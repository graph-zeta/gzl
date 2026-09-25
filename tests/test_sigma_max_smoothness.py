# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

r"""Stability tests for ``sigma_max = 4`` over the TFIM softcore graph
set, both 0qp (vacuum, k = 0) and 1qp (full Brillouin-zone grid).

For every treewidth-≤2 graph in the vendored softcore-TFIM graph
fixtures — 0qp orders 2 — 10 (401 graphs) and 1qp orders 1 — 7
(280 graphs) — on the 1D integer chain, evaluate :func:`graph_zero`
(0qp) or :func:`graph_sample` (1qp full grid) at ``sigma_max = 4``
(analytic-Epstein algebra: :func:`graph_compress` +
:func:`graph_multiply` Gamma-prefactor + :func:`graph_convolve`)
and ``sigma_max = 0`` (improved-Fourier method) across a ``nu``-grid
spanning

.. math::

    \nu \in \{1.1, 1.2, \dots, 4.0\} + \pi/300,

and check two stability criteria on the per-graph combined error

.. math::

    \mathrm{err}(\nu) = \min\!\bigl(|v_4 - v_0|,\;
                                    |v_4 - v_0| / \max(|v_4|, |v_0|)\bigr)

(maxed over k for the 1qp grid case).

1. **Smoothness in nu** — the per-graph log-10 step of :math:`\mathrm{err}`
   between adjacent grid points stays below ``JUMP_LOG10 = 2``
   (i.e. no 100-fold spike in one nu step).  Catches isolated
   instabilities of ``sigma_max = 4`` that would surface as a
   single rel.err spike (e.g. a Gamma-prefactor sign flip or a
   ``graph_compress`` regression).

2. **Absolute bound for sigma > 2** — in the safe regime where the
   Gamma poles at :math:`\nu = d + 2k` are far away,
   :math:`\mathrm{err}(\nu) < 10^{-5}` (0qp at k = 0) or
   :math:`< 10^{-3}` (1qp full grid; the higher value reflects
   larger sm = 0 truncation residuals at high-k modes for the
   same n_points).  Catches a baseline drift in the algebra that
   smoothness alone would miss.

Both tests are marked ``@pytest.mark.slow`` and run with
``pytest --runslow`` (or ``-m slow``).  The graph topologies are
loaded from the small vendored ``tests/fixtures/tfim_softcore_graphs.npz``
file (~8 KB), so the tests are fully self-contained — no external
data, no environment variable needed.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from gzl import (
    graph_from_edges_uniform,
    graph_sample,
    graph_zero,
    NotTreewidthTwoError,
    PrefactorSingularityError,
)


_FIXTURE_PATH           = Path(__file__).parent / "fixtures" / "tfim_softcore_graphs.npz"
_EXPECTED_TW2_COUNT     = 401        # tw<=2 graphs in TFIM 0qp orders 2..10
_EXPECTED_1QP_TW2_COUNT = 280        # tw<=2 graphs in TFIM 1qp orders 1..7
_N_POINTS           = 128            # 1D, cheap; sm=0 truncation safely below 1e-4 for sigma > 2
_RTOL_BOUND         = 1.0e-5         # min(abs, rel) bound for sigma > 2
_JUMP_LOG10         = 2.0            # max log10 step in err between adjacent nu
_SIGMA_TAIL_FLOOR   = 2.0            # absolute bound applies for sigma > _SIGMA_TAIL_FLOOR
_LOG10_FLOOR        = 1.0e-10        # log10 floor: errs below this count as "noise-level match"
                                     # so clean-vs-fp-noise transitions don't masquerade as jumps
_NOISE_GATE         = 1.0e-5         # if BOTH endpoints of a jump are below this, the jump is
                                     # sub-microscopic and is ignored.  Set one decade below the
                                     # 1e-4 absolute bound, so real spikes (one endpoint at or
                                     # above 1e-5) still register, but tiny-vs-tinier swings
                                     # (5e-6 -> 1e-8 etc.) don't masquerade as instabilities.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixture() -> dict:
    """Load the vendored TFIM softcore graph fixture.  It is tracked in
    git, so a missing file is an error, not a skip."""
    return dict(np.load(_FIXTURE_PATH))


def _graphs_from_fixture(qp_kind: int) -> list[tuple[str, np.ndarray, int, int]]:
    """Return all tw<=2 graphs of a given qp_kind (0 = 0qp, 1 = 1qp)
    from the vendored fixture.  Each row is ``(name, edges_flat, s, t)``.
    The 0qp set always has ``(s, t) == (0, 0)``.
    """
    f = _load_fixture()
    mask        = f["qp"] == int(qp_kind)
    orders      = f["order"][mask]
    names       = f["name"][mask]
    s_arr       = f["s"][mask]
    t_arr       = f["t"][mask]
    edges_off   = f["edges_off"]
    edges_flat  = f["edges_flat"]

    # Indices of the qp-kind subset within the global table (so we can
    # slice the global edges_off correctly for each row).
    global_idx  = np.where(f["qp"] == int(qp_kind))[0]

    out: list[tuple[str, np.ndarray, int, int]] = []
    for local_i, gi in enumerate(global_idx):
        ef = edges_flat[edges_off[gi]:edges_off[gi + 1]].astype(np.int64)
        nm = f"O{int(orders[local_i])}/{str(names[local_i])}"
        out.append((nm, ef, int(s_arr[local_i]), int(t_arr[local_i])))
    return out


def _evaluate_pair(payload: dict) -> tuple | None:
    """Worker function: build the graph and evaluate ``graph_zero`` at
    ``sigma_max = 4`` and ``sigma_max = 0``.

    Returns ``(gname, nu_idx, v4, v0)`` on success; ``None`` if
    ``sigma_max = 4`` raises :class:`PrefactorSingularityError` at this
    particular ``nu`` (no comparison is meaningful at the pole).
    """
    edges_flat = payload["edges_flat"]
    A          = payload["A"]
    n_points   = payload["n_points"]
    nu         = payload["nu"]

    try:
        g4 = graph_from_edges_uniform(
            edges_flat, nu, A, n_points, sigma_max=4.0,
        )
    except PrefactorSingularityError:
        return None
    v4 = complex(graph_zero(g4))

    g0 = graph_from_edges_uniform(
        edges_flat, nu, A, n_points, sigma_max=0.0,
    )
    v0 = complex(graph_zero(g0))

    return (payload["gname"], payload["nu_idx"], v4.real, v0.real)


def _combined_err(v4: float, v0: float) -> float:
    """Return ``min(abs_err, rel_err)`` — robust against either being
    ill-defined when the other is well-behaved.
    """
    abs_err = abs(v4 - v0)
    denom   = max(abs(v4), abs(v0))
    rel_err = abs_err / denom if denom > 0.0 else float("inf")
    return min(abs_err, rel_err)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sigma_max_smoothness_tfim_0qp_orders_2_to_10():
    A         = np.array([[1.0]])
    d         = int(A.shape[0])
    nu_grid   = np.arange(1.1, 4.001, 0.1) + np.pi / 300
    nu_grid   = np.array([float(nu) for nu in nu_grid])    # explicit Python floats
    sigma_grid = nu_grid - d
    n_nu      = len(nu_grid)

    # 1. load the tw<=2 graph set from the vendored fixture ----------------
    survivors_full = _graphs_from_fixture(qp_kind=0)
    survivors = [(name, ef) for name, ef, _, _ in survivors_full]
    assert len(survivors) == _EXPECTED_TW2_COUNT, (
        f"expected {_EXPECTED_TW2_COUNT} tw<=2 0qp graphs across orders 2-10, "
        f"got {len(survivors)}"
    )

    # 2. dispatch the (graph, nu) sweep in parallel -------------------------
    payloads = [
        dict(
            gname=name, nu_idx=k,
            edges_flat=edges_flat,
            A=A, n_points=_N_POINTS, nu=float(nu_grid[k]),
        )
        for name, edges_flat in survivors
        for k in range(n_nu)
    ]
    n_workers = max(1, min(4, (os.cpu_count() or 1)))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_evaluate_pair, payloads))

    # 3. reshape into per-graph err arrays indexed by nu --------------------
    err_per_graph = {name: np.full(n_nu, np.nan) for name, _ in survivors}
    for r in results:
        if r is None:
            continue
        gname, nu_idx, v4, v0 = r
        err_per_graph[gname][nu_idx] = _combined_err(v4, v0)

    tail_mask = sigma_grid > _SIGMA_TAIL_FLOOR

    # 4. per-graph stability metrics ----------------------------------------
    failures_jump: list = []
    failures_tail: list = []
    summary:       list = []

    for gname, errs in err_per_graph.items():
        valid = ~np.isnan(errs)
        if valid.sum() < 2:
            continue

        log_errs = np.log10(np.maximum(errs, _LOG10_FLOOR))

        # log10-jump between consecutive *valid* nu entries.  A jump is
        # ignored when both endpoints are below _NOISE_GATE: e.g. a swing
        # 8e-8 -> 5e-10 is sub-microscopic and the absolute-bound check
        # already lets it through; the smoothness check is meant to
        # catch real instabilities, not the difference between fp-noise
        # and slightly-tinier-fp-noise.
        diffs: list = []
        for k in range(n_nu - 1):
            if valid[k] and valid[k + 1]:
                if errs[k] < _NOISE_GATE and errs[k + 1] < _NOISE_GATE:
                    continue
                diffs.append((nu_grid[k], nu_grid[k + 1],
                              log_errs[k], log_errs[k + 1]))
        max_jump = max((abs(b - a) for _, _, a, b in diffs), default=0.0)

        # absolute-bound check for sigma > _SIGMA_TAIL_FLOOR
        tail_valid = tail_mask & valid
        if tail_valid.any():
            max_tail_err = float(np.max(errs[tail_valid]))
            argmax_tail  = int(np.argmax(np.where(tail_valid, errs, -np.inf)))
            tail_nu      = float(nu_grid[argmax_tail])
        else:
            max_tail_err = 0.0
            tail_nu      = float("nan")

        max_err_overall = float(np.max(errs[valid]))
        summary.append((gname, max_err_overall, max_jump, max_tail_err))

        if max_jump > _JUMP_LOG10:
            worst = max(diffs, key=lambda d: abs(d[3] - d[2]))
            failures_jump.append((
                gname,
                float(worst[0]), float(worst[1]),
                10.0 ** worst[2], 10.0 ** worst[3],
                float(max_jump),
            ))
        if max_tail_err > _RTOL_BOUND:
            failures_tail.append((gname, tail_nu, max_tail_err))

    # 5. report -------------------------------------------------------------
    if failures_jump or failures_tail:
        msg: list[str] = []
        if failures_jump:
            msg.append(
                f"sigma_max=4 vs sigma_max=0 smoothness failures "
                f"(>{_JUMP_LOG10:.1f} log10 step in one nu step) on "
                f"{len(failures_jump)} graph(s):"
            )
            for g, n0, n1, e0, e1, j in failures_jump[:10]:
                msg.append(
                    f"  {g}: nu {n0:.4f} -> {n1:.4f}, "
                    f"err {e0:.2e} -> {e1:.2e}, log10 jump {j:.2f}"
                )
        if failures_tail:
            msg.append(
                f"sigma_max=4 vs sigma_max=0 absolute-bound failures "
                f"(min(abs, rel) > {_RTOL_BOUND:.0e} for sigma > "
                f"{_SIGMA_TAIL_FLOOR}) on "
                f"{len(failures_tail)} graph(s):"
            )
            for g, nu, e in failures_tail[:10]:
                msg.append(f"  {g}: nu {nu:.4f}, err {e:.2e}")
        pytest.fail("\n".join(msg))

    # diagnostic on success — ranks worst-five by log10-jump
    summary.sort(key=lambda s: -s[2])
    print(
        f"\nsigma_max=4 smoothness summary "
        f"({len(summary)} graphs, {n_nu} nu points each, "
        f"n_points={_N_POINTS}, n_workers={n_workers}). "
        f"Worst 5 by log10-jump:"
    )
    for g, mr, jump, tail in summary[:5]:
        print(
            f"  {g}: max err {mr:.2e}, max log10-jump {jump:.2f}, "
            f"sigma>2 max err {tail:.2e}"
        )


# ---------------------------------------------------------------------------
# 1qp variant: same checks but over the FULL Brillouin-zone grid
# ---------------------------------------------------------------------------

def _evaluate_pair_grid(payload: dict) -> tuple | None:
    """Worker: build a 1qp graph and evaluate ``graph_sample`` (the full
    Brillouin-zone grid) at ``sigma_max = 4`` and ``sigma_max = 0``.

    Returns ``(gname, nu_idx, sample4, sample0)`` on success;
    ``None`` if ``sigma_max = 4`` raises :class:`PrefactorSingularityError`
    at this particular ``nu`` (no comparison is meaningful at the pole).
    """
    edges_flat = payload["edges_flat"]
    A          = payload["A"]
    n_points   = payload["n_points"]
    nu         = payload["nu"]
    s          = payload["s"]
    t          = payload["t"]

    try:
        g4 = graph_from_edges_uniform(
            edges_flat, nu, A, n_points, s=s, t=t, sigma_max=4.0,
        )
    except PrefactorSingularityError:
        return None
    sample4 = np.asarray(graph_sample(g4))

    g0 = graph_from_edges_uniform(
        edges_flat, nu, A, n_points, s=s, t=t, sigma_max=0.0,
    )
    sample0 = np.asarray(graph_sample(g0))

    return (payload["gname"], payload["nu_idx"], sample4, sample0)


def _grid_combined_err(sample4: np.ndarray, sample0: np.ndarray) -> float:
    """``max_k of min(abs_err, rel_err)`` across the Brillouin-zone grid."""
    diff    = sample4 - sample0
    abs_arr = np.abs(diff)
    denom   = np.maximum(np.abs(sample4), np.abs(sample0))
    # rel = abs / denom; where denom == 0 we already have abs == 0 so the
    # combined min collapses to 0 (no-op).  Use a safe divide.
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_arr = np.where(denom > 0.0, abs_arr / denom, 0.0)
    combined = np.minimum(abs_arr, rel_arr)
    return float(np.max(combined))


@pytest.mark.slow
def test_sigma_max_smoothness_tfim_1qp_orders_1_to_7():
    r"""1qp variant of the smoothness sweep.

    For every treewidth-≤2 graph in the softcore-TFIM 1qp graph
    fixtures (orders 1 — 7, 280 graphs total) on the 1D integer chain,
    evaluate :func:`graph_sample` (the full Brillouin-zone grid, not
    just :math:`\boldsymbol{k} = \boldsymbol{0}`) at ``sigma_max = 4``
    and ``sigma_max = 0`` across the same nu-grid as the 0qp test, and
    apply the same two stability criteria, with the per-graph err

    .. math::

        \mathrm{err}(\nu) = \max_{\boldsymbol{k}}\,
            \min\!\bigl(|v_4(\boldsymbol{k}) - v_0(\boldsymbol{k})|,\;
                        |v_4(\boldsymbol{k}) - v_0(\boldsymbol{k})| /
                        \max(|v_4|, |v_0|)\bigr)

    taken over the full :math:`(n_\text{points})^d` grid.
    """
    A          = np.array([[1.0]])
    d          = int(A.shape[0])
    nu_grid    = np.arange(1.1, 4.001, 0.1) + np.pi / 300
    nu_grid    = np.array([float(nu) for nu in nu_grid])
    sigma_grid = nu_grid - d
    n_nu       = len(nu_grid)

    # 1qp tail bound: the full Brillouin-zone grid picks up larger sm=4
    # vs sm=0 disagreement than the 0qp-at-k=0 case, because high-k
    # modes have larger truncation residuals than k=0 does for the
    # same n_points.  Empirically max_k err sits around 1e-4 in the
    # sigma=2-2.5 range and only crosses 1e-5 at sigma ~ 3.  We keep
    # the same sigma>2 floor as 0qp but loosen the absolute bound
    # by an order of magnitude — 1e-3 still catches a real algebra
    # regression (existing residuals top out at 7e-4) while letting
    # the n_points=128 grid resolution through.
    sigma_tail_floor_1qp = 2.0
    rtol_bound_1qp       = 1.0e-3

    # 1. load the tw<=2 graph set from the vendored fixture ----------------
    survivors = _graphs_from_fixture(qp_kind=1)
    assert len(survivors) == _EXPECTED_1QP_TW2_COUNT, (
        f"expected {_EXPECTED_1QP_TW2_COUNT} tw<=2 1qp graphs across orders 1-7, "
        f"got {len(survivors)}"
    )

    # 2. dispatch the (graph, nu) sweep in parallel -------------------------
    payloads = [
        dict(
            gname=name, nu_idx=k,
            edges_flat=edges_flat, s=s, t=t,
            A=A, n_points=_N_POINTS, nu=float(nu_grid[k]),
        )
        for name, edges_flat, s, t in survivors
        for k in range(n_nu)
    ]
    n_workers = max(1, min(4, (os.cpu_count() or 1)))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_evaluate_pair_grid, payloads))

    # 3. reshape into per-graph err arrays indexed by nu --------------------
    err_per_graph = {name: np.full(n_nu, np.nan) for name, _, _, _ in survivors}
    for r in results:
        if r is None:
            continue
        gname, nu_idx, sample4, sample0 = r
        err_per_graph[gname][nu_idx] = _grid_combined_err(sample4, sample0)

    tail_mask = sigma_grid > sigma_tail_floor_1qp

    # 4. per-graph stability metrics (same as 0qp test) ---------------------
    failures_jump: list = []
    failures_tail: list = []
    summary:       list = []

    for gname, errs in err_per_graph.items():
        valid = ~np.isnan(errs)
        if valid.sum() < 2:
            continue

        log_errs = np.log10(np.maximum(errs, _LOG10_FLOOR))

        diffs: list = []
        for k in range(n_nu - 1):
            if valid[k] and valid[k + 1]:
                if errs[k] < _NOISE_GATE and errs[k + 1] < _NOISE_GATE:
                    continue
                diffs.append((nu_grid[k], nu_grid[k + 1],
                              log_errs[k], log_errs[k + 1]))
        max_jump = max((abs(b - a) for _, _, a, b in diffs), default=0.0)

        tail_valid = tail_mask & valid
        if tail_valid.any():
            max_tail_err = float(np.max(errs[tail_valid]))
            argmax_tail  = int(np.argmax(np.where(tail_valid, errs, -np.inf)))
            tail_nu      = float(nu_grid[argmax_tail])
        else:
            max_tail_err = 0.0
            tail_nu      = float("nan")

        max_err_overall = float(np.max(errs[valid]))
        summary.append((gname, max_err_overall, max_jump, max_tail_err))

        if max_jump > _JUMP_LOG10:
            worst = max(diffs, key=lambda d: abs(d[3] - d[2]))
            failures_jump.append((
                gname,
                float(worst[0]), float(worst[1]),
                10.0 ** worst[2], 10.0 ** worst[3],
                float(max_jump),
            ))
        if max_tail_err > rtol_bound_1qp:
            failures_tail.append((gname, tail_nu, max_tail_err))

    # 5. report -------------------------------------------------------------
    if failures_jump or failures_tail:
        msg: list[str] = []
        if failures_jump:
            msg.append(
                f"sigma_max=4 vs sigma_max=0 1qp grid smoothness failures "
                f"(>{_JUMP_LOG10:.1f} log10 step in one nu step) on "
                f"{len(failures_jump)} graph(s):"
            )
            for g, n0, n1, e0, e1, j in failures_jump[:10]:
                msg.append(
                    f"  {g}: nu {n0:.4f} -> {n1:.4f}, "
                    f"err {e0:.2e} -> {e1:.2e}, log10 jump {j:.2f}"
                )
        if failures_tail:
            msg.append(
                f"sigma_max=4 vs sigma_max=0 1qp grid absolute-bound "
                f"failures (max_k min(abs, rel) > {rtol_bound_1qp:.0e} for "
                f"sigma > {sigma_tail_floor_1qp}) on "
                f"{len(failures_tail)} graph(s):"
            )
            for g, nu, e in failures_tail[:10]:
                msg.append(f"  {g}: nu {nu:.4f}, err {e:.2e}")
        pytest.fail("\n".join(msg))

    # diagnostic on success
    summary.sort(key=lambda s: -s[2])
    print(
        f"\nsigma_max=4 1qp grid smoothness summary "
        f"({len(summary)} graphs, {n_nu} nu points each, "
        f"n_points={_N_POINTS}, n_workers={n_workers}). "
        f"Worst 5 by log10-jump:"
    )
    for g, mr, jump, tail in summary[:5]:
        print(
            f"  {g}: max err {mr:.2e}, max log10-jump {jump:.2f}, "
            f"sigma>2 max err {tail:.2e}"
        )
