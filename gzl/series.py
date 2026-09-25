# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

"""Per-order series coefficients ``Σ_G prefactor_G · ζ_G(k; ν)`` over a
graph corpus.

The corpus is a consolidated HDF5 (per-graph subgroup layout) or NPZ
(flat / CSR layout), as written by ``tools/consolidate_corpus.py``.  For each
graph the file carries an integer ``edges`` list, integer
``multiplicities``, and a real ``prefactor`` that weights the graph's
contribution to the per-order coefficient.  The optional ``hopping``
field (``(n_graphs, 2)`` int) encodes per-graph terminals ``(s, t)``
(1qp corpora); when absent (0qp corpora) the vacuum convention
``s = t = 0`` is used.

Every graph is evaluated by :func:`gzl.evaluate_graph`, and a graph it
refuses -- a self-loop, a disconnected graph, a source or terminal that
is no vertex, ``n_points = 0`` for a block that needs a grid -- stops
the pass with that function's exception.  There is no second evaluator
behind it.  A whole-graph ``k = 0`` fallback cascade used to catch
these refusals, and it returned the value of one component for a
disconnected graph, a number growing with the torus volume, or an
unrelated ``ValueError`` in their place.

This module is application-agnostic — it operates on any corpus with the
documented schema.  Its command-line driver is ``gzl series``, which
lives in :mod:`gzl._cli_series`; ``main(argv)`` at the bottom of this
file runs that driver in-process.

Run:

    gzl series --config <CONFIG.toml>
    gzl series --corpus tfim1qp --A chain \\
        --n-points 64 --order-max 13 \\
        --nu-start 1.1 --nu-end 4.0 --nu-step 0.01 \\
        --output series.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

# zeta_circle uses scipy.integrate.quad internally and routinely emits
# IntegrationWarning ("roundoff error is detected") at high orders /
# slow-tail exponents.  The reported value is fine for our use; suppress
# the warning to keep the CLI / runner output readable.
try:
    from scipy.integrate import IntegrationWarning as _IntegrationWarning
    warnings.filterwarnings("ignore", category=_IntegrationWarning)
except ImportError:                                      # pragma: no cover
    pass

from gzl._data import CORPORA, _resolve_corpus
from gzl._lattices import LATTICES, lattice_matrix
from gzl._real import _as_real
from gzl.interaction import Interaction, is_interaction
from gzl.frontend import (  # noqa: F401  (back-compat re-export of _evaluate_via_topology)
    evaluate_graph,
    _evaluate_via_topology,
    DisconnectedGraphError,
    NPointsRequiredError,
    SelfLoopError,
    TopologyEvaluatorUnavailableError,
    UnsupportedLatticeSumError,
    VertexOutOfRangeError,
    _finite_momentum,
    _grid_size,
)


__all__ = [
    "evaluate_corpus",
    "compute_series_coefficients",
    "nu_grid",
    "main",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: The settings of the removed k = 0 fallback cascade.  The corpus
#: front-ends still accept them, ignore them and say so; they go in a
#: later release.
_CASCADE_SETTINGS = ("nu_tensor_threshold", "tw_threshold", "sigma_max",
                     "high_tw_fallback", "direct_sum_L_list", "direct_sum_K")


def _warn_cascade_settings(where: str, settings: Mapping) -> None:
    """Warn once per call about every cascade setting passed as non-None.

    ``stacklevel=3`` names the line that called the front-end."""
    given = [name for name in _CASCADE_SETTINGS
             if settings.get(name) is not None]
    if given:
        one = len(given) == 1
        warnings.warn(
            f"{where}: {', '.join(given)} {'has' if one else 'have'} no "
            f"effect and will be removed.  {'It' if one else 'They'} tuned "
            f"the k = 0 fallback cascade for graphs evaluate_graph refuses, "
            f"and that cascade is gone: such a graph now raises "
            f"evaluate_graph's own exception.",
            DeprecationWarning, stacklevel=3,
        )


def nu_grid(start: float, end: float, step: float) -> np.ndarray:
    """``np.arange``-style ν grid, inclusive of ``end`` if it lands on the step.

    Half-step extension on the right ensures the endpoint is included for
    the typical case where ``(end - start)`` is an integer multiple of
    ``step`` modulo float64 noise.
    """
    if step <= 0:
        raise ValueError(f"nu_grid: step must be positive (got {step}).")
    if end < start:
        raise ValueError(
            f"nu_grid: end must be >= start (got start={start}, end={end})."
        )
    return np.arange(start, end + step * 0.5, step)


# ---------------------------------------------------------------------------
# Corpus readers
# ---------------------------------------------------------------------------

def _read_corpus_npz(path: Path) -> dict[int, dict]:
    """Parse a flat / CSR corpus NPZ.  Returns {order: {edges_list, prefactor,
    multiplicities, graph_id, hopping}}.

    edges_list is a Python list of (E_i, 2) int arrays — one per graph in
    that order — already extracted from edges_flat / edges_off so callers
    don't need to slice repeatedly.
    """
    with np.load(path, allow_pickle=False) as f:
        orders = np.asarray(f["orders"]).astype(int)
        order_off = np.asarray(f["order_off"]).astype(int)
        order_per_graph = np.asarray(f["order"]).astype(int)
        graph_id = np.asarray(f["graph_id"])
        prefactor = np.asarray(f["prefactor"]).astype(float)
        edges_off = np.asarray(f["edges_off"]).astype(int)
        edges_flat = np.asarray(f["edges_flat"]).astype(int)
        multiplicities = np.asarray(f["multiplicities"]).astype(int)
        hopping = (
            np.asarray(f["hopping"]).astype(int) if "hopping" in f.files else None
        )

    out: dict[int, dict] = {}
    for k, order in enumerate(orders):
        gs, ge = order_off[k], order_off[k + 1]
        order_edges = []
        order_mults = []
        for i in range(gs, ge):
            es, ee = edges_off[i], edges_off[i + 1]
            order_edges.append(edges_flat[es:ee])
            order_mults.append(multiplicities[es:ee])
        order_block = {
            "edges_list": order_edges,
            "multiplicities_list": order_mults,
            "prefactor": prefactor[gs:ge],
            "graph_id": graph_id[gs:ge],
        }
        if hopping is not None:
            order_block["hopping"] = hopping[gs:ge]
        out[int(order)] = order_block

    # sanity
    if order_per_graph.size and not np.array_equal(
        order_per_graph,
        np.repeat(orders, np.diff(order_off)),
    ):
        raise RuntimeError(
            f"{path}: order_per_graph and (orders, order_off) disagree."
        )
    return out


def _read_corpus_h5(path: Path) -> dict[int, dict]:
    """Parse a consolidated H5 (per-graph subgroup layout).  Returns the
    same shape as :func:`_read_corpus_npz`.
    """
    out: dict[int, dict] = {}
    with h5py.File(path, "r") as f:
        order_keys = sorted(
            (k for k in f.keys() if k.startswith("O")),
            key=lambda s: int(s[1:]),
        )
        for ok in order_keys:
            order = int(ok[1:])
            order_grp = f[ok]
            graph_names = sorted(
                order_grp.keys(),
                key=lambda n: int(n.split("_")[1]) if "_" in n else -1,
            )
            edges_list = []
            mults_list = []
            prefs = np.empty(len(graph_names), dtype=float)
            hops = []
            ids = []
            for i, gname in enumerate(graph_names):
                g = order_grp[gname]
                edges_list.append(np.asarray(g["edges"], dtype=int))
                mults_list.append(np.asarray(g["multiplicities"], dtype=int))
                prefs[i] = float(np.asarray(g["prefactor"]))
                if "hopping" in g:
                    hops.append(np.asarray(g["hopping"], dtype=int))
                ids.append(gname.encode())
            block = {
                "edges_list": edges_list,
                "multiplicities_list": mults_list,
                "prefactor": prefs,
                "graph_id": np.array(ids),
            }
            if hops:
                block["hopping"] = np.stack(hops, axis=0)
            out[order] = block
    return out


def _load_corpus(path: str | Path) -> dict[int, dict]:
    """Open a consolidated corpus file.  Auto-detects NPZ vs H5 by suffix.

    ``path`` may also be a shipped corpus's name (``"tfim0qp"``,
    ``"tfim1qp"``); :func:`gzl._data._resolve_corpus` has the rule.
    """
    p = _resolve_corpus(path)
    if not p.is_file():
        msg = f"Corpus file not found: {p}"
        # A bare string -- one component, no suffix, so no corpus file --
        # that names nothing on disk can only have meant a name: say which
        # names there are.  Every path-shaped argument, and a bare string
        # that is an existing directory, keeps the message it always had.
        if (isinstance(path, str) and not p.suffix and len(p.parts) == 1
                and not p.exists()):
            msg += (" (nor is it a shipped corpus name: "
                    + ", ".join(repr(name) for name in CORPORA) + ")")
        raise FileNotFoundError(msg)
    suffix = p.suffix.lower()
    if suffix == ".npz":
        return _read_corpus_npz(p)
    if suffix in {".h5", ".hdf5"}:
        return _read_corpus_h5(p)
    raise ValueError(
        f"Unsupported corpus suffix {suffix!r}; expected .npz, .h5, or .hdf5."
    )


# ---------------------------------------------------------------------------
# Library function
# ---------------------------------------------------------------------------

def _parse_momentum(momentum, d: int, n_points: int, has_hopping: bool):
    """Parse the public ``momentum`` argument into
    ``(mode, mom_for_eval, trailing_shape)``.

    The corpus-level ``has_hopping`` flag selects the ``momentum=None``
    default — matching
    :func:`gzl.frontend.evaluate_graph`'s frontend semantics:

    * Vacuum / 0qp corpus (``has_hopping=False``) → scalar at k=0.
    * 1qp corpus (``has_hopping=True``) → full BZ grid (the most useful
      object for an n_points-cheap call; see the cost note at
      :func:`gzl.frontend.evaluate_graph`).

    To force scalar-at-k=0 evaluation on a 1qp corpus, pass an explicit
    ``momentum=np.zeros(d)``.

    Modes:
        ``"vacuum_scalar"`` — momentum is None, 0qp corpus; scalar.
                       ``trailing_shape = ()``.
        ``"grid"``    — momentum is None, 1qp corpus; full BZ grid
                       (``n_points^d`` points).
                       ``mom_for_eval = None`` (so
                       :func:`evaluate_graph` falls into its own
                       grid-default path for 1qp graphs).
                       ``trailing_shape = (n_points,) * d``.
        ``"single"``  — single k-vector, shape (d,).
                       ``trailing_shape = ()``.
        ``"batch"``   — batch of k-vectors, shape (N, d).
                       ``trailing_shape = (N,)``.

    Scalar / 1-D shorthand mirrors
    :func:`gzl.frontend._normalise_momentum`.
    """
    if momentum is None:
        if has_hopping:
            if int(n_points) <= 0:
                # The error evaluate_graph raises for the same request.
                raise NPointsRequiredError(
                    f"compute_series_coefficients: momentum=None on a "
                    f"1qp corpus defaults to the full BZ grid and "
                    f"requires n_points > 0 (got {n_points}).  Pass "
                    f"momentum=np.zeros({d}) to force scalar evaluation "
                    f"at k=0 on a 1qp corpus."
                )
            return "grid", None, (int(n_points),) * d
        return "vacuum_scalar", None, ()
    if isinstance(momentum, str):
        raise ValueError(
            f"momentum must be None, a scalar, or an array (got "
            f"{momentum!r}).  The 'grid' string sentinel was removed: "
            f"pass momentum=None on a 1qp corpus for the same effect."
        )
    if isinstance(momentum, (int, float)):
        if d != 1:
            raise ValueError(
                f"compute_series_coefficients: scalar momentum is "
                f"accepted only in d = 1; got d = {d}."
            )
        return "single", _finite_momentum(
            np.asarray([float(momentum)], dtype=float), momentum,
            "compute_series_coefficients"), ()
    arr = _finite_momentum(np.asarray(momentum, dtype=float), momentum,
                           "compute_series_coefficients")
    if arr.ndim == 1:
        if arr.size == d:
            return "single", arr, ()
        if d == 1:
            return "batch", arr.reshape(-1, 1), (arr.size,)
        raise ValueError(
            f"compute_series_coefficients: 1-D momentum of length "
            f"{arr.size} is ambiguous for d = {d}; pass shape ({d},) "
            f"for a single k or shape (N, {d}) for a batch."
        )
    if arr.ndim == 2 and arr.shape[1] == d:
        return "batch", arr, (arr.shape[0],)
    raise ValueError(
        f"compute_series_coefficients: momentum has unsupported shape "
        f"{arr.shape} for d = {d}; expected None, ({d},), or (N, {d})."
    )


def _coerce_sweep(nu):
    """Normalise the coupling argument of the corpus front-ends.

    Returns ``(nu_arr, sweep, scalar_request)``, the last of which says
    whether the CALLER asked about a single coupling -- a float, a NumPy
    scalar, a 0-d array, or one :class:`~gzl.Interaction` -- as opposed
    to a sweep of one point.  It cannot be recovered afterwards:
    ``np.atleast_1d`` below erases the difference, so ``3.0``, ``[3.0]``
    and ``np.array([3.0])`` all give ``nu_arr.shape == (1,)``.  Keying
    the return shape on ``n_nu == 1`` instead would silently collapse a
    one-point sweep, and with it the CLI's whole-Brillouin-zone schema.

    In detail: ``nu_arr`` is the float64 array the
    historical code built (``np.atleast_1d(np.asarray(nu, float))``,
    1-D, or a ``ValueError``) with an Interaction sweep point represented
    by its TAIL exponent, and ``sweep`` is the list of sweep points as
    the user passed them -- floats, or :class:`~gzl.Interaction`-
    likes with their identity preserved.  A plain ``Interaction(b=[1],
    nu=[nu])`` stays in ``sweep`` (its label and identity go into the
    records) and is ROUTED as its float by :func:`_iter_corpus_records`,
    so the legacy pass is byte-identical.

    The input is materialised exactly once: a generator / iterator /
    ``map`` object, and an object-dtype ndarray, are sequences here.
    The historical ``any(is_interaction(x) for x in nu)`` probe consumed
    an iterator up to its first Interaction and the pass then ran on the
    remainder -- silently, with coefficient arrays of the wrong length
    (measured: a 3-point generator sweep returned shape ``(2,)``, an
    ``iter([3.0, V1, V2])`` shape ``(1,)``).
    """
    # Was the ARGUMENT a single coupling rather than a sweep?  Keyed on
    # the argument, never on ``n_nu == 1``: a one-point sweep
    # (``[3.0]``, ``nu_grid(3, 3, 1)``) is a sweep and keeps its axis.
    scalar_request = bool(
        is_interaction(nu)
        or np.isscalar(nu)
        or (isinstance(nu, np.ndarray) and nu.ndim == 0)
    )
    if is_interaction(nu):
        seq = [nu]
    elif np.isscalar(nu) or (isinstance(nu, np.ndarray) and nu.dtype != object):
        seq, src = None, nu
    else:
        if isinstance(nu, np.ndarray):
            nu = nu.ravel()          # an object array of any shape, as coerce_nu
        seq = list(nu)      # the one materialisation
        if not seq:
            raise ValueError(
                "the nu sweep is empty; give at least one coupling (a float, "
                "an Interaction, or a non-empty sequence -- an exhausted "
                "generator is empty too)"
            )
        if not any(is_interaction(x) for x in seq):
            seq, src = None, seq
    if seq is None:
        nu_arr = np.atleast_1d(np.asarray(src, dtype=float))
        if nu_arr.ndim != 1:
            raise ValueError(f"nu must be scalar or 1-D, got shape {nu_arr.shape}.")
        if nu_arr.size == 0:
            raise ValueError("the nu sweep is empty; give at least one coupling")
        return nu_arr, list(nu_arr), scalar_request
    sweep = [x if is_interaction(x) else float(x) for x in seq]
    nu_arr = np.asarray([x.tail_exponent if is_interaction(x) else x
                         for x in sweep], dtype=float)
    return nu_arr, sweep, scalar_request


def _sweep_has_interaction(sweep) -> bool:
    return any(is_interaction(x) for x in sweep)


def _has_nontrivial_hopping(corpus, available_orders) -> bool:
    """True iff any graph in any (truncated) order has ``s != t``.

    A corpus-level property: pure-vacuum corpora — including 0qp corpora
    that carry an all-zeros placeholder ``hopping`` field — are
    k-independent and default to scalar evaluation.
    """
    for o in available_orders:
        hop = corpus[o].get("hopping")
        if hop is None:
            continue
        hop_arr = np.asarray(hop, dtype=int)
        if hop_arr.size == 0:
            continue
        if np.any(hop_arr[:, 0] != hop_arr[:, 1]):
            return True
    return False


def _iter_corpus_records(corpus, available_orders, nu_arr, A, n_points,
                         mom_for_eval, trailing_shape, *,
                         block_cache, fast_cycles, engine="hybrid",
                         richardson=True, dense_engine=None,
                         sp_n_points=None, core_grading=True):
    """Yield one record per ``(order, graph, ν)`` over the corpus.

    The single shared per-graph evaluation pass behind both
    :func:`evaluate_corpus` (which returns these records) and
    :func:`compute_series_coefficients` (which groups them by
    ``(order, ν)`` and sums ``contribution``).  Each yielded dict is
    ``{order, graph_id, nu, nu_index, route, prefactor, value,
    contribution, s, t}`` with ``value`` the raw ζ_G and
    ``contribution = prefactor · value``; whenever a sweep point is an
    Interaction every record also carries ``interaction`` (the object
    the user passed, ``None`` for a float point) and ``nu_label``.
    Orders are yielded in ascending order, graphs in corpus order, ν
    innermost — so a ``groupby`` on ``order`` sees contiguous groups.
    Every value comes from :func:`evaluate_graph`, and whatever it raises
    for a graph propagates unchanged.
    """
    with_interaction = _sweep_has_interaction(nu_arr)
    for order in available_orders:
        block = corpus[order]
        edges_list = block["edges_list"]
        mults_list = block["multiplicities_list"]
        prefactors = block["prefactor"]
        graph_ids = block["graph_id"]
        hopping_arr = block.get("hopping")     # (n_graphs, 2) int or None (0qp)
        for i in range(len(edges_list)):
            edges = edges_list[i]
            mults = mults_list[i]
            edges_flat = np.repeat(edges, mults, axis=0)
            pref = float(prefactors[i])
            gid = (
                graph_ids[i].decode()
                if isinstance(graph_ids[i], (bytes, np.bytes_))
                else str(graph_ids[i])
            )
            if hopping_arr is not None:
                s_v, t_v = int(hopping_arr[i][0]), int(hopping_arr[i][1])
            else:
                s_v, t_v = 0, 0

            for j, nu_val in enumerate(nu_arr):
                # A sweep point is a float or an Interaction-like; the
                # record's ``nu`` is its tail exponent.  A plain
                # ``|x|^-nu`` Interaction (one unit term, no table) is
                # ROUTED as its float -- the legacy pass, bit-identical --
                # while the record below keeps the object the user passed
                # and its label.
                _is_int = is_interaction(nu_val)
                _plain = nu_val.as_plain_float() if _is_int else None
                _as_kernel = _is_int and _plain is None
                nu_tail = float(nu_val.tail_exponent) if _is_int else float(nu_val)
                nu_eval = nu_val if _as_kernel else (
                    float(_plain) if _is_int else float(nu_val))
                # The topology router; a graph it refuses raises here and
                # stops the pass (see the module docstring).  The note
                # names the graph, which the router cannot: a pass has
                # thousands.
                try:
                    if s_v == t_v:
                        # Vacuum: k-independent scalar, broadcast to
                        # ``trailing_shape``.
                        val = evaluate_graph(
                            edges_flat, nu_eval, A,
                            source=s_v, terminal=None,
                            n_points=n_points,
                            richardson=richardson,
                            fast_cycles=fast_cycles,
                            block_cache=block_cache,
                            engine=engine,
                            dense_engine=dense_engine,
                            sp_n_points=sp_n_points,
                            core_grading=core_grading,
                            accuracy="floor",
                        )
                        if trailing_shape:
                            # Through the guard, never through numpy's
                            # cast: ``np.full(..., dtype=float)`` DROPS
                            # an imaginary part on a warning.
                            val = np.full(
                                trailing_shape,
                                _as_real(val, where="compute_series_coefficients"),
                                dtype=float,
                            )
                    else:
                        # 1qp; ``mom_for_eval`` is an explicit k-array or
                        # ``None`` (grid mode → full BZ grid).
                        val = evaluate_graph(
                            edges_flat, nu_eval, A,
                            source=s_v, terminal=int(t_v),
                            momentum=mom_for_eval,
                            n_points=n_points,
                            richardson=richardson,
                            fast_cycles=fast_cycles,
                            block_cache=block_cache,
                            engine=engine,
                            dense_engine=dense_engine,
                            sp_n_points=sp_n_points,
                            core_grading=core_grading,
                            accuracy="floor",
                        )
                        if trailing_shape:
                            val = _as_real(
                                val, where="compute_series_coefficients",
                            ).reshape(trailing_shape)
                except Exception as exc:
                    exc.add_note(
                        f"corpus graph {gid} at order {order}, "
                        + (nu_val.default_label if _is_int
                           else f"nu = {nu_tail:g}")
                        + (f", s = {s_v}, t = {t_v}" if s_v != t_v else "")
                    )
                    raise
                route = "topology_analytic"

                rec = {
                    "order": order,
                    "graph_id": gid,
                    "nu": nu_tail,
                    "nu_index": j,
                    "route": route,
                    "prefactor": pref,
                    "value": val,
                    "contribution": pref * val,
                    "s": s_v,
                    "t": t_v,
                }
                if with_interaction:
                    # ``nu`` above is the TAIL exponent (a float for every
                    # reader that joins on it); the sweep point itself
                    # and its label ride beside it -- present whenever
                    # the USER passed any Interaction, a plain power law
                    # included (routed as its float above, but it is
                    # still the object, and the label, the user gave).
                    rec["interaction"] = nu_val if _is_int else None
                    rec["nu_label"] = (nu_val.default_label if _is_int
                                       else f"nu={nu_tail:g}")
                yield rec


def compute_series_coefficients(
    corpus_path: str | Path,
    nu: float | np.ndarray | Interaction | Sequence[float | Interaction],
    A: "np.ndarray | str",
    n_points: int,
    *,
    momentum=None,
    order_max: int | None = None,
    nu_tensor_threshold: float | None = None,    # deprecated, no effect
    tw_threshold: int | None = None,             # deprecated, no effect
    sigma_max: float | None = None,              # deprecated, no effect
    high_tw_fallback: str | None = None,         # deprecated, no effect
    direct_sum_L_list: tuple | None = None,      # deprecated, no effect
    direct_sum_K: int | None = None,             # deprecated, no effect
    return_diagnostics: bool = False,
    progress: bool = False,
    fast_cycles: bool = False,
    richardson: bool = True,
    engine: str = "hybrid",
    dense_engine: "str | None" = None,
    sp_n_points: "int | None" = None,
    core_grading: bool = True,
) -> dict[int, np.ndarray] | tuple[..., ...]:
    r"""Per-order series coefficients ``Σ_G prefactor_G · ζ_G(k; ν)``.

    By default (``momentum=None``) a 0qp corpus is evaluated at zero
    external momentum ``k = 0`` and a 1qp corpus on the full
    Brillouin-zone grid.  Pass ``momentum`` to evaluate at one or more
    chosen k-vectors.

    ``corpus_path``, ``nu``, ``A``, ``n_points``, ``momentum``,
    ``order_max``, ``return_diagnostics`` and ``progress`` are stable
    API.  The other parameters are provisional: they may change in a
    minor release (see "API stability" in DOCUMENTATION.md).
    ``fast_cycles``, ``richardson``, ``engine``, ``dense_engine``,
    ``sp_n_points`` and ``core_grading`` select or tune the numerical
    method and are forwarded to :func:`gzl.evaluate_graph`, which
    describes the ones not described below.  ``nu_tensor_threshold``,
    ``tw_threshold``, ``sigma_max``, ``high_tw_fallback``,
    ``direct_sum_L_list`` and ``direct_sum_K`` are deprecated and have
    no effect; see below.

    Every graph goes through :func:`gzl.evaluate_graph`, and a graph it
    refuses stops the pass with its exception, a
    :class:`GraphZetaError`: :class:`SelfLoopError`,
    :class:`DisconnectedGraphError`, :class:`VertexOutOfRangeError`,
    :class:`NPointsRequiredError` (``n_points = 0`` with a block that
    needs a grid) or :class:`UnsupportedLatticeSumError`.  A note on the
    exception (PEP 678) names the corpus graph, its order and the
    coupling.

    Parameters
    ----------
    corpus_path
        Path to a consolidated graph corpus, ``.npz`` (flat/CSR layout)
        or ``.h5`` / ``.hdf5`` (per-graph subgroup layout).  See the
        consolidator script for details of the schema.  Or the name of a
        shipped TFIM corpus, as a ``str`` spelled exactly so:
        ``"tfim0qp"`` is ``data_path("tfim_softcore_corpus_0qp.npz")``,
        ``"tfim1qp"`` is ``data_path("tfim_softcore_corpus_1qp.npz")``.
        A ``Path`` is always a path.
    nu
        The coupling sweep.  A float ν — the per-edge exponent, uniform
        across edges (``np.inf`` requests the nearest-neighbour kernel)
        — or a 1-D array of them; or an :class:`~gzl.Interaction`
        (a general kernel ``V(x) = a(x) + Σ_j b_j |x|^{-ν_j}`` on every
        edge); or a sequence mixing floats and Interactions.  A single
        float or Interaction is one coupling and adds no axis; a list,
        a 1-D array or a sequence is a sweep of couplings, one
        coefficient row per point in the order given (a generator /
        iterator is materialised once), even for one point.  For
        an Interaction point the per-graph records carry ``nu`` = its
        TAIL exponent ``min_j ν_j`` (``inf`` for a purely compact
        kernel), ``interaction`` = the object itself and ``nu_label`` =
        its label (``default_label``); those two keys are present in
        every record whenever any sweep point is an Interaction.  A plain
        ``Interaction(b=[1], nu=[ν])`` keeps its identity and label in
        the records but is routed as the float ν — bit-identical to the
        legacy pass.
    A
        Lattice matrix, shape ``(d, d)``, or the name of a lattice --
        ``"chain"``, ``"square"``, ``"triangular"``, ``"cubic"``.
    n_points
        Number of discretisation points per dimension.

    Other parameters
    ----------------
    momentum
        External momentum, in fractional Brillouin-zone coordinates
        (same convention as :func:`gzl.evaluate_graph`).  Modes:

        * ``None`` (default) — corpus-determined, matching
          :func:`evaluate_graph`'s per-graph default:

          - 0qp corpus (no ``hopping`` field) → scalar per (order, ν)
            at k = 0.
          - 1qp corpus (has ``hopping``) → full BZ grid; trailing
            shape ``(n_points,) * d``.  Aggregates the entire
            single-quasi-particle dispersion in one corpus pass.

          To force a scalar k = 0 result on a 1qp corpus pass an
          explicit ``momentum=np.zeros(d)``.  ``momentum`` and ``nu``
          follow one rule: an argument that asks about a single point
          contributes no axis, and a sequence contributes one.

        * scalar (d=1) or ``(d,)`` ndarray — single k-vector; returned
          arrays have trailing shape ``()``.
        * 1-D array of length ``N`` (d=1) or ``(N, d)`` ndarray — batch
          of N k-vectors; returned arrays have trailing shape ``(N,)``.

        Vacuum (0qp) graphs are k-independent: the scalar value is
        broadcast across the requested ``momentum`` shape so the
        accumulator arithmetic stays uniform across corpora.
    order_max
        If given, truncate the result to orders ≤ ``order_max``.  Default:
        all orders present in the corpus.
    nu_tensor_threshold, tw_threshold, sigma_max, high_tw_fallback, direct_sum_L_list, direct_sum_K
        Deprecated, and without effect.  They tuned a whole-graph
        ``k = 0`` fallback cascade that took over from
        :func:`evaluate_graph` on the graphs it refuses, and that
        cascade is removed: those graphs now raise.  Passing any of them
        emits a :class:`DeprecationWarning`; they will be removed.
    return_diagnostics
        If True, additionally return a diagnostics dict, whose keys and
        values are meant for inspection and are not part of the stable
        API.  It holds route counts and
        per-order wall times.  The
        route counts (``n_<route>``, and so the CLI's diagnostics CSV)
        are recorded for the **first sweep point only** (``nu_index ==
        0``): in a mixed sweep other points may take other routes — a
        compact part disables the bridge / cycle closed forms and lifts
        the grid, a purely compact kernel runs in nearest-neighbour
        mode — and those are not counted.  Per-point routes are in the records
        :func:`evaluate_corpus` returns (``route`` beside ``nu``).
    progress
        If True, print a one-line progress summary per order.
    engine : {"hybrid", "tensor"}, optional
        Block engine forwarded to :func:`gzl.evaluate_graph`.
        ``"hybrid"`` (default) agrees with ``"tensor"`` to round-off (a
        few ulp; they are bit-identical on most but not all inputs) and
        is typically much faster; see that function for the fallback
        rules.

    Returns
    -------
    dict[int, np.ndarray]
        ``{order: coefficient}`` for each order in the corpus, truncated
        to ``order_max``.  Each value has shape ``trailing`` for a single
        coupling and ``(n_nu,) + trailing`` for a sweep, where
        ``trailing`` is set by ``momentum`` as described above.

        **A single coupling has no sweep axis.**  ``nu=3.0``, a NumPy
        scalar, a 0-d array or a single :class:`~gzl.Interaction` ask
        about one coupling, so the answer is that coupling's coefficient
        and nothing is prepended: at ``momentum=np.zeros(d)`` the value
        is the number itself.  A *sweep* -- ``[3.0]``, ``nu_grid(...)``,
        any list or 1-D array, even of length one -- keeps axis 0, so
        code that loops over sweep points never has to ask how ν was
        spelled.  The rule is the argument, never the resulting length:
        ``nu=3.0`` and ``nu=[3.0]`` compute the same numbers and differ
        only in shape.  Write ``[nu]`` to force the axis.

        The arrays are real ``float64``: a graph zeta is real on a
        Bravais lattice whose kernels are real and even, and a
        contribution that is not raises rather than losing its imaginary
        part (see :mod:`gzl._real`).

    See Also
    --------
    nu_grid : ``np.arange``-style helper for building ν grids.
    """
    _warn_cascade_settings("compute_series_coefficients", dict(
        nu_tensor_threshold=nu_tensor_threshold, tw_threshold=tw_threshold,
        sigma_max=sigma_max, high_tw_fallback=high_tw_fallback,
        direct_sum_L_list=direct_sum_L_list, direct_sum_K=direct_sum_K,
    ))
    nu_arr, sweep, scalar_request = _coerce_sweep(nu)
    A = lattice_matrix(A, "compute_series_coefficients")
    d = A.shape[0]
    n_points = _grid_size(n_points, "compute_series_coefficients")

    corpus = _load_corpus(corpus_path)
    available_orders = sorted(corpus.keys())
    if order_max is not None:
        available_orders = [o for o in available_orders if o <= order_max]

    # ``has_hopping`` drives the ``momentum=None`` default; see
    # :func:`_parse_momentum` and :func:`_has_nontrivial_hopping`.
    has_hopping = _has_nontrivial_hopping(corpus, available_orders)
    _mode, mom_for_eval, trailing_shape = _parse_momentum(
        momentum, d, int(n_points), has_hopping,
    )

    n_nu = len(sweep)
    accum_shape = (n_nu,) + trailing_shape
    coefficients: dict[int, np.ndarray] = {
        o: np.zeros(accum_shape, dtype=float) for o in available_orders
    }
    # One block cache for the whole corpus pass.  Block-cut factors
    # ζ_G = Π ζ_block and the same biconnected block (topology +
    # per-bundle ν + A + n_points + momentum) recurs across thousands
    # of graphs — bridges are a single epstein_zeta value, cycles
    # collapse to one per (length, ν), and σ-routed blocks show only a
    # few dozen distinct signatures even at order ≥ 9.  Exact
    # (isomorphism-verified) memoisation, GC'd when this call returns.
    block_cache: dict = {}
    diag = {
        "wall_per_order": {},
        "n_graphs_per_order": {},
    }

    # Group the shared per-graph record stream by order and sum the
    # contributions.  This *is* ``evaluate_corpus``'s pass, consumed
    # lazily: timing each order around its own consumption, the route
    # diagnostics (at ν-index 0).
    records = _iter_corpus_records(
        corpus, available_orders, sweep, A, n_points,
        mom_for_eval, trailing_shape,
        block_cache=block_cache, fast_cycles=fast_cycles,
        richardson=richardson,
        engine=engine, dense_engine=dense_engine,
        sp_n_points=sp_n_points,
        core_grading=core_grading,
    )
    for order, group in itertools.groupby(records, key=lambda r: r["order"]):
        n_graphs = len(corpus[order]["edges_list"])
        diag["n_graphs_per_order"][order] = n_graphs
        t0 = time.perf_counter()
        for rec in group:
            j = rec["nu_index"]
            if return_diagnostics and j == 0:
                route = rec["route"]
                diag[f"n_{route}"] = diag.get(f"n_{route}", 0) + 1
            # Real, and checked rather than assumed: a graph zeta is
            # real on a Bravais lattice with real, even kernels.
            coefficients[order][j] += _as_real(
                rec["contribution"], where="compute_series_coefficients",
            )
        wall = time.perf_counter() - t0
        diag["wall_per_order"][order] = wall
        if progress:
            # stderr: progress is status, not result, and the CLI's
            # `--output -` needs stdout to carry nothing but the CSV.
            print(f"  O{order:>2}: {n_graphs:>5} graphs × {n_nu:>3} ν → "
                  f"{wall:>6.2f}s", file=sys.stderr)

    # A single coupling, asked for as a single coupling, comes back
    # without a sweep axis: `res[order]` IS that coupling's coefficient.
    # A one-point sweep -- `[3.0]`, `nu_grid(3, 3, 1)` -- keeps its axis,
    # so code that loops over sweep points never has to branch.
    if scalar_request:
        coefficients = {o: arr[0] for o, arr in coefficients.items()}

    if return_diagnostics:
        return coefficients, diag
    return coefficients


def evaluate_corpus(
    corpus_path: str | Path,
    nu: float | np.ndarray | Interaction | Sequence[float | Interaction],
    A: "np.ndarray | str",
    n_points: int,
    *,
    momentum=None,
    order_max: int | None = None,
    nu_tensor_threshold: float | None = None,    # deprecated, no effect
    sigma_max: float | None = None,              # deprecated, no effect
    high_tw_fallback: str | None = None,         # deprecated, no effect
    direct_sum_L_list: tuple | None = None,      # deprecated, no effect
    direct_sum_K: int | None = None,             # deprecated, no effect
    fast_cycles: bool = False,
    richardson: bool = True,
    engine: str = "hybrid",
    dense_engine: "str | None" = None,
    sp_n_points: "int | None" = None,
    core_grading: bool = True,
    progress: bool = False,
) -> list[dict]:
    r"""Per-graph evaluation pass over a corpus — the values
    :func:`compute_series_coefficients` sums.

    Where :func:`evaluate_graph` evaluates one graph and
    :func:`compute_series_coefficients` returns the summed coefficient
    ``c_r(k) = Σ_{G ∈ C_r} a_r(G)·ζ_(G,ν)(k)``, ``evaluate_corpus``
    returns **every graph's value** as a flat list of records — one per
    ``(order, graph, ν)``::

        {order, graph_id, nu, nu_index, value, prefactor, contribution,
         route, s, t[, interaction, nu_label]}

    ``value`` is the raw ζ_G (the lattice embedding factor — at
    ``nu = np.inf`` the exact nearest-neighbour homomorphism count), and
    ``contribution = prefactor · value`` is its weighted contribution to
    the order coefficient.  Both are always present; pick raw or weighted
    downstream.  Grouping the records by ``(order, nu_index)`` and summing
    ``contribution`` reproduces :func:`compute_series_coefficients`
    exactly (the two share one iteration and one block cache).

    Parameters mirror :func:`compute_series_coefficients`.
    ``corpus_path``, ``nu``, ``A``, ``n_points``, ``momentum``,
    ``order_max`` and ``progress`` are stable API.  The other
    parameters are provisional: they may change in a minor release (see
    "API stability" in DOCUMENTATION.md).  ``fast_cycles``,
    ``richardson``, ``engine``, ``dense_engine``, ``sp_n_points`` and
    ``core_grading`` select or tune the numerical method.
    ``nu_tensor_threshold``, ``sigma_max``, ``high_tw_fallback``,
    ``direct_sum_L_list`` and ``direct_sum_K`` are deprecated and have
    no effect, as in :func:`compute_series_coefficients`, and a graph
    :func:`evaluate_graph` refuses raises its exception here too.  The record keys are
    stable, while the value of ``route`` names an internal evaluation
    path, meant for inspection, and may change.
    ``corpus_path`` is a corpus file, or a shipped corpus's name:
    ``"tfim0qp"`` / ``"tfim1qp"`` (a ``str``, exactly) for
    ``data_path("tfim_softcore_corpus_0qp.npz")`` / ``..._1qp.npz``.
    ``nu`` is a float (``np.inf`` requests the nearest-neighbour kernel),
    a 1-D array, an :class:`~gzl.Interaction`, or a sequence mixing
    floats and Interactions — a sweep of couplings, one ``nu_index`` per
    point in the order given (a generator / iterator is materialised
    once).  Whenever any point is an Interaction every record carries
    ``interaction`` (the object the user passed, ``None`` for a float
    point) and ``nu_label`` (its ``default_label``), and ``nu`` is the
    TAIL exponent ``min_j ν_j``; a plain ``Interaction(b=[1], nu=[ν])``
    keeps its identity and label here while it is routed as the float ν.
    Each record's ``route`` is the route that point actually took — the
    route counts :func:`compute_series_coefficients` returns under
    ``return_diagnostics`` cover the first sweep point only.  Momentum
    follows the same corpus-determined default (0qp → scalar at k=0;
    1qp → full BZ grid), so for 1qp corpora at grid momentum each
    ``value`` is an ``(n_points,)^d`` array.

    Returns
    -------
    list[dict]
        Per-graph records in (order, graph, ν) order.
    """
    _warn_cascade_settings("evaluate_corpus", dict(
        nu_tensor_threshold=nu_tensor_threshold, sigma_max=sigma_max,
        high_tw_fallback=high_tw_fallback,
        direct_sum_L_list=direct_sum_L_list, direct_sum_K=direct_sum_K,
    ))
    nu_arr, sweep, scalar_request = _coerce_sweep(nu)
    A = lattice_matrix(A, "evaluate_corpus")
    d = A.shape[0]
    n_points = _grid_size(n_points, "evaluate_corpus")

    corpus = _load_corpus(corpus_path)
    available_orders = sorted(corpus.keys())
    if order_max is not None:
        available_orders = [o for o in available_orders if o <= order_max]

    has_hopping = _has_nontrivial_hopping(corpus, available_orders)
    _mode, mom_for_eval, trailing_shape = _parse_momentum(
        momentum, d, int(n_points), has_hopping,
    )

    # One block cache for the whole pass (same factorisation reuse as
    # compute_series_coefficients).
    block_cache: dict = {}
    records = list(_iter_corpus_records(
        corpus, available_orders, sweep, A, n_points,
        mom_for_eval, trailing_shape,
        block_cache=block_cache, fast_cycles=fast_cycles,
        richardson=richardson,
        engine=engine, dense_engine=dense_engine,
        sp_n_points=sp_n_points,
        core_grading=core_grading,
    ))
    if progress:
        print(f"evaluate_corpus: {len(records)} records over "
              f"{len(available_orders)} orders", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
#
# The driver itself lives in ``gzl._cli_series``, which nothing imports.
# ``main`` keeps its flat-argv contract -- ``main(["--config", ...])``, the
# shape every caller and test uses -- and ``gzl series`` reaches the same
# parser through the same function.


def main(argv: Iterable[str] | None = None) -> int:
    """Run the series driver; return a process exit code.

    Equivalent to ``gzl series`` with the same arguments, and kept so
    that ``python -m gzl.series`` goes on working.  That spelling emits
    a :class:`RuntimeWarning` from runpy, because importing ``gzl``
    imports this module and runpy then executes it a second time as
    ``__main__``; ``gzl series`` and ``python -m gzl series`` do not.
    """
    from gzl._cli_series import run

    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
