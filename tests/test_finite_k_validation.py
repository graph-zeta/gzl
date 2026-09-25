# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Finite-momentum validation tests for :func:`evaluate_graph`.

Two families:

1. **2D-lattice cross-checks.**  Non-trivial multi-vertex graphs
   (triangle on the s-t spine, diamond, K_4) on 2D square and
   hexagonal lattices, compared against a brute-force direct
   lattice sum at finite k.  The direct-sum reference is exact
   modulo the box-truncation residual ``L^(d − ν_min)``; we choose
   σ and ``L`` so the residual sits well below the target tolerance.

2. **TFIM 1qp gap-series regression at finite momentum.**  Iterates
   the shipped 1qp corpus and accumulates the per-order coefficient at
   the antiferromagnetic mode k = π of the chain across the published
   σ values, and at k = 0 on the square lattice (physics convention
   ``k_phys``; the library's fractional convention is
   ``k_frac = k_phys / (2π)``).  Compares against the Monte Carlo
   references shipped in ``gzl/data/MC_patched/``.

The 2D cross-checks take about 2 s together and run in the default
suite.  The MC comparisons are marked ``@pytest.mark.slow``: each runs
the corpus once.  Opt in via ``pytest --runslow``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from gzl import data_path, evaluate_graph
from gzl.series import _load_corpus


CORPUS_NPZ_1QP = data_path("tfim_softcore_corpus_1qp.npz")


def _mc_file(lattice: str, name: str) -> Path:
    """A Monte Carlo reference shipped in ``gzl/data/MC_patched/TFIM``."""
    return data_path(f"MC_patched/TFIM/{lattice}/{name}")


# ---------------------------------------------------------------------------
# Brute-force direct lattice sum at finite k — slow-but-correct reference
# ---------------------------------------------------------------------------

def _direct_sum_finite_k(
    edges: np.ndarray,
    nu_vec: np.ndarray,
    A: np.ndarray,
    L: int,
    source: int,
    terminal: int,
    k_frac: np.ndarray,
) -> float:
    r"""Brute-force lattice sum

    .. math::

        \zeta_G(\bm{k}_\text{frac}) \;\approx\;
        \sum^\prime_{\bm{z}^{(v)} \in [-L, L]^d \,\forall\, v \ne s}
        e^{-2\pi i\,\bm{k}_\text{frac}\cdot(\bm{z}^{(t)} - \bm{z}^{(s)})}
        \prod_{e \in E} \lvert A(\bm{z}^{(v_e)} - \bm{z}^{(u_e)}) \rvert^{-\nu_e}

    with the source pinned at ``z=0`` and every other vertex's lattice
    multi-index summed over the box ``[-L, L]^d``.  Returns the real
    part (lattice inversion symmetry guarantees the imaginary part is
    zero up to the truncation).

    Convergence to the infinite-lattice limit is algebraic with
    leading order ``L^(d − ν_min)``; pick ``L`` and ``ν`` so the
    residual is tighter than the test tolerance.

    Memory bound: ``(2L+1)^(d · n_free)`` complex doubles, where
    ``n_free = V − 1``.  For d = 2 and V = 4 with L = 5 this is
    ``11^6 ≈ 1.77 M`` — feasible.  For larger graphs raise an error.
    """
    edges = np.asarray(edges, dtype=int)
    nu_vec = np.asarray(nu_vec, dtype=float)
    A = np.asarray(A, dtype=float)
    d = int(A.shape[0])
    V = int(edges.max()) + 1
    if V > 5:
        raise ValueError(
            f"_direct_sum_finite_k: V = {V} too large for brute force."
        )

    free = [v for v in range(V) if v != source]
    n_free = len(free)
    M = 2 * L + 1
    n_configs = M ** (n_free * d)
    if n_configs > 5_000_000:
        raise ValueError(
            f"_direct_sum_finite_k: {n_configs:.2e} configs would "
            f"blow memory; lower V, d, or L."
        )

    # Build the (n_configs, n_free, d) array of all position
    # assignments via meshgrid in flattened form.
    coords = np.arange(-L, L + 1, dtype=int)
    grids = np.meshgrid(*[coords] * (n_free * d), indexing="ij")
    pos = np.stack([g.ravel() for g in grids], axis=-1)
    pos = pos.reshape(n_configs, n_free, d)

    # Per-vertex position tensor: (n_configs, V, d).  Source row is
    # zeros throughout.
    z_vert = np.zeros((n_configs, V, d), dtype=int)
    for fi, v in enumerate(free):
        z_vert[:, v, :] = pos[:, fi, :]

    # Edge propagator product in log space (avoids underflow at high
    # ν for large displacements).  Mark configurations with any
    # zero-displacement edge as invalid.
    log_w = np.zeros(n_configs, dtype=float)
    valid = np.ones(n_configs, dtype=bool)
    for i, (u, v) in enumerate(edges):
        dz = z_vert[:, v, :] - z_vert[:, u, :]
        dx = dz @ A.T
        norm = np.linalg.norm(dx, axis=-1)
        bad = norm == 0.0
        valid &= ~bad
        # Avoid log(0) — entries with bad=True will be masked anyway.
        safe_norm = np.where(bad, 1.0, norm)
        log_w += -nu_vec[i] * np.log(safe_norm)

    # Phase: exp(-2πi k_frac · (z_t - z_s)).  z_s = 0 by pinning, so
    # this reduces to exp(-2πi k_frac · z_t).
    z_t = z_vert[:, terminal, :].astype(float)
    k_arr = np.asarray(k_frac, dtype=float).reshape(d)
    phase = np.exp(-2j * np.pi * z_t @ k_arr)

    weights = np.zeros(n_configs, dtype=complex)
    weights[valid] = np.exp(log_w[valid]) * phase[valid]
    return float(weights.sum().real)


# ---------------------------------------------------------------------------
# 2D lattice cross-checks
# ---------------------------------------------------------------------------

# 2D square lattice basis.
A_SQ = np.array([[1.0, 0.0], [0.0, 1.0]])

# 2D hexagonal lattice basis.
A_HEX = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2.0]])


class TestTwoDFiniteKAgainstDirectSum:
    """Cross-check the topology-first router's finite-k path against
    a brute-force direct lattice sum on small 2D graphs.

    We pick σ large enough that the direct-sum truncation residual
    ``L^(d − ν_min)`` sits below 1e-3 at the chosen ``L``, and run
    `evaluate_graph` at n_points high enough that its FFT-truncation
    residual is similarly small."""

    @pytest.mark.parametrize("A_lattice,name", [
        (A_SQ, "square"), (A_HEX, "hexagonal"),
    ])
    @pytest.mark.parametrize("k_frac", [
        (0.0, 0.0), (0.25, 0.0), (0.25, 0.25),
        (0.123, 0.456), (-0.31, 0.17),
    ])
    def test_bridge_2d(self, A_lattice, name, k_frac):
        # Single bridge: ζ = epstein_zeta(ν, A, 0, k_lat) (analytic).
        # No truncation in the direct sum at all — it's exact.
        from epsteinlib import epstein_zeta
        edges = np.array([[0, 1]], dtype=int)
        nu = 4.0
        k = np.asarray(k_frac, dtype=float)

        v = evaluate_graph(edges, nu, A_lattice, terminal=1, momentum=k)

        Astar = np.linalg.inv(A_lattice.T)
        k_lat = Astar @ k
        ref = float(epstein_zeta(nu, A_lattice, np.zeros(2), k_lat).real)

        assert v == pytest.approx(ref, rel=1e-10, abs=1e-12), (
            f"{name} bridge at k={k_frac}: v={v} vs ref={ref}"
        )

    @pytest.mark.parametrize("A_lattice,name", [
        (A_SQ, "square"), (A_HEX, "hexagonal"),
    ])
    @pytest.mark.parametrize("k_frac", [(0.0, 0.0), (0.25, 0.0), (0.123, 0.456)])
    def test_triangle_on_spine_2d(self, A_lattice, name, k_frac):
        # Triangle 0-1-2 with s = 0, t = 1.  The block has both s and
        # t on the spine, so it routes through the σ-routed path
        # (zeta_circle is k=0 only).  Verifies the algebra/tensor
        # finite-k path on a small 2D graph.
        edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)
        nu_vec = np.full(3, 6.0)              # σ = 4 in d = 2
        k = np.asarray(k_frac, dtype=float)

        v = evaluate_graph(
            edges, 6.0, A_lattice, source=0, terminal=1,
            momentum=k, n_points=64,
        )

        # Direct sum: V=3, n_free=2, d=2, L=10 → 21^4 = 194 481 configs.
        # ν_min - d = 4 ⇒ residual ~ L^(-4) = 10^(-4).
        ref = _direct_sum_finite_k(
            edges, nu_vec, A_lattice, L=10,
            source=0, terminal=1, k_frac=k,
        )

        assert v == pytest.approx(ref, rel=5e-3, abs=1e-5), (
            f"{name} triangle at k={k_frac}: v={v} vs ref={ref}"
        )

    @pytest.mark.parametrize("k_frac", [(0.0, 0.0), (0.25, 0.0), (0.31, 0.17)])
    def test_diamond_2d_square(self, k_frac):
        # Diamond (4-cycle 0-1-2-3 with chord 0-2), s = 0, t = 2.
        # tw = 2, SP-reducible at the chord endpoints — algebra path.
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=int,
        )
        nu_vec = np.full(5, 6.0)             # σ = 4 in d = 2
        k = np.asarray(k_frac, dtype=float)

        v = evaluate_graph(
            edges, 6.0, A_SQ, source=0, terminal=2,
            momentum=k, n_points=64,
        )

        # V=4, n_free=3, d=2, L=5 → 11^6 ≈ 1.77 M configs.
        ref = _direct_sum_finite_k(
            edges, nu_vec, A_SQ, L=5,
            source=0, terminal=2, k_frac=k,
        )
        # L=5 truncation residual ~ L^(-4) = 1.6e-3.
        assert v == pytest.approx(ref, rel=2e-2, abs=1e-3), (
            f"diamond@square at k={k_frac}: v={v} vs ref={ref}"
        )

    def test_K4_on_spine_2d_square(self):
        # K_4 with terminals (0, 3): tw = 3, so the block takes the dense
        # torus route at finite k (pinned in
        # test_frontend_finite_k.TestOnSpineDiagnostics).  This test
        # checks the VALUE, against the brute-force _direct_sum_finite_k.
        edges = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=int,
        )
        nu_vec = np.full(6, 6.0)             # σ = 4 in d = 2
        k = np.array([0.123, 0.456])

        v = evaluate_graph(
            edges, 6.0, A_SQ, source=0, terminal=3,
            momentum=k, n_points=48,
        )

        # V=4, n_free=3, d=2, L=4 → 9^6 = 531 441 configs.
        ref = _direct_sum_finite_k(
            edges, nu_vec, A_SQ, L=4,
            source=0, terminal=3, k_frac=k,
        )
        # L=4 truncation residual ~ L^(-4) = 4e-3.
        assert v == pytest.approx(ref, rel=5e-2, abs=1e-3), (
            f"K_4@square at k={k}: v={v} vs ref={ref}"
        )


# ---------------------------------------------------------------------------
# TFIM 1qp gap-series regression against Monte Carlo
# ---------------------------------------------------------------------------

def _read_mc_csv(path: Path) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["order"])] = (
                float(row["prefactor"]),
                float(row["error"]),
            )
    return out


def _accumulate_1qp_at_k(
    corpus_path: Path,
    nu: float,
    A: np.ndarray,
    n_points: int,
    k_frac: np.ndarray,
    order_max: int,
) -> dict[int, float]:
    """Iterate the 1qp corpus and accumulate
    ``Σ_G prefactor_G · ζ_G(k_frac; ν)`` per order.

    Vacuum (s == t) graphs contribute ζ_G(0); 1qp (s ≠ t) graphs
    contribute ζ_G(k_frac).  Returns ``{order: real coefficient}``.
    """
    corpus = _load_corpus(corpus_path)
    d = int(A.shape[0])
    out: dict[int, float] = {}
    for order, block in corpus.items():
        if order_max is not None and order > order_max:
            continue
        edges_list = block["edges_list"]
        mults_list = block["multiplicities_list"]
        prefactors = block["prefactor"]
        hopping = block.get("hopping")
        n_graphs = len(edges_list)

        accum = 0.0
        for i in range(n_graphs):
            edges = edges_list[i]
            mults = mults_list[i]
            edges_flat = np.repeat(edges, mults, axis=0)
            pref = float(prefactors[i])
            if hopping is not None:
                s_v = int(hopping[i][0])
                t_v = int(hopping[i][1])
            else:
                s_v, t_v = 0, 0
            if s_v == t_v:
                val = evaluate_graph(
                    edges_flat, nu, A,
                    source=s_v, terminal=None,
                    n_points=n_points,
                )
            else:
                val = evaluate_graph(
                    edges_flat, nu, A,
                    source=s_v, terminal=int(t_v),
                    momentum=np.asarray(k_frac, dtype=float),
                    n_points=n_points,
                )
            accum += pref * val
        out[order] = float(accum)
    return out


def _read_mc_dat(path: Path) -> dict[int, tuple[float, float]]:
    """Read a tab-separated MC `.dat` file with rows
    ``order \\t prefactor \\t statistical_error``.  No header.

    These ship in ``gzl/data/MC_patched`` alongside the per-order
    CSVs; the `.dat` files use the lattice-exponent ``α = ν``
    directly in their filenames (so on the 1D chain
    ``σ = α − 1``).
    """
    out: dict[int, tuple[float, float]] = {}
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            order = int(parts[0])
            val = float(parts[1])
            err = float(parts[2])
            out[order] = (val, err)
    return out


# ---------------------------------------------------------------------------
# Helpers for 1qp finite-k tests on an arbitrary lattice
# ---------------------------------------------------------------------------

def _check_1qp_dispersion(
    nu: float,
    A: np.ndarray,
    n_points: int,
    k_phys: np.ndarray,
    mc: dict[int, tuple[float, float]],
    order_max: int,
    label: str,
):
    """Compare the topology-first router's 1qp dispersion at a given
    (lattice, ν, k_phys) against the MC reference dict, for orders
    2..order_max.  Tolerance ``max(10·MC_err, 1e-5·|MC|, 1e-12)``;
    placeholder rows ``(0, 0)`` are skipped.

    ``k_phys`` is the MC's labelled momentum, which (per the MC files'
    filename convention ``kxX_kyY``) gives the dual-basis
    fractional coordinates × 2π — *not* the Cartesian wavevector.
    The library's ``momentum`` parameter is exactly ``k_phys / (2π)``;
    no A-multiplication is required.

    For an orthonormal lattice the two conventions coincide; for the
    triangular basis they diverge by ``A.T``.  Empirically validated
    by matching the Nsites-summed k = 0 triangular MC at σ = 2 to
    five digits at order 2..4 with this convention, and by matching
    the finite-k triangular MC at ``(kx, ky) = (2π/3, -2π/3)`` to
    four digits at order 2.
    """
    A = np.asarray(A, dtype=float)
    d = int(A.shape[0])
    k_phys = np.asarray(k_phys, dtype=float).reshape(d)
    k_frac = k_phys / (2.0 * np.pi)

    result = _accumulate_1qp_at_k(
        CORPUS_NPZ_1QP,
        nu=nu, A=A, n_points=n_points,
        k_frac=k_frac, order_max=order_max,
    )

    compared = 0
    for order, val in sorted(result.items()):
        if order not in mc or order < 2:
            continue
        mc_val, mc_err = mc[order]
        if mc_val == 0.0 and mc_err == 0.0:
            continue
        tol = max(10.0 * mc_err, 1e-5 * abs(mc_val), 1e-12)
        assert abs(val - mc_val) < tol, (
            f"{label} order={order}: TN={val:.10e}, "
            f"MC={mc_val:.10e}±{mc_err:.2e}, "
            f"|diff|={abs(val - mc_val):.3e} > tol={tol:.3e}"
        )
        compared += 1
    assert compared >= 3, (
        f"{label}: only {compared} orders compared (expected ≥ 3)."
    )


# ---------------------------------------------------------------------------
# 1D chain at the antiferromagnetic mode k = π via .dat
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestOneQpFiniteKSmallSigma1D:
    """1D chain 1qp gap at the antiferromagnetic (staggered) mode
    k = π across every published σ value, from the ``.dat`` files
    (``alpha`` is the lattice exponent ν, so on the chain
    ``σ = α − 1``).  This is the physically interesting mode (the 1qp
    gap closes at k = π at the AFM critical point) and the
    best-sampled k-direction in the MC data.
    """

    A1D = np.array([[1.0]])
    N_POINTS = 64

    # Per-σ MC files at k = π, indexed by α = σ + 1 (filename uses α
    # verbatim): the nine published σ-values from σ = 0.5 up to σ = 6.
    _KPI_SIGMA_ALPHA = [
        (0.5,   "1.5"),
        (2.0/3, "1.66667"),
        (1.0,   "2"),
        (1.25,  "2.25"),
        (1.5,   "2.5"),
        (2.0,   "3"),
        (3.0,   "4"),
        (4.0,   "5"),
        (6.0,   "7"),
    ]

    @pytest.mark.parametrize(
        "sigma,alpha",
        _KPI_SIGMA_ALPHA,
        ids=[f"sigma{s:g}" for s, _ in _KPI_SIGMA_ALPHA],
    )
    def test_kPi_full_sigma_sweep(self, sigma: float, alpha: str):
        """1qp gap at k = π (the antiferromagnetic mode on the 1D
        chain) across the published σ-sweep."""
        dat_path = _mc_file(
            "chain",
            f"TFIM_lr_1d__MC__list_of_prefactors__kPi__alpha_{alpha}.dat"
        )
        mc = _read_mc_dat(dat_path)

        nu = float(sigma) + 1.0           # d = 1
        k_phys = np.array([np.pi])
        _check_1qp_dispersion(
            nu=nu, A=self.A1D, n_points=self.N_POINTS,
            k_phys=k_phys, mc=mc, order_max=8,
            label=f"1qp 1D AFM σ={sigma:g} kPi",
        )


# ---------------------------------------------------------------------------
# 2D-lattice 1qp regressions: corpus is lattice-independent — we just
# pass the appropriate A matrix.
# ---------------------------------------------------------------------------

A_SQUARE = np.array([[1.0, 0.0], [0.0, 1.0]])


@pytest.mark.slow
class TestOneQpTwoDLatticeVsMC:
    """1qp gap-series regression on the square lattice using the same
    lattice-independent corpus as the 1D suite.

    ``order_max = 5``: bucket-elimination cost on 2D blocks scales as
    ``n^((tw+1)·d)`` and the corpus grows quickly past order 5 (an
    order-8 sweep is several minutes on a laptop).  Five orders still
    pin the dominant per-order coefficients well within the MC band
    and the validation cost stays in the slow-suite budget."""

    N_POINTS = 32
    ORDER_MAX = 5

    def test_square_kx0_ky0_sigma2(self):
        # 2D square TFIM 1qp at k = (0, 0), σ = 2 (ν = σ + d = 4).
        # Vacuum-like in k but the on-spine block is genuinely 2D
        # (the lattice basis enters the per-block evaluators).
        mc_path = _mc_file(
            "square",
            "1qp_gap_series_2d_square_tfim_kx0_ky0_sigma2_order11.csv"
        )
        mc = _read_mc_csv(mc_path)
        _check_1qp_dispersion(
            nu=4.0, A=A_SQUARE, n_points=self.N_POINTS,
            k_phys=np.zeros(2), mc=mc, order_max=self.ORDER_MAX,
            label="1qp 2D-square (0,0) σ=2",
        )
