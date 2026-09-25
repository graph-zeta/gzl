# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Validation of :func:`gzl.direct_sum_zero_momentum` /
:func:`gzl.direct_sum_extrapolated` at d ∈ {2, 3}.

The d > 1 generalisation replaces the 1-D ``a * pos_o``
distance with ``||A @ pos||`` and flattens each vertex's d-dim grid to
a single bucket-elim axis of length ``(2L+1)^d``.  These tests confirm:

1. **Bridge / extrapolation**: a single-edge graph in d ∈ {2, 3}
   matches the closed-form ``epstein_zeta(ν, A, 0, 0)`` after
   Richardson extrapolation in L.  Tests the new factor builder.

2. **Cross-check against the tensor evaluator**: at d = 2, a K_4
   graph matches the tw-agnostic ``graph_zeta_general_at_zero`` to
   within the union of the two methods' residuals.  Tests the
   bucket-elim engine after the d-flat axis switch.

3. **Cycle at d = 3**: a 3-cycle in d = 3 matches ``zeta_circle`` —
   an independent path with no shared code.  Tests the Z₂-symmetry
   marker weights at d > 1.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("epsteinlib")
from epsteinlib import epstein_zeta

from gzl import (
    direct_sum_extrapolated,
    direct_sum_zero_momentum,
    graph_zeta_general_at_zero,
    zeta_circle,
)


A2 = np.eye(2, dtype=float)
A3 = np.eye(3, dtype=float)
ZERO2 = np.zeros(2, dtype=float)
ZERO3 = np.zeros(3, dtype=float)


class TestBridgeExtrapolation:
    """Single-edge graph in d ∈ {2, 3}: direct_sum_extrapolated must
    converge to epstein_zeta(ν, A, 0, 0).  At ν = 4, d = 2 the
    leading residual is L^(-2); at ν = 5, d = 3 it's L^(-2).  A
    4-point L-list with K = 2 correction terms is more than enough.
    """

    def test_bridge_d2(self):
        edges = np.array([[0, 1]], dtype=int)
        nu = np.array([4.0])
        L_list = (3, 4, 5, 6, 7)
        val = direct_sum_extrapolated(
            edges, nu, A2, L_list=L_list, n_correction_terms=2,
        )
        exact = float(epstein_zeta(4.0, A2, ZERO2, ZERO2).real)
        rel = abs(val.real - exact) / abs(exact)
        # Residual ~L^-(ν-d) = L^-2 at L_max = 7 → 2%; Richardson with
        # K=2 correction terms typically buys 2-3 digits → ~5e-4.
        assert rel < 1e-3, (
            f"bridge d=2: extrapolated {val.real:.6e} vs epstein "
            f"{exact:.6e}, rel = {rel:.3e}"
        )

    def test_bridge_d3(self):
        edges = np.array([[0, 1]], dtype=int)
        nu = np.array([5.0])
        # d=3 is memory-hungry: bag tensor at L=5 is 11^3 = 1331 floats
        # for a 1-axis edge factor, OK; but the 2-vertex edge at L=5
        # would be 11^3 × 11^3 = ~2M floats = 16 MB.  Stay at L ≤ 5.
        L_list = (3, 4, 5)
        val = direct_sum_extrapolated(
            edges, nu, A3, L_list=L_list, n_correction_terms=1,
        )
        exact = float(epstein_zeta(5.0, A3, ZERO3, ZERO3).real)
        rel = abs(val.real - exact) / abs(exact)
        # Residual ~L^-(ν-d) = L^-2; K=1 correction at L_max=5 → ~3e-3.
        assert rel < 1e-2, (
            f"bridge d=3: extrapolated {val.real:.6e} vs epstein "
            f"{exact:.6e}, rel = {rel:.3e}"
        )


class TestCycleAgainstZetaCircle:
    """A 3-cycle at d = 3 with ν = 5.  Compares the d=3 direct sum
    against ``zeta_circle``, which uses Duffy quadrature and shares
    no code with the bucket-elim engine.
    """

    def test_three_cycle_d3(self):
        edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)
        nu = np.array([5.0] * 3)
        # 3-cycle is tw=2.  Bag at L=4: free-vertex axes of length
        # 9^3 = 729 each, so a (729, 729) edge factor = 4 MB.  OK.
        L_list = (3, 4, 5)
        val = direct_sum_extrapolated(
            edges, nu, A3, L_list=L_list, n_correction_terms=1,
        )
        circle = float(zeta_circle([5.0] * 3, A3))
        rel = abs(val.real - circle) / abs(circle)
        assert rel < 5e-3, (
            f"3-cycle d=3: direct-sum {val.real:.6e} vs zeta_circle "
            f"{circle:.6e}, rel = {rel:.3e}"
        )


class TestK4AgainstTensor:
    """K_4 at d = 2: cross-check the new d=2 direct sum against the
    tensor evaluator.  Both are independent truncation schemes (box
    vs torus); they should agree to within the union of their
    residuals.
    """

    def test_k4_d2(self):
        edges = np.array(
            [[i, j] for i in range(4) for j in range(i + 1, 4)],
            dtype=int,
        )
        nu = np.full(len(edges), 5.0)
        # K_4 has tw=3 → bag is 4 vertices → free-vertex tensor at L=3
        # has shape (7^2,) ** 3 = 49 ** 3 = ~118k floats = 1 MB.  OK.
        val_ds = direct_sum_extrapolated(
            edges, nu, A2, L_list=(3, 4, 5), n_correction_terms=1,
        )
        val_tn = float(np.asarray(
            graph_zeta_general_at_zero(edges, nu, A2, 32)
        ).real)
        rel = abs(val_ds.real - val_tn) / abs(val_tn)
        # Both have ~n^-(ν-d) ~ 1/L^3 and 1/n^3 residuals; agreement
        # to ~1e-3 is what we expect.
        assert rel < 5e-3, (
            f"K_4 d=2: direct-sum {val_ds.real:.6e} vs tensor "
            f"{val_tn:.6e}, rel = {rel:.3e}"
        )


# ---------------------------------------------------------------------------
# Degree-aware Richardson exponent: dense graphs must beat the old
# misspecified `d - min_edge_ν` basis that floored K_5 at ~5e-4.
# ---------------------------------------------------------------------------

class TestDenseExtrapolationExponent:
    """`direct_sum_extrapolated` now fits the leading box-truncation
    power `d - min_{v free} Σ_{e∋v} ν_e`.  For K_5 (every summed
    vertex degree 4) the true tail is `|x|^-(4ν)`; the old
    `d - min_edge_ν` basis could not represent it and floored the fit
    at ~5e-4 regardless of L.  With the correct exponent a *small* L
    reaches the converged tensor value.
    """

    NU = 2.0 + np.pi / 30        # σ ≈ 1.1, off the Γ-pole

    def _k(self, n):
        return np.array([[i, j] for i in range(n) for j in range(i + 1, n)],
                        dtype=int)

    def test_k5_d1_beats_old_floor(self):
        E = self._k(5)
        nv = np.full(len(E), self.NU)
        A1 = np.array([[1.0]])
        ds = direct_sum_extrapolated(
            E, nv, A1, L_list=(4, 5, 6, 7, 8), n_correction_terms=3,
        ).real
        # K_5 (deg-4) converges as n^-(4ν-1) ≈ n^-7, so n=32 is already
        # ~1e-9 accurate — a fast yet effectively exact reference.
        ref = float(np.real(graph_zeta_general_at_zero(E, nv, A1, 32)))
        rel = abs(ds - ref) / abs(ref)
        # Old misspecified basis floored at ~5e-4 for any L; the
        # corrected exponent must do far better at small L.  Measured
        # 1.1e-5, so 1e-4 separates the two.
        assert rel < 1e-4, (
            f"K_5 d=1 corrected-exponent direct-sum {ds:.8e} vs tensor "
            f"{ref:.8e}, rel = {rel:.2e} (old floor was ~5e-4)"
        )

    def test_k5_d2(self):
        E = self._k(5)
        nv = np.full(len(E), self.NU)
        ds = direct_sum_extrapolated(
            E, nv, A2, L_list=(3, 4, 5, 6), n_correction_terms=2,
        ).real
        # K_5 d=2 converges as n^-(4ν-2) ≈ n^-6, so n=16 is already a
        # ~1e-7 reference and far cheaper than n=24 at tw=4, d=2.
        ref = float(np.real(graph_zeta_general_at_zero(E, nv, A2, 16)))
        rel = abs(ds - ref) / abs(ref)
        # Measured 2.3e-5.
        assert rel < 1e-4, (
            f"K_5 d=2: direct-sum {ds:.8e} vs tensor {ref:.8e}, "
            f"rel = {rel:.2e}"
        )


# ---------------------------------------------------------------------------
# Finite-momentum cos-phased direct sum vs the independent tensor
# cos-weight single-k path.
# ---------------------------------------------------------------------------

class TestFiniteMomentumDirectSum:
    """`direct_sum_extrapolated(root=s, terminal=t, momentum=k)`
    multiplies the summand by the real cos(2π k·x_t) weight (x_s = 0).
    It must agree with `graph_zeta_general(..., momentum=k)` (an
    independent torus + cos-weight path) to the union of their
    truncation residuals, and require an explicit root.
    """

    NU = 2.0 + np.pi / 30

    def test_k4_d1_finite_k_vs_tensor(self):
        from gzl import graph_zeta_general
        E = np.array([[i, j] for i in range(4) for j in range(i + 1, 4)],
                     dtype=int)
        nv = np.full(len(E), self.NU)
        A1 = np.array([[1.0]])
        for kf in (0.0, 0.17, 0.5):
            k = np.array([kf])
            ds = direct_sum_extrapolated(
                E, nv, A1, L_list=(4, 5, 6, 7, 8), n_correction_terms=3,
                root=0, terminal=3, momentum=k,
            ).real
            tn = float(np.real(graph_zeta_general(
                E, nv, A1, 48, source=0, terminals=(3,), momentum=k,
            )))
            rel = abs(ds - tn) / abs(tn)
            assert rel < 5e-3, (
                f"K_4 d=1 k={kf}: phased direct-sum {ds:.6e} vs "
                f"tensor cos-weight {tn:.6e}, rel = {rel:.2e}"
            )

    def test_finite_k_requires_explicit_root(self):
        E = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)
        nv = np.full(3, self.NU)
        with pytest.raises(ValueError, match="explicit `root`"):
            direct_sum_zero_momentum(
                E, nv, np.array([[1.0]]), L=4,
                terminal=2, momentum=np.array([0.2]),
            )


# ---------------------------------------------------------------------------
# Frontend routing of dense tw≥3 blocks (torus by default, box on request).
# ---------------------------------------------------------------------------

class TestFrontendDenseRouting:
    """`evaluate_graph` routes a K_5 block, and returns the right value
    whichever engine it is sent to.

    Dense blocks take the torus by default and the box on request; the
    value assertion is what actually guards correctness — it is checked
    against the tensor engine at a finer grid, so it holds on either
    route.
    """

    NU = 2.0 + np.pi / 30

    def test_k5_routed_to_the_torus(self):
        from gzl import evaluate_graph
        E = np.array([[i, j] for i in range(5) for j in range(i + 1, 5)],
                     dtype=int)
        A1 = np.array([[1.0]])
        val, info = evaluate_graph(
            E, self.NU, A1, n_points=16, return_diagnostics=True,
        )
        assert info["n_block_direct_sum"] == 0, info
        assert info["n_block_dense_torus"] == 1, info
        ref = float(np.real(graph_zeta_general_at_zero(
            E, np.full(len(E), self.NU), A1, 32,
        )))
        assert abs(val - ref) / abs(ref) < 1e-3, (
            f"routed K_5 value {val:.8e} vs tensor {ref:.8e}"
        )

    def test_k5_still_reaches_the_box_on_request(self):
        from gzl import evaluate_graph
        E = np.array([[i, j] for i in range(5) for j in range(i + 1, 5)],
                     dtype=int)
        A1 = np.array([[1.0]])
        val, info = evaluate_graph(
            E, self.NU, A1, n_points=16, dense_engine="direct_sum",
            return_diagnostics=True,
        )
        assert info["n_block_direct_sum"] == 1, info
        assert info["n_block_dense_torus"] == 0, info
        ref = float(np.real(graph_zeta_general_at_zero(
            E, np.full(len(E), self.NU), A1, 32,
        )))
        assert abs(val - ref) / abs(ref) < 1e-3


# ---------------------------------------------------------------------------
# Single-shot full-BZ direct sum (terminal kept open) — must equal the
# per-k phased extrapolation exactly (it is the same math, factored so
# the k-independent elimination runs once), preserving the headline
# single-shot-BZ guarantee for dense on-spine blocks.
# ---------------------------------------------------------------------------

class TestSingleShotGrid:
    NU = 2.0 + np.pi / 30

    def test_grid_equals_per_k_loop(self):
        from gzl.direct_sum import (
            _direct_sum_extrapolated_grid,
            direct_sum_extrapolated,
        )
        E = np.array([[i, j] for i in range(4) for j in range(i + 1, 4)],
                     dtype=int)
        nv = np.full(len(E), self.NU)
        A1 = np.array([[1.0]])
        n = 16
        kk = (np.arange(n) / n).reshape(-1, 1)
        ss = _direct_sum_extrapolated_grid(
            E, nv, A1, kk, root=0, terminal=3,
            L_list=(4, 5, 6, 7, 8), n_correction_terms=3,
        )
        loop = np.array([
            direct_sum_extrapolated(
                E, nv, A1, L_list=(4, 5, 6, 7, 8),
                n_correction_terms=3, root=0, terminal=3, momentum=k,
            ).real
            for k in kk
        ])
        # Same math, just factored ⇒ agreement at machine precision.
        assert np.max(np.abs(ss - loop)) < 1e-12, (
            f"single-shot vs per-k loop max |Δ| = "
            f"{np.max(np.abs(ss - loop)):.2e}"
        )

    def test_grid_matches_tensor_space_k(self):
        from gzl import graph_zeta_general
        from gzl.direct_sum import _direct_sum_extrapolated_grid
        E = np.array([[i, j] for i in range(4) for j in range(i + 1, 4)],
                     dtype=int)
        nv = np.full(len(E), self.NU)
        A1 = np.array([[1.0]])
        n = 16
        kk = (np.arange(n) / n).reshape(-1, 1)
        ss = _direct_sum_extrapolated_grid(
            E, nv, A1, kk, root=0, terminal=3,
            L_list=(4, 5, 6, 7, 8), n_correction_terms=3,
        )
        tn = np.asarray(graph_zeta_general(
            E, nv, A1, n, source=0, terminals=(3,), space='k',
        )).reshape(-1).real
        # Independent torus path at the same coarse grid; agreement to
        # the union of the two small-truncation residuals.
        rel = np.max(np.abs(ss - tn)) / np.max(np.abs(tn))
        assert rel < 5e-3, f"single-shot vs tensor grid rel = {rel:.2e}"

    def test_full_bz_via_evaluate_graph(self):
        """An on-spine dense block reached through the public 1qp
        full-BZ path returns the whole grid (single shot) without
        falling back."""
        from gzl import evaluate_graph
        # K_5 with a distinct source/terminal pair on the spine.
        E = np.array([[i, j] for i in range(5) for j in range(i + 1, 5)],
                     dtype=int)
        A1 = np.array([[1.0]])
        n = 16
        grid, info = evaluate_graph(
            E, self.NU, A1, source=0, terminal=4,
            n_points=n, return_diagnostics=True,
        )
        grid = np.asarray(grid)
        assert grid.shape == (n,), grid.shape
        assert np.all(np.isfinite(grid))
        # k = 0 cell must equal the k=0 scalar route within tolerance.
        z0 = float(evaluate_graph(
            E, self.NU, A1, source=0, terminal=4,
            momentum=np.zeros(1), n_points=n,
        ))
        assert abs(grid[0] - z0) / abs(z0) < 5e-3, (
            f"grid[0]={grid[0]:.6e} vs single-k k=0 {z0:.6e}"
        )
