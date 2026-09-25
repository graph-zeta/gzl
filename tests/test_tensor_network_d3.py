# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Validation of :func:`gzl.graph_zeta_general` at d = 3.

The tensor evaluator is not restricted to d ∈ {1, 2}; this module
exercises four cases on the simple cubic lattice (A = I₃) so a regression
that re-introduces a d-restriction (or breaks the float64 k=0 path) is
caught immediately.

The reference values come from independent paths that do not share code
with `graph_zeta_general`:

* the closed-form Epstein zeta for a bridge,
* the Duffy-quadrature `zeta_circle` for a simple cycle,
* the same `graph_zeta_general` value at higher discretisation (Cauchy)
  for K₄.

Tolerances reflect the trapezoid-rule residual ``~ n^-(2-σ)`` of the
periodisation; at d = 3, σ = 1 (ν = 4) we expect ~1e-2 residuals at
``n = 16`` which dominates real disagreement with the references.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("epsteinlib")
from epsteinlib import epstein_zeta

from gzl import (
    evaluate_graph,
    graph_zeta_general,
    graph_zeta_general_at_zero,
    zeta_circle,
)


A3 = np.eye(3, dtype=float)
ZERO3 = np.zeros(3, dtype=float)


class TestBridgeAgainstEpsteinZeta:
    """Single-edge graph at d = 3: G has 2 vertices, 1 edge.  By
    construction `graph_zeta_general_at_zero(edges, [ν], A, n)` is the
    truncated Epstein zeta of ν on the cubic torus, which converges to
    `epstein_zeta(ν, A, 0, 0)` at rate ``n^-(2-σ)`` for σ < 2 and
    exponentially in n for σ ≥ 2 (smooth integrand regime).

    We use ν = 5 (σ = 2) for the value test so the periodisation
    residual at n = 16 is ~ 1/16² ≈ 0.4%.  Tolerance is set at 2% to
    leave a safety margin while still failing if the evaluator
    re-introduces a 10%-scale error.  The Cauchy convergence test at
    ν = 4 (σ = 1) checks the algebraic decay regime — failures there
    would indicate the elimination engine has lost an order of
    accuracy, not just that the discretisation is coarse.
    """

    NU_VALUE = 5.0      # σ = 2, residual ~ 1/n²
    NU_CAUCHY = 4.0     # σ = 1, residual ~ 1/n
    N_POINTS = 16

    def _bridge_value(self, nu: float, n: int) -> complex:
        edges = np.array([[0, 1]], dtype=int)
        return graph_zeta_general_at_zero(edges, np.array([nu]), A3, n)

    def test_real_within_few_percent_of_epstein(self):
        """tensor evaluator value ≈ closed-form epstein_zeta at ν = 5."""
        approx = float(np.asarray(self._bridge_value(self.NU_VALUE, self.N_POINTS)).real)
        exact = float(epstein_zeta(self.NU_VALUE, A3, ZERO3, ZERO3).real)
        rel_err = abs(approx - exact) / abs(exact)
        assert rel_err < 2e-2, (
            f"bridge d=3 at ν={self.NU_VALUE}, n={self.N_POINTS}: tensor "
            f"value {approx:.6e} vs epstein {exact:.6e}, rel_err = {rel_err:.3e}"
        )

    def test_cauchy_convergence_to_epstein(self):
        """Doubling n_points must approach the closed-form Epstein value."""
        v_low = float(np.asarray(self._bridge_value(self.NU_CAUCHY, 8)).real)
        v_hi = float(np.asarray(self._bridge_value(self.NU_CAUCHY, 16)).real)
        exact = float(epstein_zeta(self.NU_CAUCHY, A3, ZERO3, ZERO3).real)
        assert abs(v_hi - exact) < abs(v_low - exact), (
            f"bridge d=3: residual did not decrease as n grows "
            f"(8: {abs(v_low - exact):.3e}, 16: {abs(v_hi - exact):.3e})"
        )


class TestCycleAgainstZetaCircle:
    """4-cycle in d = 3 at uniform ν = 4 (σ = 1).  Compares the tensor
    evaluator against the closed-form `zeta_circle`, which uses a 3D
    Duffy-pyramid quadrature × Epstein zeta and is independent of the
    bucket-elimination code path.
    """

    NU = 4.0
    N_POINTS = 16

    def _cycle_edges(self) -> tuple[np.ndarray, np.ndarray]:
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int,
        )
        nu_vec = np.full(4, self.NU)
        return edges, nu_vec

    def test_cycle_matches_zeta_circle(self):
        edges, nu_vec = self._cycle_edges()
        tensor_val = float(np.asarray(
            graph_zeta_general_at_zero(edges, nu_vec, A3, self.N_POINTS)
        ).real)
        circle_val = float(zeta_circle([self.NU] * 4, A3))
        rel_err = abs(tensor_val - circle_val) / abs(circle_val)
        assert rel_err < 0.05, (
            f"cycle d=3 (ν={self.NU}, n={self.N_POINTS}): tensor "
            f"{tensor_val:.6e} vs zeta_circle {circle_val:.6e}, "
            f"rel_err = {rel_err:.3e}"
        )


class TestK4MinorAtD3:
    """K₄ graph at d = 3 — the smallest tw = 3 multigraph.  No
    closed-form reference exists; we Cauchy-check that the value at
    n = 8 agrees with the value at n = 6 to within the expected
    trapezoid-rule shift.  This pins the d = 3 tw > 2 code path so a
    regression in either `_eliminate_vertex` or the FFT-difference
    table reshape will surface.

    Kept tiny on purpose: K₄ at d = 3, n = 8 has a worst-bag tensor of
    size n^9 = 134M floats (1 GB) which is the upper edge of CI memory.
    n = 6 needs n^9 ≈ 10M ≈ 80 MB, comfortable.
    """

    NU = 5.0   # σ = 2 — well above the integrability threshold

    def test_n6_value_close_to_n8(self):
        # K_4: every vertex pair connected.
        edges = np.array(
            [[i, j] for i in range(4) for j in range(i + 1, 4)],
            dtype=int,
        )
        nu_vec = np.full(len(edges), self.NU)
        v6 = float(np.asarray(
            graph_zeta_general_at_zero(edges, nu_vec, A3, 6)
        ).real)
        v8 = float(np.asarray(
            graph_zeta_general_at_zero(edges, nu_vec, A3, 8)
        ).real)
        rel = abs(v8 - v6) / max(abs(v8), abs(v6))
        # σ=2 → residual scales as n^-(2-σ)? No: at σ ≥ 1.95 the
        # integrand is smooth, residual is exponentially small in n.
        # We just need v6 and v8 to agree to ~1e-3.
        assert rel < 5e-3, f"K4 d=3 Cauchy: n=6 {v6:.6e} vs n=8 {v8:.6e}, rel = {rel:.3e}"


class TestFrontendPassthroughD3:
    """The block-cut router (`evaluate_graph`) must thread d = 3 through
    to its block evaluators.  A 3-vertex linear graph (two bridges, no
    cycle) hits the closed-form Epstein zeta path twice and multiplies
    the two block values; we check the product matches a hand-computed
    expectation.
    """

    NU = 4.0

    def test_two_bridges_product(self):
        # vertex layout: 0 -- 1 -- 2 (path graph).
        edges = np.array([[0, 1], [1, 2]], dtype=int)
        val = evaluate_graph(edges, np.array([self.NU, self.NU]), A3)
        # Vacuum at k=0: factorises across the two bridge blocks, each
        # contributing epstein_zeta(ν, I_3, 0, 0).
        expected = float(epstein_zeta(self.NU, A3, ZERO3, ZERO3).real) ** 2
        rel = abs(val - expected) / abs(expected)
        assert rel < 1e-12, (
            f"path-graph d=3 router product: got {val:.6e}, "
            f"expected {expected:.6e}, rel = {rel:.3e}"
        )

    def test_bridge_plus_cycle(self):
        # vertex layout: bridge 0--1, cycle 1-2-3-1 (cut vertex at 1).
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 1]], dtype=int,
        )
        val = evaluate_graph(edges, np.array([self.NU] * 4), A3, n_points=16)
        expected_bridge = float(epstein_zeta(self.NU, A3, ZERO3, ZERO3).real)
        expected_cycle = float(zeta_circle([self.NU] * 3, A3))
        expected = expected_bridge * expected_cycle
        rel = abs(val - expected) / abs(expected)
        # Both block paths are analytic — no FFT residual.
        assert rel < 1e-10, (
            f"bridge+cycle d=3 router: got {val:.6e}, "
            f"expected {expected:.6e}, rel = {rel:.3e}"
        )
