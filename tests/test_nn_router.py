# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""Nearest-neighbour (NN) kernel router: ``evaluate_graph(edges, np.inf, A)``.

ν = ∞ requests the exact NN-indicator kernel (1 on the minimal nonzero lattice
shell, 0 elsewhere).  ζ_G then counts NN graph *homomorphisms* into the lattice.
These are exact integers, independent of ``n_points`` (above the longest-cycle
threshold), and equal the ν → ∞ limit of the power-law kernel.

A bridge returns the lattice **coordination number**; the NN kernel keys off the
minimal nonzero distance, so it is scale-invariant (works for sub-unit-norm
lattices) and reproduces the non-trivial coordinations of triangular / BCC / FCC
lattices.
"""
import numpy as np
import pytest

from gzl import evaluate_graph

# Lattices (columns of A are primitive vectors).
A_CHAIN = np.array([[1.0]])                                   # d=1, coord 2
A_SQUARE = np.eye(2)                                          # d=2, coord 4
A_SQUARE_HALF = 0.5 * np.eye(2)                               # sub-unit norm, coord 4
A_TRIANGULAR = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])  # d=2, coord 6
A_CUBIC = np.eye(3)                                           # d=3, coord 6
A_BCC = 0.5 * np.array([[-1.0, 1.0, 1.0],
                        [1.0, -1.0, 1.0],
                        [1.0, 1.0, -1.0]])                    # d=3, coord 8
A_FCC = 0.5 * np.array([[0.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0]])                     # d=3, coord 12

BRIDGE = np.array([[0, 1]])
P3 = np.array([[0, 1], [1, 2]])
C4 = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
TRIANGLE = np.array([[0, 1], [1, 2], [0, 2]])


class TestCoordinationNumber:
    """A NN bridge returns the lattice coordination number."""

    @pytest.mark.parametrize("A, coord", [
        (A_CHAIN, 2),
        (A_SQUARE, 4),
        (A_SQUARE_HALF, 4),      # scale-invariance: NN distance 0.5 < 1
        (A_TRIANGULAR, 6),       # non-trivial coordination
        (A_CUBIC, 6),
        (A_BCC, 8),              # non-trivial coordination (NN at √3/2)
        (A_FCC, 12),             # non-trivial coordination (NN at √2/2)
    ])
    def test_bridge_is_coordination(self, A, coord):
        v, info = evaluate_graph(BRIDGE, np.inf, A, n_points=0,
                                 return_diagnostics=True)
        assert v == pytest.approx(float(coord), abs=1e-9)
        assert info["nn_mode"] is True
        assert info["n_block_nn"] == 1
        assert info["n_bridges"] == 0          # routed to tensor, not epstein


class TestHomomorphismCounts:
    """Small-graph NN homomorphism counts (hand-verifiable)."""

    def test_square_path_p3(self):
        # v1: 4 NN; v2: 4 NN of v1 (may coincide with v0) -> 16.
        assert evaluate_graph(P3, np.inf, A_SQUARE) == pytest.approx(16.0, abs=1e-9)

    def test_square_4cycle(self):
        assert evaluate_graph(C4, np.inf, A_SQUARE) == pytest.approx(36.0, abs=1e-9)

    def test_square_triangle_is_zero(self):
        # The square lattice is bipartite: no NN 3-cycle.
        assert evaluate_graph(TRIANGLE, np.inf, A_SQUARE) == pytest.approx(0.0, abs=1e-9)

    def test_triangular_triangle_is_twelve(self):
        # Triangular lattice has NN triangles: 6 NN x 2 shared neighbours = 12.
        assert evaluate_graph(TRIANGLE, np.inf, A_TRIANGULAR) == pytest.approx(12.0, abs=1e-9)


class TestCrossChecksAndStability:
    """NN kernel = ν→∞ limit; integer; n_points-independent."""

    # Only lattices with NN distance == 1: there the power-law kernel
    # |x|^{-ν} → the NN indicator as ν → ∞, so a large finite ν cross-checks
    # the inf result.  (For NN distance < 1 — e.g. FCC 0.707, BCC 0.866,
    # 0.5·I — the power law DIVERGES, which is exactly why the dedicated
    # scale-invariant indicator is required; those are covered by the
    # coordination-number tests, not here.)
    @pytest.mark.parametrize("edges, A, n", [
        (P3, A_SQUARE, 12),
        (C4, A_SQUARE, 12),
        (TRIANGLE, A_TRIANGULAR, 12),
        (C4, A_CUBIC, 8),          # 4-cycle on simple cubic (count not hand-obvious)
    ])
    def test_matches_large_finite_nu(self, edges, A, n):
        v_inf = evaluate_graph(edges, np.inf, A, n_points=0)
        v_100 = evaluate_graph(edges, 100.0, A, n_points=n)
        assert v_inf == pytest.approx(v_100, abs=1e-6)
        assert v_inf == pytest.approx(round(v_inf), abs=1e-6)   # integer

    def test_subunit_norm_indicator_is_finite_and_integer(self):
        # NN distance 0.5 < 1: the indicator stays finite/exact (here the
        # FCC triangle count) where the power-law limit would diverge.
        v = evaluate_graph(TRIANGLE, np.inf, A_FCC, n_points=0)
        assert np.isfinite(v)
        assert v == pytest.approx(round(v), abs=1e-6) and v > 0

    @pytest.mark.parametrize("n_points", [0, 8, 16])
    def test_n_points_independent(self, n_points):
        # Above the longest-cycle threshold the value is exact and fixed.
        v = evaluate_graph(C4, np.inf, A_SQUARE, n_points=n_points)
        assert v == pytest.approx(36.0, abs=1e-9)


class TestRoutingAndDiagnostics:
    """In nn_mode every block is tensor-routed; closed forms are bypassed."""

    def test_dense_block_skips_direct_sum(self):
        # K5 (tw=4) must NOT hit direct_sum (undefined at ν=inf).
        K5 = np.array([(i, j) for i in range(5) for j in range(i + 1, 5)])
        v, info = evaluate_graph(K5, np.inf, A_SQUARE, n_points=0,
                                 return_diagnostics=True)
        assert np.isfinite(v)
        assert info["n_block_direct_sum"] == 0
        assert info["n_block_tensor"] == 1
        assert info["max_tensor_block_tw"] == 4

    def test_no_closed_form_blocks(self):
        # Bridge + cycle + dense, all NN: nothing counted as bridge/cycle/dsum.
        # A triangle (cycle block) with a pendant bridge.
        edges = np.array([[0, 1], [1, 2], [0, 2], [0, 3]])
        v, info = evaluate_graph(edges, np.inf, A_TRIANGULAR, n_points=0,
                                 return_diagnostics=True)
        assert info["nn_mode"] is True
        assert info["n_bridges"] == 0
        assert info["n_simple_cycles"] == 0
        assert info["n_block_direct_sum"] == 0
        assert info["n_block_nn"] == info["n_block_tensor"] >= 2


class TestFiniteMomentum:
    """NN dispersion ζ_G(k): an exact trig polynomial; n_points sets only the
    output BZ-grid resolution, the internal form factor is computed exactly."""

    def test_chain_bridge_dispersion(self):
        # 1D NN bridge: ζ(k) = 2 cos 2πk.
        for n in (4, 8, 16):
            g = evaluate_graph(BRIDGE, np.inf, A_CHAIN, terminal=1, n_points=n)
            k = np.arange(n) / n
            assert g == pytest.approx(2.0 * np.cos(2 * np.pi * k), abs=1e-12)

    def test_square_bridge_dispersion_exact(self):
        # 2D NN bridge: ζ(k) = 2(cos 2πk_x + cos 2πk_y), exact at every n_points.
        for n in (4, 8, 16, 32):
            g = evaluate_graph(BRIDGE, np.inf, A_SQUARE, terminal=1, n_points=n)
            k = np.arange(n) / n
            ana = 2.0 * (np.cos(2 * np.pi * k)[:, None]
                         + np.cos(2 * np.pi * k)[None, :])
            assert g.shape == (n, n)
            assert g == pytest.approx(ana, abs=1e-12)

    def test_single_k(self):
        # Single momentum returns a scalar matching the analytic value.
        v = evaluate_graph(BRIDGE, np.inf, A_SQUARE, terminal=1,
                           momentum=np.array([0.25, 0.0]))
        assert v == pytest.approx(2.0, abs=1e-12)   # 2(cos π/2 + cos 0)

    def test_k0_equals_vacuum(self):
        # The k=0 cell of the dispersion equals the vacuum (homomorphism) value.
        g = evaluate_graph(C4, np.inf, A_SQUARE, terminal=2, n_points=8)
        assert g[0, 0] == pytest.approx(36.0, abs=1e-9)

    def test_output_resolution_independent(self):
        # A shared k-point (here k=(1/4,1/4)) gives the SAME value whether
        # sampled on an 8- or 16-grid — n_points is pure output resolution.
        g8 = evaluate_graph(C4, np.inf, A_SQUARE, terminal=2, n_points=8)
        g16 = evaluate_graph(C4, np.inf, A_SQUARE, terminal=2, n_points=16)
        assert g8[2, 2] == pytest.approx(g16[4, 4], abs=1e-10)

    @pytest.mark.parametrize("edges, A, t", [
        (np.array([[0, 1], [1, 2]]), A_SQUARE, 2),               # path
        (np.array([[0, 1], [1, 2], [1, 3]]), A_SQUARE, 2),       # path + pendant
        (C4, A_SQUARE, 2),                                       # cycle spine
        (TRIANGLE, A_SQUARE, 1),                                 # odd cycle -> 0
        (np.array([[0, 1], [1, 2], [2, 3], [3, 0], [1, 4]]), A_SQUARE, 3),
        (np.array([[0, 1], [1, 2]]), A_TRIANGULAR, 2),           # triangular path
        (C4, A_TRIANGULAR, 2),                                   # triangular cycle
    ])
    def test_matches_large_finite_nu_dispersion(self, edges, A, t):
        # On d_min=1 lattices the ν→∞ dispersion equals the ν=100 one.
        g_inf = evaluate_graph(edges, np.inf, A, terminal=t, n_points=8)
        g_100 = evaluate_graph(edges, 100.0, A, terminal=t, n_points=8)
        assert g_inf == pytest.approx(g_100, abs=1e-6)

    def test_diagnostics_count_spine_blocks(self):
        # Path 0-1-2 (terminal 2): two on-spine NN bridges -> n_block_nn == 2.
        edges = np.array([[0, 1], [1, 2]])
        _, info = evaluate_graph(edges, np.inf, A_SQUARE, terminal=2,
                                 n_points=8, return_diagnostics=True)
        assert info["nn_mode"] is True
        assert info["n_bridges"] == 0
        assert info["n_block_nn"] == 2
