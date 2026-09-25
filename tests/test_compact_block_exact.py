# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
r"""Purely compact blocks: the exact per-block torus and the sparse peel.

A purely compact bundle (the nearest-neighbour indicator ``nu = inf``, or
any :class:`Interaction` that is a table alone) makes a block's lattice
sum a finite sum, and the torus reproduces it exactly once no
configuration can wind.  Two things are pinned here:

* ``frontend._compact_exact_grid`` -- the smallest such torus, from the
  longest fundamental cycle of the block's best spanning tree rather
  than from its vertex count -- and that the router runs a purely
  compact block THERE at both arms, whatever the pass grid asks for;
* ``tensor_network._peel_sparse`` -- the exact roll-sum peel such a
  bundle takes in place of the dense contraction: the same products and
  sums, so an integer table stays bit-identical to the dense step.
"""
import numpy as np
import pytest

import gzl.frontend as fe
import gzl.tensor_network as tn
from gzl import Interaction, evaluate_graph
from gzl.tensor_network import (
    graph_zeta_general,
    graph_zeta_general_at_zero,
)
from tests._oracles import brute_force_zeta, table_kernel

A_CHAIN = np.array([[1.0]])
A_SQUARE = np.eye(2)
A_HEX = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])

BRIDGE = np.array([[0, 1]])
C4 = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
C5 = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]])
K4 = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
K5 = np.array([(i, j) for i in range(5) for j in range(i + 1, 5)])
THETA = np.array([[0, 2], [2, 1], [0, 3], [3, 1], [0, 4], [4, 1]])
PRISM = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3],
                  [0, 3], [1, 4], [2, 5]])
# the 3 x 3 grid graph: short fundamental cycles, a long corner-to-corner
# distance -- the one bound the window term dominates
GRID33 = np.array([[r * 3 + c, r * 3 + c + 1] for r in range(3) for c in range(2)]
                  + [[r * 3 + c, (r + 1) * 3 + c] for r in range(2) for c in range(3)])


def _lift(edges, R):
    """The shipped winding-safe grid n_v R + 2, the reference torus."""
    n_v = len({int(v) for e in np.asarray(edges).tolist() for v in e})
    return n_v * max(int(R), 1) + 2


def _dense_vacuum(edges, A, n, kernels=None):
    """The historical arithmetic: eager tables, no peel."""
    old = tn._USE_SPARSE_PEEL
    tn._USE_SPARSE_PEEL = False
    try:
        nu = np.full(len(edges), np.inf)
        return float(np.real(graph_zeta_general_at_zero(
            np.asarray(edges), nu, A, int(n), pinned_vertex=0,
            kernels=kernels)))
    finally:
        tn._USE_SPARSE_PEEL = old


class TestExactGrid:
    def test_cycle_bound_beats_the_vertex_count(self):
        g = fe._compact_exact_grid
        # every fundamental cycle of a breadth-first tree of K_n is a
        # triangle: n > 3 R closes the sum
        assert g(K5, 1) == 4 and _lift(K5, 1) == 7
        assert g(K4, 1) == 4 and _lift(K4, 1) == 6
        # a simple cycle has one fundamental cycle, itself
        assert g(C5, 1) == 6 and _lift(C5, 1) == 7
        assert g(C4, 1) == 5
        # three internally disjoint 2-paths: fundamental 4-cycles
        assert g(THETA, 1) == 5 and _lift(THETA, 1) == 7
        # a bridge: no cycle, the table window 2 R + 1 alone
        assert g(BRIDGE, 1) == 3 and g(BRIDGE, 2) == 5
        # the radius scales the cycle term
        assert g(K5, 2) == 7 and g(C4, 2) == 9
        # R = 0 (a table on the origin alone) is treated as 1
        assert g(K4, 0) == 4

    def test_the_terminal_distance_widens_the_window(self):
        g = fe._compact_exact_grid
        # K5: the terminal is adjacent to the source, 2 ell R = 2 < 3
        assert g(K5, 1, 0, 1) == 4
        # the 3 x 3 grid: L = 4 or 5, but corner to corner is 4 apart
        assert g(GRID33, 1) <= 6
        assert g(GRID33, 1, 0, 8) == 9
        assert g(GRID33, 1, 0, 1) == g(GRID33, 1)
        # C4 with the terminal antipodal: 2 * 2 * R + 1 = L R + 1
        assert g(C4, 1, 0, 2) == 5 and g(C4, 2, 0, 2) == 9

    @pytest.mark.parametrize("edges", [K4, K5, THETA, PRISM, C5, GRID33])
    @pytest.mark.parametrize("A", [A_SQUARE, A_HEX])
    def test_the_exact_grid_reproduces_the_lift_bit_for_bit(self, edges, A):
        # the nearest-neighbour count is the same finite sum on every
        # admissible torus, and integer arithmetic makes that bitwise
        n_exact = fe._compact_exact_grid(edges, 1)
        want = _dense_vacuum(edges, A, _lift(edges, 1))
        got = _dense_vacuum(edges, A, n_exact)
        assert got == want and got == np.round(got)
        # ...and the shipped route lands on the same integer
        assert evaluate_graph(edges, np.inf, A, n_points=0) == want

    def test_below_the_exact_grid_the_torus_winds(self):
        # C4 on the square lattice: 36 homomorphisms; at n = 4 the
        # all-edges-forward configuration closes around the torus
        assert _dense_vacuum(C4, A_SQUARE, 5) == 36.0
        assert _dense_vacuum(C4, A_SQUARE, 4) != 36.0
        assert fe._compact_exact_grid(C4, 1) == 5

    def test_radius_two_table_against_brute_force_d1(self):
        # K5 at d = 1 with a radius-2 table: (2L+1)^4 configurations
        T = {(1,): 0.7, (-1,): 0.7, (2,): -0.3, (-2,): -0.3}
        V = Interaction.from_table(T)
        assert fe._compact_exact_grid(K5, 2) == 7
        ref = brute_force_zeta(K5, [table_kernel(T)] * 10, A_CHAIN, 8)
        got, info = evaluate_graph(K5, V, A_CHAIN, n_points=32,
                                   return_diagnostics=True)
        assert got == pytest.approx(ref, rel=1e-13)
        assert info["n_block_nn"] == 1 and info["n_block_compact_lift"] == 0


class TestSparsePeel:
    @pytest.mark.parametrize("edges", [K4, K5, THETA, PRISM])
    def test_integer_tables_are_bit_identical_to_the_dense_step(self, edges):
        n = _lift(edges, 1)
        nu = np.full(len(edges), np.inf)
        got = float(np.real(graph_zeta_general_at_zero(
            edges, nu, A_HEX, n, pinned_vertex=0)))
        assert got == _dense_vacuum(edges, A_HEX, n)
        assert got == np.round(got)

    def test_the_sparse_peel_fires_and_the_fft_peel_does_not(self, monkeypatch):
        sparse, conv = [], []
        orig_s, orig_c = tn._peel_sparse, tn._peel_convolution
        monkeypatch.setattr(tn, "_peel_sparse",
                            lambda *a, **k: (sparse.append(1), orig_s(*a, **k))[1])
        monkeypatch.setattr(tn, "_peel_convolution",
                            lambda *a, **k: (conv.append(1), orig_c(*a, **k))[1])
        graph_zeta_general_at_zero(K4, np.full(6, np.inf), A_SQUARE, 6,
                                   pinned_vertex=0)
        assert sparse and not conv
        sparse.clear()
        graph_zeta_general_at_zero(K4, np.full(6, 2.5), A_SQUARE, 6,
                                   pinned_vertex=0)
        assert conv and not sparse

    def test_the_kill_switch_restores_the_dense_representation(self, monkeypatch):
        monkeypatch.setattr(tn, "_USE_SPARSE_PEEL", False)
        peels = []
        orig = tn.TorusTruncation.peel_step
        monkeypatch.setattr(tn.TorusTruncation, "peel_step",
                            lambda self, *a: (peels.append(1), orig(self, *a))[1])
        v = graph_zeta_general_at_zero(K5, np.full(10, np.inf), A_HEX, 5,
                                       pinned_vertex=0)
        assert not peels and float(np.real(v)) == np.round(float(np.real(v)))

    def test_a_real_table_agrees_with_the_dense_step_to_round_off(self):
        V = Interaction.from_shells(A_HEX, {1.0: 0.5, np.sqrt(3): 0.2})
        nu = np.full(len(PRISM), np.inf)
        n = fe._compact_exact_grid(PRISM, 2)
        got = float(np.real(graph_zeta_general_at_zero(
            PRISM, nu, A_HEX, n, pinned_vertex=0, kernels=[V] * 9)))
        want = _dense_vacuum(PRISM, A_HEX, n, kernels=[V] * 9)
        assert got == pytest.approx(want, rel=1e-13)
        # a kept terminal: the whole form factor, both arithmetics
        M = np.real(graph_zeta_general(PRISM, nu, A_HEX, n, source=0,
                                       terminals=(3,), space="z",
                                       kernels=[V] * 9))
        old = tn._USE_SPARSE_PEEL
        tn._USE_SPARSE_PEEL = False
        try:
            M_dense = np.real(graph_zeta_general(
                PRISM, nu, A_HEX, n, source=0, terminals=(3,), space="z",
                kernels=[V] * 9))
        finally:
            tn._USE_SPARSE_PEEL = old
        assert np.max(np.abs(M - M_dense)) <= 1e-13 * np.max(np.abs(M_dense))

    def test_nearest_neighbour_table_is_bit_identical_to_nu_inf(self):
        NN = Interaction.nearest_neighbour(A_HEX)
        nu = np.full(len(K5), np.inf)
        a = graph_zeta_general_at_zero(K5, nu, A_HEX, 4, pinned_vertex=0)
        b = graph_zeta_general_at_zero(K5, nu, A_HEX, 4, pinned_vertex=0,
                                       kernels=[NN] * 10)
        assert float(np.real(a)) == float(np.real(b))


class TestRouting:
    def _record(self, monkeypatch):
        seen = []
        orig = fe.graph_zeta_general
        orig0 = fe.graph_zeta_general_at_zero

        def rec(edges, nu, A, n, **kw):
            seen.append(("k", int(n)))
            return orig(edges, nu, A, n, **kw)

        def rec0(edges, nu, A, n, **kw):
            seen.append(("0", int(n)))
            return orig0(edges, nu, A, n, **kw)
        monkeypatch.setattr(fe, "graph_zeta_general", rec)
        monkeypatch.setattr(fe, "graph_zeta_general_at_zero", rec0)
        return seen

    def test_a_purely_compact_block_runs_at_its_exact_grid_at_both_arms(
            self, monkeypatch):
        seen = self._record(monkeypatch)
        # k = 0: the pass grid 16 is neither honoured nor lifted to
        # n_v R + 2 = 7 -- the block runs at 4
        val = evaluate_graph(K5, np.inf, A_HEX, n_points=16)
        assert seen == [("0", 4)]
        seen.clear()
        # the BZ grid: the form factor at the exact grid, delivered on 16
        grid = evaluate_graph(K5, np.inf, A_HEX, terminal=1, n_points=16)
        assert seen == [("k", 4)] and grid.shape == (16, 16)
        assert grid[0, 0] == pytest.approx(val, abs=1e-9)
        # the same integer at every pass grid, including none
        for n in (0, 4, 8, 32):
            assert evaluate_graph(K5, np.inf, A_HEX, n_points=n) == val

    def test_a_table_block_takes_the_same_route(self, monkeypatch):
        seen = self._record(monkeypatch)
        V = Interaction.from_shells(A_HEX, {1.0: 0.5, np.sqrt(3): 0.2})
        _, info = evaluate_graph(PRISM, V, A_HEX, n_points=16,
                                 return_diagnostics=True)
        assert seen == [("0", fe._compact_exact_grid(PRISM, 2))]
        assert info["n_block_nn"] == 1 and info["n_block_tensor"] == 1
        assert info["n_block_compact_lift"] == 0
        seen.clear()
        # a pass grid below the exact grid counts as a lift
        _, info = evaluate_graph(PRISM, V, A_HEX, n_points=4,
                                 return_diagnostics=True)
        assert seen == [("0", fe._compact_exact_grid(PRISM, 2))]
        assert info["n_block_compact_lift"] == 1

    def test_finite_k_form_factor_against_brute_force(self):
        T = {(1, 0): 0.4, (-1, 0): 0.4, (0, 1): 0.6, (0, -1): 0.6,
             (1, 1): -0.2, (-1, -1): -0.2}
        V = Interaction.from_table(T)
        k = np.array([0.15, 0.4])
        import networkx as nx
        for edges, t in ((K4, 3), (THETA, 1)):
            got = evaluate_graph(edges, V, A_HEX, terminal=t, momentum=k,
                                 n_points=0)
            # the box must hold every configuration: the pin's
            # eccentricity times the radius
            G = nx.Graph(edges.tolist())
            ref = brute_force_zeta(edges, [table_kernel(T)] * len(edges),
                                   A_HEX, nx.eccentricity(G, 0),
                                   terminal=t, k_frac=k)
            assert got == pytest.approx(ref, rel=1e-12)
        # the on-grid samples of the exact trigonometric polynomial
        g8 = evaluate_graph(K4, V, A_HEX, terminal=3, n_points=8)
        g16 = evaluate_graph(K4, V, A_HEX, terminal=3, n_points=16)
        assert g8[2, 2] == pytest.approx(g16[4, 4], rel=1e-13)
