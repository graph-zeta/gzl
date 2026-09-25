# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""The router with general interactions: the demoted power law is
bit-identical to the float call, the nearest-neighbour table is
bit-identical to nu = inf, closed forms and engines agree with exact
oracles, the routing counters say which arm ran, and the block cache
never conflates two kernels that share a tail exponent."""
from __future__ import annotations

import math

import numpy as np
import pytest

from gzl import (
    Interaction,
    InteractionSupportError,
    evaluate_graph,
    zeta_circle,
)
from gzl import frontend
from gzl.frontend import (
    GraphZetaError,
    TopologyEvaluatorUnavailableError,
    UnsupportedLatticeSumError,
    UnsupportedRequestError,
)
from gzl.interaction import _KernelProduct
from gzl.hybrid import hybrid_zeta
from gzl.slab import slab_zeta
from gzl.direct_sum import direct_sum_extrapolated
from tests._oracles import brute_force_zeta, mixed_kernel, power_law_kernel, table_kernel

A_CHAIN = np.array([[1.0]])
A_SQUARE = np.eye(2)
A_SQUARE_HALF = 0.5 * np.eye(2)
A_HEX = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
A_CUBIC = np.eye(3)
A_BCC = 0.5 * np.array([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]])
A_FCC = 0.5 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
A_SHEARED_SHORT = np.array([[1.0, 0.9], [0.0, 0.436]])   # shortest vector (-0.1, 0.436): not a column
NN_LATTICES = [A_CHAIN, A_SQUARE, A_SQUARE_HALF, A_HEX, A_CUBIC, A_BCC, A_FCC, A_SHEARED_SHORT]

BRIDGE = np.array([[0, 1]])
P3 = np.array([[0, 1], [1, 2]])
TRIANGLE = np.array([[0, 1], [1, 2], [0, 2]])
C4 = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
THETA = np.array([[0, 1], [0, 1], [0, 2], [2, 1]])        # parallel pair + a 2-path
DIAMOND = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
K4 = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
K4S = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 4], [4, 3]])   # K4, edge 2-3 subdivided
K5 = np.array([[i, j] for i in range(5) for j in range(i + 1, 5)])
BRIDGE_TRI = np.array([[0, 1], [1, 2], [2, 3], [1, 3]])   # bridge + triangle


def cross(d, val, origin=0.0):
    tab = {}
    for i in range(d):
        for s in (-1, 1):
            m = [0] * d
            m[i] = s
            tab[tuple(m)] = val
    if origin:
        tab[(0,) * d] = origin
    return tab


def kernel_fn(V, A):
    """A tests._oracles kernel function for an Interaction-like."""
    if isinstance(V, _KernelProduct):
        fns = [kernel_fn(f, A) for f in V.factors]
        return lambda dz: np.prod([f(dz) for f in fns], axis=0)
    return mixed_kernel(dict(V.compact), list(V.b), list(V.nu), A)


class TestLegacyIdentity:
    @pytest.mark.parametrize("edges", [BRIDGE, TRIANGLE, C4, DIAMOND, K4, BRIDGE_TRI])
    @pytest.mark.parametrize("A", [A_CHAIN, A_SQUARE])
    def test_plain_power_law_is_bit_identical(self, edges, A):
        d = A.shape[0]
        nu = d + 0.5
        want = evaluate_graph(edges, nu, A, n_points=8)
        assert evaluate_graph(edges, Interaction.power_law(nu), A, n_points=8) == want
        per_edge = [Interaction.power_law(nu) if i % 2 else nu for i in range(len(edges))]
        assert evaluate_graph(edges, per_edge, A, n_points=8) == want

    def test_plain_power_law_finite_k_is_bit_identical(self):
        nu = 2.5
        for edges in (BRIDGE, TRIANGLE, DIAMOND):
            want = evaluate_graph(edges, nu, A_SQUARE, terminal=1, n_points=8)
            got = evaluate_graph(edges, Interaction.power_law(nu), A_SQUARE, terminal=1, n_points=8)
            assert np.array_equal(want, got)
            k = np.array([0.1, 0.3])
            assert (evaluate_graph(edges, nu, A_SQUARE, terminal=1, momentum=k, n_points=8)
                    == evaluate_graph(edges, Interaction.power_law(nu), A_SQUARE, terminal=1,
                                      momentum=k, n_points=8))

    def test_multigraph_attribute_may_be_an_interaction(self):
        nx = pytest.importorskip("networkx")
        V = Interaction.from_table(cross(2, 0.3), b=[1.0], nu=[3.0])
        mg = nx.MultiGraph()
        mg.add_edge(0, 1, nu=V)
        mg.add_edge(1, 2, nu=V)
        mg.add_edge(0, 2, nu=3.0)
        want = evaluate_graph(TRIANGLE, [V, V, 3.0], A_SQUARE, n_points=8)
        got = evaluate_graph(mg, A_SQUARE, n_points=8)
        assert got == want
        mg2 = nx.MultiGraph()
        mg2.add_edge(0, 1, nu=V)
        mg2.add_edge(0, 1, nu=V)
        # the doubled edge is the pointwise square of the kernel: an
        # independent real-space sum of V(x)^2 (steep enough at nu = 3
        # squared for the box to settle at L = 60 to 1e-6)
        got = evaluate_graph(mg2, A_SQUARE)
        ref = brute_force_zeta(BRIDGE, [kernel_fn(V ** 2, A_SQUARE)], A_SQUARE, 60)
        assert got == pytest.approx(ref, rel=1e-6)
        assert got == (V ** 2).lattice_sum(A_SQUARE)

    @pytest.mark.parametrize("A", NN_LATTICES)
    def test_nearest_neighbour_table_is_bit_identical_to_nu_inf(self, A):
        NN = Interaction.nearest_neighbour(A)
        for edges in (BRIDGE, P3, TRIANGLE, C4):
            want, info_w = evaluate_graph(edges, np.inf, A, n_points=0, return_diagnostics=True)
            got, info_g = evaluate_graph(edges, NN, A, n_points=0, return_diagnostics=True)
            assert got == want
            assert info_g["nn_mode"] and info_g["n_block_nn"] == info_w["n_block_nn"]
        if A.shape[0] <= 2:
            g_w = evaluate_graph(BRIDGE, np.inf, A, terminal=1, n_points=8)
            g_g = evaluate_graph(BRIDGE, NN, A, terminal=1, n_points=8)
            assert np.array_equal(g_w, g_g)


class TestClosedForms:
    def test_mixed_bridge_is_the_lattice_sum(self):
        V = Interaction.from_shells(A_HEX, {1.0: 0.5, np.sqrt(3): 0.2}, b=[0.3], nu=[6.0])
        val, info = evaluate_graph(BRIDGE, V, A_HEX, return_diagnostics=True)
        assert val == V.lattice_sum(A_HEX) and info["n_bridges"] == 1
        assert info["interaction_mode"] and info["n_block_mixed_kernel"] == 1
        ref = brute_force_zeta(BRIDGE, [kernel_fn(V, A_HEX)], A_HEX, 60)
        assert val == pytest.approx(ref, rel=1e-6)          # window tail ~ 2 pi / (4 * 60^4)
        k = np.array([0.2, 0.35])
        assert evaluate_graph(BRIDGE, V, A_HEX, terminal=1, momentum=k) == V.lattice_sum(A_HEX, k)
        grid = evaluate_graph(BRIDGE, V, A_HEX, terminal=1, n_points=6)
        axes = np.arange(6) / 6
        kk = np.stack(np.meshgrid(axes, axes, indexing="ij"), -1).reshape(-1, 2)
        assert np.array_equal(grid.ravel(), V.lattice_sum(A_HEX, kk))

    def test_mixed_cycle_takes_the_generalised_closed_form(self):
        V = Interaction.from_shells(A_HEX, {1.0: 0.5}, b=[0.3], nu=[4.5])
        val, info = evaluate_graph(TRIANGLE, V, A_HEX, return_diagnostics=True)
        assert info["n_simple_cycles"] == 1 and info["n_block_cycle_fallback"] == 0
        assert val == zeta_circle([V] * 3, A_HEX)

    def test_cycle_fallback_takes_the_sigma_router_and_is_not_cached(self, monkeypatch):
        from gzl import circle as C
        V = Interaction.from_table(cross(2, 0.4), b=[1.0], nu=[4.5])
        # negative: no two rungs can agree, whatever their rounding
        monkeypatch.setattr(C, "_CYCLE_SELF_BAND", -1.0)
        cache = {}
        val, info = evaluate_graph(TRIANGLE, V, A_SQUARE, n_points=16,
                                   return_diagnostics=True, block_cache=cache)
        assert info["n_block_cycle_fallback"] == 1 and info["n_simple_cycles"] == 0
        assert info["n_block_tensor"] + info["n_block_algebra"] == 1
        assert not any(k[0] == "cycle" for k in cache)
        # ...and it is the same number the sigma-router gives with fast_cycles
        assert val == evaluate_graph(TRIANGLE, V, A_SQUARE, n_points=16, fast_cycles=True)

    def test_pure_power_law_with_weight_cycle(self):
        V = Interaction.power_law(2.5, b=2.0)
        val, info = evaluate_graph(TRIANGLE, V, A_SQUARE, return_diagnostics=True)
        assert info["n_simple_cycles"] == 1
        assert val == pytest.approx(8.0 * zeta_circle([2.5] * 3, A_SQUARE), rel=1e-14)


class TestCompactMode:
    def test_purely_compact_graph_inherits_nn_routing(self):
        T = Interaction.from_table(cross(2, 1.0, origin=1.0))
        for edges, reach in ((TRIANGLE, 1), (C4, 1), (K4, 1)):
            val, info = evaluate_graph(edges, T, A_SQUARE, n_points=0, return_diagnostics=True)
            assert info["nn_mode"] and info["n_bridges"] == 0 and info["n_simple_cycles"] == 0
            assert info["n_block_nn"] == 1
            ref = brute_force_zeta(edges, [kernel_fn(T, A_SQUARE)] * len(edges), A_SQUARE, 3)
            assert val == ref                                   # integer-exact
        R2 = {m: 1.0 for m in [(1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)]}
        T2 = Interaction.from_table(R2)
        val = evaluate_graph(C4, T2, A_SQUARE, n_points=0)
        ref = brute_force_zeta(C4, [kernel_fn(T2, A_SQUARE)] * 4, A_SQUARE, 4)
        assert val == ref

    def test_a_purely_compact_block_runs_at_its_exact_grid_at_k0(self, monkeypatch):
        # no tail, nothing to truncate: the torus is exact at every
        # n >= the block's smallest exact grid (``_compact_exact_grid``,
        # here 3R + 1 for K4), so the vacuum arm runs the block THERE
        # rather than at the larger of the lift and the pass grid
        # (measured: a K4 of a J1 + J2 sweep at n_points = 48 took 21 s
        # at 48 against 0.03 s at its lift of 10).
        grids = []
        orig = frontend.graph_zeta_general_at_zero

        def recording(edges, nu, A, n, *a, **k):
            grids.append(int(n))
            return orig(edges, nu, A, n, *a, **k)
        monkeypatch.setattr(frontend, "graph_zeta_general_at_zero", recording)
        for R in (1, 2):
            table = {(i, j): 0.3 for i in range(-R, R + 1) for j in range(-R, R + 1)
                     if (i, j) != (0, 0) and max(abs(i), abs(j)) == R}
            V = Interaction.from_table(table)
            assert V.support_radius == R and not np.isfinite(V.tail_exponent)
            exact = frontend._compact_exact_grid(K4, R)
            assert exact == 3 * R + 1 < 4 * R + 2
            grids.clear()
            big, info = evaluate_graph(K4, V, A_SQUARE, n_points=32, return_diagnostics=True)
            assert info["n_block_nn"] == 1 and grids == [exact]
            small = evaluate_graph(K4, V, A_SQUARE, n_points=4 * R + 2)
            assert big == small
            ref = brute_force_zeta(K4, [kernel_fn(V, A_SQUARE)] * 6, A_SQUARE, 2 * R + 1)
            assert big == pytest.approx(ref, rel=1e-13)

    def test_compact_finite_k_form_factor(self):
        T = Interaction.from_table(cross(2, 0.5, origin=0.25))
        k = np.array([0.15, 0.4])
        val = evaluate_graph(P3, T, A_SQUARE, terminal=2, momentum=k, n_points=0)
        ref = brute_force_zeta(P3, [kernel_fn(T, A_SQUARE)] * 2, A_SQUARE, 3, terminal=2, k_frac=k)
        assert val == pytest.approx(ref, rel=1e-14)
        grid8 = evaluate_graph(P3, T, A_SQUARE, terminal=2, n_points=8)
        grid16 = evaluate_graph(P3, T, A_SQUARE, terminal=2, n_points=16)
        assert grid8[2, 2] == pytest.approx(grid16[4, 4], rel=1e-14)

    def test_compact_part_keeps_the_split_below_the_core_floor(self):
        # the split is value-correct with tables (hybrid samples the table
        # on the fine SP grid) and is resolved as for a float; at n = 8
        # the grading rule never fires (``_CORE_N_FLOOR``), for a table
        # as for a float; the ladder's rungs [8, 6, 4] sit below the
        # winding-safe grid n_v R + 2 = 6, so it is skipped as a window
        # refusal, not fitted
        V = Interaction.from_table(cross(2, 0.3), b=[1.0], nu=[2.5])
        val, info = evaluate_graph(K4, V, A_SQUARE, n_points=8, return_diagnostics=True)
        assert info["sp_n_points"] == frontend._SPLIT_SP_N_POINTS[2]
        assert info["n_block_split"] == 1 and info["n_block_core_graded"] == 0
        assert info["n_block_richardson_theory"] == 0
        assert info["n_block_richardson_skipped_window"] == 1
        want = hybrid_zeta(K4, np.full(6, 2.5), A_SQUARE, 8,
                           sp_n_points=info["sp_n_points"], kernels=[V] * 6)
        assert val == pytest.approx(float(np.real(want)), rel=1e-12)

    def test_a_pure_block_does_not_depend_on_a_table_elsewhere_in_the_graph(self):
        # the split used to be switched off for the WHOLE call as soon as
        # any kernel carried a table (measured 12x worse on a K4 beside a
        # pendant bridge that carried the only table): the dense block's
        # value is now what the all-float call gives, bit for bit
        G = np.vstack([K4S, [[3, 5]]])
        vf, inf_f = evaluate_graph(G, 2.5, A_CHAIN, n_points=16, return_diagnostics=True)
        Vb = Interaction.from_table({(1,): 0.3, (-1,): 0.3}, b=[1.0], nu=[2.5])
        vm, inf_m = evaluate_graph(G, [2.5] * 7 + [Vb], A_CHAIN, n_points=16,
                                   return_diagnostics=True)
        bf = evaluate_graph(BRIDGE, 2.5, A_CHAIN)
        bm = evaluate_graph(BRIDGE, Vb, A_CHAIN)
        assert inf_f["sp_n_points"] == inf_m["sp_n_points"] == frontend._SPLIT_SP_N_POINTS[1]
        assert vf / bf == vm / bm
        # and the pure block SHARES its cache entry between the two calls
        # (a plain power law in the mixed list keys as its float)
        cache = {}
        evaluate_graph(G, 2.5, A_CHAIN, n_points=16, block_cache=cache)
        n_keys = len(cache)
        _, info = evaluate_graph(G, [2.5] * 7 + [Vb], A_CHAIN, n_points=16, block_cache=cache,
                                 return_diagnostics=True)
        assert info["n_block_cache_hits"] >= 1 and len(cache) == n_keys + 1

    def test_ladder_rungs_below_the_winding_safe_grid_are_not_fitted(self):
        # K4 with a subdivided edge, R = 1: lift 5 * 1 + 2 = 7; at n = 8
        # the rungs [8, 6, 4] would fit the all-table winding term of the
        # 4-cycles (measured: the accepted ladder was 6x worse than the
        # raw grid) -- a window refusal now; at n = 12 the rungs
        # [12, 10, 8] all clear the lift and the ladder is attempted
        V = Interaction.from_shells(A_HEX, {1.0: 0.5}, b=[0.1], nu=[3.0], total=True)
        ref = float(np.real(direct_sum_extrapolated(K4S, np.full(7, 3.0), A_HEX, (4, 5, 6, 7),
                                                    kernels=[V] * 7)))
        val, info = evaluate_graph(K4S, V, A_HEX, n_points=8, accuracy="strict",
                                   return_diagnostics=True)
        assert info["n_block_richardson_skipped_window"] == 1
        assert info["n_block_richardson_theory"] == 0
        assert val == pytest.approx(ref, rel=5e-5)          # measured 1.5e-05 (split on)
        _, info12 = evaluate_graph(K4S, V, A_HEX, n_points=12, accuracy="strict",
                                   return_diagnostics=True)
        assert info12["n_block_richardson_skipped_window"] == 0

    def test_a_purely_compact_bundle_beside_finite_tails_keeps_the_power_law_routes(self):
        # the call-wide nearest-neighbour flag used to send this block to
        # the raw pass-grid algebra (1.5e-2 off at n = 8); the block mixes
        # one purely compact bundle (NN times a mixed kernel: tail inf) with
        # finite tails, so it takes the compact-block routes -- here the
        # exact closed-form cycle
        NN = Interaction.nearest_neighbour(A_CHAIN)
        V2 = Interaction.from_table({(1,): 0.4, (-1,): 0.4, (2,): -0.1, (-2,): -0.1},
                                    b=[1.0], nu=[2.3])
        edges = np.array([[0, 1], [0, 1], [1, 2], [0, 2]])
        val, info = evaluate_graph(edges, [NN, V2, V2, V2], A_CHAIN, n_points=8,
                                   return_diagnostics=True)
        assert info["nn_mode"] is False and info["n_block_nn"] == 0
        assert info["n_simple_cycles"] == 1
        want = zeta_circle([NN * V2, V2, V2], A_CHAIN)
        assert val == pytest.approx(want, rel=1e-12)
        ref = brute_force_zeta(edges, [kernel_fn(k, A_CHAIN) for k in (NN, V2, V2, V2)],
                               A_CHAIN, 200)
        assert val == pytest.approx(ref, rel=1e-5)          # L^-3.6 truncation of the oracle
        # an all-compact block beside finite tails stays exact on its lifted torus
        G = np.array([[0, 1], [1, 2], [0, 2], [2, 3]])
        val2, info2 = evaluate_graph(G, [NN, NN, NN, V2], A_CHAIN, n_points=8,
                                     return_diagnostics=True)
        assert info2["n_block_nn"] == 1 and info2["nn_mode"] is False
        assert val2 == pytest.approx(evaluate_graph(TRIANGLE, NN, A_CHAIN)
                                     * evaluate_graph(BRIDGE, V2, A_CHAIN), rel=1e-12)
        # the legacy float path keeps its call-wide flag, byte for byte
        _, info3 = evaluate_graph(TRIANGLE, [np.inf, 2.5, 2.5], A_CHAIN, n_points=8,
                                  return_diagnostics=True)
        assert info3["nn_mode"] is True and info3["n_block_nn"] == 1
        # ... but the flag is a routing HINT, never a proof of exactness:
        # this block is MIXED, so its value must still converge in
        # n_points.  Asserting only the two flags above is what let the
        # purely-compact exact grid reach this input and freeze it.
        mixed = [evaluate_graph(TRIANGLE, [np.inf, 2.5, 2.5], A_CHAIN, n_points=n)
                 for n in (8, 32, 128)]
        assert len({float(v).hex() for v in mixed}) == 3
        assert mixed[2] == pytest.approx(mixed[1], rel=1e-8)

    def test_a_mixed_inf_block_is_not_pinned_to_the_purely_compact_grid(self):
        # REGRESSION.  ``_compact_exact_grid`` drops ``n_points`` from the
        # answer, and it may do so only because a PURELY compact block has
        # finitely many configurations.  Gating it on the CALL-WIDE
        # nearest-neighbour flag handed that licence to every block of any
        # call carrying one ν = inf edge: the mixed path graph below froze
        # at exactly 4.0, 11.25 % below the exact 4ζ(3.5), identically at
        # every n_points -- so the one convergence check a caller has
        # reported perfect stability on a wrong number.
        want = (evaluate_graph(BRIDGE, np.inf, A_CHAIN)
                * evaluate_graph(BRIDGE, 3.5, A_CHAIN))       # = 2 · 2ζ(3.5)
        assert want == pytest.approx(4.506935469268229, rel=1e-13)
        vals = [evaluate_graph(P3, [np.inf, 3.5], A_CHAIN, n_points=n)
                for n in (8, 16, 32, 64)]
        assert vals[-1] == pytest.approx(want, rel=1e-9)
        # it must CONVERGE rather than sit on one fixed grid: monotone in
        # the error and three orders better at n = 64 than at n = 8.  The
        # defect made all four of these bit-identical.
        errs = [abs(v - want) for v in vals]
        assert errs == sorted(errs, reverse=True)
        assert errs[0] > 1e3 * errs[-1]

    def test_an_inf_edge_does_not_pin_an_unrelated_power_law_block(self):
        # The contamination crossed blocks.  A K4 with no ν = inf edge in
        # it was pinned to the tiny torus because a pendant NN bridge
        # elsewhere in the SAME CALL set the call-wide flag: 0.375 against
        # a converged 0.104431, 3.6x high and bit-identical at n = 8...64.
        # Block-cut factorisation is exact at k = 0, so the joint value is
        # the K4's own value at that grid times the bridge's.
        joint_edges = np.vstack([K4, [3, 4]])
        joint_nu = np.array([2.5] * 6 + [np.inf])
        for n in (16, 32, 64):
            joint = evaluate_graph(joint_edges, joint_nu, A_CHAIN, n_points=n)
            factored = (evaluate_graph(K4, 2.5, A_CHAIN, n_points=n)
                        * evaluate_graph(BRIDGE, np.inf, A_CHAIN))
            assert joint == pytest.approx(factored, rel=1e-12)
        # and n_points still reaches the power-law block at all
        coarse = evaluate_graph(joint_edges, joint_nu, A_CHAIN, n_points=8)
        fine = evaluate_graph(joint_edges, joint_nu, A_CHAIN, n_points=64)
        assert float(coarse).hex() != float(fine).hex()

    def test_a_purely_inf_block_keeps_its_exact_grid_shortcut(self):
        # The gate must not cost the shortcut its reason to exist: when
        # every bundle of the block IS ν = inf the tiny torus is exact and
        # n_points is correctly irrelevant.  C4's value is its count of
        # nearest-neighbour homomorphisms, 6 on the chain and 6² on the
        # square lattice.
        for A, want in ((A_CHAIN, 6.0), (A_SQUARE, 36.0)):
            vals = [evaluate_graph(C4, np.inf, A, n_points=n) for n in (4, 6, 8, 16)]
            assert all(v == pytest.approx(want, rel=1e-12) for v in vals)
            assert len({float(v).hex() for v in vals}) == 1

    def test_lifted_single_momentum_shares_the_vacuum_arms_number_and_key(self):
        # lift 5 * 2 + 2 = 12 > n = 8: the on-spine k = 0 evaluation and the
        # vacuum arm used to be different evaluators under one cache key
        # (1.7e-05 apart), so a shared cache served whichever came first
        V = Interaction.from_table({(2,): 0.4, (-2,): 0.4, (1,): 0.2, (-1,): 0.2},
                                   b=[1.0], nu=[2.5])
        vac = evaluate_graph(K4S, V, A_CHAIN, n_points=8)
        k0 = evaluate_graph(K4S, V, A_CHAIN, n_points=8, terminal=4, momentum=np.zeros(1))
        assert k0 == pytest.approx(vac, rel=1e-10)          # one evaluator, reassociated
        for first in ("vac", "k0"):
            cache = {}
            calls = {"vac": lambda: evaluate_graph(K4S, V, A_CHAIN, n_points=8, block_cache=cache),
                     "k0": lambda: evaluate_graph(K4S, V, A_CHAIN, n_points=8, terminal=4,
                                                 momentum=np.zeros(1), block_cache=cache)}
            got_first = calls[first]()
            second = "k0" if first == "vac" else "vac"
            got_second = calls[second]()
            assert got_first == (vac if first == "vac" else k0)
            assert got_second == pytest.approx(k0 if second == "k0" else vac, rel=1e-10)

    def test_the_gate_reads_the_graded_grid_for_a_compact_block_as_for_a_pure_one(
            self, monkeypatch):
        # a compact K5 at a pass grid of 16 in 3D takes the pure K5's route
        # step for step: graded to 8 (the winding-safe floor 5 * 1 + 2 = 7
        # is below it), judged by the gate at that 8, declined, and shipped
        # on the floor-mode slab at the same 8 -- never the box.  An
        # earlier reading declined the graded core for every compact
        # block, so the gate saw the full 16 and the block ran the slab
        # at 16; grading with the core floored at the residual tables'
        # winding-safe size makes the two kernels indistinguishable here.
        asked, seen = [], []
        orig = frontend._dense_torus_is_accurate

        def recording(*a, **k):
            asked.append(int(a[5]))
            return orig(*a, **k)

        def fake_slab(edges, nu, A, n, **kw):
            seen.append(int(n))
            return (0.0, 0, 0)
        monkeypatch.setattr(frontend, "_dense_torus_is_accurate", recording)
        monkeypatch.setattr(frontend, "_slab_zeta", fake_slab)
        V = Interaction.from_shells(A_CUBIC, {1.0: 0.5}, b=[0.1], nu=[4.0], total=True)
        seen_by = {}
        for label, nu in (("compact", V), ("pure", 4.0)):
            asked.clear()
            seen.clear()
            _, info = evaluate_graph(K5, nu, A_CUBIC, n_points=16, accuracy="floor",
                                     return_diagnostics=True)
            assert info["n_block_dense_torus_declined"] == 1
            assert info["n_block_direct_sum"] == 0 and info["n_block_slab"] == 1
            seen_by[label] = (list(asked), list(seen))
        assert seen_by["compact"] == seen_by["pure"] == ([8], [8])
        assert seen_by["compact"][1][0] >= frontend._compact_core_floor(K5, [V] * 10) - 1

    def test_the_class_pair_is_lifted_not_dropped(self, monkeypatch):
        # strict single k at d = 3 on a gate-declined R = 2 block: the pair
        # (8, 10) sits below the lift 12 and used to be dropped (box, which
        # refuses on reach, then an open-terminal tensor at 12^9 cells)
        rungs = []

        def fake_fk(edges, nu, A, n, **kw):
            rungs.append(int(n))
            return (np.zeros(1), 0)
        monkeypatch.setattr(frontend, "_slab_zeta_finite_k", fake_fk)
        V = Interaction.from_shells(A_CUBIC, {1.0: 0.5, 2.0: 0.2}, b=[0.1], nu=[4.0],
                                    total=True)                     # (2, 0, 0): radius 2, lift 12
        assert V.support_radius == 2
        evaluate_graph(K5, V, A_CUBIC, n_points=8, terminal=1, momentum=np.array([0.1, 0.2, 0.3]),
                       accuracy="strict")
        assert rungs == [12, 14]

    def test_slab_grid_is_floored_at_the_winding_safe_size(self):
        # R = 2 on K5 (cubic): lift 5 * 2 + 2 = 12 > the pass grid 8, so the
        # floor-mode slab runs at 12, not 8, and the lift is counted
        V = Interaction.from_shells(A_CUBIC, {1.0: 0.5, 2.0: 0.2}, b=[0.1], nu=[4.0],
                                    total=True)                     # (2, 0, 0): radius 2, lift 12
        assert V.support_radius == 2
        val, info = evaluate_graph(K5, V, A_CUBIC, n_points=8, accuracy="floor",
                                   return_diagnostics=True)
        assert info["n_block_slab"] == 1 and info["n_block_compact_lift"] == 1
        at12 = float(slab_zeta(K5, np.full(10, 4.0), A_CUBIC, 12, kernels=[V] * 10)[0])
        assert val == pytest.approx(at12, rel=1e-13)
        assert val != float(slab_zeta(K5, np.full(10, 4.0), A_CUBIC, 8, kernels=[V] * 10)[0])

    @pytest.mark.slow
    def test_finite_k_pinned_grid_is_floored_at_the_winding_safe_size(self):
        V = Interaction.from_shells(A_CUBIC, {1.0: 0.5, 2.0: 0.2}, b=[0.1], nu=[4.0],
                                    total=True)                     # radius 2, lift 12
        grid, info = evaluate_graph(K5, V, A_CUBIC, n_points=8, terminal=1, accuracy="floor",
                                    return_diagnostics=True)
        assert info["n_block_compact_lift"] == 1
        k0 = evaluate_graph(K5, V, A_CUBIC, n_points=8, terminal=1, momentum=np.zeros(3),
                            accuracy="floor")
        assert grid[0, 0, 0] == pytest.approx(k0, rel=1e-12)

    def test_compact_block_takes_the_slab_at_the_pass_grid_in_floor_mode(self):
        # the d = 3 (3, 3) class is declined by the accuracy gate at n = 8;
        # a compact block used to fall to the box ladder (hours-class in a
        # pass), now it ships the pass-grid slab like a power law does
        V = Interaction.from_shells(A_CUBIC, {1.0: 0.5}, b=[0.1], nu=[4.0], total=True)
        val, info = evaluate_graph(K5, V, A_CUBIC, n_points=8, accuracy="floor",
                                   return_diagnostics=True)
        assert info["n_block_dense_torus_declined"] == 1 and info["n_block_slab"] == 1
        assert info["n_block_direct_sum"] == 0
        want = float(slab_zeta(K5, np.full(10, 4.0), A_CUBIC, 8, kernels=[V] * 10)[0])
        assert val == pytest.approx(want, rel=1e-13)


class TestGradedCoreWithTables:
    """The graded core reaches blocks with a compact part, floored at the
    winding-safe size of the SP-reduced residual tables."""

    K4S3 = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [4, 5], [5, 6], [6, 3], [2, 3]])

    @staticmethod
    def _table(R, val=0.1, tail=3.0):
        pts = {(i, j): val for i in range(-R, R + 1) for j in range(-R, R + 1)
               if (i, j) != (0, 0)}
        return Interaction.from_table(pts, b=[1.0], nu=[tail])

    def test_core_floor_sums_series_radii_and_takes_the_parallel_max(self):
        floor = frontend._compact_core_floor
        # K4: no chain, r_max = R; the cycle bound n_v R + 2 wins
        assert floor(K4, [self._table(1)] * 6) == 6
        assert floor(K4, [self._table(2)] * 6) == 10
        # one subdivided edge: the residual chain has radius 2R
        assert floor(K4S, [self._table(1)] * 7) == 8
        assert floor(K4S, [self._table(2)] * 7) == 12
        # a 4-edge chain: radius 4R, window 2 * 4R + 2 beats the cycle bound
        assert floor(self.K4S3, [self._table(1)] * 9) == 10
        assert floor(self.K4S3, [self._table(2)] * 9) == 18
        # mixed radii on one block: the max radius sets the cycle bound
        mixed = [self._table(2)] + [self._table(1)] * 5
        assert floor(K4, mixed) == 10
        # no table anywhere: no floor
        assert floor(K4, [Interaction.power_law(3.0)] * 6) == 0
        assert floor(K4, None) == 0
        assert frontend._graded_core_with_tables(10, 32, K4, [self._table(1)] * 6) == 10
        assert frontend._graded_core_with_tables(10, 32, K4, [self._table(4)] * 6) == 18
        # floored above the marginal-shrink line: leave the block alone
        assert frontend._graded_core_with_tables(10, 20, K4, [self._table(4)] * 6) == 20

    def test_mixed_dense_block_is_graded_at_k0_with_the_core_floored(self, monkeypatch):
        grids = []
        orig = frontend.hybrid_zeta

        def recording(edges, nu, A, n, *a, **k):
            grids.append((int(n), k.get("core_n_points")))
            return orig(edges, nu, A, n, *a, **k)
        monkeypatch.setattr(frontend, "hybrid_zeta", recording)
        V = Interaction.from_shells(A_HEX, {1.0: 0.5}, b=[0.1], nu=[3.0], total=True)
        raw = evaluate_graph(K4, V, A_HEX, n_points=32, core_grading=False, richardson=False)
        grids.clear()
        grd, info = evaluate_graph(K4, V, A_HEX, n_points=32, richardson=False,
                                   return_diagnostics=True)
        assert info["n_block_core_graded"] == 1
        assert info["n_block_core_graded_compact"] == 1
        core = grids[-1][0]
        assert core % 2 == 0 and frontend._compact_core_floor(K4, [V] * 6) <= core < 32
        ref = float(np.real(direct_sum_extrapolated(
            K4, np.full(6, 3.0), A_HEX, L_list=(6, 7, 8, 9, 10), n_correction_terms=3,
            root=0, kernels=[V] * 6)))
        # measured: raw 9.6e-10, graded 1.5e-08 against this reference, the
        # same 15x the pure nu = 3 kernel pays (2.7e-08 -> 4.7e-07)
        assert raw == pytest.approx(ref, rel=1e-7)
        assert grd == pytest.approx(ref, rel=1e-6)
        assert grd != raw

    def test_mixed_dense_block_is_graded_on_the_bz_grid(self, monkeypatch):
        cores = []
        orig = frontend.hybrid_zeta

        def recording(edges, nu, A, n, *a, **k):
            cores.append((int(n), k.get("core_n_points")))
            return orig(edges, nu, A, n, *a, **k)
        monkeypatch.setattr(frontend, "hybrid_zeta", recording)
        V = Interaction.from_shells(A_HEX, {1.0: 0.5}, b=[0.1], nu=[3.0], total=True)
        raw = evaluate_graph(K4, V, A_HEX, n_points=32, terminal=3, core_grading=False)
        cores.clear()
        grd, info = evaluate_graph(K4, V, A_HEX, n_points=32, terminal=3,
                                   return_diagnostics=True)
        assert info["n_block_core_graded"] == 1
        assert info["n_block_core_graded_compact"] == 1
        n, core = cores[-1]
        assert n == 32 and core is not None
        assert core % 2 == 0 and frontend._compact_core_floor(K4, [V] * 6) <= core < 32
        # measured against the box at the nodes (0,0) / (8,0) / (8,8):
        # raw 9.6e-10 / 1.5e-09 / 1.0e-09, graded 1.5e-08 / 3.7e-10 / 1.2e-10
        assert np.max(np.abs(grd - raw)) <= 1e-6 * np.max(np.abs(raw))
        assert not np.array_equal(grd, raw)

    def test_a_large_support_declines_below_its_floor_and_grades_above_it(self, monkeypatch):
        cores = []
        orig = frontend.hybrid_zeta

        def recording(edges, nu, A, n, *a, **k):
            cores.append((int(n), k.get("core_n_points")))
            return orig(edges, nu, A, n, *a, **k)
        monkeypatch.setattr(frontend, "hybrid_zeta", recording)
        V = self._table(4, val=0.05)                # Chebyshev radius 4: K4 floor 18
        assert frontend._compact_core_floor(K4, [V] * 6) == 18
        # n = 16: the floor exceeds the grid, the block is left alone --
        # bit-identical to grading switched off
        raw16 = evaluate_graph(K4, V, A_SQUARE, n_points=16, core_grading=False, richardson=False)
        grd16, info16 = evaluate_graph(K4, V, A_SQUARE, n_points=16, richardson=False,
                                       return_diagnostics=True)
        assert info16["n_block_core_graded"] == 0 and grd16 == raw16
        # n = 32: the rule's core is lifted to (at least) 18 and grading fires
        cores.clear()
        grd32, info32 = evaluate_graph(K4, V, A_SQUARE, n_points=32, richardson=False,
                                       return_diagnostics=True)
        assert info32["n_block_core_graded_compact"] == 1
        assert 18 <= cores[-1][0] <= 24
        raw32 = evaluate_graph(K4, V, A_SQUARE, n_points=32, core_grading=False, richardson=False)
        assert grd32 == pytest.approx(raw32, rel=1e-6)


class TestOracles:
    @pytest.mark.parametrize("edges", [P3, TRIANGLE, THETA, DIAMOND])
    def test_mixed_kernels_against_brute_force_d1(self, edges):
        A = A_CHAIN
        V = Interaction.from_table({(1,): 0.4, (-1,): 0.4, (0,): 0.2}, b=[0.7, -0.3], nu=[4.5, 6.0])
        val = evaluate_graph(edges, V, A, n_points=64)
        ref = brute_force_zeta(edges, [kernel_fn(V, A)] * len(edges), A, 40)
        # both truncated (torus n = 64 ~ 64^-3.5, box L = 40 ~ 40^-3.5)
        assert val == pytest.approx(ref, rel=1e-5)
        # the compact part is really in: the pure tail differs at O(1)
        pure = evaluate_graph(edges, Interaction(b=[0.7, -0.3], nu=[4.5, 6.0]), A, n_points=64)
        assert abs(pure / val - 1.0) > 1e-2

    def test_two_term_power_law_k4_multilinear(self):
        # exact linearity in the FIRST edge at fixed routing: mixed on edge 0
        # only, the rest plain; the engines see the same block topology
        b, nu = (0.7, -0.3), (2.5, 4.0)
        V = Interaction(b=list(b), nu=list(nu))
        kern = [V] + [Interaction.power_law(3.0)] * 5
        got = evaluate_graph(K4, kern, A_SQUARE, n_points=8, richardson=False)
        want = sum(bj * evaluate_graph(K4, [nj] + [3.0] * 5, A_SQUARE, n_points=8, richardson=False)
                   for bj, nj in zip(b, nu))
        assert got == pytest.approx(want, rel=1e-12)

    def test_mixed_kernel_diamond_through_the_algebra_route(self):
        # a diamond (4-cycle + chord) at sigma = 1.2 < 1.49 -> the tw-2
        # algebra route (graph_from_edges with kernels); a theta graph
        # would NOT do: its parallel pair merges into a triangle, which
        # is a simple cycle and takes the closed form.
        V = Interaction.from_table(cross(1, 0.3, origin=0.2), b=[1.0], nu=[2.2])
        val, info = evaluate_graph(DIAMOND, V, A_CHAIN, n_points=64, return_diagnostics=True)
        assert info["n_block_algebra"] == 1
        ref = brute_force_zeta(DIAMOND, [kernel_fn(V, A_CHAIN)] * 5, A_CHAIN, 40)
        assert val == pytest.approx(ref, rel=1e-4)            # both truncated; agreement in the window
        theta, info = evaluate_graph(THETA, V, A_CHAIN, return_diagnostics=True)
        assert info["n_simple_cycles"] == 1                   # the merged theta IS a triangle
        assert theta == zeta_circle([V ** 2, V, V], A_CHAIN)


class TestGuards:
    def test_block_grid_is_lifted_for_a_compact_part(self):
        # a block with a compact part runs on max(n_points, n_v R + 2):
        # a pass grid that cannot hold the support is lifted, not refused
        V = Interaction.from_table({(3,): 0.5, (-3,): 0.5}, b=[1.0], nu=[2.5])
        lifted = evaluate_graph(DIAMOND, V, A_CHAIN, n_points=4)
        assert lifted == evaluate_graph(DIAMOND, V, A_CHAIN, n_points=4 * 3 + 2)
        assert lifted != evaluate_graph(DIAMOND, V, A_CHAIN, n_points=32)
        lifted, info = evaluate_graph(DIAMOND, V, A_CHAIN, n_points=4, return_diagnostics=True)
        assert info["n_block_compact_lift"] == 1
        # an algebra-route block (sigma < 1.49) whose composed support
        # exceeds the pass grid keeps the algebra (its analytic tail) on a
        # grid of sum_e R_e + 1 = 16 points: the same value as a pass that
        # asks for 16
        W = Interaction.from_table({(3,): 0.5, (-3,): 0.5}, b=[1.0], nu=[2.2])
        val, info = evaluate_graph(DIAMOND, W, A_CHAIN, n_points=8, return_diagnostics=True)
        assert info["n_block_algebra"] == 1 and info["n_block_tensor"] == 0
        assert val == evaluate_graph(DIAMOND, W, A_CHAIN, n_points=16)
        # the standalone engine keeps the loud guard
        with pytest.raises(InteractionSupportError):
            from gzl.tensor_network import graph_zeta_general_at_zero
            graph_zeta_general_at_zero(DIAMOND, np.full(5, 2.5), A_CHAIN, 4, kernels=[V] * 5)

    def test_compact_part_at_finite_k_takes_the_lifted_form_factor_path(self):
        V = Interaction.from_table({(1,): 0.4, (-1,): 0.4, (0,): 0.2}, b=[1.0], nu=[6.0])
        k = np.array([0.2])
        val, info = evaluate_graph(DIAMOND, V, A_CHAIN, terminal=2, momentum=k, n_points=8,
                                   return_diagnostics=True)
        assert info["n_block_tensor"] == 1 and info["n_block_mixed_kernel"] == 1
        ref = brute_force_zeta(DIAMOND, [kernel_fn(V, A_CHAIN)] * 5, A_CHAIN, 30,
                               terminal=2, k_frac=k)
        assert val == pytest.approx(ref, rel=1e-5)
        grid = evaluate_graph(DIAMOND, V, A_CHAIN, terminal=2, n_points=8)
        assert grid.shape == (8,) and np.isfinite(grid).all()

    def test_unsupported_tail_on_a_dense_block_at_finite_k(self):
        # the compact finite-k path used to sit ABOVE the dense refusal:
        # a tail <= d on a treewidth-3 spine block returned a truncation
        # artefact at finite k while raising at k = 0
        V = Interaction.from_table({(1,): 0.3, (-1,): 0.3}, b=[1.0], nu=[1.0])
        for kw in (dict(terminal=1, momentum=np.array([0.2])),
                   dict(terminal=1, momentum=np.zeros(1)),
                   dict(terminal=1)):
            with pytest.raises(UnsupportedLatticeSumError):
                evaluate_graph(K4, V, A_CHAIN, n_points=8, **kw)

    def test_compact_block_at_finite_k_keeps_the_split(self):
        # a mixed block whose winding-safe grid fits the pass grid takes
        # the legacy finite-k dispatch with its split (d = 1 keeps the
        # split at finite k), not a separate path
        V = Interaction.from_table({(1,): 0.3, (-1,): 0.3}, b=[1.0], nu=[2.5])
        k = np.array([0.3])
        val, info = evaluate_graph(K4S, V, A_CHAIN, n_points=16, terminal=4, momentum=k,
                                   return_diagnostics=True)
        assert info["n_block_split"] == 1 and info["n_block_compact_lift"] == 0
        want = hybrid_zeta(K4S, np.full(7, 2.5), A_CHAIN, 16, source=0, terminal=4,
                           momentum=k, sp_n_points=info["sp_n_points"], kernels=[V] * 7)
        assert val == pytest.approx(float(np.real(want)), rel=1e-12)

    def test_finite_k_lift_for_a_support_the_pass_grid_cannot_hold(self):
        # R = 2 on the diamond at n_points = 4 (lift 5 * 2 + 2 = 12): the
        # exact form factor on the lifted torus; replacing the lift by the
        # nearest-neighbour rule n_v + 2 was 61 % wrong here
        V = Interaction.from_table({(2,): 0.4, (-2,): 0.4, (1,): 0.2, (-1,): 0.2, (0,): 0.1},
                                   b=[1.0], nu=[6.0])
        k = np.array([0.2])
        val, info = evaluate_graph(DIAMOND, V, A_CHAIN, n_points=4, terminal=2, momentum=k,
                                   return_diagnostics=True)
        assert info["n_block_compact_lift"] == 1
        ref = brute_force_zeta(DIAMOND, [kernel_fn(V, A_CHAIN)] * 5, A_CHAIN, 30,
                               source=0, terminal=2, k_frac=k)
        assert val == pytest.approx(ref, rel=1e-6)          # measured 1.6e-07
        # a pass grid that holds the lift takes the legacy arm (with its
        # split), which is at least as close to the reference
        v12 = evaluate_graph(DIAMOND, V, A_CHAIN, n_points=12, terminal=2, momentum=k)
        assert abs(v12 - ref) <= abs(val - ref)
        grid = evaluate_graph(DIAMOND, V, A_CHAIN, n_points=4, terminal=2)
        for j in range(4):
            single = evaluate_graph(DIAMOND, V, A_CHAIN, n_points=4, terminal=2,
                                    momentum=np.array([j / 4]))
            assert grid[j] == pytest.approx(single, rel=1e-12)

    def test_finite_k_algebra_lift_delivers_the_callers_grid(self):
        # sum_e R_e + 1 = 16 > n_points = 8 on a sigma < 1.49 spine block:
        # the algebra runs at 16 and samples the caller's 8-point grid
        # exactly (the regular part is a trigonometric polynomial)
        W = Interaction.from_table({(3,): 0.5, (-3,): 0.5}, b=[1.0], nu=[2.2])
        grid8, info = evaluate_graph(DIAMOND, W, A_CHAIN, n_points=8, terminal=2,
                                     return_diagnostics=True)
        assert info["n_block_algebra"] == 1
        grid16 = evaluate_graph(DIAMOND, W, A_CHAIN, n_points=16, terminal=2)
        assert np.allclose(grid8, grid16[::2], rtol=1e-12, atol=0)
        for j in (1, 3):
            single = evaluate_graph(DIAMOND, W, A_CHAIN, n_points=8, terminal=2,
                                    momentum=np.array([j / 8]))
            assert grid8[j] == pytest.approx(single, rel=1e-12)

    def test_unsupported_tail_on_a_dense_block(self):
        V = Interaction(b=[1.0, 0.5], nu=[1.0, 3.0])         # tail 1.0 <= d = 1
        with pytest.raises(UnsupportedLatticeSumError):
            evaluate_graph(K4, V, A_CHAIN, n_points=8)
        T = Interaction.from_table(cross(1, 0.3))
        assert np.isfinite(evaluate_graph(K4, T, A_CHAIN, n_points=0))


class TestBlockCache:
    def test_equal_tails_different_tables_never_collide(self):
        V1 = Interaction.from_table(cross(1, 0.3), b=[1.0], nu=[3.0])
        V2 = Interaction.from_table(cross(1, 0.31), b=[1.0], nu=[3.0])
        for edges in (BRIDGE, TRIANGLE, DIAMOND, BRIDGE_TRI):
            a = evaluate_graph(edges, V1, A_CHAIN, n_points=16)
            b = evaluate_graph(edges, V2, A_CHAIN, n_points=16)
            assert a != b
            for order in ((V1, V2), (V2, V1)):
                cache = {}
                got = {V: evaluate_graph(edges, V, A_CHAIN, n_points=16, block_cache=cache)
                       for V in order}
                assert got[V1] == a and got[V2] == b
                assert len(cache) == (4 if edges is BRIDGE_TRI else 2)
            cache = {}
            evaluate_graph(edges, V1, A_CHAIN, n_points=16, block_cache=cache)
            _, info = evaluate_graph(edges, V1, A_CHAIN, n_points=16, block_cache=cache,
                                     return_diagnostics=True)
            assert info["n_block_cache_hits"] >= 1

    def test_product_bundles_key_on_their_tables_not_their_tails(self):
        # a bundle is a lazy product whose key carries the factor tables;
        # keying it on the summed tail alone would serve V1's block value
        # to V2 on every graph whose bundles are all products (the shape
        # of a corpus sweep with multiplicity >= 2), silently
        V1 = Interaction.from_table(cross(1, 0.3), b=[1.0], nu=[3.0])
        V2 = Interaction.from_table(cross(1, 0.31), b=[1.0], nu=[3.0])
        assert (V1 * V1).key() != (V2 * V2).key() and (V1 * V2).key() != (V1 * V1).key()
        doubled_tri = np.array([[0, 1], [0, 1], [1, 2], [1, 2], [0, 2], [0, 2]])
        for edges in (np.array([[0, 1], [0, 1]]), doubled_tri):
            a = evaluate_graph(edges, V1, A_CHAIN, n_points=16)
            b = evaluate_graph(edges, V2, A_CHAIN, n_points=16)
            assert a != b
            for order in ((V1, V2), (V2, V1)):
                cache = {}
                got = {V: evaluate_graph(edges, V, A_CHAIN, n_points=16, block_cache=cache)
                       for V in order}
                assert got[V1] == a and got[V2] == b

    def test_kernel_keys_are_uniform_and_sortable(self):
        V = Interaction.from_table(cross(1, 0.3), b=[1.0], nu=[3.0])
        cache = {}
        evaluate_graph(BRIDGE_TRI, [V, 3.0, V, 3.0], A_CHAIN, n_points=16, block_cache=cache)
        for key in cache:
            assert key[0] in ("bridge", "cycle", "sigma")
        assert frontend._bundle_key(3.0) == 3.0 and frontend._bundle_key(V) == V.key()

    def test_parallel_plain_power_laws_key_through_the_cache(self):
        # REGRESSION: a bundle of PARALLEL plain power laws in a per-edge
        # mix arrives at ``_bundle_key`` as a ``_KernelProduct``, which has
        # no ``nu``.  Reading ``x.nu[0]`` raised AttributeError, and only
        # when a block_cache was supplied -- which every corpus pass
        # supplies, and which the README recommends for passes.
        V = Interaction.from_table(cross(1, 0.25), b=[1.0], nu=[4.0])
        edges = np.array([[0, 1], [0, 1], [1, 2], [2, 0]])   # parallel pair
        mix = [3.0, 3.0, V, V]
        loose = evaluate_graph(edges, mix, A_CHAIN, n_points=16)
        cached = evaluate_graph(edges, mix, A_CHAIN, n_points=16, block_cache={})
        assert float(loose).hex() == float(cached).hex()
        # and the demotion promise survives the cache: the Hadamard SUM of
        # two parallel ν = 3 bundles keys exactly like the float ν = 6 one,
        # so an all-plain-Interaction graph shares the all-float entries
        c_float, c_plain = {}, {}
        a = evaluate_graph(edges, [3.0, 3.0, 3.0, 3.0], A_CHAIN,
                           n_points=16, block_cache=c_float)
        b = evaluate_graph(edges, [Interaction.power_law(3.0)] * 4, A_CHAIN,
                           n_points=16, block_cache=c_plain)
        assert float(a).hex() == float(b).hex()
        assert set(c_float) == set(c_plain)
        assert frontend._bundle_key(Interaction.power_law(2.5)
                                    * Interaction.power_law(3.0)) == 5.5


class TestMixedInfAtFiniteMomentum:
    """``nu = inf`` beside finite exponents, at k != 0.

    The nearest-neighbour form factor treats EVERY edge of the block as
    the NN indicator, and the closed forms are undefined at ``nu = inf``.
    Gating that arm on the call-wide flag meant a graph carrying one
    ``inf`` edge returned the all-``inf`` dispersion for the whole graph.
    """

    def test_inf_and_finite_in_different_blocks_is_correct_not_all_nn(self):
        # each BLOCK here is pure, so both are servable and the answer is
        # simply right; the Interaction route is an independent evaluator
        NN = Interaction.nearest_neighbour(A_CHAIN)
        legacy = evaluate_graph(P3, np.array([np.inf, 2.5]), A_CHAIN,
                                terminal=2, n_points=8)
        kern = evaluate_graph(P3, [NN, Interaction.power_law(2.5)], A_CHAIN,
                              terminal=2, n_points=8)
        assert np.allclose(np.asarray(legacy, dtype=float),
                           np.asarray(kern, dtype=float), rtol=1e-12, atol=0.0)
        # the defect returned the all-NN dispersion, whose k = 0 entry is 4
        all_nn = evaluate_graph(P3, np.array([np.inf, np.inf]), A_CHAIN,
                                terminal=2, n_points=8)
        assert not np.allclose(np.asarray(legacy, dtype=float),
                               np.asarray(all_nn, dtype=float))

    def test_a_genuinely_mixed_block_is_refused_at_finite_k(self):
        # one 2-connected block whose own edges mix inf with finite has no
        # evaluator at k != 0: refuse rather than pick one of two wrong
        # answers.  The vacuum value is unaffected, and the kernels channel
        # serves every momentum.  The refusal is of the request, so it is an
        # UnsupportedRequestError, a GraphZetaError:
        # TopologyEvaluatorUnavailableError, which it used to be, means that
        # networkx cannot be imported.
        for momentum in (None, 0.0, 0.25):
            with pytest.raises(UnsupportedRequestError, match="mixes") as exc:
                evaluate_graph(TRIANGLE, np.array([np.inf, 2.5, 2.5]),
                               A_CHAIN, terminal=2, momentum=momentum,
                               n_points=8)
            assert isinstance(exc.value, GraphZetaError)
            assert not isinstance(exc.value,
                                  TopologyEvaluatorUnavailableError)
        assert evaluate_graph(TRIANGLE, np.array([np.inf, 2.5, 2.5]), A_CHAIN,
                              n_points=32) > 0.0
        NN = Interaction.nearest_neighbour(A_CHAIN)
        served = evaluate_graph(TRIANGLE, [NN] + [Interaction.power_law(2.5)] * 2,
                                A_CHAIN, terminal=2, n_points=8)
        assert np.asarray(served, dtype=float).shape == (8,)

    def test_an_all_inf_graph_at_finite_k_is_unchanged(self):
        g = evaluate_graph(P3, np.array([np.inf, np.inf]), A_CHAIN,
                           terminal=2, n_points=8)
        assert np.asarray(g, dtype=float)[0] == pytest.approx(4.0, rel=1e-14)
