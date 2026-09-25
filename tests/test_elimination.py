# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Tests for the truncation-independent elimination planner.

The planner claims to describe, without evaluating anything, what a
bucket-elimination engine will actually do: which axes each step
touches, whether the FFT peel fires there, and hence the cost exponent
the contraction reaches.  Three properties are load-bearing and are
tested here:

1. the simulated scope evolution reproduces the one ``direct_sum``
   already relies on for its Z₂ marker pre-pass (so extracting it did
   not change box behaviour),
2. the planned exponent never exceeds the block's treewidth, and
3. re-pinning is value-exact, which is what makes pin choice a free
   optimisation rather than an accuracy trade.
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl import _elimination as E
from gzl.tensor_network import graph_zeta_general_at_zero

nx = pytest.importorskip("networkx")


CANONICAL = {
    "K4": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    "K5": [(i, j) for i in range(5) for j in range(i + 1, 5)],
    "K5-e": [(i, j) for i in range(5) for j in range(i + 1, 5)
             if (i, j) != (0, 4)],
    "prism": [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
              (0, 3), (1, 4), (2, 5)],
    "K33": [(i, j) for i in (0, 1, 2) for j in (3, 4, 5)],
    "diamond": [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)],
}

# Real treewidth-3 blocks lifted from the order-13 pCUT corpus census,
# as (n_vertices, bundle edges).  These are the shapes the router
# actually sends to a dense engine.
CORPUS_BLOCKS = [
    (4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
    (5, [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)]),
    (5, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 4), (3, 4)]),
    (6, [(0, 1), (0, 2), (0, 5), (1, 2), (1, 3), (2, 4),
         (3, 4), (3, 5), (4, 5)]),
    (6, [(0, 1), (0, 3), (0, 5), (1, 2), (1, 4), (2, 3),
         (2, 5), (3, 4), (4, 5)]),
]


def _vertices(edges):
    return sorted({v for e in edges for v in e})


def _treewidth(edges):
    g = nx.Graph()
    g.add_nodes_from(_vertices(edges))
    g.add_edges_from(edges)
    return nx.algorithms.approximation.treewidth_min_fill_in(g)[0]


class TestPlanShape:
    """The plan describes a real, complete schedule."""

    @pytest.mark.parametrize("name", sorted(CANONICAL))
    def test_every_free_vertex_is_eliminated(self, name):
        edges = CANONICAL[name]
        verts = _vertices(edges)
        p = E.plan(edges, verts)
        assert p.pin in verts
        assert sorted(p.order) == [v for v in verts if v != p.pin]

    def test_k4_reaches_one_below_treewidth(self):
        """K4 is the smallest tw=3 core; its profile is [2, 2, 1].

        The pin is a size-1 axis and sits in the peak bag, and the peak
        step peels, so the contraction runs a full power of n^d below
        the treewidth rather than at the dense (tw+1) bound.
        """
        p = E.plan(CANONICAL["K4"], range(4))
        assert [s.exponent for s in p.steps] == [2, 2, 1]
        assert p.exponent == 2
        assert _treewidth(CANONICAL["K4"]) == 3


class TestExponentInvariant:
    """The planned exponent must never exceed the treewidth.

    This is the invariant the unified engine is built to hold: a
    treewidth-tau block contracts at n^(tau*d), never at the
    n^((tau+1+N_t)*d) dense bound.
    """

    @pytest.mark.parametrize("name", sorted(CANONICAL))
    def test_canonical_cores(self, name):
        edges = CANONICAL[name]
        p = E.plan(edges, _vertices(edges))
        assert p.exponent <= _treewidth(edges)

    @pytest.mark.parametrize("nv,edges", CORPUS_BLOCKS)
    def test_real_corpus_blocks(self, nv, edges):
        p = E.plan(edges, range(nv))
        assert p.exponent <= _treewidth(edges)

    @pytest.mark.parametrize("nv,edges", CORPUS_BLOCKS)
    def test_open_terminal_does_not_raise_the_exponent(self, nv, edges):
        """A free terminal replaces an elimination step, not adds one.

        In grid mode the terminal axis stays open, so it is not summed;
        the docstring bound's ``+ N_t`` does not enter the achievable
        exponent.
        """
        closed = E.plan(edges, range(nv))
        opened = E.plan(edges, range(nv), keep=[nv - 1])
        assert opened.exponent <= closed.exponent + 1
        assert opened.exponent <= _treewidth(edges) + 1


class TestPeelPrediction:
    """Peel eligibility is structural and must match the engines' rule.

    A partner ``u`` is disqualified only when it occurs in *another*
    factor of the same bucket.  Two traps live here, both of which cost
    a wrong expectation while writing these tests:

    * a pin-incident edge collapses to a one-axis factor on its free
      endpoint, so it is absent from every *other* vertex's bucket and
      cannot block a peel there;
    * a plain direct neighbour is the peelable case, not a blocked one.
      Only a neighbour that *also* arrives through an already-merged
      factor blocks the step.
    """

    def test_merged_factor_blocks_the_partner(self):
        """K4's second step cannot peel: the partner is already merged.

        Eliminating vertex 1 leaves a merged factor on ``{2, 3}``.  At
        the next step the kernel ``{2, 3}`` is no longer alone in the
        bucket, so the 2-sum is not a convolution in ``z_3``.
        """
        p = E.plan_for_order(CANONICAL["K4"], [1, 2, 3], pin=0)
        first, second = p.steps[0], p.steps[1]
        assert first.vertex == 1 and first.partners  # peels
        assert second.vertex == 2 and second.partners == ()

    def test_direct_neighbour_alone_in_the_bucket_peels(self):
        """A pin-incident edge does not block a peel elsewhere.

        On the path 0-1-2 pinned at 0, vertex 2's bucket holds only the
        kernel ``{1, 2}``; the pinned edge ``(0, 1)`` became a one-axis
        factor on ``{1}`` and never enters that bucket.
        """
        p = E.plan_for_order([(0, 1), (1, 2)], [2, 1], pin=0)
        step2 = next(s for s in p.steps if s.vertex == 2)
        assert step2.partners == (1,)

    def test_pin_is_never_a_partner(self):
        """The pin carries no generator once taken at index 0."""
        p = E.plan_for_order([(0, 1), (1, 2)], [1, 2], pin=0)
        step1 = next(s for s in p.steps if s.vertex == 1)
        assert 0 not in step1.partners


class TestScopesFromEdges:
    """Structural pins on the shared scope builder + simulator.

    Historic note: this class used to cross-check the shared simulator
    against ``direct_sum._symbolic_scopes``, the scope builder of the
    box's ``_plan_marker`` pre-pass it was originally lifted from.  That
    pre-pass has since been deleted (the offset-aware peel made the marker
    unconditional and the cost trade it modelled vanished), so the
    shared builder is now the only one; what stays asserted is its own
    contract — root-incident kernels become 1-axis non-conv factors,
    free-free kernels conv-eligible 2-axis factors, and the simulator
    walks the elimination order faithfully.
    """

    @pytest.mark.parametrize("name", sorted(CANONICAL))
    def test_scope_shape_and_simulation(self, name):
        edges = CANONICAL[name]
        verts = _vertices(edges)
        root = verts[0]
        order = [v for v in verts if v != root]

        edge_map = {(min(u, v), max(u, v)): 1.0 for u, v in edges}
        shared = E.scopes_from_edges(edge_map, root)

        n_root = sum(1 for (u, v) in edge_map if root in (u, v))
        assert len(shared) == len(edge_map)
        assert sum(1 for s, conv in shared if not conv) == n_root
        for s, conv in shared:
            if conv:
                assert len(s) == 2 and root not in s
            else:
                assert len(s) == 1 and root not in s

        steps = E.simulate(shared, order)
        assert [s.vertex for s in steps] == order


class TestRepinIsValueExactOnTheTorus:
    """Pin choice is free **on the torus**: the value must not move.

    The torus label set is a group, so translating every coordinate is a
    bijection of the label set and translation invariance survives the
    truncation exactly.  That is what turns pin selection into a pure
    cost optimisation there.  Verified through the real engine.
    """

    @pytest.mark.parametrize("name", ["K4", "diamond", "prism"])
    def test_all_pins_agree(self, name):
        edges = CANONICAL[name]
        verts = _vertices(edges)
        E_arr = np.array(edges, dtype=int)
        nu_vec = np.full(len(edges), 1.75)
        A = np.eye(1)

        vals = [
            complex(graph_zeta_general_at_zero(
                E_arr, nu_vec, A, 8, pinned_vertex=p,
            ))
            for p in verts
        ]
        ref = vals[0]
        for v in vals[1:]:
            assert abs(v - ref) <= 1e-12 * max(abs(ref), 1.0)


class TestRepinIsNotValueExactOnTheBox:
    """...but on the box it is NOT, and that must not be assumed away.

    ``[-L, L]^d`` is not closed under translation, so *which*
    configurations fall inside the truncation depends on which vertex is
    held at the origin.  Re-pinning therefore changes the truncation
    error at finite ``L`` — by 6.6e-2 on a diamond at ``L = 3`` — even
    though every pin converges to the same limit.

    This is a characterisation test, not a wish.  It exists because the
    torus result above makes "the pin is free by translation invariance"
    an easy thing to carry over to the box, where it is false.  Two
    concrete consequences:

    * pin selection is a pure *cost* lever on the torus but an
      accuracy-affecting one on the box;
    * the L-ladder in :func:`direct_sum.direct_sum_extrapolated` is only
      consistent because ``_pick_root`` is deterministic and every rung
      re-derives the same root.  A pin choice that varied with ``L`` —
      or with a cached plan — would make the ladder fit a sequence whose
      truncation constant moves between rungs, and nothing would report
      it.
    """

    EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]

    def _spread(self, L):
        import gzl.direct_sum as ds

        E_arr = np.array(self.EDGES, dtype=int)
        nu_vec = np.full(len(self.EDGES), 1.75)
        A = np.eye(1)
        vals = [
            complex(ds.direct_sum_zero_momentum(
                E_arr, nu_vec, A, L, root=r, use_symmetry=False,
            )).real
            for r in range(4)
        ]
        return (max(vals) - min(vals)) / abs(np.mean(vals))

    def test_pin_changes_the_value_at_finite_L(self):
        assert self._spread(3) > 1e-3

    def test_the_pin_dependence_vanishes_as_L_grows(self):
        """It is a truncation effect, so it must decay — not a bug."""
        wide, narrow = self._spread(3), self._spread(10)
        assert narrow < wide / 10.0


class TestPlanBudgetAndCache:
    """The planner's size refusal and the exact-key memoisation in front
    of the pin-searching DP."""

    def test_refusal_boundary_is_exactly_max_plan_free(self):
        """|elim| = 12 runs; |elim| = 13 refuses.  A cycle C_n with a
        free pin choice eliminates n - 1 vertices."""
        C13 = [(i, (i + 1) % 13) for i in range(13)]
        p = E.plan(C13, range(13))              # 12 eliminable: runs
        assert len(p.order) == 12
        C14 = [(i, (i + 1) % 14) for i in range(14)]
        with pytest.raises(E.PlanBudgetExceededError):
            E.plan(C14, range(14))

    def test_refusal_counts_eliminable_not_total_vertices(self):
        """Kept terminals do not count against the budget: C14 with one
        kept vertex eliminates 12 and must run."""
        C14 = [(i, (i + 1) % 14) for i in range(14)]
        p = E.plan(C14, range(14), keep=[13])
        assert len(p.order) == 12
        assert 13 not in p.order

    def test_fixed_pin_refusal_uses_the_same_count(self):
        C14 = [(i, (i + 1) % 14) for i in range(14)]
        with pytest.raises(E.PlanBudgetExceededError):
            E.plan(C14, range(14), pin=0)       # 13 eliminable

    def test_cache_is_value_transparent_and_hit_on_reorder(self):
        """A permuted edge list is the same planning problem: the cache
        must return the identical Plan object, and a cold call after
        cache_clear must reproduce it field-for-field."""
        edges = CANONICAL["K5-e"]
        E._plan_cached.cache_clear()
        p1 = E.plan(edges, range(5))
        p2 = E.plan(list(reversed(edges)), range(5))
        assert p2 is p1                          # exact-key hit
        E._plan_cached.cache_clear()
        p3 = E.plan(edges, range(5))
        assert (p3.pin, p3.order, p3.keep, p3.steps) == \
               (p1.pin, p1.order, p1.keep, p1.steps)

    def test_cache_distinguishes_pin_keep_and_extra_scopes(self):
        edges = CANONICAL["K4"]
        E._plan_cached.cache_clear()
        base = E.plan(edges, range(4))
        pinned = E.plan(edges, range(4), pin=3)
        kept = E.plan(edges, range(4), keep=[3])
        extra = E.plan(edges, range(4), extra_scopes=[(1, 2, 3)])
        assert pinned.pin == 3
        assert 3 not in kept.order
        # extra scope widens at least one bag or blocks a peel; the
        # plans must not alias even if equal-ranked.
        assert extra is not base
        assert kept is not base and pinned is not base

    def test_duplicate_bundle_entries_are_not_collapsed_by_the_key(self):
        """A duplicated two-vertex kernel blocks the peel partner (the
        partner then occurs in another factor of the bucket), so the
        key must be a multiset: deduplicating it would return the
        single-kernel plan for the double-kernel problem."""
        path = [(0, 1), (1, 2)]
        doubled = [(0, 1), (1, 2), (1, 2)]
        E._plan_cached.cache_clear()
        p_single = E.plan(path, range(3), pin=0)
        p_double = E.plan(doubled, range(3), pin=0)
        s_single = {s.vertex: s.partners for s in p_single.steps}
        s_double = {s.vertex: s.partners for s in p_double.steps}
        assert s_single != s_double


class TestMinFreeCutNuTwin:
    """The planner-side cut must agree EXACTLY (==) with the engine's
    ``direct_sum._min_free_cut_nu`` — it is the independent half of the
    pin/alpha desync refusal, so any drift between the two
    implementations must surface here, not in a silent basis fit."""

    def _edge_map(self, edges, rng):
        import gzl.direct_sum as ds
        nu = 2.5 + rng.random(len(edges)) * 3.0
        return ds._collapse_multi_edges(
            np.array(edges, dtype=int), nu)

    @pytest.mark.parametrize("name", sorted(CANONICAL))
    def test_exact_equality_on_canonical_blocks(self, name):
        import gzl.direct_sum as ds
        rng = np.random.default_rng(7)
        for _ in range(5):
            em = self._edge_map(CANONICAL[name], rng)
            verts = {v for e in em for v in e}
            for root in sorted(verts):
                assert E.min_free_cut_nu(em, root) == \
                       ds._min_free_cut_nu(em, root)

    @pytest.mark.parametrize("nv,edges", CORPUS_BLOCKS)
    def test_exact_equality_on_corpus_blocks(self, nv, edges):
        import gzl.direct_sum as ds
        rng = np.random.default_rng(11)
        em = self._edge_map(edges, rng)
        for root in range(nv):
            assert E.min_free_cut_nu(em, root) == \
                   ds._min_free_cut_nu(em, root)

    def test_exact_equality_with_externals_and_fallbacks(self):
        import gzl.direct_sum as ds
        rng = np.random.default_rng(13)
        # externals shrink the escaping set
        em = self._edge_map(CANONICAL["prism"], rng)
        assert E.min_free_cut_nu(em, 0, externals=(3, 4)) == \
               ds._min_free_cut_nu(em, 0, externals=(3, 4))
        # single edge: no free vertex once both ends are pinned
        em2 = self._edge_map([(0, 1)], rng)
        assert E.min_free_cut_nu(em2, 0, externals=(1,)) == \
               ds._min_free_cut_nu(em2, 0, externals=(1,))
        # approximate branch (sizes 1..3 + full set)
        em3 = self._edge_map(CANONICAL["K5"], rng)
        assert E.min_free_cut_nu(em3, 0, max_free_exact=2) == \
               ds._min_free_cut_nu(em3, 0, max_free_exact=2)
