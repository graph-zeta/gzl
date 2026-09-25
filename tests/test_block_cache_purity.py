# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""``block_cache`` is a memo, and a memo must not change the answer.

The invariant is one line: for any call, passing ``block_cache={}`` must
return exactly what ``block_cache=None`` returns.  It is worth its own
module because it is the property a *shared* cache silently violates,
and ``compute_series_coefficients`` shares one dict across a whole
corpus pass -- so a violation is a wrong series coefficient whose value
depends on graph ORDER, not an exception anyone would notice.

HOW THIS WAS FOUND, AND WHAT IT COST.  The d = 3 finite-k split
suppression (``_SPLIT_FINITE_K_DIMS``) makes an ON-SPINE block evaluate
WITHOUT the split while an OFF-SPINE block of the same shape, in the
same graph, evaluates WITH it.  The cache key recorded the *pass-level*
resolved grid for both, and ``_momentum_cache_key`` collapses k = 0 to
``None``, so the two keyed identically and one engine's value was served
to the other.  Measured on the shape below: 150231.4152957815 with no
cache against 150036.5470942466 with one, **1.30e-03 apart**, the winner
set by block iteration order.  Before that change the same call was
0.0e+00 apart, so the change introduced it; it was not pre-existing.

The fix is in ``frontend._make_key``: the key now records the split grid
the block is ACTUALLY evaluated at (``_fk_sp_pass`` on the spine,
``dense_sp_n`` off it) rather than the pass-level one.

THE SECOND VIOLATION OF THE SAME CLASS.  ``_momentum_cache_key``
collapses k = 0 to ``None``, and the key used to fold the spine-endpoint
roles into the WL signature only when the momentum survived that
collapse -- so an ON-SPINE block at k = 0 keyed identically to an
OFF-SPINE twin, all roles 0.  But the on-spine block is dispatched
through ``_block_at_finite_k``, where the sigma_eff Richardson ladder is
deliberately absent (the finite-k tail is not a clean power law), while
the off-spine block takes the k = 0 arm and GETS the ladder.  The cached
value was route-dependent and the key was not.  Measured before the fix:
``TWIN_K4SUB`` at d = 2, n = 8 gave 5.675e-05, and two isomorphic
K_{2,3} blocks sharing a cut vertex -- treewidth 2, so no dense routing
is involved at all -- 9.2e-07 at n = 32 and 2.9e-08 at n = 64.  Every
one of these is 0.0e+00 with ``richardson=False``, which names the
ladder, and unchanged with ``core_grading=False``, which acquits the
grading.  ``richardson=True`` is the default in ``evaluate_graph`` and
in ``compute_series_coefficients``, so a shipped corpus pass hit this
whenever an on-spine block was isomorphic to another graph's off-spine
block.  The fix folds the spine roles into the signature whenever the
block is on the spine, at every momentum; ``TestTheKeyNamesTheRoute``
pins it, and d = 2 rejoins the parametrisation above.

THE THIRD, ONE LEVEL DOWN AGAIN: WHAT COUNTS AS k = 0.  The key collapsed
every momentum with ``max |k| <= 1e-15`` onto the exact zero, but
``hybrid_zeta`` rewrites only an EXACT zero as the vacuum problem
(terminal dropped, re-pinned at the planner pin) and keeps the terminal
for 1e-16 -- on a graded core, a different truncation -- and the closed
forms resolve the |k|^sigma cusp of the lattice sum at 1e-16.  So ``0``
and ``1e-16`` were two values under one key.  Measured without a cache,
k = 0 against k = 1e-16: the treewidth-2 corpus block below, chain,
n = 512, 2.4e-05 (1.6e-03 while the exact zero was still graded);
K4SUB with the terminal on its subdivision vertex,
square lattice, n = 32, 5.7e-04; an on-spine bridge at sigma = 0.1,
3.0e-02, stored under the bridge key every vacuum graph with that
exponent reads.  The key now uses hybrid's own predicate
(``hybrid._momentum_is_zero``, exact, either sign);
``TestExactZeroIsTheOnlyZero`` pins it.

AND THE ROUNDING NEXT TO IT.  Every other momentum was rounded to 12
decimals, so momenta within ~5e-13 of each other shared an entry.  At a
reciprocal-lattice point (k integer in fractional coordinates) the bridge
closed form resolves the same cusp: at sigma = 0.1, 2e-15 then 1e-13 was
served 2.1e-02 off, and 1 - 1e-16 then 1.0 was served 3.0e-02 off; where
zeta is smooth a neighbour's value was ~2e-12 off.  The key is now the
exact float values; ``TestTheMomentumKeyIsExact`` pins it.
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl import evaluate_graph, hybrid_zeta
from gzl.frontend import _momentum_cache_key

#: Two isomorphic K4SUB blocks sharing cut vertex 0.  With ``terminal``
#: inside one of them, exactly one block is on the spine and the other is
#: not -- which is the whole point: they are the SAME shape evaluated two
#: ways in one call.
TWIN_K4SUB = np.array(
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (4, 3),
     (0, 5), (0, 6), (0, 7), (5, 6), (5, 7), (6, 8), (8, 7)], dtype=int,
)

K5_PLUS_BRIDGE = np.array(
    [(i, j) for i in range(5) for j in range(i + 1, 5)] + [(0, 5)], dtype=int,
)

#: Two isomorphic K_{2,3} blocks sharing cut vertex 4: hubs (0, 1) over
#: (2, 3, 4) and hubs (4, 5) over (6, 7, 8).  Treewidth 2 -- the tensor
#: route with the ladder, no dense routing -- so it isolates the ladder
#: as the mechanism.  ``source=0, terminal=1`` puts the first block on
#: the spine; ``source=5, terminal=4`` mirrors that onto the second.
TWIN_K23 = np.array(
    [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4),
     (4, 6), (4, 7), (4, 8), (5, 6), (5, 7), (5, 8)], dtype=int,
)
LONE_K23 = TWIN_K23[:6]

#: A treewidth-2 block of the 1qp TFIM corpus, listed with its
#: multiplicities.  With the hopping endpoints (2, 3) both kept it has a
#: genuine core (plan exponent 2), so at a non-zero k its core is graded;
#: at an exact zero it is not, and the vacuum rewrite reduces
#: vertex 3 away on the fine grid.
TWO_TERMINAL_CORE = np.repeat(
    np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (3, 5), (4, 5)]),
    [3, 1, 2, 2, 1, 1, 1], axis=0,
)

#: One block of TWIN_K4SUB: K4 with the edge 2-3 subdivided by vertex 4,
#: the dense-on-torus route at d = 2.
K4SUB = TWIN_K4SUB[:7]

#: A triangle with a pendant bridge 2-3: a vacuum graph that reads the
#: shared closed-form bridge entry.
TRI_BRIDGE = np.array([(0, 1), (1, 2), (2, 0), (2, 3)], dtype=int)


def _both_ways(edges, nu, A, **kw):
    a = evaluate_graph(edges, nu, A, block_cache=None, **kw)
    b = evaluate_graph(edges, nu, A, block_cache={}, **kw)
    return float(np.asarray(a).reshape(-1)[0]), float(np.asarray(b).reshape(-1)[0])


class TestACacheDoesNotChangeTheAnswer:

    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_the_twin_block_shape_at_finite_k(self, d):
        r"""The split regression at d = 1 and 3; at d = 2 the same call
        was the ladder violation of the module docstring."""
        a, b = _both_ways(
            TWIN_K4SUB, 3.5, np.eye(d),
            source=0, terminal=1, momentum=np.zeros(d), n_points=8,
        )
        assert a == b, f"d={d}: cache changed the value by {abs(a - b) / abs(b):.3e}"

    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_the_twin_block_shape_at_zero_momentum(self, d):
        a, b = _both_ways(TWIN_K4SUB, 3.5, np.eye(d), n_points=8)
        assert a == b

    @pytest.mark.parametrize("d", [1, 2])
    def test_a_dense_block_beside_a_bridge(self, d):
        r"""A dense K5 beside a bridge, evaluated twice on one cache: the
        second call is served both blocks and must reproduce the first
        bit for bit.  (Each block occurs once per call, so without the
        first call the cache is written and never read.  d = 3 takes
        the slab arm at 9.4 s a call, too slow for the suite.)"""
        cache = {}
        a = evaluate_graph(K5_PLUS_BRIDGE, 3.5, np.eye(d), n_points=8,
                           block_cache=cache)
        b, info = evaluate_graph(K5_PLUS_BRIDGE, 3.5, np.eye(d), n_points=8,
                                 block_cache=cache, return_diagnostics=True)
        assert info["n_block_cache_hits"] == 2
        assert a == b
        assert a == evaluate_graph(K5_PLUS_BRIDGE, 3.5, np.eye(d),
                                   n_points=8)


class TestTheKeyNamesTheRoute:
    r"""An on-spine block at k = 0 and an off-spine twin are the same
    mathematical object evaluated by two routines.  The key must say
    which, or the memo changes the answer."""

    @pytest.mark.parametrize("n_points", [32, 64])
    @pytest.mark.parametrize("s, t", [(0, 1), (5, 4)])
    def test_twin_k23_at_zero_momentum_is_bit_identical(self, n_points, s, t):
        r"""The reproducer, at the shipped default ``richardson=True``,
        with either block on the spine.  Bit-identity, not a tolerance:
        the fix is a key change and moves no value."""
        a, b = _both_ways(
            TWIN_K23, 3.5, np.eye(2), source=s, terminal=t,
            momentum=np.zeros(2), n_points=n_points, richardson=True,
        )
        assert a == b, (
            f"n={n_points} (s,t)=({s},{t}): the cache changed the value "
            f"by {abs(a - b) / abs(b):.3e}"
        )

    def test_the_on_spine_block_gets_its_own_entry(self):
        r"""Structural: the two blocks must not share a key, so the
        single call stores two sigma entries and serves no hit."""
        cache: dict = {}
        _, info = evaluate_graph(
            TWIN_K23, 3.5, np.eye(2), source=0, terminal=1,
            momentum=np.zeros(2), n_points=32, block_cache=cache,
            return_diagnostics=True,
        )
        assert info["n_block_cache_hits"] == 0
        assert sum(1 for k in cache if k[0] == "sigma") == 2

    def test_sound_reuse_between_on_spine_blocks_survives(self):
        r"""Guards against over-correcting: the same block on the spine
        of ANOTHER graph, at the same endpoints, is still served from
        the entry -- and served the fresh value, bit for bit."""
        cache: dict = {}
        kw = dict(source=0, terminal=1, momentum=np.zeros(2), n_points=32)
        evaluate_graph(TWIN_K23, 3.5, np.eye(2), block_cache=cache, **kw)
        fresh = evaluate_graph(LONE_K23, 3.5, np.eye(2), **kw)
        served, info = evaluate_graph(
            LONE_K23, 3.5, np.eye(2), block_cache=cache,
            return_diagnostics=True, **kw,
        )
        assert info["n_block_cache_hits"] == 1
        assert served == fresh

    def test_the_vacuum_block_is_not_served_the_on_spine_value(self):
        r"""The vacuum K_{2,3} takes the ladder; the on-spine entry did
        not.  A shared cache must evaluate it afresh."""
        cache: dict = {}
        evaluate_graph(
            TWIN_K23, 3.5, np.eye(2), source=0, terminal=1,
            momentum=np.zeros(2), n_points=32, block_cache=cache,
        )
        fresh = evaluate_graph(LONE_K23, 3.5, np.eye(2), n_points=32)
        served, info = evaluate_graph(
            LONE_K23, 3.5, np.eye(2), n_points=32, block_cache=cache,
            return_diagnostics=True,
        )
        # The off-spine twin's entry IS the vacuum value (same k = 0
        # arm, all roles 0), so this is a hit -- of the right entry.
        assert info["n_block_cache_hits"] == 1
        assert served == fresh


class TestExactZeroIsTheOnlyZero:
    r"""k = 0 is an exact zero, for the key as for hybrid.

    Each case below evaluates to a different number at k = 0 and at
    k = 1e-16, by a different mechanism: the treewidth-2 block (the
    exact zero ungraded as the vacuum problem, 1e-16 graded with the
    terminal kept), the dense-on-torus route (graded at both, vacuum
    rewrite against kept terminal), and a closed-form bridge resolving
    the |k|^sigma cusp.  The key used to give both momenta one entry."""

    CASES = [
        pytest.param(TWO_TERMINAL_CORE, 2.0, np.eye(1),
                     dict(source=2, terminal=3, n_points=512,
                          accuracy="floor"), id="tw2-graded-chain"),
        pytest.param(K4SUB, 2.5, np.eye(2),
                     dict(source=0, terminal=4, n_points=32),
                     id="dense-torus-square"),
        pytest.param(np.array([(0, 1)]), 1.1, np.eye(1),
                     dict(source=0, terminal=1, n_points=16),
                     id="bridge-sigma-0.1"),
    ]

    @pytest.mark.parametrize("edges, nu, A, kw", CASES)
    def test_zero_and_1e16_in_both_orders_are_the_uncached_values(
            self, edges, nu, A, kw):
        d = A.shape[0]
        ks = (np.zeros(d), np.full(d, 1e-16))
        fresh = [evaluate_graph(edges, nu, A, momentum=k, **kw) for k in ks]
        # The premise: the two momenta are different numbers here, so a
        # shared entry is visible.  If a change ever makes them equal,
        # this case no longer tests anything and needs replacing.
        assert fresh[0] != fresh[1]
        for order in ((0, 1), (1, 0)):
            cache: dict = {}
            for i in order:
                got = evaluate_graph(edges, nu, A, momentum=ks[i],
                                     block_cache=cache, **kw)
                assert got == fresh[i], (
                    f"order {order}: k = {ks[i]} served "
                    f"{abs(got - fresh[i]) / abs(fresh[i]):.3e} off its "
                    f"uncached value"
                )

    def test_a_batch_through_zero_is_the_uncached_values(self):
        r"""The same thing inside ONE call: a d = 1 batch shares its
        cache across its momenta, which is how a dispersion sampled
        through the zone centre meets it."""
        kw = dict(source=2, terminal=3, n_points=512, accuracy="floor")
        fresh = {k: evaluate_graph(TWO_TERMINAL_CORE, 2.0, np.eye(1),
                                   momentum=k, **kw) for k in (0.0, 1e-16)}
        for batch in ([0.0, 1e-16], [1e-16, 0.0]):
            got = evaluate_graph(TWO_TERMINAL_CORE, 2.0, np.eye(1),
                                 momentum=np.array(batch), block_cache={},
                                 **kw)
            assert list(got) == [fresh[k] for k in batch]

    def test_a_near_zero_bridge_does_not_reach_the_vacuum_entry(self):
        r"""The bridge key is shared with every vacuum graph, so a
        near-zero on-spine bridge used to contaminate OTHER graphs:
        3.0e-02 on this one at sigma = 0.1."""
        nu, n = 1.1, 16
        cache: dict = {}
        evaluate_graph(np.array([(0, 1)]), nu, np.eye(1), source=0,
                       terminal=1, momentum=[1e-16], n_points=n,
                       block_cache=cache)
        fresh = evaluate_graph(TRI_BRIDGE, nu, np.eye(1), n_points=n)
        served = evaluate_graph(TRI_BRIDGE, nu, np.eye(1), n_points=n,
                                block_cache=cache)
        assert served == fresh

    @pytest.mark.parametrize("k, is_zero", [
        ([0.0], True), ([-0.0], True),
        ([1e-15], False),     # the old tolerance's boundary: was collapsed
        ([1e-16], False), ([5e-324], False),
    ])
    def test_the_key_collapses_exactly_what_hybrid_rewrites(self, k, is_zero):
        r"""Behavioural, on hybrid itself: on a graded core its vacuum
        rewrite is the vacuum call bit for bit, and a kept terminal is a
        different truncation.  The key must collapse the same set."""
        kw = dict(n_points=512, core_n_points=16)
        nu = np.full(len(TWO_TERMINAL_CORE), 2.0)
        vacuum = hybrid_zeta(TWO_TERMINAL_CORE, nu, np.eye(1), source=2, **kw)
        got = hybrid_zeta(TWO_TERMINAL_CORE, nu, np.eye(1), source=2,
                          terminal=3, momentum=np.array(k), **kw)
        assert (got == vacuum) is is_zero
        assert (_momentum_cache_key(np.array(k)) is None) is is_zero

    @pytest.mark.parametrize("k", [[0.0, -0.0], [-0.0, -0.0]])
    def test_signed_zeros_are_one_momentum(self, k):
        r"""Mixed-sign zeros collapse to the one k = 0 entry, which is
        sound only because every route evaluates them bit-identically
        (measured per route, see ``hybrid._momentum_is_zero``); this
        pins it on the dense-on-torus route."""
        assert _momentum_cache_key(np.array(k)) is None
        kw = dict(source=0, terminal=4, n_points=32)
        assert (evaluate_graph(K4SUB, 2.5, np.eye(2), momentum=np.array(k), **kw)
                == evaluate_graph(K4SUB, 2.5, np.eye(2), momentum=np.zeros(2), **kw))


class TestTheMomentumKeyIsExact:
    r"""Every non-zero momentum is keyed by its exact float values.

    The key used to round momenta to 12 decimals.  At a reciprocal-lattice
    point the bridge closed form resolves the |k|^sigma cusp, so momenta
    that round alike are different numbers there; where zeta is smooth
    they are ~2e-12 apart, which a memo must not paper over either."""

    BRIDGE = np.array([(0, 1)])
    KW = dict(source=0, terminal=1, n_points=16)

    @pytest.mark.parametrize("k1, k2", [
        pytest.param(2e-15, 1e-13, id="beside-the-zone-centre"),
        pytest.param(1.0 - 1e-16, 1.0, id="at-the-next-zone-centre"),
        pytest.param(0.3, 0.3 + 4e-13, id="smooth"),
    ])
    def test_rounding_equal_momenta_in_both_orders_are_the_uncached_values(
            self, k1, k2):
        nu = 1.1                              # sigma = 0.1, the sharpest cusp
        fresh = {k: evaluate_graph(self.BRIDGE, nu, np.eye(1), momentum=k,
                                   **self.KW) for k in (k1, k2)}
        # The premise, as above: two different numbers, so a shared
        # entry is visible.
        assert fresh[k1] != fresh[k2]
        for order in ((k1, k2), (k2, k1)):
            cache: dict = {}
            for k in order:
                got = evaluate_graph(self.BRIDGE, nu, np.eye(1), momentum=k,
                                     block_cache=cache, **self.KW)
                assert got == fresh[k], (
                    f"order {order}: k = {k!r} served "
                    f"{abs(got - fresh[k]) / abs(fresh[k]):.3e} off its "
                    f"uncached value"
                )

    def test_a_signed_zero_component_is_one_momentum(self):
        r"""``-0.0 == 0.0`` inside the key tuple, which is sound only
        because every route evaluates the two bit-identically (measured on
        square and triangular cells); pinned here on the bridge of a
        triangular cell, where ``A^-T k`` mixes the components."""
        A = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
        k_pos, k_neg = np.array([0.3, 0.0]), np.array([0.3, -0.0])
        assert _momentum_cache_key(k_pos) == _momentum_cache_key(k_neg)
        assert (evaluate_graph(self.BRIDGE, 2.1, A, momentum=k_neg, **self.KW)
                == evaluate_graph(self.BRIDGE, 2.1, A, momentum=k_pos, **self.KW))
