# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""The dense-block routing flag (``dense_engine`` / ``sp_n_points``).

Treewidth->=3 blocks go to the torus by default at d <= 3
(``_DENSE_ENGINE_BY_D``); ``dense_engine="direct_sum"`` keeps them on
the box.  The torus route carries the sigma_eff Richardson ladder at
k = 0 and, with ``sp_n_points``, hybrid's split resolution.

Two guards are subtle enough to state:

* ``nu <= d`` is refused outright on either setting
  (``UnsupportedLatticeSumError``).  A torus engine sums a finite grid
  and returns a confident number that is ~50% wrong at nu = 0.5, so
  the flag must not be able to switch the refusal off.
* The block cache must separate the settings.  Its lookup guard
  verifies topology, nu and role by isomorphism -- never numerics -- so
  a key that omitted the routing would serve a box value to a torus
  caller with no symptom at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl import Interaction, evaluate_graph
from gzl import frontend


# K4: the smallest treewidth-3 block, and 3-connected, so SP reduction
# has nothing to remove and the split is a no-op on it by construction.
K4 = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=int)

# K4 with one edge subdivided: a treewidth-3 core plus a degree-2
# vertex.  The degree-2 vertex is the cheap escape that sets the
# truncation rate, and removing it is exactly what the split does, so
# this is the shape on which the split is NOT a no-op.
K4SUB = np.array(
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 4], [4, 3]], dtype=int,
)

# K5: 4-connected, tw = 4.  At d = 1, nu = 2.5, n_points = 32 its core is
# graded (to 20), and how far is set by the pass floor alone.
K5 = np.array([[i, j] for i in range(5) for j in range(i + 1, 5)],
              dtype=int)

A1 = np.eye(1)
A2 = np.eye(2)


class TestDefaultRoutesToTheTorus:
    r"""The shipped default sends dense blocks to the torus at d <= 3.

    The ``dense_engine="direct_sum"`` round trip below keeps the box
    route pinned --- it remains the default at d >= 4, takes a
    gate-declined block the slab cannot, runs on request, and is the
    MemoryError fallback.
    """

    def test_default_bills_the_block_to_the_torus(self):
        _, info = evaluate_graph(
            K4, 2.5, A1, n_points=16, return_diagnostics=True,
        )
        assert info["dense_engine"] == "torus"
        assert info["n_block_direct_sum"] == 0
        assert info["n_block_dense_torus"] == 1
        assert info["max_tensor_block_tw"] >= 3

    def test_finite_k_default_bills_to_the_torus(self):
        _, info = evaluate_graph(
            K4, 2.5, A1, source=0, terminal=1, n_points=8,
            return_diagnostics=True,
        )
        assert info["dense_engine"] == "torus"
        assert info["n_block_direct_sum"] == 0
        assert info["n_block_dense_torus"] == 1

    @pytest.mark.parametrize("nu", [1.5, 2.5, 4.0])
    def test_default_equals_explicit_torus(self, nu):
        assert (evaluate_graph(K4SUB, nu, A1, n_points=16)
                == evaluate_graph(K4SUB, nu, A1, n_points=16,
                                  dense_engine="torus"))

    def test_the_box_route_is_still_reachable_and_still_differs(self):
        # The old default, pinned: it is not dead code.  It stays live at
        # d >= 4, for gate-declined blocks the slab cannot take, and on
        # MemoryError, so it must keep working AND keep producing its own
        # (different) number --- if these were equal the flag would be
        # inert and every test here vacuous.
        torus = evaluate_graph(K4SUB, 2.5, A1, n_points=16)
        box = evaluate_graph(K4SUB, 2.5, A1, n_points=16,
                             dense_engine="direct_sum")
        assert torus != box
        _, info = evaluate_graph(
            K4SUB, 2.5, A1, n_points=16, dense_engine="direct_sum",
            return_diagnostics=True,
        )
        assert info["n_block_direct_sum"] == 1
        assert info["n_block_dense_torus"] == 0

    def test_d3_now_defaults_to_the_torus(self):
        r"""d = 3 routes to the torus, like d = 1 and d = 2.

        It stayed on the box until the d = 3 cycle quadrature made a
        cubic pass affordable to measure at all.  Measured cold, torus
        arm first: 5.5 s / 0.90 GB against direct_sum's 54.2 s / 3.44 GB,
        agreeing to 1.35e-09 and both 9/9 inside the Fey cubic MC noise.

        The box is NOT retired at d = 3 -- it takes gate-declined blocks
        the slab cannot and is the MemoryError fallback; see
        TestMemoryFallbackToTheBox.
        """
        assert frontend._DENSE_ENGINE_BY_D[3] == "torus"

    def test_the_split_and_the_graded_core_are_both_on_at_d3(self):
        r"""Both d = 3 knobs are now present and deliberate.

        The d = 3 route first shipped alone, with both knobs pinned off on
        the stated grounds that the fine grid costs ``sp_n_points**d``.
        At d = 3, sp = 128 is 17 MB per distinct nu -- against the 2 MB
        the d = 2 entry's own comment calls trivial -- and the split is
        worth 23x -> 2442x the box on K4SUB at the shipped n_points = 16
        (slab-referenced; see the constant's comment).

        ``_CORE_KAPPA[3]`` followed, but only once the accuracy gate was
        taught to read the GRADED grid rather than ``n_points``.  The two
        are not independent: grading sends an exponent-3 core to a
        SMALLER torus, and on its own at d = 3 that is a loss (V6E11
        graded to n = 8 with the split off is 1.44e-03, against the box's
        1.37e-04).  It pays only WITH the split -- V6E11 then lands at
        3.6e-07 for at least 57.7x the box (resolved bound; the point
        estimate is 382x but sits below the reference band) at a cost
        flat in ``n_points``: 9.2/9.5/10.0 s at n = 16/32/64 against the
        box's ~75 s.  The exponent-3 core that does NOT clear the gate
        after grading (K5, which grades to n = 8) is still sent to the
        box, which is what
        ``test_the_gate_reads_the_graded_grid_not_the_delivery_grid``
        pins.
        """
        assert frontend._SPLIT_SP_N_POINTS[3] == 128
        assert frontend._CORE_KAPPA[3] == 3.0
        eng, sp, grade = frontend._resolve_dense_routing(
            3, None, None, 16, False, True)
        assert (eng, sp, grade) == ("torus", 128, True)

    def test_unlisted_dimensions_are_conservative(self):
        # d >= 4 is unmeasured; it must fall back to the box rather than
        # inherit the flip by accident.
        _, _ = frontend._resolve_dense_routing(5, None, None, 8, False)[:2]
        assert frontend._resolve_dense_routing(
            5, None, None, 8, False)[0] == "direct_sum"


class TestThePreferredSlabRoute:
    r"""`_SLAB_PREFERRED_RUNGS` -- the (3, 2) class rides the second pin.

    Corpus census of the exponent-2 dense class, on the two symbolic
    prices the route reads: 463 of 2598 occurrences (17.8%, 113 shapes)
    are slab-inner-exponent 1 with a NO-OP split -- the sub-class that
    carried the shipped defect (the torus at the pass grid, 3.6x-58x
    worse than the box at n_points = 8), and the one where n = 18 costs
    less than the box (0.3-0.9 s vs 0.6-1.7 s) and beats it on accuracy
    on every band-resolved shape.  The 2127 split-ACTING occurrences
    are excluded: their incumbent route is measured ~2000x better than
    a raw n = 18 truncation (K4SUB below).
    """

    K4 = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    K5 = np.array([[i, j] for i in range(5) for j in range(i + 1, 5)])
    V6E11 = np.array([[0, 1], [0, 4], [1, 2], [1, 3], [1, 5], [2, 3],
                      [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]])

    def test_the_predicate_takes_exactly_the_measured_class(self):
        A3, nu = np.eye(3), np.full(6, 3.5)
        got = frontend._slab_preferred(self.K4, nu, 3, A3)
        assert got == ((16, 18), 0)
        # exponent 3 is the gate + decline arm's business, not this one
        assert frontend._slab_preferred(
            self.K5, np.full(10, 3.5), 3, A3) is None
        # inner exponent 2: the slab costs 30-60 min; held for the
        # through-router measurement
        assert frontend._slab_preferred(
            self.V6E11, np.full(11, 3.5), 3, A3) is None
        # divergent, wrong dimension, poor cell, nn: all decline
        assert frontend._slab_preferred(self.K4, np.full(6, 2.5), 3, A3) is None
        assert frontend._slab_preferred(
            self.K4, np.full(6, 2.5), 2, np.eye(2)) is None
        assert frontend._slab_preferred(
            self.K4, nu, 3,
            np.array([[1.0, 0.3, 0.0], [0.0, 1.1, 0.2],
                      [0.0, 0.0, 0.9]])) is None
        assert frontend._slab_preferred(
            self.K4, np.full(6, np.inf), 3, A3, nn_mode=True) is None

    def test_a_split_acting_block_is_excluded_and_keeps_its_split(self):
        r"""The exclusion the full suite forced.  The first draft of the
        predicate did not require the split to be a no-op, and hijacked
        K4SUB -- whose split route runs at 1.94e-08 from a resolved
        reference (2442x the box, the table beside
        ``_SPLIT_SP_N_POINTS``) -- onto a raw n = 18 truncation sitting
        4.3e-05 out: a ~2000x regression delivered by the change that
        claimed to fix accuracy.  The degree-2 escape that makes the
        split act is the same mode that makes the raw truncation slow,
        so the two conditions are one condition.
        """
        K4SUB = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3],
                          [2, 4], [4, 3]])
        nu, A3 = np.full(7, 3.5), np.eye(3)
        assert frontend._slab_preferred(K4SUB, nu, 3, A3) is None
        _, info = evaluate_graph(K4SUB, nu, A3, n_points=16,
                                 richardson=True, return_diagnostics=True)
        assert info["n_block_slab_preferred"] == 0
        assert info["n_block_dense_torus"] == 1
        assert info["n_block_split"] == 1

    def test_k4_rides_it_at_every_pass_grid(self):
        r"""The value is the n = 18 slab's, INDEPENDENT of n_points --
        that is the point: the pass grid no longer decides this block's
        accuracy.  Pinned against the slab n = 20 reference
        30.825193801160868: this route sits 1.01e-07 out, the box
        5.63e-07, the old torus-at-pass-grid route 2.83e-05.
        """
        nu, A3 = np.full(6, 3.5), np.eye(3)
        ref = 30.825193801160868
        vals = []
        for n_points in (8, 16, 32):
            v, info = evaluate_graph(self.K4, nu, A3, n_points=n_points,
                                     richardson=True,
                                     return_diagnostics=True)
            assert info["n_block_slab_preferred"] == 1
            assert info["n_block_direct_sum"] == 0
            assert info["n_block_dense_torus"] == 0
            vals.append(float(v))
        assert vals[0] == vals[1] == vals[2]
        assert abs(vals[0] - ref) / ref < 2e-7

    def test_the_self_band_tripwire_falls_back_to_the_box(self):
        r"""The two-rung self-band is the per-block safety: a pair that
        disagrees means THIS shape's truncation has not settled by
        n = 18 (nothing measured does this), and the block must land on
        the box rather than ship a number the route cannot vouch for.
        Forced here by making the lower rung lie.
        """
        nu, A3 = np.full(6, 3.5), np.eye(3)
        real = frontend._slab_zeta

        def lying_lower(edges, nuv, A, n, **kw):
            v, cells, ex = real(edges, nuv, A, n, **kw)
            if int(n) == 16:
                v = v * (1.0 + 10.0 * frontend._SLAB_SELF_BAND_MAX)
            return v, cells, ex

        frontend._slab_zeta = lying_lower
        try:
            _, info = evaluate_graph(self.K4, nu, A3, n_points=8,
                                     richardson=True,
                                     return_diagnostics=True)
        finally:
            frontend._slab_zeta = real
        assert info["n_block_slab_preferred"] == 0
        assert info["n_block_slab_selfband_declined"] == 1
        assert info["n_block_direct_sum"] == 1

    def test_a_slab_refusal_keeps_the_incumbent_torus_route(self):
        # A byte-budget refusal must not strand the block: the torus
        # ladder is still there.
        nu, A3 = np.full(6, 3.5), np.eye(3)
        real = frontend._slab_zeta

        def refusing(*a, **kw):
            raise MemoryError("forced refusal")

        frontend._slab_zeta = refusing
        try:
            v, info = evaluate_graph(self.K4, nu, A3, n_points=8,
                                     richardson=True,
                                     return_diagnostics=True)
        finally:
            frontend._slab_zeta = real
        assert info["n_block_slab_preferred"] == 0
        assert info["n_block_dense_torus"] == 1
        assert np.isfinite(float(v))


class TestFlagRoutes:
    r"""``dense_engine="torus"`` actually diverts the block."""

    def test_vacuum_routes_to_the_torus(self):
        _, info = evaluate_graph(
            K4SUB, 2.5, A1, n_points=16, dense_engine="torus",
            return_diagnostics=True,
        )
        assert info["n_block_direct_sum"] == 0
        assert info["n_block_dense_torus"] == 1
        assert info["n_block_tensor"] == 1     # not disjoint, by design
        assert info["max_tensor_block_tw"] >= 3

    def test_finite_k_routes_to_the_torus(self):
        _, info = evaluate_graph(
            K4SUB, 2.5, A1, source=0, terminal=1, n_points=16,
            dense_engine="torus", return_diagnostics=True,
        )
        assert info["n_block_direct_sum"] == 0
        assert info["n_block_dense_torus"] == 1

    def test_grid_routes_to_the_torus(self):
        arr, info = evaluate_graph(
            K4SUB, 2.5, A1, source=0, terminal=1, n_points=8,
            dense_engine="torus", return_diagnostics=True,
        )
        assert np.asarray(arr).shape == (8,)
        assert info["n_block_dense_torus"] == 1

    def test_the_two_routes_disagree_but_converge(self):
        # A liveness control on every other test in this class: if the
        # flag did nothing, the routes would agree EXACTLY and a test
        # that merely checks counters could still be measuring one
        # engine twice.  They must differ at a coarse grid and close
        # the gap as the torus is refined.
        box = evaluate_graph(
            K4SUB, 2.5, A1, n_points=16, dense_engine="direct_sum")
        coarse = evaluate_graph(
            K4SUB, 2.5, A1, n_points=16, dense_engine="torus",
            richardson=False,
        )
        fine = evaluate_graph(
            K4SUB, 2.5, A1, n_points=256, dense_engine="torus",
            richardson=False,
        )
        assert coarse != box
        assert abs(fine - box) < abs(coarse - box)

    @pytest.mark.parametrize("A", [A1, A2])
    def test_treewidth_2_blocks_are_untouched_by_the_flag(self, A):
        # The flag is scoped to dense blocks.  A 4-cycle is tw=2 and
        # must give the identical number either way -- if it moved, the
        # flag would be silently rerouting far more than it claims.
        cycle = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
        nu = float(A.shape[0]) + 1.5
        a = evaluate_graph(cycle, nu, A, n_points=8)
        b = evaluate_graph(
            cycle, nu, A, n_points=8, dense_engine="torus", sp_n_points=64,
        )
        assert a == b


class TestSplitScope:
    r"""``sp_n_points`` reaches dense torus blocks and nothing else."""

    def test_split_is_applied_and_counted(self):
        _, info = evaluate_graph(
            K4SUB, 2.5, A1, n_points=16, dense_engine="torus",
            sp_n_points=256, return_diagnostics=True,
        )
        assert info["n_block_dense_torus"] == 1
        assert info["n_block_split"] == 1
        assert info["sp_n_points"] == 256

    def test_split_changes_the_value_on_a_block_with_sp_structure(self):
        # K4SUB has a degree-2 vertex, so the split has something to do.
        plain = evaluate_graph(
            K4SUB, 2.5, A1, n_points=16, dense_engine="torus",
            richardson=False,
        )
        split = evaluate_graph(
            K4SUB, 2.5, A1, n_points=16, dense_engine="torus",
            sp_n_points=256, richardson=False,
        )
        assert plain != split

    def test_split_is_a_noop_on_a_3_connected_block(self):
        # K4 has no degree-2 vertex, so SP reduction removes nothing and
        # the split must be EXACTLY inert.  This is the control that
        # keeps the previous test honest: it shows the difference there
        # comes from the SP part, not from the split perturbing
        # everything it touches.
        plain = evaluate_graph(
            K4, 2.5, A1, n_points=16, dense_engine="torus", richardson=False,
        )
        split = evaluate_graph(
            K4, 2.5, A1, n_points=16, dense_engine="torus",
            sp_n_points=256, richardson=False,
        )
        assert plain == split

    def test_sp_n_points_equal_to_n_points_is_folded_to_no_split(self):
        _, info = evaluate_graph(
            K4SUB, 2.5, A1, n_points=32, dense_engine="torus",
            sp_n_points=32, return_diagnostics=True,
        )
        assert info["sp_n_points"] is None
        assert info["n_block_split"] == 0

    def test_a_coarser_sp_grid_is_raised_not_refused(self):
        # hybrid itself raises ValueError below n_points; the frontend
        # resolver clamps instead, so a caller sweeping n_points with a
        # fixed sp_n_points does not fall off a cliff mid-sweep.
        v, info = evaluate_graph(
            K4SUB, 2.5, A1, n_points=64, dense_engine="torus",
            sp_n_points=16, return_diagnostics=True,
        )
        assert info["sp_n_points"] is None      # clamped to 64, then folded
        assert np.isfinite(v)

    def test_split_is_ignored_off_the_torus_route(self):
        a = evaluate_graph(
            K4SUB, 2.5, A1, n_points=16, dense_engine="direct_sum")
        b = evaluate_graph(
            K4SUB, 2.5, A1, n_points=16, dense_engine="direct_sum",
            sp_n_points=512,
        )
        assert a == b


class TestNuBelowDIsRefusedOnEitherEngine:
    r"""``nu <= d`` is refused outright, on either engine setting.

    The class name and docstring used to say the box was KEPT here.  That
    was already stale: the frontend raises before the box branch is
    consulted, which is what the test below has always asserted.

    The refusal is a SUPPORT limit, not a divergence verdict.
    ``direct_sum_extrapolated`` refuses ``nu_min <= d`` and the torus
    engines have no guard at all, so neither can judge the case -- but
    convergence is governed by the cluster cut, not by ``min nu``, and
    K_4 at ``nu = 0.7``, ``d = 1`` has cut 2.10 > d.  So this guard is
    conservative: it declines graphs whose sums converge.  See
    ``frontend.UnsupportedLatticeSumError``.
    """

    @pytest.mark.parametrize("dense_engine", ["direct_sum", "torus"])
    def test_below_the_threshold_it_refuses(self, dense_engine):
        r"""A divergent lattice sum is an ERROR, not a number.

        This used to assert that the block "landed on the tensor path via
        the existing cascade" -- i.e. it codified the artefact.  The box's
        refusal was swallowed by an ``except (ValueError, MemoryError)``
        and the torus returned a truncation value that keeps climbing with
        the grid: at d = 1, nu = 0.8 it reads 5.86, 10.89, 13.62 over
        n_points = 6, 10, 14.  The class docstring above always described
        this correctly; only the assertion disagreed.
        """
        with pytest.raises(frontend.UnsupportedLatticeSumError):
            evaluate_graph(K4, 0.7, A1, n_points=16,
                           dense_engine=dense_engine)

    def test_the_old_name_still_catches(self):
        """The rename must not break a caller catching the old name."""
        assert (frontend.DivergentLatticeSumError
                is frontend.UnsupportedLatticeSumError)
        with pytest.raises(frontend.DivergentLatticeSumError):
            evaluate_graph(K4, 0.7, A1, n_points=16)

    def test_the_refusal_is_conservative_not_a_divergence_verdict(self):
        r"""The refused graph's cluster cut clears ``d``.

        This is the whole reason the error is named for lack of support
        rather than for divergence: ``min nu = 0.7 <= d = 1`` refuses,
        but convergence is governed by the minimum escaping cluster cut,
        which is ``3 nu = 2.10 > 1`` here.  If this assertion ever fails,
        the justification in ``UnsupportedLatticeSumError`` is wrong and
        the docstring must be corrected with it.
        """
        from gzl import _elimination

        edge_map = {(int(u), int(v)): 0.7 for u, v in K4}
        cut = _elimination.min_free_cut_nu(edge_map, 0)
        assert cut == pytest.approx(2.10)
        assert cut > 1.0          # > d, i.e. the sum converges

    def test_above_the_threshold_the_flag_does_divert(self):
        # Liveness control for the test above: at nu > d the same call
        # shape DOES take the new route, so a zero there means the
        # guard fired, not that the flag is inert.
        _, info = evaluate_graph(
            K4, 1.7, A1, n_points=16, dense_engine="torus",
            return_diagnostics=True,
        )
        assert info["n_block_dense_torus"] == 1


class TestBlockCacheSeparatesSettings:
    r"""One shared cache must not serve one routing's value to another.

    ``_cache_lookup`` verifies a hit by graph isomorphism on topology,
    nu and role.  It cannot see that two entries were produced by
    different engines, so the separation has to be in the key.  A
    corpus pass shares one dict across thousands of graphs, which is
    what makes a missing key field a silent, global corruption rather
    than a local one.
    """

    def test_sigma_ref_is_in_the_key(self):
        r"""A shared cache must not serve one block's CORE SIZE to another.

        The core-grading floor is read from the block's PER-EDGE
        exponents, and the signature sees only the merged bundles.  K5 at
        nu = 2.5 and K5 with one edge split into two parallel edges of
        1.25 have the same signature and describe the same lattice sum,
        but read floors 4.0 and 1.5, so at d = 1, n = 32 their cores are
        graded to 20 and 8 and their values differ by 11%.  The boolean
        ``dense_core_grade`` in the key records only that grading is on,
        never what it decided.

        Measured before the key carried the floor, with the floor still
        read over the whole graph: K5 plus a bridge at nu 2.5 or 1.2
        graded the core to 20 or 8, and whichever graph came second
        inherited the other's core -- 9.9% wrong one way, 11.0% the
        other.  Silent, with the sign set by graph ordering.

        This test asks only that the key carries what the evaluation
        read, not that a split edge should lower the floor.  If the split
        stops moving the core, the guard below fails and a new pair is
        needed.  Not reachable from ``compute_series_coefficients`` (it
        promotes a SCALAR nu, so every edge reads the same floor), which
        is why this is pinned at the ``evaluate_graph`` level.
        """
        edges_a, nu_a = K5, np.full(len(K5), 2.5)
        edges_b = np.vstack([K5, [[0, 1]]])
        nu_b = np.array([1.25] + [2.5] * (len(K5) - 1) + [1.25])

        ref_a = evaluate_graph(edges_a, nu_a, A1, n_points=32, block_cache={})
        ref_b = evaluate_graph(edges_b, nu_b, A1, n_points=32, block_cache={})
        # The two must genuinely differ, or the test is vacuous.
        assert abs(ref_a - ref_b) > 1e-3 * abs(ref_a)

        graphs = {"a": (edges_a, nu_a, ref_a), "b": (edges_b, nu_b, ref_b)}
        for first, second in (("a", "b"), ("b", "a")):
            shared = {}
            for name in (first, second):
                edges, nu, want = graphs[name]
                assert evaluate_graph(edges, nu, A1, n_points=32,
                                      block_cache=shared) == want

    def test_a_block_is_shared_across_neighbours(self):
        r"""The floor is the block's own, so a block's cache entry no
        longer depends on what sits beside it: K5 beside a diamond at
        nu 2.5 and beside one at 1.2 is one entry, served to the second
        graph bit for bit as it would be computed.
        """
        diamond = np.array([[4, 5], [4, 6], [5, 6], [5, 7], [6, 7]])
        edges = np.vstack([K5, diamond])
        nu_a = np.array([2.5] * len(K5) + [2.5] * len(diamond))
        nu_b = np.array([2.5] * len(K5) + [1.2] * len(diamond))
        fresh_b = evaluate_graph(edges, nu_b, A1, n_points=32,
                                 block_cache={})
        shared: dict = {}
        evaluate_graph(edges, nu_a, A1, n_points=32, block_cache=shared)
        got_b, info = evaluate_graph(edges, nu_b, A1, n_points=32,
                                     block_cache=shared,
                                     return_diagnostics=True)
        assert info["n_block_cache_hits"] == 1     # the K5, not the diamond
        assert got_b == fresh_b

    def test_routes_do_not_cross_serve(self):
        cache: dict = {}
        box = evaluate_graph(
            K4SUB, 2.5, A1, n_points=16, dense_engine="direct_sum",
            block_cache=cache,
        )
        torus = evaluate_graph(
            K4SUB, 2.5, A1, n_points=16, dense_engine="torus",
            block_cache=cache,
        )
        assert box != torus
        # ...and re-asking each way returns its own number, i.e. both
        # entries coexist rather than the second overwriting the first.
        assert evaluate_graph(
            K4SUB, 2.5, A1, n_points=16, dense_engine="direct_sum",
            block_cache=cache,
        ) == box

    def test_sp_grids_do_not_cross_serve(self):
        cache: dict = {}
        coarse = evaluate_graph(
            K4SUB, 2.5, A1, n_points=16, dense_engine="torus",
            sp_n_points=64, block_cache=cache, richardson=False,
        )
        fine = evaluate_graph(
            K4SUB, 2.5, A1, n_points=16, dense_engine="torus",
            sp_n_points=512, block_cache=cache, richardson=False,
        )
        assert coarse != fine

    def test_a_repeat_call_is_actually_served_from_the_cache(self):
        # Without this the tests above would pass on an inert cache.
        cache: dict = {}
        evaluate_graph(K4SUB, 2.5, A1, n_points=16, block_cache=cache)
        _, info = evaluate_graph(
            K4SUB, 2.5, A1, n_points=16, block_cache=cache,
            return_diagnostics=True,
        )
        assert info["n_block_cache_hits"] >= 1


class TestNnModeIsUnaffected:
    r"""nu = inf never takes the split.

    hybrid refuses ``sp_n_points`` at nu = inf outright (the kernel is a
    nearest-neighbour indicator, so an FFT collapse returns 2.0 as
    1.9999999999999998 and there is no power-law tail to truncate
    anyway).  The resolver suppresses it upstream rather than letting
    that ValueError escape from inside a corpus pass.
    """

    def test_infinite_nu_suppresses_the_split(self):
        v, info = evaluate_graph(
            K4, np.inf, A1, dense_engine="torus", sp_n_points=512,
            return_diagnostics=True,
        )
        assert info["nn_mode"] is True
        assert info["sp_n_points"] is None
        assert info["n_block_split"] == 0
        assert np.isfinite(v)


class TestValidation:
    def test_unknown_dense_engine_is_rejected(self):
        with pytest.raises(ValueError, match="dense_engine"):
            evaluate_graph(K4, 2.5, A1, n_points=16, dense_engine="box")

    def test_non_positive_sp_n_points_is_rejected(self):
        with pytest.raises(ValueError, match="sp_n_points"):
            evaluate_graph(
                K4, 2.5, A1, n_points=16, dense_engine="torus",
                sp_n_points=0,
            )


class TestResolver:
    r"""``_resolve_dense_routing`` in isolation.

    Worth testing directly because it is the single place the cache key
    and the route both read: if it were not deterministic in its inputs
    the two could disagree, and nothing downstream would notice.
    """

    def test_none_resolves_from_the_per_d_table(self):
        for d in (1, 2, 3):
            eng, sp, _grade = frontend._resolve_dense_routing(d, None, None, 64, False)
            assert eng == frontend._DENSE_ENGINE_BY_D[d]
            # the split table covers exactly the routed dimensions
            if eng == "torus":
                assert sp == frontend._SPLIT_SP_N_POINTS.get(d)
            else:
                assert sp is None

    def test_a_dimension_absent_from_the_split_table_gets_no_split(self):
        r"""d = 3 used to be the example here and is now in the table, so
        the property is asserted on a dimension that genuinely is not:
        d = 4 routes to the box and must carry no split either way.
        """
        assert 4 not in frontend._SPLIT_SP_N_POINTS
        eng, sp, _grade = frontend._resolve_dense_routing(4, "torus", None, 8, False)
        assert (eng, sp) == ("torus", None)

    def test_unlisted_dimensions_fall_back_to_the_box(self):
        eng, sp, _grade = frontend._resolve_dense_routing(7, None, None, 8, False)
        assert (eng, sp) == ("direct_sum", None)

    def test_explicit_grid_is_clamped_up_to_n_points(self):
        assert frontend._resolve_dense_routing(
            1, "torus", 8, 64, False, False,
        ) == ("torus", None, False)
        assert frontend._resolve_dense_routing(
            1, "torus", 256, 64, False, False,
        ) == ("torus", 256, False)

    def test_nn_mode_suppresses_the_split(self):
        assert frontend._resolve_dense_routing(
            1, "torus", 512, 64, True,
        ) == ("torus", None, False)


class TestMemoryFallbackToTheBox:
    r"""When the torus runs out, the box takes over --- and only then.

    This is the branch that preserves the d = 3 / treewidth->=4
    capability: the box's cost is set by ``L``, not by ``n_points``, so
    it survives grids the torus cannot allocate.  It is unreachable at
    the sizes the suite runs, so it is forced here; an untested
    ``except MemoryError`` is exactly the kind of code that is wrong for
    a year.

    The second test in each pair is the control that keeps the first
    honest: a MemoryError from a block we did NOT divert must still
    propagate, or the handler would be quietly swallowing real
    allocation failures everywhere.
    """

    @staticmethod
    def _boom(*a, **k):
        raise MemoryError("forced")

    def test_vacuum_falls_back_and_is_rebilled(self, monkeypatch):
        monkeypatch.setattr(frontend, "_block_general", self._boom)
        v, info = evaluate_graph(
            K4SUB, 2.5, A1, n_points=16, dense_engine="torus",
            sp_n_points=256, return_diagnostics=True,
        )
        # The counters must be UNWOUND, not merely left alone: the block
        # ran on the box, so billing it as a torus block would misreport
        # the very migration these counters exist to measure.
        assert info["n_block_direct_sum"] == 1
        assert info["n_block_dense_torus"] == 0
        assert info["n_block_split"] == 0
        assert v == evaluate_graph(K4SUB, 2.5, A1, n_points=16)

    def test_vacuum_memory_error_off_the_route_propagates(self, monkeypatch):
        # A tw = 2 block at sigma >= 1.49 reaches _block_general but is
        # NEVER diverted to the torus (only tw >= 3 is), so its
        # MemoryError is a real allocation failure and must surface.
        #
        # The original spelling used K4 at nu = 0.7 <= d and relied on
        # the box refusing the DIVERGENT sum to push the block this far.
        # That path no longer exists -- a divergent lattice sum is
        # refused outright (TestNuBelowDIsRefusedOnEitherEngine) -- and
        # a plain box-route block never reaches _block_general at all,
        # so it cannot serve as the control either: the patch would not
        # fire and the test would pass for the wrong reason.
        theta = np.array([(0, 2), (2, 1), (0, 3), (3, 1), (0, 4), (4, 1)],
                         dtype=int)
        monkeypatch.setattr(frontend, "_block_general", self._boom)
        with pytest.raises(MemoryError):
            evaluate_graph(
                theta, 4.0, A1, n_points=16, dense_engine="torus",
            )

    @pytest.mark.parametrize("momentum", [np.array([0.25]), None])
    def test_finite_k_falls_back_and_is_rebilled(self, monkeypatch, momentum):
        monkeypatch.setattr(frontend, "_block_general", self._boom)
        kw = dict(source=0, terminal=1, n_points=8)
        if momentum is not None:
            kw["momentum"] = momentum
        v, info = evaluate_graph(
            K4SUB, 2.5, A1, dense_engine="torus", sp_n_points=64,
            return_diagnostics=True, **kw,
        )
        assert info["n_block_direct_sum"] == 1
        assert info["n_block_dense_torus"] == 0
        assert info["n_block_tensor"] == 0
        assert np.all(np.isfinite(np.asarray(v)))
        assert np.allclose(
            np.asarray(v), np.asarray(evaluate_graph(K4SUB, 2.5, A1, **kw)),
        )

    def test_finite_k_memory_error_off_the_route_propagates(self, monkeypatch):
        # Same control on the spine path, and via a different mechanism:
        # a treewidth-2 block is never diverted by this flag at all, so
        # it reaches _block_general on its own terms.
        monkeypatch.setattr(frontend, "_block_general", self._boom)
        diamond = np.array(
            [[0, 1], [0, 2], [1, 3], [2, 3]], dtype=int,
        )
        with pytest.raises(MemoryError):
            evaluate_graph(
                diamond, 4.0, A1, source=0, terminal=3, n_points=8,
                dense_engine="torus", sp_n_points=64,
            )


class TestSplitLadderBasis:
    r"""With the split on, the Richardson ladder must use the CORE cut.

    The ladder fits an ``n^-sigma`` tail.  The split does not reduce the
    block-cut mode, it RELOCATES it onto the fine grid -- so with the
    split on there is no ``n^-sigma_blk`` tail left to fit, and the
    guards correctly refuse.  Measured at the shipped config before this
    was fixed: ``not_tail_like`` / ``unstable`` on 4 of 4 blocks where
    the split acts, and the resulting value WORSE than not splitting at
    all.

    The two are complements, not substitutes::

        err(n, sp)  ~  A_blk * sp^-sigma_blk  +  A_core * n^-sigma_core

    the split shrinks the first term, the ladder removes the second.
    """

    @staticmethod
    def _cuts(edges, nu, d):
        from gzl._elimination import (
            min_free_cut_nu, sp_reduced_cut_nu,
        )
        emap = {}
        for u, v in edges:
            k = (min(int(u), int(v)), max(int(u), int(v)))
            emap[k] = emap.get(k, 0.0) + float(nu)
        return (min_free_cut_nu(emap, 0, ()) - d,
                sp_reduced_cut_nu(emap, 0, ()) - d)

    def test_the_core_cut_exceeds_the_block_cut_on_an_acting_block(self):
        # K4SUB's degree-2 vertex is the cheap escape (2 nu); the K4 core
        # underneath has edge connectivity 3, hence >= 3 nu.
        blk, core = self._cuts(K4SUB, 2.5, 1)
        assert blk == pytest.approx(2 * 2.5 - 1)
        assert core == pytest.approx(3 * 2.5 - 1)
        assert core > blk

    def test_the_two_cuts_agree_when_nothing_reduces(self):
        # The control.  K4 is 3-connected, so SP reduction removes
        # nothing and the basis switch must be an exact no-op.
        blk, core = self._cuts(K4, 2.5, 1)
        assert blk == core

    @pytest.mark.parametrize("edges", [
        np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=int),   # 4-cycle
        np.array([[0, 1], [1, 2], [0, 2]], dtype=int),           # 3-cycle
        np.array([[0, 1]], dtype=int),                           # bridge
        np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=int),   # path
    ])
    def test_fully_reducible_shapes_fall_back_to_the_block_cut(self, edges):
        # These have no irreducible core at all.  Reducing them to
        # nothing must not produce a nonsense exponent -- and the
        # degree recheck matters here: suppressing one degree-2 vertex
        # drops its neighbour to degree 1, which a stale candidate list
        # would then try to suppress as if it still had two edges.
        blk, core = self._cuts(edges, 2.5, 1)
        assert core == blk

    def test_the_ladder_is_accepted_on_acting_blocks_with_the_split_on(self):
        # The defect this class exists for: before the basis switch the
        # guards refused here, so the ladder silently contributed
        # nothing exactly where the split had made it necessary.
        from gzl.frontend import _richardson_ladder
        from gzl.hybrid import hybrid_zeta
        blk, core = self._cuts(K4SUB, 2.5, 1)
        nu_vec = np.full(len(K4SUB), 2.5)

        def f(m):
            return float(np.real(hybrid_zeta(K4SUB, nu_vec, A1, m,
                                             sp_n_points=max(1024, m))))

        _, why_blk = _richardson_ladder(f, 64, 1, blk, f(64))
        v_core, why_core = _richardson_ladder(f, 64, 1, core, f(64))
        assert why_blk != "ok", (
            "block basis unexpectedly accepted -- if the guards stopped "
            "refusing, this test no longer demonstrates anything"
        )
        assert why_core == "ok"
        assert v_core is not None

    def test_splitting_beats_not_splitting_at_the_shipped_size(self):
        # End-to-end, and scored against the BOX family, which shares no
        # machinery with the torus and so cannot flatter it.  This is the
        # property the shipped sp value has to have and did not: at
        # sp = 1024 the split was worse than no split.
        from gzl.direct_sum import direct_sum_extrapolated
        from gzl import frontend
        nu_vec = np.full(len(K4SUB), 2.5)
        ref = float(np.real(direct_sum_extrapolated(
            K4SUB, nu_vec, A1, L_list=(40, 42, 44, 46),
            n_correction_terms=3)))

        def run(sp):
            orig = dict(frontend._SPLIT_SP_N_POINTS)
            frontend._SPLIT_SP_N_POINTS = {} if sp is None else {1: sp}
            try:
                v = evaluate_graph(K4SUB, 2.5, A1, n_points=64)
            finally:
                frontend._SPLIT_SP_N_POINTS = orig
            return abs(v - ref) / abs(ref)

        assert run(frontend._SPLIT_SP_N_POINTS[1]) < run(None)


class TestCoreGrading:
    r"""The connectivity-graded core: ``n_core = κ·n^(σ_blk/σ_core)``.

    The rule is applied per block from rates the router already has, so
    what needs gating is not the arithmetic but the SCOPE: it must act
    only where there is a core to decouple, and must leave everything
    else — including the default path — untouched.
    """

    def test_on_by_default(self):
        r"""Enabled by default: the core converges a full nu faster than
        its block, so at a shared grid it is over-resolved, and it is the
        half that costs n^(tau*d).  Explicitly disable-able, and it is
        still suppressed in nn_mode and for any d without a measured
        kappa -- but NOT off the torus route: a tw ≤ 2 block runs hybrid
        whatever the dense engine is, so its grading cannot depend on it.
        """
        assert frontend._resolve_dense_routing(
            2, None, None, 16, False,
        )[2] is True
        assert frontend._resolve_dense_routing(
            2, None, None, 16, False, False,
        )[2] is False

    def test_engine_independent_but_suppressed_in_nn_mode(self):
        r"""Grading is offered to every block that reaches hybrid, so the
        dense-engine choice does not switch it off: a tw ≤ 2 block runs
        hybrid whatever the dense engine is, and the dense arms add their
        own ``dense_on_torus`` gate on top.  nn_mode still suppresses it
        -- at ν = inf a coarser core changes which walks exist.
        """
        assert frontend._resolve_dense_routing(
            2, "direct_sum", None, 16, False, True,
        )[2] is True
        assert frontend._resolve_dense_routing(
            2, "torus", None, 16, True, True,
        )[2] is False

    def test_uncalibrated_dimensions_are_left_alone(self):
        r"""A dimension absent from ``_CORE_KAPPA`` means "not calibrated
        here", not "grade with some default" — the same convention a
        missing ``_SPLIT_SP_N_POINTS`` entry follows.

        d = 1/2/3 are all calibrated now; d >= 4 is not, and is the live
        case this pins.  It is checked on the explicit ``"torus"`` flag
        because the DEFAULT engine at d >= 4 is the box, so a default
        call would return ``False`` for the wrong reason.
        """
        assert 4 not in frontend._CORE_KAPPA
        assert frontend._resolve_dense_routing(
            4, "torus", None, 16, False, True,
        )[2] is False
        for calibrated in (2, 3):
            assert frontend._resolve_dense_routing(
                calibrated, "torus", None, 16, False, True,
            )[2] is True

    def test_a_block_with_nothing_to_reduce_is_returned_untouched(self):
        r"""σ_core == σ_blk by construction when SP reduction removes
        nothing, so the exponent is 1 and the rule must be the identity.
        This is what makes it safe to switch on corpus-wide: it cannot
        coarsen a block that has no core to coarsen.
        """
        for n in (16, 24, 48, 64):
            assert frontend._core_n_for_block(n, 3.0, 3.0, 2) == n

    def test_the_core_shrinks_and_the_saving_grows_with_n_points(self):
        r"""The defining property: the exponent σ_blk/σ_core is < 1
        exactly when the split has something to do, so the RATIO of core
        to delivery grid must fall as the delivery grid rises.
        """
        ns = (16, 24, 32, 48, 64)
        cores = [frontend._core_n_for_block(n, 3.0, 8.0, 2) for n in ns]
        assert all(c <= n for c, n in zip(cores, ns))
        ratios = [c / n for c, n in zip(cores, ns)]
        assert all(b < a for a, b in zip(ratios, ratios[1:]))

    def test_the_core_never_goes_below_the_pre_asymptotic_floor(self):
        r"""Below ``_CORE_N_FLOOR`` the torus is pre-asymptotic and the
        power law the rule is derived from does not hold, so the rule
        must clip rather than extrapolate off the end of its own model.
        """
        for n in (16, 24, 48):
            assert frontend._core_n_for_block(
                n, 1.0, 40.0, 2) >= frontend._CORE_N_FLOOR

    def test_grading_is_on_the_cache_key(self):
        r"""Two calls whose resolved grading differs must not share block
        cache entries — they compute the same block on different grids.
        """
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (4, 3)]
        nu = [2.5] * len(edges)
        A = np.eye(2)
        cache: dict = {}
        # n_points = 32: this block's core is graded to 20 there.  At 16
        # the minimum-shrink gate declines it (14 of 16 is not worth the
        # ladder it would cost), which is TestTheMinimumShrinkGate's job.
        a = frontend.evaluate_graph(edges, nu, A, n_points=32,
                                    core_grading=False, block_cache=cache)
        n_after_off = len(cache)
        b = frontend.evaluate_graph(edges, nu, A, n_points=32,
                                    core_grading=True, block_cache=cache)
        assert len(cache) > n_after_off
        assert a != b            # different grids, so different numbers

    def test_a_no_op_block_is_bit_identical_with_grading_on(self):
        r"""The liveness control.  A 3-connected block has no degree-2
        vertex to suppress, so grading is an exact no-op there — and if
        this ever stopped being bitwise it would mean the flag had
        started perturbing blocks it has no business touching.
        """
        K4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        nu = [2.5] * len(K4)
        for A in (np.eye(1), np.eye(2)):
            off = frontend.evaluate_graph(K4, nu, A, n_points=16)
            on = frontend.evaluate_graph(K4, nu, A, n_points=16,
                                         core_grading=True)
            assert off == on

    @pytest.mark.parametrize("momentum", [None, (0.25, 0.125)])
    def test_finite_k_grades_and_stays_close(self, momentum):
        r"""The finite-k arm has NO Richardson ladder, so the graded core
        plus the split is the whole accuracy story there.  Gated as an
        agreement bound against the ungraded value rather than as a
        ratio: the point is that grading trades surplus accuracy, not
        that it is free.
        """
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (4, 3)]
        nu = [2.5] * len(edges)
        A, n = np.eye(2), 32
        # momentum=None with a terminal is the full BZ grid; an explicit
        # k is the single-point branch, which reads the trigonometric
        # polynomial straight off the coarse array with no embedding.
        kw = ({} if momentum is None
              else {"momentum": np.asarray(momentum, dtype=float)})
        off = np.asarray(frontend.evaluate_graph(
            edges, nu, A, n_points=n, terminal=1, **kw))
        on, info = frontend.evaluate_graph(
            edges, nu, A, n_points=n, terminal=1, core_grading=True,
            return_diagnostics=True, **kw)
        on = np.asarray(on)
        assert info["n_block_core_graded"] >= 1
        assert on.shape == off.shape
        assert np.allclose(on, off, rtol=2e-3, atol=0.0)

    def test_the_ladder_is_gated_on_the_sp_floor_not_skipped_wholesale(self):
        r"""The ladder removes ``A_core*n^-sigma_core`` and nothing else.

        Measured directly, the SP residual is a CONSTANT offset across
        any ladder in ``n_core``, so no Richardson fit can remove it.
        Once grading pushes the core term below that floor,
        extrapolating removes nothing; while it is still above, the
        ladder is worth 13-15x and must keep running.  So this is a gate
        on two ANALYTIC quantities, not a blanket skip.
        """
        sp, s_blk = 512, 3.0
        # core term still above the floor by more than the margin -> run
        assert not frontend._core_term_below_sp_floor(20, 5.5, sp, s_blk)
        # fallen below, or within the margin of it -> skip
        assert frontend._core_term_below_sp_floor(12, 8.0, sp, s_blk)
        assert frontend._core_term_below_sp_floor(10, 8.0, sp, s_blk)
        # no split means no SP floor, so the ladder is untouched
        assert not frontend._core_term_below_sp_floor(12, 8.0, None, s_blk)

    def test_a_graded_block_above_the_floor_keeps_its_ladder(self):
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (4, 3)]
        nu = [2.5] * len(edges)
        _, info = frontend.evaluate_graph(
            edges, nu, np.eye(2), n_points=32, richardson=True,
            core_grading=True, return_diagnostics=True)
        assert info["n_block_core_graded"] == 1        # core was coarsened
        assert info["n_block_richardson_theory"] == 1  # ...and still fitted
        assert info["n_block_richardson_skipped_graded"] == 0


class TestTheCoreIsSizedAgainstThePassFloor:
    r"""The numerator is the PASS's rate, not the block's own.

    Sizing against ``sigma_blk`` is circular — it targets the error this
    block would have at the delivery grid, so it can never beat the
    delivery grid, and on a 3-connected block the two cuts coincide by
    construction so the exponent is exactly 1 and the rule self-disables.
    That is on precisely the blocks that cost the most: at d = 2 the one
    block carrying 66% of an order-9 pass was 3-connected.

    ``_pass_floor_sigma`` is ``2 nu - d``, MEASURED: over every block
    reaching an n_points-dependent engine on the d = 2 1qp corpus, cut/nu
    = 2 carries 85.0% of distinct blocks and 83.5% of occurrences, and
    cut/nu = 3 never occurs as a block minimum there at all.
    """

    def test_the_pass_floor_is_the_degree_two_escape(self):
        assert frontend._pass_floor_sigma(2.5, 2) == pytest.approx(3.0)
        assert frontend._pass_floor_sigma(1.5, 1) == pytest.approx(2.0)
        # An array coupling takes the MINIMUM: the floor is the slowest edge.
        assert frontend._pass_floor_sigma(np.array([4.0, 2.5]), 2) == \
            pytest.approx(3.0)
        # No grid block left, no floor.  Which edges the array holds is
        # the router's choice, pinned in TestTheCoreFloorIsTheBlocksOwn.
        assert frontend._pass_floor_sigma([], 2) is None

    def test_a_three_connected_block_is_now_graded(self):
        r"""sigma_blk == sigma_core == 3nu - d = 5.5, and the pass floor
        is 2nu - d = 3.0.  The old numerator gave exponent 1 (no grading);
        the pass floor gives 3.0/5.5 and a real reduction.
        """
        for n in (24, 32, 48, 64):
            legacy = frontend._core_n_for_block(n, 5.5, 5.5, 2)
            rated = frontend._core_n_for_block(n, 5.5, 5.5, 2, sigma_ref=3.0)
            assert legacy == n, "legacy must self-disable here (that is the bug)"
            assert rated < n, f"pass-floor rule must grade at n_points={n}"

    def test_a_core_no_faster_than_the_floor_is_left_alone(self):
        """No gap to spend => identity, whatever the block's own rate."""
        for n in (16, 32, 64):
            assert frontend._core_n_for_block(n, 9.0, 3.0, 2, sigma_ref=3.0) == n
            assert frontend._core_n_for_block(n, 9.0, 2.0, 2, sigma_ref=3.0) == n

    def test_the_saving_still_grows_with_n_points(self):
        ns = (24, 32, 48, 64)
        cores = [frontend._core_n_for_block(n, 5.5, 5.5, 2, sigma_ref=3.0)
                 for n in ns]
        ratios = [c / n for c, n in zip(cores, ns)]
        assert all(b < a for a, b in zip(ratios, ratios[1:]))

    def test_the_floor_still_clips(self):
        for n in (16, 24, 48):
            assert frontend._core_n_for_block(
                n, 1.0, 40.0, 2, sigma_ref=3.0) >= frontend._CORE_N_FLOOR


class TestTheCoreFloorIsTheBlocksOwn:
    r"""A dense core reads its own floor, whatever sits beside it.

    ``_pass_floor_sigma`` was read over the whole graph, so with per-edge
    nu a low-exponent neighbour pulled the floor down and a dense block
    beside it was graded to a core sized for an error nothing in the
    graph carries.  Measured before the fix, K5 at nu = 2.5 on the chain,
    n_points = 32, the K5 factor (the value divided by the bridge's
    closed form):

        bridge nu        K5 factor                  against K5 alone
        (none)           8.914606862887881e-05
        3.5, 2.5         8.914606862887881e-05      0
        1.2, 0.9, 0.6    9.894414128070352e-05      1.10e-01  (core 8)
        0.4, 0.2         8.914605002823246e-05      grading switched off

    A pendant triangle at nu = 1.2 (a closed-form cycle) did the same,
    and with the K5 on the spine a bridge at 1.2, on the spine or off
    it, moved the factor 9.5e-02 at k = 0.25 and 3.9e-01 across the BZ
    grid.  Leaving the closed forms out of the floor was not enough: a
    grid block, a pendant diamond at nu = 1.2, still moved it 1.10e-01
    while its own error at n_points = 32 is about 6.5e-04.  Each block
    now reads its own per-edge exponents.  K5 and K4 look the same from
    every vertex, so where they attach does not matter and their factor
    is their standalone value.  K5 alone at n_points = 64 is
    8.914604836886712e-05.

    ``rel=1e-12`` separates K5's own graded value from both failure
    modes (1.1e-01, and 2.1e-07 for the ungraded core) and leaves room
    for the round-off of one division.
    """

    N = 32

    def _alone(self, **kw):
        return evaluate_graph(K5, 2.5, A1, n_points=self.N, **kw)

    def test_the_k5_core_is_graded_here(self):
        # Liveness: without grading the tests below would compare two
        # ungraded values and pass whatever the floor read.
        graded, info = self._alone(return_diagnostics=True)
        ungraded = self._alone(core_grading=False)
        assert info["n_block_core_graded"] == 1
        assert graded != pytest.approx(ungraded, rel=1e-12)

    @pytest.mark.parametrize("nu_b", [3.5, 2.5, 1.2, 0.9, 0.6, 0.4])
    def test_a_pendant_bridge_leaves_the_core_alone(self, nu_b):
        value = evaluate_graph(np.vstack([K5, [[4, 5]]]),
                               np.array([2.5] * len(K5) + [nu_b]), A1,
                               n_points=self.N)
        bridge = evaluate_graph(np.array([[0, 1]]), nu_b, A1,
                                n_points=self.N)
        assert value / bridge == pytest.approx(self._alone(), rel=1e-12)

    def test_a_closed_form_cycle_leaves_the_core_alone(self):
        tri = np.array([[4, 5], [5, 6], [4, 6]])
        value, info = evaluate_graph(
            np.vstack([K5, tri]), np.array([2.5] * len(K5) + [1.2] * 3),
            A1, n_points=self.N, return_diagnostics=True,
        )
        assert info["n_simple_cycles"] == 1      # the closed form ran
        cycle = evaluate_graph(tri - 4, 1.2, A1, n_points=self.N)
        assert value / cycle == pytest.approx(self._alone(), rel=1e-12)

    def test_an_interaction_bridge_leaves_the_core_alone(self):
        r"""The same on the kernel path, where the floor reads tail
        exponents.  Measured before the fix: K5 with two-term power-law
        kernels (nu 2.5 and 3.5) beside a bridge whose tail is 1.2 was
        9.35e-02 off.
        """
        kern = Interaction(b=(1.0, 0.3), nu=(2.5, 3.5))
        bridge = Interaction(b=(1.0, 0.2), nu=(1.2, 2.0))
        alone, info = evaluate_graph(K5, [kern] * len(K5), A1,
                                     n_points=self.N,
                                     return_diagnostics=True)
        assert info["interaction_mode"]          # not demoted to a float
        assert info["n_block_core_graded"] == 1
        value = evaluate_graph(np.vstack([K5, [[4, 5]]]),
                               [kern] * len(K5) + [bridge], A1,
                               n_points=self.N)
        assert value / float(bridge.lattice_sum(A1)) == \
            pytest.approx(alone, rel=1e-12)

    @pytest.mark.parametrize("momentum", [np.array([0.25]), None],
                             ids=["k", "grid"])
    @pytest.mark.parametrize("source", [0, 5],
                             ids=["bridge_off_spine", "bridge_on_spine"])
    def test_a_bridge_leaves_the_core_alone_at_finite_k(self, source,
                                                        momentum):
        # source 0: the spine is K5 alone (0 -> 1) and the bridge hangs
        # off it.  source 5: the spine is the bridge (5 -> 4), then K5
        # (4 -> 1).
        value = evaluate_graph(np.vstack([K5, [[4, 5]]]),
                               np.array([2.5] * len(K5) + [1.2]), A1,
                               source=source, terminal=1, momentum=momentum,
                               n_points=self.N)
        if source == 0:
            bridge = evaluate_graph(np.array([[0, 1]]), 1.2, A1,
                                    n_points=self.N)
            alone = self._alone(source=0, terminal=1, momentum=momentum)
        else:
            bridge = evaluate_graph(np.array([[0, 1]]), 1.2, A1, source=0,
                                    terminal=1, momentum=momentum,
                                    n_points=self.N)
            alone = self._alone(source=4, terminal=1, momentum=momentum)
        np.testing.assert_allclose(np.asarray(value) / np.asarray(bridge),
                                   alone, rtol=1e-12)

    @pytest.mark.parametrize("extra, kw, alone_kw, extra_kw", [
        ([[4, 5], [4, 6], [5, 6], [5, 7], [6, 7]], {}, {}, {}),
        ([[4, 5], [5, 6], [4, 6]], {"fast_cycles": True}, {},
         {"fast_cycles": True}),
        ([[4, 5], [5, 6], [4, 6]],
         {"source": 5, "terminal": 1, "momentum": np.array([0.25])},
         {"source": 4, "terminal": 1, "momentum": np.array([0.25])},
         {"source": 1, "terminal": 0, "momentum": np.array([0.25])}),
    ], ids=["diamond", "fast_cycles_triangle", "triangle_on_spine"])
    def test_a_grid_block_leaves_the_core_alone(self, extra, kw, alone_kw,
                                                extra_kw):
        r"""Grid blocks at nu = 1.2, which reach an ``n_points``-dependent
        engine: a diamond (K4 minus an edge), a triangle sent to the sigma
        router by ``fast_cycles``, and a triangle on the spine at finite
        k.  Each moved the K5 factor 1.10e-01 while the floor was read
        over the whole graph.  The neighbour's factor is its own
        evaluation, with the same local labels and endpoints.
        """
        extra = np.array(extra)
        value = evaluate_graph(
            np.vstack([K5, extra]),
            np.array([2.5] * len(K5) + [1.2] * len(extra)), A1,
            n_points=self.N, **kw)
        neighbour = evaluate_graph(extra - 4, 1.2, A1, n_points=self.N,
                                   **extra_kw)
        assert value / neighbour == pytest.approx(self._alone(**alone_kw),
                                                  rel=1e-12)

    def test_the_core_does_not_read_its_neighbours_exponents(self):
        r"""V6E11 on the square lattice, n_points = 32, with a diamond on
        its vertex 5.  V6E11 does not look the same from every vertex, so
        attached it is pinned elsewhere than alone and its factor is not
        its standalone value; what must hold is that the diamond's
        exponent does not reach it.  With the floor read over the whole
        graph the diamond at 2.2 graded it to 10 instead of 12 and moved
        its factor 4.0e-06 against the diamond at 2.5.
        """
        v6e11 = np.array([[0, 1], [0, 4], [1, 2], [1, 3], [1, 5], [2, 3],
                          [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]])
        diamond = np.array([[5, 6], [5, 7], [6, 7], [6, 8], [7, 8]])
        factors = []
        for nu_x in (2.5, 2.2):
            value = evaluate_graph(
                np.vstack([v6e11, diamond]),
                np.array([2.5] * len(v6e11) + [nu_x] * len(diamond)), A2,
                n_points=32)
            factors.append(value / evaluate_graph(diamond - 5, nu_x, A2,
                                                  n_points=32))
        assert factors[1] == pytest.approx(factors[0], rel=1e-12)

    def test_a_grid_block_leaves_the_core_alone_at_d2(self):
        r"""K4 at nu = 3.0 on the square lattice, n_points = 32, beside a
        diamond at 2.2.  Here the whole-graph floor stayed inside the
        diamond's own error (it graded the K4 to 10 instead of 22, a
        shift of 1.9e-05 against the diamond's 2.8e-04), and the K4
        factor is now its standalone value.
        """
        diamond = np.array([[3, 4], [3, 5], [4, 5], [4, 6], [5, 6]])
        alone, info = evaluate_graph(K4, 3.0, A2, n_points=32,
                                     return_diagnostics=True)
        assert info["n_block_core_graded"] == 1   # grading is live here
        value = evaluate_graph(
            np.vstack([K4, diamond]),
            np.array([3.0] * len(K4) + [2.2] * len(diamond)), A2,
            n_points=32)
        neighbour = evaluate_graph(diamond - 3, 2.2, A2, n_points=32)
        assert value / neighbour == pytest.approx(alone, rel=1e-12)


class TestTheDenseCoreByteCeiling:
    r"""Cost must be bounded by construction, not by hoping the rate rule
    was generous.

    The accuracy rule can legitimately DECLINE (returning ``n_points``),
    and a declined exponent-3 block at d = 2, n_points = 32 allocates
    17.2 GB — one such block took 78% of an order-11 pass.  The exponent
    grows with treewidth and with d, so an uncapped core is a memory wall
    waiting for the first tw = 5 shape or the first d = 3 run.
    """

    def test_the_cap_is_monotone_in_exponent_and_dimension(self):
        for d in (1, 2, 3):
            caps = [frontend._core_n_for_budget(p, d) for p in (2, 3, 4)]
            assert all(b <= a for a, b in zip(caps, caps[1:])), (d, caps)
        for p in (2, 3):
            caps = [frontend._core_n_for_budget(p, d) for d in (1, 2, 3)]
            assert all(b <= a for a, b in zip(caps, caps[1:])), (p, caps)

    @pytest.mark.parametrize("d,p", [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3)])
    def test_the_capped_contraction_fits_the_budget(self, d, p):
        n = frontend._core_n_for_budget(p, d)
        assert n >= 2
        assert (n ** d) ** p * 16 <= frontend._CORE_MAX_BYTES
        # ...and it is the LARGEST even n that does, i.e. not needlessly small.
        assert ((n + 2) ** d) ** p * 16 > frontend._CORE_MAX_BYTES

    def test_no_configuration_exceeds_the_budget(self):
        """The property that matters: bounded for every (d, exponent)."""
        for d in (1, 2, 3, 4):
            for p in (1, 2, 3, 4, 5):
                n = frontend._core_n_for_budget(p, d)
                assert (n ** d) ** p * 16 <= frontend._CORE_MAX_BYTES, (d, p, n)

    def test_it_never_raises_the_accuracy_choice(self):
        """A budget is not a licence to refine."""
        E = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        V = [0, 1, 2, 3]
        for n_rate in (8, 10, 12, 16):
            assert frontend._core_n_capped(n_rate, E, V, 0, (), 2) <= n_rate

    def test_it_lowers_a_declined_block(self):
        """The case the rate rule cannot help with."""
        # V6E11, treewidth 4, exponent 3 at every pin.
        E = np.array([(0, 1), (0, 4), (1, 2), (1, 3), (1, 5), (2, 3),
                      (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)])
        V = list(range(6))
        assert frontend._core_n_capped(32, E, V, 1, (), 2) < 32

    def test_an_unknown_exponent_is_not_a_constraint(self):
        assert frontend._core_n_for_budget(None, 2) is None
        assert frontend._core_n_for_budget(0, 2) is None
        # A planner refusal must leave the accuracy choice alone.
        assert frontend._core_n_capped(24, np.zeros((0, 2), dtype=int),
                                       [], 0, (), 2) == 24


class TestTheMinimumShrinkGate:
    r"""``_CORE_MIN_SHRINK`` declines a core that would barely shrink.

    Grading is not free: a coarser core costs the block its place on the
    ``n_points`` Richardson ladder, so a core that comes out at 14 of 16
    buys a 1.3x saving for a rung.  The gate is what refuses that trade,
    and it had no direct test -- another test's comment referred to this
    class by name while it did not exist.
    """

    def test_the_gate_declines_a_marginal_shrink(self):
        """At n = 16 the rule asks for 14, which is > 0.75 * 16."""
        assert frontend._core_n_for_block(16, 3.0, 5.5, 2) == 16

    def test_the_gate_admits_a_worthwhile_shrink(self):
        """At n = 32 the rule asks for 20, which is <= 0.75 * 32."""
        assert frontend._core_n_for_block(32, 3.0, 5.5, 2) == 20

    def test_the_gate_is_what_declines_it(self, monkeypatch):
        """Relaxing only the gate must let the n = 16 core through.

        Without this the first test above is satisfied by any code path
        that returns ``n_points`` -- including the rule never firing.
        """
        monkeypatch.setattr(frontend, "_CORE_MIN_SHRINK", 1.0)
        assert frontend._core_n_for_block(16, 3.0, 5.5, 2) == 14

    def test_the_gate_never_raises_the_core(self):
        """Whatever it returns is at most ``n_points``, for every rung."""
        for n in (12, 16, 24, 32, 48, 64):
            assert frontend._core_n_for_block(n, 3.0, 5.5, 2) <= n
