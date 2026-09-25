# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""The accuracy gate beside the byte-budget gate on the torus route.

``hybrid``'s ``max_core_bytes`` decides whether a dense block FITS on
the torus.  It has never decided whether the block is BETTER there, and
the two questions have different answers: measured at d = 3 against a
slab reference, the two exponent-3 cores a byte-budget change moves are
5.0x (K5) and 10.6x (V6E11) WORSE on an ``n = 8`` torus than in the
real-space box they displace.  ``frontend._TORUS_MIN_N_BY_CLASS`` is the
second question, and this module gates the mechanism.

Three properties carry the change and each has a class below.

* **Monotone toward the incumbent.**  A class with no table entry is
  unconstrained, and a gated class goes to the box -- which is the
  engine it used before any budget change.  So the gate can restore
  today's routing and cannot invent a third behaviour.
* **Keyed on the plan exponent, never the treewidth.**  A census of
  3684 random dense shapes splits 333 as (tw 3, exponent 2), 21 as
  (tw 4, exponent 3) and 7 as (tw 3, exponent 3).  A treewidth-keyed
  rule prices the seven as if they were the 333.
* **The ladder is part of the gate.**  The measured ``n = 12`` entry
  clears the box on V6E11 only with the k = 0 Richardson ladder
  running; its raw value there is still 2.3x worse.  A caller with
  ``richardson=False``, and every finite-k caller, does not inherit it.

The d = 1 K4 used throughout is a stand-in chosen for cost, not for
physics: the table is monkeypatched onto its class so the mechanism can
be exercised in milliseconds.  The one test that runs the SHIPPED entry
end to end at d = 3 is marked ``slow`` -- the box arm it falls back to
costs about 80 s per block.
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl import evaluate_graph, frontend
from gzl.direct_sum import direct_sum_extrapolated

from tests._env_gate import assert_pinned


# K4: the smallest treewidth-3 block.  Post-peel plan exponent 2 with a
# free pin, and still 2 with one terminal kept -- which is itself the
# point of TestTheKeyIsTheCostClass below.
K4 = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=int)
NU_K4 = np.full(len(K4), 1.5)
A1 = np.eye(1)


def _route(**kw):
    """``(value, non-zero n_block_* counters)`` for the d = 1 K4."""
    v, info = evaluate_graph(K4, NU_K4, A1, return_diagnostics=True, **kw)
    return float(np.asarray(v).reshape(-1)[0]), {
        k: int(x) for k, x in info.items()
        if isinstance(x, (int, np.integer)) and k.startswith("n_block") and x
    }


def _box_k0():
    return float(np.real(direct_sum_extrapolated(
        K4, NU_K4, A1, L_list=frontend._DENSE_DSUM_L_LIST[1],
        n_correction_terms=3)))


class TestTheHelperItself:
    r"""``_dense_torus_is_accurate`` in isolation.

    Kept separate from the routing tests because the helper is also the
    thing a future re-measurement edits, and a table change should fail
    here first -- with a readable assertion -- rather than as a routing
    surprise three layers up.
    """

    def test_a_class_with_no_entry_is_unconstrained(self):
        assert frontend._dense_torus_is_accurate(
            K4, list(range(4)), 0, (), 1, 2, ladder=True)
        assert frontend._dense_torus_is_accurate(
            K4, list(range(4)), 0, (), 1, 2, ladder=False)

    def test_the_entry_binds_below_its_grid_and_releases_above(
            self, monkeypatch):
        monkeypatch.setattr(frontend, "_TORUS_MIN_N_BY_CLASS", {(1, 2): 12})
        for n in (2, 4, 8, 10, 11):
            assert not frontend._dense_torus_is_accurate(
                K4, list(range(4)), 0, (), 1, n, ladder=True)
        for n in (12, 13, 16, 64):
            assert frontend._dense_torus_is_accurate(
                K4, list(range(4)), 0, (), 1, n, ladder=True)

    def test_the_grid_alone_is_not_enough_without_the_ladder(
            self, monkeypatch):
        r"""The threshold is earned partly BY the ladder, so it does not
        transfer to a caller that has none.  V6E11 at the shipped
        threshold n = 12 is 2.3x worse than the box on its raw value and
        8.3x better with the accepted ladder; only the second of those
        justifies the entry.
        """
        monkeypatch.setattr(frontend, "_TORUS_MIN_N_BY_CLASS", {(1, 2): 12})
        assert frontend._dense_torus_is_accurate(
            K4, list(range(4)), 0, (), 1, 16, ladder=True)
        assert not frontend._dense_torus_is_accurate(
            K4, list(range(4)), 0, (), 1, 16, ladder=False)

    def test_an_unclassifiable_block_is_admitted(self, monkeypatch):
        r"""``plan`` can refuse (budget, or a shape it cannot order).
        With no cost class there is no measurement to refuse ON, so the
        block keeps today's routing.  Failing CLOSED here would move
        blocks to the box on an unknown-probability planner failure --
        a regression risk taken for no measured gain.
        """
        def boom(*a, **k):
            raise RuntimeError("planner refused")
        monkeypatch.setattr(frontend._elimination, "plan", boom)
        monkeypatch.setattr(frontend, "_TORUS_MIN_N_BY_CLASS", {(1, 2): 999})
        assert frontend._dense_torus_is_accurate(
            K4, list(range(4)), 0, (), 1, 2, ladder=True)

    def test_the_cost_class_is_not_the_treewidth(self, monkeypatch):
        r"""The shapes make the case on their own: K4, the prism and K33
        are treewidth 3 and exponent 2, while K5 and V6E11 are treewidth
        4 and exponent 3.  A rule keyed on treewidth 3 would gate the
        first three and miss the last two; the shipped ``(3, 3)`` entry
        does the opposite, which is what the measurement asked for.
        """
        K5 = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        PRISM = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
                 (0, 3), (1, 4), (2, 5)]
        plan = frontend._elimination.plan
        assert plan([tuple(e) for e in K4.tolist()], list(range(4)),
                    pin=0, keep=()).exponent == 2
        assert plan(PRISM, list(range(6)), pin=0, keep=()).exponent == 2
        assert plan(K5, list(range(5)), pin=0, keep=()).exponent == 3
        # ...and the table, keyed on the exponent, separates them.
        monkeypatch.setattr(frontend, "_TORUS_MIN_N_BY_CLASS", {(1, 3): 999})
        assert frontend._dense_torus_is_accurate(
            K4, list(range(4)), 0, (), 1, 8, ladder=True)
        assert not frontend._dense_torus_is_accurate(
            np.array(K5), list(range(5)), 0, (), 1, 8, ladder=True)

    def test_the_pin_and_kept_set_are_part_of_the_class(self):
        r"""The exponent is a property of (graph, pin, kept set), so the
        gate must be handed the ones the core is actually contracted at
        -- the same pin desync ``_core_n_capped`` is careful about.  The
        lookup is exercised here rather than asserted about K4 alone,
        whose exponent happens to be 2 for every kept set.
        """
        assert frontend._dense_torus_min_n(
            K4, list(range(4)), 0, (), 1) is None
        assert frontend._dense_torus_min_n(
            K4, list(range(4)), 0, (1,), 1) is None


class TestTheGateRoutesBackToTheIncumbent:
    r"""End to end at d = 1, where the box arm is cheap.

    The load-bearing assertion is not "it declined" but "the value it
    returned is the BOX's, bit for bit".  A gate that declined and then
    quietly produced a third number would pass a counter check and fail
    the only thing that matters.
    """

    def test_ungated_the_router_is_untouched(self):
        assert frontend._TORUS_MIN_N_BY_CLASS.get((1, 2)) is None
        v, route = _route(n_points=16, richardson=True)
        assert route.get("n_block_dense_torus") == 1
        assert route.get("n_block_dense_torus_declined", 0) == 0

    def test_a_gated_block_takes_the_box_and_reports_it(self, monkeypatch):
        before, _ = _route(n_points=16, richardson=True)
        monkeypatch.setattr(frontend, "_TORUS_MIN_N_BY_CLASS", {(1, 2): 64})
        after, route = _route(n_points=16, richardson=True)
        assert route.get("n_block_dense_torus_declined") == 1
        assert route.get("n_block_direct_sum") == 1
        assert route.get("n_block_dense_torus", 0) == 0
        assert after == _box_k0()
        assert after != before          # the gate actually changed the route

    def test_the_gate_releases_at_its_own_threshold(self, monkeypatch):
        monkeypatch.setattr(frontend, "_TORUS_MIN_N_BY_CLASS", {(1, 2): 12})
        v16, r16 = _route(n_points=16, richardson=True)
        v8, r8 = _route(n_points=8, richardson=True)
        assert r16.get("n_block_dense_torus") == 1
        assert r8.get("n_block_dense_torus_declined") == 1
        assert v8 == _box_k0()

    def test_richardson_false_does_not_inherit_the_threshold(
            self, monkeypatch):
        monkeypatch.setattr(frontend, "_TORUS_MIN_N_BY_CLASS", {(1, 2): 12})
        _, on = _route(n_points=16, richardson=True)
        v_off, off = _route(n_points=16, richardson=False)
        assert on.get("n_block_dense_torus") == 1
        assert off.get("n_block_dense_torus_declined") == 1
        assert v_off == _box_k0()

    def test_finite_k_keeps_the_box_at_every_grid(self, monkeypatch):
        r"""There is no k != 0 ladder (the tail carries a cos(2 pi k.x)
        factor and stops being a power law), and the table's entry is
        earned partly by the ladder, so a gated class does not get its
        grid back on this path at any ``n``.
        """
        box = float(np.real(direct_sum_extrapolated(
            K4, NU_K4, A1, L_list=frontend._DENSE_DSUM_L_LIST[1],
            n_correction_terms=3, root=0, terminal=1,
            momentum=np.array([0.25]))))
        monkeypatch.setattr(frontend, "_TORUS_MIN_N_BY_CLASS", {(1, 2): 4})
        for n in (8, 16, 64):
            v, info = evaluate_graph(
                K4, NU_K4, A1, source=0, terminal=1, momentum=[0.25],
                n_points=n, return_diagnostics=True)
            assert info["n_block_dense_torus_declined"] == 1
            assert float(v) == box


class TestTheSplitReleasesTheGate:
    r"""An acting split removes the mode the threshold is about.

    The table's grid requirement is a statement about how long the
    block's OWN slow escape takes to die on the coarse grid.  The
    split-resolution SP collapse relocates that escape onto a fine grid,
    so what is left decays at the SP-reduced cut instead and the
    requirement no longer describes the block.  Measured on V6E11 at
    d = 3, nu = 3.5, n_points = 8 -- a grid the table refuses --
    1.44e-03 without the split (10.6x worse than the box, resolved at 737
    reference band-widths) against 3.6e-07 with it, which is 0.18
    band-widths and NOT resolved.  What IS resolved is the bound: from
    10.6x worse than the box to at least 57.7x better -- computed as
    (box_dev - band) / (split_dev + band), the strict form; box_dev/band
    over-states it at 69.8x.

    The release is keyed on the same predicate that makes the split an
    exact no-op elsewhere (block cut < SP-reduced cut), so it cannot
    release a block the split does not help.
    """

    def test_the_escape_hatch_releases_a_gated_class(self, monkeypatch):
        monkeypatch.setattr(frontend, "_TORUS_MIN_N_BY_CLASS", {(1, 2): 999})
        args = (K4, list(range(4)), 0, (), 1, 8)
        assert not frontend._dense_torus_is_accurate(*args, ladder=True)
        assert frontend._dense_torus_is_accurate(
            *args, ladder=True, split_acts=True)
        # ...and with no table entry the hatch is irrelevant either way.
        monkeypatch.setattr(frontend, "_TORUS_MIN_N_BY_CLASS", {})
        assert frontend._dense_torus_is_accurate(*args, ladder=False)

    def test_a_gated_block_the_split_acts_on_reaches_the_torus(
            self, monkeypatch):
        r"""End to end at d = 1, where the box arm is cheap.  K4SUB has a
        degree-2 vertex, so its block cut (2 nu) is below its SP-reduced
        cut (3 nu) and the split acts; K4 is 3-connected and it does not.
        With the class gated, only the first must get through.
        """
        K4SUB = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 4], [4, 3]],
            dtype=int)
        nu_sub = np.full(len(K4SUB), 1.5)
        monkeypatch.setattr(frontend, "_TORUS_MIN_N_BY_CLASS", {(1, 2): 999})
        _, acting = evaluate_graph(
            K4SUB, nu_sub, A1, n_points=16, richardson=True,
            sp_n_points=1024, return_diagnostics=True)
        _, inert = evaluate_graph(
            K4, NU_K4, A1, n_points=16, richardson=True,
            sp_n_points=1024, return_diagnostics=True)
        assert acting["n_block_dense_torus"] == 1
        assert acting["n_block_dense_torus_declined"] == 0
        assert inert["n_block_dense_torus_declined"] == 1
        assert inert["n_block_direct_sum"] == 1

    @pytest.mark.slow
    def test_whether_the_hatch_fires_depends_on_which_vertex_is_external(self):
        r"""The example in the docstring is not a guaranteed release, and
        this pins both directions so nobody reads it as one.

        The cuts are evaluated at the router's root -- ``min(external)``
        where the block has a boundary.  On V6E11 they read (4.0, 11.0)
        at roots 1..5 and (4.0, 4.0) at root 0, because pinning the
        degree-2 vertex suppresses the escape the split would relocate.
        So on a standalone single-block call, where the source IS the
        boundary, ``source=0`` declines and ``source=1`` releases -- at
        d = 3, where the table's one entry lives and the split ships.
        """
        V6E11 = np.array(
            [[0, 1], [0, 4], [1, 2], [1, 3], [1, 5], [2, 3],
             [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]], dtype=int)
        G = np.vstack([V6E11, [[1, 6]]])         # pendant, so v1 can be a cut
        nu = np.full(len(G), 3.5)
        A3 = np.eye(3)
        out = {}
        for src in (0, 1, 6):
            _, info = evaluate_graph(G, nu, A3, n_points=8, source=src,
                                     richardson=True, return_diagnostics=True)
            out[src] = info
        # source 0 pins the degree-2 vertex -> the two cuts coincide there
        # -> no release -> DECLINED.  It went to the box when this test
        # was written; the slab arm (`_SLAB_N_BY_CLASS`) landed later on
        # the same branch and now takes the declined block instead.  The
        # DECLINE is what this test is about and is unchanged; asserting
        # the box specifically was asserting the arm that happened to be
        # behind it.
        assert out[0]["n_block_dense_torus_declined"] == 1
        assert out[0]["n_block_slab"] + out[0]["n_block_direct_sum"] == 1
        # any other boundary leaves the degree-2 escape alive -> released.
        for src in (1, 6):
            assert out[src]["n_block_dense_torus_declined"] == 0
            assert out[src]["n_block_dense_torus"] == 1
            assert out[src]["n_block_split"] == 1

    def test_the_cuts_that_make_that_happen(self):
        from gzl._elimination import (
            min_free_cut_nu, sp_reduced_cut_nu,
        )
        V6E11 = [(0, 1), (0, 4), (1, 2), (1, 3), (1, 5), (2, 3),
                 (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
        emap = {}
        for u, v in V6E11:
            emap[(min(u, v), max(u, v))] = 3.5
        assert min_free_cut_nu(emap, 0, ()) == pytest.approx(7.0)
        assert sp_reduced_cut_nu(emap, 0, ()) == pytest.approx(7.0)
        for r in (1, 2, 3, 4, 5):
            assert min_free_cut_nu(emap, r, ()) == pytest.approx(7.0)
            assert sp_reduced_cut_nu(emap, r, ()) == pytest.approx(14.0)

    def test_the_hatch_needs_the_split_switched_ON_not_merely_reducible(
            self, monkeypatch):
        r"""``sp_n_points=None`` means the mode is still on the coarse
        grid, however reducible the block is.  The predicate is the
        conjunction, and getting it wrong would release exactly the
        blocks the gate exists for.
        """
        K4SUB = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 4], [4, 3]],
            dtype=int)
        nu_sub = np.full(len(K4SUB), 1.5)
        monkeypatch.setattr(frontend, "_TORUS_MIN_N_BY_CLASS", {(1, 2): 999})
        _, off = evaluate_graph(
            K4SUB, nu_sub, A1, n_points=16, richardson=True,
            sp_n_points=16, return_diagnostics=True)
        assert off["n_block_dense_torus_declined"] == 1


class TestTheD3SplitEntryIsANoOpWhereItCannotAct:
    r"""Adding ``_SPLIT_SP_N_POINTS[3]`` switches the split on for EVERY
    d = 3 dense block, including the exponent-2 cores that ship today.

    It cannot disturb them, and the reason is structural rather than
    statistical: the split reduces the series/parallel part, and on a
    3-connected block there is none, so the collapse is the identity.
    The three cores that ship at d = 3 are all in that group.  Asserted
    on the cuts here (fast) and measured bit-identical at sp = 64, 128
    and 256 in the constant's own comment.
    """

    CORES = {
        "K4": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        "prism": [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
                  (0, 3), (1, 4), (2, 5)],
        "K33": [(i, j) for i in (0, 1, 2) for j in (3, 4, 5)],
        "K5": [(i, j) for i in range(5) for j in range(i + 1, 5)],
    }

    @pytest.mark.parametrize("name", ["K4", "prism", "K33", "K5"])
    def test_the_shipped_d3_cores_have_nothing_for_the_split_to_reduce(
            self, name):
        from gzl._elimination import (
            min_free_cut_nu, sp_reduced_cut_nu,
        )
        emap = {}
        for u, v in self.CORES[name]:
            k = (min(u, v), max(u, v))
            emap[k] = emap.get(k, 0.0) + 3.5
        root = max({v for e in self.CORES[name] for v in e},
                   key=lambda r: (sp_reduced_cut_nu(emap, r, ()), -r))
        assert (sp_reduced_cut_nu(emap, root, ())
                == pytest.approx(min_free_cut_nu(emap, root, ())))

    def test_a_subdivided_edge_is_what_makes_the_split_act(self):
        from gzl._elimination import (
            min_free_cut_nu, sp_reduced_cut_nu,
        )
        K4SUB = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (4, 3)]
        emap = {}
        for u, v in K4SUB:
            emap[(min(u, v), max(u, v))] = 3.5
        assert min_free_cut_nu(emap, 0, ()) == pytest.approx(7.0)    # 2 nu
        assert sp_reduced_cut_nu(emap, 0, ()) == pytest.approx(10.5)  # 3 nu

    def test_the_entry_is_the_measured_one(self):
        r"""128 rather than 64 (still in the trough: 96x the box against
        2442x on K4SUB at n_points = 16) and rather than 256 (better on
        K4SUB, worse on V6E9a -- both at the core floor -- for 8x the
        grid).
        """
        assert frontend._SPLIT_SP_N_POINTS[3] == 128


class TestTheD3SplitIsScopedToZeroMomentum:
    r"""The d = 3 entry is a k = 0 measurement and does not carry to the
    finite-k arm.

    There is no third family at finite k -- the sigma_eff ladder does not
    run there and the slab is a k = 0 instrument -- so all that can be
    measured is whether the split moves the torus toward the box or away
    from it.  Measured on K4SUB at d = 3, n_points = 16, it moves it
    AWAY at every non-zero k tested (0.92x-0.93x on the |torus - box|
    gap) and toward it only at k = 0 (1.64x), while K4, which the split
    cannot act on, is 1.00x at every k -- the control that says the
    0.92x is the split and not the harness.  That is not evidence the
    split is wrong at finite k; it is evidence that nothing available can
    tell, and a default must not be set on that.

    An explicit ``sp_n_points`` is still honoured, because a finite-k
    campaign has to be able to drive the path it is measuring.
    """

    K4SUB = np.array(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 4], [4, 3]], dtype=int)
    A3 = np.eye(3)

    def _at(self, sp, d=3):
        nu = np.full(len(self.K4SUB), 3.5)
        A = np.eye(d)
        k = [0.25, 0.25, 0.0][:d]
        return float(evaluate_graph(
            self.K4SUB, nu, A, source=0, terminal=1, momentum=k,
            n_points=16, sp_n_points=sp))

    def test_d3_is_in_the_split_table_but_not_the_finite_k_set(self):
        assert 3 in frontend._SPLIT_SP_N_POINTS
        assert 3 not in frontend._SPLIT_FINITE_K_DIMS
        assert {1, 2} <= set(frontend._SPLIT_SP_N_POINTS)
        assert {1, 2} <= frontend._SPLIT_FINITE_K_DIMS

    def test_the_d3_finite_k_default_is_the_no_split_value(self):
        assert self._at(None) == self._at(16)      # 16 == n_points -> off

    def test_an_explicit_grid_is_still_honoured_at_finite_k(self):
        assert self._at(128) != self._at(None)

    def test_the_k_zero_arm_of_the_same_dimension_keeps_the_split(self):
        nu = np.full(len(self.K4SUB), 3.5)
        on = float(evaluate_graph(self.K4SUB, nu, self.A3, n_points=16,
                                  richardson=True))
        off = float(evaluate_graph(self.K4SUB, nu, self.A3, n_points=16,
                                   richardson=True, sp_n_points=16))
        assert on != off

    def test_d2_finite_k_is_untouched(self):
        r"""The suppression is keyed on the dimension, so the dimensions
        that already shipped the split at finite k must not notice.
        """
        assert self._at(None, d=2) == self._at(512, d=2)


class TestTheShippedCalibration:
    r"""Pins what was measured, so a later edit has to restate it.

    These are not redundant with the routing tests: those check the
    MECHANISM on a monkeypatched table, and would keep passing if the
    shipped entry were deleted or moved.
    """

    def test_the_table_is_the_measured_one(self):
        r"""d = 3, exponent 3, n >= 12.  Measured against a slab
        reference at nu = 3.25/3.5/4.0/5.0 (K5) and 3.5 (V6E11): n = 12
        is the first grid at which both cores clear the box, n = 10 is a
        wash (0.95x / 1.05x) and n = 8 loses by 5.0x / 10.6x.
        """
        assert frontend._TORUS_MIN_N_BY_CLASS == {(3, 3): 12}

    def test_the_gate_reads_the_graded_grid_not_the_delivery_grid(self):
        r"""The connectivity-graded core can only LOWER the grid a block
        actually runs at, so gating on ``n_points`` would admit a block
        and then evaluate it BELOW its own threshold.

        This is not hypothetical: an earlier version of this test was a
        tripwire asserting ``3 not in _CORE_KAPPA``, and it fired the
        moment the d = 3 entry landed.  K5 at ``n_points`` 12, 16 and 32
        grades to 8 and lands at 3.075e-07 -- 0.20x the box, the exact
        regression the table exists to prevent.

        The two grading calls below are the ones ``_evaluate_via_topology``
        makes at its grading site, in the same order and at the same pin,
        which is what makes "the number gated on is the number evaluated
        at" checkable here rather than only end to end.
        """
        assert frontend._CORE_KAPPA[3] == 3.0
        edges = np.array([[i, j] for i in range(5) for j in range(i + 1, 5)],
                         dtype=int)
        nu = np.full(len(edges), 3.5)
        verts, pin, keep, d = list(range(5)), 0, (), 3

        # K5 is 4-regular: SP reduction removes nothing, so the split does
        # not act on it and the gate is live.
        emap = {(int(u), int(v)): float(w)
                for (u, v), w in zip(edges.tolist(), nu.tolist())}
        s_blk = frontend._min_free_cut_nu(emap, pin) - d
        s_core = frontend._sp_reduced_cut_nu(emap, pin) - d
        assert s_core == pytest.approx(s_blk)

        for n_points in (12, 16, 32):
            graded = frontend._core_n_for_block(
                n_points, s_blk, s_core, d,
                sigma_ref=frontend._pass_floor_sigma(nu, d))
            graded = frontend._core_n_capped(
                graded, edges, verts, pin, keep, d)
            n_eff = min(int(n_points), int(graded))
            assert n_eff < frontend._TORUS_MIN_N_BY_CLASS[(3, 3)]
            # Gating on n_points would ADMIT ...
            assert frontend._dense_torus_is_accurate(
                edges, verts, pin, keep, d, n_points, True)
            # ... gating on the graded grid, which is what ships, declines.
            assert not frontend._dense_torus_is_accurate(
                edges, verts, pin, keep, d, n_eff, True)

    def test_the_gate_is_fed_the_graded_grid_END_TO_END(self):
        r"""Through the router, not re-implemented beside it.

        ``test_the_gate_reads_the_graded_grid_not_the_delivery_grid``
        above checks the MECHANISM by calling `_core_n_for_block` /
        `_core_n_capped` itself.  That is worth having, and it is not
        enough: it passes whether or not the shipped call site actually
        USES the number, because it never touches the call site.

        It did not catch a real regression.  One commit shipped the
        gate reading `_n_eff`; a later one -- a block-cache fix that
        should not have gone near it -- reverted that one line to
        `int(n_points)` while leaving the `_n_eff` computation in place,
        dead.  The whole suite stayed green, including `--runslow`.

        Measured with the regression live, K5 at n_points = 16: admitted
        to the torus, graded to n = 8, landing 3.0849e-07 from the slab
        reference against the box's 6.35e-08 -- **4.9x WORSE than the
        box**, which is precisely what this table exists to prevent.
        With the gate reading `_n_eff` it is declined at every
        n_points and the slab takes it at 1.3441e-08.

        So this test asserts the ROUTE, which is the only thing that
        cannot be true while the call site is wrong.
        """
        edges = np.array([[i, j] for i in range(5) for j in range(i + 1, 5)],
                         dtype=int)
        nu, A3 = np.full(len(edges), 3.5), np.eye(3)
        for n_points in (12, 16, 32):
            _, info = evaluate_graph(edges, nu, A3, n_points=n_points,
                                     richardson=True, return_diagnostics=True)
            assert info["n_block_dense_torus_declined"] == 1, (
                f"K5 at n_points={n_points} was ADMITTED: the gate is being "
                f"fed the delivery grid, not the graded one"
            )
            assert info["n_block_dense_torus"] == 0

    def test_the_exponent_2_class_at_d3_is_not_gated(self):
        r"""The control on "cannot regress what ships".  K4, the prism
        and K33 are exponent-2 at d = 3 and route to the torus today;
        the table must leave them alone.
        """
        assert (3, 2) not in frontend._TORUS_MIN_N_BY_CLASS


@pytest.mark.slow
class TestTheShippedEntryEndToEnd:
    r"""V6E11, a d = 3 core the entry was measured on, through the router.

    K5, the other one, is declined and taken by the slab, which
    ``test_slab.TestTheArmRoutesEndToEnd`` pins.
    """

    V6E11 = np.array(
        [[0, 1], [0, 4], [1, 2], [1, 3], [1, 5], [2, 3],
         [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]], dtype=int)

    def test_v6e11_is_no_longer_declined_because_the_split_acts_on_it(self):
        r"""V6E11 at ``n_points = 8``: admitted to the torus, not declined.

        The split predicate is read at the evaluation's own pin
        (``hybrid._planner_pin``), and V6E11 is one of the 172 of 2636
        dense-block occurrences where that pin disagrees with the
        router's root.  Read at the root, the gate would refuse a block
        the split takes from 10.6x worse than the box to >= 57.7x better.
        """
        edges, nu, A3 = self.V6E11, np.full(len(self.V6E11), 3.5), np.eye(3)
        _, info = evaluate_graph(edges, nu, A3, n_points=8, richardson=True,
                                 return_diagnostics=True)
        assert info["n_block_dense_torus_declined"] == 0
        assert info["n_block_dense_torus"] == 1
        assert info["n_block_split"] == 1
        assert info["n_block_slab"] == 0
        assert info["n_block_direct_sum"] == 0


class TestTheBalancedFiniteKRungs:
    r"""The finite-k arm's class-keyed rungs (``_FK_PAIR_BY_P``).

    The BALANCED policy: each class ships the cheapest rung pair that
    stays at least box-accurate, so no single block dominates a pass.
    Measured (512 pass k-points vs the n = 18 pinned
    reference): p >= 10 ships (8, 10) -- 4.5e-08 median, worse than
    the box at exactly one point (k = 0, 1.57x, 9.9e-08 rel there),
    22% of the (10, 12) cost.  7 <= p < 10 keeps (10, 12): every
    top-rung-10 scheme fails that class (69/512 points, up to 9.5x
    the box) -- its p = 7.5 asymptotics are not clean below n = 12.
    """

    K5 = np.array([[i, j] for i in range(5) for j in range(i + 1, 5)],
                  dtype=int)
    S01 = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [1, 4],
                    [2, 3], [2, 4], [3, 4]], dtype=int)

    @staticmethod
    def _rungs(p):
        return next(
            (r for pmin, r in frontend._FK_PAIR_BY_P if p >= pmin), None)

    def test_the_table_and_its_selection(self):
        assert frontend._FK_PAIR_BY_P == ((10.0, (8, 10)),
                                          (7.0, (10, 12)))
        assert self._rungs(11.0) == (8, 10)     # K5 class
        assert self._rungs(10.0) == (8, 10)     # boundary in
        assert self._rungs(9.99) == (10, 12)    # just below
        assert self._rungs(7.5) == (10, 12)     # S01 class
        assert self._rungs(7.0) == (10, 12)     # floor in
        assert self._rungs(6.99) is None        # ladder-borne: the box

    def test_the_emblem_classes_have_the_measured_basis(self):
        r"""The basis excludes BOTH pins (the terminal's z-sum is
        exact on the torus).  S01 = K5-e is the sharp case: with the
        hop across the missing edge (t = 4, the other deg-3 vertex)
        the slow escapes are pinned away and p = 11; with t = 1 a
        deg-3 vertex stays free and p = 7.5 -- the sector measured above.
        """
        from gzl import _elimination
        for edges, t, want in ((self.K5, 1, 11.0), (self.S01, 1, 7.5),
                               (self.S01, 4, 11.0)):
            emap = {}
            for u, v in edges.tolist():
                k = (min(u, v), max(u, v))
                emap[k] = emap.get(k, 0.0) + 3.5
            p = float(_elimination.min_free_cut_nu(emap, 0, (t,))) - 3
            assert p == want

    def test_k5_finite_k_ships_the_cheap_pair(self):
        r"""K5 at k = 0 through the router: the finite-k arm fires and
        ships the (8, 10) pair, 3.9x cheaper than the (10, 12) it
        replaced (measured 5.7 s vs 21 s).  Its value is the
        frozen key ``router_k0|K5|d3|n8`` of test_engine_reference.py.
        """
        _, info = evaluate_graph(
            self.K5, 3.5, np.eye(3), source=0, terminal=1,
            momentum=np.zeros(3), n_points=8, return_diagnostics=True)
        assert info["n_block_slab_finite_k"] == 1
        assert info["n_block_direct_sum"] == 0

    def test_grid_requests_ride_the_floor_tier(self):
        r"""A pass-grid request ships the block PINNED AT THE PASS n:
        the pass floor is the ordinary blocks' raw truncation at that
        n (measured med 1.4e-05 / 3.3e-05 on the O9 census
        neighbours), and the pinned block sits 10x below it (med
        1.3e-06, max 2.5e-05) at ~1 s -- the block stops being
        special.  Single-k requests keep the class pair (the frozen
        keys' standard); asserted in the slow test above.
        """
        g, info = evaluate_graph(
            self.S01, 3.5, np.eye(3), source=0, terminal=4,
            n_points=8, return_diagnostics=True)
        assert info["n_block_slab_fk_floor"] == 1
        assert info["n_block_slab_finite_k"] == 0
        assert info["n_block_direct_sum"] == 0
        assert_pinned(g[0, 0, 0], "0x1.a22ab77245874p+4",
                      "S01 grid request, pinned at the pass grid")


class TestFloorMode:
    r"""``accuracy="floor"`` — the corpus-pass standard.

    In a pass, the work must stay approximately LINEAR in the number
    of graphs: no block may buy precision the pass cannot see.  The
    pass's accuracy floor is its ordinary blocks' own raw truncation
    at the shared ``n_points`` (p = 7.5 class med 1.4e-05–3.3e-05 at
    n = 8; the p = 4 class that dominates order ≥ 10 med
    2.3e-03–5.6e-03), so a declined block shipped at the pass grid —
    the exact fold of the same torus truncation — lands level with
    its neighbours (K5 vacuum raw-8: 3.1e-07, 45x BELOW the tight
    floor; the O11 (7,11) p = 4 sector pinned-8: 2.7e-03, level) at
    ~1 s instead of 20 s (slab@12) or an hours-class box grid.
    ``"strict"`` (the default) keeps the box-dominance standard the
    frozen reference keys pin; ``compute_series_coefficients``
    passes ``"floor"``.
    """

    K5 = np.array([[i, j] for i in range(5) for j in range(i + 1, 5)],
                  dtype=int)
    V6E11 = np.array(
        [[0, 1], [0, 4], [1, 2], [1, 3], [1, 5], [2, 3],
         [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]], dtype=int)

    def test_the_kwarg_validates_and_defaults_strict(self):
        with pytest.raises(ValueError, match="accuracy"):
            evaluate_graph(self.K5, 3.5, np.eye(3), n_points=8,
                           accuracy="fast")
        # Where nothing declines, floor is bit-identical to strict --
        # d = 1 has no gate entries at all.
        E, A1 = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3],
                          [2, 3]]), np.eye(1)
        assert (evaluate_graph(E, 1.5, A1, n_points=16)
                == evaluate_graph(E, 1.5, A1, n_points=16,
                                  accuracy="floor"))

    def test_floor_vacuum_ships_the_pass_grid(self):
        r"""K5 at k = 0: declined as always, but the slab arm runs at
        the PASS grid (the fold-exact torus-8 value, dev 3.1e-07 vs
        the n = 18 ladder) instead of the class n = 12 -- ~1 s, not
        ~16 s.  Strict on the same block is pinned bit-exactly by the
        frozen key router_vac|K5|d3|n8.
        """
        v, info = evaluate_graph(
            self.K5, 3.5, np.eye(3), n_points=8, richardson=True,
            return_diagnostics=True, accuracy="floor")
        assert info["n_block_dense_torus_declined"] == 1
        assert info["n_block_slab"] == 1
        assert info["n_block_direct_sum"] == 0
        assert_pinned(v, "0x1.7adaf8cfecb1ap+2",
                      "K5 floor vacuum at the pass grid")

    def test_floor_finite_k_takes_any_p(self):
        r"""V6E11 with both terminals on deg-4 vertices: the deg-2
        vertex stays free, p = 4 < 7, and strict sends it to the box
        (minutes single-k, hours as a grid -- the order-11 wall).  In
        floor mode the pinned pass-grid tier takes it at ~1 s.
        """
        v, info = evaluate_graph(
            self.V6E11, 3.5, np.eye(3), source=1, terminal=2,
            momentum=np.zeros(3), n_points=8,
            return_diagnostics=True, accuracy="floor")
        assert info["n_block_dense_torus_declined"] == 1
        assert info["n_block_slab_fk_floor"] == 1
        assert info["n_block_fk_outer_pin"] == 0    # a tie sector stays
        assert info["n_block_direct_sum"] == 0
        assert_pinned(v, "0x1.960668f3511fap+6",
                      "V6E11 floor finite-k, p = 4")

    def test_the_default_resolves_by_request_shape(self):
        r"""``accuracy=None`` (the default): a full-BZ grid request is
        a pass context and resolves to floor -- bit-identical to the
        explicit kwarg; vacuum scalars and explicit-k requests resolve
        to strict, which is what the frozen reference keys assert
        every slow-gate run (they call with no kwarg).
        """
        g1 = evaluate_graph(self.V6E11, 3.5, np.eye(3), source=1,
                            terminal=2, n_points=8)
        g2 = evaluate_graph(self.V6E11, 3.5, np.eye(3), source=1,
                            terminal=2, n_points=8, accuracy="floor")
        assert np.array_equal(np.asarray(g1), np.asarray(g2))
        with pytest.raises(ValueError, match="accuracy"):
            evaluate_graph(self.V6E11, 3.5, np.eye(3), source=1,
                           terminal=2, n_points=8, accuracy="pass")
