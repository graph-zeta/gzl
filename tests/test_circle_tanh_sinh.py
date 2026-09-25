# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Fixed-node (tanh-sinh) cycle quadrature, and its Epstein table cache.

``zeta_circle`` is the single largest line item in a d = 2 corpus pass --
measured ~42 s, and INDEPENDENT of ``n_points``, so no amount of grid
work touches it.  Profiling puts 83-84% of that in ``epstein_zeta`` at
15-18 us a call, and the adaptive rule re-derives its own node set for
every cycle, so nothing is ever reused.

The fixed rule changes that twice over: 4424 nodes against 17k-35k, flat
in cycle length; and because the nodes are the SAME for every cycle and
every lattice, ``zeta_E(nu, A, .)`` is tabulated once per exponent.  The
shipped corpora carry six distinct exponents.

WHAT THESE TESTS PROTECT.  Not the speed -- the risk is that a fixed rule
silently loses accuracy somewhere the adaptive one held, so the agreement
gates sweep the axes where that could happen: sigma down to the ``nu > d``
edge, non-uniform nu from bundle multiplicities, and eight lattices from
cubic to strongly sheared.  Every case measured at round-off during
development (0.0 to 3.7e-16), including two anisotropic cells where it is
the ADAPTIVE rule that drifts (1.7e-14 on diag(1,5)) and the fixed one
that sits at 0.

The parameters are NOT free.  ``M = 48, h = 0.08`` is the validated pair,
and the 14-node Duffy angular rule must not be swapped for plain
Gauss-Legendre -- that was measured to break by 40% at every order.
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl import circle as C
from gzl.circle import (
    zeta_circle,
    zeta_circle_tanh_sinh,
    _ts_grid_2d,
    _ts_table_clear,
)

SQUARE = np.eye(2)
SCALED = 1.7 * np.eye(2)
RECT = np.diag([1.0, 2.0])
ANISO = np.diag([1.0, 5.0])
TRI = np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2.0]])
OBLIQUE = np.array([[1.0, np.cos(np.deg2rad(75.0))],
                    [0.0, np.sin(np.deg2rad(75.0))]])
SHEAR = np.array([[1.0, 0.9], [0.0, 1.0]])
GENERIC = np.array([[1.3, 0.37], [0.21, 0.94]])
ALL_LATTICES = [SQUARE, SCALED, RECT, ANISO, TRI, OBLIQUE, SHEAR, GENERIC]


class TestTheSwitch:

    def test_it_is_on(self):
        r"""ON by default.  Agreement with the adaptive rule is round-off
        rather than bitwise, so this was a deliberate flip, validated
        across sigma, uniform and non-uniform nu, and eight lattices --
        see the agreement gates above.
        """
        assert C.USE_TANH_SINH is True

    def test_it_actually_routes(self, monkeypatch):
        r"""Liveness, both ways.  With the flag ON the fixed rule must be
        taken; with it OFF the adaptive one must be.  Without this, every
        agreement test above would pass just as well on a flag that does
        nothing.
        """
        def boom(*a, **k):
            raise AssertionError("routed here")
        monkeypatch.setattr(C, "zeta_circle_tanh_sinh", boom)
        with pytest.raises(AssertionError):
            zeta_circle(np.array([2.5] * 4), SQUARE)      # ON -> fixed rule
        monkeypatch.setattr(C, "USE_TANH_SINH", False)
        zeta_circle(np.array([2.5] * 4), SQUARE)          # OFF -> adaptive

    def test_the_kill_switch_restores_the_adaptive_value(self, monkeypatch):
        r"""The escape hatch has to actually work: with the flag off the
        result must be the adaptive rule's, exactly."""
        from gzl.circle import int_full_2d, epstein_zeta_prod
        nu_vec = np.array([2.5] * 5)
        monkeypatch.setattr(C, "USE_TANH_SINH", False)
        got = zeta_circle(nu_vec, SQUARE)
        A_star = np.linalg.inv(SQUARE).T
        ref = int_full_2d(lambda *ys: epstein_zeta_prod(
            nu_vec, SQUARE, A_star, np.array(ys, dtype=float)))
        assert got == ref


class TestRefusals:

    @pytest.mark.parametrize("d", [1, 4])
    def test_unsupported_dimensions_are_refused(self, d):
        r"""d = 1 keeps its adaptive rule (one cheap 1-D integral) and
        d >= 4 has no Duffy decomposition here.  A silent fall-through
        would return a d = 2 answer for another lattice entirely.
        """
        with pytest.raises(NotImplementedError, match="d = 2 and d = 3"):
            zeta_circle_tanh_sinh(np.array([4.5 + d] * 3), np.eye(d))

    def test_three_dimensions_are_supported(self):
        r"""d = 3 is now a fixed-node rule, not a refusal.

        Pinned against the ADAPTIVE rule, which is the oracle here: the
        fixed rule has to reproduce ``int_full_3d``'s decomposition term
        for term, and a mis-stated permutation or Jacobian shows up as a
        gross disagreement rather than a small one.
        """
        nu_vec = np.array([3.5, 3.5, 3.5])
        got = zeta_circle_tanh_sinh(nu_vec, np.eye(3))
        want = C.int_full_3d(
            lambda *y: C.epstein_zeta_prod(
                nu_vec, np.eye(3), np.linalg.inv(np.eye(3)).T, np.asarray(y)
            )
        )
        assert abs(got - want) / abs(want) < 1e-11

    @pytest.mark.parametrize("a", [(1.0, 2.0, 3.0), (0.7, 1.9, 3.3)])
    def test_the_tiling_is_pinned_by_an_asymmetric_analytic_integral(self, a):
        r"""The grid must integrate a NON-symmetric function correctly.

        ``sum(w) == 1`` and the ``eye(3)`` value test are both blind to a
        wrong permutation: on a cubic cell ``G = A^T A = I`` is invariant
        under every signed permutation, so ANY permutation triple over the
        same base pyramid integrates to the same number.  A mutated
        ``PERMS`` that covers one argument slot twice and another never
        passes the whole suite while being ~10-58% wrong on any cell whose
        Gram matrix is not signed-permutation invariant.

        ``cosh(a . y)`` closes that hole and costs nothing: it is even
        under ``y -> -y`` (so it respects the rule's factor-2 sign
        folding) but NOT coordinate-symmetric, and

            \int_{[-1/2,1/2]^d} cosh(a . y) dy = prod_i 2 sinh(a_i/2)/a_i

        exactly, with zero Epstein evaluations.  Measured: shipped grid
        2.9e-15, mutated PERMS 6.2e-02.
        """
        a = np.asarray(a, dtype=float)
        ys, wt = C._ts_grid_3d()
        got = float(wt @ np.cosh(ys @ a))
        want = float(np.prod(2.0 * np.sinh(a / 2.0) / a))
        assert abs(got - want) <= 1e-13 * abs(want)

    @pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-6, 1e-9])
    def test_the_gram_test_is_scale_invariant(self, scale):
        r"""The sheet grouping must not depend on the cell's SIZE.

        The Gram-invariance test decides which sheets share a table.  With
        the tolerance clamped to an absolute floor, a cell whose Gram
        entries all fall below it compares equal to everything: at
        |A| ~ 1e-6 a sheared cell collapsed 12 sheets into 1 and the table
        was silently wrong.  Scaling a lattice cannot change its symmetry,
        so the class count must not move.
        """
        ops, _ = C._TS_GRID_3D_OPS[("3d", 48, 0.08)] if C._TS_GRID_3D_OPS \
            else (C._ts_grid_3d() and C._TS_GRID_3D_OPS[("3d", 48, 0.08)])
        sheared = np.array([[1.0, .3, .1], [0.0, 1.2, .2], [.1, 0.0, .9]])
        assert len(C._ts_sheet_classes(scale * sheared, ops)) == len(ops)
        assert len(C._ts_sheet_classes(scale * np.eye(3), ops)) == 1

    @pytest.mark.parametrize("A", [np.eye(2), 1.7 * np.eye(2),
                                   np.array([[1.0, 0.5], [0.0, 0.8660254037844386]])])
    def test_the_direct_table_is_chunk_size_independent(self, A, monkeypatch):
        r"""Blocking the node axis must not change the answer.

        The unchunked version of this was a 74 GB phase matrix at d = 3
        (measured 49 GB peak on a real pass), so the chunk exists for
        memory.  It is EXACT in exact arithmetic -- each node's dot
        product is independent -- but not bitwise: BLAS picks a different
        inner blocking per shape.  Round-off is the only allowed
        difference, and nothing in the suite gated this path before.
        """
        ys, _ = C._ts_grid_2d()
        monkeypatch.setattr(C, "_DIRECT_TABLE_MAX_BYTES", 10 ** 12)
        one = C._epstein_table_direct(20.0, A, ys)
        monkeypatch.setattr(C, "_DIRECT_TABLE_MAX_BYTES", 4096)
        many = C._epstein_table_direct(20.0, A, ys)
        assert np.max(np.abs(one - many)) <= 1e-14 * np.max(np.abs(one))

    def test_the_sheet_blocks_stay_index_aligned(self):
        r"""Sheet ``s``'s node ``i`` IS ``S_s @ (sheet 0's node i)``.

        The symmetry reduction copies one sheet's table to its whole
        class, which is only valid under that correspondence.  The code
        asserts ``len(ys) % len(ops) == 0`` -- a COUNT, which a
        misaligned-but-equal-sized tiling would also satisfy.  This
        checks the alignment itself, and it holds exactly (0.0, not to a
        tolerance) because the weight is independent of the sheet, so the
        combined-weight floor removes the same nodes from every block.
        """
        ys, wt = C._ts_grid_3d()
        ops, block = C._TS_GRID_3D_OPS[("3d", 48, 0.08)]
        assert len(ops) * block == len(ys)
        base = ys[:block]
        for s in range(len(ops)):
            got = ys[s * block:(s + 1) * block]
            assert np.array_equal(got, base @ ops[s].T), f"sheet {s} misaligned"
            assert np.array_equal(wt[s * block:(s + 1) * block], wt[:block])

    def test_the_three_dimensional_weights_integrate_to_the_cell(self):
        r"""Sum of the combined weights is the BZ volume, 1.

        This is the cheap structural check that the 12 angular sheets
        (2 signs x 2 signs x 3 cyclic reorderings) and the ``(2x)^2``
        Duffy Jacobian are all present exactly once.
        """
        _, wt = C._ts_grid_3d()
        assert abs(float(wt.sum()) - 1.0) < 1e-12

    def test_nu_below_the_convergence_bound_is_refused(self):
        with pytest.raises(ValueError, match="nu_i > d"):
            zeta_circle_tanh_sinh(np.array([2.5, 1.5]), SQUARE)


class TestTheTableCache:

    def setup_method(self):
        _ts_table_clear()

    def test_cached_equals_cold(self):
        nu_vec = np.array([2.5, 2.5, 5.0, 2.5])
        cold = zeta_circle_tanh_sinh(nu_vec, SQUARE)
        warm = zeta_circle_tanh_sinh(nu_vec, SQUARE)
        _ts_table_clear()
        again = zeta_circle_tanh_sinh(nu_vec, SQUARE)
        assert cold == warm == again

    def test_the_lattice_is_in_the_key(self):
        r"""The table is ``zeta_E(nu, A, 0, A* y)`` -- it depends on the
        cell.  Without ``A`` in the key a sheared lattice would be served
        the cubic table: finite, plausible, wrong everywhere.
        """
        nu_vec = np.array([2.5] * 5)
        a = zeta_circle_tanh_sinh(nu_vec, SQUARE)
        b = zeta_circle_tanh_sinh(nu_vec, TRI)
        assert a != b
        _ts_table_clear()
        assert a == zeta_circle_tanh_sinh(nu_vec, SQUARE)

    def test_the_exponent_is_in_the_key(self):
        a = zeta_circle_tanh_sinh(np.array([2.5] * 5), SQUARE)
        b = zeta_circle_tanh_sinh(np.array([3.5] * 5), SQUARE)
        assert a != b

    def test_one_table_serves_many_cycles(self):
        r"""The whole point.  Cycles of every length and multiplicity
        drawn from ONE exponent must need exactly ONE table.
        """
        for L in range(3, 12):
            zeta_circle_tanh_sinh(np.array([2.5] * L), SQUARE)
        assert len(C._TS_TABLE_CACHE) == 1

    def test_the_budget_is_respected_and_clear_resets_it(self, monkeypatch):
        monkeypatch.setattr(C, "TS_TABLE_MAX_BYTES", 100_000)
        _ts_table_clear()
        for nu in (2.5, 3.0, 3.5, 4.0, 4.5, 5.0):
            zeta_circle_tanh_sinh(np.array([nu] * 4), SQUARE)
        assert C._TS_TABLE_BYTES <= 100_000
        _ts_table_clear()
        assert C._TS_TABLE_BYTES == 0 and not C._TS_TABLE_CACHE

    def test_a_table_is_read_only(self):
        zeta_circle_tanh_sinh(np.array([2.5] * 4), SQUARE)
        tab = next(iter(C._TS_TABLE_CACHE.values()))
        with pytest.raises(ValueError):
            tab[0] = 0.0


class TestTheGrid:

    def test_it_is_independent_of_nu_and_lattice(self):
        r"""This is exactly what makes the table cacheable: the node set
        is universal, so only the Epstein VALUES depend on (nu, A).
        """
        a_y, a_w = _ts_grid_2d()
        b_y, b_w = _ts_grid_2d()
        assert a_y is b_y and a_w is b_w

    def test_the_node_count_is_the_validated_one(self):
        r"""4032 nodes.

        The nominal construction is 4 sub-integrals x 14 angular nodes x 77
        radial nodes = 4424.  392 of those have a COMBINED weight below
        ``_TS_WEIGHT_FLOOR`` and are dropped: the 2D weight multiplies in
        the Duffy Jacobian ``2x`` with ``x`` down to ~1e-16, so a node can
        clear ``_tanh_sinh_1d``'s own 1e-18 filter and still arrive
        weighing ~4e-33.  They contribute nothing (see
        ``test_the_dropped_weight_is_zero_in_double_precision``) and they
        sit exactly where ``epstein_zeta`` overflows at large ``nu``.

        Pinned because M and h are validated parameters, not knobs.
        """
        ys, wt = _ts_grid_2d()
        assert len(ys) == len(wt) == 4032


class TestLargeExponentsDoNotPoisonTheGrid:
    r"""The regression that shipped NaN, and the guard that stops it.

    ``epstein_zeta(nu, A, 0, y)`` returns ``nan+nanj`` for large ``nu`` at
    machine-epsilon ``|y|`` — an intermediate ``|y|^-nu`` overflows double
    precision, with the frontier at ``nu * |log10|y|| ~ 350`` (epsteinlib
    0.6.2 and 0.5.1; reported upstream).  The double-exponential rule is
    exactly the one that puts nodes at ``|y| ~ 1e-16``; the adaptive rule
    never samples that close, which is why only the fixed rule tripped.

    One NaN node makes the whole dot product NaN, so a cycle came back
    NaN on 8.4e-31 of the total weight.  Before the guard: ``nu_max``
    21.935030 finite, 21.935031 NaN.  In corpus terms a bundle of
    multiplicity ``m`` merges to exponent ``m*nu``, so this reached
    order 11 at sigma = 0.5 and **order 5** at sigma = 8.

    These gate the failure directly rather than the fix: they assert
    finiteness and agreement with the adaptive rule ABOVE the old
    threshold, so they stay meaningful if the guard is reimplemented.
    """

    # Straddles the old threshold (21.935) and reaches what the corpus
    # actually demands: a bundle of multiplicity m merges to m*nu, so the
    # d = 2 order-11 corpus reaches 27.5 at sigma = 0.5, 44 at sigma = 2
    # and 110 at sigma = 8.
    BIG = [22.0, 22.5, 24.0, 30.0, 40.0, 44.0, 66.0, 110.0]

    @pytest.mark.parametrize("nu_max", BIG)
    def test_a_big_bundle_is_finite_and_correct(self, nu_max):
        r"""Finite, and equal to the refined rule.

        The referee is a REFINED fixed rule (M = 80, h = 0.04), the same
        internal device the module's own validation used, because the
        ADAPTIVE rule cannot serve as the oracle out here: it calls
        ``epstein_zeta`` at its own nodes and so goes NaN itself above
        ``nu_max ~ 88``.  This was never a regression unique to the fixed
        rule; the fixed rule merely lowered an existing ceiling from ~88
        to ~22, and the direct Fourier table removes it at both.

        nu = 110 is a completely regular point of the integrand — the
        value there is just the nearest-neighbour limit — so "NaN" was
        only ever an artifact of how the sum was evaluated.
        """
        nu_vec = np.array([nu_max, 2.5, 2.5])
        fixed = zeta_circle_tanh_sinh(nu_vec, SQUARE)
        assert np.isfinite(fixed), (
            f"tanh-sinh returned {fixed} at nu_max = {nu_max}; a node at "
            f"machine-epsilon |y| has poisoned the dot product"
        )
        refined = zeta_circle_tanh_sinh(nu_vec, SQUARE, M=80, h=0.04)
        assert np.isfinite(refined)
        assert abs(fixed - refined) <= 1e-12 * abs(refined)

    @pytest.mark.parametrize("nu_max", [22.0, 22.5, 24.0, 30.0, 40.0, 44.0, 66.0])
    def test_it_matches_the_adaptive_rule_while_that_rule_still_works(self, nu_max):
        """Below the adaptive rule's own ceiling the two must agree."""
        nu_vec = np.array([nu_max, 2.5, 2.5])
        C.USE_TANH_SINH = False
        try:
            adaptive = zeta_circle(nu_vec, SQUARE)
        finally:
            C.USE_TANH_SINH = True
        # If this ever goes NaN the comparison would be vacuous, so gate it.
        assert np.isfinite(adaptive), (
            f"the adaptive reference is itself NaN at nu_max={nu_max}; "
            f"this parametrisation must stay below its ceiling (~88)"
        )
        fixed = zeta_circle_tanh_sinh(nu_vec, SQUARE)
        assert abs(fixed - adaptive) <= 1e-12 * abs(adaptive)

    def test_the_corpus_graph_that_actually_failed(self):
        """Triangle with a nine-fold bundle, d = 2, sigma = 0.5.

        Two of 22,677 order-11 graph evaluations were this shape, and both
        returned NaN, which made the whole order-11 coefficient NaN.
        """
        nu_vec = np.array([9 * 2.5, 2.5, 2.5])
        assert np.isfinite(zeta_circle(nu_vec, SQUARE))

    # Length 3 is test_a_big_bundle_is_finite_and_correct[22.5].
    @pytest.mark.parametrize("length", [4, 5, 6])
    def test_longer_cycles_too(self, length):
        nu_vec = np.array([22.5] + [2.5] * (length - 1))
        assert np.isfinite(zeta_circle_tanh_sinh(nu_vec, SQUARE))

    def test_the_dropped_weight_is_zero_in_double_precision(self):
        """The guard must not cost accuracy — it may only drop nothing."""
        ys, wt = _ts_grid_2d()
        assert len(ys) == len(wt)
        # Every surviving node clears the floor, and the survivors carry
        # the entire weight of the rule.
        assert np.all(np.abs(wt) > C._TS_WEIGHT_FLOOR)
        assert abs(float(np.abs(wt).sum()) - 1.0) < 1e-15

    def test_small_exponents_are_untouched_by_the_guard(self):
        """Values below the old threshold must not have moved."""
        for nu_vec in ([2.5, 2.5, 2.5], [21.0, 2.5, 2.5], [5.0, 3.0, 4.0]):
            v = np.asarray(nu_vec, dtype=float)
            C.USE_TANH_SINH = False
            try:
                adaptive = zeta_circle(v, SQUARE)
            finally:
                C.USE_TANH_SINH = True
            assert abs(zeta_circle_tanh_sinh(v, SQUARE) - adaptive) <= 1e-12 * abs(adaptive)


class TestTheDirectFourierTable:
    r"""Above ``_TS_DIRECT_NU_MIN`` the table is summed, not called out for.

    ``zeta_E(nu, A, 0, A* y) = sum_{z != 0} exp(-2 pi i z.y) |A z|^-nu`` and
    for large ``nu`` that series is a handful of shells.  These gate the
    replacement against ``epstein_zeta`` in the window where the library
    still works, which is the only place the two can be compared at all.
    """

    OVERLAP = [20.0, 22.0, 25.0, 30.0, 35.0]

    @pytest.mark.parametrize("A", ALL_LATTICES)
    @pytest.mark.parametrize("nu", OVERLAP)
    def test_it_matches_epstein_zeta_where_both_are_in_range(self, nu, A):
        from epsteinlib import epstein_zeta
        ys, _ = _ts_grid_2d()
        A_star = np.linalg.inv(A).T
        zeros = np.zeros(len(A))
        lib = np.array([epstein_zeta(float(nu), A, zeros, A_star @ y) for y in ys])
        finite = np.isfinite(lib)
        # The comparison must actually have something to compare.
        assert finite.sum() > 0.9 * len(ys), (
            f"epstein_zeta itself is non-finite on {(~finite).sum()} of "
            f"{len(ys)} nodes at nu={nu}; nothing to gate against"
        )
        direct = C._epstein_table_direct(float(nu), A, ys)
        assert np.all(np.isfinite(direct))
        assert np.abs(direct[finite] - lib[finite]).max() < 1e-13

    @pytest.mark.parametrize("A", ALL_LATTICES)
    def test_it_is_finite_far_beyond_the_library(self, A):
        ys, _ = _ts_grid_2d()
        for nu in (44.0, 66.0, 110.0, 200.0):
            v = C._epstein_table_direct(float(nu), A, ys)
            assert np.all(np.isfinite(v)), f"non-finite at nu={nu}, A={A}"

    def test_the_truncation_shrinks_as_nu_grows(self):
        """Sanity on the radius rule: bigger nu needs fewer shells."""
        ys, _ = _ts_grid_2d()
        # Values must be stable against a deliberately larger radius.
        for nu in (25.0, 60.0, 110.0):
            loose = C._epstein_table_direct(nu, SQUARE, ys, tol=1e-18)
            tight = C._epstein_table_direct(nu, SQUARE, ys, tol=1e-24)
            assert np.abs(loose - tight).max() < 1e-14

    def test_a_singular_lattice_is_refused(self):
        ys, _ = _ts_grid_2d()
        with pytest.raises(ValueError, match="singular lattice"):
            C._epstein_table_direct(30.0, np.zeros((2, 2)), ys)


class TestSigmaMaxCircleRouting:
    r"""``SIGMA_MAX_CIRCLE`` must actually route, and route at the right place.

    Without this the direct table could be dead code and every other test
    in the class above would still pass — the two paths agree to round-off,
    which is exactly what makes a routing bug invisible.
    """

    def test_the_threshold_sits_inside_its_window(self):
        """Bounded below by cost, above by correctness.

        Below sigma ~ 14 the series stops being small and epsteinlib is
        cheaper; above sigma ~ 36.6 epsteinlib is not slower but WRONG.
        """
        assert 14.0 <= C.SIGMA_MAX_CIRCLE <= 30.0

    def test_above_the_threshold_the_direct_table_is_used(self, monkeypatch):
        seen = []
        real = C._epstein_table_direct
        monkeypatch.setattr(
            C, "_epstein_table_direct",
            lambda *a, **k: (seen.append(a[0]), real(*a, **k))[1],
        )
        C._ts_table_clear()
        nu_hi = C.SIGMA_MAX_CIRCLE + 2.0 + 1.0      # d = 2, comfortably above
        zeta_circle_tanh_sinh(np.array([nu_hi, 2.5, 2.5]), SQUARE)
        assert any(abs(n - nu_hi) < 1e-9 for n in seen), (
            f"nu={nu_hi} (sigma={nu_hi - 2}) should have taken the direct "
            f"path; _epstein_table_direct saw {seen}"
        )

    def test_below_the_threshold_it_is_not(self, monkeypatch):
        seen = []
        real = C._epstein_table_direct
        monkeypatch.setattr(
            C, "_epstein_table_direct",
            lambda *a, **k: (seen.append(a[0]), real(*a, **k))[1],
        )
        C._ts_table_clear()
        zeta_circle_tanh_sinh(np.array([4.0, 2.5, 2.5]), SQUARE)
        assert seen == [], f"low-sigma exponents must use epstein_zeta; saw {seen}"

    def test_the_two_paths_agree_where_they_overlap(self):
        """The switch must not be observable in the value."""
        nu_vec = np.array([9 * 2.5, 2.5, 2.5])      # the order-11 offender
        try:
            C.SIGMA_MAX_CIRCLE = 16.0
            C._ts_table_clear()
            direct = zeta_circle_tanh_sinh(nu_vec, SQUARE)
            C.SIGMA_MAX_CIRCLE = 1e9                # force epsteinlib
            C._ts_table_clear()
            library = zeta_circle_tanh_sinh(nu_vec, SQUARE)
        finally:
            C.SIGMA_MAX_CIRCLE = 16.0
            C._ts_table_clear()
        assert np.isfinite(direct) and np.isfinite(library)
        assert abs(direct - library) <= 1e-14 * abs(library)


class TestTheThreeDimensionalSheetSymmetry:
    r"""The 12 angular sheets collapse only when the LATTICE allows it.

    ``zeta_E(nu, A, 0, S y) = zeta_E(nu, A, 0, y)`` exactly when the signed
    permutation ``S`` preserves ``G = A^T A``.  Skipping sheets on that
    basis is an identity, not an approximation -- so the test that matters
    is the one where the symmetry does NOT hold and nothing may be skipped.
    """

    def setup_method(self):
        _ts_table_clear()

    def test_the_class_count_follows_the_cell(self):
        _, _ = C._ts_grid_3d()
        ops, _ = C._TS_GRID_3D_OPS[("3d", 48, 0.08)]
        counts = {
            name: len(C._ts_sheet_classes(A, ops))
            for name, A in (
                ("cubic", np.eye(3)),
                ("diagonal", np.diag([1.0, 2.0, 3.0])),
                ("sheared", np.array([[1.0, .3, .1], [0.0, 1.2, .2], [.1, 0.0, .9]])),
            )
        }
        # Cubic: every signed permutation preserves the identity Gram.
        # Diagonal: the sign flips do, the cyclic permutations do not.
        # Sheared: none do, so the reduction must be inert.
        assert counts["cubic"] == 1
        assert 1 < counts["diagonal"] < len(ops)
        assert counts["sheared"] == len(ops)

    @pytest.mark.parametrize("A", [
        np.eye(3),
        np.diag([1.0, 2.0, 3.0]),
        np.array([[1.0, .3, .1], [0.0, 1.2, .2], [.1, 0.0, .9]]),
    ])
    def test_the_table_matches_a_direct_evaluation(self, A):
        r"""Whatever the grouping decides, the table is the Epstein value.

        Includes the sheared cell, where no sheet may be skipped -- if the
        grouping ever became optimistic, that row is where it shows.
        """
        from epsteinlib import epstein_zeta
        ys, _ = C._ts_grid_3d()
        A_star = np.linalg.inv(A).T
        got = C._ts_zeta_table(4.0, A, A_star)
        idx = np.random.default_rng(1).choice(len(ys), 200, replace=False)
        want = np.array([
            epstein_zeta(4.0, A, np.zeros(3), A_star @ ys[j]) for j in idx
        ])
        assert np.max(np.abs(got[idx] - want)) / np.max(np.abs(want)) < 1e-14
