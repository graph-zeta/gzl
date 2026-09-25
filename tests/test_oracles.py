# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Self-checks of the brute-force oracles in :mod:`tests._oracles`.

The oracles gate the multi-kernel feature, so they are gated
themselves — against the references that already exist in the tree and
against closed forms that hold for compactly supported kernels.

Three families:

1. **Reproduction.**  With ``power_law_kernel`` on every edge, the
   general-kernel enumeration must reproduce the shipped power-law
   references: ``tests.test_finite_k_validation._direct_sum_finite_k``
   (finite k, box truncation) and the library's own box engine
   ``direct_sum_zero_momentum`` (k = 0).  These are float-reassociation
   comparisons — the same terms summed in a different order — so the
   tolerance is round-off, not physics.

2. **Closed forms for compact kernels.**  A compactly supported table
   makes the finite box *exact*, and a path factorises: the oracle's
   value is then an algebraic identity, not a numerical agreement.
   Where the table's values are dyadic the identity holds *bitwise*,
   because every partial sum is exactly representable and float
   addition is then associative on those operands.

3. **Torus cycle.**  ``cycle_torus_reference`` is checked twice: it is
   exact against the box enumeration for a compact kernel that cannot
   wrap, and it converges at the documented rate to
   ``gzl.zeta_circle`` for a power law.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import gzl.direct_sum as ds
from gzl import zeta_circle
from gzl.tensor_network import _balanced_z_axis

from tests._oracles import (
    box_enumeration,
    brute_force_zeta,
    cycle_torus_reference,
    mixed_kernel,
    power_law_kernel,
    table_kernel,
)
from tests.test_finite_k_validation import A_HEX, A_SQ, _direct_sum_finite_k


TRIANGLE = np.array([(0, 1), (1, 2), (0, 2)])
CYCLE3 = np.array([(0, 1), (1, 2), (2, 0)])
PATH2 = np.array([(0, 1), (1, 2)])
EDGE1 = np.array([(0, 1)])

A_1D = np.eye(1)

# A radius-1 even table whose values are dyadic rationals with
# denominator 8.  Every partial sum and every pairwise product that the
# enumeration forms is an exact multiple of 1/64, well inside the
# float64 mantissa, so the enumeration's result does not depend on the
# summation order and the closed forms below hold BITWISE.
TAB_1D = {(-1,): 0.375, (0,): 0.5, (1,): 0.375}
TAB_1D_SUM = 1.25            # 0.375 + 0.5 + 0.375, exact

# The same shape with an illustrative a(0) = 0.3.  The values are not
# dyadic, so only the exact-arithmetic identity is claimed here and the
# tolerance is a few ULP.
TAB_1D_NONDYADIC = {(-1,): 0.7, (0,): 0.3, (1,): 0.7}

# d = 2, radius 1 in the sup-norm, even, dyadic (denominator 16).
TAB_2D = dict(zip(
    itertools.product((-1, 0, 1), repeat=2),
    [0.0625, 0.125, 0.0625,
     0.125, 0.5, 0.125,
     0.0625, 0.125, 0.0625],
))
TAB_2D_SUM = 1.25            # exact


# ---------------------------------------------------------------------------
# 1.  Reproduction of the shipped power-law references
# ---------------------------------------------------------------------------

class TestReproducesFiniteKDirectSum:
    r"""``brute_force_zeta`` with power-law kernels IS
    ``_direct_sum_finite_k``.

    Same box, same pin, same phase convention; the only difference is
    that the shipped reference accumulates the edge product in log
    space while the oracle multiplies the kernel values directly.  That
    is a reassociation of identical terms, so the agreement is
    round-off (``rtol = 1e-14`` is ~50x above the observed 3e-16).
    """

    @pytest.mark.parametrize(
        "name, edges, A, nu, L, k_frac",
        [
            ("triangle d1 k=0", TRIANGLE, A_1D, [2.5, 3.0, 3.5], 6, [0.0]),
            ("triangle d1 k",   TRIANGLE, A_1D, [2.5, 3.0, 3.5], 6, [0.25]),
            ("path2 d1 k=0",    PATH2,    A_1D, [2.5, 3.5],      6, [0.0]),
            ("path2 d1 k",      PATH2,    A_1D, [2.5, 3.5],      6, [0.375]),
            ("triangle sq k=0", TRIANGLE, A_SQ, [3.5, 4.0, 4.5], 4, [0.0, 0.0]),
            ("triangle sq k",   TRIANGLE, A_SQ, [3.5, 4.0, 4.5], 4, [0.25, 0.125]),
            ("triangle hex k",  TRIANGLE, A_HEX, [3.5, 4.0, 4.5], 4, [0.25, 0.125]),
            ("path2 sq k",      PATH2,    A_SQ, [3.5, 4.5],      4, [0.25, 0.125]),
        ],
    )
    def test_matches_shipped_reference(self, name, edges, A, nu, L, k_frac):
        nu = np.asarray(nu, dtype=float)
        A = np.asarray(A, dtype=float)
        k_frac = np.asarray(k_frac, dtype=float)
        ref = _direct_sum_finite_k(edges, nu, A, L, 0, 2, k_frac)
        got = brute_force_zeta(
            edges, [power_law_kernel(x, A) for x in nu], A, L,
            source=0, terminal=2, k_frac=k_frac,
        )
        np.testing.assert_allclose(got, ref, rtol=1e-14, atol=0.0)

    def test_the_comparison_can_fail(self):
        """Liveness: perturbing one exponent moves the oracle well
        outside the 1e-14 band, so the agreements above are not
        vacuous."""
        nu = np.array([2.5, 3.0, 3.5])
        ref = _direct_sum_finite_k(TRIANGLE, nu, A_1D, 6, 0, 2,
                                   np.array([0.25]))
        got = brute_force_zeta(
            TRIANGLE,
            [power_law_kernel(x, A_1D) for x in (2.5, 3.0, 3.6)],
            A_1D, 6, source=0, terminal=2, k_frac=np.array([0.25]),
        )
        assert abs(got - ref) / abs(ref) > 1e-3


class TestBoxEnumerationIsTheBoxEngine:
    r"""``box_enumeration`` sums exactly the box the library's
    ``direct_sum`` engine sums.

    The pin is passed through as ``root`` because ``[-L, L]^d`` is not
    translation-closed: at finite ``L`` the value depends on which
    vertex is held at the origin, so an agreement here is only
    meaningful with the roots matched (this is the same reason
    ``tests/test_box_nu_inf.py`` passes ``root=0`` explicitly)."""

    @pytest.mark.parametrize(
        "edges, A, nu, L",
        [
            (PATH2,    A_1D,  [2.5, 3.5],      5),
            (TRIANGLE, A_1D,  [2.5, 3.0, 3.5], 5),
            (PATH2,    A_SQ,  [3.5, 4.5],      3),
            (PATH2,    A_HEX, [3.5, 4.5],      3),
            (TRIANGLE, A_SQ,  [3.5, 4.0, 4.5], 3),
        ],
    )
    def test_matches_direct_sum_zero_momentum(self, edges, A, nu, L):
        nu = np.asarray(nu, dtype=float)
        A = np.asarray(A, dtype=float)
        lib = complex(ds.direct_sum_zero_momentum(edges, nu, A, L, root=0)).real
        orc = box_enumeration(
            edges, [power_law_kernel(x, A) for x in nu], A, L, root=0,
        )
        np.testing.assert_allclose(orc, lib, rtol=1e-13, atol=0.0)

    def test_d4_two_path_twins_the_box_engines_own_reference(self):
        """``tests/test_box_nu_inf.py::test_d4_two_path_matches_brute_force``
        gates the box engine's d-guard lift with a hand-rolled
        ``itertools.product`` sum at d = 4, root = 0, L = 1.  The oracle
        must reproduce both sides of that check, which is what makes it
        a drop-in replacement for those hand-rolled references."""
        nu = np.array([2.5, 3.5])
        A = np.eye(4)
        lib = complex(ds.direct_sum_zero_momentum(
            PATH2, nu, A, 1, root=0)).real
        coords = np.array(list(itertools.product(range(-1, 2), repeat=4)),
                          dtype=float)
        hand = 0.0
        for x1 in coords:
            r1 = np.linalg.norm(x1)
            if r1 == 0.0:
                continue
            r2 = np.linalg.norm(coords - x1, axis=1)
            hand += r1 ** -2.5 * np.sum(r2[r2 > 0.0] ** -3.5)
        orc = box_enumeration(
            PATH2, [power_law_kernel(x, A) for x in nu], A, 1, root=0,
        )
        np.testing.assert_allclose(orc, hand, rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(orc, lib, rtol=1e-13, atol=0.0)


# ---------------------------------------------------------------------------
# 2.  Closed forms for compactly supported kernels
# ---------------------------------------------------------------------------

class TestCompactKernelClosedForms:
    r"""A compact table makes the finite box EXACT, and the closed
    forms are algebraic identities.

    *Single edge, pin 0.*  ``Σ_{m ∈ [-L,L]^d} a(m) = Σ_m a(m)`` once
    ``L ≥ R`` (the reach), because the box then contains the whole
    support.

    *2-path, pin the MIDDLE vertex 1.*  With ``m_1 = 0`` the two edge
    arguments are ``m_1 - m_0 = -m_0`` and ``m_2 - m_1 = m_2``, which
    are independent, so the sum factorises:

        ``Σ_{m_0} a(-m_0) · Σ_{m_2} a(m_2) = (Σ_m a(m))^2``

    (the two factors are equal because the box is symmetric under
    ``m → -m``, so this needs no evenness of ``a``).  Exact for
    ``L ≥ R``.

    *2-path, pin the END vertex 0.*  Now the arguments are ``m_1`` and
    ``m_2 - m_1``, which are coupled: the inner sum over ``m_2`` is
    complete only when ``L ≥ |m_1| + R`` for every ``m_1`` in the
    support, i.e. ``L ≥ 2R``.  Below that the value is short — which
    the test asserts too, so the bound is pinned rather than assumed.
    """

    @pytest.mark.parametrize("L", [1, 2, 3, 5])
    def test_single_edge_is_the_table_sum(self, L):
        got = brute_force_zeta(EDGE1, [table_kernel(TAB_1D)], A_1D, L, source=0)
        assert got == TAB_1D_SUM

    @pytest.mark.parametrize("L", [1, 2, 3, 5])
    def test_two_path_middle_pin_factorises(self, L):
        got = brute_force_zeta(
            PATH2, [table_kernel(TAB_1D)] * 2, A_1D, L, source=1,
        )
        assert got == TAB_1D_SUM ** 2

    @pytest.mark.parametrize("L", [2, 3, 5])
    def test_two_path_end_pin_needs_twice_the_reach(self, L):
        got = brute_force_zeta(
            PATH2, [table_kernel(TAB_1D)] * 2, A_1D, L, source=0,
        )
        assert got == TAB_1D_SUM ** 2

    def test_end_pin_is_short_below_twice_the_reach(self):
        """Liveness for the ``L >= 2R`` bound: at L = R = 1 the box
        clips the outer vertex and the value is strictly smaller."""
        got = brute_force_zeta(
            PATH2, [table_kernel(TAB_1D)] * 2, A_1D, 1, source=0,
        )
        assert got < TAB_1D_SUM ** 2

    @pytest.mark.parametrize("L", [1, 2, 3])
    def test_two_dimensional_table_factorises(self, L):
        got = brute_force_zeta(
            PATH2, [table_kernel(TAB_2D)] * 2, np.eye(2), L, source=1,
        )
        assert got == TAB_2D_SUM ** 2

    @pytest.mark.parametrize("L", [1, 2, 3])
    def test_nondyadic_table_matches_to_roundoff(self, L):
        """The non-dyadic table TAB_1D_NONDYADIC (a(0) = 0.3).  The
        identity is the same; only the float summation order separates
        the two sides, so this is a round-off comparison rather than
        the bitwise one above."""
        s = sum(TAB_1D_NONDYADIC.values())
        got = brute_force_zeta(
            PATH2, [table_kernel(TAB_1D_NONDYADIC)] * 2, A_1D, L, source=1,
        )
        np.testing.assert_allclose(got, s * s, rtol=1e-15, atol=0.0)


class TestOriginWeightIsHonoured:
    """``a(0) != 0`` is the whole point of the compact part: a
    configuration whose two endpoints coincide must be WEIGHTED, not
    dropped."""

    def test_origin_only_table_gives_exactly_a_of_zero(self):
        """A single edge whose kernel is supported only at the origin
        has exactly one surviving configuration (m_1 = 0), of weight
        a(0).  A power law would give 0 there."""
        got = brute_force_zeta(EDGE1, [table_kernel({(0,): 0.5})], A_1D, 4,
                               source=0)
        assert got == 0.5

    def test_power_law_drops_the_same_configuration(self):
        """The control: the regularised power law is exactly 0 at the
        origin, so the analogous single-edge sum contains no
        coincident-endpoint term."""
        k = power_law_kernel(3.5, A_1D)
        assert k(np.zeros((1, 1), dtype=int))[0] == 0.0

    def test_the_origin_weight_changes_the_value(self):
        """Zeroing a(0) must move the 2-path value by exactly the
        amount the factorised identity predicts."""
        tab_no0 = {k: v for k, v in TAB_1D.items() if k != (0,)}
        got = brute_force_zeta(
            PATH2, [table_kernel(tab_no0)] * 2, A_1D, 3, source=1,
        )
        assert got == (TAB_1D_SUM - TAB_1D[(0,)]) ** 2
        assert got != TAB_1D_SUM ** 2


class TestMixedKernel:
    def test_value_at_the_origin_is_the_table_alone(self):
        """Every power law vanishes at the origin, so V(0) = a(0)."""
        V = mixed_kernel(TAB_1D, [1.5, -0.25], [2.5, 4.0], A_1D)
        assert V(np.zeros((1, 1), dtype=int))[0] == TAB_1D[(0,)]

    def test_is_the_pointwise_sum_of_its_parts(self):
        dz = np.arange(-4, 5, dtype=int).reshape(-1, 1)
        a = table_kernel(TAB_1D)
        p1 = power_law_kernel(2.5, A_1D)
        p2 = power_law_kernel(4.0, A_1D)
        V = mixed_kernel(TAB_1D, [1.5, -0.25], [2.5, 4.0], A_1D)
        np.testing.assert_array_equal(
            V(dz), a(dz) + 1.5 * p1(dz) - 0.25 * p2(dz),
        )

    def test_enumeration_is_multilinear_in_the_kernel_terms(self):
        r"""The strong check on the mixed kernel inside the sum: the
        product over edges of ``(a + b K)`` expands into the 2^E sums
        over pure kernels, so

            ``Z[a + bK, a + bK] = Z[a,a] + b Z[a,K] + b Z[K,a]
                                  + b^2 Z[K,K]``

        term by term.  This is an exact algebraic identity of the
        enumeration; only float reassociation separates the sides."""
        b, nu, L = 1.5, 3.5, 4
        a = table_kernel(TAB_1D)
        K = power_law_kernel(nu, A_1D)
        V = mixed_kernel(TAB_1D, [b], [nu], A_1D)
        full = brute_force_zeta(PATH2, [V, V], A_1D, L, source=0)
        expanded = 0.0
        for f0, c0 in ((a, 1.0), (K, b)):
            for f1, c1 in ((a, 1.0), (K, b)):
                expanded += c0 * c1 * brute_force_zeta(
                    PATH2, [f0, f1], A_1D, L, source=0,
                )
        np.testing.assert_allclose(full, expanded, rtol=1e-13, atol=0.0)


# ---------------------------------------------------------------------------
# 3.  The torus cycle reference
# ---------------------------------------------------------------------------

def _grid_kernel(fn, n: int, d: int) -> np.ndarray:
    """Sample ``fn`` on the balanced ``n^d`` label grid, index 0 = origin."""
    z = _balanced_z_axis(n)
    mesh = np.meshgrid(*[z] * d, indexing="ij")
    return fn(np.stack(mesh, axis=-1))


class TestCycleTorusReference:

    @pytest.mark.parametrize("n", [5, 6, 8, 12])
    def test_compact_kernel_is_exact_against_the_box(self, n):
        r"""For a compact kernel of reach R = 1 the torus truncation is
        EXACT once no closed walk can wind: an ``E``-cycle's total
        displacement is at most ``E R``, so ``n > E R = 3`` forbids an
        alias.  The box enumeration at ``L = 4 >= 2R`` is the same
        (exact) infinite-lattice value, and the two must agree to FFT
        round-off."""
        V = _grid_kernel(table_kernel(TAB_1D), n, 1)
        torus = cycle_torus_reference(V, 3)
        box = box_enumeration(CYCLE3, [table_kernel(TAB_1D)] * 3, A_1D, 4,
                              root=0)
        np.testing.assert_allclose(torus, box, rtol=1e-13, atol=0.0)

    def test_the_exactness_bound_is_n_greater_than_E_times_R(self):
        r"""The docstring's bound is ``n > E R``, not ``n > 2R``.  At
        ``E = 3, R = 1`` the ``n = 3`` torus satisfies ``n > 2R`` (no
        single displacement wraps) yet counts the two three-step walks
        ``(+1, +1, +1)`` and ``(-1, -1, -1)`` that wind once, so it is
        off by exactly ``2 * 0.375**3``; ``n = 4 = E R + 1`` is exact."""
        box = box_enumeration(CYCLE3, [table_kernel(TAB_1D)] * 3, A_1D, 4,
                              root=0)
        aliased = cycle_torus_reference(_grid_kernel(table_kernel(TAB_1D), 3, 1), 3)
        assert aliased != pytest.approx(box, rel=1e-6)
        np.testing.assert_allclose(aliased - box, 2 * TAB_1D[(1,)] ** 3,
                                   rtol=1e-12, atol=0.0)
        exact = cycle_torus_reference(_grid_kernel(table_kernel(TAB_1D), 4, 1), 3)
        np.testing.assert_allclose(exact, box, rtol=1e-13, atol=0.0)

    def test_power_law_matches_zeta_circle(self):
        r"""ν = 4.5, d = 2, E = 3 against the closed-form cycle zeta.
        The torus truncation converges like ``n^(d - 2ν) = n^-7`` (the
        minimum cut isolating one free vertex of a cycle is two edges),
        so ``n = 256`` sits far below the 1e-12 band."""
        A = np.eye(2)
        V = _grid_kernel(power_law_kernel(4.5, A), 256, 2)
        got = cycle_torus_reference(V, 3)
        ref = zeta_circle([4.5] * 3, A)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)

    def test_the_documented_convergence_rate(self):
        r"""Pin the ``n^-7`` claim in the docstring rather than assume
        it: doubling n must cut the error by ~2^7."""
        A = np.eye(2)
        ref = zeta_circle([4.5] * 3, A)
        errs = [
            abs(cycle_torus_reference(_grid_kernel(power_law_kernel(4.5, A),
                                                   n, 2), 3) - ref)
            for n in (64, 128)
        ]
        ratio = errs[0] / errs[1]
        assert 2 ** 6 < ratio < 2 ** 8, f"ratio {ratio} is not ~2^7"

    def test_refuses_a_non_cube_grid(self):
        with pytest.raises(ValueError, match="cube"):
            cycle_torus_reference(np.zeros((4, 5)), 3)

    def test_refuses_a_degenerate_cycle(self):
        with pytest.raises(ValueError, match="E >= 2"):
            cycle_torus_reference(np.zeros((4,)), 1)


# ---------------------------------------------------------------------------
# Builders and guards
# ---------------------------------------------------------------------------

class TestKernelBuilders:
    def test_power_law_uses_the_metric(self):
        """The kernel argument is a LABEL difference; the metric enters
        through A.  On a doubled cell the same label is twice as far."""
        dz = np.array([[2]])
        assert power_law_kernel(3.0, A_1D)(dz)[0] == 2.0 ** -3.0
        assert power_law_kernel(3.0, 2.0 * A_1D)(dz)[0] == 4.0 ** -3.0

    def test_power_law_on_a_sheared_cell(self):
        dz = np.array([[1, 1]])
        r = np.linalg.norm(A_HEX @ np.array([1.0, 1.0]))
        assert power_law_kernel(2.5, A_HEX)(dz)[0] == r ** -2.5

    def test_table_is_zero_off_support(self):
        a = table_kernel(TAB_1D)
        dz = np.array([[-2], [-1], [0], [1], [2]])
        np.testing.assert_array_equal(a(dz), [0.0, 0.375, 0.5, 0.375, 0.0])

    def test_table_refuses_ragged_keys(self):
        with pytest.raises(ValueError, match="same length"):
            table_kernel({(0,): 1.0, (0, 1): 2.0})

    def test_table_refuses_a_dimension_mismatch(self):
        a = table_kernel(TAB_1D)
        with pytest.raises(ValueError, match="expected d = 1"):
            a(np.zeros((3, 2), dtype=int))

    def test_mixed_refuses_mismatched_b_and_nu(self):
        with pytest.raises(ValueError, match="len"):
            mixed_kernel(TAB_1D, [1.0, 2.0], [3.0], A_1D)


class TestGuards:
    def test_refuses_an_oversized_enumeration(self):
        with pytest.raises(ValueError, match="blow memory"):
            brute_force_zeta(
                TRIANGLE, [power_law_kernel(3.5, np.eye(3))] * 3,
                np.eye(3), 12,
            )

    def test_refuses_a_kernel_count_mismatch(self):
        with pytest.raises(ValueError, match="kernel functions"):
            brute_force_zeta(TRIANGLE, [power_law_kernel(3.5, A_1D)] * 2,
                             A_1D, 2)

    def test_refuses_a_half_specified_phase(self):
        fns = [power_law_kernel(3.5, A_1D)] * 2
        with pytest.raises(ValueError, match="both"):
            brute_force_zeta(PATH2, fns, A_1D, 2, terminal=2)
        with pytest.raises(ValueError, match="both"):
            brute_force_zeta(PATH2, fns, A_1D, 2, k_frac=[0.25])

    def test_refuses_an_out_of_range_pin(self):
        fns = [power_law_kernel(3.5, A_1D)] * 2
        with pytest.raises(ValueError, match="source"):
            brute_force_zeta(PATH2, fns, A_1D, 2, source=7)
        with pytest.raises(ValueError, match="terminal"):
            brute_force_zeta(PATH2, fns, A_1D, 2, terminal=7,
                             k_frac=[0.25])

    def test_refuses_a_non_square_lattice(self):
        with pytest.raises(ValueError, match="square"):
            brute_force_zeta(EDGE1, [power_law_kernel(3.5, A_1D)],
                             np.zeros((1, 2)), 2)

    def test_zero_momentum_phase_is_a_no_op(self):
        """k = 0 must reproduce the vacuum sum, which is what lets the
        finite-k parametrisations above double as k = 0 tests.  The
        phase factor is exactly ``1 + 0j`` at every configuration, so
        the two differ only in that one reduction is real and the other
        complex — numpy associates those two reductions differently, so
        this is round-off (measured 1.6e-16), not bitwise."""
        fns = [power_law_kernel(x, A_1D) for x in (2.5, 3.5)]
        vac = brute_force_zeta(PATH2, fns, A_1D, 5, source=0)
        at0 = brute_force_zeta(PATH2, fns, A_1D, 5, source=0, terminal=2,
                               k_frac=[0.0])
        np.testing.assert_allclose(at0, vac, rtol=1e-15, atol=0.0)
