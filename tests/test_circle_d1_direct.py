# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
r"""The d = 1 cycle at large exponents: a certified positive lattice sum.

The quadrature integrates ``prod_i Z(nu_i, y)`` over the Brillouin zone, and
at large exponents that integrand cancels (``Z ~ 2 cos 2 pi y``): the
(4, 4, 4)-bundle triangle at nu = 11, exponents 44, came out 1.5e-3 relative
off (value 3.4e-13 against an integrand of order 8).  ``_cycle_1d_direct``
sums ``prod |d_i|^-nu_i`` over compositions of zero with ``|d_i| <= M``
instead -- positive terms only -- and accepts the sum once a rigorous tail
bound is below ``2^-60`` of it.

References: 34-digit mpmath evaluations of ``2 int_0^{1/2} prod_i 2
clcos(nu_i, 2 pi y) dy`` (tanh-sinh, estimated error below 1e-42), which
agree with brute-force real-space sums of the triangles to 1e-29 or better.
"""
from __future__ import annotations

import numpy as np
import pytest

from gzl import zeta_circle
from gzl.circle import _cycle_1d_direct, _cycle_1d_partial

UNIT = np.eye(1)

REF = {
    (44.0, 44.0, 44.0): 3.410605131648480892188323e-13,
    (33.0, 44.0, 44.0): 2.330580173293128610071033e-10,
    (33.0, 33.0, 33.0): 6.984919309616091380173004e-10,
    (24.0, 24.0, 24.0): 3.576278686548762605471288e-7,
    (8.0, 8.0, 8.0): 0.02344467430256788005055522,
    (11.0,) * 13: 10.05835757580594200906101,
}


@pytest.mark.parametrize("nus", sorted(REF))
def test_large_exponent_cycles_match_the_reference(nus):
    got = zeta_circle(list(nus), UNIT)
    assert abs(got - REF[nus]) / REF[nus] < 1e-14


@pytest.mark.parametrize("nus", sorted(REF))
def test_the_direct_sum_certifies_them(nus):
    assert _cycle_1d_direct(list(nus)) is not None


@pytest.mark.parametrize("nus", [[1.1, 1.1, 1.1], [1.5] * 5, [1.25, 3.0, 3.0]])
def test_slow_tails_keep_the_quadrature(nus):
    """Small exponents cannot certify within the truncation cap."""
    assert _cycle_1d_direct(nus) is None


def test_partial_sum_counts_compositions_of_zero():
    """``C_M`` by hand: the triangle at M = 2 has the compositions of zero
    with entries in {-2, -1, 1, 2}: (1, 1, -2) and permutations and signs,
    six tuples, each with one entry of modulus 2."""
    nu = 5.0
    want = 6 * 2.0 ** -nu
    assert abs(_cycle_1d_partial(np.array([nu] * 3), 2) - want) < 1e-16 * want
    # An odd cycle of +-1 steps cannot close: the M = 1 sum is empty.
    assert _cycle_1d_partial(np.array([nu] * 3), 1) == 0.0


def test_direct_and_quadrature_agree_where_both_are_accurate():
    """At moderate exponents both paths are accurate; they must agree."""
    from gzl.circle import int_full_1d, epstein_zeta_prod
    nus = [6.0, 6.0, 7.0]
    direct = _cycle_1d_direct(nus)
    assert direct is not None
    A = np.eye(1)
    quad = int_full_1d(lambda y: epstein_zeta_prod(nus, A, A, np.array([y])))
    assert abs(direct - quad) / direct < 1e-12
