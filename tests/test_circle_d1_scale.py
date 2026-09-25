# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
r"""The d = 1 cycle closed form is accurate at every scale of its integrand.

``int_full_1d`` integrates with an ABSOLUTE tolerance sized for an
integrand of order one.  Two things make a d = 1 cycle integrand small, and
both used to cost digits:

* a stretched chain ``A = (a)``: every term carries ``|a|^-nu_e``, so the
  integrand is ``|a|^-sum(nu)`` times the unit chain's --
  ``[69, 1.5, 1.5]`` on ``(1.7)`` came out 4.7e-4 relative off, and
  ``[100, 2.5, 2.5]`` 7.4e-7.  A power-law cycle is now evaluated on the
  unit chain and rescaled exactly (``_zeta_circle_1d_rescaled``);
* small couplings in an Interaction kernel: ``b = 1e-6`` on a triangle
  was 5.7e-9 off.  The kernel path now sizes its tolerance by a sup-norm
  bound of the integrand, and only ever tightens it, so every kernel of
  scale >= 1 keeps its historical value bit for bit.

The unit-chain power-law path is untouched.

References: the unit-chain constants below are 30-digit mpmath
evaluations of ``2 int_0^{1/2} prod_e 2 clcos(nu_e, 2 pi y) dy`` (tanh-sinh,
estimated error below 1e-42), cross-checked against brute-force real-space
sums ``sum_{d_1 + d_2 + d_3 = 0, d_i != 0} prod |d_i|^-nu_i`` to 30 digits
where those converge (all three exponents >= 11).  Everything else is an
exact identity -- scale covariance in ``a`` and homogeneity in the kernel's
couplings -- so no mpmath is needed at test time.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from gzl import GraphZetaError, Interaction, zeta_circle
from gzl.circle import CycleQuadratureError

UNIT = np.eye(1)

#: 30-digit unit-chain references (see the module docstring).
REF_UNIT = {
    (69.0, 1.5, 1.5): 1.907129017543924104895,
    (100.0, 2.5, 2.5): 0.7642968823386050566208,
    (5.0, 4.0, 3.0): 0.4511013653364743098646,
    (2.5, 2.5, 2.5): 1.242623505678824891372,
}


def _rel(x, ref):
    return abs(x - ref) / abs(ref)


@pytest.mark.parametrize("nus", sorted(REF_UNIT))
@pytest.mark.parametrize("a", [1.3, 1.7, 2.2, 0.6, 0.35])
def test_stretched_chain_matches_the_reference(nus, a):
    ref = REF_UNIT[nus] * math.prod(a ** -nu for nu in nus)
    assert _rel(zeta_circle(list(nus), [[a]]), ref) < 1e-14


@pytest.mark.parametrize("nus", sorted(REF_UNIT))
def test_unit_chain_matches_the_reference(nus):
    assert _rel(zeta_circle(list(nus), UNIT), REF_UNIT[nus]) < 1e-14


@pytest.mark.parametrize("nus", [[1.1, 1.1, 1.1], [1.5] * 4, [3.0, 3.0, 6.0],
                                 [2.2] * 7, [121.0, 11.0, 11.0]])
@pytest.mark.parametrize("a", [1.7, 0.6, -1.3])
def test_scale_covariance_is_exact(nus, a):
    """``zeta_circle(nu, (a)) = prod_e |a|^-nu_e zeta_circle(nu, (1))``, to
    the handful of roundings of the per-edge factors."""
    want = zeta_circle(nus, UNIT) * math.prod(abs(a) ** -nu for nu in nus)
    assert _rel(zeta_circle(nus, [[a]]), want) < 1e-15 * len(nus)


def test_out_of_range_value_raises_instead_of_inf():
    with pytest.raises(CycleQuadratureError):
        zeta_circle([400.0, 11.0, 11.0], [[0.01]])
    assert issubclass(CycleQuadratureError, GraphZetaError)


@pytest.mark.parametrize("b", [1e-6, 1e-3])
@pytest.mark.parametrize("a", [1.0, 1.7])
def test_small_coupling_kernel_is_homogeneous(b, a):
    """A triangle of ``b |x|^-3`` edges is ``b^3`` times the plain cycle --
    the power-law path, which never saw the kernel tolerance."""
    V = Interaction(b=[b], nu=[3.0])
    want = b ** 3 * zeta_circle([3.0, 3.0, 3.0], [[a]])
    assert _rel(zeta_circle([V, V, V], [[a]]), want) < 1e-13


@pytest.mark.parametrize("E", [3, 4])
@pytest.mark.parametrize("a", [1.0, 1.7])
def test_compact_kernel_is_homogeneous_in_its_couplings(E, a):
    """Scaling every coupling of a table-plus-tail kernel by ``s`` scales an
    E-cycle by ``s^E``; at ``s = 1e-4`` the historical absolute tolerance
    left 1e-10..1e-9 of that relative error."""
    def kernel(s):
        table = {(1,): 0.7 * s, (-1,): 0.7 * s, (2,): 0.2 * s, (-2,): 0.2 * s}
        return Interaction.from_table(table, b=[0.3 * s], nu=[3.0])
    s = 1e-4
    big = zeta_circle([kernel(1.0)] * E, [[a]])
    small = zeta_circle([kernel(s)] * E, [[a]])
    assert _rel(small, s ** E * big) < 1e-13
