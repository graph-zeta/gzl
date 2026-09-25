# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The finite-momentum box tail oscillates; its Richardson basis must too.

``direct_sum_zero_momentum`` weights the summand by
``cos(2π k · n_terminal)``, so the box faces at ``n_i = ±L`` make the
truncation tail oscillate with frequency ``k_i`` in ``L``.  Fitting pure
powers through that injects error larger than the truncation it removes.

A bridge at finite k has an exact closed form -- the momentum-space
Epstein zeta -- so these assertions carry no reference uncertainty.
"""
import numpy as np
import pytest
from epsteinlib import epstein_zeta

from gzl import direct_sum_extrapolated, direct_sum_zero_momentum

BRIDGE = np.array([(0, 1)])


def _exact(nu, A, k):
    Astar = np.linalg.inv(A).T
    return float(np.real(epstein_zeta(nu, A, np.zeros(A.shape[0]), Astar @ k)))


def _both(d, L_list, kv):
    A = np.eye(d)
    nu = d + 0.75
    nv = np.array([nu])
    k = np.array([kv] + [0.0] * (d - 1))
    exact = _exact(nu, A, k)
    raw = float(np.real(direct_sum_zero_momentum(
        BRIDGE, nv, A, max(L_list), root=0, momentum=k, terminal=1)))
    ext = float(np.real(direct_sum_extrapolated(
        BRIDGE, nv, A, L_list=L_list, root=0, momentum=k, terminal=1,
        n_correction_terms=3)))
    return (abs(raw - exact) / abs(exact), abs(ext - exact) / abs(exact))


class TestFiniteKBasis:

    @pytest.mark.parametrize("d,L_list,kv", [
        (1, (4, 5, 6, 7, 8), 0.5),
        (1, (4, 5, 6, 7, 8), 0.25),
        (1, (4, 5, 6, 7, 8), 0.125),
        (1, tuple(range(6, 25)), 0.5),
        (1, tuple(range(6, 25)), 0.25),
        (2, (3, 4, 5, 6), 0.5),
        (2, tuple(range(4, 13)), 0.5),
        (2, tuple(range(4, 13)), 0.25),
    ])
    def test_never_worse_than_the_raw_box(self, d, L_list, kv):
        """Extrapolating must not be worse than not extrapolating."""
        raw, ext = _both(d, L_list, kv)
        assert ext <= raw * 1.05, (ext, raw)

    @pytest.mark.parametrize("d,kv", [(1, 0.5), (1, 0.25), (2, 0.5)])
    def test_long_ladder_gains_orders_of_magnitude(self, d, kv):
        L_list = tuple(range(6, 25)) if d == 1 else tuple(range(4, 13))
        raw, ext = _both(d, L_list, kv)
        assert ext < raw / 100.0, (ext, raw)

    def test_short_ladder_falls_back_rather_than_guessing(self):
        """Below one oscillation period the fit is declined, not faked."""
        # k=0.125 has period 8; L=4..8 spans 4 < 8, so the gate fires and
        # the raw box value is returned unchanged.
        raw, ext = _both(1, (4, 5, 6, 7, 8), 0.125)
        assert ext == pytest.approx(raw, rel=1e-12)

    def test_zero_momentum_path_is_unchanged(self):
        """k=0 must still use the plain Euler-Maclaurin power basis."""
        A = np.eye(1)
        nv = np.array([1.75])
        exact = _exact(1.75, A, np.zeros(1))
        ext = float(np.real(direct_sum_extrapolated(
            BRIDGE, nv, A, L_list=(6, 8, 10, 12, 14), n_correction_terms=3)))
        assert abs(ext - exact) / abs(exact) < 1e-4
