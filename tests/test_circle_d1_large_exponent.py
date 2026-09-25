# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
r"""The d = 1 cycle closed form at large exponents: finite, and never a
silent NaN.

``epsteinlib`` returns ``nan+nanj`` at large ``nu`` and small ``|y|``, and
the adaptive d = 1 rule samples there: ``zeta_circle([121, 11, 11],
eye(1))`` was NaN, and with it ``c_13`` of the 0qp chain series at
``nu = 11`` (a multiplicity-11 bundle beside two single edges).  The fix
sums the Epstein series directly at exactly those nodes.

Oracle: the three-cycle in REAL space, independent of any Epstein
evaluation,

    Z(a, b, c) = sum_{x, y != 0, x != y} |x|^-a |y|^-b |x - y|^-c,

with the large exponent on the outer sum (``|x| <= 4``; the dropped tail is
below ``2 * 4^(1 - a) / (a - 1) * 2 zeta(b)``, 3e-43 at ``a = 70``) and the
inner sum exact: explicit for ``|y| <= 40`` plus both tails as convergent
Hurwitz-zeta series.  Agrees with a 50-digit mpmath evaluation of the same
sum to round-off.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import zeta as hurwitz

from gzl import GraphZetaError, Interaction, evaluate_graph, zeta_circle
from gzl import circle as C

A_CHAIN = np.eye(1)


def _inner(x, b, c, N=40):
    """``sum_{y != 0, x} |y|^-b |x - y|^-c``: explicit for ``|y| <= N``, and
    ``sum_{y > N} y^-b (y -+ x)^-c = sum_k (c)_k / k! (+-x)^k
    zeta(b + c + k, N + 1)`` for the two tails (``N >= 4 |x|``)."""
    terms = [abs(y) ** -b * abs(x - y) ** -c
             for y in range(-N, N + 1) if y not in (0, x)]
    for sgn in (1, -1):
        coef, k = 1.0, 0
        while True:
            t = coef * (sgn * x) ** k * hurwitz(b + c + k, N + 1)
            terms.append(t)
            if k > 5 and abs(t) < 1e-40:
                break
            coef *= (c + k) / (k + 1)
            k += 1
    return math.fsum(terms)


def ring3(w, b, c, N=40):
    """The real-space three-cycle with kernel ``w`` (a dict ``x -> w(x)``)
    on one edge and ``|.|^-b``, ``|.|^-c`` on the other two."""
    return math.fsum(wx * _inner(x, b, c, N) for x, wx in w.items())


def power(a, X=4):
    return {x: abs(x) ** -float(a) for x in range(-X, X + 1) if x}


FRONTIER = [[121, 11, 11], [70, 1.5, 1.5], [115, 9, 9], [200, 2.5, 2.5],
            [400, 11, 11]]
PREVIOUSLY_FINITE = [[2.5] * 3, [5] * 3, [11] * 3, [30, 5, 5],
                     [60, 1.5, 1.5], [69, 1.5, 1.5]]


def test_the_oracle_reproduces_a_cycle_the_old_rule_handled():
    """Convention check of the oracle itself: ``[5, 4, 3]`` needs a wide
    outer window (tail bound 1.5e-9 relative at ``|x| <= 200``)."""
    ref = ring3(power(5, X=200), 4, 3, N=800)
    assert zeta_circle([5, 4, 3], A_CHAIN) == pytest.approx(ref, rel=1e-8)


@pytest.mark.parametrize("nu", FRONTIER)
def test_frontier_cycle_is_finite_and_matches_the_real_space_sum(nu):
    a, b, c = nu
    got = zeta_circle(nu, A_CHAIN)
    assert math.isfinite(got)
    assert got == pytest.approx(ring3(power(a), b, c), rel=1e-12)


@pytest.mark.parametrize("nu", PREVIOUSLY_FINITE)
def test_the_direct_sum_runs_only_where_epsteinlib_failed(nu, monkeypatch):
    """Bit-identity by construction: where the library was finite the
    substitution is never called, so the arithmetic is the old one."""
    calls = []
    real = C._epstein_zeta_1d_direct
    monkeypatch.setattr(C, "_epstein_zeta_1d_direct",
                        lambda *a: calls.append(a) or real(*a))
    assert math.isfinite(zeta_circle(nu, A_CHAIN))
    assert calls == []
    # ... and it is what fixes the frontier on the quadrature path.  A cycle
    # this large is normally taken by the certified lattice sum
    # (``_cycle_1d_direct``) before any quadrature; declining it here
    # exercises the retry, which a large-max / small-min cycle still needs.
    monkeypatch.setattr(C, "_cycle_1d_direct", lambda *a: None)
    assert math.isfinite(zeta_circle([121, 11, 11], A_CHAIN)) and calls


def test_direct_sum_convention_matches_epsteinlib():
    """``zeta_E(nu, A, 0, A* y)``, sign and scale, on the chain and two
    scaled cells, where the library is finite (``y = 1/4`` is left out:
    the leading term cancels there and both sides are round-off of it)."""
    from epsteinlib import epstein_zeta
    for a in (1.0, 1.7, 0.6):
        A = np.array([[a]])
        for nu in (5.5, 8.0, 11.0):
            for y in (0.5, 0.37, 0.1, 0.013, -0.2, 1e-3):
                ref = epstein_zeta(nu, A, np.zeros(1), np.array([y / a])).real
                got = C._epstein_zeta_1d_direct(nu, A, y)
                assert got == pytest.approx(ref, rel=1e-14)


def test_kernel_path_shares_the_fix():
    P = [Interaction.power_law(nu, b=2.0) for nu in (121.0, 11.0, 11.0)]
    ref = ring3(power(121), 11, 11)
    assert zeta_circle(P, A_CHAIN) == pytest.approx(8.0 * ref, rel=1e-12)
    # a compact part beside the large tail: V(+-1) = 0.5 + 1
    V = Interaction.from_table({(1,): 0.5, (-1,): 0.5}, b=[1.0], nu=[121.0])
    w = {x: (0.5 if abs(x) == 1 else 0.0) + abs(x) ** -121.0
         for x in range(-4, 5) if x}
    assert zeta_circle([V, P[1], P[2]], A_CHAIN) == pytest.approx(
        4.0 * ring3(w, 11, 11), rel=1e-12)


def test_evaluate_graph_on_the_corpus_bundle():
    """The 0qp corpus shape that made ``c_13`` NaN at ``nu = 11``."""
    E = np.array([[0, 1]] * 11 + [[0, 2], [1, 2]])
    val, info = evaluate_graph(E, 11.0, A_CHAIN, n_points=16,
                               return_diagnostics=True)
    assert info["n_simple_cycles"] == 1
    assert val == pytest.approx(ring3(power(121), 11, 11), rel=1e-12)


class TestNonFiniteRaises:
    def test_plain_path(self, monkeypatch):
        monkeypatch.setattr(C, "_cycle_1d_direct", lambda *a: None)
        monkeypatch.setattr(C, "_epstein_zeta_1d_direct", lambda *a: math.nan)
        with pytest.raises(C.CycleQuadratureError, match="d = 1 cycle integral"):
            zeta_circle([121, 11, 11], A_CHAIN)
        assert issubclass(C.CycleQuadratureError, GraphZetaError)

    def test_every_node_non_finite(self, monkeypatch):
        monkeypatch.setattr(C, "epstein_zeta", lambda *a: complex(math.nan, math.nan))
        monkeypatch.setattr(C, "_epstein_zeta_1d_direct", lambda *a: math.nan)
        with pytest.raises(C.CycleQuadratureError):
            zeta_circle([2.5, 2.5, 2.5], A_CHAIN)
        with pytest.raises(C.CycleQuadratureError):
            zeta_circle([Interaction.power_law(2.5, b=2.0)] * 3, A_CHAIN)

    def test_kernel_path(self, monkeypatch):
        monkeypatch.setattr(C, "_epstein_zeta_1d_direct", lambda *a: math.nan)
        P = [Interaction.power_law(nu, b=2.0) for nu in (121.0, 11.0, 11.0)]
        with pytest.raises(C.CycleQuadratureError):
            zeta_circle(P, A_CHAIN)

    def test_front_end_falls_back_to_the_sigma_router(self, monkeypatch):
        """A declined plain cycle takes the sigma-router, as the Interaction
        path already did, and is not cached under the closed-form key."""
        monkeypatch.setattr(C, "_cycle_1d_direct", lambda *a: None)
        monkeypatch.setattr(C, "_epstein_zeta_1d_direct", lambda *a: math.nan)
        E = np.array([[0, 1]] * 11 + [[0, 2], [1, 2]])
        cache = {}
        val, info = evaluate_graph(E, 11.0, A_CHAIN, n_points=16,
                                   return_diagnostics=True, block_cache=cache)
        assert info["n_block_cycle_fallback"] == 1 and info["n_simple_cycles"] == 0
        assert not any(k[0] == "cycle" for k in cache)
        assert val == evaluate_graph(E, 11.0, A_CHAIN, n_points=16, fast_cycles=True)
        assert val == pytest.approx(ring3(power(121), 11, 11), rel=1e-12)
