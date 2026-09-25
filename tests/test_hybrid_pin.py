# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""The pin ``hybrid_zeta`` eliminates around, on the vacuum path.

On the torus the pin is FREE — translation invariance makes ``zeta_B``
independent of which vertex sits at the origin — so it changes only cost.
It changes it by a full power of ``n^d``.

``hybrid_zeta`` had always pinned the caller's ``source`` (default 0) and
never asked ``_elimination.plan``, which computes the cost-optimal pin and
was already computing it for the ORDER.  On K5-minus-an-edge with the
terminals at the missing edge's endpoints — a real corpus block, and the
one that made the d = 2 order-9 pass 21x the box — that is contraction
exponent 3 at vertex 0 against 2 at vertex 1: 33.0 s / 27.5 GB against
0.065 s / 0.20 GB at ``n_points = 32``, agreeing to 1.55e-16.

WHAT THESE TESTS PROTECT.  Three things, and the first two are
preconditions the fix silently depends on:

* ``plan(pin=None)`` really does return the argmin-exponent pin
  (measured: 763/763 corpus blocks, 63/63 random connected graphs);
* it does so DETERMINISTICALLY — same pin under shuffled edge order and
  under different ``PYTHONHASHSEED`` — because the block cache keys on
  topology, so a pin that wandered would serve two different values for
  one key;
* the re-pin is value-exact, and is scoped to the vacuum path only.
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl import _elimination as EL
from gzl import hybrid as H
from gzl.hybrid import hybrid_zeta

# K5 minus the edge (0, 4); terminals are the missing edge's endpoints.
K5E = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
K5E_V = [0, 1, 2, 3, 4]
NU = [2.5] * 9
SQUARE = np.eye(2)


class TestThePlannerPinIsWellDefined:

    def test_it_attains_the_minimum_exponent(self):
        best = min(EL.plan(K5E, K5E_V, pin=p).exponent for p in K5E_V)
        assert EL.plan(K5E, K5E_V).exponent == best
        # And the caller's default pin is genuinely worse — otherwise this
        # whole test module is gating a no-op.
        assert EL.plan(K5E, K5E_V, pin=0).exponent > best

    def test_it_is_stable_under_edge_reordering(self):
        rng = np.random.default_rng(7)
        ref = EL.plan(K5E, K5E_V)
        for _ in range(20):
            shuffled = [K5E[i] for i in rng.permutation(len(K5E))]
            got = EL.plan(shuffled, K5E_V)
            assert (got.pin, got.exponent) == (ref.pin, ref.exponent)

    def test_helper_falls_back_to_the_caller_on_refusal(self):
        # A planner that raises must not fail the evaluation.
        assert H._planner_pin([], [], 3) == 3

    def test_the_kill_switch_restores_the_callers_pin(self, monkeypatch):
        monkeypatch.setattr(H, "USE_PLANNER_PIN", False)
        assert H._planner_pin(K5E, NU, 0) == 0
        monkeypatch.setattr(H, "USE_PLANNER_PIN", True)
        assert H._planner_pin(K5E, NU, 0) == EL.plan(K5E, K5E_V).pin


class TestTheRePinIsValueExact:

    @pytest.mark.parametrize("n", [12, 16, 20])
    def test_vacuum_value_is_unchanged_by_the_pin(self, n, monkeypatch):
        monkeypatch.setattr(H, "USE_PLANNER_PIN", True)
        planner = hybrid_zeta(K5E, NU, SQUARE, n_points=n)
        monkeypatch.setattr(H, "USE_PLANNER_PIN", False)
        caller = hybrid_zeta(K5E, NU, SQUARE, n_points=n)
        assert np.isfinite(planner) and np.isfinite(caller)
        assert abs(planner - caller) <= 1e-13 * abs(caller)

    @pytest.mark.parametrize("n", [12, 16])
    def test_zero_momentum_equals_the_vacuum_value(self, n):
        """zeta_B(k=0) IS the vacuum value; that identity is what frees the pin."""
        at_zero = hybrid_zeta(K5E, NU, SQUARE, n_points=n, source=0,
                              terminal=4, momentum=np.zeros(2))
        vac = hybrid_zeta(K5E, NU, SQUARE, n_points=n)
        assert abs(at_zero - vac) <= 1e-13 * abs(vac)

    def test_it_is_cheaper(self):
        """The point of the change, gated as an EXPONENT not a wall clock."""
        pin = EL.plan(K5E, K5E_V).pin
        assert EL.plan(K5E, K5E_V, pin=pin).exponent < \
               EL.plan(K5E, K5E_V, pin=0).exponent


class TestItIsScopedToTheVacuumPath:

    def test_a_finite_k_call_keeps_the_callers_pin(self, monkeypatch):
        """At finite k the kept indices are what the caller asked for."""
        seen = []
        real = H._planner_pin
        monkeypatch.setattr(H, "_planner_pin",
                            lambda e, nv, s: (seen.append(s), real(e, nv, s))[1])
        hybrid_zeta(K5E, NU, SQUARE, n_points=12, source=0, terminal=4,
                    momentum=np.array([0.25, 0.0]))
        assert seen == [], "the planner pin must not be taken at finite k"

    def test_the_full_grid_call_keeps_the_callers_pin(self, monkeypatch):
        seen = []
        real = H._planner_pin
        monkeypatch.setattr(H, "_planner_pin",
                            lambda e, nv, s: (seen.append(s), real(e, nv, s))[1])
        hybrid_zeta(K5E, NU, SQUARE, n_points=12, source=0, terminal=4)
        assert seen == [], "the planner pin must not be taken on the BZ grid path"

    def test_finite_k_is_unaffected_in_value(self, monkeypatch):
        k = np.array([0.25, 0.125])
        monkeypatch.setattr(H, "USE_PLANNER_PIN", True)
        a = hybrid_zeta(K5E, NU, SQUARE, n_points=12, source=0, terminal=4, momentum=k)
        monkeypatch.setattr(H, "USE_PLANNER_PIN", False)
        b = hybrid_zeta(K5E, NU, SQUARE, n_points=12, source=0, terminal=4, momentum=k)
        assert a == b


class TestTheTieBreakOnTheSpReducedCut:
    r"""When the exponent cannot choose, the SP-reduced cut must.

    The pin is never eliminated, so pinning a DEGREE-2 vertex keeps its
    escape alive and holds sigma_core down at the block's own rate — which
    IS the pass floor, so the core-sizing rule then correctly concludes
    there is no gap to spend and declines to grade.  Pinning anywhere else
    lets that vertex reduce away and the core's cut jumps.

    On the d = 2 order-11 block that dominated the pass this is 17.2 GB
    against 48 MB, decided entirely by a tie-break.
    """

    # V6E11, treewidth 4, degrees [2, 4, 4, 4, 4, 4].
    V6E11 = [(0, 1), (0, 4), (1, 2), (1, 3), (1, 5), (2, 3),
             (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
    V6E11_NU = [2.5] * 11

    def test_the_exponent_really_does_tie(self):
        """Otherwise this class is gating a case that cannot arise."""
        V = sorted({v for e in self.V6E11 for v in e})
        exps = {p: EL.plan(self.V6E11, V, pin=p).exponent for p in V}
        assert len(set(exps.values())) == 1, exps

    def test_the_degree_two_vertex_is_not_pinned(self):
        import networkx as nx
        g = nx.Graph(self.V6E11)
        deg2 = [v for v, dd in g.degree() if dd == 2]
        assert deg2, "the block must have a degree-2 vertex for this to bite"
        assert H._planner_pin(self.V6E11, self.V6E11_NU, 0) not in deg2

    def test_the_chosen_pin_maximises_the_sp_reduced_cut(self):
        from gzl._elimination import sp_reduced_cut_nu
        emap = {}
        for (u, v), w in zip(self.V6E11, self.V6E11_NU):
            emap[(min(u, v), max(u, v))] = emap.get((min(u, v), max(u, v)), 0.0) + w
        V = sorted({v for e in self.V6E11 for v in e})
        pin = H._planner_pin(self.V6E11, self.V6E11_NU, 0)
        assert sp_reduced_cut_nu(emap, pin) == max(
            sp_reduced_cut_nu(emap, q) for q in V)

    def test_it_is_deterministic(self):
        rng = np.random.default_rng(11)
        ref = H._planner_pin(self.V6E11, self.V6E11_NU, 0)
        for _ in range(15):
            idx = rng.permutation(len(self.V6E11))
            e = [self.V6E11[i] for i in idx]
            nu = [self.V6E11_NU[i] for i in idx]
            assert H._planner_pin(e, nu, 0) == ref

    def test_the_value_is_unchanged(self):
        a = hybrid_zeta(self.V6E11, self.V6E11_NU, SQUARE, n_points=12)
        import gzl.hybrid as _H
        _H.USE_PLANNER_PIN = False
        try:
            b = hybrid_zeta(self.V6E11, self.V6E11_NU, SQUARE, n_points=12)
        finally:
            _H.USE_PLANNER_PIN = True
        assert abs(a - b) <= 1e-12 * abs(b)


class TestNuInfIsRefused:
    r"""``nu = inf`` is not supported by this engine, by design.

    The value there is an INTEGER homomorphism count, but hybrid collapses
    the series/parallel part by FFT, so it comes back off the integer --
    99.99999999999996 for 100 on a theta graph at d = 2.
    ``graph_zeta_general`` is exact because it never leaves integer
    arithmetic, and ``frontend._block_general`` has always gated on
    ``not np.any(np.isinf(nu_arr))``, so the router never sent nu = inf
    here.  Refusing makes that skip explicit rather than incidental.

    The alternative -- teaching ``_planner_pin`` the no-peel cost model so
    the nu = inf path is CHEAPER -- was rejected: it would optimise a path
    whose answers are wrong.
    """

    THETA = np.array([(0, 2), (2, 1), (0, 3), (3, 1), (0, 4), (4, 1)],
                     dtype=int)

    def test_hybrid_refuses(self):
        with pytest.raises(ValueError, match="nu = inf"):
            hybrid_zeta(self.THETA, np.full(6, np.inf), np.eye(2), 7)

    def test_the_router_still_answers_and_is_exact(self):
        """Refusing costs no capability: evaluate_graph routes to the
        tensor and returns the exact integer."""
        from gzl.frontend import evaluate_graph
        got = evaluate_graph(self.THETA, np.inf, np.eye(2), n_points=7)
        assert got == 100.0

    def test_finite_nu_is_untouched(self):
        assert np.isfinite(
            hybrid_zeta(self.THETA, np.full(6, 2.5), np.eye(2), 7))
