# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Planner-adoption guards, written ahead of the adoption — and, since
the adoption landed, holding it in place.

They cover the engines' use of ``_elimination.plan`` and pin two
defect classes:

**Pin/alpha desync.**  ``direct_sum_extrapolated`` derives the
Richardson basis exponent ``alpha = d - _min_free_cut_nu(edge_map,
root_eff)`` while the per-L sums pin ``root_eff``.  Since the
adoption, the effective root is resolved once and threaded into every
per-L call (the green spy tests pin that), and the basis exponent is
cross-checked against the independent planner-side twin
``_elimination.min_free_cut_nu`` — a drifted basis fits the wrong
power with the least-squares solve still succeeding, the value finite
and plausible, and every raw per-L value individually correct, so the
cross-check is the only thing that can catch it.  The former
xfail(strict) negative arm now asserts the refusal directly (its
fixture was corrected at adoption — see the test docstring).

**Planner budget.**  ``_elimination.plan`` costs 0.24 ms on K4 but
8.97 s on C16 against a sub-second corpus pass; the adoption ships a
hard refusal above ``|free| = 12`` (``PlanBudgetExceededError``,
asserted in test_elimination.py), an exact-key memoisation in front
of ``plan`` (see ``_elimination._plan_cached`` — exact-key rather than
WL-class, so the cache is value-transparent under the label-dependent
DP tie-breaks), and simple cycles keep never reaching ``plan`` at all.

Note on the spy in the cycle test: it patches the *module attribute*
``_elimination.plan``.  If the adoption imports ``plan`` into an
engine's namespace at import time, that early binding would detach
this spy — resolve it through the module dict at call time (the same
construction rule every Truncation method follows).
"""

from __future__ import annotations

import numpy as np
import pytest

import gzl._elimination as E
import gzl.direct_sum as ds
from gzl import evaluate_graph


# K4 with a pendant vertex: asymmetric enough that the pin is a real
# choice (max degree = vertex 3) and the minimum free cut depends on
# which vertex is pinned (pendant 4 free: min cut nu; pendant pinned:
# min cut 3 nu).
K4_TAIL = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4)]
NU = 2.5


def _spies(monkeypatch):
    seen = {"picked": [], "alpha_roots": [], "pick_calls": 0}
    orig_pick = ds._pick_root
    orig_cut = ds._min_free_cut_nu

    def pick(edge_map, V):
        seen["pick_calls"] += 1
        r = orig_pick(edge_map, V)
        seen["picked"].append(r)
        return r

    def cut(edge_map, root):
        seen["alpha_roots"].append(int(root))
        return orig_cut(edge_map, root)

    monkeypatch.setattr(ds, "_pick_root", pick)
    monkeypatch.setattr(ds, "_min_free_cut_nu", cut)
    return seen


class TestPinAlphaConsistency:
    def test_auto_root_is_single_and_reaches_the_basis(self, monkeypatch):
        """root=None: every per-L sum and the alpha derivation must
        resolve to ONE root.  Any adoption that lets them diverge —
        a planner pin for the sums, ``_pick_root`` for the basis —
        trips this before any value is compared."""
        seen = _spies(monkeypatch)
        nu_vec = np.full(len(K4_TAIL), NU)
        ds.direct_sum_extrapolated(K4_TAIL, nu_vec, np.eye(1),
                                   L_list=(4, 5, 6, 7))
        # Exactly ONE pick: the adoption resolves root_eff once at the
        # extrapolated site and THREADS it into every per-L call.  A
        # >= 1 assertion would also pass if each per-L call re-picked
        # (deterministically identical today, but then consistency
        # would rest on determinism again instead of dataflow).
        assert seen["pick_calls"] == 1
        assert seen["alpha_roots"], "alpha never derived — vacuous run"
        roots = set(seen["picked"]) | set(seen["alpha_roots"])
        assert len(roots) == 1, (
            f"pin/alpha roots diverged: picked {set(seen['picked'])}, "
            f"alpha from {set(seen['alpha_roots'])}"
        )

    def test_explicit_root_reaches_the_basis_unrepicked(self,
                                                       monkeypatch):
        """root=4: the basis must be derived from 4 and ``_pick_root``
        must not be consulted at all."""
        seen = _spies(monkeypatch)
        nu_vec = np.full(len(K4_TAIL), NU)
        ds.direct_sum_extrapolated(K4_TAIL, nu_vec, np.eye(1),
                                   L_list=(4, 5, 6, 7), root=4)
        assert seen["pick_calls"] == 0
        assert set(seen["alpha_roots"]) == {4}

    def test_deliberate_desync_is_refused(self, monkeypatch):
        """A corrupted basis exponent must be refused, not fitted.

        Since the planner adoption, both extrapolated sites cross-check
        the engine's ``_min_free_cut_nu`` against the independent
        planner-side twin ``_elimination.min_free_cut_nu`` at the same
        root and raise ``PinAlphaDesyncError`` on any mismatch — the
        wrong-power fit would otherwise succeed silently with every raw
        per-L value individually correct.

        NOTE the fixture changed at adoption, deliberately: the
        original arm redirected the cut to root 0 and its docstring
        claimed 'pendant pinned: min cut 3nu'.  That premise belonged
        to the OLD single-vertex incident rule — the shipped *cluster*
        cut with no externals is root-INDEPENDENT on a connected graph
        (see test_cluster_cut_is_root_independent below), so the
        root-redirected basis was numerically correct and refusing it
        would have gated nothing.  The refusable desync class is a
        *value* drift of the basis exponent, which also covers every
        root-plumbing bug that could actually change alpha (they enter
        via ``externals`` or the incident-rule fallback)."""
        orig_cut = ds._min_free_cut_nu
        monkeypatch.setattr(
            ds, "_min_free_cut_nu",
            lambda edge_map, root: orig_cut(edge_map, root) - 1.0)
        nu_vec = np.full(len(K4_TAIL), NU)
        with pytest.raises(ds.PinAlphaDesyncError):
            ds.direct_sum_extrapolated(K4_TAIL, nu_vec, np.eye(1),
                                       L_list=(4, 5, 6, 7), root=4)

    def test_cluster_cut_is_root_independent_without_externals(self):
        """Pinned fact, discovered while flipping the negative arm: an
        admissible escaping bipartition serves every root (the root
        just lands on one side or the other, and the S/complement
        roles swap), so with ``externals=()`` the cluster cut of a
        connected graph cannot depend on the root — verified 200/200
        on random connected graphs at adoption time.  The single-vertex
        *incident* rule IS root-dependent (7.5 at the pendant of
        K4_TAIL vs 2.5 elsewhere), which is what the original fixture
        premise had in mind.  Pinned so nobody re-derives a
        root-plumbing guard from that stale premise."""
        nu_vec = np.full(len(K4_TAIL), NU)
        em = ds._collapse_multi_edges(np.array(K4_TAIL, dtype=int),
                                      nu_vec)
        cuts = {ds._min_free_cut_nu(em, r) for r in range(5)}
        assert len(cuts) == 1
        assert ds._min_free_incident_nu(em, 4) \
            != ds._min_free_incident_nu(em, 0)


class TestPlannerBudget:
    def test_simple_cycles_never_reach_plan(self, monkeypatch):
        """Cycles route to the closed form; the exact-DP planner must
        never see them (a C16 reaching `plan` costs 8.97 s)."""
        calls = {"n": 0}
        orig = E.plan

        def counting(*a, **kw):
            calls["n"] += 1
            return orig(*a, **kw)

        monkeypatch.setattr(E, "plan", counting)
        edges = np.array([(i, (i + 1) % 8) for i in range(8)])
        v = evaluate_graph(edges, NU, np.eye(1), n_points=16,
                           block_cache={})
        assert np.isfinite(float(np.real(v))), "vacuous evaluation"
        assert calls["n"] == 0, (
            f"a simple cycle reached _elimination.plan {calls['n']}x"
        )
