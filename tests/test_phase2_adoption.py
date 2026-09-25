# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Planner adoption: behavioural tests.

The guards in ``test_phase2_guards.py`` pin the refusal and desync
contracts; this file pins the *adoption mechanics*: that the engines
actually consult ``_elimination.plan`` (engagement sentinels — the
values match either way, so a detached planner would be invisible to
value tests), that the exact-key cache makes a Richardson ladder pay
for one DP rather than ``len(L_list)``, that a budget refusal falls
back to the shipped min-degree schedule, and that re-ordering is
value-exact so the adoption moves nothing but round-off.
"""

from __future__ import annotations

import numpy as np
import pytest

import gzl._elimination as E
import gzl.direct_sum as ds

K4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
K4_TAIL = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4)]
PRISM = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
         (0, 3), (1, 4), (2, 5)]
NU = 2.5


def _spy_plan(monkeypatch):
    """Count calls into the module-attribute ``plan`` and record kwargs."""
    calls = {"n": 0, "keeps": [], "pins": []}
    orig = E.plan

    def counting(*a, **kw):
        calls["n"] += 1
        calls["keeps"].append(tuple(kw.get("keep", ())))
        calls["pins"].append(kw.get("pin"))
        return orig(*a, **kw)

    monkeypatch.setattr(E, "plan", counting)
    return calls


class TestBoxAdoption:
    def test_zero_momentum_consults_the_planner(self, monkeypatch):
        calls = _spy_plan(monkeypatch)
        nu_vec = np.full(len(K4), NU)
        v = ds.direct_sum_zero_momentum(np.array(K4), nu_vec, np.eye(1), 5)
        assert np.isfinite(complex(v).real)
        assert calls["n"] == 1
        assert calls["pins"] == [ds._pick_root(
            ds._collapse_multi_edges(np.array(K4), nu_vec), 4)]

    def test_ladder_pays_for_one_dp_via_the_cache(self, monkeypatch):
        """Four rungs, one planner DP: the exact-key cache absorbs the
        per-L repetition.  ``plan`` itself is still entered per rung
        (the spy sees every call), but only the first misses."""
        calls = _spy_plan(monkeypatch)
        E._plan_cached.cache_clear()
        before = E._plan_cached.cache_info().misses
        nu_vec = np.full(len(K4_TAIL), NU)
        ds.direct_sum_extrapolated(K4_TAIL, nu_vec, np.eye(1),
                                   L_list=(4, 5, 6, 7))
        after = E._plan_cached.cache_info().misses
        assert calls["n"] == 4                    # one per rung
        assert after - before == 1                # one DP

    def test_open_terminal_keeps_the_terminal_open(self, monkeypatch):
        calls = _spy_plan(monkeypatch)
        nu_vec = np.full(len(K4_TAIL), NU)
        ds._direct_sum_extrapolated_grid(
            K4_TAIL, nu_vec, np.eye(1),
            np.array([[0.0], [0.25]]), root=0, terminal=4,
            L_list=(3, 4, 5, 6),
        )
        assert calls["n"] >= 1
        assert set(calls["keeps"]) == {(4,)}
        assert set(calls["pins"]) == {0}

    def test_finite_momentum_without_root_is_still_refused(self):
        """Pinned regression: root_eff threading must not swallow the
        per-L engine's 'finite momentum requires an explicit root'
        refusal — the phase source is caller semantics, never the
        engine's to guess via _pick_root."""
        nu_vec = np.full(len(K4_TAIL), NU)
        with pytest.raises(ValueError, match="explicit `root`"):
            ds.direct_sum_extrapolated(
                K4_TAIL, nu_vec, np.eye(1), L_list=(4, 5, 6, 7),
                momentum=np.array([0.25]), terminal=4)

    def test_budget_refusal_falls_back_to_min_degree(self, monkeypatch):
        """When plan refuses, the engine must run the shipped heuristic
        and agree with the planned value to round-off (re-ordering is
        the same finite sum contracted differently)."""
        nu_vec = np.full(len(PRISM), NU)
        planned = complex(ds.direct_sum_zero_momentum(
            np.array(PRISM), nu_vec, np.eye(1), 4))

        def refusing(*a, **kw):
            raise E.PlanBudgetExceededError("forced by test")

        monkeypatch.setattr(E, "plan", refusing)
        md_calls = {"n": 0}
        orig_md = ds._min_degree_order

        def counting_md(adj, free, root):
            md_calls["n"] += 1
            return orig_md(adj, free, root)

        monkeypatch.setattr(ds, "_min_degree_order", counting_md)
        fallback = complex(ds.direct_sum_zero_momentum(
            np.array(PRISM), nu_vec, np.eye(1), 4))
        assert md_calls["n"] == 1
        assert fallback == pytest.approx(planned, rel=1e-12)

    @pytest.mark.parametrize("edges", [K4, K4_TAIL, PRISM],
                             ids=["K4", "K4_TAIL", "prism"])
    def test_planned_order_is_value_exact_vs_min_degree(self, edges,
                                                        monkeypatch):
        """The adoption's whole value claim: order changes move nothing
        but round-off.  Both schedules through the same engine."""
        nu_vec = np.full(len(edges), NU)
        planned = complex(ds.direct_sum_zero_momentum(
            np.array(edges), nu_vec, np.eye(2), 3))
        monkeypatch.setattr(
            E, "plan",
            lambda *a, **kw: (_ for _ in ()).throw(
                E.PlanBudgetExceededError("forced")))
        heuristic = complex(ds.direct_sum_zero_momentum(
            np.array(edges), nu_vec, np.eye(2), 3))
        assert planned == pytest.approx(heuristic, rel=1e-12)

    def test_torus_vacuum_default_source_consults_planner(self,
                                                          monkeypatch):
        """graph_zeta_general_at_zero with the default pin lets the
        planner choose it — and the value moves nothing but round-off
        against every explicit pin (translation invariance)."""
        import gzl.tensor_network as tn
        calls = _spy_plan(monkeypatch)
        nu_vec = np.full(len(K4_TAIL), 3.5)
        planned = complex(tn.graph_zeta_general_at_zero(
            np.array(K4_TAIL), nu_vec, np.eye(1), 12))
        assert calls["n"] == 1
        assert calls["pins"] == [None]           # free pin search
        for p in range(5):
            explicit = complex(tn.graph_zeta_general_at_zero(
                np.array(K4_TAIL), nu_vec, np.eye(1), 12,
                pinned_vertex=p))
            assert planned == pytest.approx(explicit, rel=1e-12)
        assert calls["n"] == 1                   # explicit pins skip it

    def test_torus_terminal_and_momentum_calls_skip_planner(self,
                                                            monkeypatch):
        """Finite-k layouts encode x_source = 0; the planner must not
        touch their pin even when the caller leaves source unset."""
        import gzl.tensor_network as tn
        calls = _spy_plan(monkeypatch)
        nu_vec = np.full(len(K4_TAIL), 3.5)
        tn.graph_zeta_general(np.array(K4_TAIL), nu_vec, np.eye(1), 8,
                              terminals=(4,), space="k")
        tn.graph_zeta_general(np.array(K4_TAIL), nu_vec, np.eye(1), 8,
                              terminals=(4,), momentum=np.array([0.25]))
        assert calls["n"] == 0

    def test_frontend_internal_vacuum_block_reaches_the_planner(
            self, monkeypatch):
        """End-to-end: a purely internal σ-routed vacuum block through
        evaluate_graph(engine='tensor') hands the engine source=None,
        and the value matches the hybrid engine to round-off."""
        from gzl import evaluate_graph
        DIAMOND = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
        nu = 4.0                                  # sigma = 3 >= 1.49
        v_hybrid = float(evaluate_graph(
            np.array(DIAMOND), nu, np.eye(1), n_points=16,
            block_cache={}))
        calls = _spy_plan(monkeypatch)
        v_tensor = float(evaluate_graph(
            np.array(DIAMOND), nu, np.eye(1), n_points=16,
            block_cache={}, engine="tensor"))
        assert calls["n"] >= 1
        assert None in calls["pins"]
        assert v_tensor == pytest.approx(v_hybrid, rel=1e-10)

    def test_finite_momentum_declares_the_cos_scope(self, monkeypatch):
        """The single-k weight is an extra factor on the terminal; the
        planner must be told (extra_scopes) so the schedule prices it."""
        seen = {}
        orig = E.plan

        def capture(*a, **kw):
            seen.setdefault("extra", []).append(
                tuple(kw.get("extra_scopes", ())))
            return orig(*a, **kw)

        monkeypatch.setattr(E, "plan", capture)
        nu_vec = np.full(len(K4_TAIL), NU)
        ds.direct_sum_zero_momentum(
            np.array(K4_TAIL), nu_vec, np.eye(1), 5,
            root=0, momentum=np.array([0.25]), terminal=4)
        assert seen["extra"] == [((4,),)]


def _reference_torus_min_degree(adj, vertices, terminals):
    """The torus heuristic exactly as shipped before the min-degree merge."""
    nb = {v: set(adj[v]) for v in vertices}
    remaining = set(vertices) - terminals
    order = []
    active = remaining | terminals
    while remaining:
        v = min(remaining, key=lambda u: len(nb[u] & active))
        order.append(v)
        active_nb = list(nb[v] & active)
        for a in active_nb:
            for b in active_nb:
                if a != b:
                    nb[a].add(b)
        active.discard(v)
        remaining.discard(v)
    return order


def _reference_box_min_degree(adj, free, root):
    """The box heuristic exactly as shipped before the min-degree merge."""
    elimination = []
    remaining = set(free)
    while remaining:
        next_v = min(
            remaining,
            key=lambda v: sum(
                1 for u in adj[v] if u in remaining or u == root),
        )
        elimination.append(next_v)
        remaining.remove(next_v)
    return elimination


class TestMinDegreeMerge:
    """One home for the two min-degree heuristics, bodies moved
    verbatim.  The facades must be order-identical to the pre-merge
    implementations (tie-breaks included), and the two MODES must stay
    genuinely different — collapsing them would silently change one
    engine's schedule."""

    def _random_graph(self, rng):
        V = int(rng.integers(3, 9))
        edges = [(i, int(rng.integers(0, i))) for i in range(1, V)]
        for _ in range(int(rng.integers(0, V + 2))):
            u, v = rng.integers(0, V, 2)
            if u != v:
                edges.append((min(int(u), int(v)), max(int(u), int(v))))
        adj = {v: set() for v in range(V)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return V, adj

    def test_torus_facade_is_order_identical_to_the_old_code(self):
        import gzl.tensor_network as tn
        rng = np.random.default_rng(21)
        for _ in range(200):
            V, adj = self._random_graph(rng)
            terminals = set(
                int(t) for t in
                rng.choice(V, size=int(rng.integers(0, 3)),
                           replace=False))
            got = tn._min_degree_order(adj, list(range(V)), terminals)
            want = _reference_torus_min_degree(
                adj, list(range(V)), terminals)
            assert got == want

    def test_box_facade_is_order_identical_to_the_old_code(self):
        rng = np.random.default_rng(22)
        for _ in range(200):
            V, adj = self._random_graph(rng)
            root = int(rng.integers(0, V))
            free = [v for v in range(V) if v != root]
            got = ds._min_degree_order(adj, free, root)
            want = _reference_box_min_degree(adj, free, root)
            assert got == want

    def test_the_two_modes_still_diverge_somewhere(self):
        """Liveness: over the random battery the fill-in and static
        heuristics must produce at least one differing order — the
        merge must not have collapsed the semantics."""
        rng = np.random.default_rng(23)
        diverged = 0
        for _ in range(300):
            V, adj = self._random_graph(rng)
            root = 0
            free = [v for v in range(V) if v != root]
            torus = E.min_degree_order(
                adj, list(range(V)), terminals={root}, fill_in=True)
            box = E.min_degree_order(adj, free, root=root,
                                     fill_in=False)
            if torus != box:
                diverged += 1
        assert diverged > 0

    def test_mode_flags_are_required(self):
        with pytest.raises(TypeError):
            E.min_degree_order({0: set()}, [0], fill_in=True)
        with pytest.raises(TypeError):
            E.min_degree_order({0: set()}, [0], fill_in=False)
