# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""The tau*d acceptance test: the achieved exponent equals the plan's.

With the planner live in both engines, this file asserts — per
instrumented elimination step, on real corpus blocks, at k = 0 AND on
the open-terminal grid path, for BOTH truncations — that the achieved
maximum post-peel step exponent equals the plan's:

    dense step   pays  n^(|bag| * d)
    peeled step  pays  n^((|bag| - 1) * d)

so the whole contraction runs at ``n^(plan.exponent * d)`` with
``plan.exponent <= tau`` (treewidth).  The box's COST gate is the one
sanctioned exception: dense really is faster at small L, so a
structurally peelable step the gate keeps dense may raise the achieved
exponent — but only if the engine SURFACED it in the per-call
``direct_sum._LAST_GATE_DECLINED`` diagnostic.  A silent exponent
degradation anywhere is a failure.

Instrumentation notes (each earned the hard way):
- step records are collected in a fresh local list per run, never a
  module-level one — an accumulating step list once inverted a pin
  measurement by making the denominator a prefix of the numerator;
- engines are called DIRECTLY: through ``evaluate_graph`` a cycle or
  bridge resolves to a closed form and a 0.00e+00 comparison measures
  the router, not the engine;
- the peel/dense split is recorded by wrapping the Truncation methods
  the shared skeleton actually dispatches to, so chunked peels and the
  marker path count exactly once.
"""

from __future__ import annotations

import numpy as np
import pytest

import gzl._elimination as E
import gzl.direct_sum as ds
import gzl.tensor_network as tn

nx = pytest.importorskip("networkx")

CANONICAL = {
    "K4": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    "K5": [(i, j) for i in range(5) for j in range(i + 1, 5)],
    "K5-e": [(i, j) for i in range(5) for j in range(i + 1, 5)
             if (i, j) != (0, 4)],
    "prism": [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
              (0, 3), (1, 4), (2, 5)],
    "K33": [(i, j) for i in (0, 1, 2) for j in (3, 4, 5)],
    "diamond": [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)],
}
CORPUS_BLOCKS = [
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)],
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 4), (3, 4)],
    [(0, 1), (0, 2), (0, 5), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5),
     (4, 5)],
    [(0, 1), (0, 3), (0, 5), (1, 2), (1, 4), (2, 3), (2, 5), (3, 4),
     (4, 5)],
]
NU = 2.5


def _tw(edges):
    g = nx.Graph()
    g.add_nodes_from({v for e in edges for v in e})
    g.add_edges_from(edges)
    return nx.algorithms.approximation.treewidth_min_fill_in(g)[0]


def _record_steps(monkeypatch, cls):
    """Wrap ``cls.peel_step``/``cls.dense_step`` and return the fresh
    per-run record list: entries ``(vertex, |bag|, peeled)`` with the
    eliminated vertex included in the bag count, matching Step.bag."""
    steps: list = []
    orig_peel = cls.peel_step
    orig_dense = cls.dense_step

    def peel(self, bucket, token, v, out_axes, dtype):
        steps.append((v, len(out_axes) + 1, True))
        return orig_peel(self, bucket, token, v, out_axes, dtype)

    def dense(self, bucket, v, out_axes, dtype):
        steps.append((v, len(out_axes) + 1, False))
        return orig_dense(self, bucket, v, out_axes, dtype)

    monkeypatch.setattr(cls, "peel_step", peel)
    monkeypatch.setattr(cls, "dense_step", dense)
    return steps


def _achieved(steps):
    return max((bag - 1 if peeled else bag)
               for _, bag, peeled in steps)


def _assert_box_invariant(steps, plan, where):
    """achieved == plan.exponent, or every dense step at the achieved
    peak is present in the surfaced gate diagnostic."""
    achieved = _achieved(steps)
    declined = ds._LAST_GATE_DECLINED
    assert all(e["reason"] in ("gate", "min_bag") for e in declined), \
        f"{where}: unexpected decline reason in {declined}"
    if achieved == plan.exponent:
        return
    assert achieved > plan.exponent, (
        f"{where}: engine beat the plan ({achieved} < {plan.exponent})"
        " — the plan is supposed to be exact, not a bound"
    )
    peak_dense = [(v, bag) for v, bag, peeled in steps
                  if not peeled and bag == achieved]
    surfaced = {(e["vertex"], e["axes"]) for e in declined}
    assert peak_dense and all(pd in surfaced for pd in peak_dense), (
        f"{where}: achieved exponent {achieved} exceeds planned "
        f"{plan.exponent} with peak dense steps {peak_dense} NOT all "
        f"surfaced in the gate diagnostic {sorted(surfaced)} — a "
        f"silent exponent degradation"
    )


def _run_torus_k0(monkeypatch, edges, n=8, d=1):
    A = np.eye(d)
    steps = _record_steps(monkeypatch, tn.TorusTruncation)
    tn.graph_zeta_general_at_zero(
        np.array(edges), np.full(len(edges), NU), A, n)
    pairs = sorted({(min(u, v), max(u, v)) for u, v in edges})
    V = max(max(e) for e in edges) + 1
    return steps, E.plan(pairs, range(V))


def _run_torus_grid(monkeypatch, edges, terminal, n=8, d=1):
    A = np.eye(d)
    steps = _record_steps(monkeypatch, tn.TorusTruncation)
    tn.graph_zeta_general(
        np.array(edges), np.full(len(edges), NU), A, n,
        terminals=(terminal,), space="k")
    pairs = sorted({(min(u, v), max(u, v)) for u, v in edges})
    V = max(max(e) for e in edges) + 1
    return steps, E.plan(pairs, range(V), pin=0, keep=[terminal])


def _run_box_k0(monkeypatch, edges, L=4, d=1):
    A = np.eye(d)
    nu_vec = np.full(len(edges), NU)
    em = ds._collapse_multi_edges(np.array(edges), nu_vec)
    V = max(max(e) for e in em) + 1
    root = ds._pick_root(em, V)
    steps = _record_steps(monkeypatch, ds.BoxTruncation)
    ds.direct_sum_zero_momentum(np.array(edges), nu_vec, A, L)
    return steps, E.plan(em, range(V), pin=root)


def _run_box_grid(monkeypatch, edges, terminal, L=4, d=1, root=0):
    A = np.eye(d)
    nu_vec = np.full(len(edges), NU)
    em = ds._collapse_multi_edges(np.array(edges), nu_vec)
    V = max(max(e) for e in em) + 1
    steps = _record_steps(monkeypatch, ds.BoxTruncation)
    ds._direct_sum_open_terminal(em, A, L, root, terminal, d)
    return steps, E.plan(em, range(V), pin=root, keep=[terminal])


ALL_BLOCKS = [(f"canon-{k}", v) for k, v in sorted(CANONICAL.items())] \
           + [(f"corpus-{i}", b) for i, b in enumerate(CORPUS_BLOCKS)]


class TestTorusAchievesThePlan:
    """The torus peel is ungated (finite nu), so the achieved exponent
    must equal the plan's on every block — no exemptions."""

    @pytest.mark.parametrize("name,edges", ALL_BLOCKS,
                             ids=[n for n, _ in ALL_BLOCKS])
    def test_k0_d1(self, name, edges, monkeypatch):
        steps, plan = _run_torus_k0(monkeypatch, edges)
        assert _achieved(steps) == plan.exponent
        assert plan.exponent <= _tw(edges)

    @pytest.mark.parametrize("name,edges", ALL_BLOCKS,
                             ids=[n for n, _ in ALL_BLOCKS])
    def test_grid_d1(self, name, edges, monkeypatch):
        t = max(max(e) for e in edges)
        steps, plan = _run_torus_grid(monkeypatch, edges, t)
        assert _achieved(steps) == plan.exponent
        assert plan.exponent <= _tw(edges) + 1

    @pytest.mark.parametrize("name", ["K4", "prism", "diamond"])
    def test_k0_d2(self, name, monkeypatch):
        steps, plan = _run_torus_k0(monkeypatch, CANONICAL[name],
                                    n=6, d=2)
        assert _achieved(steps) == plan.exponent
        assert plan.exponent <= _tw(CANONICAL[name])


class TestBoxAchievesThePlanOrSurfacesTheGate:
    """d=1 at the shipped rungs has the FFT gate CLOSED (open only from
    L = 15), so structurally peelable steps run dense — the acceptance
    is that every such peak step is surfaced, never silent.  d=2 at
    L = 4 has the gate OPEN, so the plan must be achieved outright
    wherever the per-step bag test passes."""

    @pytest.mark.parametrize("name,edges", ALL_BLOCKS,
                             ids=[n for n, _ in ALL_BLOCKS])
    def test_k0_d1_L4(self, name, edges, monkeypatch):
        steps, plan = _run_box_k0(monkeypatch, edges, L=4)
        _assert_box_invariant(steps, plan, f"{name} k0 d1 L4")

    @pytest.mark.parametrize("name,edges", ALL_BLOCKS,
                             ids=[n for n, _ in ALL_BLOCKS])
    def test_grid_d1_L4(self, name, edges, monkeypatch):
        t = max(max(e) for e in edges)
        steps, plan = _run_box_grid(monkeypatch, edges, t, L=4)
        _assert_box_invariant(steps, plan, f"{name} grid d1 L4")

    @pytest.mark.parametrize("name", ["K4", "prism", "diamond"])
    def test_k0_d2_L3_gate_closed(self, name, monkeypatch):
        steps, plan = _run_box_k0(monkeypatch, CANONICAL[name],
                                  L=3, d=2)
        _assert_box_invariant(steps, plan, f"{name} k0 d2 L3")

    @pytest.mark.parametrize("name", ["K4", "prism", "diamond"])
    def test_k0_d2_L4_gate_open(self, name, monkeypatch):
        steps, plan = _run_box_k0(monkeypatch, CANONICAL[name],
                                  L=4, d=2)
        _assert_box_invariant(steps, plan, f"{name} k0 d2 L4")

    def test_d1_gate_closure_is_actually_surfaced(self, monkeypatch):
        """Liveness control for the exemption arm: at d=1 L=4 the gate
        is closed (_conv_possible False for L=4..14), K4 has
        structurally peelable steps, so the diagnostic must be
        NON-EMPTY — an exemption that never fires guards nothing."""
        steps, plan = _run_box_k0(monkeypatch, CANONICAL["K4"], L=4)
        assert ds._LAST_GATE_DECLINED, (
            "gate diagnostic empty although the d=1 gate is closed "
            "and K4 plans peelable steps"
        )
        assert any(not peeled for _, _, peeled in steps)

    def test_diagnostic_is_replaced_per_call_not_accumulated(
            self, monkeypatch):
        """The accumulator trap, pinned: two consecutive calls must
        not concatenate their diagnostics."""
        _run_box_k0(monkeypatch, CANONICAL["K4"], L=4)
        first = list(ds._LAST_GATE_DECLINED)
        _run_box_k0(monkeypatch, CANONICAL["K4"], L=4)
        second = list(ds._LAST_GATE_DECLINED)
        assert first == second, "diagnostic accumulated across calls"


class TestInfKernelsPlanHonestly:
    """A closed loophole: nu = inf bundles never take the FFT peel
    (integer-exact NN contraction), so a nu-blind plan would claim
    peels an engine refuses, its exponent would read optimistic, and
    achieved > planned would surface with NO gate diagnostic — the
    exact silent degradation this file forbids.  The box declares its
    inf bundles to the planner via non_kernel_edges, so the invariant
    holds outright.  The torus has two honest plans: with
    ``tensor_network._USE_SPARSE_PEEL`` on (the default) an inf bundle
    peels by the exact roll-sum under the same structural condition as
    a kernel edge, so the honest plan treats it as one; with the switch
    off the bundle is dense and is declared, exactly as before the
    sparse peel existed."""

    def test_box_inf_achieved_equals_the_honest_plan(self, monkeypatch):
        edges = CANONICAL["K4"]
        nu_vec = np.full(len(edges), np.inf)
        em = ds._collapse_multi_edges(np.array(edges), nu_vec)
        root = ds._pick_root(em, 4)
        steps = _record_steps(monkeypatch, ds.BoxTruncation)
        val = complex(ds.direct_sum_zero_momentum(
            np.array(edges), nu_vec, np.eye(1), 3)).real
        p = E.plan(em, range(4), pin=root,
                   non_kernel_edges=tuple(em))
        assert all(not s.partners for s in p.steps)
        assert _achieved(steps) == p.exponent
        assert not ds._LAST_GATE_DECLINED, (
            "an inf decline is a correctness opt-out, not a cost "
            "exemption — it must not pollute the gate diagnostic"
        )
        assert val == np.round(val)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_torus_inf_achieved_equals_the_honest_plan(self, monkeypatch,
                                                      sparse):
        edges = CANONICAL["K4"]
        nu_vec = np.full(len(edges), np.inf)
        pairs = sorted({(min(u, v), max(u, v)) for u, v in edges})
        monkeypatch.setattr(tn, "_USE_SPARSE_PEEL", sparse)
        steps = _record_steps(monkeypatch, tn.TorusTruncation)
        val = tn.graph_zeta_general_at_zero(
            np.array(edges), nu_vec, np.eye(1), 6)
        p = E.plan(pairs, range(4),
                   non_kernel_edges=() if sparse else pairs)
        if sparse:
            # the sparse peel IS a peel: the honest plan partners, the
            # engine peels, and the exponent drops by one
            assert any(s.partners for s in p.steps)
            assert any(peeled for _, _, peeled in steps)
            assert p.exponent == E.plan(pairs, range(4),
                                        non_kernel_edges=pairs).exponent - 1
        else:
            assert all(not s.partners for s in p.steps)
            assert not any(peeled for _, _, peeled in steps)
        assert _achieved(steps) == p.exponent
        assert float(np.real(val)) == np.round(float(np.real(val)))

    @pytest.mark.parametrize("sparse", [True, False])
    def test_torus_mixed_inf_finite_achieved_equals_plan(self,
                                                         monkeypatch,
                                                         sparse):
        """One inf bundle inside a finite block.  Sparse peel on: the
        bundle peels by rolls and the honest plan is the kernel plan.
        Off: only that bundle is excluded from peeling, and the honest
        plan matches the engine."""
        edges = CANONICAL["prism"]
        nu_vec = np.full(len(edges), NU)
        nu_vec[0] = np.inf                        # bundle (0, 1)
        monkeypatch.setattr(tn, "_USE_SPARSE_PEEL", sparse)
        steps = _record_steps(monkeypatch, tn.TorusTruncation)
        tn.graph_zeta_general_at_zero(
            np.array(edges), nu_vec, np.eye(1), 6)
        pairs = sorted({(min(u, v), max(u, v)) for u, v in edges})
        p = E.plan(pairs, range(6),
                   non_kernel_edges=() if sparse else [(0, 1)])
        assert _achieved(steps) == p.exponent
        if not sparse:
            for s in p.steps:            # the inf bundle never partners
                if s.vertex == 0:
                    assert 1 not in s.partners
                if s.vertex == 1:
                    assert 0 not in s.partners


@pytest.mark.slow
class TestShippedCorpusCensus:
    """The full acceptance sweep over every distinct tw >= 3 block of
    the shipped corpora, both truncations, k = 0 and grid."""

    @pytest.fixture(scope="class")
    def census(self):
        from tests._block_census import DATA_DIR, enumerate_blocks
        blocks, _ = enumerate_blocks(DATA_DIR, keep_mults=False)
        return blocks

    def test_census_k0_and_grid_both_truncations(self, census,
                                                 monkeypatch):
        checked = 0
        for rec in census:
            G = rec["g"]
            nv = G.number_of_nodes()
            if nv > 8:                    # keep the sweep tractable
                continue
            local = {v: i for i, v in enumerate(sorted(G.nodes()))}
            edges = sorted(
                (min(local[u], local[v]), max(local[u], local[v]))
                for u, v in G.edges()
            )
            with pytest.MonkeyPatch.context() as mp:
                steps, plan = _run_torus_k0(mp, edges, n=6)
                assert _achieved(steps) == plan.exponent, edges
                assert plan.exponent <= _tw(edges)
            with pytest.MonkeyPatch.context() as mp:
                t = max(max(e) for e in edges)
                steps, plan = _run_torus_grid(mp, edges, t, n=6)
                assert _achieved(steps) == plan.exponent, edges
            with pytest.MonkeyPatch.context() as mp:
                steps, plan = _run_box_k0(mp, edges, L=4)
                _assert_box_invariant(steps, plan, f"{edges} box k0")
            with pytest.MonkeyPatch.context() as mp:
                t = max(max(e) for e in edges)
                steps, plan = _run_box_grid(mp, edges, t, L=4)
                _assert_box_invariant(steps, plan, f"{edges} box grid")
            checked += 1
        assert checked >= 100, (
            f"census sweep covered only {checked} blocks — "
            "coverage collapsed, not passed"
        )
