# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""The shared series/parallel graph rewrite (:mod:`gzl._sp`).

``_sp.sp_reduce`` is the third piece of the truncation-agnostic layer,
after the executor (``_contract``) and the planner (``_elimination``).
It is parameterised by a ``Truncation``, so the same rewrite serves the
periodic torus and the zero-padded box.

The two truncations differ **in kind** here, and that is the thing these
tests are built around:

* the **torus** composes CYCLICALLY, which is exact for its own finite
  sum, so a torus collapse is value-preserving and gated bitwise;
* the **box** composes LINEARLY and only approximately.  The exact box
  composition depends on both endpoints separately rather than on their
  difference — ``[-L, L]^d`` is not translation-closed and the product
  of two Toeplitz matrices is not Toeplitz — so no difference kernel
  represents it.  A box collapse is therefore gated on CONVERGENCE in
  ``L``, never on agreement at fixed ``L``.

Writing a single tolerance across both would have certified the box
where it cannot hold and the torus where it is far too weak.
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl import _contract, _sp
from gzl.direct_sum import BoxTruncation, direct_sum_zero_momentum
from gzl.hybrid import hybrid_zeta
from gzl.tensor_network import (
    TorusTruncation,
    _edge_kernel_torus,
    graph_zeta_general,
)


NU = 2.5
A1 = np.eye(1)
A2 = np.eye(2)
AHEX = np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2.0]])

CYCLE4 = [(0, 1), (1, 2), (2, 3), (0, 3)]
TRIANGLE = [(0, 1), (1, 2), (0, 2)]
K4SUB = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (4, 3)]


def _slope(xs, ys):
    return float(np.polyfit(np.log(np.asarray(xs, float)),
                            np.log(np.asarray(ys, float)), 1)[0])


# ---------------------------------------------------------------------------
# trace — the one formula that changed when the rewrite moved into _sp
# ---------------------------------------------------------------------------

class TestTrace:
    r"""``trace`` is the kernel at ZERO displacement, not its sum.

    A loop closes both endpoints onto the same vertex, so it contributes
    ``K(x - x) = K(0)`` — one entry.  ``gen.sum()``, which the torus
    rewrite used before it moved into ``_sp``, is the DISCONNECTED
    answer and is larger by orders of magnitude (2.66 against 0 on the
    box generator below).  Under the regularised kernel the right
    answer is exactly the zero that drops every coincident-endpoint
    configuration.
    """

    @pytest.mark.parametrize("d, A", [(1, A1), (2, A2)])
    def test_torus_trace_is_the_origin_entry(self, d, A):
        t = TorusTruncation(8, d, A)
        gen = _edge_kernel_torus(NU, A, 8)
        assert t.trace(gen) == gen[(0,) * d]
        assert t.trace(gen) == 0.0            # regularised kernel
        assert gen.sum() > 1.0                # ... and not the sum

    @pytest.mark.parametrize("d, A", [(1, A1), (2, A2)])
    def test_box_trace_is_the_centre_of_the_difference_range(self, d, A):
        t = BoxTruncation(6, d, A)
        gen = t.generator(NU)
        # the box holds K(delta) at delta + 2L, so zero is the CENTRE
        assert t.trace(gen) == gen[(2 * 6,) * d]
        assert t.trace(gen) == 0.0
        assert gen.sum() > 1.0

    def test_the_old_sum_formula_was_the_disconnected_answer(self):
        r"""Names the defect exactly, rather than gating a magnitude.

        For a composed bridge ``K = K1 * K2`` the convolution theorem
        gives ``sum(K) == sum(K1) * sum(K2)`` — the value the two legs
        would contribute if the intermediate vertex were NOT summed
        against both of them, i.e. the disconnected configuration.  The
        loop contribution is ``K(0) = sum_z K1(z) K2(-z)``, a single
        entry.  Asserting the identity shows *why* ``gen.sum()`` was
        wrong, not merely that it differs.
        """
        t = TorusTruncation(16, 1, A1)
        gen = _edge_kernel_torus(NU, A1, 16)
        comp = t.compose(gen, gen)

        assert comp.sum() == pytest.approx(gen.sum() ** 2, rel=1e-12)
        assert t.trace(comp) == comp[0]
        assert t.trace(comp) == pytest.approx(
            float(np.sum(gen * gen[(-np.arange(16)) % 16])), rel=1e-12)
        assert t.trace(comp) < 0.5 * comp.sum()


class TestSelfLoopRefused:
    r"""A self-loop in the input is refused, not traced.

    ``direct_sum._collapse_multi_edges`` and ``frontend`` both refuse
    self-loops already.  The alternative here is worse than an
    exception: under the regularised kernel a self-loop makes the whole
    graph zeta vanish, so accepting one would return a zero
    indistinguishable from a computed value.
    """

    def test_sp_reduce_refuses(self):
        t = TorusTruncation(8, 1, A1)
        gen = _edge_kernel_torus(NU, A1, 8)
        with pytest.raises(ValueError, match="self-loop at vertex 1"):
            _sp.sp_reduce([(0, 1), (1, 1)], [gen, gen], 0, 0, t)

    def test_hybrid_zeta_refuses(self):
        with pytest.raises(ValueError, match="self-loop"):
            hybrid_zeta([(0, 1), (1, 1)], [NU, NU], A1, 8)


# ---------------------------------------------------------------------------
# torus — the collapse is exact, so it is gated bitwise
# ---------------------------------------------------------------------------

class TestTorusIsValuePreserving:

    @pytest.mark.parametrize("edges", [TRIANGLE, CYCLE4, K4SUB])
    @pytest.mark.parametrize("n", [8, 9, 16])
    def test_collapse_reproduces_the_uncollapsed_engine(self, edges, n):
        r"""The torus composes cyclically, which IS the truncated sum —
        so collapsing changes association, not value.  Gated against the
        independent tensor engine at round-off.
        """
        nu_vec = [NU] * len(edges)
        ref = float(np.real(graph_zeta_general(edges, nu_vec, A1, n)))
        got = hybrid_zeta(edges, nu_vec, A1, n)
        assert got == pytest.approx(ref, rel=1e-12)

    def test_the_rewrite_actually_shrinks_the_graph(self):
        r"""Liveness: a test that collapses nothing would pass every
        value gate above for the wrong reason."""
        t = TorusTruncation(8, 1, A1)
        gen = _edge_kernel_torus(NU, A1, 8)
        nodes, r_edges, _, _ = _sp.sp_reduce(
            CYCLE4, [gen] * 4, 0, 0, t)
        assert (len(nodes), len(r_edges)) == (2, 1), (nodes, r_edges)

        # ... and on a 3-connected block it is a genuine no-op
        k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        nodes, r_edges, _, _ = _sp.sp_reduce(k4, [gen] * 6, 0, 0, t)
        assert (len(nodes), len(r_edges)) == (4, 6)


class TestParallelMergeOrientation:
    r"""The parallel merge must not reverse an already-canonical kernel.

    Every stored kernel is canonical ``min -> max``, so their Hadamard
    product is too.  The merge used to take its endpoint pair straight
    from ``tuple(frozenset(pair))``, which is HASH-ordered rather than
    sorted — descending on 31 of the 120 pairs drawn from 0..15 — and a
    descending pair makes the orientation helper reverse a kernel that
    was already right.

    Invisible on every orthogonal lattice and at every odd n, because
    the kernel is even there and reversal is a no-op.  Real on a sheared
    cell at even n, where the balanced axis holds n/2 without -n/2.
    """

    @pytest.mark.parametrize("pair", [(1, 2), (1, 8), (3, 11), (2, 9)])
    def test_bundle_kernel_stays_canonical(self, pair):
        t = TorusTruncation(8, 2, AHEX)
        ka = _edge_kernel_torus(2.5, AHEX, 8)
        kb = _edge_kernel_torus(3.5, AHEX, 8)     # distinct: product not even
        assert not np.array_equal(ka, t.reverse(ka)), "kernel must be non-even"

        u, v = pair
        _, r_edges, r_gens, _ = _sp.sp_reduce(
            [(u, v), (u, v)], [ka, kb], u, v, t)
        assert r_edges == [(min(u, v), max(u, v))]

        lo = u < v
        want = t.hadamard(ka if lo else t.reverse(ka),
                          kb if lo else t.reverse(kb))
        assert np.array_equal(r_gens[0], want), (
            f"pair {pair} (frozenset order {tuple(frozenset(pair))}): "
            f"bundle kernel is "
            f"{'REVERSED' if np.array_equal(r_gens[0], t.reverse(want)) else 'wrong'}"
            f", max|diff| = {np.max(np.abs(r_gens[0] - want)):.3e}"
        )

    def test_end_to_end_against_the_tensor_engine(self):
        r"""Contiguous labels — a gap in the vertex numbering would make
        ``graph_zeta_general`` count isolated vertices at ``n^d`` each and
        the comparison meaningless."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
                 (7, 8), (8, 0), (1, 8), (1, 8)]      # bundle on {1, 8}
        assert sorted({v for e in edges for v in e}) == list(range(9))

        # {1, 8} iterates DESCENDING on CPython (hash(8) % 8 == 0 lands
        # ahead of hash(1) % 8 == 1), which is what made this shape the
        # reproducer.  That is an implementation detail, not a language
        # guarantee, so it is reported rather than asserted: the
        # invariant below must hold on any iteration order, and if some
        # build orders the pair ascending this shape simply stops being
        # the interesting one instead of failing spuriously.
        descending = tuple(frozenset((1, 8)))[0] > tuple(frozenset((1, 8)))[1]

        nu = [NU] * len(edges)
        for n in (8, 12):
            got = hybrid_zeta(edges, nu, AHEX, n)
            ref = float(np.real(graph_zeta_general(edges, nu, AHEX, n)))
            assert got == pytest.approx(ref, rel=1e-12), (
                f"n={n}, bundle pair {{1,8}} iterates "
                f"{'descending' if descending else 'ascending'}"
            )


# ---------------------------------------------------------------------------
# box — the collapse is a surrogate, so it is gated on convergence
# ---------------------------------------------------------------------------

class TestBoxCollapse:

    def _sp_value(self, L):
        t = BoxTruncation(L, 1, A1)
        nodes, r_edges, r_gens, scalar = _sp.sp_reduce(
            CYCLE4, [t.generator(NU)] * 4, 0, 0, t)
        assert len(nodes) == 2 and len(r_edges) == 1
        a, b = r_edges[0]
        row = t.pin_slot(r_gens[0], b if a == 0 else a, pin_row=(a == 0))
        return scalar * float(row.sum())

    def test_converges_to_the_uncollapsed_box_one_signed(self):
        r"""End-to-end: the box collapse converges to the box value.

        NOTE what this does and does not pin down.  Instrumenting the
        reduction shows both series collapses here have the pin as an
        endpoint (w=1 -> neighbours (0,2); w=2 -> neighbours (3,0)), and
        the composed generator is then read through ``pin_slot`` as a
        window SUM — which is nearly invariant under a translation of
        the kernel.  So this test is deliberately NOT the guard on
        ``box_compose``'s centring; ``test_compose_matches_an_explicit
        _convolution`` below is, and it is entry-by-entry.  An earlier
        version of this docstring claimed the collapsed vertex had two
        free neighbours.  That was false, and it mattered: a +/-1 crop
        offset in box_compose leaves every assertion here passing while
        the composed kernel is wrong by a factor 5.4 at zero
        displacement.

        Both gates here matter.  ONE-SIGNED: the surrogate extends the
        intermediate sum with zero padding, so it can only over-count;
        a sign flip would mean the error is not the truncation being
        modelled.  CONVERGENT at ``2 nu - d``: that is the block's own
        truncation exponent, i.e. the collapse costs nothing in rate.
        """
        Ls = (4, 6, 8, 12, 16, 24, 32, 48)
        rel = []
        for L in Ls:
            e = float(np.real(direct_sum_zero_momentum(
                CYCLE4, [NU] * 4, A1, L, root=0)))
            rel.append((self._sp_value(L) - e) / e)

        assert all(r > 0 for r in rel), f"error must be one-signed; got {rel}"
        assert all(b < a for a, b in zip(rel, rel[1:])), (
            f"error must fall monotonically in L; got {rel}"
        )
        s = _slope(Ls, rel)
        assert -4.6 < s < -3.5, (
            f"box collapse converges at {s:+.2f}; expected 2 nu - d = -4.0"
        )

    def test_box_restriction_of_an_original_edge_is_exact(self):
        r"""A box generator is a pure power law on a symmetric
        difference range, so restricting it is a centred crop and must
        reproduce a directly-built coarse generator entry for entry.
        """
        for d, A in ((1, A1), (2, A2)):
            for L_f, L_c in ((8, 8), (12, 4), (16, 5)):
                fine, coarse = BoxTruncation(L_f, d, A), BoxTruncation(L_c, d, A)
                got = fine.restrict_to(fine.generator(NU), coarse)
                assert np.array_equal(got, coarse.generator(NU)), (d, L_f, L_c)

    def test_restriction_guards(self):
        f = BoxTruncation(8, 1, A1)
        with pytest.raises(ValueError, match="restrict upward"):
            f.restrict_to(f.generator(NU), BoxTruncation(16, 1, A1))
        with pytest.raises(ValueError, match="dimension mismatch"):
            f.restrict_to(f.generator(NU), BoxTruncation(4, 2, A2))
        with pytest.raises(ValueError, match="share the lattice"):
            f.restrict_to(f.generator(NU), BoxTruncation(4, 1, 2.0 * A1))

    def test_compose_rejects_a_wrongly_sized_generator(self):
        with pytest.raises(ValueError, match="difference range"):
            _contract.box_compose(np.ones(5), np.ones(5), 8, 1)

    @pytest.mark.parametrize("d, L", [(1, 4), (1, 7), (2, 3), (2, 4)])
    def test_compose_matches_an_explicit_convolution(self, d, L):
        r"""Entry-by-entry gate on box_compose — centring included.

        This is the guard the convergence test above cannot be.  The
        brute-force reference is the definition:
        ``K_new(z) = sum_v K1(z - v) K2(v)``, zero outside the
        difference range.  Comparing every entry catches the two
        defects a window-sum comparison is blind to: a crop offset
        (which shifts the kernel and moves ``trace``'s zero-displacement
        entry by a factor 5.4) and dropping the zero padding, i.e.
        computing a CYCLIC convolution where the docstring promises a
        linear one.
        """
        m = 4 * L + 1
        rng = np.random.default_rng(0)
        g1 = rng.random((m,) * d)
        g2 = rng.random((m,) * d)

        # brute force, straight from the definition
        want = np.zeros((m,) * d)
        for zi in np.ndindex(*(m,) * d):
            z = np.array(zi) - 2 * L
            tot = 0.0
            for vi in np.ndindex(*(m,) * d):
                v = np.array(vi) - 2 * L
                w = z - v + 2 * L
                if np.all((w >= 0) & (w < m)):
                    tot += g1[tuple(w)] * g2[vi]
            want[zi] = tot

        got = _contract.box_compose(g1, g2, L, d)
        assert got.shape == want.shape
        assert np.allclose(got, want, rtol=0, atol=1e-11 * np.max(want)), (
            f"max|diff| = {np.max(np.abs(got - want)):.3e} on a scale of "
            f"{np.max(want):.3e}"
        )

    def test_compose_places_zero_displacement_where_trace_reads_it(self):
        r"""``trace`` indexes ``(2L,)*d``; the composition must actually
        put ``sum_v K1(-v) K2(v)`` there.  A crop offset breaks exactly
        this and nothing else visible."""
        for d, L in ((1, 5), (2, 3)):
            t = BoxTruncation(L, d, A1 if d == 1 else A2)
            g = t.generator(NU)
            comp = t.compose(g, g)
            m = 4 * L + 1
            rev = g[tuple(slice(None, None, -1) for _ in range(d))]
            assert t.trace(comp) == pytest.approx(
                float(np.sum(g * rev)), rel=1e-12), (d, L)


# ---------------------------------------------------------------------------
# the rewrite buys rate, not cost — and that is a claim, so it is gated
# ---------------------------------------------------------------------------

class TestReductionDoesNotChangeCost:

    @pytest.mark.slow
    def test_planner_exponent_is_unchanged_on_the_whole_corpus(self):
        r"""SP reduction must not be sold as a speedup.

        Suppressing a degree-2 vertex preserves treewidth for tw >= 2,
        and the planner already eliminates such vertices at a bag below
        the peak — so the achieved cost exponent is identical on the
        block and on its SP-reduced core.  The entire value of the
        rewrite is the truncation RATE.

        This is gated rather than asserted in prose because the claim is
        exactly the kind that gets repeated from a scratch document and
        quietly goes stale.  It runs in ~4 s.
        """
        nx = pytest.importorskip("networkx")
        from gzl import _elimination
        from tests._block_census import DATA_DIR, enumerate_blocks

        t = TorusTruncation(6, 1, A1)
        blocks, n_inst = enumerate_blocks(DATA_DIR, True)
        assert n_inst > 100, f"corpus collapsed to {n_inst} occurrences"

        seen, reduced, examined = {}, 0, 0
        for e in blocks:
            g, cnt = e["g"], e["count"]
            relab = {v: i for i, v in enumerate(sorted(g.nodes()))}
            edges = [(relab[u], relab[v]) for u, v in g.edges()]
            gens = [_edge_kernel_torus(NU * float(dd.get("w", 1)), A1, 6)
                    for _, _, dd in g.edges(data=True)]

            nodes, r_edges, _, _ = _sp.sp_reduce(edges, gens, 0, 0, t)
            verts = sorted({v for pair in edges for v in pair})
            try:
                e_block = _elimination.plan(edges, verts, pin=0).exponent
                e_core = _elimination.plan(
                    r_edges, sorted(nodes), pin=0).exponent
            except _elimination.PlanBudgetExceededError:
                continue
            examined += 1
            seen[(e_block, e_core)] = seen.get((e_block, e_core), 0) + cnt
            if e_block != e_core:
                reduced += cnt

        assert examined > 0, "liveness: no block reached the planner"
        assert reduced == 0, (
            f"SP reduction moved the cost exponent on {reduced} occurrences; "
            f"the module docstring claims it never does.  Histogram "
            f"(block -> core): {seen}"
        )
        # the core NEVER gets cheaper, and never gets more expensive either
        assert all(a == b for a, b in seen), seen


# ---------------------------------------------------------------------------
# nothing is routed
# ---------------------------------------------------------------------------

class TestDirectSumIsUntouched:
    r"""The capability exists; the shipped box path does not use it.

    Wiring SP reduction into ``direct_sum`` would move the ``conv_nu``
    peel tag and the Richardson ``alpha`` basis, both of which are
    guarded against desync.  That is a separate change, and this test keeps
    it from being made by accident.
    """

    def test_direct_sum_never_calls_the_rewrite(self, monkeypatch):
        calls = []
        real = _sp.sp_reduce
        monkeypatch.setattr(
            _sp, "sp_reduce",
            lambda *a, **k: (calls.append(1), real(*a, **k))[1])

        direct_sum_zero_momentum(CYCLE4, [NU] * 4, A1, 6, root=0)
        direct_sum_zero_momentum(K4SUB, [NU] * 7, A1, 5, root=0)
        assert calls == [], (
            f"direct_sum reached the SP rewrite {len(calls)} time(s); the "
            f"shipped box path must not use it"
        )

        # the spy is live — hybrid does reach it
        hybrid_zeta(CYCLE4, [NU] * 4, A1, 8)
        assert calls, "spy never fired; the test proves nothing"
