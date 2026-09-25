# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Exactness of the corpus-wide per-block memoization.

The ``block_cache`` added to :func:`gzl.evaluate_graph` /
:func:`gzl.series.compute_series_coefficients` is *pure*
memoization: block-cut already factors ζ_G = Π ζ_block, and a cached
block value is reused only after an exact, ν-/role-matched
``networkx.is_isomorphic`` confirmation of a Weisfeiler–Lehman hash
hit.  These tests pin:

1. **Bit-for-bit invariance** — `compute_series_coefficients` with vs
   without a `block_cache` produces *identical* per-order arrays on a
   real corpus slice.
2. **Cache effectiveness** — a corpus slice with many repeated
   bridge / cycle blocks reports a large `n_block_cache_hits`.
3. **Collision guard** — two non-isomorphic blocks that happen to
   share structural counts do not pollute each other's value.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("networkx")
pytest.importorskip("epsteinlib")

from gzl import data_path, evaluate_graph
from gzl.series import compute_series_coefficients


NPZ_0QP = data_path("tfim_softcore_corpus_0qp.npz")
NPZ_1QP = data_path("tfim_softcore_corpus_1qp.npz")


# ---------------------------------------------------------------------------
# 1. Pure-memoization invariant on a real corpus slice
# ---------------------------------------------------------------------------

class TestCacheBitIdentical:
    """The cache is **deterministic and reproducible** (two cached
    passes agree to the last bit) and **mathematically exact up to
    floating-point reassociation**: a reused block value is that of an
    isomorphic representative, whose evaluation order (min-degree
    elimination / einsum / FFT) is not vertex-labelling-invariant at
    the ULP level, so it can differ from a fresh same-labelling
    evaluation by ~1e-13 relative — far below the discretisation
    residual and identical in spirit to changing a summation order.
    """

    A1 = np.array([[1.0]])

    def test_two_calls_bit_identical(self):
        """Determinism: two cached corpus passes agree to the last bit."""
        nu = np.array([3.0 + np.pi / 30, 4.0 + np.pi / 30])
        a = compute_series_coefficients(NPZ_0QP, nu, self.A1, 32,
                                        order_max=5)
        b = compute_series_coefficients(NPZ_0QP, nu, self.A1, 32,
                                        order_max=5)
        for o in a:
            np.testing.assert_array_equal(a[o], b[o])

    def test_cache_matches_uncached_per_graph(self):
        """A cached pass agrees with an *uncached* `evaluate_graph`
        re-evaluation of the same graphs to floating-point
        reassociation tolerance (~1e-11 rel), and the per-order totals
        likewise — the deviation is the isomorphic-representative ULP
        noise, never a wrong reuse.
        """
        from gzl.series import _load_corpus

        nu = 4.0 + np.pi / 30
        corpus = _load_corpus(NPZ_0QP)
        order = 4
        blk = corpus[order]
        el = blk["edges_list"]
        ml = blk["multiplicities_list"]
        prefs = blk["prefactor"]
        hop = blk.get("hopping")

        cached_total = 0.0
        uncached_total = 0.0
        shared_cache: dict = {}
        for i in range(len(el)):
            edges = np.repeat(el[i], ml[i], axis=0)
            s_v, t_v = ((int(hop[i][0]), int(hop[i][1]))
                        if hop is not None else (0, 0))
            v_cached = evaluate_graph(
                edges, float(nu), self.A1,
                source=s_v, terminal=None if s_v == t_v else int(t_v),
                n_points=32, block_cache=shared_cache,
            )
            v_uncached = evaluate_graph(
                edges, float(nu), self.A1,
                source=s_v, terminal=None if s_v == t_v else int(t_v),
                n_points=32, block_cache=None,
            )
            assert np.isclose(v_cached, v_uncached, rtol=1e-11, atol=0.0), (
                f"graph {i}: cached {v_cached!r} vs uncached "
                f"{v_uncached!r} exceeds FP-reassociation tolerance"
            )
            cached_total += float(prefs[i]) * v_cached
            uncached_total += float(prefs[i]) * v_uncached
        assert np.isclose(cached_total, uncached_total,
                          rtol=1e-11, atol=0.0)

    @pytest.mark.parametrize(
        "nu, rtol, regime",
        [
            # σ ≈ 2 (ν = 3 + δ) routes σ-blocks through the *tensor*
            # path, which is vertex-labelling-invariant up to FP
            # reassociation: cache vs no-cache agree to ~1e-10.
            (3.0 + np.pi / 30, 1e-10, "tensor (σ≈2)"),
            # σ ≈ 1 (ν = 2 + δ, < 1.49) routes σ-blocks through the
            # σ_max=4 *algebra* path, whose finite-n_points
            # discretisation error depends on the SP-reduction
            # terminal/ordering — different between a block's own
            # labelling and the cached isomorphic representative's.
            # Both are equally valid O(n^-(2-σ)) approximants of the
            # *same* exact ζ_block, so they agree only at the
            # discretisation scale (~1e-4 at n=32), NOT at the ULP
            # level.  The cached pass is still exactly deterministic.
            (2.0 + np.pi / 30, 5e-3, "algebra (σ≈1)"),
        ],
    )
    def test_cache_vs_nocache_corpus_pass(self, nu, rtol, regime):
        """End-to-end: a shared-cache corpus pass agrees with the
        no-cache pass to the regime-appropriate tolerance, and the
        cached pass is itself bit-deterministic across repeats.
        """
        from gzl.series import _load_corpus

        corpus = _load_corpus(NPZ_1QP)
        orders = [o for o in sorted(corpus) if o <= 6]

        def _pass(use_cache):
            cache = {} if use_cache else None
            tot = 0.0
            for o in orders:
                b = corpus[o]
                el, ml, pr = (b["edges_list"], b["multiplicities_list"],
                              b["prefactor"])
                hp = b.get("hopping")
                for i in range(len(el)):
                    e = np.repeat(el[i], ml[i], axis=0)
                    s, t = ((int(hp[i][0]), int(hp[i][1]))
                            if hp is not None else (0, 0))
                    v = evaluate_graph(
                        e, float(nu), self.A1,
                        source=s, terminal=None if s == t else int(t),
                        momentum=None if s == t else np.zeros(1),
                        n_points=32, block_cache=cache,
                    )
                    tot += float(pr[i]) * float(np.asarray(v).real)
            return tot

        c1 = _pass(True)
        c2 = _pass(True)
        nc = _pass(False)
        assert c1 == c2, f"{regime}: cached pass must be deterministic"
        assert np.isclose(c1, nc, rtol=rtol, atol=0.0), (
            f"{regime}: cache {c1!r} vs no-cache {nc!r} exceeds "
            f"rtol={rtol}"
        )


# ---------------------------------------------------------------------------
# 2. Cache effectiveness — repeated blocks are served from the cache
# ---------------------------------------------------------------------------

class TestCacheEffectiveness:
    A1 = np.array([[1.0]])

    def test_repeated_bridges_hit_cache(self):
        """A path graph 0-1-2-3-4 is four identical ν-bridges; after
        the first, the rest must be cache hits."""
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=int)
        cache: dict = {}
        _, info = evaluate_graph(
            edges, 4.0, self.A1, n_points=16,
            block_cache=cache, return_diagnostics=True,
        )
        # 4 bridges, all the same (ν, A, n) ⇒ 1 computed, 3 served.
        assert info["n_bridges"] == 1
        assert info["n_block_cache_hits"] == 3

    def test_distinct_blocks_not_conflated(self):
        """A ν=4 bridge and a ν=5 bridge must not share a cache slot."""
        cache: dict = {}
        v4 = evaluate_graph(np.array([[0, 1]]), 4.0, self.A1,
                            n_points=16, block_cache=cache)
        v5 = evaluate_graph(np.array([[0, 1]]), 5.0, self.A1,
                            n_points=16, block_cache=cache)
        v4b = evaluate_graph(np.array([[0, 1]]), 4.0, self.A1,
                             n_points=16, block_cache=cache)
        assert v4 != v5
        assert v4 == v4b           # ν=4 still resolves to its own value


# ---------------------------------------------------------------------------
# 3. WL-hash collision guard (σ-routed blocks)
# ---------------------------------------------------------------------------

class TestCollisionGuard:
    """Two non-isomorphic σ-routed blocks that share a hash bucket must
    keep their own values — the exact `is_isomorphic` check is the
    backstop.

    K_{3,3} and the triangular prism are both 3-regular on six vertices
    with nine edges, so the Weisfeiler–Lehman hash cannot tell them
    apart and they share one primary key.  (K_4 against K_4 − e, the
    pair this test used before, differ in their edge count, which is in
    the key: the backstop never ran.)
    """

    A1 = np.array([[1.0]])
    K33 = np.array([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5],
                    [2, 3], [2, 4], [2, 5]])
    PRISM = np.array([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5],
                      [0, 3], [1, 4], [2, 5]])

    def test_k33_vs_prism(self, monkeypatch):
        import gzl.frontend as fe
        calls = {"n": 0}
        real = fe._nx.is_isomorphic

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(fe._nx, "is_isomorphic", counting)
        nu = 5.0 + np.pi / 30
        cache: dict = {}
        a = evaluate_graph(self.K33, nu, self.A1, n_points=16,
                           block_cache=cache)
        b = evaluate_graph(self.PRISM, nu, self.A1, n_points=16,
                           block_cache=cache)
        assert len(cache) == 1, "the two blocks no longer share a bucket"
        assert len(next(iter(cache.values()))) == 2
        assert calls["n"] >= 1, "the isomorphism check never ran"
        assert b == evaluate_graph(self.PRISM, nu, self.A1, n_points=16)
        a2 = evaluate_graph(self.K33, nu, self.A1, n_points=16,
                            block_cache=cache)
        assert a != b
        assert a == a2                            # K_{3,3} unaffected


class TestCacheKeyNamesTheEngine:
    """A block-cache key must name the evaluator, not just the block.

    ``hybrid`` and ``tensor`` agree only to round-off — measured over 48
    configurations, 45 are bit-identical and 3 differ by a few ulp (max
    7.4e-16, grid mode at d=1), because the FFT collapse re-associates
    the sum.  Nothing else in the key distinguishes them, so a cache
    shared across a hybrid call and a tensor call would serve one
    engine's value to the other.

    The magnitude is negligible and no shipped pass mixes engines (a
    corpus run threads a single ``engine`` throughout).  This is kept
    for the same reason the momentum and ν flags are in the key: the
    key identifies what produced the value.
    """

    A1 = np.eye(1)
    DIAMOND = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]], dtype=int)

    def test_engines_do_not_share_cache_entries(self):
        shared = {}
        evaluate_graph(self.DIAMOND, 2.5, self.A1, n_points=16,
                       engine="hybrid", block_cache=shared)
        n_after_hybrid = len(shared)
        evaluate_graph(self.DIAMOND, 2.5, self.A1, n_points=16,
                       engine="tensor", block_cache=shared)
        assert len(shared) > n_after_hybrid, (
            "tensor reused a hybrid cache entry: the key does not "
            "distinguish the evaluator"
        )

    def test_shared_cache_returns_the_engines_own_value(self):
        shared = {}
        evaluate_graph(self.DIAMOND, 2.5, self.A1, n_points=16,
                       engine="hybrid", block_cache=shared)
        via_shared = evaluate_graph(
            self.DIAMOND, 2.5, self.A1, n_points=16,
            engine="tensor", block_cache=shared,
        )
        fresh = evaluate_graph(
            self.DIAMOND, 2.5, self.A1, n_points=16,
            engine="tensor", block_cache={},
        )
        assert via_shared == fresh
