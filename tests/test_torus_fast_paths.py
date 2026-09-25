# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Two fast paths in :mod:`gzl.tensor_network`, and their gates.

Both exist because the dense-block census is evaluated per BLOCK while
its inputs are shared across blocks:

1. **The edge-kernel cache.**  At d = 2, nu = 2.5 the whole shipped
   corpus carries six distinct edge exponents across 13800 edge
   instances, yet ``hybrid_zeta`` rebuilt them for every block --
   measured at ~85 ms of a 113 ms per-block budget, against 246 ms to
   build all six once.  Correctness gate: a cached kernel must be
   BITWISE the freshly built one.  Safety gate: the cache is bounded in
   BYTES, because the same entry count is 0.03 MB at d = 1 n = 4096 and
   134 MB at d = 3 n = 256.

2. **The real-transform convolution** in ``TorusTruncation.compose``.
   Both generators are real, so the complex ``fftn`` carried a zero
   imaginary part through three transforms and discarded it.  This one
   is NOT bit-identical -- a different transform decomposition
   reassociates the sum -- so it is gated on a measured agreement bound
   and carries a kill switch, exactly as ``_USE_CONV`` does.
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl import _sp
from gzl import tensor_network as tn
from gzl.hybrid import hybrid_zeta
from gzl.tensor_network import (
    TorusTruncation,
    _edge_kernel_torus,
    _edge_kernel_torus_uncached,
    _kernel_cache_clear,
)

A1 = np.eye(1)
A2 = np.eye(2)
AHEX = np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2.0]])


class TestEdgeKernelCache:

    def setup_method(self):
        _kernel_cache_clear()

    @pytest.mark.parametrize("d, A", [(1, A1), (2, A2), (2, AHEX)])
    @pytest.mark.parametrize("nu", [1.5, 2.5, 5.0, np.inf])
    @pytest.mark.parametrize("n", [8, 11, 16])
    def test_cached_is_bitwise_the_freshly_built_one(self, d, A, nu, n):
        got = _edge_kernel_torus(nu, A, n)
        assert np.array_equal(got, _edge_kernel_torus_uncached(nu, A, n))
        # ...and again, now served from the cache
        assert np.array_equal(_edge_kernel_torus(nu, A, n),
                              _edge_kernel_torus_uncached(nu, A, n))

    def test_the_lattice_is_part_of_the_key(self):
        r"""Two lattices share ``(nu, n)`` but not their kernels.  Without
        ``A`` in the key a hexagonal cell would silently be served the
        cubic kernel -- finite, plausible, and wrong everywhere.
        """
        cub = _edge_kernel_torus(2.5, A2, 16)
        hexa = _edge_kernel_torus(2.5, AHEX, 16)
        assert not np.array_equal(cub, hexa)
        assert np.array_equal(hexa, _edge_kernel_torus_uncached(2.5, AHEX, 16))

    def test_a_cached_kernel_is_read_only(self):
        r"""Consumers treat it as read-only by contract; the flag makes a
        future consumer that forgets fail loudly instead of corrupting
        every other block that shares the entry.
        """
        K = _edge_kernel_torus(2.5, A2, 16)
        with pytest.raises(ValueError):
            K[0, 0] = 1.0

    def test_the_budget_is_in_bytes_and_is_respected(self, monkeypatch):
        monkeypatch.setattr(tn, "KERNEL_CACHE_MAX_BYTES", 40_000)
        _kernel_cache_clear()
        for nu in (1.5, 2.5, 3.5, 4.5, 5.5, 6.5):
            _edge_kernel_torus(nu, A2, 32)          # 8192 B each
        assert tn._KERNEL_CACHE_BYTES <= 40_000
        assert len(tn._KERNEL_CACHE) < 6            # something was evicted

    def test_an_oversized_kernel_is_served_uncached(self, monkeypatch):
        r"""A single kernel bigger than the whole budget must not evict
        the cache and then fail to fit anyway -- it is returned, just not
        stored."""
        monkeypatch.setattr(tn, "KERNEL_CACHE_MAX_BYTES", 100)
        _kernel_cache_clear()
        K = _edge_kernel_torus(2.5, A2, 32)
        assert np.array_equal(K, _edge_kernel_torus_uncached(2.5, A2, 32))
        assert len(tn._KERNEL_CACHE) == 0

    def test_clear_releases_the_accounting_too(self):
        _edge_kernel_torus(2.5, A2, 16)
        assert tn._KERNEL_CACHE_BYTES > 0
        _kernel_cache_clear()
        assert tn._KERNEL_CACHE_BYTES == 0 and not tn._KERNEL_CACHE


class TestRealTransformCompose:

    @pytest.mark.parametrize("d, A, n", [
        (1, A1, 64), (1, A1, 65), (2, A2, 24), (2, A2, 25), (2, AHEX, 16),
    ])
    def test_it_agrees_with_the_complex_path(self, d, A, n, monkeypatch):
        r"""Agreement to a few ulp of scale, NOT bit-identity.  Odd ``n``
        is parametrised deliberately: ``irfftn`` cannot infer an odd
        final axis length from the half-spectrum, so without an explicit
        ``s=`` it would silently return an even-length axis -- a wrong
        SHAPE, which downstream would broadcast rather than raise.
        """
        g1 = _edge_kernel_torus(2.5, A, n)
        g2 = _edge_kernel_torus(4.0, A, n)
        T = TorusTruncation(n, d, A)
        monkeypatch.setattr(tn, "_USE_RFFT", False)
        ref = T.compose(g1, g2)
        monkeypatch.setattr(tn, "_USE_RFFT", True)
        got = T.compose(g1, g2)
        assert got.shape == ref.shape == g1.shape
        assert np.max(np.abs(got - ref)) <= 1e-13 * np.max(np.abs(ref))

    def test_the_kill_switch_restores_the_complex_arithmetic(
            self, monkeypatch):
        g1 = _edge_kernel_torus(2.5, A2, 16)
        g2 = _edge_kernel_torus(4.0, A2, 16)
        T = TorusTruncation(16, 2, A2)
        monkeypatch.setattr(tn, "_USE_RFFT", False)
        a = T.compose(g1, g2)
        b = np.fft.ifftn(np.fft.fftn(g1) * np.fft.fftn(g2)).real
        assert np.array_equal(a, b)

    def test_a_complex_operand_falls_back(self, monkeypatch):
        r"""``rfftn`` is only valid on real input.  A complex generator
        (a finite-momentum phase has entered the contraction) must take
        the complex path rather than have its imaginary part discarded.
        """
        monkeypatch.setattr(tn, "_USE_RFFT", True)
        g1 = _edge_kernel_torus(2.5, A2, 16).astype(complex)
        g1 = g1 + 1j * np.roll(g1, 1, axis=0)
        g2 = _edge_kernel_torus(4.0, A2, 16)
        T = TorusTruncation(16, 2, A2)
        got = T.compose(g1, g2)
        ref = np.fft.ifftn(np.fft.fftn(g1) * np.fft.fftn(g2)).real
        assert np.allclose(got, ref, rtol=0, atol=1e-13 * np.max(np.abs(ref)))


class TestBridgeCache:
    r"""The SP collapse memoised by canonical SP-tree.

    Series is convolution and parallel is Hadamard; both are associative
    AND commutative, so an SP part's value is a commutative expression
    tree whose canonical name is the flattened, SORTED tree with leaves
    ``('E', nu)``.  Sibling order is free, tree SHAPE is not.

    What the gates below protect is the KEY, because every way of
    getting it wrong returns a plausible finite number rather than
    failing: a key missing the lattice serves a hexagonal cell the cubic
    bridge; a key missing the orientation serves a reversed leg on a
    sheared cell at even n; a key missing the grid serves the wrong
    resolution entirely.
    """

    def setup_method(self):
        _sp._sp_cache_clear()

    E = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (4, 3)]
    NU = [2.5] * 7

    @pytest.mark.parametrize("d, A", [(1, A1), (2, A2), (2, AHEX)])
    @pytest.mark.parametrize("n, sp", [(8, 32), (12, 48), (11, 33)])
    def test_cached_equals_cold(self, d, A, n, sp):
        r"""Bit-identity, not approximate agreement: the cache returns
        the very array the uncached path computed, so anything short of
        equality means the key is wrong rather than the arithmetic.
        """
        _sp._sp_cache_clear()
        cold = hybrid_zeta(self.E, self.NU, A, n, sp_n_points=sp)
        warm = hybrid_zeta(self.E, self.NU, A, n, sp_n_points=sp)
        _sp._sp_cache_clear()
        again = hybrid_zeta(self.E, self.NU, A, n, sp_n_points=sp)
        assert np.array_equal(np.asarray(cold), np.asarray(warm))
        assert np.array_equal(np.asarray(cold), np.asarray(again))

    def test_the_lattice_is_in_the_key(self):
        r"""hybrid hands the rewrite an IDENTITY-lattice truncation on
        purpose (the rules are lattice-independent index arithmetic), so
        the real cell reaches the cache only through the key.  Without
        it, these two would share a bridge.
        """
        _sp._sp_cache_clear()
        cub = hybrid_zeta(self.E, self.NU, A2, 12, sp_n_points=48)
        hexa = hybrid_zeta(self.E, self.NU, AHEX, 12, sp_n_points=48)
        _sp._sp_cache_clear()
        assert cub == hybrid_zeta(self.E, self.NU, A2, 12, sp_n_points=48)
        _sp._sp_cache_clear()
        assert hexa == hybrid_zeta(self.E, self.NU, AHEX, 12, sp_n_points=48)
        assert cub != hexa

    def test_the_grid_is_in_the_key(self):
        _sp._sp_cache_clear()
        a = hybrid_zeta(self.E, self.NU, A2, 12, sp_n_points=48)
        b = hybrid_zeta(self.E, self.NU, A2, 12, sp_n_points=96)
        _sp._sp_cache_clear()
        assert a == hybrid_zeta(self.E, self.NU, A2, 12, sp_n_points=48)
        assert a != b

    def test_distinct_exponents_do_not_share_a_bridge(self):
        _sp._sp_cache_clear()
        a = hybrid_zeta(self.E, self.NU, A2, 12, sp_n_points=48)
        b = hybrid_zeta(self.E, [3.5] * 7, A2, 12, sp_n_points=48)
        assert a != b

    def test_tree_shape_is_load_bearing(self):
        r"""``P(nu, S(nu, nu))`` and ``P(nu, nu, nu)`` carry the same nu
        multiset and are different numbers.  A crude "(top kind, sorted
        nu multiset)" key would conflate them.
        """
        _sp._sp_cache_clear()
        # 0-1 doubled, plus a 2-path 0-2-1: P(nu, nu, S(nu, nu))
        a = hybrid_zeta([(0, 1), (0, 1), (0, 2), (2, 1)], [2.5] * 4,
                        A2, 12, sp_n_points=48)
        # 0-1 tripled: P(nu, nu, nu)
        b = hybrid_zeta([(0, 1), (0, 1), (0, 1)], [2.5] * 3,
                        A2, 12, sp_n_points=48)
        assert a != b

    def test_the_budget_is_respected_and_clear_resets_it(self, monkeypatch):
        monkeypatch.setattr(_sp, "SP_CACHE_MAX_BYTES", 20_000)
        _sp._sp_cache_clear()
        for nu in (2.5, 3.0, 3.5, 4.0, 4.5, 5.0):
            hybrid_zeta(self.E, [nu] * 7, A2, 12, sp_n_points=48)
        assert _sp._SP_CACHE_BYTES <= 20_000
        _sp._sp_cache_clear()
        assert _sp._SP_CACHE_BYTES == 0 and not _sp._SP_CACHE

    def test_it_actually_hits(self):
        r"""Liveness.  A cache that never hits would pass every
        correctness gate above while buying nothing.
        """
        _sp._sp_cache_clear()
        for nu_extra in (2.5, 5.0, 7.5):
            hybrid_zeta(self.E + [(0, 1)], self.NU + [nu_extra],
                        A2, 12, sp_n_points=48)
        assert _sp._SP_CACHE_STATS["hit"] > 0
