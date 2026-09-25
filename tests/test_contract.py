# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The shared factor-supply layer and the two Truncation subclasses.

``gzl._contract`` holds the pure index arithmetic both engines'
factor supplies share; ``TorusTruncation`` (in ``tensor_network``) and
``BoxTruncation`` (in ``direct_sum``) wrap it with each engine's
geometry.  Three kinds of thing are pinned here:

1. **Delegation moved no value.**  The facade wrappers must reproduce
   the arithmetic they used to own, bit for bit, and the intra-module
   patch chains must survive — a test that monkeypatches
   ``tensor_network._edge_difference_table`` must still intercept the
   lazy ``_factor_array`` build.

2. **The behavioural differences are asserted directly, never carried
   as flags.**  Torus row sums are constant (a circulant table's rows
   are permutations of the generator); box row sums are not (measured
   spread well over 20%).  The torus's ``full(N, gen.sum())`` peel
   shortcut is exact on one truncation and wrong on the other, and any
   shared code path must learn that from a method call, not from a
   boolean someone remembered to set.

3. **``axis_size`` is explicit.**  Deriving it from ``gen.size`` is a
   circulant-only identity — exact on the torus, wrong by
   ``(2 - 1/(2L+1))^d`` on the box's difference-range generator.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import gzl._contract as ct
import gzl.direct_sum as ds
import gzl.tensor_network as tn

A2_TRI = np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2.0]])


# ---------------------------------------------------------------------------
# 1. Delegation
# ---------------------------------------------------------------------------

class TestDelegationMovedNoValue:

    @staticmethod
    def _reference_circulant_table(K, n):
        """The body tensor_network owned before the move, verbatim."""
        d = int(K.ndim)
        n_total = n ** d
        if d == 1:
            i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n),
                                       indexing="ij")
            return K[(i_idx - j_idx) % n]
        base = np.arange(n)
        ax = (base[:, None] - base[None, :]) % n
        idx = []
        for c in range(d):
            shape = [1] * (2 * d)
            shape[c] = shape[d + c] = n
            idx.append(ax.reshape(shape))
        return K[tuple(idx)].reshape(n_total, n_total)

    @pytest.mark.parametrize("d,n", [(1, 8), (1, 13), (2, 6), (3, 4)])
    def test_circulant_table_is_bit_identical_to_the_old_body(self, d, n):
        rng = np.random.default_rng(d * 100 + n)
        K = rng.standard_normal((n,) * d)
        assert np.array_equal(tn._edge_difference_table(K, n),
                              self._reference_circulant_table(K, n))

    @pytest.mark.parametrize("d,n", [(1, 8), (2, 6), (3, 4)])
    def test_reverse_generator_matches_the_cyclic_map(self, d, n):
        rng = np.random.default_rng(d)
        K = rng.standard_normal((n,) * d)
        rev = tn._reverse_generator(K)
        # K_rev[m] = K[(-m) mod n], checked pointwise.
        for idx in itertools.product(range(n), repeat=d):
            neg = tuple((-i) % n for i in idx)
            assert rev[idx] == K[neg]

    def test_patch_chain_survives_the_move(self, monkeypatch):
        """Patching the facade name must still intercept the lazy build.

        The mutation-detector in test_executor_goldens and the
        orientation tests all rely on this chain; if ``_factor_array``
        ever called ``_contract.circulant_table`` directly, they would
        go vacuous without failing.
        """
        gen = np.arange(1.0, 9.0)
        sentinel = np.full((8, 8), 7.0)
        monkeypatch.setattr(tn, "_edge_difference_table",
                            lambda K, n: sentinel)
        out = tn._factor_array(None, gen)
        assert out is sentinel

    def test_box_helpers_still_reproduce_positions(self):
        """The ds wrappers delegate; the position-level reference in
        test_direct_sum_factors keeps them honest.  Here: one spot
        check that the delegation composes."""
        L, d = 3, 2
        gen = ds._conv_kernel_diff(2.5, np.eye(d), L, d)
        t = ds._table_from_generator(
            gen, ds._axis_extents(L, d, False), ds._axis_extents(L, d, True),
            ds._gen_offsets(L, d, ds._axis_origins(L, d, False),
                            ds._axis_origins(L, d, True)), d)
        assert t.shape == ((2 * L + 1) ** d, (L + 1) * (2 * L + 1))


# ---------------------------------------------------------------------------
# 2. The Truncation base is abstract
# ---------------------------------------------------------------------------

def test_every_base_method_raises():
    t = ct.Truncation()
    for name, args in [("axis_size", (0,)), ("coords", (0,)),
                       ("generator", (2.5,)), ("table", (None, 0, 1)),
                       ("pin_slot", (None, 0, True)), ("reverse", (None,)),
                       ("weight", (0,)), ("empty_bucket_scalar", (0,)),
                       ("peel_constant", (None, 0))]:
        with pytest.raises(NotImplementedError):
            getattr(t, name)(*args)


def test_factor_defaults():
    f = ct.Factor(scope=(0, 1))
    assert f.peelable is True and f.arr is None and f.gen is None


# ---------------------------------------------------------------------------
# 3. TorusTruncation
# ---------------------------------------------------------------------------

class TestTorusTruncation:

    @pytest.mark.parametrize("d,n,A", [(1, 16, np.eye(1)), (2, 8, A2_TRI),
                                       (3, 4, np.eye(3))])
    def test_supply_matches_the_engine(self, d, n, A):
        tr = tn.TorusTruncation(n, d, A)
        gen = tr.generator(2.5)
        assert np.array_equal(gen, tn._edge_kernel_torus(2.5, A, n))
        assert tr.axis_size(0) == n ** d
        T = tr.table(gen, 0, 1)
        assert np.array_equal(T, tn._edge_difference_table(gen, n))
        # pin_slot: row 0 and column 0 of the table, both orientations.
        assert np.array_equal(tr.pin_slot(gen, 0, True), T[0, :])
        assert np.array_equal(tr.pin_slot(gen, 0, False), T[:, 0])
        assert tr.weight(0) is None
        assert tr.empty_bucket_scalar(0) == float(n ** d)

    def test_coords_are_the_balanced_z_labels(self):
        n, d = 8, 2
        tr = tn.TorusTruncation(n, d, A2_TRI)
        z = tn._balanced_z_axis(n)
        want = np.stack(np.meshgrid(z, z, indexing="ij"),
                        axis=-1).reshape(-1, 2)
        assert np.array_equal(tr.coords(0), want)
        assert tr.coords(0).shape == (tr.axis_size(0), d)

    @pytest.mark.parametrize("d,n,A", [(1, 16, np.eye(1)), (2, 8, A2_TRI)])
    def test_peel_constant_equals_the_true_row_sums(self, d, n, A):
        """The shortcut is exact BECAUSE circulant row sums are constant
        — both halves asserted, so neither can rot independently."""
        tr = tn.TorusTruncation(n, d, A)
        gen = tr.generator(2.5)
        T = tr.table(gen, 0, 1)
        row_sums = T.sum(axis=1)
        spread = (row_sums.max() - row_sums.min()) / abs(row_sums).max()
        assert spread <= 1e-15, f"circulant row sums not constant: {spread}"
        got = tr.peel_constant(gen, 0)
        assert np.max(np.abs(got - row_sums)) <= \
            4 * np.spacing(np.abs(row_sums).max())

    def test_axis_size_equals_gen_size_only_here(self):
        """The circulant identity the box does NOT satisfy."""
        tr = tn.TorusTruncation(8, 2, A2_TRI)
        assert tr.axis_size(0) == tr.generator(2.5).size


# ---------------------------------------------------------------------------
# 4. BoxTruncation
# ---------------------------------------------------------------------------

def _positions(L, d, half):
    axes = ([np.arange(0, L + 1)] if half else [np.arange(-L, L + 1)])
    axes += [np.arange(-L, L + 1)] * (d - 1)
    return np.array(list(itertools.product(*axes)), dtype=float)


class TestBoxTruncation:

    @pytest.mark.parametrize("d,L,A", [(1, 4, np.eye(1)), (2, 3, A2_TRI),
                                       (3, 2, np.eye(3))])
    def test_supply_matches_the_engine(self, d, L, A):
        tr = ds.BoxTruncation(L, d, A, marker=2)
        gen = tr.generator(2.5)
        assert np.array_equal(gen, ds._conv_kernel_diff(2.5, A, L, d))
        assert tr.axis_size(0) == (2 * L + 1) ** d
        assert tr.axis_size(2) == (L + 1) * (2 * L + 1) ** (d - 1)
        # Table against the position-built reference, marker and not.
        for a, b in [(0, 1), (2, 1), (0, 2)]:
            pa = _positions(L, d, a == 2)
            pb = _positions(L, d, b == 2)
            diff = pb[None, :, :] - pa[:, None, :]
            dist = np.linalg.norm(diff @ A.T, axis=-1)
            with np.errstate(divide="ignore"):
                want = dist ** (-2.5)
            want[dist == 0.0] = 0.0
            assert np.array_equal(tr.table(gen, a, b), want), (a, b)

    def test_pin_slot_is_the_kernel_at_the_positions(self):
        L, d, A = 4, 2, A2_TRI
        tr = ds.BoxTruncation(L, d, A, marker=2)
        gen = tr.generator(3.0)
        for v in (0, 2):
            pos = _positions(L, d, v == 2)
            dist = np.linalg.norm(pos @ A.T, axis=-1)
            with np.errstate(divide="ignore"):
                want = dist ** (-3.0)
            want[dist == 0.0] = 0.0
            got = tr.pin_slot(gen, v, pin_row=False)
            assert np.array_equal(got, want)
            # The kernel is even, so orientation is a numerical no-op
            # on every lattice this engine builds — assert that too,
            # since pin_slot honours the flag for generality.
            assert np.allclose(tr.pin_slot(gen, v, pin_row=True), want,
                               rtol=1e-15, atol=0)

    def test_weights_and_empty_bucket(self):
        L, d = 4, 1
        tr = ds.BoxTruncation(L, d, np.eye(1), marker=2)
        w = tr.weight(2)
        assert w is not None and w.sum() == tr.n_full
        assert np.array_equal(w, np.where(np.arange(L + 1) == 0, 1.0, 2.0))
        assert tr.weight(0) is None
        # The half-box defect class: the marker's empty bucket must be
        # the FULL count (weights sum), never the half-axis size.
        assert tr.empty_bucket_scalar(2) == float(tr.n_full)
        assert tr.empty_bucket_scalar(2) != float(tr.axis_size(2))
        assert tr.empty_bucket_scalar(0) == float(tr.n_full)

    def test_weights_at_d2_encode_the_half_box_pattern(self):
        L, d = 3, 2
        tr = ds.BoxTruncation(L, d, np.eye(2), marker=1)
        pos = _positions(L, d, True)
        want = np.where(pos[:, 0] == 0.0, 1.0, 2.0)
        assert np.array_equal(tr.weight(1), want)
        assert tr.weight(1).sum() == tr.n_full

    @pytest.mark.parametrize("d,L,A", [(1, 4, np.eye(1)), (2, 3, A2_TRI)])
    def test_peel_constant_is_not_constant_here(self, d, L, A):
        """The other half of the torus assertion: box row sums vary, by
        a lot — a shared path assuming constancy would be wrong on
        every box call that reached it."""
        tr = ds.BoxTruncation(L, d, A, marker=None)
        gen = tr.generator(2.5)
        got = tr.peel_constant(gen, 0)
        T = tr.table(gen, 0, 1)
        assert np.array_equal(got, T.sum(axis=1))
        spread = (got.max() - got.min()) / got.max()
        assert spread > 0.20, (
            f"box row-sum spread fell to {spread:.3f}; if this became "
            f"constant the torus shortcut would quietly become valid "
            f"here and this test's premise is stale"
        )

    def test_coords_match_the_positions(self):
        L, d = 3, 2
        tr = ds.BoxTruncation(L, d, np.eye(2), marker=1)
        assert np.array_equal(tr.coords(0), _positions(L, d, False))
        assert np.array_equal(tr.coords(1), _positions(L, d, True))

    def test_axis_size_never_equals_gen_size(self):
        """The hazard the explicit method exists for, as numbers."""
        for d, L in [(1, 4), (2, 3), (3, 2)]:
            tr = ds.BoxTruncation(L, d, np.eye(d), marker=None)
            gen = tr.generator(2.5)
            assert gen.size == (2 * (2 * L + 1) - 1) ** d
            assert tr.axis_size(0) == (2 * L + 1) ** d
            ratio = gen.size / tr.axis_size(0)
            assert ratio == pytest.approx((2 - 1 / (2 * L + 1)) ** d)
            assert gen.size != tr.axis_size(0)

    def test_generator_cache_is_shared_with_the_engine_convention(self):
        """float(nu) keys, coexisting with ('hat', nu, m) tuples."""
        cache = {}
        tr = ds.BoxTruncation(3, 1, np.eye(1), kdiff_cache=cache)
        g1 = tr.generator(2.5)
        g2 = tr.generator(2.5)
        assert g1 is g2 and set(cache) == {2.5}
        cache[("hat", 2.5, 7)] = "sentinel"
        assert tr.generator(2.5) is g1     # tuple keys never collide


# ---------------------------------------------------------------------------
# 5. Cross-truncation: the same abstract question, two honest answers
# ---------------------------------------------------------------------------

def test_empty_bucket_scalars_agree_on_what_they_count():
    """Both truncations count the vertex's positions — n^d on the torus
    and (2L+1)^d on the box — and so do both engines, through the shared
    loop.  The case is unreachable from the frontend (edge lists cannot
    carry isolated vertices; every shipped corpus is gap-free), and the
    method pins the correct definition.  If this test ever fails because
    the torus number changed, check the gap-label probe in
    tests/test_shared_loop.py (TestIsolatedVertexSemantics) before
    "fixing" either side.
    """
    tor = tn.TorusTruncation(8, 2, np.eye(2))
    box = ds.BoxTruncation(4, 2, np.eye(2), marker=None)
    assert tor.empty_bucket_scalar(0) == 8 ** 2
    assert box.empty_bucket_scalar(0) == 9 ** 2
