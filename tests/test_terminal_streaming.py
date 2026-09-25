# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The streaming path's per-terminal-index table rebuild.

Pre-fix, ``_streaming_path`` materialised the full ``(n^d, n^d)``
circulant table of every still-lazy terminal-incident kernel ONCE PER
TERMINAL INDEX only to ``np.take`` one slice — ``n^(d·|terminals|)``
table builds per call (measured 24 rebuilds at n = 24 against 0 on the
batched path).  The fix gathers the pinned row/column straight off the
generator with :func:`_pin_from_generator_at`.  Gates:

1. the gather is BIT-IDENTICAL to slicing the materialised table, for
   every index, both orientations, d = 1, 2, 3 — so the fix is a pure
   substitution and the existing streaming-parity tolerances cannot
   have moved;
2. on a graph whose only lazy kernel is terminal-incident and whose
   streamed suffix runs table-free, the build count is ZERO at every
   torus size (pre-fix: one build per terminal index).

Suffix-internal dense consumptions of non-terminal lazy kernels still
rebuild per index — that is the streaming path's deliberate
memory-for-time trade, distinct from the defect fixed here.
"""

from __future__ import annotations

import numpy as np
import pytest

import gzl.tensor_network as tn

K4_TAIL = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
                    (3, 4)])
PATH3 = np.array([(0, 1), (1, 2)])


class TestPinFromGeneratorAt:
    @pytest.mark.parametrize("n,d", [(5, 1), (7, 1), (4, 2), (3, 3)])
    def test_bitwise_equal_to_table_slice(self, n, d):
        rng = np.random.default_rng(5)
        gen = rng.random((n,) * d)          # shaped, as kernels are
        table = tn._edge_difference_table(gen, n)
        for t in range(n ** d):
            row = tn._pin_from_generator_at(gen, t, pin_row=True)
            col = tn._pin_from_generator_at(gen, t, pin_row=False)
            assert np.array_equal(row, np.take(table, t, axis=0))
            assert np.array_equal(col, np.take(table, t, axis=1))

    def test_index_zero_matches_the_special_case(self):
        rng = np.random.default_rng(6)
        gen = rng.random((4, 4))
        for pin_row in (True, False):
            assert np.array_equal(
                tn._pin_from_generator_at(gen, 0, pin_row),
                tn._pin_from_generator(gen, pin_row),
            )


class TestNoPerIndexTableRebuild:
    def _count_builds(self, n):
        calls = {"n": 0}
        orig = tn._edge_difference_table

        def counting(gen, n_points):
            calls["n"] += 1
            return orig(gen, n_points)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(tn, "_edge_difference_table", counting)
            mp.setattr(tn, "_FORCE_STREAMING", True)
            mp.setattr(tn, "_MEM_BUDGET_BYTES", 1)
            tn.graph_zeta_general(
                PATH3, np.full(len(PATH3), 2.5), np.eye(1), n,
                source=0, terminals=(2,), space="z",
            )
        return calls["n"]

    def test_terminal_pinning_builds_no_table_at_all(self):
        """PATH3's only lazy kernel is the terminal-incident (1, 2);
        its (0, 1) partner is source-pinned off the generator and the
        streamed suffix multiplies 1-axis vectors.  So every table
        build in this call would be the per-index rebuild the fix
        removes: the count must be zero, at every size."""
        assert self._count_builds(8) == 0
        assert self._count_builds(12) == 0

    def test_streamed_value_still_matches_batched(self, monkeypatch):
        nu = np.full(len(K4_TAIL), 2.5)
        batched = np.asarray(tn.graph_zeta_general(
            K4_TAIL, nu, np.eye(1), 8, source=0, terminals=(4,),
            space="z"))
        monkeypatch.setattr(tn, "_FORCE_STREAMING", True)
        monkeypatch.setattr(tn, "_MEM_BUDGET_BYTES", 1)
        streamed = np.asarray(tn.graph_zeta_general(
            K4_TAIL, nu, np.eye(1), 8, source=0, terminals=(4,),
            space="z"))
        np.testing.assert_allclose(streamed, batched, rtol=1e-12)


# ---------------------------------------------------------------------------
# Streaming ported to the box open-terminal engine
# ---------------------------------------------------------------------------

import gzl.direct_sum as ds  # noqa: E402


def _em(edges, nu_val=2.5):
    nu = np.full(len(edges), nu_val)
    return ds._collapse_multi_edges(np.asarray(edges), nu)


class TestBoxSliceAt:
    @pytest.mark.parametrize("d,L", [(1, 4), (2, 2)])
    def test_bitwise_equal_to_table_slice(self, d, L):
        """The per-position gather equals slicing the materialised
        Toeplitz table, both pinning directions, every index."""
        rng = np.random.default_rng(9)
        gen = rng.random((4 * L + 1,) * d)
        ext = tuple([2 * L + 1] * d)
        off = tuple([2 * L] * d)
        table = ds._table_from_generator(gen, ext, ext, off, d)
        n = (2 * L + 1) ** d
        for t in range(n):
            t_idx = np.unravel_index(t, ext)
            row = ds._box_slice_at(gen, ext, off, d, t_idx,
                                   keep_is_b=True)
            col = ds._box_slice_at(gen, ext, off, d, t_idx,
                                   keep_is_b=False)
            assert np.array_equal(row, np.take(table, t, axis=0))
            assert np.array_equal(col, np.take(table, t, axis=1))


class TestBoxOpenTerminalStreaming:
    @pytest.mark.parametrize("d,L", [(1, 4), (1, 6), (2, 3)])
    @pytest.mark.parametrize("use_symmetry", [True, False])
    def test_forced_streaming_matches_batched(self, d, L,
                                              use_symmetry,
                                              monkeypatch):
        em = _em(K4_TAIL)
        A = np.eye(d)
        M_b, pos_b = ds._direct_sum_open_terminal(
            em, A, L, 0, 4, d, use_symmetry=use_symmetry)
        monkeypatch.setattr(ds, "_FORCE_STREAMING", True)
        M_s, pos_s = ds._direct_sum_open_terminal(
            em, A, L, 0, 4, d, use_symmetry=use_symmetry)
        assert np.array_equal(pos_s, pos_b)
        np.testing.assert_allclose(M_s, M_b, rtol=1e-12)

    def test_budget_breach_engages_streaming(self, monkeypatch):
        """With a tiny memory budget the schedule must switch on its
        own (no force flag) and still reproduce the batched M.  The
        engagement sentinel is the per-position gather helper — the
        value matches either way, so without it a broken switch would
        be invisible."""
        em = _em(K4_TAIL)
        M_b, _ = ds._direct_sum_open_terminal(em, np.eye(1), 5, 0, 4, 1)
        calls = {"n": 0}
        orig = ds._box_slice_at

        def counting(*a, **kw):
            calls["n"] += 1
            return orig(*a, **kw)

        monkeypatch.setattr(ds, "_box_slice_at", counting)
        monkeypatch.setattr(ds, "_MEM_BUDGET_BYTES", 64)
        M_s, _ = ds._direct_sum_open_terminal(em, np.eye(1), 5, 0, 4, 1)
        assert calls["n"] >= 1, "budget breach did not engage streaming"
        np.testing.assert_allclose(M_s, M_b, rtol=1e-12)

    def test_grid_extrapolated_end_to_end_under_streaming(
            self, monkeypatch):
        """The Richardson grid ladder on top of streamed per-L
        eliminations must reproduce the batched dispersion."""
        k_grid = np.array([[0.0], [0.2], [0.4]])
        nu = np.full(len(K4_TAIL), 2.5)
        ref = ds._direct_sum_extrapolated_grid(
            K4_TAIL, nu, np.eye(1), k_grid, root=0, terminal=4,
            L_list=(3, 4, 5, 6))
        monkeypatch.setattr(ds, "_FORCE_STREAMING", True)
        streamed = ds._direct_sum_extrapolated_grid(
            K4_TAIL, nu, np.eye(1), k_grid, root=0, terminal=4,
            L_list=(3, 4, 5, 6))
        np.testing.assert_allclose(streamed, ref, rtol=1e-10)

    def test_mid_schedule_split_keeps_prefix_intermediates_shaped(
            self, monkeypatch):
        """Pinned defect: a budget split in the MIDDLE of the schedule
        pins the terminal inside a shaped multi-axis prefix intermediate;
        flattening that residue crashes _dense_step_sliced /
        _chunked_peel, which index one array axis per scope vertex.  The
        k_split = 0 tests are structurally blind to this (only original
        <= 2-scope kernels exist there).  This graph + an 8 MB budget
        lands the split after a 3-axis intermediate at d = 2, L = 3, and
        the gate-closed suffix routes its 49^4-element bag to the sliced
        dense branch."""
        edges = ([(1, 2), (1, 3), (1, 6), (2, 4), (2, 5), (3, 4),
                  (3, 5), (4, 5), (4, 6), (5, 6), (0, 4), (0, 5),
                  (7, 3), (7, 4), (7, 5)]
                 + [(0, v) for v in range(8, 15)])
        em = _em(edges)
        M_b, _ = ds._direct_sum_open_terminal(em, np.eye(2), 3, 0, 6, 2)
        monkeypatch.setattr(ds, "_MEM_BUDGET_BYTES", 8 * 1024 * 1024)
        M_s, _ = ds._direct_sum_open_terminal(em, np.eye(2), 3, 0, 6, 2)
        np.testing.assert_allclose(M_s, M_b, rtol=1e-12)
