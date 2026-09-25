# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The shared bucket skeleton, driven through both engines.

``_contract.eliminate_one`` owns control flow only; every arithmetic
decision routes through a Truncation method whose body is its engine's
shipped code.  Three structural properties are pinned here, because the
value tests (the frozen goldens, the conv parity suites) cannot see
them:

1. **Late binding survives the extraction.**  A raising stub patched
   onto ``direct_sum._fft_conv_step`` or
   ``tensor_network._peel_convolution`` must abort a real engine call —
   if the skeleton or a Truncation captured those names at import or
   construction time, the stub would be bypassed and the suite's peel
   sentinels would go quietly vacuous.

2. **Both engines actually run through the skeleton.**  Otherwise this
   file certifies a code path nothing uses.

3. **The one deliberate semantic change is what it claims to be.**  The
   torus used to *skip* a vertex with no incident factor (contributing
   1 where the sum over its positions is n^d); the box counted it
   correctly.  Both now count — asserted as cross-engine agreement on
   the gap-label input where they used to disagree.
"""

from __future__ import annotations

import numpy as np
import pytest

import gzl._contract as ct
import gzl.direct_sum as ds
import gzl.tensor_network as tn

K4 = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])


class TestLateBindingSurvives:
    """Raising stubs on the module attributes must reach the executor."""

    def test_box_peel_resolves_through_the_module_dict(self):
        class Boom(RuntimeError):
            pass

        def stub(*a, **kw):
            raise Boom

        old = (ds._fft_conv_step, ds._FFT_MARGIN, ds._FFT_MIN_BAG)
        ds._fft_conv_step = stub
        ds._FFT_MARGIN, ds._FFT_MIN_BAG = 0.0, 0
        try:
            with pytest.raises(Boom):
                ds.direct_sum_zero_momentum(
                    K4, np.full(6, 2.5), np.eye(1), 6, root=0)
        finally:
            ds._fft_conv_step, ds._FFT_MARGIN, ds._FFT_MIN_BAG = old

    def test_torus_peel_resolves_through_the_module_dict(self):
        class Boom(RuntimeError):
            pass

        def stub(*a, **kw):
            raise Boom

        old = tn._peel_convolution
        tn._peel_convolution = stub
        try:
            with pytest.raises(Boom):
                tn.graph_zeta_general_at_zero(
                    K4, np.full(6, 2.5), np.eye(1), 16)
        finally:
            tn._peel_convolution = old

    def test_kill_switch_still_bypasses_the_stub(self):
        """The same stub with _USE_CONV off must never be reached —
        the flag is read per step through the module dict, so the
        raising stub doubles as the flag-half detector."""
        def stub(*a, **kw):                      # pragma: no cover
            raise RuntimeError("peel ran with the switch off")

        old_tn = (tn._peel_convolution, tn._USE_CONV)
        tn._peel_convolution, tn._USE_CONV = stub, False
        try:
            tn.graph_zeta_general_at_zero(K4, np.full(6, 2.5), np.eye(1), 16)
        finally:
            tn._peel_convolution, tn._USE_CONV = old_tn

        old_ds = (ds._fft_conv_step, ds._USE_CONV,
                  ds._FFT_MARGIN, ds._FFT_MIN_BAG)
        ds._fft_conv_step, ds._USE_CONV = stub, False
        ds._FFT_MARGIN, ds._FFT_MIN_BAG = 0.0, 0
        try:
            ds.direct_sum_zero_momentum(K4, np.full(6, 2.5), np.eye(1), 6,
                                        root=0)
        finally:
            (ds._fft_conv_step, ds._USE_CONV,
             ds._FFT_MARGIN, ds._FFT_MIN_BAG) = old_ds


class TestBothEnginesUseTheSkeleton:

    def test_box_steps_run_through_eliminate_one(self):
        calls = []
        real = ct.eliminate_one

        def spy(factors, v, trunc):
            calls.append((type(trunc).__name__, v))
            return real(factors, v, trunc)

        ct.eliminate_one = spy
        try:
            ds.direct_sum_zero_momentum(K4, np.full(6, 2.5), np.eye(1), 5,
                                        root=0)
        finally:
            ct.eliminate_one = real
        assert calls, "the box engine bypassed the shared skeleton"
        assert all(name == "BoxTruncation" for name, _ in calls)

    def test_torus_steps_run_through_eliminate_one(self):
        calls = []
        real = ct.eliminate_one

        def spy(factors, v, trunc):
            calls.append((type(trunc).__name__, v))
            return real(factors, v, trunc)

        ct.eliminate_one = spy
        try:
            tn.graph_zeta_general_at_zero(K4, np.full(6, 2.5), np.eye(1), 8)
        finally:
            ct.eliminate_one = real
        assert calls, "the torus engine bypassed the shared skeleton"
        assert all(name == "TorusTruncation" for name, _ in calls)


class TestIsolatedVertexSemantics:
    """Below the label boundary both engines count an isolated vertex.

    A gap in the vertex labelling (edges ``0-2-4`` with labels 1 and 3
    absent) creates degree-0 free vertices INSIDE the engines.  The two
    engine loops used to *disagree* on them (the box counted each
    isolated vertex's positions, the torus silently skipped it); the
    shared-loop adoption made both count.  Since the label-convention
    unification (gzl/_labels.py) the case is unreachable from
    ANY public entry — the vertex set is the edge support, gaps are
    compressed away — so the engine-level semantics are probed with the
    normalisation bypassed, and the public boundary is pinned
    separately below (and across all engines in
    tests/test_label_convention.py).
    """

    GAP = np.array([[0, 2], [2, 4]])
    COMPACT = np.array([[0, 1], [1, 2]])
    NU = np.full(2, 2.5)

    @staticmethod
    def _bypass(monkeypatch):
        ident = lambda edges, refs=None: (edges, refs)  # noqa: E731
        monkeypatch.setattr(tn, "_relabel_to_support", ident)
        monkeypatch.setattr(ds, "_relabel_to_support", ident)

    def test_torus_counts_positions_below_the_boundary(self, monkeypatch):
        self._bypass(monkeypatch)
        n = 8
        v_gap = complex(np.asarray(tn.graph_zeta_general_at_zero(
            self.GAP, self.NU, np.eye(1), n)).item()).real
        v_ref = complex(np.asarray(tn.graph_zeta_general_at_zero(
            self.COMPACT, self.NU, np.eye(1), n)).item()).real
        # Two isolated vertices, n^d positions each.
        assert v_gap == pytest.approx(v_ref * n ** 2, rel=1e-12)

    def test_box_counts_positions_below_the_boundary(self, monkeypatch):
        self._bypass(monkeypatch)
        L = 4
        n_full = 2 * L + 1
        v_gap = complex(ds.direct_sum_zero_momentum(
            self.GAP, self.NU, np.eye(1), L, root=0)).real
        v_ref = complex(ds.direct_sum_zero_momentum(
            self.COMPACT, self.NU, np.eye(1), L, root=0)).real
        assert v_gap == pytest.approx(v_ref * n_full ** 2, rel=1e-12)

    def test_the_two_engines_agree_on_the_ratio(self, monkeypatch):
        """The cross-engine inconsistency the shared loop removed."""
        self._bypass(monkeypatch)
        n, L = 8, 4
        r_tn = (complex(np.asarray(tn.graph_zeta_general_at_zero(
            self.GAP, self.NU, np.eye(1), n)).item()).real
            / complex(np.asarray(tn.graph_zeta_general_at_zero(
                self.COMPACT, self.NU, np.eye(1), n)).item()).real)
        r_ds = (complex(ds.direct_sum_zero_momentum(
            self.GAP, self.NU, np.eye(1), L, root=0)).real
            / complex(ds.direct_sum_zero_momentum(
                self.COMPACT, self.NU, np.eye(1), L, root=0)).real)
        assert r_tn == pytest.approx(float(8 ** 2), rel=1e-12)
        assert r_ds == pytest.approx(float((2 * L + 1) ** 2), rel=1e-12)

    def test_the_public_boundary_compresses_the_gap_away(self):
        """Through the PUBLIC entries the same gap input is the same
        graph: no bypass, ratio exactly 1 in both engines."""
        n, L = 8, 4
        assert complex(np.asarray(tn.graph_zeta_general_at_zero(
            self.GAP, self.NU, np.eye(1), n)).item()) \
            == complex(np.asarray(tn.graph_zeta_general_at_zero(
                self.COMPACT, self.NU, np.eye(1), n)).item())
        assert complex(ds.direct_sum_zero_momentum(
            self.GAP, self.NU, np.eye(1), L, root=0)) \
            == complex(ds.direct_sum_zero_momentum(
                self.COMPACT, self.NU, np.eye(1), L, root=0))


class TestSkeletonBranchEquivalence:
    """Why moving the torus scalar branch after the peel attempt is a
    pure reordering: a peelable factor has axes {v, u} with u != v, so
    its bucket's output-axis set is never empty.  Asserted on the
    structure rather than argued in a comment.
    """

    def test_no_partner_exists_when_out_axes_is_empty(self):
        n = 8
        gen = tn._edge_kernel_torus(2.5, np.eye(1), n)
        # A bucket of pure 1-axis factors (out_axes would be empty).
        bucket = [([1], gen.reshape(-1).copy(), None),
                  ([1], None, gen)]
        assert tn._find_conv_peel(bucket, 1) is None

    def test_two_axis_factor_forces_nonempty_out_axes(self):
        n = 8
        gen = tn._edge_kernel_torus(2.5, np.eye(1), n)
        bucket = [([1, 2], None, gen)]
        found = tn._find_conv_peel(bucket, 1)
        assert found is not None
        _, u = found
        assert u == 2      # ... which is an output axis by definition
