# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Parity of the FFT elimination fast path in ``gzl.direct_sum``.

Elimination steps whose bucket isolates one original two-vertex kernel
are contracted by zero-padded real FFT (``_eliminate_all`` /
``_fft_conv_step``).  These tests A/B the fast path against the dense
branch via the private ``_USE_CONV`` switch: both are exact
contractions differing only in summation order, so agreement is
required at ``1e-12`` relative (measured ~1e-13 at the worst
``ν = d + 0.25``).

Coverage notes:

* K4 (and every core with merged ψ factors mid-elimination) exercises
  the structural fallback — the second K4 step has bucket
  ``{K(b), K(b−c), K(t−b), ψ₁}`` where no kernel can be peeled because
  ψ₁ carries the would-be partner; that step must run dense.
* The path graph with ``root = 1`` makes the sole convolution partner
  of the first-eliminated vertex the Z₂ marker, exercising the
  ``u == marker`` refusal.
* A sentinel test wraps ``_fft_conv_step`` with a counter so the
  parity tests cannot silently degenerate into dense-vs-dense.
"""

from __future__ import annotations

import numpy as np
import pytest

import gzl.direct_sum as ds
from gzl.direct_sum import (
    _direct_sum_extrapolated_grid,
    direct_sum_zero_momentum,
)


A1 = np.eye(1, dtype=float)
A2 = np.eye(2, dtype=float)

K4 = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
K5 = np.array([(i, j) for i in range(5) for j in range(i + 1, 5)])
K5_ST = np.array([e for e in K5.tolist() if tuple(e) != (0, 4)])
PRISM = np.array([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
                  (0, 3), (1, 4), (2, 5)])
K33 = np.array([(i, j) for i in (0, 1, 2) for j in (3, 4, 5)])
PATH = np.array([(0, 1), (1, 2)])

GRAPHS = {
    "K4": K4,
    "K5": K5,
    "K5_ST": K5_ST,
    "prism": PRISM,
    "K33": K33,
}

RTOL = 1.0e-12


@pytest.fixture
def force_conv(monkeypatch):
    """Open the performance gate so small test boxes still peel.

    The fast path is gated on a cost model (``_FFT_MIN_BAG`` /
    ``_FFT_MARGIN``) that keeps the dense branch where the two are
    comparable — which at the small boxes used here is everywhere.
    Correctness must hold whenever the peel *is* taken, so the parity
    tests disable the gate; :class:`TestPerformanceGate` covers the
    gate itself.
    """
    monkeypatch.setattr(ds, "_FFT_MIN_BAG", 0)
    monkeypatch.setattr(ds, "_FFT_MARGIN", 0.0)


def _ab(monkeypatch, fn, *args, expect_peel=True, **kwargs):
    """Return (conv, dense) results of ``fn`` toggling ``_USE_CONV``.

    ANTI-VACUITY GUARD — see the torus twin in
    ``tests/test_tensor_network_conv.py``.  :func:`_assert_parity` only
    bounds ``|on - off|``, so a switch that stopped reaching the
    executor would make every parametrisation here compare dense against
    dense and pass.  Counting the FFT step makes the switch itself the
    object under test.

    Note this file's peel is additionally cost-gated, so callers must
    also request the ``force_conv`` fixture; without it the gate keeps
    the dense branch at these small boxes and ``n_on`` is legitimately
    zero.
    """
    calls = {"n": 0}
    orig = ds._fft_conv_step

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(ds, "_fft_conv_step", counting)

    monkeypatch.setattr(ds, "_USE_CONV", True)
    on = np.asarray(fn(*args, **kwargs))
    n_on = calls["n"]

    calls["n"] = 0
    monkeypatch.setattr(ds, "_USE_CONV", False)
    off = np.asarray(fn(*args, **kwargs))
    n_off = calls["n"]

    if expect_peel:
        assert n_on >= 1, (
            "the FFT step never fired with _USE_CONV=True: this "
            "parametrisation compares dense against dense and asserts "
            "nothing (is the force_conv fixture requested?)"
        )
    else:
        # The caller is testing a REFUSAL.  Zero peels is the property
        # under test, not an accident, so it is asserted rather than
        # tolerated -- otherwise the refusal could silently stop
        # happening and the value comparison (dense vs dense) would
        # still pass.
        assert n_on == 0, (
            f"expected the peel to be refused here, but it fired {n_on}x"
        )
    assert n_off == 0, (
        f"the FFT step fired {n_off}x with _USE_CONV=False: the kill "
        f"switch no longer reaches the executor"
    )
    return on, off


def _assert_parity(on, off):
    scale = float(np.max(np.abs(off)))
    assert scale > 0.0
    assert np.max(np.abs(on - off)) <= RTOL * scale


class TestVacuumParity:
    @pytest.mark.parametrize("name", sorted(GRAPHS))
    @pytest.mark.parametrize("d,L", [(1, 5), (1, 8), (2, 2), (2, 3)])
    @pytest.mark.parametrize("nu_off", [0.25, 1.25])
    @pytest.mark.parametrize("use_symmetry", [True, False])
    def test_zero_momentum(self, monkeypatch, force_conv, name, d, L,
                           nu_off, use_symmetry):
        edges = GRAPHS[name]
        A = A1 if d == 1 else A2
        nu = np.full(len(edges), d + nu_off)
        on, off = _ab(
            monkeypatch, direct_sum_zero_momentum, edges, nu, A, L,
            use_symmetry=use_symmetry,
        )
        _assert_parity(on, off)

    def test_large_nu(self, monkeypatch, force_conv):
        nu = np.full(len(K5), 4.0)
        on, off = _ab(
            monkeypatch, direct_sum_zero_momentum, K5, nu, A1, 8,
        )
        _assert_parity(on, off)

    def test_mixed_nu_multigraph(self, monkeypatch, force_conv):
        # Parallel edges in both orientations with different exponents:
        # exercises the multi-edge collapse feeding the conv tag.
        edges = np.array([(0, 1), (1, 0), (1, 2), (2, 1), (0, 2)])
        nu = np.array([2.5, 4.0, 2.5, 4.0, 2.5])
        on, off = _ab(
            monkeypatch, direct_sum_zero_momentum, edges, nu, A1, 8,
        )
        _assert_parity(on, off)


class TestFiniteMomentumParity:
    @pytest.mark.parametrize("name,terminal",
                             [("K4", 3), ("K5_ST", 4), ("prism", 5)])
    @pytest.mark.parametrize("d,L", [(1, 5), (2, 3)])
    @pytest.mark.parametrize("kf", [0.17, 0.5])
    def test_single_k(self, monkeypatch, force_conv, name, terminal, d, L, kf):
        edges = GRAPHS[name]
        A = A1 if d == 1 else A2
        nu = np.full(len(edges), 2.5)
        on, off = _ab(
            monkeypatch, direct_sum_zero_momentum, edges, nu, A, L,
            root=0, terminal=terminal, momentum=np.full(d, kf),
        )
        _assert_parity(on, off)


class TestGridParity:
    @pytest.mark.parametrize("name,terminal", [("K4", 3), ("K5", 4)])
    @pytest.mark.parametrize(
        "d,L_list", [(1, (3, 4, 5, 6)), (2, (2, 3, 4))],
    )
    def test_grid(self, monkeypatch, force_conv, name, terminal, d, L_list):
        edges = GRAPHS[name]
        A = A1 if d == 1 else A2
        nu = np.full(len(edges), 2.5)
        k_grid = (np.linspace(0.0, 0.5, 5)[:, None]
                  * np.ones(d)[None, :])
        on, off = _ab(
            monkeypatch, _direct_sum_extrapolated_grid, edges, nu, A,
            k_grid, root=0, terminal=terminal, L_list=L_list,
            n_correction_terms=2,
        )
        _assert_parity(on, off)


class TestMarkerCollision:
    """Historic name: the peel used to REFUSE marker-touching steps.

    The offset-aware peel now handles them (a half-box axis is a
    window with a different origin and extent, and the transform
    geometry is derived from the endpoint windows), so the two
    refusal tests and the ``_plan_marker`` pin that lived here are
    deleted — their behaviour no longer exists.  The marker-peel
    correctness gates live in ``tests/test_marker_peel.py``; what
    remains here is the structural path-layout parity test, which is
    marker-independent.
    """

    def test_path_layouts_match_dense(self, monkeypatch, force_conv):
        """Value parity on both root layouts of the path 0-1-2.

        ``root=0``: vertex 2 is eliminated against the free-free kernel
        ``{1, 2}``, so the peel is available and parity is a real test.

        ``root=1``: pinning the middle vertex makes BOTH kernels
        pin-incident, hence one-axis, so no two-vertex kernel exists and
        no peel is structurally possible.  That arm is dense-vs-dense
        and is marked as such rather than left to look like coverage it
        does not provide.
        """
        nu = np.full(2, 2.5)
        for root, expect_peel in ((0, True), (1, False)):
            on, off = _ab(
                monkeypatch, direct_sum_zero_momentum, PATH, nu, A1, 8,
                root=root, expect_peel=expect_peel,
            )
            _assert_parity(on, off)


class TestMarkerRule:
    """The Z2 marker is the last eliminated vertex, unconditionally, in
    BOTH kill-switch states — the ``_plan_marker`` cost pre-pass that
    traded the half axis against an FFT-blocking marker was deleted
    along with the marker's peel exclusions.  The surviving switch is
    ``_USE_CONV`` itself: it decides whether steps may peel, never
    whether the marker exists.
    """

    def test_marker_is_switch_independent_and_values_agree(
            self, monkeypatch, force_conv):
        """Both switch states run with the same marker; the on-arm
        actually peels a marker-touching step (sentinel), and the two
        agree to round-off.  PATH root=0 makes the first step peel
        toward the marker (u = marker), which the earlier marker rule
        could never do."""
        markers = []
        orig_init = ds.BoxTruncation.__init__

        def recording(self, L, d, A, **kw):
            orig_init(self, L, d, A, **kw)
            markers.append(self.marker)

        monkeypatch.setattr(ds.BoxTruncation, "__init__", recording)

        marker_peels = {"n": 0}
        orig_peel = ds.BoxTruncation.peel_step

        def counting(self, bucket, token, w, out_axes, dtype):
            u, _ = token
            if self._half(w) or self._half(u):
                marker_peels["n"] += 1
            return orig_peel(self, bucket, token, w, out_axes, dtype)

        monkeypatch.setattr(ds.BoxTruncation, "peel_step", counting)
        nu = np.full(len(PATH), 2.5)

        monkeypatch.setattr(ds, "_USE_CONV", True)
        v_on = complex(direct_sum_zero_momentum(PATH, nu, A1, 5, root=0))
        assert marker_peels["n"] >= 1, (
            "no marker-touching step peeled — the offset-aware marker "
            "peel is dead"
        )
        monkeypatch.setattr(ds, "_USE_CONV", False)
        v_off = complex(direct_sum_zero_momentum(PATH, nu, A1, 5, root=0))

        assert len(markers) == 2 and markers[0] == markers[1] is not None, (
            f"marker differs between switch states: {markers}"
        )
        assert abs(v_on - v_off) <= RTOL * abs(v_off)


class TestPerformanceGate:
    """The gate is a pure performance heuristic: it decides *whether*
    a step takes the FFT path, never what the step computes."""

    def test_small_box_stays_dense(self, monkeypatch):
        # A d=1 box this small can never pay for the padded transform.
        calls = {"n": 0}
        orig = ds._fft_conv_step

        def counting(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(ds, "_fft_conv_step", counting)
        monkeypatch.setattr(ds, "_USE_CONV", True)
        nu = np.full(len(K5), 2.5)
        direct_sum_zero_momentum(K5, nu, A1, 6)
        assert calls["n"] == 0
        assert not ds._conv_possible(13, 13, 1)

    def test_gate_open_at_production_box(self):
        # d=2 at the production ladder's upper end does pay: the box
        # admits the FFT path, and a three-axis bag clears the per-step
        # floor (BoxTruncation.peel_gate checks both) …
        assert ds._conv_possible(13 ** 2, 13, 2)
        assert 13 ** 6 >= ds._FFT_MIN_BAG
        # …but not a bag small enough to be sub-millisecond anyway.
        assert 13 ** 2 < ds._FFT_MIN_BAG

    def test_gate_does_not_change_values(self, monkeypatch):
        # Same call with the gate closed, open, and the fast path
        # disabled entirely — three routes, one value.
        nu = np.full(len(K4), 2.5)
        monkeypatch.setattr(ds, "_USE_CONV", True)
        gated = direct_sum_zero_momentum(K4, nu, A2, 4)
        monkeypatch.setattr(ds, "_FFT_MIN_BAG", 0)
        monkeypatch.setattr(ds, "_FFT_MARGIN", 0.0)
        forced = direct_sum_zero_momentum(K4, nu, A2, 4)
        monkeypatch.setattr(ds, "_USE_CONV", False)
        dense = direct_sum_zero_momentum(K4, nu, A2, 4)
        _assert_parity(np.asarray(forced), np.asarray(dense))
        _assert_parity(np.asarray(gated), np.asarray(dense))


class TestPeelFires:
    def test_sentinel(self, monkeypatch, force_conv):
        calls = {"n": 0}
        orig = ds._fft_conv_step

        def counting(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(ds, "_fft_conv_step", counting)
        nu = np.full(len(K5), 2.5)

        monkeypatch.setattr(ds, "_USE_CONV", True)
        direct_sum_zero_momentum(K5, nu, A1, 6)
        assert calls["n"] >= 1

        calls["n"] = 0
        monkeypatch.setattr(ds, "_USE_CONV", False)
        direct_sum_zero_momentum(K5, nu, A1, 6)
        assert calls["n"] == 0


class TestEmptyBucketScalar:
    """A vertex with no incident factor contributes its full axis count.

    On the Z2 marker axis the elimination ranges over a HALF box, and
    ``z2_weights`` is what restores the other half.  A bucket-less
    vertex therefore contributes ``z2_weights.sum()`` — which is
    ``n_full`` — and not the half-axis size.

    Getting this wrong scales the entire block by ``(L+1)/(2L+1)`` with
    no exception, no NaN and no shape mismatch, and every downstream
    Richardson fit converges smoothly to the wrong number.

    Whether it is reached is a property of the ORDER rather than the
    engine.  The fallback ``_min_degree_order`` eliminates degree-0 free
    vertices first while ``marker = elimination[-1]``, so the marker
    never has an empty bucket there.  The production order comes from
    ``_elimination.plan``, a cost-ordered schedule that need not keep
    that property, so the branch must be correct on its own;
    ``test_branch_is_unreachable_on_the_shipped_schedule`` checks that
    it stays unreached on the canonical cores.
    """

    @staticmethod
    def _z2_weights(m, d):
        axes = [np.arange((m + 1) // 2)] + [np.arange(m)] * (d - 1)
        pos = np.array(np.meshgrid(*axes, indexing="ij")).reshape(d, -1).T
        return np.where(pos[:, 0] == 0, 1.0, 2.0)

    @pytest.mark.parametrize("d,L", [(1, 4), (2, 3), (3, 2)])
    def test_marker_axis_contributes_the_full_count(self, d, L):
        m = 2 * L + 1
        n_full = m ** d
        z2 = self._z2_weights(m, d)
        out = ds._eliminate_all(
            [{"scope": (1,), "tensor": np.ones(n_full)}],
            [1, 2], marker=2,
            n_full=n_full, n_half=len(z2), z2_weights=z2,
            A=np.eye(d), L=L, d=d,
        )
        got = float(np.prod(
            [float(np.asarray(f["tensor"])) for f in out]
        ))
        assert got == pytest.approx(float(n_full) ** 2, rel=1e-12), (
            f"d={d} L={L}: bucket-less marker vertex contributed "
            f"{got / n_full} instead of {n_full}"
        )

    def test_branch_is_unreachable_on_the_shipped_schedule(self):
        """The branch never fires on the shipped schedule and moves no value.

        Instrumented over the canonical high-treewidth cores at the
        shipped root and order.  If this ever starts firing, the branch
        is live on the shipped schedule and values computed there depend
        on it.
        """
        seen = {"n": 0}
        orig = ds._eliminate_all

        def spy(factors, elimination, marker, *a, **kw):
            for w in elimination:
                if not [f for f in factors if w in f["scope"]]:
                    seen["n"] += 1
            return orig(factors, elimination, marker, *a, **kw)

        ds._eliminate_all = spy
        try:
            for E in (K4, K5, PRISM, K33):
                nu = np.full(len(E), 2.5)
                ds.direct_sum_zero_momentum(E, nu, A1, 4)
        finally:
            ds._eliminate_all = orig
        assert seen["n"] == 0, (
            f"the empty-bucket branch fired {seen['n']}x on the shipped "
            f"schedule; values on that schedule now depend on it"
        )
