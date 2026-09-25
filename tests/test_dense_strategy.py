# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""The box dense step's strategy dispatch.

The machinery is ``_contract.resolve_dense_strategy``, a
``dense_strategy`` method per truncation, and the box's axis-sliced
dense branch.  The torus stays pinned to its sliced einsum loop.

``BoxTruncation.dense_strategy`` is ``"auto"``: sliced at or above
``direct_sum._DENSE_SLICE_MIN_VOLUME`` bag elements.  The threshold is
measured, not modelled (M1 Max, bag 3-4 bucket shapes): sliced is 3-7x
slower below ~1e5 bag elements, breaks even around ~5e5, wins 1.5-3x
above ~5e6 — and everywhere replaces the broadcast peak of TWO full-bag
arrays by the out-scope volume, a factor ``2*axis_size(w)`` smaller.
With the peel chunked, that broadcast peak was the box's memory ceiling
before the sliced branch.  1e5 keeps every shipped d = 1 rung on
broadcast (largest d = 1 bag is K5's marker-halved 17^3 * 9 = 4.4e4 at
L = 8, where sliced is a measured 3.5-11.5x wall-clock loss) and flips
exactly the class that matters at d >= 2: the big UNPEELABLE dense bags
(K5/prism bag-4 buckets, 3.3e6 elements at d = 2 L = 3; 1.5e8 at d = 3
L = 2) that set that ceiling.
Measured per call: K4's d >= 2 buckets are peel-covered at L >= 4 and
marker-halved to 6.7e4 < 1e5 at L = 3, so K4 correctly stays on
broadcast throughout — the flip follows bag volume, not dimension.

The sliced branch re-associates the w-axis sum (a loop of in-place
adds versus ``np.sum``'s pairwise reduction), so values move at the
1e-15 scale where it engages; gated at 1e-13 against the broadcast
branch here, 1e-12 plus the Richardson fit residual end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

import gzl._contract as _contract
import gzl.direct_sum as ds


K4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
K5 = [(a, b) for a in range(5) for b in range(a + 1, 5)]
PRISM = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
         (0, 3), (1, 4), (2, 5)]


class TestResolver:
    def test_pinned_modes_pass_through(self):
        assert _contract.resolve_dense_strategy(
            "broadcast", 10 ** 12, 1) == "broadcast"
        assert _contract.resolve_dense_strategy(
            "sliced", 1, 10 ** 12) == "sliced"

    def test_auto_thresholds_on_bag_volume(self):
        thr = 1000
        assert _contract.resolve_dense_strategy(
            "auto", 999, thr) == "broadcast"
        assert _contract.resolve_dense_strategy(
            "auto", 1000, thr) == "sliced"
        assert _contract.resolve_dense_strategy(
            "auto", 1001, thr) == "sliced"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown dense strategy"):
            _contract.resolve_dense_strategy("fused", 1, 1)


def _force(monkeypatch, strategy: str):
    monkeypatch.setattr(ds.BoxTruncation, "dense_strategy",
                        lambda self, bag_volume: strategy)


def _sliced_counter(monkeypatch):
    """Count sliced-branch executions; record whether w was the marker."""
    calls = {"n": 0, "marker": 0}
    orig = ds.BoxTruncation._dense_step_sliced

    def counting(self, bucket, w, union_scope):
        calls["n"] += 1
        if self.marker is not None and w == self.marker:
            calls["marker"] += 1
        return orig(self, bucket, w, union_scope)

    monkeypatch.setattr(ds.BoxTruncation, "_dense_step_sliced", counting)
    return calls


CASES = [
    ("k4_d1_L4", K4, 2.5, 1, 4, {}),
    ("k4_d2_L3", K4, 3.0, 2, 3, {}),
    ("k5_d1_L4", K5, 2.5, 1, 4, {}),
    ("prism_d2_L2", PRISM, 3.0, 2, 2, {}),
    ("k4_d1_L5_1qp", K4, 2.5, 1, 5,
     {"momentum": np.array([0.25]), "terminal": 3, "root": 0}),
]


class TestSlicedEqualsBroadcast:
    """The two dense branches agree wherever either can run.

    Forced through the strategy method (not the tunable), so this holds
    whether the box's strategy is pinned or ``"auto"``.  The sliced
    counter proves the forced branch actually ran — an equality between
    two runs of the same branch would certify nothing.
    """

    @pytest.mark.parametrize("name,edges,nu,d,L,kw",
                             CASES, ids=[c[0] for c in CASES])
    def test_engine_value(self, name, edges, nu, d, L, kw, monkeypatch):
        A = np.eye(d)
        nu_vec = np.full(len(edges), nu)
        _force(monkeypatch, "broadcast")
        vb = complex(ds.direct_sum_zero_momentum(edges, nu_vec, A, L, **kw))
        calls = _sliced_counter(monkeypatch)
        _force(monkeypatch, "sliced")
        vs = complex(ds.direct_sum_zero_momentum(edges, nu_vec, A, L, **kw))
        assert calls["n"] >= 1, "forced sliced branch never ran"
        assert abs(vs - vb) <= 1e-13 * abs(vb)

    def test_marker_elimination_through_the_sliced_branch(self, monkeypatch):
        """The Z2 marker weights become per-slice scalars; a sliced
        branch that dropped them would undercount by ~2x silently.
        The counter asserts a marker elimination really took the
        sliced branch (use_symmetry=True installs the marker)."""
        nu_vec = np.full(len(K4), 2.5)
        _force(monkeypatch, "broadcast")
        vb = complex(ds.direct_sum_zero_momentum(K4, nu_vec, np.eye(1), 4,
                                                 use_symmetry=True))
        calls = _sliced_counter(monkeypatch)
        _force(monkeypatch, "sliced")
        vs = complex(ds.direct_sum_zero_momentum(K4, nu_vec, np.eye(1), 4,
                                                 use_symmetry=True))
        assert calls["marker"] >= 1, (
            "no marker vertex went through the sliced dense branch — "
            "the Z2-weight path is untested"
        )
        assert abs(vs - vb) <= 1e-13 * abs(vb)

    def test_complex_dtype_promotes_identically(self, monkeypatch):
        """A complex extra factor must promote in the sliced branch
        exactly as the broadcast product does naturally."""
        L, d = 4, 1
        m = 2 * L + 1
        rng = np.random.default_rng(7)
        phase = np.exp(1j * rng.uniform(0, 2 * np.pi, m))
        factors = [
            {"scope": (1, 2), "tensor": rng.random((m, m)) + 0.5},
            {"scope": (1,), "tensor": phase},
            {"scope": (2,), "tensor": rng.random(m) + 0.5},
        ]
        common = dict(marker=None, n_full=m, n_half=(L + 1),
                      z2_weights=None, A=np.eye(d), L=L, d=d)

        def run():
            out = ds._eliminate_all(
                [dict(f) for f in factors], [1, 2], **common)
            r = 1.0 + 0.0j
            for f in out:
                r *= complex(np.asarray(f["tensor"]))
            return r

        monkeypatch.setattr(ds, "_USE_CONV", False)   # all-dense
        _force(monkeypatch, "broadcast")
        vb = run()
        calls = _sliced_counter(monkeypatch)
        _force(monkeypatch, "sliced")
        vs = run()
        assert calls["n"] >= 1
        # Liveness: the complex factor must actually reach the value —
        # a real result would mean the promotion path was never tested.
        assert abs(vb.imag) > 0
        assert abs(vs - vb) <= 1e-13 * abs(vb)


class TestAutoEngagement:
    """The measured threshold's shipped-rung split, pinned structurally.

    Wall clock cannot be asserted portably, but WHICH branch runs can:
    if a later threshold change flips a shipped d = 1 rung to sliced
    (a measured 3.5-11.5x loss there), the first test fails; if d >= 2
    stops slicing (giving back the memory ceiling), the second does.
    """

    def test_d1_shipped_rungs_stay_broadcast(self, monkeypatch):
        calls = _sliced_counter(monkeypatch)
        for L in (4, 5, 6, 7, 8):
            ds.direct_sum_zero_momentum(K5, np.full(10, 2.5), np.eye(1), L)
        assert calls["n"] == 0, (
            "a shipped d=1 rung took the sliced branch (largest d=1 bag "
            "is K5's marker-halved 17^3 * 9 = 4.4e4 "
            "< _DENSE_SLICE_MIN_VOLUME)"
        )

    def test_d2_unpeelable_bags_slice(self, monkeypatch):
        """The class the flip exists for: dense bags the FFT peel cannot
        absorb (K5's bag-4 buckets — the broadcast memory ceiling) slice
        from the first d = 2 rung (3.3e6 elements at L = 3).  K4's
        buckets by contrast are peel-covered at L >= 4 and marker-halved
        below 1e5 at L = 3, so K4 correctly stays on broadcast — sliced
        is a measured 2-4x loss at that volume and the bag is 0.5 MB."""
        calls = _sliced_counter(monkeypatch)
        ds.direct_sum_zero_momentum(K5, np.full(10, 3.0), np.eye(2), 3)
        assert calls["n"] >= 1, (
            "K5's unpeelable bag-4 bucket (49^3 * 28 = 3.3e6) did not slice"
        )
        before = calls["n"]
        ds.direct_sum_zero_momentum(K4, np.full(6, 3.0), np.eye(2), 3)
        assert calls["n"] == before, (
            "K4 d=2 L=3 sliced despite its marker-halved 6.7e4 bag "
            "sitting below the measured threshold"
        )

    def test_threshold_is_read_at_call_time(self, monkeypatch):
        """Monkeypatching the tunable must reach the running step —
        the construction rule's late-binding, asserted for this knob."""
        nu_vec = np.full(len(K4), 2.5)
        _force(monkeypatch, "broadcast")
        vb = complex(ds.direct_sum_zero_momentum(K4, nu_vec, np.eye(1), 4))
        monkeypatch.undo()
        calls = _sliced_counter(monkeypatch)
        monkeypatch.setattr(ds, "_DENSE_SLICE_MIN_VOLUME", 1)
        va = complex(ds.direct_sum_zero_momentum(K4, nu_vec, np.eye(1), 4))
        assert calls["n"] >= 1, "threshold=1 did not engage the sliced branch"
        assert abs(va - vb) <= 1e-13 * abs(vb)


class TestTheLadderAgreesAcrossStrategies:
    """End to end: the raw rungs and the extrapolated value agree
    between the two dense strategies.

    There is no fit-residual check: four rungs against a constant and
    three powers is an exact solve for any exponent, and with a fifth
    rung (L = 7) the residual does not single out the right basis
    either (measured 2.0e-9 of the value at the true exponent -10,
    2.0e-9 at -9 and 6.9e-10 at -8).
    """

    def test_k5_d2_ladder(self, monkeypatch):
        edges = K5
        nu_vec = np.full(len(edges), 3.0)
        A = np.eye(2)
        Ls = (3, 4, 5, 6)

        def raws():
            return np.array([
                float(np.real(complex(ds.direct_sum_zero_momentum(
                    edges, nu_vec, A, L)))) for L in Ls
            ])

        def extrapolated():
            return complex(ds.direct_sum_extrapolated(
                edges, nu_vec, A, L_list=Ls))

        _force(monkeypatch, "broadcast")
        S_b = raws()
        v_b = extrapolated()
        monkeypatch.undo()
        calls = _sliced_counter(monkeypatch)
        S_a = raws()
        v_a = extrapolated()
        assert calls["n"] >= 1, "auto did not slice on the d=2 ladder"

        assert np.max(np.abs(S_a - S_b)) <= 1e-13 * np.max(np.abs(S_b))
        assert abs(v_a - v_b) <= 1e-12 * abs(v_b)
