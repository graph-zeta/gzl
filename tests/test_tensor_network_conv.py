# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Parity of the FFT elimination fast path in
``gzl.tensor_network``.

``_eliminate_vertex`` contracts a bucket by cyclic FFT convolution
whenever one original circulant edge tensor can be peeled
(``_find_conv_peel`` / ``_peel_convolution``).  These tests A/B the
fast path against the dense slice-einsum via the private ``_USE_CONV``
switch: both are exact contractions differing only in summation order,
so agreement is required at ``1e-12`` relative (measured ~1e-13 at the
worst ``ν = d + 0.25``).

The K5-minus-st core is the canonical structural-fallback case: its
second elimination step has bucket ``{K(b), K(b−c), K(t−b), ψ₁(b,c,t)}``
where no kernel can be peeled (ψ₁ carries every would-be partner), so
that step must run dense while the first (peak) step accelerates.
"""

from __future__ import annotations

import numpy as np
import pytest

import gzl.tensor_network as tn
from gzl.tensor_network import (
    graph_zeta_general,
    graph_zeta_general_at_zero,
)


A1 = np.eye(1, dtype=float)
# Hexagonal d=2 cell, matching tests/test_hybrid.py.
A2 = np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2.0]])

K4 = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
K5 = np.array([(i, j) for i in range(5) for j in range(i + 1, 5)])
K5_ST = np.array([e for e in K5.tolist() if tuple(e) != (0, 4)])
PRISM = np.array([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
                  (0, 3), (1, 4), (2, 5)])
K33 = np.array([(i, j) for i in (0, 1, 2) for j in (3, 4, 5)])

GRAPHS = {
    "K4": (K4, 3),
    "K5": (K5, 4),
    "K5_ST": (K5_ST, 4),
    "prism": (PRISM, 5),
    "K33": (K33, 5),
}

RTOL = 1.0e-12


def _ab(monkeypatch, fn, *args, **kwargs):
    """Run ``fn`` with the FFT peel on and off, asserting it really moved.

    ANTI-VACUITY GUARD.  :func:`_assert_parity` only bounds
    ``|on - off|``.  If ``_USE_CONV`` ever stopped reaching the
    executor — which is exactly what a refactor that relocates the
    elimination loop into a shared module can do — both arms would
    compute the identical thing and every parametrisation in this file
    would pass while the FFT path went completely untested.

    Counting the peel makes the switch itself the object under test, and
    does so for every parametrisation rather than for the single
    fixture :class:`TestPeelFires` covers.  ``off == 0`` is the
    no-op detector; ``on >= 1`` is the vacuity detector.
    """
    calls = {"n": 0}
    orig = tn._peel_convolution

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(tn, "_peel_convolution", counting)

    monkeypatch.setattr(tn, "_USE_CONV", True)
    on = np.asarray(fn(*args, **kwargs))
    n_on = calls["n"]

    calls["n"] = 0
    monkeypatch.setattr(tn, "_USE_CONV", False)
    off = np.asarray(fn(*args, **kwargs))
    n_off = calls["n"]

    assert n_on >= 1, (
        "the FFT peel never fired with _USE_CONV=True: this "
        "parametrisation compares dense against dense and asserts nothing"
    )
    assert n_off == 0, (
        f"the FFT peel fired {n_off}x with _USE_CONV=False: the kill "
        f"switch no longer reaches the executor"
    )
    return on, off


def _assert_parity(on, off):
    scale = float(np.max(np.abs(off)))
    assert scale > 0.0
    assert np.max(np.abs(on - off)) <= RTOL * scale


class TestModeParity:
    @pytest.mark.parametrize("name", sorted(GRAPHS))
    @pytest.mark.parametrize("d,n", [(1, 16), (1, 32), (2, 8)])
    @pytest.mark.parametrize("nu_off", [0.25, 1.25])
    def test_vacuum(self, monkeypatch, name, d, n, nu_off):
        edges, _ = GRAPHS[name]
        A = A1 if d == 1 else A2
        nu = np.full(len(edges), d + nu_off)
        on, off = _ab(
            monkeypatch, graph_zeta_general_at_zero, edges, nu, A, n,
        )
        _assert_parity(on, off)

    @pytest.mark.parametrize("name", sorted(GRAPHS))
    @pytest.mark.parametrize("d,n", [(1, 16), (2, 8)])
    @pytest.mark.parametrize("space", ["z", "k"])
    def test_free_terminal(self, monkeypatch, name, d, n, space):
        edges, t = GRAPHS[name]
        A = A1 if d == 1 else A2
        nu = np.full(len(edges), 2.5)
        on, off = _ab(
            monkeypatch, graph_zeta_general, edges, nu, A, n,
            source=0, terminals=(t,), space=space,
        )
        _assert_parity(on, off)

    @pytest.mark.parametrize("name", sorted(GRAPHS))
    @pytest.mark.parametrize("d,n", [(1, 16), (2, 8)])
    def test_single_momentum(self, monkeypatch, name, d, n):
        edges, t = GRAPHS[name]
        A = A1 if d == 1 else A2
        nu = np.full(len(edges), 2.5)
        on, off = _ab(
            monkeypatch, graph_zeta_general, edges, nu, A, n,
            source=0, terminals=(t,), momentum=np.full(d, 0.3),
        )
        _assert_parity(on, off)

    @pytest.mark.parametrize("name", sorted(GRAPHS))
    def test_vacuum_d2_n12(self, monkeypatch, name):
        edges, _ = GRAPHS[name]
        nu = np.full(len(edges), 2.5)
        on, off = _ab(
            monkeypatch, graph_zeta_general_at_zero, edges, nu, A2, 12,
        )
        _assert_parity(on, off)


class TestStreamingParity:
    @pytest.mark.parametrize("name", ["prism", "K5_ST"])
    def test_forced_streaming(self, monkeypatch, name):
        edges, t = GRAPHS[name]
        nu = np.full(len(edges), 2.5)
        kwargs = dict(source=0, terminals=(t,), space="z")

        batched = np.asarray(
            graph_zeta_general(edges, nu, A1, 16, **kwargs)
        )
        monkeypatch.setattr(tn, "_FORCE_STREAMING", True)
        monkeypatch.setattr(tn, "_MEM_BUDGET_BYTES", 1)
        on, off = _ab(
            monkeypatch, graph_zeta_general, edges, nu, A1, 16, **kwargs,
        )
        _assert_parity(on, off)
        # The streamed suffix must also agree with the batched result.
        _assert_parity(on, batched)


class TestOrientationAndMerge:
    def test_reversed_orientation_exact(self):
        fwd = np.array([(0, 1), (1, 2), (0, 2)])
        rev = np.array([(1, 0), (2, 1), (2, 0)])
        nu = np.full(3, 2.5)
        a = graph_zeta_general_at_zero(fwd, nu, A1, 16)
        b = graph_zeta_general_at_zero(rev, nu, A1, 16)
        assert a == b

    def test_mixed_nu_parallel_edges(self, monkeypatch):
        # Parallel edges in both orientations with different exponents:
        # exercises the merged-generator Hadamard path.
        edges = np.array([(0, 1), (1, 0), (1, 2), (2, 1), (0, 2)])
        nu = np.array([2.5, 4.0, 2.5, 4.0, 2.5])
        on, off = _ab(
            monkeypatch, graph_zeta_general_at_zero, edges, nu, A1, 16,
        )
        _assert_parity(on, off)

    @pytest.mark.parametrize("edges,terminal", [
        (np.array([(0, 1), (1, 2)]), 2),
        (np.array([(0, 1), (1, 2), (2, 3), (3, 0)]), 2),
        (np.array([(0, 1), (1, 2), (0, 2)]), 2),
    ])
    def test_nn_kernel_stays_exact(self, monkeypatch, edges, terminal):
        # ν = inf values are homomorphism counts — integers.  The
        # dense contraction produces them exactly; an FFT round trip
        # would not (2.0 comes back as 1.9999999999999998), so the
        # fast path must decline these edges outright.
        nu = np.full(len(edges), np.inf)
        for kwargs in ({}, dict(source=0, terminals=(terminal,),
                                space="z")):
            fn = (graph_zeta_general_at_zero if not kwargs
                  else graph_zeta_general)
            monkeypatch.setattr(tn, "_USE_CONV", True)
            val = np.asarray(fn(edges, nu, A1, 8, **kwargs)).real
            assert np.array_equal(val, np.round(val))

    def test_nn_kernel_never_peels(self, monkeypatch):
        calls = {"n": 0}
        orig = tn._peel_convolution

        def counting(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(tn, "_peel_convolution", counting)
        monkeypatch.setattr(tn, "_USE_CONV", True)
        c4 = np.array([(0, 1), (1, 2), (2, 3), (3, 0)])
        graph_zeta_general(c4, np.full(len(c4), np.inf), A1, 8,
                           source=0, terminals=(2,), space="z")
        assert calls["n"] == 0

    def test_mixed_finite_and_nn_exponents(self, monkeypatch):
        # A bundle mixing a finite and an infinite exponent must also
        # withhold the generator, not multiply one in.
        edges = np.array([(0, 1), (1, 0), (1, 2), (0, 2)])
        nu = np.array([2.5, np.inf, 2.5, 2.5])
        on, off = _ab(
            monkeypatch, graph_zeta_general_at_zero, edges, nu, A1, 16,
        )
        _assert_parity(on, off)


class TestPeelFires:
    def test_sentinel(self, monkeypatch):
        calls = {"n": 0}
        orig = tn._peel_convolution

        def counting(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(tn, "_peel_convolution", counting)
        nu = np.full(len(K5), 2.5)

        monkeypatch.setattr(tn, "_USE_CONV", True)
        graph_zeta_general_at_zero(K5, nu, A1, 16)
        assert calls["n"] >= 1

        calls["n"] = 0
        monkeypatch.setattr(tn, "_USE_CONV", False)
        graph_zeta_general_at_zero(K5, nu, A1, 16)
        assert calls["n"] == 0
