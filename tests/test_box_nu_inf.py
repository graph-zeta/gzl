# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""The box ν = ∞ minimal-shell kernel and the d-guard.

The latent defect: ``_conv_kernel_diff`` evaluated ``dist ** -inf``
literally, which is 1 / nan / 0 for NN distances = 1 / < 1 / > 1 — so
the 2-path NN count came out 4 / nan / 0 for ``A ∈ {I, I/2, 2I}``
where 4 is correct for all three.  Unreachable only because the
frontend's nn_mode routes every ∞ block to the tensor; fixed before
any routing exposes it.  Acceptance: exact integrality
(``np.array_equal(val, round(val))``) plus peel-count 0, over all
three lattices.

The ``d ∈ {1, 2, 3}`` guard is lifted alongside (the machinery is
dimension-generic); d = 4 is gated against a brute-force reference.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import gzl.direct_sum as ds
import gzl.tensor_network as tn

PATH2 = np.array([(0, 1), (1, 2)])
K4 = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


class TestBoxInfKernel:
    @pytest.mark.parametrize("scale", [1.0, 0.5, 2.0])
    def test_two_path_nn_count_is_exactly_four(self, scale):
        """The acceptance fixture: 4 for every lattice scale,
        integer-exact."""
        nu = np.array([np.inf, np.inf])
        val = complex(ds.direct_sum_zero_momentum(
            PATH2, nu, scale * np.eye(1), 3))
        assert val.imag == 0.0
        v = np.float64(val.real)
        assert np.array_equal(v, np.round(v))
        assert v == 4.0

    @pytest.mark.parametrize("scale", [1.0, 0.5, 2.0])
    def test_inf_steps_never_peel(self, scale, monkeypatch):
        """Peel-count == 0 with the cost gate forced OPEN — the
        structural opt-out (no conv_nu tag), not the gate, is what
        keeps the NN contraction integer-exact."""
        calls = {"n": 0}
        orig = ds._fft_conv_step

        def counting(*a, **kw):
            calls["n"] += 1
            return orig(*a, **kw)

        monkeypatch.setattr(ds, "_fft_conv_step", counting)
        monkeypatch.setattr(ds, "_conv_possible",
                            lambda *a, **kw: True)
        monkeypatch.setattr(ds, "_FFT_MIN_BAG", 0)
        nu = np.array([np.inf, np.inf])
        # root=0 keeps the (1, 2) kernel free-free, so a peelable
        # STRUCTURE exists and the missing conv_nu tag is what stops
        # the peel (the default max-degree root would make every
        # kernel root-incident and the assertion vacuous).
        val = complex(ds.direct_sum_zero_momentum(
            PATH2, nu, scale * np.eye(1), 3, root=0))
        assert calls["n"] == 0
        assert val.real == 4.0

    def test_liveness_finite_kernel_does_peel_under_forced_gate(
            self, monkeypatch):
        """Control for the sentinel above: the same graph with finite
        ν and the same forced gate must peel at least once — otherwise
        the zero-count assertion is vacuous."""
        calls = {"n": 0}
        orig = ds._fft_conv_step

        def counting(*a, **kw):
            calls["n"] += 1
            return orig(*a, **kw)

        monkeypatch.setattr(ds, "_fft_conv_step", counting)
        monkeypatch.setattr(ds, "_conv_possible",
                            lambda *a, **kw: True)
        monkeypatch.setattr(ds, "_FFT_MIN_BAG", 0)
        nu = np.array([2.5, 2.5])
        ds.direct_sum_zero_momentum(PATH2, nu, np.eye(1), 3, root=0)
        assert calls["n"] >= 1

    @pytest.mark.parametrize("d,L,n_torus", [(1, 3, 8), (2, 2, 6)])
    def test_nn_counts_match_the_torus_engine_exactly(self, d, L,
                                                      n_torus):
        """NN homomorphism counts are local, so box and torus must
        agree INTEGER-EXACTLY on K4 once the truncation exceeds the
        shell reach."""
        nu = np.full(len(K4), np.inf)
        box = complex(ds.direct_sum_zero_momentum(
            K4, nu, np.eye(d), L)).real
        torus = complex(tn.graph_zeta_general_at_zero(
            K4, nu, np.eye(d), n_torus, pinned_vertex=0)).real
        assert np.array_equal(np.float64(box), np.round(box))
        assert box == torus

    def test_open_terminal_inf_grid_is_integer_exact(self):
        """The open-terminal engine shares the factor build; its
        M(x_t) for an all-inf 2-path is the integer NN pair count per
        terminal position."""
        nu = np.array([np.inf, np.inf])
        em = ds._collapse_multi_edges(PATH2, nu)
        M, pos = ds._direct_sum_open_terminal(em, np.eye(1), 3, 0, 2, 1)
        assert np.array_equal(M, np.round(M))
        assert M.sum() == 4.0                 # total count over x_t


class TestDGuardLift:
    def test_d4_bridge_matches_brute_force(self):
        nu = np.array([2.5])
        val = complex(ds.direct_sum_zero_momentum(
            np.array([(0, 1)]), nu, np.eye(4), 2)).real
        coords = np.array(list(itertools.product(range(-2, 3),
                                                 repeat=4)), float)
        r = np.linalg.norm(coords, axis=1)
        ref = np.sum(r[r > 0.0] ** -2.5)
        np.testing.assert_allclose(val, ref, rtol=1e-13)

    def test_d4_two_path_matches_brute_force(self):
        nu = np.array([2.5, 3.5])
        # root=0 to match the reference's pin: on a box the pin is
        # value-RELEVANT at finite L (the default max-degree root
        # would pin the middle vertex and sum a different truncation).
        val = complex(ds.direct_sum_zero_momentum(
            PATH2, nu, np.eye(4), 1, root=0)).real
        coords = np.array(list(itertools.product(range(-1, 2),
                                                 repeat=4)), float)
        ref = 0.0
        for x1 in coords:
            r1 = np.linalg.norm(x1)
            if r1 == 0.0:
                continue
            diff = coords - x1
            r2 = np.linalg.norm(diff, axis=1)
            ref += r1 ** -2.5 * np.sum(r2[r2 > 0.0] ** -3.5)
        np.testing.assert_allclose(val, ref, rtol=1e-13)

    def test_d4_inf_path_counts_coordination_squared(self):
        nu = np.array([np.inf, np.inf])
        val = complex(ds.direct_sum_zero_momentum(
            PATH2, nu, np.eye(4), 1)).real
        assert val == float((2 * 4) ** 2)     # (2d)^2 = 64

    def test_d0_still_refused(self):
        with pytest.raises(ValueError):
            ds.direct_sum_zero_momentum(
                PATH2, np.array([2.5, 2.5]), np.zeros((0, 0)), 2)
