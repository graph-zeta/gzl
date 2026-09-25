# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The finite-k box tail sees the momentum only modulo 1 AND up to sign.

The box faces sit at integer lattice coordinates, so for integer ``L``

    cos(2*pi*(1-f)*L) = cos(2*pi*f*L),

which makes ``f`` and ``1-f`` the same mode.  ``zeta_G(k)`` is therefore
exactly even and 1-periodic in fractional momentum, and the raw box
satisfies that to round-off.  Any extrapolator laid on top of the box
must preserve it: the unfolded frequency let a slow mode near ``f = 1``
pass the adequacy gate as if it were fast, so the grid disagreed between
``k`` and ``1-k`` by up to 5e-4 on real treewidth-3 corpus blocks.

These assertions need no reference value -- they are exact symmetries of
the object -- so they carry no reference uncertainty.
"""
import numpy as np
import pytest

from gzl.direct_sum import (
    _direct_sum_extrapolated_grid,
    _direct_sum_open_terminal,
    _collapse_multi_edges,
    _osc_freqs,
)

# A real treewidth-3 block from the order-13 1qp corpus (K4 with one
# doubled edge), the most frequent 4-vertex tw=3 block in that corpus.
BLOCK = [(0, 1, 1), (2, 0, 1), (2, 1, 1), (2, 3, 2), (3, 0, 1), (3, 1, 1)]

D1_L_LIST = (4, 5, 6, 7, 8)      # shipped d=1 ladder
D2_L_LIST = (3, 4, 5, 6)         # shipped d=2 ladder


def _block_arrays(nu0):
    edges = np.array([[u, v] for u, v, _ in BLOCK], dtype=int)
    nu = np.array([m * nu0 for _, _, m in BLOCK], dtype=float)
    return edges, nu


def _bz(n, d):
    ax = [np.arange(n, dtype=float) / n] * d
    return np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, d)


class TestOscFreqFolding:

    @pytest.mark.parametrize("f", [0.0625, 0.125, 0.25, 0.375])
    def test_f_and_one_minus_f_are_the_same_mode(self, f):
        assert _osc_freqs([f]) == _osc_freqs([1.0 - f])

    def test_folded_into_the_half_open_unit_half(self):
        for f in (0.0625, 0.3, 0.5, 0.7, 0.9375):
            got = _osc_freqs([f])
            assert all(0.0 < g <= 0.5 + 1e-12 for g in got), (f, got)

    def test_zero_and_integer_momenta_do_not_oscillate(self):
        assert _osc_freqs([0.0]) == []
        assert _osc_freqs([1.0]) == []
        assert _osc_freqs(None) == []


class TestGridMomentumSymmetry:
    """zeta_G(k) == zeta_G(1-k) must survive the extrapolator."""

    @pytest.mark.parametrize("d,n,L_list,nu0", [
        (1, 16, D1_L_LIST, 2.75),
        (1, 16, D1_L_LIST, 2.25),
        (2, 8, D2_L_LIST, 2.75),
    ])
    def test_grid_is_even_in_momentum(self, d, n, L_list, nu0):
        edges, nu = _block_arrays(nu0)
        kg = _bz(n, d)
        out = _direct_sum_extrapolated_grid(
            edges, nu, np.eye(d), kg, root=0, terminal=1,
            L_list=L_list, n_correction_terms=min(3, len(L_list) - 1),
        ).reshape((n,) * d)
        axes = tuple(range(d))
        mirrored = np.flip(np.roll(out, -1, axis=axes), axis=axes)
        rel = np.max(np.abs(out - mirrored) / np.maximum(np.abs(out), 1e-300))
        assert rel < 1e-11, f"k <-> 1-k asymmetry {rel:.3e}"


class TestNeverWorseThanTheRawBox:
    """On the shipped ladders the fit must never lose to no fit.

    Neutral reference: the box at a far larger L than the ladder uses,
    which is the same object the ladder is trying to reach.  Quoted at a
    reach (d=1 L=60, d=2 L=18) where its own residual is orders below
    the errors compared.
    """

    @pytest.mark.parametrize("d,n,L_list,L_ref,nu0", [
        (1, 16, D1_L_LIST, 60, 2.75),
        (2, 8, D2_L_LIST, 18, 2.75),
    ])
    def test_extrapolated_grid_never_loses_to_the_raw_box(
            self, d, n, L_list, L_ref, nu0):
        edges, nu = _block_arrays(nu0)
        A = np.eye(d)
        kg = _bz(n, d)
        em = _collapse_multi_edges(edges, nu)

        M_ref, pos_ref = _direct_sum_open_terminal(em, A, L_ref, 0, 1, d)
        ref = np.cos(2 * np.pi * (kg @ pos_ref.T)) @ M_ref

        M_raw, pos_raw = _direct_sum_open_terminal(em, A, max(L_list), 0, 1, d)
        raw = np.cos(2 * np.pi * (kg @ pos_raw.T)) @ M_raw

        ext = _direct_sum_extrapolated_grid(
            edges, nu, A, kg, root=0, terminal=1,
            L_list=L_list, n_correction_terms=min(3, len(L_list) - 1),
        )
        e_raw = np.abs(raw - ref) / np.abs(ref)
        e_ext = np.abs(ext - ref) / np.abs(ref)
        worse = int(np.sum(e_ext > e_raw * 1.05))
        assert worse == 0, (
            f"{worse}/{e_raw.size} momenta worse than the raw box; "
            f"worst blow-up {np.max(e_ext / e_raw):.2f}x"
        )
