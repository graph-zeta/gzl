# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

"""Tests for ``graph_compress(compress_singularities=True)`` and the
zero-coefficient filter inside ``_merge_duplicate_exponents``.

The flag exists to absorb exponents at the ``Gamma((d - nu)/2)`` pole
family ``nu = d + 2k`` (k >= 0) into ``aMat`` before
:func:`graph_multiply` reaches its prefactor singularity check.  It is
opt-in (``False`` by default), so existing behaviour is unchanged for
all callers that don't ask for it; ``graph_multiply`` itself opts in
on its inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gzl import (
    PrefactorSingularityError,
    graph_compress,
    graph_multiply,
    graph_multiply_power,
    graph_zero,
    make_epstein_graph,
    make_graph_obj,
)
from gzl.core import _merge_duplicate_exponents, _vint_array


A_1D = np.array([[1.0]])


# ---------------------------------------------------------------------------
# 1. Default behaviour is unchanged when the flag is False.
# ---------------------------------------------------------------------------


def test_compress_default_unchanged_below_threshold():
    """nu safely below d + sigma_max stays in (bVec, nuVec); aMat untouched."""
    n_pts = 16
    g = make_epstein_graph(nu=1.5 + math.pi / 300.0, A=A_1D, n_samples=n_pts)
    out = graph_compress(g, sigma_max=4.0)

    assert out.nuVec.size == 1
    assert np.allclose(out.aMat, 0.0)
    assert np.isclose(float(out.nuVec[0]), 1.5 + math.pi / 300.0)


def test_compress_default_drops_above_threshold():
    """nu above d + sigma_max gets absorbed into aMat (existing behaviour)."""
    n_pts = 16
    g = make_epstein_graph(nu=6.0, A=A_1D, n_samples=n_pts)
    out = graph_compress(g, sigma_max=4.0)

    assert out.nuVec.size == 0
    assert np.linalg.norm(out.aMat) > 0


def test_compress_flag_off_vs_default_identical():
    """Explicit compress_singularities=False matches the default exactly."""
    n_pts = 16
    nu = 1.5 + math.pi / 300.0
    g = make_epstein_graph(nu=nu, A=A_1D, n_samples=n_pts)

    out_default = graph_compress(g, sigma_max=4.0)
    out_explicit = graph_compress(g, sigma_max=4.0, compress_singularities=False)

    assert np.array_equal(out_default.aMat, out_explicit.aMat)
    assert np.array_equal(out_default.bVec, out_explicit.bVec)
    assert np.array_equal(out_default.nuVec, out_explicit.nuVec)


# ---------------------------------------------------------------------------
# 2. Flag-on absorbs nu = d + 2k into aMat.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3])
def test_singularity_drop_at_d_plus_2k(k: int):
    """nu exactly d + 2k (k>=1) goes from (bVec, nuVec) into aMat; remaining nuVec is empty.

    k = 0 (nu = d, the convergence boundary) is intentionally excluded
    from the absorbing rule — see graph_compress's docstring.
    """
    n_pts = 16
    d = 1
    nu = float(d + 2 * k)
    g = make_epstein_graph(nu=nu, A=A_1D, n_samples=n_pts)

    out = graph_compress(g, sigma_max=4.0, compress_singularities=True)

    assert out.nuVec.size == 0, (
        f"expected nu = d + 2*{k} = {nu} to be absorbed into aMat, "
        f"but nuVec is {out.nuVec}"
    )
    # aMat should hold _vint_array(Az, nu) for the corresponding lattice
    pos = np.arange(0, int(np.floor(n_pts / 2)) + 1, dtype=int)
    neg = np.arange(-int(np.ceil(n_pts / 2)) + 1, 0, dtype=int)
    z_axis = np.concatenate([pos, neg])
    z_grids = np.meshgrid(*([z_axis] * d), indexing="ij")
    z_array = np.stack(z_grids, axis=-1)
    Az = np.tensordot(z_array, A_1D.T, axes=([d], [0]))
    expected = _vint_array(Az, nu)

    assert np.allclose(out.aMat, expected)


def test_singularity_drop_within_tolerance():
    """nu within singularity_tol of d + 2k is absorbed; outside is kept."""
    n_pts = 16
    g_in_band = make_epstein_graph(nu=3.02, A=A_1D, n_samples=n_pts)  # |3.02 - 3| = 0.02
    g_out_band = make_epstein_graph(nu=3.10, A=A_1D, n_samples=n_pts)  # |3.10 - 3| = 0.10

    out_in = graph_compress(
        g_in_band, sigma_max=4.0, compress_singularities=True, singularity_tol=0.05
    )
    out_out = graph_compress(
        g_out_band, sigma_max=4.0, compress_singularities=True, singularity_tol=0.05
    )

    assert out_in.nuVec.size == 0, "nu = 3.02 should be absorbed (|3.02-3| < 0.05)"
    assert out_out.nuVec.size == 1, "nu = 3.10 should be kept (|3.10-3| > 0.05)"
    assert np.isclose(float(out_out.nuVec[0]), 3.10)


def test_singularity_below_threshold_still_dropped_when_flag_set():
    """A pole-class nu (d+2k, k>=1) sitting below d+sigma_max is dropped only
    with the flag — without it, it would stay analytic."""
    n_pts = 16
    nu = 3.0  # d + 2 in d=1 (k=1)
    g = make_epstein_graph(nu=nu, A=A_1D, n_samples=n_pts)

    out_no_flag = graph_compress(g, sigma_max=4.0)
    out_flag = graph_compress(g, sigma_max=4.0, compress_singularities=True)

    # Without the flag: nu=3 < 1+4=5, so it stays analytic.
    assert out_no_flag.nuVec.size == 1
    assert np.allclose(out_no_flag.aMat, 0.0)

    # With the flag: nu=3 = d + 2 (k=1) hits the pole, absorbed into aMat.
    assert out_flag.nuVec.size == 0
    assert np.linalg.norm(out_flag.aMat) > 0


def test_boundary_at_d_not_absorbed():
    """nu close to d (the k=0 case, convergence boundary) is intentionally
    NOT absorbed by compress_singularities — that band is where sigma_max
    algebra works best."""
    n_pts = 16
    # nu = 1.05 in d=1: |nu - d| = 0.05, would be in the band, but k=0 excluded.
    g = make_epstein_graph(nu=1.05, A=A_1D, n_samples=n_pts)
    out = graph_compress(g, sigma_max=4.0, compress_singularities=True)
    assert out.nuVec.size == 1
    assert np.isclose(float(out.nuVec[0]), 1.05)
    assert np.allclose(out.aMat, 0.0)


def test_mixed_pole_and_regular_exponents():
    """Multiple exponents — the pole-class one is absorbed, the regular one is kept."""
    n_pts = 16
    A = A_1D
    aMat = np.zeros((n_pts,), dtype=complex)
    bVec = np.array([1.0, 1.0], dtype=complex)
    nuVec = np.array([3.0, 1.5 + math.pi / 300.0], dtype=float)
    g = make_graph_obj(aMat, bVec, nuVec, A)

    out = graph_compress(g, sigma_max=4.0, compress_singularities=True)

    assert out.nuVec.size == 1
    assert np.isclose(float(out.nuVec[0]), 1.5 + math.pi / 300.0)
    assert np.linalg.norm(out.aMat) > 0


# ---------------------------------------------------------------------------
# 3. graph_multiply at nu = d + 2k no longer raises.
# ---------------------------------------------------------------------------


def test_graph_multiply_pole_path_does_not_double_back():
    """Sanity: applying graph_multiply twice (i.e. nu=d+2 chained) still
    returns finite values — exercises the auto-clean of cross-product
    zeros at nu1+nu2 = 2d+2n."""
    n_pts = 32
    g = make_epstein_graph(nu=3.0, A=A_1D, n_samples=n_pts)
    out2 = graph_multiply(g, g, sigma_max=4.0)
    out3 = graph_multiply(out2, g, sigma_max=4.0)
    assert math.isfinite(graph_zero(out2))
    assert math.isfinite(graph_zero(out3))


def test_graph_multiply_power_at_pole():
    """graph_multiply_power inherits the fix from graph_multiply."""
    n_pts = 32
    g = make_epstein_graph(nu=3.0, A=A_1D, n_samples=n_pts)
    out = graph_multiply_power(g, n_exp=3, sigma_max=4.0)
    assert math.isfinite(graph_zero(out))


# ---------------------------------------------------------------------------
# 4. Cross-zero filter in _merge_duplicate_exponents.
# ---------------------------------------------------------------------------


def test_merge_filters_zero_coefficients():
    """An entry with |b| < 1e-16 is removed; one with |b| just above it is kept."""
    b_vec = np.array([1e-17, 1e-15, 2.0], dtype=complex)
    nu_vec = np.array([2.0, 3.0, 4.0], dtype=float)
    b_out, nu_out = _merge_duplicate_exponents(b_vec, nu_vec)

    assert b_out.shape == (2,)
    assert np.allclose(nu_out, [3.0, 4.0])


def test_merge_summing_to_zero_is_dropped():
    """Two entries at the same nu with opposite-sign coefficients merge to zero
    and the merged entry is dropped."""
    b_vec = np.array([1.0, -1.0, 2.0], dtype=complex)
    nu_vec = np.array([2.5, 2.5, 3.5], dtype=float)
    b_out, nu_out = _merge_duplicate_exponents(b_vec, nu_vec)

    # The (1, -1) entries at nu=2.5 cancel; only nu=3.5 survives.
    assert b_out.shape == (1,)
    assert np.isclose(float(nu_out[0]), 3.5)


# ---------------------------------------------------------------------------
# 5. Existing legacy tolerance offset path still works (no regression).
# ---------------------------------------------------------------------------


def test_irrational_offset_path_still_works():
    """The pre-existing pi/300 offset workaround is unaffected."""
    n_pts = 16
    g = make_epstein_graph(nu=3.0 + math.pi / 300.0, A=A_1D, n_samples=n_pts)
    out = graph_multiply(g, g, sigma_max=4.0)
    assert math.isfinite(graph_zero(out))
