# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Regression tests for the sigma-resonance fix in the SP algebra.

A ``graph_multiply`` cross term whose exponent lands on the
``Gamma((d - nu)/2)`` pole family ``nu = d + 2n`` is a 0 x inf limit: its
prefactor has a structural zero while the next multiplication's Gamma pole
restores an O(1) exponent tower above it.  Dropping the near-pole entry
(the historical behaviour, via merge-drop at the exact zero or the
``compress_singularities`` band otherwise) deleted that tower and cost
~3 orders of accuracy at resonant sigma (m * sigma in 2N for a
chain-reachable m) — e.g. 8.7e-2 relative error for a 13-cycle at
sigma = 0.25, n = 124, against a ~1e-5 background.

The fix emits degenerate channels at the canonical exponent
``d + 2n + _POLE_CANONICAL_OFFSET`` with the residue-limit prefactor
(:func:`gzl.core._mult_prefactor_degenerate`), and
``graph_compress(compress_singularities=True)`` exempts exactly those
canonical exponents while still absorbing genuine near-pole *inputs*
(a sigma = 2 edge, a Hadamard bundle at sigma = k - d/2), which the band
handles accurately as long as a convolution follows the product
(``TestAbsorbedInputs``).

These tests pin: the resonant cycles at background accuracy, the leaf-pole
path unchanged, off-window bit-exactness (the fix is window-gated), the
canonical exemption in graph_compress, and the residue-limit prefactor
against a finite difference of the ordinary prefactor.
"""

import numpy as np
import pytest

import gzl.core as core
from gzl.circle import zeta_circle
from gzl.construction import graph_from_edges_uniform
from gzl.core import (
    _mult_prefactor,
    _mult_prefactor_degenerate,
    _POLE_CANONICAL_OFFSET,
    graph_compress,
    graph_zero,
    make_graph_obj,
)

A1 = np.array([[1.0]])


def _cycle_algebra(L: int, sigma: float, n: int) -> float:
    edges = [(i, (i + 1) % L) for i in range(L)]
    g = graph_from_edges_uniform(edges, 1.0 + sigma, A1, n)
    return graph_zero(g)


def _cycle_rel_err(L: int, sigma: float, n: int) -> float:
    ref = zeta_circle([1.0 + sigma] * L, A1)
    return abs(_cycle_algebra(L, sigma, n) - ref) / abs(ref)


class TestResonantCycles:
    """The headline resonances sit at background accuracy, not 1e-2."""

    @pytest.mark.parametrize(
        "sigma, bound",
        [
            (0.25, 1e-4),   # m = 8 hits nu = 3 = d + 2; was 8.7e-2
            (0.20, 1e-4),   # m = 10 hits nu = 3; was 1.7e-2
            (0.50, 2e-4),   # m = 4, 8 hit nu = 3 and 5; was 1.1e-2
            (1.00, 1e-4),   # m = 2, 4, ... all resonate; was 1.5e-4
        ],
    )
    def test_resonant_sigma_at_background(self, sigma, bound):
        assert _cycle_rel_err(13, sigma, 124) < bound

    def test_band_detuning_stays_bounded(self):
        # Inside the old 1e-4 band but off the canonical offset the
        # representation is frozen at the canonical detuning; the induced
        # error grows ~2 * |delta_nu - offset| — bounded, and far below
        # the historical 8.7e-2 plateau.
        err = _cycle_rel_err(13, 0.25 + 5e-5 / 8.0, 124)
        assert err < 1e-3

    def test_leaf_pole_route_unchanged(self):
        # sigma = 2: the *edge* exponent sits on the pole (nu = 3 = d+2).
        # This is the band-absorption route (accurate, measured 8.2e-8 at
        # n = 124) and must stay untouched by the birth canonicalisation.
        assert _cycle_rel_err(13, 2.0, 124) < 5e-7


class TestWindowGating:
    """Off the pole window the fix is a bit-exact no-op."""

    @pytest.mark.parametrize("sigma", [0.3547, 0.25 + np.pi / 30, 0.75])
    def test_off_window_bit_exact(self, sigma, monkeypatch):
        after = _cycle_algebra(9, sigma, 64)
        # _POLE_BIRTH_WINDOW = 0 reproduces the pre-fix code path exactly:
        # degenerate channels fall back to _mult_prefactor + integer snap,
        # and no canonical exponent exists for the compress exemption.
        monkeypatch.setattr(core, "_POLE_BIRTH_WINDOW", 0.0)
        before = _cycle_algebra(9, sigma, 64)
        assert after == before  # bit-exact


class TestCanonicalExemption:
    def test_canonical_entry_kept_leaf_absorbed(self):
        d = 1
        n_pts = 16
        nu_canon = (d + 2.0) + _POLE_CANONICAL_OFFSET
        g = make_graph_obj(
            np.zeros((n_pts,), dtype=complex),
            [1.0, 1.0],
            [nu_canon, d + 2.0],
            A1,
        )
        out = graph_compress(g, 4.0, compress_singularities=True)
        # The exact-pole leaf is absorbed; the canonical entry survives.
        assert out.nuVec.size == 1
        assert out.nuVec[0] == nu_canon

    def test_in_band_non_canonical_still_absorbed(self):
        d = 1
        n_pts = 16
        g = make_graph_obj(
            np.zeros((n_pts,), dtype=complex),
            [1.0],
            [d + 2.0 + 5e-5],
            A1,
        )
        out = graph_compress(g, 4.0, compress_singularities=True)
        assert out.nuVec.size == 0


class TestDegeneratePrefactor:
    @pytest.mark.parametrize("d", [1, 2, 3])
    @pytest.mark.parametrize("n", [1, 2])
    @pytest.mark.parametrize("s1", [0.3, 0.7, 1.3])
    def test_matches_finite_difference_slope(self, d, n, s1):
        # _mult_prefactor vanishes linearly in the detuning t at
        # nu1 + nu2 = 2d + 2n; the degenerate form is (slope * offset).
        nu1 = d + s1
        nu2_pole = 2.0 * d + 2.0 * n - nu1
        t = 1e-7
        slope = _mult_prefactor(nu1, nu2_pole + t, d, 1.0) / t
        got = _mult_prefactor_degenerate(nu1, n, d, 1.0)
        want = slope * _POLE_CANONICAL_OFFSET
        assert got == pytest.approx(want, rel=1e-5)

    def test_mpmath_matches_float64(self):
        # mpmath is an optional dependency: _mult_prefactor imports it
        # lazily and only for precision="mpmath".
        pytest.importorskip("mpmath")
        for d, n, s1 in [(1, 1, 0.3), (2, 2, 0.7), (3, 1, 1.3)]:
            f64 = _mult_prefactor_degenerate(d + s1, n, d, 1.0)
            mp = _mult_prefactor_degenerate(
                d + s1, n, d, 1.0, precision="mpmath"
            )
            assert mp == pytest.approx(f64, rel=1e-10)

    def test_exact_pole_input_returns_zero(self):
        # nu1 exactly on the pole family cannot legitimately reach the
        # degenerate branch (the band absorbs it first); the conservative
        # guard returns 0.0 instead of a Gamma pole.
        assert _mult_prefactor_degenerate(3.0, 1, 1, 1.0) == 0.0


class TestButterfly:
    """1-sum of two triangles — the historically dramatic failure class."""

    @pytest.mark.parametrize(
        "sigma, bound",
        [
            (1.0, 5e-4),          # mid-chain resonance in each triangle
            (2.0, 5e-4),          # every edge on the pole family (leaf route)
            (0.5 + np.pi / 300, 5e-4),   # generic control
        ],
    )
    def test_butterfly_matches_block_factorisation(self, sigma, bound):
        edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]
        g = graph_from_edges_uniform(edges, 1.0 + sigma, A1, 124)
        got = graph_zero(g)
        ref = zeta_circle([1.0 + sigma] * 3, A1) ** 2
        assert abs(got - ref) / abs(ref) < bound


class TestAbsorptionRing:
    """The pole flank: O(1)-coefficient entries just OUTSIDE the band.

    Such an entry is amplified by ~1/distance through the multiplication
    prefactor (measured 15.5% relative on a 6-cycle with one chord at
    sigma near 1/2).  The coefficient-gated ring absorbs it; degenerate
    birth generators, whose coefficient is proportional to that same
    distance, must be kept.
    """

    HEX = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]

    def _hex(self, sigma, n):
        from gzl.construction import graph_from_edges
        g = graph_from_edges(
            np.asarray(self.HEX), np.full(7, 1.0 + sigma), A1, n,
            s=0, t=0, sigma_max=4.0,
        )
        return graph_zero(g)

    # 2e-2 is deliberately absent: _POLE_ABSORB_RATIO caps a share-0.65
    # flank at dist ~ 0.0065, because beyond ~0.02 absorption stops being
    # reliably better (measured geo-mean 1.8 but with 1-in-6 blocks made
    # worse, and no locally computable way to tell which).
    @pytest.mark.parametrize("d_nu", [1.04e-4, 2e-4, 1e-3, 5e-3])
    def test_flank_is_absorbed(self, d_nu, monkeypatch):
        sigma = 0.5 + d_nu / 2.0
        ref = self._hex(sigma, 1024)
        got = self._hex(sigma, 124)
        assert abs(got - ref) / abs(ref) < 1e-4      # was up to 1.6e-1

    def test_leaf_flank_is_continuous_across_the_band_edge(self):
        # bowtie: two triangles sharing a vertex; at sigma = 2 the edge
        # exponent sits ON the pole (absorbed by the band), and just
        # outside it used to jump by four orders.
        edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]
        errs = []
        for d_nu in (0.0, 2e-4, 1e-3, 5e-3):
            sigma = 2.0 + d_nu
            ref = zeta_circle([1.0 + sigma] * 3, A1) ** 2
            got = graph_zero(
                graph_from_edges_uniform(edges, 1.0 + sigma, A1, 124)
            )
            errs.append(abs(got - ref) / abs(ref))
        assert max(errs) < 1e-7                      # was 5.0e-5 at 2e-4
        assert max(errs) / min(errs) < 10.0          # continuous

    @pytest.mark.parametrize("L", [5, 9, 13, 14])
    def test_generators_are_never_absorbed(self, L, monkeypatch):
        # Cycles are pure-generator blocks: their only convolution is the
        # last operation, so every near-pole entry reaching a multiply is
        # a degenerate cross birth.  The ring must not touch them.
        for sigma in (0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.297, 1.3325):
            after = graph_zero(
                graph_from_edges_uniform(
                    [(i, (i + 1) % L) for i in range(L)], 1.0 + sigma, A1, 64
                )
            )
            ref = zeta_circle([1.0 + sigma] * L, A1)
            assert abs(after - ref) / abs(ref) < 1e-3


class TestConditioningGuard:
    """Deep chains at small sigma destroy the float64 conditioning.

    The chain stays healthy; the final convolution shifts every exponent
    up by one edge weight, collapsing the Epstein-weight spread that had
    been holding the alternating-sign coefficients in check.  The
    frontend refuses such a block rather than shipping the result.
    """

    @staticmethod
    def _theta(m1, m2, m3):
        edges, nxt = [], 2
        for length in (m1, m2, m3):
            prev = 0
            for _ in range(length - 1):
                edges.append((prev, nxt))
                prev, nxt = nxt, nxt + 1
            edges.append((prev, 1))
        return np.asarray(edges)

    def test_condition_tracks_the_actual_error(self):
        from gzl.core import graph_zero_conditioned
        for L, sigma, lo, hi in [(13, 0.02, 1e13, 1e16), (13, 0.5, 1.0, 1e7)]:
            g = graph_from_edges_uniform(
                [(i, (i + 1) % L) for i in range(L)], 1.0 + sigma, A1, 124
            )
            _, condition = graph_zero_conditioned(g)
            assert lo < condition < hi

    @pytest.mark.parametrize("sigma", [0.02, 0.05])
    def test_ill_conditioned_block_is_refused_not_shipped(self, sigma):
        from gzl.frontend import evaluate_graph
        edges = self._theta(12, 12, 1)
        value, info = evaluate_graph(
            edges, 1.0 + sigma, A1, n_points=64, return_diagnostics=True,
        )
        assert info["n_block_algebra_refused"] == 1
        assert info["n_block_algebra"] == 0
        # zeta_G(0) is a sum of non-negative kernel products; the algebra
        # used to return a large NEGATIVE number here.
        assert value > 0.0

    @pytest.mark.parametrize("sigma", [0.5, 1.0])
    def test_well_conditioned_blocks_keep_the_algebra(self, sigma):
        from gzl.frontend import evaluate_graph
        for edges in (self._theta(3, 3, 1), self._theta(12, 12, 1)):
            _, info = evaluate_graph(
                edges, 1.0 + sigma, A1, n_points=64, return_diagnostics=True,
            )
            assert info["n_block_algebra_refused"] == 0
            assert info["n_block_algebra"] == 1


class TestAbsorbedInputs:
    """An O(1) operand exponent on nu = d + 2n is absorbed into aMat with
    its kernel cut off at the grid window.

    A convolution after the product suppresses the lost tail, which is
    what every block of evaluate_graph and the T^n benchmarks of the
    mathematics paper do.  A product sampled directly keeps it, and an
    irrational offset of nu is still the remedy there.  DOCUMENTATION.md
    ("The sigma_max parameter") states both halves; these tests hold it
    to them.  Here the triangle carries the exponent 2 nu = 3 = d + 2.
    """

    @staticmethod
    def _triangle(nu, n):
        from gzl.core import graph_convolve, graph_multiply, make_epstein_graph
        g = make_epstein_graph(nu, A1, n)
        return graph_convolve(graph_multiply(g, g), g), g

    def _open(self, nu, n):
        from gzl.core import graph_multiply
        tri, g = self._triangle(nu, n)
        return graph_zero(graph_multiply(tri, g))

    def _closed(self, nu, n):
        from gzl.core import graph_convolve, graph_multiply
        tri, g = self._triangle(nu, n)
        return graph_zero(graph_convolve(graph_multiply(tri, g), g))

    @staticmethod
    def _open_exact(nu):
        from scipy.special import zeta
        return zeta_circle([nu] * 3, A1) * 2.0 * zeta(nu)

    def _open_err(self, nu, n):
        return abs(self._open(nu, n) / self._open_exact(nu) - 1.0)

    def test_closed_product_is_at_background(self):
        # (triangle * edge) conv edge, the block T^2 of the paper; the
        # reference is the same algebra at n = 2048, where the error has
        # dropped by ~500x.  Measured: 7.8e-7 at nu = 1.5, 9.6e-7 at
        # nu = 1.5 + pi/30.
        errs = {}
        for nu in (1.5, 1.5 + np.pi / 30):
            ref = self._closed(nu, 2048)
            errs[nu] = abs(self._closed(nu, 250) / ref - 1.0)
        assert errs[1.5] < 2.0 * errs[1.5 + np.pi / 30]

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="an absorbed input loses its tail in a directly sampled "
        "product (1.7e-4 at nu = 1.5 vs 1.4e-8 off the pole); fixing it "
        "must also update DOCUMENTATION.md, 'The sigma_max parameter'",
    )
    def test_open_product_at_resonance_is_at_background(self):
        assert self._open_err(1.5, 250) < 10.0 * self._open_err(
            1.5 + np.pi / 30, 250
        )

    def test_irrational_offset_rescues_the_open_product(self):
        assert self._open_err(1.5 + np.pi / 30, 250) < 1e-7   # 1.4e-8

    def test_prefactor_error_only_at_or_below_d(self):
        from gzl.core import (
            PrefactorSingularityError,
            graph_multiply,
            make_epstein_graph,
        )
        # nu = d + 2 on the chain and the square lattice: absorbed.
        for nu, A in ((3.0, A1), (4.0, np.eye(2))):
            g = make_epstein_graph(nu, A, 16)
            assert np.isfinite(graph_zero(graph_multiply(g, g)))
        # nu = d, and nu1 + nu2 = d: the prefactor has a pole.
        for nu1, nu2 in ((1.0, 1.0), (0.3, 0.7)):
            with pytest.raises(PrefactorSingularityError):
                graph_multiply(
                    make_epstein_graph(nu1, A1, 16),
                    make_epstein_graph(nu2, A1, 16),
                )
