# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

"""Tests for gzl."""

import numpy as np
import pytest

from gzl import (
    PrefactorSingularityError,
    NotSeriesParallelError,
    NotTreewidthTwoError,
    make_epstein_graph,
    graph_multiply,
    graph_multiply_power,
    graph_convolve,
    graph_convolve_power,
    graph_attach,
    graph_from_sp,
    graph_from_sp_uniform,
    graph_from_tw2,
    graph_from_tw2_uniform,
    graph_from_edges,
    graph_from_edges_uniform,
    graph_sample,
    graph_zero,
    periodic_convolve_nd,
    zeta_circle,
    direct_sum_zero_momentum,
    direct_sum_extrapolated,
    UnsupportedLatticeSumError,
)
from gzl.circle import epstein_zeta_prod

nx = pytest.importorskip("networkx")
epstein_zeta = pytest.importorskip("epsteinlib").epstein_zeta


# ---------------------------------------------------------------------------
# 3D lattice test
# ---------------------------------------------------------------------------

class Test3DLattice:
    """Nontrivial 3D non-square lattice with exponents close to d = 3."""

    A = np.array(
        [[1.0, 0.1, 0.0],
         [0.0, 1.1, 0.0],
         [0.0, 0.2, 1.2]],
        dtype=float,
    )
    N = 20
    REF = 300.7476236777885

    def _compute(self):
        g1 = make_epstein_graph(3.05, self.A, self.N)
        g2 = make_epstein_graph(3.10, self.A, self.N)
        g3 = make_epstein_graph(3.15, self.A, self.N)
        g4 = make_epstein_graph(3.20, self.A, self.N)
        g_res = graph_convolve(graph_multiply(g1, g2), graph_multiply(g3, g4))
        return graph_zero(g_res)

    def test_relative_error(self):
        val = self._compute()
        rel_err = abs(val - self.REF) / abs(self.REF)
        assert rel_err < 1e-3, f"3D lattice rel error {rel_err:.3e} > 1e-3"


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

class TestValidation:
    """Lattice / grid mismatch checks for multiply and convolve."""

    def test_multiply_different_A(self):
        g1 = make_epstein_graph(2.5, np.array([[1.0]]), 100)
        g2 = make_epstein_graph(2.5, np.array([[2.0]]), 100)
        with pytest.raises(ValueError, match="lattice matrices"):
            graph_multiply(g1, g2)

    def test_multiply_different_n(self):
        g1 = make_epstein_graph(2.5, np.array([[1.0]]), 100)
        g2 = make_epstein_graph(2.5, np.array([[1.0]]), 50)
        with pytest.raises(ValueError, match="sample grids"):
            graph_multiply(g1, g2)

    def test_convolve_different_A(self):
        g1 = make_epstein_graph(2.5, np.array([[1.0]]), 100)
        g2 = make_epstein_graph(2.5, np.array([[2.0]]), 100)
        with pytest.raises(ValueError, match="lattice matrices"):
            graph_convolve(g1, g2)

    def test_convolve_different_n(self):
        g1 = make_epstein_graph(2.5, np.array([[1.0]]), 100)
        g2 = make_epstein_graph(2.5, np.array([[1.0]]), 50)
        with pytest.raises(ValueError, match="sample grids"):
            graph_convolve(g1, g2)


# ---------------------------------------------------------------------------
# Prefactor singularity guard
# ---------------------------------------------------------------------------

class TestPrefactorSingularity:
    """graph_multiply absorbs nu = d + 2n into aMat (compress_singularities)
    so the Gamma prefactor never sees the pole.  These tests pin down the
    behaviour: the multiply succeeds, and at nu = d + 2 the product is
    the exact two-edge path to the documented truncation.
    """

    def test_nu_equals_d_plus_2_1d_no_raise(self):
        """d=1, nu=3 (= d + 2): pre-absorbed into aMat by graph_multiply's
        entry compress; no PrefactorSingularityError, finite result."""
        g = make_epstein_graph(3.0, np.array([[1.0]]), 100)
        out = graph_multiply(g, g)  # must not raise
        val = graph_zero(out)
        assert np.isfinite(val)

    def test_nu_equals_d_plus_2_matches_the_exact_two_edge_path(self):
        """d=1, nu=3 (= d + 2).  At k = 0 the two-edge path factorises,
        so its value is zeta_E(3)^2 = (2 zeta(3))^2 on the chain.

        The absorbed input keeps its kernel only inside the grid window,
        and a product sampled directly carries that loss (the open
        product of TestAbsorbedInputs in test_pole_resonance.py):
        measured 3.3e-4 at n_points = 100.  The offset nu = 3 + pi/300
        this test used to compare with differs from the pole value by
        3.4e-3 in the exact sums alone, so it could not see an error
        below that.
        """
        from scipy.special import zeta
        g = make_epstein_graph(3.0, np.array([[1.0]]), 100)
        v = graph_zero(graph_multiply(g, g))
        exact = (2.0 * zeta(3.0)) ** 2
        assert abs(v / exact - 1.0) < 5e-4

    def test_nu_equals_d_plus_4_2d_no_raise(self):
        """d=2, nu=6 (= d + 2*2): pre-absorbed; no error."""
        A = np.eye(2)
        g = make_epstein_graph(6.0, A, 20)
        out = graph_multiply(g, g)
        val = graph_zero(out)
        assert np.isfinite(val)

    def test_cross_term_singularity_check_still_works(self):
        """Direct call to _check_multiply_singularities still flags the
        below-d cross-sum pole (nu1 + nu2 = d - 2n).  This case cannot
        arise from physical inputs (nu_i > d) reaching graph_multiply,
        but the underlying guard is preserved for callers that build
        GraphZeta objects manually."""
        from gzl.core import _check_multiply_singularities
        nu1 = np.array([1.5])
        nu2 = np.array([0.5])
        with pytest.raises(PrefactorSingularityError, match="nu1 \\+ nu2"):
            _check_multiply_singularities(nu1, nu2, d=2)

    def test_compress_singularities_flag_off_still_raises(self):
        """If a caller bypasses the auto-absorb (uses graph_compress with
        the flag off and feeds the result into _check_multiply_singularities
        manually), the guard still works as before."""
        from gzl.core import _check_multiply_singularities
        nu = np.array([3.0])  # d + 2 in d=1
        with pytest.raises(PrefactorSingularityError):
            _check_multiply_singularities(nu, nu, d=1)


# ---------------------------------------------------------------------------
# 1D circle graph vs adaptive integration reference
# ---------------------------------------------------------------------------

class TestCircle1D:
    """Compare graph algebra to zeta_circle reference for the 1D circle."""

    N_NODES = 13
    N_POINTS = 500

    def _graph_value(self, nu: float) -> float:
        g0 = make_epstein_graph(nu, np.array([[1.0]]), self.N_POINTS)
        g_mul = g0
        for _ in range(self.N_NODES - 2):
            g_mul = graph_multiply(g_mul, g0)
        g_test = graph_convolve(g_mul, g0)
        return graph_zero(g_test)

    def _ref_value(self, nu: float) -> float:
        return zeta_circle(np.full(self.N_NODES, nu), np.array([[1.0]]))

    @pytest.mark.parametrize("nu", [1.5 + np.pi / 300, 2.5 + np.pi / 300])
    def test_relative_error(self, nu):
        val = self._graph_value(nu)
        ref = self._ref_value(nu)
        rel_err = abs(val - ref) / abs(ref)
        assert rel_err < 1e-4, f"nu={nu:.4f}: rel error {rel_err:.3e}"


class TestEpsteinZetaProdMemo:
    """``epstein_zeta_prod`` evaluates one Epstein zeta per *distinct*
    exponent.  The product must stay bit-identical to evaluating every
    edge separately — the reuse is a caching change, not a numerical
    one."""

    @staticmethod
    def _reference(nu_vec, A, A_star, y_vec):
        """One epstein_zeta call per edge, as before the memoisation."""
        A = np.array(A, dtype=np.float64)
        A_star = np.array(A_star, dtype=np.float64)
        y_vec = np.array(y_vec, dtype=np.float64)
        zeros = np.zeros(len(A), dtype=np.float64)
        zetas = np.real(
            [epstein_zeta(nu, A, zeros, A_star @ y_vec) for nu in nu_vec]
        )
        return float(np.prod(zetas))

    @pytest.mark.parametrize("A", [
        np.eye(1),
        np.eye(2),
        np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2.0]]),
        np.eye(3),
    ])
    @pytest.mark.parametrize("nu_vec", [
        [2.5, 2.5, 2.5],                     # uniform — one call
        [3.5] * 8,                           # uniform, longer cycle
        [2.5, 3.5, 2.5, 4.25],               # repeats, not adjacent
        [3.25, 4.5, 5.75],                   # all distinct — no reuse
    ])
    def test_bit_identical_to_per_edge(self, A, nu_vec):
        d = A.shape[0]
        A_star = np.linalg.inv(A).T
        for y in ([0.0] * d, [0.3] * d, [-0.21] * d):
            y = np.asarray(y, dtype=float)
            assert epstein_zeta_prod(nu_vec, A, A_star, y) == \
                self._reference(nu_vec, A, A_star, y)

    def test_calls_once_per_distinct_exponent(self, monkeypatch):
        calls = []
        import gzl.circle as circle_mod
        orig = circle_mod.epstein_zeta

        def counting(nu, *args, **kwargs):
            calls.append(float(nu))
            return orig(nu, *args, **kwargs)

        monkeypatch.setattr(circle_mod, "epstein_zeta", counting)
        A = np.eye(1)
        circle_mod.epstein_zeta_prod(
            [2.5] * 12, A, np.eye(1), np.array([0.3]),
        )
        assert calls == [2.5]

        calls.clear()
        circle_mod.epstein_zeta_prod(
            [2.5, 4.0, 2.5, 4.0], A, np.eye(1), np.array([0.3]),
        )
        assert calls == [2.5, 4.0]


# ---------------------------------------------------------------------------
# Periodic convolution utility
# ---------------------------------------------------------------------------

class TestPeriodicConvolve:

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="[Ss]hape"):
            periodic_convolve_nd(np.zeros(10), np.zeros(20))

    def test_delta_identity(self):
        """Convolution with a delta should return the original (scaled)."""
        n = 64
        f = np.random.default_rng(42).standard_normal(n)
        delta = np.zeros(n)
        delta[0] = float(n)  # compensate normalisation
        result = periodic_convolve_nd(f, delta)
        np.testing.assert_allclose(result, f, atol=1e-12)


# ---------------------------------------------------------------------------
# Power helpers
# ---------------------------------------------------------------------------

class TestGraphMultiplyPower:
    """``graph_multiply_power`` — repeated pointwise multiplication."""

    A = np.array([[1.0]])
    nu = 2.5 + np.pi / 300
    n_points = 64

    def _g0(self):
        return make_epstein_graph(self.nu, self.A, self.n_points)

    def test_identity(self):
        """``n_exp = 1`` returns the input unchanged."""
        g = self._g0()
        g_pow = graph_multiply_power(g, 1)
        assert g_pow is g

    def test_matches_repeated_multiply(self):
        """``graph_multiply_power(g, 3)`` matches the manual chain."""
        g = self._g0()
        manual = graph_multiply(graph_multiply(g, g), g)
        via_pow = graph_multiply_power(g, 3)
        np.testing.assert_allclose(
            graph_zero(via_pow), graph_zero(manual), rtol=1e-12
        )
        np.testing.assert_allclose(
            graph_sample(via_pow), graph_sample(manual), rtol=1e-10, atol=1e-12
        )

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="n_exp must be >= 1"):
            graph_multiply_power(self._g0(), 0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="n_exp must be >= 1"):
            graph_multiply_power(self._g0(), -2)

    def test_non_integer_raises(self):
        with pytest.raises(TypeError, match="must be a positive int"):
            graph_multiply_power(self._g0(), 2.0)

    def test_bool_rejected(self):
        with pytest.raises(TypeError, match="must be a positive int"):
            graph_multiply_power(self._g0(), True)


class TestGraphConvolvePower:
    """``graph_convolve_power`` — repeated convolution."""

    A = np.array([[1.0]])
    nu = 1.5 + np.pi / 300
    n_points = 128

    def _g0(self):
        return make_epstein_graph(self.nu, self.A, self.n_points)

    def test_identity(self):
        g = self._g0()
        g_pow = graph_convolve_power(g, 1)
        assert g_pow is g

    def test_matches_repeated_convolve(self):
        """``graph_convolve_power(g, 3)`` matches the manual chain."""
        g = self._g0()
        manual = graph_convolve(graph_convolve(g, g), g)
        via_pow = graph_convolve_power(g, 3)
        np.testing.assert_allclose(
            graph_zero(via_pow), graph_zero(manual), rtol=1e-12
        )
        np.testing.assert_allclose(
            graph_sample(via_pow), graph_sample(manual), rtol=1e-10, atol=1e-12
        )

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="n_exp must be >= 1"):
            graph_convolve_power(self._g0(), 0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="n_exp must be >= 1"):
            graph_convolve_power(self._g0(), -1)

    def test_non_integer_raises(self):
        with pytest.raises(TypeError, match="must be a positive int"):
            graph_convolve_power(self._g0(), 3.5)


# ---------------------------------------------------------------------------
# Binary-exponentiation parity (binary vs linear path)
# ---------------------------------------------------------------------------

class TestPowerBinaryVsLinear:
    """Parity between ``_method='binary'`` (default) and ``_method='linear'``.

    Tolerance is ``rtol=1e-8`` against the dominant grid magnitude to
    absorb floating-point reordering: the two paths perform different
    numbers of compress/FFT round-trips, so bit-exact agreement is not
    expected.  Grid samples span many orders of magnitude, so absolute
    tolerance is scaled to ``max(|reference|)`` rather than fixed.
    """

    A = np.array([[1.0]])
    n_points = 200
    rtol = 1e-8

    @staticmethod
    def _grid_close(a, b, rtol):
        ref_scale = max(np.max(np.abs(b)), 1.0)
        max_abs = np.max(np.abs(a - b))
        assert max_abs <= rtol * ref_scale, (
            f"max|a - b| = {max_abs:.3e}, rtol*scale = "
            f"{rtol * ref_scale:.3e}"
        )

    @pytest.mark.parametrize("n_exp", [1, 2, 3, 4, 5, 7, 8, 9])
    def test_multiply_power_parity(self, n_exp):
        nu = 2.5 + np.pi / 300
        g = make_epstein_graph(nu, self.A, self.n_points)
        binary = graph_multiply_power(g, n_exp, _method="binary")
        linear = graph_multiply_power(g, n_exp, _method="linear")
        np.testing.assert_allclose(
            graph_zero(binary), graph_zero(linear), rtol=self.rtol,
        )
        self._grid_close(graph_sample(binary), graph_sample(linear),
                         self.rtol)

    @pytest.mark.parametrize("n_exp", [13, 16])
    def test_multiply_power_parity_high_n(self, n_exp):
        """Higher exponents accumulate FFT noise inside ``graph_multiply``;
        agreement degrades to ~ ``sqrt(n_exp) * eps`` per operation."""
        nu = 2.5 + np.pi / 300
        g = make_epstein_graph(nu, self.A, self.n_points)
        binary = graph_multiply_power(g, n_exp, _method="binary")
        linear = graph_multiply_power(g, n_exp, _method="linear")
        np.testing.assert_allclose(
            graph_zero(binary), graph_zero(linear), rtol=1e-7,
        )
        self._grid_close(graph_sample(binary), graph_sample(linear),
                         1e-7)

    @pytest.mark.parametrize("n_exp", [1, 2, 3, 4, 5, 7, 8, 9, 13, 16])
    def test_convolve_power_parity(self, n_exp):
        nu = 1.5 + np.pi / 300
        g = make_epstein_graph(nu, self.A, self.n_points)
        binary = graph_convolve_power(g, n_exp, _method="binary")
        linear = graph_convolve_power(g, n_exp, _method="linear")
        np.testing.assert_allclose(
            graph_zero(binary), graph_zero(linear), rtol=self.rtol,
        )
        self._grid_close(graph_sample(binary), graph_sample(linear),
                         self.rtol)

    def test_multiply_power_parity_at_sigma_max_zero_near_a_pole(self):
        """nu = 2.9105 is 0.09 from the prefactor pole at d + 2.  With
        sigma_max = 0 (no Gamma algebra) the two schedules still agree:
        12 edges at n_points = 500."""
        g = make_epstein_graph(2.9105, self.A, 500)
        binary = graph_multiply_power(g, 12, sigma_max=0.0, _method="binary")
        linear = graph_multiply_power(g, 12, sigma_max=0.0, _method="linear")
        np.testing.assert_allclose(
            graph_zero(binary), graph_zero(linear), rtol=self.rtol,
        )

    def test_invalid_method_raises(self):
        g = make_epstein_graph(2.5 + np.pi / 300, self.A, self.n_points)
        with pytest.raises(ValueError, match="_method must be"):
            graph_multiply_power(g, 4, _method="ladder")
        with pytest.raises(ValueError, match="_method must be"):
            graph_convolve_power(g, 4, _method="ladder")


# ---------------------------------------------------------------------------
# 1-sum attachment
# ---------------------------------------------------------------------------

class TestGraphAttach:
    """``graph_attach`` — 1-sum attachment via zero-momentum scalar."""

    A = np.array([[1.0]])
    nu_host = 1.7 + np.pi / 300
    nu_dec = 2.3 + np.pi / 300
    n_points = 128

    def _host(self):
        # Build a non-trivial host with both Fourier and Epstein content.
        g0 = make_epstein_graph(self.nu_host, self.A, self.n_points)
        return graph_multiply(g0, g0)

    def _decoration(self, n_points=None):
        n = self.n_points if n_points is None else n_points
        return make_epstein_graph(self.nu_dec, self.A, n)

    def test_zero_momentum_factorizes(self):
        """``graph_zero(attach(g, d)) == graph_zero(g) * graph_zero(d)``."""
        g = self._host()
        d = self._decoration()
        attached = graph_attach(g, d)
        expected = graph_zero(g) * graph_zero(d)
        np.testing.assert_allclose(graph_zero(attached), expected, rtol=1e-12)

    def test_full_sample_scales(self):
        """Sampled zeta scales uniformly by the decoration scalar."""
        g = self._host()
        d = self._decoration()
        c = graph_zero(d)
        attached = graph_attach(g, d)
        np.testing.assert_allclose(
            graph_sample(attached), c * graph_sample(g),
            rtol=1e-10, atol=1e-12,
        )

    def test_nuVec_preserved(self):
        """The singular-exponent list is inherited from the host."""
        g = self._host()
        d = self._decoration()
        attached = graph_attach(g, d)
        np.testing.assert_array_equal(attached.nuVec, g.nuVec)

    def test_lattice_preserved(self):
        g = self._host()
        d = self._decoration()
        attached = graph_attach(g, d)
        np.testing.assert_array_equal(attached.A, g.A)

    def test_grid_sizes_may_differ(self):
        """Host and decoration may live on different grids; only A must match."""
        g = self._host()  # n_points = 128
        d_coarse = self._decoration(n_points=64)
        d_fine = self._decoration(n_points=256)
        z_coarse = graph_zero(graph_attach(g, d_coarse))
        z_fine = graph_zero(graph_attach(g, d_fine))
        # Both should converge to the same limit; tolerate coarse-grid slack.
        np.testing.assert_allclose(z_coarse, z_fine, rtol=1e-3)

    def test_lattice_mismatch_raises(self):
        g = self._host()
        d_other = make_epstein_graph(
            self.nu_dec, np.array([[1.5]]), self.n_points
        )
        with pytest.raises(ValueError, match="lattice matrices A must match"):
            graph_attach(g, d_other)


# ---------------------------------------------------------------------------
# graph_from_sp / graph_from_sp_uniform
# ---------------------------------------------------------------------------

class TestGraphFromSP:
    """Series-parallel multigraph constructor.

    Exercises the SP-reduction decomposer by comparing its output to
    equivalent graphs built by hand from the algebraic primitives.
    """

    A = np.array([[1.0]])
    n_points = 64
    nu = 1.0 + np.pi / 30  # irrational offset to avoid Gamma poles

    @staticmethod
    def _sample_close(g1, g2, atol=1e-10, rtol=1e-10):
        """Two GraphZetas agree on the full k-grid."""
        s1 = graph_sample(g1)
        s2 = graph_sample(g2)
        np.testing.assert_allclose(s1, s2, atol=atol, rtol=rtol)

    # --- smoke ------------------------------------------------------------

    def test_single_edge(self):
        """An s-t graph with a single edge reduces to make_epstein_graph."""
        G = nx.MultiGraph()
        G.add_edge("s", "t", nu=self.nu)
        out = graph_from_sp(G, "s", "t", self.A, self.n_points)
        ref = make_epstein_graph(self.nu, self.A, self.n_points)
        self._sample_close(out, ref)

    def test_path_two_edges(self):
        """s -- u -- t reduces to graph_multiply."""
        G = nx.MultiGraph()
        G.add_edge("s", "u", nu=self.nu)
        G.add_edge("u", "t", nu=self.nu)
        out = graph_from_sp(G, "s", "t", self.A, self.n_points)
        g0 = make_epstein_graph(self.nu, self.A, self.n_points)
        ref = graph_multiply(g0, g0)
        self._sample_close(out, ref)

    def test_theta_graph_three_parallel(self):
        """Three parallel s-t edges reduce to graph_convolve_power(g, 3)."""
        G = nx.MultiGraph()
        for _ in range(3):
            G.add_edge("s", "t", nu=self.nu)
        out = graph_from_sp(G, "s", "t", self.A, self.n_points)
        g0 = make_epstein_graph(self.nu, self.A, self.n_points)
        ref = graph_convolve_power(g0, 3)
        self._sample_close(out, ref)

    def test_triangle_plus_direct(self):
        """Triangle s-u-t in parallel with a direct s-t edge."""
        G = nx.MultiGraph()
        G.add_edge("s", "u", nu=self.nu)
        G.add_edge("u", "t", nu=self.nu)
        G.add_edge("s", "t", nu=self.nu)
        out = graph_from_sp(G, "s", "t", self.A, self.n_points)
        g0 = make_epstein_graph(self.nu, self.A, self.n_points)
        ref = graph_convolve(graph_multiply(g0, g0), g0)
        self._sample_close(out, ref)

    def test_per_edge_distinct_nu(self):
        """Different nu on each edge of a path is honoured."""
        nu1 = 1.0 + np.pi / 30
        nu2 = 1.3 + np.pi / 30
        G = nx.MultiGraph()
        G.add_edge("s", "u", nu=nu1)
        G.add_edge("u", "t", nu=nu2)
        out = graph_from_sp(G, "s", "t", self.A, self.n_points)
        g1 = make_epstein_graph(nu1, self.A, self.n_points)
        g2 = make_epstein_graph(nu2, self.A, self.n_points)
        ref = graph_multiply(g1, g2)
        self._sample_close(out, ref)

    def test_sp_bridge_diamond(self):
        """SP 'diamond': (s-a-t) || (s-b-t) — two parallel 2-paths."""
        G = nx.MultiGraph()
        G.add_edge("s", "a", nu=self.nu)
        G.add_edge("a", "t", nu=self.nu)
        G.add_edge("s", "b", nu=self.nu)
        G.add_edge("b", "t", nu=self.nu)
        out = graph_from_sp(G, "s", "t", self.A, self.n_points)
        g0 = make_epstein_graph(self.nu, self.A, self.n_points)
        path = graph_multiply(g0, g0)
        ref = graph_convolve(path, path)
        self._sample_close(out, ref)

    def test_longer_path(self):
        """A 4-edge path reduces to graph_multiply_power(g, 4)."""
        G = nx.MultiGraph()
        G.add_edge("s", "u", nu=self.nu)
        G.add_edge("u", "v", nu=self.nu)
        G.add_edge("v", "w", nu=self.nu)
        G.add_edge("w", "t", nu=self.nu)
        out = graph_from_sp(G, "s", "t", self.A, self.n_points)
        g0 = make_epstein_graph(self.nu, self.A, self.n_points)
        ref = graph_multiply_power(g0, 4)
        self._sample_close(out, ref)

    # --- non-SP rejection -------------------------------------------------

    def test_k4_not_series_parallel(self):
        """K_4 between opposite corners has tw = 3 and must be rejected."""
        G = nx.MultiGraph()
        verts = ["s", "u", "v", "t"]
        for i, a in enumerate(verts):
            for b in verts[i + 1:]:
                G.add_edge(a, b, nu=self.nu)
        with pytest.raises(NotSeriesParallelError):
            graph_from_sp(G, "s", "t", self.A, self.n_points)

    def test_dangling_tree_rejected(self):
        """Stage-1 SP reduction rejects graphs with leaves.

        Such graphs are tw-2 but require 1-sum attachments (graph_attach),
        which stage 1 does not implement.
        """
        G = nx.MultiGraph()
        G.add_edge("s", "t", nu=self.nu)
        G.add_edge("s", "dangle", nu=self.nu)
        with pytest.raises(NotSeriesParallelError):
            graph_from_sp(G, "s", "t", self.A, self.n_points)

    # --- validation -------------------------------------------------------

    def test_terminals_must_differ(self):
        G = nx.MultiGraph()
        G.add_edge("s", "t", nu=self.nu)
        with pytest.raises(ValueError, match="distinct"):
            graph_from_sp(G, "s", "s", self.A, self.n_points)

    def test_terminal_missing(self):
        G = nx.MultiGraph()
        G.add_edge("s", "t", nu=self.nu)
        with pytest.raises(ValueError, match="terminals must be in"):
            graph_from_sp(G, "s", "missing", self.A, self.n_points)

    def test_self_loop_rejected(self):
        G = nx.MultiGraph()
        G.add_edge("s", "t", nu=self.nu)
        G.add_edge("s", "s", nu=self.nu)
        with pytest.raises(ValueError, match="self-loops"):
            graph_from_sp(G, "s", "t", self.A, self.n_points)

    def test_missing_nu_attribute(self):
        G = nx.MultiGraph()
        G.add_edge("s", "t")  # no nu
        with pytest.raises(ValueError, match="missing the required 'nu'"):
            graph_from_sp(G, "s", "t", self.A, self.n_points)

    def test_disconnected_terminals(self):
        # An ISOLATED terminal hits the isolated-node refusal (its
        # lattice-sum factor diverges; label-convention unification)...
        G = nx.MultiGraph()
        G.add_node("s")
        G.add_node("t")
        G.add_edge("s", "x", nu=self.nu)  # t is isolated
        with pytest.raises(ValueError, match="isolated"):
            graph_from_sp(G, "s", "t", self.A, self.n_points)
        # ...while terminals in two genuine components keep the
        # connectivity refusal this test was written for.
        G2 = nx.MultiGraph()
        G2.add_edge("s", "x", nu=self.nu)
        G2.add_edge("t", "y", nu=self.nu)
        with pytest.raises(ValueError, match="not connected"):
            graph_from_sp(G2, "s", "t", self.A, self.n_points)

    def test_non_multigraph_rejected(self):
        """Plain Graph is not a MultiGraph and must be rejected."""
        G = nx.Graph()
        G.add_edge("s", "t", nu=self.nu)
        with pytest.raises(ValueError, match="networkx.MultiGraph"):
            graph_from_sp(G, "s", "t", self.A, self.n_points)

    def test_extra_component_silently_dropped(self):
        """Nodes in components not containing the terminals are ignored."""
        G = nx.MultiGraph()
        G.add_edge("s", "t", nu=self.nu)
        G.add_edge("x", "y", nu=self.nu)  # disconnected side-component
        out = graph_from_sp(G, "s", "t", self.A, self.n_points)
        ref = make_epstein_graph(self.nu, self.A, self.n_points)
        self._sample_close(out, ref)

    # --- uniform wrapper --------------------------------------------------

    def test_uniform_wrapper_matches_explicit(self):
        """graph_from_sp_uniform matches graph_from_sp with per-edge nu."""
        G_uniform = nx.MultiGraph()
        G_uniform.add_edge("s", "u", nu=999.0)  # should be overwritten
        G_uniform.add_edge("u", "t", nu=999.0)
        G_uniform.add_edge("s", "t", nu=999.0)

        G_explicit = nx.MultiGraph()
        G_explicit.add_edge("s", "u", nu=self.nu)
        G_explicit.add_edge("u", "t", nu=self.nu)
        G_explicit.add_edge("s", "t", nu=self.nu)

        out_uniform = graph_from_sp_uniform(
            G_uniform, "s", "t", self.nu, self.A, self.n_points
        )
        out_explicit = graph_from_sp(
            G_explicit, "s", "t", self.A, self.n_points
        )
        self._sample_close(out_uniform, out_explicit)

    def test_uniform_non_multigraph_rejected(self):
        G = nx.Graph()
        G.add_edge("s", "t")
        with pytest.raises(ValueError, match="networkx.MultiGraph"):
            graph_from_sp_uniform(
                G, "s", "t", self.nu, self.A, self.n_points
            )

    # --- 2D sanity --------------------------------------------------------

    def test_2d_triangle_plus_direct(self):
        """Same topology as test_triangle_plus_direct but on a 2D lattice."""
        A_2d = np.array([[1.0, 0.2], [0.0, 1.1]])
        nu = 2.1 + np.pi / 30
        n_points = 24

        G = nx.MultiGraph()
        G.add_edge("s", "u", nu=nu)
        G.add_edge("u", "t", nu=nu)
        G.add_edge("s", "t", nu=nu)
        out = graph_from_sp(G, "s", "t", A_2d, n_points)
        g0 = make_epstein_graph(nu, A_2d, n_points)
        ref = graph_convolve(graph_multiply(g0, g0), g0)
        self._sample_close(out, ref, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# graph_from_tw2 / graph_from_tw2_uniform
# ---------------------------------------------------------------------------

class TestGraphFromTw2:
    """Stage-2 constructor: block-cut decomposition + decorations.

    All tests compare the output of :func:`graph_from_tw2` on a
    :class:`networkx.MultiGraph` to a hand-built reference using the
    primitive operations (:func:`graph_multiply`,
    :func:`graph_convolve`, :func:`graph_attach`).
    """

    A = np.array([[1.0]])
    n_points = 64
    nu = 1.0 + np.pi / 30

    @staticmethod
    def _sample_close(g1, g2, atol=1e-10, rtol=1e-10):
        s1 = graph_sample(g1)
        s2 = graph_sample(g2)
        np.testing.assert_allclose(s1, s2, atol=atol, rtol=rtol)

    def _g0(self, nu=None):
        return make_epstein_graph(
            self.nu if nu is None else nu, self.A, self.n_points
        )

    # --- stage-1 equivalence on 2-connected SP inputs ---------------------

    def test_2connected_matches_graph_from_sp(self):
        """On 2-connected SP inputs graph_from_tw2 = graph_from_sp."""
        G = nx.MultiGraph()
        # diamond: two parallel 2-paths s-a-t and s-b-t
        G.add_edge("s", "a", nu=self.nu)
        G.add_edge("a", "t", nu=self.nu)
        G.add_edge("s", "b", nu=self.nu)
        G.add_edge("b", "t", nu=self.nu)

        out_tw2 = graph_from_tw2(G, "s", "t", self.A, self.n_points)
        out_sp = graph_from_sp(G, "s", "t", self.A, self.n_points)
        self._sample_close(out_tw2, out_sp)

    def test_theta_matches_stage1(self):
        G = nx.MultiGraph()
        for _ in range(4):
            G.add_edge("s", "t", nu=self.nu)
        out_tw2 = graph_from_tw2(G, "s", "t", self.A, self.n_points)
        out_sp = graph_from_sp(G, "s", "t", self.A, self.n_points)
        self._sample_close(out_tw2, out_sp)

    # --- single decoration at a terminal ---------------------------------

    def test_leaf_at_terminal(self):
        """s-t plus a leaf hanging off s = s-t edge scaled by zeta_g0(0)."""
        G = nx.MultiGraph()
        G.add_edge("s", "t", nu=self.nu)
        G.add_edge("s", "leaf", nu=self.nu)

        out = graph_from_tw2(G, "s", "t", self.A, self.n_points)
        g0 = self._g0()
        ref = graph_attach(g0, g0)
        self._sample_close(out, ref)

    def test_leaf_at_other_terminal(self):
        """Symmetry: leaf at t gives the same result as leaf at s."""
        G = nx.MultiGraph()
        G.add_edge("s", "t", nu=self.nu)
        G.add_edge("t", "leaf", nu=self.nu)

        out = graph_from_tw2(G, "s", "t", self.A, self.n_points)
        g0 = self._g0()
        ref = graph_attach(g0, g0)
        self._sample_close(out, ref)

    # --- decoration at a spine-internal articulation --------------------

    def test_leaf_at_middle(self):
        """s-u-t plus u-leaf = graph_multiply(g0,g0) scaled by zeta_g0(0)."""
        G = nx.MultiGraph()
        G.add_edge("s", "u", nu=self.nu)
        G.add_edge("u", "t", nu=self.nu)
        G.add_edge("u", "leaf", nu=self.nu)

        out = graph_from_tw2(G, "s", "t", self.A, self.n_points)
        g0 = self._g0()
        path = graph_multiply(g0, g0)
        ref = graph_attach(path, g0)
        self._sample_close(out, ref)

    def test_multi_edge_chain_decoration(self):
        """Decoration is a 3-edge chain hanging off s."""
        G = nx.MultiGraph()
        G.add_edge("s", "t", nu=self.nu)
        G.add_edge("s", "a", nu=self.nu)
        G.add_edge("a", "b", nu=self.nu)
        G.add_edge("b", "c", nu=self.nu)

        out = graph_from_tw2(G, "s", "t", self.A, self.n_points)
        g0 = self._g0()
        chain3 = graph_multiply_power(g0, 3)
        ref = graph_attach(g0, chain3)
        self._sample_close(out, ref)

    def test_sp_decoration_triangle(self):
        """Decoration is a triangle (2-connected SP) attached at s."""
        G = nx.MultiGraph()
        G.add_edge("s", "t", nu=self.nu)
        # triangle s-x-y-s
        G.add_edge("s", "x", nu=self.nu)
        G.add_edge("x", "y", nu=self.nu)
        G.add_edge("y", "s", nu=self.nu)

        out = graph_from_tw2(G, "s", "t", self.A, self.n_points)
        g0 = self._g0()
        # triangle as a zeta: one direct edge in parallel with 2-path
        triangle = graph_convolve(graph_multiply(g0, g0), g0)
        ref = graph_attach(g0, triangle)
        self._sample_close(out, ref)

    # --- multiple / nested decorations -----------------------------------

    def test_two_leaves_same_vertex(self):
        G = nx.MultiGraph()
        G.add_edge("s", "t", nu=self.nu)
        G.add_edge("s", "leaf1", nu=self.nu)
        G.add_edge("s", "leaf2", nu=self.nu)

        out = graph_from_tw2(G, "s", "t", self.A, self.n_points)
        g0 = self._g0()
        ref = graph_attach(graph_attach(g0, g0), g0)
        self._sample_close(out, ref)

    def test_leaves_on_both_terminals(self):
        G = nx.MultiGraph()
        G.add_edge("s", "t", nu=self.nu)
        G.add_edge("s", "leaf_s", nu=self.nu)
        G.add_edge("t", "leaf_t", nu=self.nu)

        out = graph_from_tw2(G, "s", "t", self.A, self.n_points)
        g0 = self._g0()
        ref = graph_attach(graph_attach(g0, g0), g0)
        self._sample_close(out, ref)

    # --- decorations combined with a nontrivial spine -------------------

    def test_decorated_diamond(self):
        """Diamond spine + leaves at the two internal vertices."""
        G = nx.MultiGraph()
        G.add_edge("s", "a", nu=self.nu)
        G.add_edge("a", "t", nu=self.nu)
        G.add_edge("s", "b", nu=self.nu)
        G.add_edge("b", "t", nu=self.nu)
        G.add_edge("a", "la", nu=self.nu)
        G.add_edge("b", "lb", nu=self.nu)

        out = graph_from_tw2(G, "s", "t", self.A, self.n_points)
        g0 = self._g0()
        diamond = graph_convolve(
            graph_multiply(g0, g0), graph_multiply(g0, g0)
        )
        # Both leaves contribute the same scalar
        ref = graph_attach(graph_attach(diamond, g0), g0)
        self._sample_close(out, ref)

    def test_serial_diamonds_with_midpoint_decoration(self):
        """Two diamonds joined at articulation v, with theta at v."""
        G = nx.MultiGraph()
        # Diamond 1: s-a-v, s-b-v
        G.add_edge("s", "a", nu=self.nu)
        G.add_edge("a", "v", nu=self.nu)
        G.add_edge("s", "b", nu=self.nu)
        G.add_edge("b", "v", nu=self.nu)
        # Diamond 2: v-c-t, v-d-t
        G.add_edge("v", "c", nu=self.nu)
        G.add_edge("c", "t", nu=self.nu)
        G.add_edge("v", "d", nu=self.nu)
        G.add_edge("d", "t", nu=self.nu)
        # Decoration at v: theta of 3 parallel v-w edges
        for _ in range(3):
            G.add_edge("v", "w", nu=self.nu)

        out = graph_from_tw2(G, "s", "t", self.A, self.n_points)
        g0 = self._g0()
        diamond = graph_convolve(
            graph_multiply(g0, g0), graph_multiply(g0, g0)
        )
        spine = graph_multiply(diamond, diamond)
        theta3 = graph_convolve_power(g0, 3)
        ref = graph_attach(spine, theta3)
        self._sample_close(out, ref, atol=1e-10, rtol=1e-10)

    # --- terminal that is itself an articulation point ------------------

    def test_terminal_is_articulation(self):
        """s is an articulation point (decoration branches from s directly)."""
        # s has degree 2: one edge into the main (s-t), one decoration (s-x-y)
        G = nx.MultiGraph()
        G.add_edge("s", "t", nu=self.nu)
        G.add_edge("s", "x", nu=self.nu)
        G.add_edge("x", "y", nu=self.nu)

        out = graph_from_tw2(G, "s", "t", self.A, self.n_points)
        g0 = self._g0()
        chain2 = graph_multiply(g0, g0)
        ref = graph_attach(g0, chain2)
        self._sample_close(out, ref)

    # --- rejection -------------------------------------------------------

    def test_k4_on_spine_rejected(self):
        """A K_4 block anywhere (spine or decoration) raises."""
        G = nx.MultiGraph()
        verts = ["s", "u", "v", "t"]
        for i, a in enumerate(verts):
            for b in verts[i + 1:]:
                G.add_edge(a, b, nu=self.nu)
        with pytest.raises(NotTreewidthTwoError):
            graph_from_tw2(G, "s", "t", self.A, self.n_points)

    def test_k4_in_decoration_rejected(self):
        """K_4 hanging off the spine is still treewidth > 2."""
        G = nx.MultiGraph()
        G.add_edge("s", "t", nu=self.nu)
        # K_4 on (s, p, q, r)
        verts = ["s", "p", "q", "r"]
        for i, a in enumerate(verts):
            for b in verts[i + 1:]:
                G.add_edge(a, b, nu=self.nu)
        with pytest.raises(NotTreewidthTwoError):
            graph_from_tw2(G, "s", "t", self.A, self.n_points)

    _SEED_PROBE = """
import sys
sys.path.insert(0, {root!r})
import numpy as np, networkx as nx
from gzl import graph_from_tw2_uniform, graph_zero
nu = 1.5
G = nx.MultiGraph()
spine = ["s", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "t"]
for u, v in zip(spine, spine[1:]):
    G.add_edge(u, v, nu=nu)
G.add_edge("s", "t", nu=nu)
G.add_edge("m4", "c1", nu=nu); G.add_edge("c1", "c2", nu=nu)
G.add_edge("c2", "m4", nu=nu)
G.add_edge("c1", "t1", nu=nu); G.add_edge("t1", "t2", nu=nu)
g = graph_from_tw2_uniform(G, "s", "t", nu, np.array([[1.0]]), 250)
print(float(graph_zero(g)).hex())
"""

    def test_decoration_does_not_depend_on_the_hash_seed(self):
        """The NetworkX example of DOCUMENTATION.md at nu = 1.5.

        The second terminal of a decoration used to be the first vertex
        of a string-labelled subgraph, which follows Python's randomised
        hash.  On some seeds it was the end of the pendant path, the
        decoration's spine then ran through triangle times edge, and the
        value lost three orders (4.2e-5 against 1.4e-7 at n = 500).  The
        value must be bit-identical across seeds and at background.
        """
        import os
        import subprocess
        import sys
        from pathlib import Path

        import gzl

        root = str(Path(gzl.__file__).resolve().parents[1])
        probe = self._SEED_PROBE.format(root=root)
        values = set()
        for seed in ("0", "3", "11"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            out = subprocess.run([sys.executable, "-c", probe], env=env,
                                 capture_output=True, text=True, timeout=300)
            assert out.returncode == 0, out.stderr[-2000:]
            values.add(out.stdout.strip())
        assert len(values) == 1, values
        nu = 1.5
        z = float(np.real(epstein_zeta(nu, self.A, np.zeros(1), np.zeros(1))))
        ref = (zeta_circle(np.full(9, nu), self.A).real
               * zeta_circle(np.full(3, nu), self.A).real * z ** 2)
        val = float.fromhex(values.pop())
        assert abs(val / ref - 1.0) < 1e-5          # 1.4e-6 at n = 250

    def test_induced_subgraph_keeps_the_graph_order(self):
        from gzl.construction import _induced_subgraph
        G = nx.MultiGraph()
        labels = [f"v{i}" for i in range(40)]
        for u, v in zip(labels, labels[1:]):
            G.add_edge(u, v, nu=self.nu)
        G.add_edge("v3", "v4", nu=2.0 * self.nu)          # a parallel edge
        keep = {"v9", "v3", "v5", "v4"}
        H = _induced_subgraph(G, keep)
        assert list(H) == ["v3", "v4", "v5", "v9"]
        assert list(H.edges(keys=True, data="nu")) == [
            ("v3", "v4", 0, self.nu),
            ("v3", "v4", 1, 2.0 * self.nu),
            ("v4", "v5", 0, self.nu),
        ]

    def test_tw2_error_is_subclass_of_sp_error(self):
        """NotTreewidthTwoError is a NotSeriesParallelError."""
        G = nx.MultiGraph()
        verts = ["s", "u", "v", "t"]
        for i, a in enumerate(verts):
            for b in verts[i + 1:]:
                G.add_edge(a, b, nu=self.nu)
        # `except NotSeriesParallelError` must still catch the tw-2 error
        with pytest.raises(NotSeriesParallelError):
            graph_from_tw2(G, "s", "t", self.A, self.n_points)

    # --- validation errors shared with stage 1 --------------------------

    def test_terminals_must_differ(self):
        G = nx.MultiGraph()
        G.add_edge("s", "t", nu=self.nu)
        with pytest.raises(ValueError, match="distinct"):
            graph_from_tw2(G, "s", "s", self.A, self.n_points)

    def test_self_loop_rejected(self):
        G = nx.MultiGraph()
        G.add_edge("s", "t", nu=self.nu)
        G.add_edge("s", "s", nu=self.nu)
        with pytest.raises(ValueError, match="self-loops"):
            graph_from_tw2(G, "s", "t", self.A, self.n_points)

    def test_missing_nu(self):
        G = nx.MultiGraph()
        G.add_edge("s", "t")  # no nu
        with pytest.raises(ValueError, match="missing the required 'nu'"):
            graph_from_tw2(G, "s", "t", self.A, self.n_points)

    def test_non_multigraph_rejected(self):
        G = nx.Graph()
        G.add_edge("s", "t", nu=self.nu)
        with pytest.raises(ValueError, match="networkx.MultiGraph"):
            graph_from_tw2(G, "s", "t", self.A, self.n_points)

    # --- uniform wrapper ------------------------------------------------

    def test_uniform_wrapper_matches_explicit(self):
        G_u = nx.MultiGraph()
        G_u.add_edge("s", "t")
        G_u.add_edge("s", "leaf")
        G_u.add_edge("t", "x")
        G_u.add_edge("x", "y")

        G_e = nx.MultiGraph()
        G_e.add_edge("s", "t", nu=self.nu)
        G_e.add_edge("s", "leaf", nu=self.nu)
        G_e.add_edge("t", "x", nu=self.nu)
        G_e.add_edge("x", "y", nu=self.nu)

        out_u = graph_from_tw2_uniform(
            G_u, "s", "t", self.nu, self.A, self.n_points
        )
        out_e = graph_from_tw2(G_e, "s", "t", self.A, self.n_points)
        self._sample_close(out_u, out_e)


# ---------------------------------------------------------------------------
# graph_from_edges (low-level array constructor)
# ---------------------------------------------------------------------------

class TestGraphFromEdges:
    """Array-based constructor: edges + per-edge nu vector.

    Mirrors the formal definition G = (V, E, nu, s, t) with E given as an
    indexed list (numpy array of shape (n, 2)) and nu as a length-n vector.
    Multigraphs are encoded by row repetition.
    """

    nu = 1.5 + np.pi / 30  # 1D-safe (> d = 1) and irrational
    A = np.array([[1.0]])
    n_points = 500

    # ------------------------------------------------------------------
    # parity vs the existing constructors and reference functions
    # ------------------------------------------------------------------

    def test_single_edge_matches_make_epstein_graph(self):
        edges = np.array([[0, 1]])
        nu_vec = np.array([self.nu])
        g = graph_from_edges(edges, nu_vec, self.A, self.n_points, s=0, t=1)
        g_ref = make_epstein_graph(self.nu, self.A, self.n_points)
        assert np.isclose(
            float(np.real(graph_zero(g))),
            float(np.real(graph_zero(g_ref))),
            rtol=1e-12, atol=0.0,
        )

    def test_triangle_matches_zeta_circle(self):
        edges = np.array([[0, 1], [1, 2], [0, 2]])
        nu_vec = np.full(3, self.nu)
        g = graph_from_edges(edges, nu_vec, self.A, self.n_points, s=0, t=1)
        ref = float(np.real(zeta_circle(np.full(3, self.nu), self.A)))
        assert np.isclose(float(np.real(graph_zero(g))), ref, rtol=1e-7)

    def test_four_cycle_matches_zeta_circle(self):
        edges = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])
        nu_vec = np.full(4, self.nu)
        g = graph_from_edges(edges, nu_vec, self.A, self.n_points, s=0, t=2)
        ref = float(np.real(zeta_circle(np.full(4, self.nu), self.A)))
        assert np.isclose(float(np.real(graph_zero(g))), ref, rtol=1e-7)

    def test_double_bond_matches_zeta_E_2nu(self):
        """Two parallel bonds between vertices 0 and 1, expanded by row
        repetition.  The vacuum value at k = 0 is ζ_E(2ν)."""
        from epsteinlib import epstein_zeta

        edges = np.array([[0, 1], [0, 1]])
        nu_vec = np.full(2, self.nu)
        g = graph_from_edges(edges, nu_vec, self.A, self.n_points, s=0, t=1)
        ref = float(
            epstein_zeta(2.0 * self.nu, self.A,
                         np.zeros(1), np.zeros(1)).real
        )
        assert np.isclose(
            float(np.real(graph_zero(g))), ref, rtol=1e-12, atol=0.0
        )

    def test_path_two_edges_matches_graph_multiply(self):
        """Path 0—1—2 with two equal-ν edges should match the algebraic
        composition graph_multiply(g0, g0)."""
        edges = np.array([[0, 1], [1, 2]])
        nu_vec = np.full(2, self.nu)
        g_path = graph_from_edges(
            edges, nu_vec, self.A, self.n_points, s=0, t=2
        )
        g0 = make_epstein_graph(self.nu, self.A, self.n_points)
        g_ref = graph_multiply(g0, g0)
        assert np.isclose(
            float(np.real(graph_zero(g_path))),
            float(np.real(graph_zero(g_ref))),
            rtol=1e-12, atol=0.0,
        )

    # ------------------------------------------------------------------
    # vacuum / zero-momentum (s == t)
    # ------------------------------------------------------------------

    def test_vacuum_default_terminals_match_explicit(self):
        """At k = 0 the vacuum value is independent of the terminal pair,
        so s = t = 0 (default) must agree with any explicit s != t."""
        edges = np.array([[0, 1], [1, 2], [0, 2]])
        nu_vec = np.full(3, self.nu)
        g_vac = graph_from_edges(edges, nu_vec, self.A, self.n_points)
        g_ex = graph_from_edges(
            edges, nu_vec, self.A, self.n_points, s=0, t=2
        )
        assert np.isclose(
            float(np.real(graph_zero(g_vac))),
            float(np.real(graph_zero(g_ex))),
            rtol=1e-12, atol=0.0,
        )

    def test_vacuum_matches_zeta_circle_3(self):
        edges = np.array([[0, 1], [1, 2], [0, 2]])
        nu_vec = np.full(3, self.nu)
        g = graph_from_edges(edges, nu_vec, self.A, self.n_points)  # s=t=0
        ref = float(np.real(zeta_circle(np.full(3, self.nu), self.A)))
        assert np.isclose(float(np.real(graph_zero(g))), ref, rtol=1e-7)

    def test_vacuum_picks_smallest_aux_when_s_nonzero(self):
        """s = t = 2 must still produce a valid GraphZeta whose value at
        k = 0 equals the vacuum sum."""
        edges = np.array([[0, 1], [1, 2], [0, 2]])
        nu_vec = np.full(3, self.nu)
        g = graph_from_edges(
            edges, nu_vec, self.A, self.n_points, s=2, t=2
        )
        ref = float(np.real(zeta_circle(np.full(3, self.nu), self.A)))
        assert np.isclose(float(np.real(graph_zero(g))), ref, rtol=1e-7)

    # ------------------------------------------------------------------
    # parity vs graph_from_tw2
    # ------------------------------------------------------------------

    def test_matches_graph_from_tw2_on_4cycle(self):
        edges = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])
        nu_vec = np.full(4, self.nu)

        g_arr = graph_from_edges(
            edges, nu_vec, self.A, self.n_points, s=0, t=2
        )

        G = nx.MultiGraph()
        G.add_edge(0, 1, nu=self.nu)
        G.add_edge(1, 2, nu=self.nu)
        G.add_edge(2, 3, nu=self.nu)
        G.add_edge(0, 3, nu=self.nu)
        g_mg = graph_from_tw2(G, 0, 2, self.A, self.n_points)

        s_arr = graph_sample(g_arr)
        s_mg = graph_sample(g_mg)
        denom = max(float(np.max(np.abs(s_mg))), 1e-30)
        assert float(np.max(np.abs(s_arr - s_mg))) < 1e-10 * denom

    # ------------------------------------------------------------------
    # uniform wrapper
    # ------------------------------------------------------------------

    def test_uniform_wrapper_matches_explicit_per_edge(self):
        edges = np.array([[0, 1], [1, 2], [0, 2]])
        g_u = graph_from_edges_uniform(edges, self.nu, self.A, self.n_points)
        g_e = graph_from_edges(
            edges, np.full(3, self.nu), self.A, self.n_points
        )
        assert np.isclose(
            float(np.real(graph_zero(g_u))),
            float(np.real(graph_zero(g_e))),
            rtol=1e-12, atol=0.0,
        )

    # ------------------------------------------------------------------
    # input validation
    # ------------------------------------------------------------------

    def test_bad_edge_shape_rejected(self):
        with pytest.raises(ValueError, match="shape"):
            graph_from_edges(
                np.array([0, 1]), np.array([self.nu]),
                self.A, self.n_points,
            )

    def test_empty_edge_list_rejected(self):
        with pytest.raises(ValueError, match="at least one edge"):
            graph_from_edges(
                np.zeros((0, 2), dtype=int), np.zeros(0),
                self.A, self.n_points,
            )

    def test_nu_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="nu must have shape"):
            graph_from_edges(
                np.array([[0, 1], [1, 2]]),
                np.array([self.nu]),
                self.A, self.n_points,
            )

    def test_self_loop_rejected(self):
        with pytest.raises(ValueError, match="self-loop"):
            graph_from_edges(
                np.array([[0, 1], [1, 1]]),
                np.full(2, self.nu),
                self.A, self.n_points, s=0, t=1,
            )

    def test_terminal_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="not a vertex"):
            graph_from_edges(
                np.array([[0, 1]]), np.array([self.nu]),
                self.A, self.n_points, s=0, t=5,
            )

    def test_nu_at_or_below_d_rejected(self):
        """Re(nu_e) > d is required for convergence."""
        with pytest.raises(ValueError, match="strictly greater"):
            graph_from_edges(
                np.array([[0, 1]]), np.array([1.0]),  # nu = d = 1
                self.A, self.n_points,
            )

    def test_negative_vertex_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            graph_from_edges(
                np.array([[-1, 0]]), np.array([self.nu]),
                self.A, self.n_points,
            )

    def test_uniform_wrapper_validates_shape(self):
        with pytest.raises(ValueError, match="shape"):
            graph_from_edges_uniform(
                np.array([0, 1]), self.nu, self.A, self.n_points,
            )

    # ------------------------------------------------------------------
    # tw-2 boundary: K_4 minor must raise NotTreewidthTwoError
    # ------------------------------------------------------------------

    def test_K4_raises_not_treewidth_two(self):
        # K_4 = complete graph on 4 vertices, 6 edges.
        edges = np.array([
            [0, 1], [0, 2], [0, 3],
            [1, 2], [1, 3], [2, 3],
        ])
        nu_vec = np.full(6, self.nu)
        with pytest.raises(NotTreewidthTwoError):
            graph_from_edges(
                edges, nu_vec, self.A, self.n_points, s=0, t=1
            )


# ---------------------------------------------------------------------------
# direct_sum_zero_momentum: treewidth-agnostic truncated lattice summation
# ---------------------------------------------------------------------------

class TestDirectSum:
    """Direct truncated lattice sum via variable elimination.

    Two roles:

    * **Cross-check on tw <= 2 graphs**: Richardson-extrapolated direct
      sum should agree with `zeta_circle` (closed-form) and with
      `graph_from_edges_uniform` (FFT-based) to truncation precision.
      Catches bugs in *either* method.
    * **Reference for tw > 2 graphs**: the K_4 and K_5 ladders at
      d = 1 are pinned against frozen values in test_marker_peel.py.
    """

    A = np.array([[1.0]])
    nu = 2.5                       # well above d=1, away from poles
    L_list = (5, 6, 7, 8, 9)

    # ------------------------------------------------------------------
    # Validation against closed-form / library references on tw <= 2 graphs
    # ------------------------------------------------------------------

    def test_single_edge_matches_epstein_zeta(self):
        from epsteinlib import epstein_zeta

        edges  = np.array([[0, 1]])
        nu_vec = np.array([self.nu])
        ref = float(epstein_zeta(self.nu, self.A,
                                  np.zeros(1), np.zeros(1)).real)

        v = direct_sum_extrapolated(edges, nu_vec, self.A,
                                     self.L_list).real
        # K=3 integer-shifted Richardson empirically reaches 1e-6 at this nu
        assert np.isclose(v, ref, rtol=1e-5, atol=0.0), (
            f"computed={v:.12e}, ref={ref:.12e}, "
            f"rel.err={abs(v-ref)/abs(ref):.3e}"
        )

    def test_double_bond_matches_epstein_zeta_2nu(self):
        """V=2 with two parallel edges: sum = zeta_E(2*nu) (multi-edge
        collapse multiplies effective exponent)."""
        from epsteinlib import epstein_zeta

        edges  = np.array([[0, 1], [0, 1]])
        nu_vec = np.full(2, self.nu)
        ref = float(epstein_zeta(2 * self.nu, self.A,
                                  np.zeros(1), np.zeros(1)).real)

        v = direct_sum_extrapolated(edges, nu_vec, self.A,
                                     self.L_list).real
        # 2nu_eff = 5: K=3 Richardson is essentially fp-precision here.
        assert np.isclose(v, ref, rtol=1e-7, atol=0.0)

    @pytest.mark.parametrize("n_nodes", [3, 4, 5, 6])
    def test_cycle_matches_zeta_circle(self, n_nodes):
        edges = np.array([(i, (i + 1) % n_nodes) for i in range(n_nodes)])
        nu_vec = np.full(n_nodes, self.nu)
        ref = float(zeta_circle(np.full(n_nodes, self.nu), self.A).real)

        v = direct_sum_extrapolated(edges, nu_vec, self.A,
                                     self.L_list).real
        # K=3 Richardson reaches ~1e-4 for small cycles, degrading to
        # ~1e-3 by n_nodes=6 (more vertices => more boundary configurations
        # contributing to the leading-order constants).
        assert np.isclose(v, ref, rtol=5e-3, atol=0.0), (
            f"n_nodes={n_nodes}: computed={v:.12e}, ref={ref:.12e}"
        )

    def test_lollipop_matches_graph_from_edges(self):
        """V=4 lollipop: triangle (0,1,2) + tail edge (2,3).  Both
        SP-reduction and direct sum should land on the same value
        (within their respective error bars)."""
        edges = np.array([[0, 1], [1, 2], [0, 2], [2, 3]])
        nu_vec = np.full(4, self.nu)

        v_direct = direct_sum_extrapolated(
            edges, nu_vec, self.A, self.L_list
        ).real
        # graph_from_edges_uniform: FFT-based, fp-precision at this nu
        g = graph_from_edges_uniform(
            edges, self.nu, self.A, n_points=500, sigma_max=4.0
        )
        v_sp = float(graph_zero(g).real)

        # Direct-sum K=3 Richardson is ~1e-3 here (V=4 has lots of
        # boundary configurations contributing); SP is fp-precision.
        assert np.isclose(v_direct, v_sp, rtol=1e-3, atol=0.0), (
            f"direct={v_direct:.12e}, sp={v_sp:.12e}, "
            f"rel.diff={abs(v_direct-v_sp)/abs(v_sp):.3e}"
        )

    # ------------------------------------------------------------------
    # Single-shot direct_sum_zero_momentum: convergence with L
    # ------------------------------------------------------------------

    def test_single_shot_converges_with_L(self):
        """For a 3-cycle, the relative error vs zeta_circle should
        decrease at least like L^(d - nu) ~ L^{-1.5} as L grows."""
        edges = np.array([[0, 1], [1, 2], [0, 2]])
        nu_vec = np.full(3, self.nu)
        ref = float(zeta_circle(np.full(3, self.nu), self.A).real)

        errs = []
        for L in (5, 10, 20):
            s = direct_sum_zero_momentum(edges, nu_vec, self.A, L).real
            errs.append(abs(s - ref) / abs(ref))

        # Each doubling of L should reduce the error by at least ~2x
        # (theoretical factor 2^1.5 = 2.83 for nu=2.5)
        assert errs[1] < errs[0] / 2.0, (
            f"L=5 -> L=10: errs {errs[0]:.3e} -> {errs[1]:.3e}"
        )
        assert errs[2] < errs[1] / 2.0, (
            f"L=10 -> L=20: errs {errs[1]:.3e} -> {errs[2]:.3e}"
        )

    # ------------------------------------------------------------------
    # Origin-choice independence (translational invariance)
    # ------------------------------------------------------------------

    def test_root_choice_independent_at_extrapolation(self):
        """Translational invariance ⇒ the answer doesn't depend on
        which vertex is pinned at the origin (only the truncation-error
        constants do).  Extrapolation should bring different root
        choices into agreement."""
        # Triangle: all vertices equivalent, so this is trivially
        # exact at every L.  Use a lollipop instead — vertex 0 has
        # degree 2 and vertex 3 has degree 1, so different root choices
        # give different intermediate tensors.
        edges = np.array([[0, 1], [1, 2], [0, 2], [2, 3]])
        nu_vec = np.full(4, self.nu)

        v_default = direct_sum_extrapolated(
            edges, nu_vec, self.A, self.L_list
        ).real
        # Force pin at the leaf vertex (degree-1 vertex 3).
        v_leaf = direct_sum_extrapolated(
            edges, nu_vec, self.A, self.L_list, root=3
        ).real

        # Both extrapolations should agree to truncation precision.
        # The leaf-pinned version converges slower, so tolerance is
        # generous.
        assert np.isclose(v_default, v_leaf, rtol=5e-2, atol=0.0), (
            f"default root: {v_default:.12e}, leaf root: {v_leaf:.12e}"
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_self_loop_rejected(self):
        with pytest.raises(ValueError, match="self-loop"):
            direct_sum_zero_momentum(
                np.array([[0, 1], [1, 1]]),
                np.full(2, self.nu),
                self.A, L=5,
            )

    def test_extrapolation_refuses_below_d(self):
        with pytest.raises(UnsupportedLatticeSumError, match="not supported"):
            direct_sum_extrapolated(
                np.array([[0, 1]]),
                np.array([0.5]),       # nu = 0.5 < d = 1
                self.A,
                self.L_list,
            )

    def test_nu_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="nu must"):
            direct_sum_zero_momentum(
                np.array([[0, 1], [1, 2]]),
                np.array([self.nu]),    # wrong length
                self.A, L=5,
            )

    # ------------------------------------------------------------------
    # -Λ = Λ symmetry (chain Z_2): use_symmetry=True must give identical
    # results to use_symmetry=False
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("L", [5, 7, 10])
    def test_symmetry_parity_triangle(self, L):
        edges = np.array([[0, 1], [1, 2], [0, 2]])
        nu_vec = np.full(3, self.nu)
        v_sym = direct_sum_zero_momentum(edges, nu_vec, self.A, L,
                                          use_symmetry=True).real
        v_no  = direct_sum_zero_momentum(edges, nu_vec, self.A, L,
                                          use_symmetry=False).real
        assert np.isclose(v_sym, v_no, rtol=1e-12, atol=0.0), (
            f"L={L}: sym={v_sym:.15e}, no_sym={v_no:.15e}"
        )

    @pytest.mark.parametrize("L", [5, 7, 10])
    def test_symmetry_parity_K4(self, L):
        """K_4 (tw=3) — multiple variables in intermediate tensors;
        confirms the marker-axis indexing is consistent through all
        elimination steps."""
        edges = np.array([
            [0, 1], [0, 2], [0, 3],
            [1, 2], [1, 3], [2, 3],
        ])
        nu_vec = np.full(6, self.nu)
        v_sym = direct_sum_zero_momentum(edges, nu_vec, self.A, L,
                                          use_symmetry=True).real
        v_no  = direct_sum_zero_momentum(edges, nu_vec, self.A, L,
                                          use_symmetry=False).real
        assert np.isclose(v_sym, v_no, rtol=1e-12, atol=0.0)

    def test_symmetry_matches_on_lollipop(self):
        """The Z₂-marker path gives the same value as the unsymmetric
        path on a lollipop, a triangle with a pendant edge."""
        edges = np.array([[0, 1], [1, 2], [0, 2], [2, 3]])
        nu_vec = np.full(4, self.nu)
        v_sym = direct_sum_zero_momentum(edges, nu_vec, self.A, 30,
                                          use_symmetry=True).real
        v_no = direct_sum_zero_momentum(edges, nu_vec, self.A, 30,
                                         use_symmetry=False).real
        assert np.isclose(v_sym, v_no, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# Epstein-grid memoisation in graph_sample (pure cache; must be exact)
# ---------------------------------------------------------------------------

class TestEpsteinGridCache:
    """The lru_cache in graph_sample is pure memoisation — it must not
    change any value, must survive a cache clear, and must protect its
    cached arrays from mutation."""

    A = np.array([[1.0]])

    def _block(self, nu):
        edges = np.array([[0, 1], [1, 2], [0, 2]])      # triangle: tw=2, σ-routed
        return graph_from_edges_uniform(
            edges, nu, self.A, 24, s=0, t=1, sigma_max=4.0,
        )

    def test_cache_clear_gives_identical_result(self):
        from gzl.core import _epstein_grid_cached
        g = self._block(2.5)
        first = graph_sample(g)
        _epstein_grid_cached.cache_clear()              # force recompute
        second = graph_sample(g)
        assert np.array_equal(first, second)

    def test_repeated_calls_identical(self):
        g = self._block(1.7)
        a = graph_sample(g)
        b = graph_sample(g)                              # cache hit
        assert np.array_equal(a, b)

    def test_cached_array_is_protected(self):
        from gzl.core import _epstein_grid_cached
        _epstein_grid_cached.cache_clear()
        graph_sample(self._block(3.1))
        # the cached grids must be read-only so callers cannot corrupt them
        a = _epstein_grid_cached(3.1, self.A.tobytes(), self.A.shape, 24)
        assert a.flags.writeable is False
