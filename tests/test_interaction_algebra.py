# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""The semi-analytic algebra route with general interaction kernels:
``graph_from_edges(..., kernels=)`` and the ``kern`` edge attribute of
:func:`graph_from_tw2` / :func:`graph_from_sp`.

The algebra is exact only up to its ``n_points`` window — a series
collapse is a cyclic convolution on the balanced grid, a parallel merge
a pointwise product there — so every oracle here is an identity that
holds at FIXED ``n``:

* bit-identity with the legacy power-law path (values pinned before the
  change, objects compared field by field);
* multilinearity in each edge kernel, at every ``sigma_max``, including
  exponents on the Gamma-pole family ``nu = d + 2, d + 4``;
* brute-force enumeration for purely compact kernels once the composed
  support fits the window, and the two-path identity
  ``zeta = (sum_x V(x))^2`` for a mixed kernel;
* the exact Hadamard product of parallel edges (one product edge ==
  parallel factor edges, bit-exactly);
* the window guard, and what it does NOT cover (composition aliasing);
* the conditioning ratio on the kernel path: a table's own ℓ¹ magnitude
  is tracked through the algebra so ``graph_zero_conditioned`` sees a
  cancelling table, while the power-law kappa is pinned unchanged;
* the Interaction-in-``nu`` guard on every array form of ``nu``.

Tolerances are relative to the sum of term magnitudes, never to a value
that could cancel, and every equality has an anti-vacuity control.
"""
from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import pytest
from scipy.special import zeta as riemann_zeta

from gzl import (
    GraphZetaError,
    Interaction,
    InteractionSupportError,
    UnsupportedLatticeSumError,
    graph_from_edges,
    graph_from_edges_uniform,
    graph_from_sp,
    graph_from_tw2,
    graph_zero,
)
from gzl.core import graph_attach, graph_zero_conditioned, make_epstein_graph
from gzl.interaction import _KernelProduct

A_CHAIN = np.array([[1.0]])
A_SHEARED = np.array([[1.0, 0.0], [0.3, 1.1]])

TRIANGLE = [[0, 1], [1, 2], [0, 2]]
THETA = [[0, 1], [0, 1], [0, 1]]          # three parallel edges: graph_convolve twice
C4 = [[0, 1], [1, 2], [2, 3], [3, 0]]
GRAPHS = [
    pytest.param(TRIANGLE, id="triangle"),
    pytest.param(THETA, id="theta"),
    pytest.param(C4, id="C4"),
]
LATTICES = [
    pytest.param(1, A_CHAIN, 21, id="d1-n21"),
    pytest.param(2, A_SHEARED, 13, id="d2-n13"),
]
SIGMA_MAXES = [0.5, 2.0, 4.0, 8.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cross_table(d: int, val: float, origin: float = 0.0) -> dict:
    """``val`` on the 2d axis neighbours, ``origin`` at m = 0."""
    tab = {}
    for i in range(d):
        for s in (-1, 1):
            m = [0] * d
            m[i] = s
            tab[tuple(m)] = val
    if origin != 0.0:
        tab[(0,) * d] = origin
    return tab


def radial_table_1d(R: int, c: float, origin: float, decay: float = 0.85) -> dict:
    """An even 1-D table ``a(m) = c * decay^|m|`` for ``0 < |m| <= R``,
    ``a(0) = origin`` (non-zero: coincident endpoints are weighted)."""
    tab = {(m,): c * decay ** abs(m) for m in range(-R, R + 1) if m != 0}
    tab[(0,)] = origin
    return tab


def zeta0(edges, kernels, A, n, sigma_max=4.0, s=0, t=0) -> float:
    """``graph_zero`` of the algebra object with per-edge kernels; ``nu``
    carries the tails, as the router will pass them."""
    nu = [k.tail_exponent for k in kernels]
    return graph_zero(graph_from_edges(edges, nu, A, n, s=s, t=t,
                                       sigma_max=sigma_max, kernels=kernels))


def build(edges, kernels, A, n, s=0, t=1, sigma_max=4.0):
    nu = [k.tail_exponent for k in kernels]
    return graph_from_edges(edges, nu, A, n, s=s, t=t, sigma_max=sigma_max,
                            kernels=kernels)


def same_object(g1, g2) -> bool:
    """Field-by-field ``array_equal`` (a bit-identity of the algebra
    object, stronger than equal values)."""
    return (
        g1.aMat.shape == g2.aMat.shape
        and g1.aMat.dtype == g2.aMat.dtype
        and np.array_equal(g1.aMat, g2.aMat)
        and np.array_equal(g1.bVec, g2.bVec)
        and np.array_equal(g1.nuVec, g2.nuVec)
        and np.array_equal(g1.A, g2.A)
    )


def ulp_distance(a: float, b: float) -> float:
    return abs(a - b) / np.spacing(max(abs(a), abs(b)))


def brute_force_compact(edges, tables, d: int):
    """Exact ``zeta_G(0)`` for a graph whose every edge carries a finite
    even table ``a_e(m)`` (zero off the table): pin vertex 0 and
    enumerate every free vertex over the Chebyshev box its supports
    allow (``|x_v|_inf <= R``-weighted graph distance from the pin).
    Returns ``(value, scale)`` with ``scale = sum |terms|``."""
    edges = [tuple(e) for e in edges]
    n_v = 1 + max(max(e) for e in edges)
    R = [max((max(abs(x) for x in m) for m in tab), default=0) for tab in tables]
    dist = [math.inf] * n_v
    dist[0] = 0
    for _ in range(n_v):
        for (u, v), r in zip(edges, R):
            dist[v] = min(dist[v], dist[u] + r)
            dist[u] = min(dist[u], dist[v] + r)
    ranges = [itertools.product(range(-int(dist[v]), int(dist[v]) + 1), repeat=d)
              for v in range(1, n_v)]
    terms = []
    for config in itertools.product(*ranges):
        x = [(0,) * d] + [tuple(c) for c in config]
        term = 1.0
        for (u, v), tab in zip(edges, tables):
            w = tab.get(tuple(x[v][i] - x[u][i] for i in range(d)), 0.0)
            if w == 0.0:
                term = 0.0
                break
            term *= w
        if term != 0.0:
            terms.append(term)
    return math.fsum(terms), math.fsum(abs(t) for t in terms)


def mixed_kernel(d: int, off1: float, off2: float) -> Interaction:
    """``0.7 K_{d+off1} - 0.4 K_{d+off2} + a``, ``a`` the radius-1 cross
    table with ``a(0) = 0.3``."""
    return Interaction.from_table(cross_table(d, 0.25, origin=0.3),
                                  b=[0.7, -0.4], nu=[d + off1, d + off2])


# ---------------------------------------------------------------------------
# (1) Legacy identity
# ---------------------------------------------------------------------------

#: graph_zero(graph_from_edges(edges, full(E, d + 1.6), A, n)) computed
#: BEFORE the kernels= argument existed (macOS/arm64).  Exact on that
#: toolchain.  Elsewhere the pins move with the np.fft build: measured
#: on Linux CI (2026-09-04) the triangle values sit 96-224 ULP from
#: these pins, and the py3.11 and py3.12 runners differ from EACH OTHER
#: by 128 ULP -- a 13-/21-point transform spreads wider than the 64-ULP
#: bound tests/test_executor_goldens.py calibrated on its own grids.
#: The bitwise legacy gate is tests/test_engine_reference.py on its
#: freezing toolchain; this test keeps a 512-ULP band (2x the measured
#: spread, below the 848-ULP smallest real defect the repo has
#: measured) so that a moved legacy number is still refused on every
#: platform.
LEGACY_ULP_BAND = 512.0
LEGACY_PINS = {
    ("triangle", 1): "0x1.2342858ec5640p+0",
    ("triangle", 2): "0x1.64e02811a36e0p+2",
    ("theta", 1): "0x1.01342498fbf68p+1",
    ("theta", 2): "0x1.0ef9812bb5882p+1",
}


class TestLegacyIdentity:
    @pytest.mark.parametrize("name, edges", [("triangle", TRIANGLE), ("theta", THETA)])
    @pytest.mark.parametrize("d, A, n", LATTICES)
    def test_legacy_values_pinned_before_the_change(self, name, edges, d, A, n):
        nu = np.full(len(edges), d + 1.6)
        v = graph_zero(graph_from_edges(edges, nu, A, n))
        v_none = graph_zero(graph_from_edges(edges, nu, A, n, kernels=None))
        assert v == v_none, "kernels=None must be the omitted-argument path"
        pin = float.fromhex(LEGACY_PINS[(name, d)])
        assert ulp_distance(v, pin) <= LEGACY_ULP_BAND, (
            f"legacy value moved: {v.hex()} vs pinned {pin.hex()} "
            f"({ulp_distance(v, pin):.1f} ULP)"
        )

    @pytest.mark.parametrize("edges", GRAPHS)
    @pytest.mark.parametrize("d, A, n", LATTICES)
    def test_plain_power_law_kernels_build_the_same_object(self, edges, d, A, n):
        nu = [d + 1.6, d + 2.3, d + 1.9, d + 2.7][: len(edges)]
        g_legacy = graph_from_edges(edges, nu, A, n, s=0, t=1)
        g_kernel = graph_from_edges(edges, nu, A, n, s=0, t=1,
                                    kernels=[Interaction.power_law(x) for x in nu])
        assert same_object(g_legacy, g_kernel)
        assert graph_zero(g_legacy) == graph_zero(g_kernel)
        # Anti-vacuity: a genuinely different kernel on edge 0 changes the
        # object and the value (b = 2 doubles the leaf's bVec).
        other = [Interaction.power_law(nu[0], b=2.0)] + [Interaction.power_law(x) for x in nu[1:]]
        g_other = graph_from_edges(edges, nu, A, n, s=0, t=1, kernels=other)
        assert not same_object(g_legacy, g_other)
        assert graph_zero(g_other) != graph_zero(g_legacy)

    @pytest.mark.parametrize("d, A, n", LATTICES)
    def test_power_law_leaf_is_make_epstein_graph(self, d, A, n):
        nu = d + 1.6
        leaf = graph_from_edges([[0, 1]], [nu], A, n, s=0, t=1,
                                kernels=[Interaction.power_law(nu)])
        ref = make_epstein_graph(nu, A, n)
        assert same_object(leaf, ref)
        assert leaf.aMat.dtype == np.complex128 and leaf.nuVec.dtype == np.float64


# ---------------------------------------------------------------------------
# The leaf: where the table lands
# ---------------------------------------------------------------------------

class TestLeaf:
    @pytest.mark.parametrize("d, A, n", LATTICES + [pytest.param(1, A_CHAIN, 6, id="d1-n6-even")])
    def test_compact_leaf_scatters_the_table_on_the_balanced_grid(self, d, A, n):
        table = cross_table(d, 0.25, origin=0.3)
        if d == 1:
            table[(2,)] = table[(-2,)] = 0.125         # radius 2: even n = 6 holds +-2
        V = Interaction.from_table(table)
        leaf = build([[0, 1]], [V], A, n)
        assert leaf.bVec.size == 0 and leaf.nuVec.size == 0
        assert leaf.aMat.dtype == np.complex128 and leaf.aMat.shape == (n,) * d
        # The balanced layout is the FFT layout: label m sits at index m mod n.
        expected = np.zeros((n,) * d, dtype=complex)
        for m, val in table.items():
            expected[tuple(x % n for x in m)] = val
        assert np.array_equal(leaf.aMat, expected)
        # graph_zero of a bridge is sum_m a(m) exactly (an inverse FFT of
        # a few reals; 1e-15 is roundoff, the table sums to ~1).
        total = math.fsum(table.values())
        assert abs(graph_zero(leaf) - total) <= 1e-15 * math.fsum(abs(v) for v in table.values())

    def test_mixed_leaf_fields(self):
        V = mixed_kernel(1, 1.6, 3.2)
        leaf = build([[0, 1]], [V], A_CHAIN, 21)
        assert np.array_equal(leaf.bVec, np.asarray(V.b, dtype=complex))
        assert np.array_equal(leaf.nuVec, np.asarray(V.nu, dtype=float))
        assert leaf.aMat[0] == 0.3 and leaf.aMat[1] == 0.25 and leaf.aMat[20] == 0.25
        assert np.count_nonzero(leaf.aMat) == 3
        # Value: the bridge closed form (table sum + Epstein terms), which
        # is what Interaction.lattice_sum computes independently.
        ref = V.lattice_sum(A_CHAIN)
        assert abs(graph_zero(leaf) - ref) <= 1e-14 * abs(ref)


# ---------------------------------------------------------------------------
# (2) Multilinearity in each edge kernel at fixed n (and every sigma_max)
# ---------------------------------------------------------------------------

def _multilinearity_terms(edges, d, A, n, off1, off2, sigma_max):
    """``lhs`` and the three single-kernel terms of edge 0, the other
    edges carrying ``K_{d+1.6}``.  Returns ``(lhs, t1, t2, ta, scale)``."""
    nu1, nu2 = d + off1, d + off2
    table = cross_table(d, 0.25, origin=0.3)
    mixed = Interaction.from_table(table, b=[0.7, -0.4], nu=[nu1, nu2])
    others = [Interaction.power_law(d + 1.6)] * (len(edges) - 1)
    lhs = zeta0(edges, [mixed] + others, A, n, sigma_max)
    t1 = zeta0(edges, [Interaction.power_law(nu1)] + others, A, n, sigma_max)
    t2 = zeta0(edges, [Interaction.power_law(nu2)] + others, A, n, sigma_max)
    ta = zeta0(edges, [Interaction.from_table(table)] + others, A, n, sigma_max)
    scale = 0.7 * abs(t1) + 0.4 * abs(t2) + abs(ta)
    return lhs, t1, t2, ta, scale


def _assert_multilinear(lhs, t1, t2, ta, scale):
    assert abs(lhs - (0.7 * t1 - 0.4 * t2 + ta)) <= 1e-13 * scale
    # Anti-vacuity: every term carries weight (none is a rounding-level
    # bystander; measured >= 0.17 of scale), and a wrong coefficient is
    # rejected by orders of magnitude (measured ~3e-2 of scale).
    assert min(0.7 * abs(t1), 0.4 * abs(t2), abs(ta)) >= 0.1 * scale
    assert abs(lhs - (0.75 * t1 - 0.4 * t2 + ta)) >= 1e-3 * scale


class TestMultilinearity:
    @pytest.mark.parametrize("off1, off2", [(1.6, 3.2), (2.0, 4.0)],
                             ids=["generic", "gamma-pole-family"])
    @pytest.mark.parametrize("edges", GRAPHS)
    @pytest.mark.parametrize("d, A, n", LATTICES)
    def test_edge_kernel_is_multilinear(self, edges, d, A, n, off1, off2):
        # (2.0, 4.0) puts both exponents on the Gamma-pole family
        # nu = d + 2k, which graph_multiply pre-absorbs into aMat
        # (compress_singularities); the theta never multiplies, so there
        # the pole family stays analytic through graph_convolve.
        _assert_multilinear(*_multilinearity_terms(edges, d, A, n, off1, off2, 4.0))

    @pytest.mark.parametrize("sigma_max", SIGMA_MAXES)
    def test_multilinear_at_every_sigma_max(self, sigma_max):
        # The analytic/Fourier split moves the power-law terms between
        # (bVec, nuVec) and aMat; the compact table's coupling into either
        # representation must stay exact.
        _assert_multilinear(*_multilinearity_terms(TRIANGLE, 1, A_CHAIN, 21, 1.6, 3.2, sigma_max))

    def test_purely_compact_edge_has_no_analytic_part(self):
        # The `ta` term above runs with an empty bVec on edge 0 (tail +inf);
        # make that explicit on the built object of the theta, whose two
        # convolutions keep the other edges' exponents analytic.
        table = cross_table(1, 0.25, origin=0.3)
        g = build(THETA, [Interaction.from_table(table)] + [Interaction.power_law(2.6)] * 2, A_CHAIN, 21)
        assert g.nuVec.size == 0 and g.bVec.size == 0
        # sum_x a(x) K_2.6(x)^2 = 2 * 0.25 * 1^(-5.2) (the origin has K = 0)
        assert abs(graph_zero(g) - 0.5) <= 1e-14


# ---------------------------------------------------------------------------
# (3) Absolute correctness
# ---------------------------------------------------------------------------

class TestAbsolute:
    TABLES_R7 = [radial_table_1d(7, c, o) for c, o in ((0.9, 0.3), (0.7, -0.2), (1.1, 0.5))]

    @pytest.mark.parametrize("n, fits", [(21, False), (41, True), (81, True)])
    def test_compact_triangle_vs_brute_force(self, n, fits):
        """d = 1, three radius-7 tables with a(0) != 0.  The series collapse
        of the triangle is a cyclic convolution with composed support 2R =
        14; at n = 21 <= 3R its aliases land inside the third edge's
        support (label 14 -> -7), so the value is a window artefact; from
        n = 3R + 1 on the composition is exact and n-independent."""
        kernels = [Interaction.from_table(t) for t in self.TABLES_R7]
        ref, scale = brute_force_compact(TRIANGLE, self.TABLES_R7, 1)
        if not fits:
            # The composed support 3 R = 21 does not fit an n = 21 window:
            # the block-level guard refuses it (the wrap was a real 2e-3
            # effect, measured before the guard existed).
            with pytest.raises(InteractionSupportError, match="compose"):
                zeta0(TRIANGLE, kernels, A_CHAIN, n)
            return
        v = zeta0(TRIANGLE, kernels, A_CHAIN, n)
        err = abs(v - ref) / scale
        assert err <= 1e-13, f"n={n}: {err:.3e}"

    def test_compact_triangle_is_window_independent_once_it_fits(self):
        kernels = [Interaction.from_table(t) for t in self.TABLES_R7]
        v41 = zeta0(TRIANGLE, kernels, A_CHAIN, 41)
        v81 = zeta0(TRIANGLE, kernels, A_CHAIN, 81)
        _, scale = brute_force_compact(TRIANGLE, self.TABLES_R7, 1)
        assert abs(v41 - v81) <= 1e-13 * scale

    @pytest.mark.parametrize("n", [5, 13])
    def test_compact_triangle_d2_vs_brute_force(self, n):
        tables = [cross_table(2, 0.4, 0.3), cross_table(2, 0.6, -0.2), cross_table(2, 0.5, 0.7)]
        ref, scale = brute_force_compact(TRIANGLE, tables, 2)
        v = zeta0(TRIANGLE, [Interaction.from_table(t) for t in tables], A_SHEARED, n)
        assert abs(v - ref) <= 1e-13 * scale
        # Anti-vacuity: the enumeration carries cancelling terms (the sum
        # of magnitudes exceeds the magnitude of the sum; measured 1.49x).
        assert scale > abs(ref)

    @pytest.mark.parametrize("d, A, n", LATTICES)
    def test_two_path_identity_with_the_middle_vertex_pinned(self, d, A, n):
        """``zeta_{s-m-t}(0) = (sum_x V(x))^2``.  With the middle vertex as
        the pin (vertex 0), the two edges are two bridges of the block-cut
        tree: a leaf each, multiplied as scalars — exact at any n."""
        V = mixed_kernel(d, 1.6, 3.2)
        ref = V.lattice_sum(A) ** 2
        v = zeta0([[1, 0], [0, 2]], [V, V], A, n)
        assert abs(v - ref) <= 1e-14 * abs(ref)

    @pytest.mark.parametrize("d, A, ns", [(1, A_CHAIN, (21, 41, 81)), (2, A_SHEARED, (13, 25))],
                             ids=["d1", "d2"])
    def test_two_path_identity_with_an_end_vertex_pinned(self, d, A, ns):
        """Same identity with s = 0, t = 2: now the middle vertex is
        series-collapsed by graph_multiply, whose closing graph_compress
        absorbs the cross exponents nu_i + nu_j - d > d + sigma_max into
        aMat as WINDOW-TRUNCATED sums.  The residual is that truncation:
        it is real (>> roundoff) and shrinks with n (measured d = 1:
        3.8e-5, 1.5e-6, 5.9e-8 at n = 21, 41, 81)."""
        V = mixed_kernel(d, 1.6, 3.2)
        ref = V.lattice_sum(A) ** 2
        res = [abs(zeta0([[0, 1], [1, 2]], [V, V], A, n, s=0, t=2) - ref) / abs(ref) for n in ns]
        assert res[0] >= 1e-7                       # anti-vacuity: a genuine residual
        assert all(a > b for a, b in zip(res, res[1:])), res
        assert res[-1] <= res[0] / 10.0, res
        assert res[-1] <= 1e-4, res


# ---------------------------------------------------------------------------
# (4) Products: one product edge == parallel factor edges
# ---------------------------------------------------------------------------

I1 = Interaction.from_table(cross_table(1, 0.3, 0.2), b=[1.0], nu=[2.6])
I2 = Interaction.from_table(cross_table(1, 0.5, 0.4), b=[0.5], nu=[3.1])
I3 = Interaction.from_table(cross_table(1, 0.2, 0.1), b=[0.8], nu=[2.9])
K26 = Interaction.power_law(2.6)


def _pointwise_product_kernel(factors) -> Interaction:
    """The product of positive kernels spelled as ONE Interaction: its
    pure power-law tail ``prod b_j K_{sum nu_j}`` plus the table
    ``prod V_j(m) - tail(m)`` on the union support (radius 1 here)."""
    prod = factors[0]
    for f in factors[1:]:
        prod = prod * f
    labels = np.arange(-2, 3).reshape(-1, 1)
    samp = prod.sample(labels, A_CHAIN)
    b = float(np.prod([f.b[0] for f in factors]))
    nu = float(sum(f.nu[0] for f in factors))
    dist = np.abs(labels[:, 0]).astype(float)
    tail = np.where(dist > 0.0, b * np.where(dist > 0.0, dist, 1.0) ** (-nu), 0.0)
    table = {(int(m),): float(s - t) for m, s, t in zip(labels[:, 0], samp, tail) if s - t != 0.0}
    return Interaction.from_table(table, b=[b], nu=[nu])


class TestProducts:
    @pytest.mark.parametrize("factors", [(I1, I2), (I1, I2, I3)], ids=["two", "three"])
    def test_product_edge_equals_parallel_factor_edges_bit_exactly(self, factors):
        prod = factors[0]
        for f in factors[1:]:
            prod = prod * f
        assert isinstance(prod, _KernelProduct)
        # The bundle as one edge carrying the lazy product ...
        g_one = build(TRIANGLE, [prod, K26, K26], A_CHAIN, 21)
        # ... and as parallel edges carrying the factors: the same leaves in
        # the same order, so the same graph_convolve calls.
        edges = [[0, 1]] * len(factors) + [[1, 2], [0, 2]]
        g_par = build(edges, list(factors) + [K26, K26], A_CHAIN, 21)
        assert same_object(g_one, g_par)
        assert graph_zero(g_one) == graph_zero(g_par)
        # Anti-vacuity: a different factor set is a different value.
        g_alt = build(TRIANGLE, [I1 * I3, K26, K26], A_CHAIN, 21)
        assert graph_zero(g_alt) != graph_zero(g_one)

    @pytest.mark.parametrize("factors", [(I1, I2), (I1, I2, I3)], ids=["two", "three"])
    def test_product_edge_equals_the_pointwise_product_kernel(self, factors):
        # All kernels here are positive, so the value is its own sum of
        # term magnitudes and a value-relative tolerance is legitimate.
        prod = factors[0]
        for f in factors[1:]:
            prod = prod * f
        v_prod = zeta0(TRIANGLE, [prod, K26, K26], A_CHAIN, 21)
        v_tab = zeta0(TRIANGLE, [_pointwise_product_kernel(factors), K26, K26], A_CHAIN, 21)
        assert v_prod > 0.0 and v_tab > 0.0
        assert abs(v_prod - v_tab) <= 1e-13 * v_prod

    def test_bundle_object_is_the_hadamard_product(self):
        """A two-factor bundle between the terminals is one graph_convolve
        of the two leaves: the pure power-law cross term ``b1 b2
        K_{nu1+nu2}`` stays analytic, the table-power-law cross terms
        and the table product go into aMat on the support."""
        prod = I1 * I2
        assert prod.tail_exponent == 2.6 + 3.1
        g = build([[0, 1]], [prod], A_CHAIN, 21)
        assert np.array_equal(g.nuVec, [prod.tail_exponent])
        assert np.array_equal(g.bVec, [1.0 * 0.5 + 0j])
        assert np.count_nonzero(g.aMat) == 3
        # origin: a1(0) a2(0) alone (K = 0 there)
        assert abs(g.aMat[0] - 0.2 * 0.4) <= 1e-16
        # |x| = 1: V1 V2 - b1 b2 K_5.7 = (0.3 + 1)(0.5 + 0.5) - 0.5 = 0.8
        assert abs(g.aMat[1] - 0.8) <= 1e-15 and abs(g.aMat[20] - 0.8) <= 1e-15
        # The summed tail passes the divergence check inside a block.
        graph_from_edges(TRIANGLE, [prod.tail_exponent, 2.6, 2.6], A_CHAIN, 21,
                         kernels=[prod, K26, K26])


# ---------------------------------------------------------------------------
# (5) The window guard, and what it does not cover
# ---------------------------------------------------------------------------

R6 = radial_table_1d(6, 1.0, 0.5)


class TestWindowGuard:
    @pytest.mark.parametrize("n", [9, 12], ids=["odd-9", "even-12"])
    def test_support_outside_the_balanced_window_raises(self, n):
        # n = 12 holds +6 but not -6: the even window is [-5, 6].
        with pytest.raises(InteractionSupportError):
            zeta0(TRIANGLE, [Interaction.from_table(R6)] * 3, A_CHAIN, n)
        assert issubclass(InteractionSupportError, GraphZetaError)
        assert issubclass(InteractionSupportError, ValueError)

    def test_guard_passes_at_2R_plus_1_and_the_leaf_is_exact(self):
        V = Interaction.from_table(R6)
        leaf = build([[0, 1]], [V], A_CHAIN, 13)
        ref = V.lattice_sum(A_CHAIN)
        assert abs(graph_zero(leaf) - ref) <= 1e-14 * abs(ref)

    def test_composition_aliasing_is_guarded_at_the_block_level(self):
        """The leaf window is not enough: at n = 13 every radius-6 table
        fits, yet the triangle's series collapse has composed support
        3 R = 18 whose aliases land inside the third table (measured 18 %
        wrong before the block-level guard).  The block is refused up to
        n = 18 and exact from n = 3 R + 1 = 19."""
        kernels = [Interaction.from_table(R6)] * 3
        ref, scale = brute_force_compact(TRIANGLE, [R6] * 3, 1)
        for n in (13, 18):
            with pytest.raises(InteractionSupportError, match="compose"):
                zeta0(TRIANGLE, kernels, A_CHAIN, n)
        err19 = abs(zeta0(TRIANGLE, kernels, A_CHAIN, 19) - ref) / scale
        err25 = abs(zeta0(TRIANGLE, kernels, A_CHAIN, 25) - ref) / scale
        assert err19 <= 1e-13 and err25 <= 1e-13, (err19, err25)
        # the nearest-neighbour indicator (R = 1) on cycles: the exact
        # closed walk counts from n = E + 1, refused below
        NN = Interaction.from_table({(1,): 1.0, (-1,): 1.0})
        C6 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
        with pytest.raises(InteractionSupportError):
            zeta0(C6, [NN] * 6, A_CHAIN, 6)
        assert zeta0(C6, [NN] * 6, A_CHAIN, 7) == pytest.approx(20.0, rel=1e-14)

    def test_guard_propagates_through_graph_from_tw2(self):
        nx = pytest.importorskip("networkx")
        mg = nx.MultiGraph()
        for u, v in TRIANGLE:
            mg.add_edge(u, v, nu=math.inf, kern=Interaction.from_table(R6))
        with pytest.raises(InteractionSupportError):
            graph_from_tw2(mg, 0, 1, A_CHAIN, 9)
        with pytest.raises(InteractionSupportError):
            graph_from_tw2(mg, 0, 1, A_CHAIN, 13)      # leaves fit, the composition does not
        graph_from_tw2(mg, 0, 1, A_CHAIN, 19)          # fits: no raise

    def test_n_points_zero_is_refused_on_the_kernel_path(self):
        with pytest.raises(InteractionSupportError):
            build([[0, 1]], [Interaction.from_table({(0,): 0.5})], A_CHAIN, 0)

    def test_interaction_in_nu_is_a_clear_error(self):
        V = Interaction.from_table({(1,): 0.3, (-1,): 0.3})
        with pytest.raises(ValueError, match="kernels="):
            graph_from_edges([[0, 1]], V, A_CHAIN, 9)
        with pytest.raises(ValueError, match="kernels="):
            graph_from_edges([[0, 1]], [V], A_CHAIN, 9)

    def test_table_sampled_on_another_lattice_is_refused(self):
        V = Interaction.from_function(lambda x: np.exp(-np.linalg.norm(x, axis=-1)),
                                      A_CHAIN, radius=2.0)
        build([[0, 1]], [V], A_CHAIN, 21)               # its own lattice: fine
        with pytest.raises(ValueError, match="different lattice"):
            build([[0, 1]], [V], np.array([[1.5]]), 21)


# ---------------------------------------------------------------------------
# (6) sigma_max: the split never touches a compact table
# ---------------------------------------------------------------------------

class TestSigmaMax:
    def test_compact_kernels_are_sigma_max_invariant_bit_exactly(self):
        """graph_compress moves power-law terms between (bVec, nuVec) and
        aMat; with no power-law term there is nothing to move, so every
        sigma_max runs the identical arithmetic."""
        tables = TestAbsolute.TABLES_R7
        kernels = [Interaction.from_table(t) for t in tables]
        vals = [zeta0(TRIANGLE, kernels, A_CHAIN, 41, sm) for sm in SIGMA_MAXES]
        assert all(v == vals[0] for v in vals), vals
        ref, scale = brute_force_compact(TRIANGLE, tables, 1)
        assert abs(vals[0] - ref) <= 1e-13 * scale

    def test_mixed_spread_is_the_power_law_truncation_and_shrinks_with_n(self):
        """For a mixed kernel the value DOES depend on sigma_max: an
        exponent above d + sigma_max is absorbed into aMat as a
        window-truncated sum, exactly as on the legacy path (the pure
        nu = 2.6 triangle moves 1.3e-5 at n = 21, 2.8e-7 at n = 81).  A
        1e-13 sigma_max-independence is therefore not a property of this
        algebra at finite n; what holds is that the spread shrinks with
        the window (measured 1.2e-5, 1.8e-6, 1.8e-7 at n = 21, 41, 81)."""
        V = mixed_kernel(1, 1.6, 3.2)
        spreads = []
        for n in (21, 41, 81):
            vals = [zeta0(TRIANGLE, [V, K26, K26], A_CHAIN, n, sm) for sm in SIGMA_MAXES]
            spreads.append((max(vals) - min(vals)) / abs(vals[-1]))
        assert spreads[0] >= 1e-7, spreads          # anti-vacuity: a real spread
        assert spreads[0] > spreads[1] > spreads[2], spreads
        assert spreads[2] <= spreads[0] / 10.0, spreads


# ---------------------------------------------------------------------------
# (7) MultiGraph input with `kern` attributes
# ---------------------------------------------------------------------------

DECORATED = [[0, 1], [1, 2], [0, 2], [2, 3]]           # triangle + pendant edge
DECORATED_KERNELS = [
    mixed_kernel(1, 1.6, 3.2),
    Interaction.power_law(2.6),
    Interaction.from_table(cross_table(1, 0.2, 0.1), b=[1.0], nu=[2.6]),
    Interaction.from_table(radial_table_1d(2, 0.6, 0.3)),          # purely compact pendant
]


def _multigraph(edges, kernels, nx, with_kern=True):
    mg = nx.MultiGraph()
    for (u, v), k in zip(edges, kernels):
        if with_kern:
            mg.add_edge(u, v, nu=k.tail_exponent, kern=k)
        else:
            mg.add_edge(u, v, nu=k.tail_exponent)
    return mg


class TestMultiGraphKern:
    def test_graph_from_tw2_kern_equals_graph_from_edges_kernels(self):
        nx = pytest.importorskip("networkx")
        mg = _multigraph(DECORATED, DECORATED_KERNELS, nx)
        g_mg = graph_from_tw2(mg, 0, 1, A_CHAIN, 21)
        g_ed = build(DECORATED, DECORATED_KERNELS, A_CHAIN, 21)
        assert same_object(g_mg, g_ed)
        assert graph_zero(g_mg) == graph_zero(g_ed)
        # Anti-vacuity: the pendant's table matters (it is the decoration
        # scalar); doubling it changes the value.
        doubled = DECORATED_KERNELS[:3] + [2.0 * DECORATED_KERNELS[3]]
        assert graph_zero(graph_from_tw2(_multigraph(DECORATED, doubled, nx), 0, 1, A_CHAIN, 21)) \
            != graph_zero(g_mg)

    def test_graph_from_sp_kern_on_the_triangle(self):
        nx = pytest.importorskip("networkx")
        mg = _multigraph(TRIANGLE, DECORATED_KERNELS[:3], nx)
        g_sp = graph_from_sp(mg, 0, 1, A_CHAIN, 21)
        g_ed = build(TRIANGLE, DECORATED_KERNELS[:3], A_CHAIN, 21)
        assert same_object(g_sp, g_ed)

    def test_edges_with_and_without_kern_may_mix(self):
        nx = pytest.importorskip("networkx")
        mg = nx.MultiGraph()
        mg.add_edge(0, 1, nu=2.6, kern=mixed_kernel(1, 1.6, 3.2))
        mg.add_edge(1, 2, nu=2.6)                         # legacy edge
        mg.add_edge(0, 2, nu=2.9)                         # legacy edge
        g_mg = graph_from_tw2(mg, 0, 1, A_CHAIN, 21)
        g_ed = build(TRIANGLE, [mixed_kernel(1, 1.6, 3.2), Interaction.power_law(2.6),
                                Interaction.power_law(2.9)], A_CHAIN, 21)
        assert same_object(g_mg, g_ed)
        # The mixed-kernel edge changes the value against the all-legacy graph.
        g_plain = graph_from_tw2(_multigraph(TRIANGLE, DECORATED_KERNELS[:3], nx, with_kern=False),
                                 0, 1, A_CHAIN, 21)
        assert graph_zero(g_plain) != graph_zero(g_mg)


# ---------------------------------------------------------------------------
# Validation: nu carries the tail; NaN / 0 / <= d are refused, +inf passes
# ---------------------------------------------------------------------------

class TestValidation:
    def test_kernels_length_mismatch(self):
        with pytest.raises(ValueError, match="kernels has length 2"):
            graph_from_edges(TRIANGLE, [2.6] * 3, A_CHAIN, 21, kernels=[K26, K26])

    def test_kernel_entry_must_be_an_interaction(self):
        with pytest.raises(ValueError, match="'kern' must be an Interaction"):
            graph_from_edges(TRIANGLE, [2.6] * 3, A_CHAIN, 21, kernels=[K26, 2.6, K26])

    def test_nu_must_be_the_kernel_tail(self):
        with pytest.raises(ValueError, match="not the tail exponent"):
            graph_from_edges(TRIANGLE, [3.0, 2.6, 2.6], A_CHAIN, 21, kernels=[K26] * 3)
        # A few ULP of float summation are tolerated (a bundle tail formed
        # as a float sum by the router), a real mismatch is not.
        graph_from_edges(TRIANGLE, [2.6 * (1 + 1e-15), 2.6, 2.6], A_CHAIN, 21, kernels=[K26] * 3)

    def test_nan_tail_is_refused(self):
        # NaN passes every `<=` comparison, so the divergence check alone
        # would let it through; the tail check must catch it.
        with pytest.raises(ValueError, match="not the tail exponent"):
            graph_from_edges(TRIANGLE, [math.nan, 2.6, 2.6], A_CHAIN, 21, kernels=[K26] * 3)

    def test_zero_tail_is_refused(self):
        with pytest.raises(ValueError):
            graph_from_edges(TRIANGLE, [0.0, 2.6, 2.6], A_CHAIN, 21, kernels=[K26] * 3)

    @pytest.mark.parametrize("d, A, n", LATTICES)
    def test_divergence_check_reads_the_tail(self, d, A, n):
        # A consistent kernel whose tail is <= d hits the nu <= d check.
        V = Interaction.from_table(cross_table(d, 0.25, 0.3), b=[1.0, 1.0], nu=[d - 0.2, d + 3.0])
        assert V.tail_exponent == d - 0.2
        with pytest.raises(UnsupportedLatticeSumError,
                           match="not strictly greater"):
            graph_from_edges(TRIANGLE, [V.tail_exponent, d + 1.6, d + 1.6], A, n,
                             kernels=[V, Interaction.power_law(d + 1.6), Interaction.power_law(d + 1.6)])

    @pytest.mark.parametrize("d, A, n", LATTICES)
    def test_inf_tail_passes_the_divergence_check(self, d, A, n):
        V = Interaction.from_table(cross_table(d, 0.25, 0.3))
        assert V.tail_exponent == math.inf
        g = graph_from_edges(TRIANGLE, [math.inf] * 3, A, n, kernels=[V] * 3)
        assert g.nuVec.size == 0
        assert math.isfinite(graph_zero(g))

    def test_multigraph_interaction_in_nu_is_refused(self):
        nx = pytest.importorskip("networkx")
        mg = nx.MultiGraph()
        mg.add_edge(0, 1, nu=K26)
        with pytest.raises(ValueError, match="put it in the 'kern' attribute"):
            graph_from_tw2(mg, 0, 1, A_CHAIN, 21)

    def test_multigraph_kern_is_validated(self):
        nx = pytest.importorskip("networkx")
        mg = nx.MultiGraph()
        mg.add_edge(0, 1, nu=2.6, kern="not a kernel")
        with pytest.raises(ValueError, match="'kern' must be an Interaction"):
            graph_from_tw2(mg, 0, 1, A_CHAIN, 21)
        mg = nx.MultiGraph()
        mg.add_edge(0, 1, nu=2.7, kern=K26)
        with pytest.raises(ValueError, match="not the tail exponent"):
            graph_from_sp(mg, 0, 1, A_CHAIN, 21)


# ---------------------------------------------------------------------------
# (8) Conditioning: the table's own magnitude
# ---------------------------------------------------------------------------

EPS = float(np.finfo(float).eps)
C6 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
DIAMOND = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]]


def cancelling_table(c: float) -> dict:
    """``a(±1) = c, a(±2) = −c, a(±3) = 0.1``: six entries of size ~c that
    sum to 0.2."""
    return {(1,): c, (-1,): c, (2,): -c, (-2,): -c, (3,): 0.1, (-3,): 0.1}


def power_law_kappa(g) -> float:
    """``graph_zero_conditioned``'s ratio computed with the power-law
    formula on the same object (the field cleared)."""
    return graph_zero_conditioned(dataclasses.replace(g, table_magnitude=None))[1]


class TestConditioning:
    """``graph_zero_conditioned`` on the kernel path.  The power-law
    formula folds ``aMat`` into ONE term ``|Σ a|``, so a table whose
    entries cancel inside ``np.sum(aMat)`` reported ``kappa = 1`` while
    losing digits; a kernel-built object now carries
    ``GraphZeta.table_magnitude`` (``Σ|a|`` at the leaf, an ℓ¹ bound
    through the algebra) and the ratio uses ``max(|Σ a|, M)``.  The
    power-law path is untouched: its kappa is pinned here as computed
    BEFORE the field existed."""

    #: name -> (edges, nu, n, kappa.hex() before the field existed).
    #: Bridge and theta are exactly 1 (one positive Epstein term; three
    #: convolved ones with an all-zero aMat); the triangle's kappa carries
    #: the FFT-built aMat of its series collapse and gets the
    #: LEGACY_ULP_BAND of the value pins.
    LEGACY_KAPPA = {
        "bridge": ([[0, 1]], [3.0], 9, "0x1.0000000000000p+0"),
        "theta": (THETA, [2.6] * 3, 21, "0x1.0000000000000p+0"),
        "triangle": (TRIANGLE, [2.6] * 3, 21, "0x1.0ce38c29ae217p+7"),
    }

    def test_cancelling_table_bridge_kappa_sees_the_table(self):
        """A bridge carrying the cancelling table at c = 1e9: measured 1.8e-8
        relative loss with kappa = 1 before the fix (eps * kappa = 2e-16
        claimed no loss)."""
        K = Interaction.from_table(cancelling_table(1e9), b=[1.0], nu=[3.0])
        g = build([[0, 1]], [K], A_CHAIN, 9)
        v, kappa = graph_zero_conditioned(g)
        exact = 2.0 * riemann_zeta(3.0) + 0.2          # Z_3(0) on the chain + Σ a
        rel = abs(v - exact) / exact
        assert rel >= 1e-10                            # anti-vacuity: a genuine loss
        assert kappa >= 1e8                            # was exactly 1
        assert rel <= 10.0 * EPS * kappa               # eps * kappa is an error bar again
        assert kappa <= 1e11                           # and not an absurd one (measured 1.5e9)
        assert power_law_kappa(g) == 1.0               # what |Σ a| alone reports
        assert g.table_magnitude == float(np.sum(np.abs(g.aMat)))

    def test_series_of_cancelling_tables_is_flagged_above_the_ceiling(self):
        """Two such tables in series (path 0-1-2, s = 0, t = 2) compose to
        ``(Σ a)² = 0.04`` through FFTs whose operands are ~1e12: measured
        7.1e-3 relative error at c = 1e6, with kappa = 1 before the fix.
        The tracked bound puts kappa above the front-end's 1e12 ceiling
        (``_ALGEBRA_MAX_CONDITION``), the refusal that error deserves."""
        K = Interaction.from_table(cancelling_table(1e6))
        g = graph_from_edges([[0, 1], [1, 2]], [math.inf] * 2, A_CHAIN, 13,
                             s=0, t=2, kernels=[K, K])
        v, kappa = graph_zero_conditioned(g)
        rel = abs(v - 0.04) / 0.04
        assert rel >= 1e-6                             # anti-vacuity: genuinely damaged
        assert kappa >= 1e12
        assert power_law_kappa(g) == 1.0
        assert rel <= EPS * kappa
        # The same tables in PARALLEL are a pointwise product without
        # cancellation (Σ a² = 4 c² + 0.02): the bound sits its factor-2
        # double count above the honest ratio 1, no more.
        g = graph_from_edges([[0, 1], [0, 1]], [math.inf] * 2, A_CHAIN, 13,
                             s=0, t=1, kernels=[K, K])
        v, kappa = graph_zero_conditioned(g)
        assert abs(v - (4e12 + 0.02)) <= 1e-15 * 4e12
        assert 1.0 <= kappa <= 2.0 * (1.0 + 1e-6)      # + the 0.1 / c of the two ±3 entries

    @pytest.mark.parametrize("name", ["bridge", "theta", "triangle"])
    def test_power_law_path_kappa_is_unchanged(self, name):
        edges, nu, n, pin = self.LEGACY_KAPPA[name]
        g = graph_from_edges(edges, nu, A_CHAIN, n, s=0, t=1)
        assert g.table_magnitude is None
        _, kappa = graph_zero_conditioned(g)
        assert ulp_distance(kappa, float.fromhex(pin)) <= LEGACY_ULP_BAND, (
            f"power-law kappa moved: {kappa.hex()} vs pinned {pin}")
        if name != "triangle":
            assert kappa == 1.0

    def test_power_law_kernels_keep_the_power_law_kappa(self):
        nu = [2.6, 3.1, 2.9]
        g_legacy = graph_from_edges(TRIANGLE, nu, A_CHAIN, 21, s=0, t=1)
        g_kernel = graph_from_edges(TRIANGLE, nu, A_CHAIN, 21, s=0, t=1,
                                    kernels=[Interaction.power_law(x) for x in nu])
        assert g_kernel.table_magnitude == 0.0          # an empty table, tracked
        assert graph_zero_conditioned(g_kernel) == graph_zero_conditioned(g_legacy)

    @pytest.mark.parametrize("d, A, n", LATTICES)
    def test_leaf_magnitude_is_the_table_l1_and_kappa_its_ratio(self, d, A, n):
        table = cross_table(d, -0.25, origin=0.3)      # signed: |Σ a| < Σ |a|
        leaf = build([[0, 1]], [Interaction.from_table(table)], A, n)
        l1 = math.fsum(abs(x) for x in table.values())
        total = math.fsum(table.values())
        assert abs(total) < l1                          # anti-vacuity
        assert leaf.table_magnitude == float(np.sum(np.abs(leaf.aMat)))
        assert abs(leaf.table_magnitude - l1) <= 1e-15 * l1
        v, kappa = graph_zero_conditioned(leaf)
        assert abs(v - total) <= 1e-15 * l1
        assert abs(kappa - l1 / abs(total)) <= 1e-12 * kappa
        assert power_law_kappa(leaf) == 1.0

    def test_magnitude_bounds_the_composed_tables(self):
        """Purely compact operands: every coefficient of the composed
        object descends from the tables, so the tracked bound dominates
        ``Σ|aMat|`` after a parallel merge, a series collapse, a
        triangle, a C4 and a three-fold bundle — within the documented
        looseness (measured 2.9x, 4.3x, 20x, 13x, 5.2x)."""
        tables = [radial_table_1d(2, 0.7, -0.3), radial_table_1d(1, -0.5, 0.4),
                  radial_table_1d(2, 0.9, 0.2)]
        kernels = [Interaction.from_table(t) for t in tables] * 2
        cases = [([[0, 1], [0, 1]], 0, 1), ([[0, 1], [1, 2]], 0, 2),
                 (TRIANGLE, 0, 1), (C4, 0, 1), (THETA, 0, 1)]
        for edges, s, t in cases:
            g = graph_from_edges(edges, [math.inf] * len(edges), A_CHAIN, 21,
                                 s=s, t=t, kernels=kernels[:len(edges)])
            l1 = float(np.sum(np.abs(g.aMat)))
            assert l1 > 0.0
            assert g.table_magnitude * (1.0 + 1e-12) >= l1, (edges, g.table_magnitude, l1)
            assert g.table_magnitude <= 50.0 * l1, (edges, g.table_magnitude, l1)

    def test_realistic_mixed_blocks_stay_far_below_the_ceiling(self):
        """A J1+J2-like signed table on a σ = 1.2 tail: kappa is never
        below the power-law formula (``max`` can only raise it) and stays
        ≤ 1e4 (measured ≤ 2.3e3 against an honest ``Σ|aMat| / |value|``
        of up to 9.2e2 on the C6), eight orders below the front-end's
        1e12 — no healthy block is refused for carrying a table."""
        K = Interaction.from_table({(1,): -0.6, (-1,): -0.6, (0,): 0.4},
                                   b=[1.0], nu=[2.2])
        for edges, s, t in ((C4, 0, 1), (THETA, 0, 1), (DIAMOND, 0, 2), (C6, 0, 1)):
            g = graph_from_edges(edges, [2.2] * len(edges), A_CHAIN, 21, s=s, t=t,
                                 kernels=[K] * len(edges))
            v, kappa = graph_zero_conditioned(g)
            assert kappa >= power_law_kappa(g)
            assert kappa <= 1e4, (edges, kappa)
        # Anti-vacuity: on the C6 the honest ratio itself is large (the
        # value is a small residual of cancelling shells).
        assert float(np.sum(np.abs(g.aMat))) / abs(v) >= 100.0

    def test_decorations_scale_the_magnitude(self):
        """A pendant multiplies the host by ``ζ_pendant(0)``: the tracked
        bound follows, identically through ``graph_from_tw2`` (the
        block-cut decoration path) and ``graph_attach``."""
        nx = pytest.importorskip("networkx")
        K = Interaction.from_table(cancelling_table(1e3))
        P = Interaction.from_table(radial_table_1d(1, 0.5, 0.25))    # Σ a = 1.1
        mg = nx.MultiGraph()
        mg.add_edge(0, 1, nu=math.inf, kern=K)
        mg.add_edge(1, 2, nu=math.inf, kern=P)
        g = graph_from_tw2(mg, 0, 1, A_CHAIN, 9)
        leaf = build([[0, 1]], [K], A_CHAIN, 9)
        pendant = build([[0, 1]], [P], A_CHAIN, 9)
        c = abs(complex(graph_zero(pendant)))
        assert abs(c - 1.1) <= 1e-14
        assert g.table_magnitude == leaf.table_magnitude * c
        assert graph_attach(leaf, pendant).table_magnitude == g.table_magnitude
        assert g.table_magnitude != leaf.table_magnitude              # anti-vacuity
        _, kappa = graph_zero_conditioned(g)
        assert kappa >= 1e3 and power_law_kappa(g) == 1.0
        # A power-law host stays on the power-law path.
        host = graph_from_edges([[0, 1]], [3.0], A_CHAIN, 9, s=0, t=1)
        assert graph_attach(host, pendant).table_magnitude is None


# ---------------------------------------------------------------------------
# Validation: the Interaction-in-nu guard on every array form of nu
# ---------------------------------------------------------------------------

class TestInteractionInNuGuard:
    """The friendly 'pass kernels=' ValueError on every form of ``nu``.
    An object-dtype ndarray of Interactions used to fall through to
    numpy's opaque ``float() argument must be ...`` TypeError, and the
    uniform wrapper's ``float(nu)`` to the same."""

    V = Interaction.from_table({(1,): 0.3, (-1,): 0.3})

    def test_object_ndarray_of_interactions(self):
        with pytest.raises(ValueError, match="kernels="):
            graph_from_edges(TRIANGLE, np.array([self.V] * 3, dtype=object), A_CHAIN, 9)
        with pytest.raises(ValueError, match="kernels="):
            graph_from_edges(TRIANGLE, np.array([2.6, self.V, 2.6], dtype=object), A_CHAIN, 9)
        # An object array of floats is just floats.
        g = graph_from_edges(TRIANGLE, np.array([2.6] * 3, dtype=object), A_CHAIN, 9)
        assert graph_zero(g) == graph_zero(graph_from_edges(TRIANGLE, [2.6] * 3, A_CHAIN, 9))

    def test_uniform_wrapper(self):
        with pytest.raises(ValueError, match="kernels="):
            graph_from_edges_uniform(TRIANGLE, self.V, A_CHAIN, 9)
        with pytest.raises(ValueError, match="kernels="):
            graph_from_edges_uniform(TRIANGLE, np.array(self.V, dtype=object), A_CHAIN, 9)
