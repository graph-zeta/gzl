# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
r"""The box engine with general interaction kernels.

Every "equals" here is against an EXACT oracle — a brute-force
enumeration of the truncated sum written from the kernel's definition
(no engine code, no ``Interaction.sample``: see :func:`kernel_table` and
:func:`enumerate_box`), or an identity that holds exactly (per-edge
multilinearity, the hand count of nearest-neighbour walks) — and every
such assertion has an anti-vacuity control showing that the number moves
when it should (a poisoned cache, a window one rung short, a wrong
coefficient, a torus that wraps).  Tolerances are relative to the sum of
the term MAGNITUDES, never to a value a mixed-sign kernel may cancel.

Legacy byte-identity is the frozen harness's job
(``tests/test_engine_reference.py``, ``tests/test_executor_goldens.py``);
this module pins the legacy build site and cache keys directly so a
regression there names the mechanism, not the number.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

import gzl.direct_sum as ds
from gzl import hybrid
from gzl._errors import GraphZetaError
from gzl.interaction import (
    Interaction,
    InteractionSupportError,
    _KernelProduct,
    is_interaction,
)
from gzl.tensor_network import _balanced_z_axis

A1 = np.eye(1)
A2 = np.eye(2)
A_TRI = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])

BRIDGE = np.array([(0, 1)])
PATH2 = np.array([(0, 1), (1, 2)])
PATH3 = np.array([(0, 1), (1, 2), (2, 3)])
TRIANGLE = np.array([(0, 1), (1, 2), (0, 2)])
K4 = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])

RTOL = 1e-13          # fixed-L agreement, relative to the term-magnitude scale


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def cross_table(d: int, val: float, origin: float = 0.0) -> dict:
    """``val`` on the ``2d`` axis-neighbours ``±e_i``, ``origin`` at 0."""
    tab = {}
    for i in range(d):
        for s in (-1, 1):
            m = [0] * d
            m[i] = s
            tab[tuple(m)] = val
    if origin != 0.0:
        tab[(0,) * d] = origin
    return tab


def mixed(d: int) -> Interaction:
    """``0.7 K_{d+1.5} - 0.3 K_{d+2.5} + a``, ``a`` the cross table of radius
    1 with ``a(±e_i) = 0.3`` and ``a(0) = 0.4`` (a coincident-endpoint
    weight — the origin is not special for the compact part)."""
    return Interaction.from_table(cross_table(d, 0.3, origin=0.4),
                                  b=[0.7, -0.3], nu=[d + 1.5, d + 2.5])


def compact01(d: int) -> Interaction:
    """The 0/1 table of radius 1 with ``a(0) = 1``: ``K = 1`` on ``{0, ±e_i}``,
    so every product is 0 or 1 and every sum is an integer."""
    return Interaction.from_table(cross_table(d, 1.0, origin=1.0))


def dyadic(d: int) -> Interaction:
    """A purely compact table with dyadic values (1 on ``±e_i``, 1/2 at 0):
    every product and partial sum is exact in binary, so equality is ``==``."""
    return Interaction.from_table(cross_table(d, 1.0, origin=0.5))


def torus_labels(n: int, d: int) -> np.ndarray:
    """The balanced-window integer labels of an ``n^d`` torus, ``(n,)*d + (d,)``
    — the layout ``hybrid._dense_core`` takes its kernels in."""
    z = _balanced_z_axis(n)
    return np.stack(np.meshgrid(*([z] * d), indexing="ij"), axis=-1)


# ---------------------------------------------------------------------------
# The oracle: brute-force enumeration from the kernel's definition
# ---------------------------------------------------------------------------

def kernel_table(kern, L: int, d: int, A) -> np.ndarray:
    """``K(m)`` on the difference grid ``[-2L, 2L]^d`` from the DEFINITION —
    ``Σ_j b_j |A m|^{-ν_j}`` for ``m ≠ 0`` plus ``a(m)`` — independent of
    ``Interaction.sample``.  A lazy product multiplies its factors' tables;
    a float is the legacy power law (``inf``: the minimal-shell indicator)."""
    if isinstance(kern, _KernelProduct):
        out = None
        for f in kern.factors:
            t = kernel_table(f, L, d, A)
            out = t if out is None else out * t
        return out
    ax = np.arange(-2 * L, 2 * L + 1)
    labels = np.stack(np.meshgrid(*([ax] * d), indexing="ij"),
                      axis=-1).reshape(-1, d)
    dist = np.linalg.norm(labels @ np.asarray(A, dtype=float).T, axis=1)
    out = np.zeros(len(labels))
    nz = dist > 0.0
    if is_interaction(kern):
        for bj, nj in zip(kern.b, kern.nu):
            out[nz] += bj * dist[nz] ** (-nj)
        table = dict(kern.compact)
        for i, m in enumerate(map(tuple, labels.tolist())):
            out[i] += table.get(m, 0.0)
    else:
        nu = float(kern)
        if math.isinf(nu):
            dmin = dist[nz].min()
            out[nz & np.isclose(dist, dmin, rtol=1e-9, atol=0.0)] = 1.0
        else:
            out[nz] = dist[nz] ** (-nu)
    return out.reshape((4 * L + 1,) * d)


def enumerate_box(edges, kernels, A, L: int, root: int, *,
                  terminal=None, k=None):
    r"""``Σ`` over every free-vertex position in ``[-L, L]^d`` (root at the
    origin) of ``Π_e K_e(x_v - x_u)`` — times ``cos(2π k·x_terminal)`` when
    a momentum is given — as the full broadcast product summed once (no
    elimination, no FFT).  Returns ``(value, Σ |terms|)``; the second is
    the scale every tolerance in this module is relative to."""
    edges = np.asarray(edges, dtype=int)
    A = np.asarray(A, dtype=float)
    d = A.shape[0]
    V = int(edges.max()) + 1
    free = [v for v in range(V) if v != root]
    axis_of = {v: i for i, v in enumerate(free)}
    ax = np.arange(-L, L + 1)
    pos = np.stack(np.meshgrid(*([ax] * d), indexing="ij"),
                   axis=-1).reshape(-1, d)
    N = len(pos)

    def lookup(tab, delta):
        return tab[tuple(delta[..., c] + 2 * L for c in range(d))]

    prod = np.ones((N,) * len(free))
    for (u, v), kern in zip(edges.tolist(), kernels):
        tab = kernel_table(kern, L, d, A)
        if u == root or v == root:
            other = v if u == root else u
            vals = lookup(tab, pos if v == other else -pos)
            shape = [1] * len(free)
            shape[axis_of[other]] = N
            prod = prod * vals.reshape(shape)
        else:
            delta = pos[None, :, :] - pos[:, None, :]          # x_v - x_u
            vals = lookup(tab, delta)
            if axis_of[u] > axis_of[v]:
                vals = vals.T
            shape = [1] * len(free)
            shape[axis_of[u]] = N
            shape[axis_of[v]] = N
            prod = prod * vals.reshape(shape)
    if k is not None and terminal is not None and terminal != root:
        kf = np.asarray(k, dtype=float).reshape(d)
        phase = np.cos(2.0 * np.pi * (pos @ kf))
        shape = [1] * len(free)
        shape[axis_of[terminal]] = N
        prod = prod * phase.reshape(shape)
    return float(prod.sum()), float(np.abs(prod).sum())


def box(edges, kernels, A, L: int, root: int, **kw) -> float:
    """The engine on per-edge kernels; ``nu`` carries the tails."""
    nu = np.array([k.tail_exponent for k in kernels], dtype=float)
    return complex(ds.direct_sum_zero_momentum(
        edges, nu, A, L, root=root, kernels=list(kernels), **kw)).real


def assert_close(got, want, scale, rtol=RTOL):
    assert abs(got - want) <= rtol * scale, (got, want, scale)


def _eccentricity(edge_map, root) -> int:
    """Edge-count eccentricity of ``root``: what a reach of ``R x ecc``
    would assume, the contrast for ``_compact_reach``."""
    return nx.eccentricity(nx.Graph(list(edge_map)), v=root)


class TestTheOracleItself:
    """The enumeration must reproduce the legacy engine on a plain power
    law — otherwise every agreement below could be a shared mistake."""

    @pytest.mark.parametrize("d,A,L", [(1, A1, 4), (2, A2, 2), (2, A_TRI, 2)])
    def test_enumeration_matches_the_legacy_box_on_power_laws(self, d, A, L):
        nu = np.array([d + 1.5] * 6)
        legacy = complex(ds.direct_sum_zero_momentum(K4, nu, A, L, root=0)).real
        want, scale = enumerate_box(K4, [d + 1.5] * 6, A, L, 0)
        assert_close(legacy, want, scale)
        # and the two-path NN count of tests/test_box_nu_inf.py
        assert enumerate_box(PATH2, [np.inf, np.inf], A1, 3, 0)[0] == 4.0


# ---------------------------------------------------------------------------
# 1. The legacy build site and the cache keys
# ---------------------------------------------------------------------------

class TestLegacyBuildSite:
    @pytest.mark.parametrize("A", [A1, A2, A_TRI])
    def test_conv_kernel_diff_is_unchanged_with_interaction_none(self, A):
        d, L = A.shape[0], 3
        for nu in (2.5, 3.5, np.inf):
            legacy = ds._conv_kernel_diff(nu, A, L, d)
            assert np.array_equal(legacy, ds._conv_kernel_diff(nu, A, L, d,
                                                               interaction=None))
        # Anti-vacuity: an interaction is consulted when given — a table
        # of radius 1 on the same tail moves exactly the 2d neighbour
        # labels and nothing else, and the plain power law is bit-equal.
        legacy = ds._conv_kernel_diff(2.5, A, L, d)
        T = Interaction.from_table(cross_table(d, 0.3), b=[1.0], nu=[2.5])
        withT = ds._conv_kernel_diff(2.5, A, L, d, interaction=T)
        assert np.flatnonzero(withT != legacy).size == 2 * d
        assert withT[(2 * L,) * d] == 0.0
        P = Interaction.power_law(2.5)
        assert np.array_equal(ds._conv_kernel_diff(2.5, A, L, d, interaction=P),
                              legacy)

    @staticmethod
    def _capture_eliminate(monkeypatch):
        """Record the ``kdiff_cache`` and the factor list of the next
        ``_eliminate_all`` call (the 10th positional argument)."""
        seen = {}
        real = ds._eliminate_all

        def capture(factors, *a, **kw):
            seen["factors"] = list(factors)
            seen["cache"] = a[8] if len(a) >= 9 else kw.get("kdiff_cache")
            return real(factors, *a, **kw)

        monkeypatch.setattr(ds, "_eliminate_all", capture)
        return seen

    @staticmethod
    def _force_peel_gate(monkeypatch):
        monkeypatch.setattr(ds, "_conv_possible", lambda *a, **kw: True)
        monkeypatch.setattr(ds, "_FFT_MIN_BAG", 0)

    def test_legacy_cache_keys_and_tags_are_floats(self, monkeypatch):
        seen = self._capture_eliminate(monkeypatch)
        self._force_peel_gate(monkeypatch)      # so hat entries appear too
        ds.direct_sum_zero_momentum(K4, np.full(6, 2.5), A1, 6, root=0)
        keys = list(seen["cache"])
        floats = [k for k in keys if isinstance(k, float)]
        hats = [k for k in keys if isinstance(k, tuple)]
        assert floats == [2.5]
        assert hats and all(h[0] == "hat" and isinstance(h[1], float)
                            for h in hats)
        tags = [f["conv_nu"] for f in seen["factors"] if "conv_nu" in f]
        assert tags and all(isinstance(t, float) for t in tags)
        assert not any("conv_kernel" in f for f in seen["factors"])

    def test_kernel_cache_keys_are_key_tuples_never_floats(self, monkeypatch):
        seen = self._capture_eliminate(monkeypatch)
        self._force_peel_gate(monkeypatch)
        M = mixed(1)
        box(K4, [M] * 6, A1, 6, 0)
        keys = list(seen["cache"])
        assert not any(isinstance(k, float) for k in keys)
        gen_keys = [k for k in keys if k[0] == "K"]
        assert gen_keys == [("K",) + M.key()]
        hats = [k for k in keys if k[0] == "hat"]
        assert hats and all(h[1] == ("K",) + M.key() for h in hats)
        tagged = [f for f in seen["factors"] if "conv_nu" in f]
        assert tagged and all(f["conv_nu"] == ("K",) + M.key()
                              and f["conv_kernel"] is M for f in tagged)

    def test_equal_tails_different_tables_get_distinct_entries(self,
                                                               monkeypatch):
        """The stale-entry regression: two kernels with the SAME tail and
        different tables on one path must each get their own generator.
        Served one generator, the value would be the [T1, T1] number,
        6.75e-3 of the scale away — the control below."""
        seen = self._capture_eliminate(monkeypatch)
        T1 = Interaction.from_table(cross_table(1, 0.3), b=[1.0], nu=[3.0])
        T2 = Interaction.from_table(cross_table(1, 0.31), b=[1.0], nu=[3.0])
        assert T1.tail_exponent == T2.tail_exponent
        got = box(PATH2, [T1, T2], A1, 4, 0)
        assert sorted(k for k in seen["cache"] if k[0] == "K") == sorted(
            [("K",) + T1.key(), ("K",) + T2.key()])
        want, scale = enumerate_box(PATH2, [T1, T2], A1, 4, 0)
        assert_close(got, want, scale)
        poisoned, _ = enumerate_box(PATH2, [T1, T1], A1, 4, 0)
        assert abs(poisoned - want) > 1e-3 * scale

    def test_box_truncation_generator_keys_interactions_separately(self):
        cache: dict = {}
        tr = ds.BoxTruncation(3, 1, A1, kdiff_cache=cache)
        T = Interaction.from_table(cross_table(1, 0.3), b=[1.0], nu=[2.5])
        g = tr.generator(T)
        assert tr.generator(T) is g
        assert set(cache) == {("K",) + T.key()}
        assert np.array_equal(g, ds._conv_kernel_diff(2.5, A1, 3, 1,
                                                      interaction=T))
        # float keys keep their own space, and a plain power law passed as
        # an OBJECT is a third entry (never normalised into the float's)
        f = tr.generator(2.5)
        p = tr.generator(Interaction.power_law(2.5))
        assert set(cache) == {("K",) + T.key(), 2.5,
                              ("K",) + Interaction.power_law(2.5).key()}
        assert p is not f and np.array_equal(p, f)
        cache[("hat", 2.5, 7)] = "sentinel"
        assert tr.generator(2.5) is f and tr.generator(T) is g

    def test_peel_cache_miss_rebuilds_from_the_kernel_object(self):
        """A hand-built factor list (no pre-filled cache) peels a kernel
        factor by rebuilding its generator from ``conv_kernel``."""
        L, d = 3, 1
        M = mixed(1)
        gen = ds._conv_kernel_diff(M.tail_exponent, A1, L, d, interaction=M)
        ext = ds._axis_extents(L, d, False)
        org = ds._axis_origins(L, d, False)
        off2 = ds._gen_offsets(L, d, org, org)
        off1 = tuple(int(org[c] + 2 * L) for c in range(d))
        key = ("K",) + M.key()
        factors = [
            {"scope": (1,), "gen": gen, "ext": (ext,), "off": off1, "d": d},
            {"scope": (1, 2), "gen": gen, "ext": (ext, ext), "off": off2,
             "d": d, "conv_nu": key, "conv_kernel": M},
        ]
        old = (ds._USE_CONV, ds._FFT_MARGIN, ds._FFT_MIN_BAG)
        ds._USE_CONV, ds._FFT_MARGIN, ds._FFT_MIN_BAG = True, 0.0, 0
        try:
            cache: dict = {}
            out = ds._eliminate_all(factors, [1, 2], None, 2 * L + 1, L + 1,
                                    np.where(np.arange(L + 1) == 0, 1.0, 2.0),
                                    A1, L, d, cache)
        finally:
            ds._USE_CONV, ds._FFT_MARGIN, ds._FFT_MIN_BAG = old
        assert key in cache and ("hat", key, 2 * L + 1) in cache
        got = float(np.prod([ds._factor_array(f) for f in out]))
        want, scale = enumerate_box(PATH2, [M, M], A1, L, 0)
        assert_close(got, want, scale)


# ---------------------------------------------------------------------------
# 2. Fixed L against the enumeration
# ---------------------------------------------------------------------------

GRAPHS = [("bridge", BRIDGE), ("path2", PATH2), ("triangle", TRIANGLE), ("K4", K4)]
WINDOWS = [(1, 3), (1, 4), (1, 5), (2, 2), (2, 3)]


class TestFixedLEnumeration:
    @pytest.mark.parametrize("name,edges", GRAPHS, ids=[g[0] for g in GRAPHS])
    @pytest.mark.parametrize("d,L", WINDOWS)
    @pytest.mark.parametrize("use_symmetry", [True, False])
    def test_mixed_kernels(self, name, edges, d, L, use_symmetry):
        A = A1 if d == 1 else A2
        kern = [mixed(d)] * len(edges)
        want, scale = enumerate_box(edges, kern, A, L, 0)
        got = box(edges, kern, A, L, 0, use_symmetry=use_symmetry)
        assert_close(got, want, scale)
        # anti-vacuity: the kernel is what makes the number — the plain
        # tail alone (b = 0.7 K_{d+1.5} only) is a different sum
        tail_only, _ = enumerate_box(edges, [0.7 * Interaction.power_law(d + 1.5)]
                                     * len(edges), A, L, 0)
        assert abs(tail_only - want) > 1e-3 * scale

    def test_on_the_triangular_cell(self):
        M = Interaction.from_table(cross_table(2, 0.3, origin=0.4),
                                   b=[0.7, -0.3], nu=[3.5, 4.5])
        want, scale = enumerate_box(K4, [M] * 6, A_TRI, 2, 0)
        assert_close(box(K4, [M] * 6, A_TRI, 2, 0), want, scale)

    @pytest.mark.parametrize("d,A,L,k", [(1, A1, 4, [0.3]), (2, A2, 2, [0.3, 0.1])])
    def test_single_momentum(self, d, A, L, k):
        kern = [mixed(d)] * 3
        want, scale = enumerate_box(TRIANGLE, kern, A, L, 0, terminal=2, k=k)
        got = box(TRIANGLE, kern, A, L, 0, momentum=np.array(k), terminal=2)
        assert_close(got, want, scale)
        vac, _ = enumerate_box(TRIANGLE, kern, A, L, 0)
        assert abs(vac - want) > 1e-3 * scale       # the phase acts

    def test_open_terminal_residual_transforms_to_the_enumeration(self):
        """The open-terminal engine with a kernel map: ``Σ_x cos(2π k x)
        M(x)`` at fixed L is the finite-k enumeration for every k."""
        L, M = 4, mixed(1)
        nu = np.full(6, M.tail_exponent)
        em = ds._collapse_multi_edges(K4, nu)
        km = ds._collapse_multi_kernels(K4, nu, [M] * 6)
        Mres, pos = ds._direct_sum_open_terminal(em, A1, L, 0, 3, 1, kern_map=km)
        for k in (0.0, 0.25, 0.3):
            got = float(np.cos(2.0 * np.pi * k * pos[:, 0]) @ Mres)
            want, scale = enumerate_box(K4, [M] * 6, A1, L, 0, terminal=3, k=[k])
            assert_close(got, want, scale)

    def test_parallel_rows_multiply_in_row_order(self):
        """Repeated rows are one bundle: the lazy product of the row
        kernels, bit-identical to passing the product on a single row."""
        T1 = Interaction.from_table(cross_table(1, 0.3), b=[1.0], nu=[2.5])
        T2 = Interaction(b=[1.0, -0.2], nu=[3.0, 5.0])
        km = ds._collapse_multi_kernels(np.array([(0, 1), (1, 0), (1, 2)]),
                                        np.array([2.5, 3.0, 3.0]), [T1, T2, T2])
        assert isinstance(km[(0, 1)], _KernelProduct)
        assert km[(0, 1)].factors == (T1, T2) and km[(1, 2)] is T2
        two_rows = box(np.array([(0, 1), (1, 0)]), [T1, T2], A1, 4, 0)
        one_row = box(BRIDGE, [T1 * T2], A1, 4, 0)
        assert two_rows == one_row
        want, scale = enumerate_box(BRIDGE, [T1 * T2], A1, 4, 0)
        assert_close(one_row, want, scale)

    def test_an_interaction_in_nu_names_kernels(self):
        """The box accepts a general kernel only through ``kernels=``; an
        Interaction in ``nu`` used to die in numpy's float coercion with
        an opaque TypeError.  Every entry point, list entry or bare."""
        M = mixed(1)
        kk = np.zeros((1, 1))
        calls = (
            lambda nu: ds.direct_sum_zero_momentum(TRIANGLE, nu, A1, 3),
            lambda nu: ds.direct_sum_extrapolated(TRIANGLE, nu, A1,
                                                  L_list=(2, 3, 4, 5)),
            lambda nu: ds._direct_sum_extrapolated_grid(
                TRIANGLE, nu, A1, kk, root=0, terminal=1, L_list=(2, 3, 4, 5)),
        )
        for call in calls:
            with pytest.raises(ValueError, match=r"nu_vec\[2\] is an Interaction.*kernels="):
                call([2.5, 2.5, M])
            with pytest.raises(ValueError, match=r"nu_vec is an Interaction.*kernels="):
                call(M)
            with pytest.raises(ValueError, match="kernels="):
                call([M * M] * 3)
        assert np.isfinite(box(TRIANGLE, [M] * 3, A1, 3, 0))

    def test_kernel_list_validation(self):
        M = mixed(1)
        nu = np.full(3, M.tail_exponent)
        with pytest.raises(ValueError, match="length"):
            ds.direct_sum_zero_momentum(TRIANGLE, nu, A1, 3, kernels=[M, M])
        with pytest.raises(TypeError, match="Interaction"):
            ds.direct_sum_zero_momentum(TRIANGLE, nu, A1, 3, kernels=[M, M, 2.5])
        # nu is the TAIL: a row whose nu says "finite" while its kernel
        # has none (or the reverse) is refused, not silently re-keyed
        with pytest.raises(ValueError, match="tail"):
            ds.direct_sum_zero_momentum(TRIANGLE, nu, A1, 3,
                                        kernels=[M, M, compact01(1)])
        with pytest.raises(ValueError, match="tail"):
            ds.direct_sum_zero_momentum(TRIANGLE, np.array([np.inf, 2.5, 2.5]),
                                        A1, 3, kernels=[M, M, M])
        # a table label outside [-2L, 2L]^d is refused at the build site
        far = Interaction.from_table({(7,): 0.1, (-7,): 0.1}, b=[1.0], nu=[2.5])
        with pytest.raises(InteractionSupportError):
            ds.direct_sum_zero_momentum(BRIDGE, np.array([2.5]), A1, 3,
                                        kernels=[far])


# ---------------------------------------------------------------------------
# 3. Per-edge multilinearity
# ---------------------------------------------------------------------------

class TestMultilinearity:
    @pytest.mark.parametrize("edges", [TRIANGLE, K4], ids=["triangle", "K4"])
    @pytest.mark.parametrize("d,L", [(1, 4), (2, 2)])
    def test_edge_zero_mixed_rest_legacy(self, edges, d, L):
        r"""``ζ(e_0 → 0.7 K_a - 0.3 K_b + a) = 0.7 ζ(e_0 → K_a) - 0.3 ζ(e_0 →
        K_b) + ζ(e_0 → a)`` at fixed L, the other edges the legacy power law
        ``K_{d+1.5}``.  The first two terms come from the LEGACY float path
        (``kernels=None``), so this also ties the kernel path to it."""
        A = A1 if d == 1 else A2
        E = len(edges)
        M, P = mixed(d), Interaction.power_law(d + 1.5)
        T = Interaction.from_table(cross_table(d, 0.3, origin=0.4))
        lhs = box(edges, [M] + [P] * (E - 1), A, L, 0)
        za = complex(ds.direct_sum_zero_momentum(
            edges, np.array([d + 1.5] * E), A, L, root=0)).real
        zb = complex(ds.direct_sum_zero_momentum(
            edges, np.array([d + 2.5] + [d + 1.5] * (E - 1)), A, L, root=0)).real
        zt = box(edges, [T] + [P] * (E - 1), A, L, 0)
        scale = 0.7 * abs(za) + 0.3 * abs(zb) + abs(zt)
        assert abs(lhs - (0.7 * za - 0.3 * zb + zt)) <= RTOL * scale
        # anti-vacuity: a wrong coefficient breaks the identity by percent
        assert abs(lhs - (0.6 * za - 0.3 * zb + zt)) > 1e-3 * scale


# ---------------------------------------------------------------------------
# 4. A purely compact block is exact — box, enumeration and torus agree
# ---------------------------------------------------------------------------

class TestPurelyCompactIsExactAcrossEngines:
    @pytest.mark.parametrize("d,A,L,n", [(1, A1, 2, 6), (2, A2, 2, 6)])
    def test_k4_box_equals_enumeration_equals_dense_core(self, d, A, L, n):
        """K4 on the 0/1 table of radius 1 with ``a(0) = 1``: with ``L >=
        R·ecc = 1`` the box is the exact infinite-lattice count, and so is
        the torus once no cycle can wrap (``n >= n_v R + 2 = 6``)."""
        C = compact01(d)
        b = box(K4, [C] * 6, A, L, 0)
        e, _ = enumerate_box(K4, [C] * 6, A, L, 0)
        Ks = [C.sample(torus_labels(n, d), A)] * 6
        t = float(hybrid._dense_core([0, 1, 2, 3], [tuple(x) for x in K4.tolist()],
                                     Ks, 0, 0, n, d, A, peelable=False))
        assert b == e and t == e
        assert e == float(int(e)) and e == (15.0 if d == 1 else 29.0)
        assert box(K4, [C] * 6, A, L + 1, 0) == e            # L-independent
        # anti-vacuity: a torus that wraps (n = 3 < n_v R) overcounts
        Ks3 = [C.sample(torus_labels(3, d), A)] * 6
        t3 = float(hybrid._dense_core([0, 1, 2, 3], [tuple(x) for x in K4.tolist()],
                                      Ks3, 0, 0, 3, d, A, peelable=False))
        assert t3 > e

    def test_nearest_neighbour_table_is_the_legacy_inf_indicator(self):
        for d, A, L in [(1, A1, 2), (2, A2, 2), (2, A_TRI, 2)]:
            NN = Interaction.nearest_neighbour(A)
            got = box(K4, [NN] * 6, A, L, 0)
            legacy = complex(ds.direct_sum_zero_momentum(
                K4, np.full(6, np.inf), A, L, root=0)).real
            assert got == legacy


# ---------------------------------------------------------------------------
# 5. Peel liveness: compact never, mixed always (under the forced gate)
# ---------------------------------------------------------------------------

class TestPeelLiveness:
    @staticmethod
    def _count_peels(monkeypatch):
        calls = {"n": 0}
        orig = ds._fft_conv_step

        def counting(*a, **kw):
            calls["n"] += 1
            return orig(*a, **kw)

        monkeypatch.setattr(ds, "_fft_conv_step", counting)
        monkeypatch.setattr(ds, "_conv_possible", lambda *a, **kw: True)
        monkeypatch.setattr(ds, "_FFT_MIN_BAG", 0)
        return calls

    def test_purely_compact_kernel_never_peels(self, monkeypatch):
        """root=0 keeps the (1, 2) kernel free-free, so a peelable
        STRUCTURE exists and the missing tag is what stops the peel; the
        dense contraction of a dyadic table is then exact (``==``)."""
        calls = self._count_peels(monkeypatch)
        D = dyadic(1)
        got = box(PATH2, [D, D], A1, 3, 0)
        assert calls["n"] == 0
        assert got == enumerate_box(PATH2, [D, D], A1, 3, 0)[0]

    @pytest.mark.parametrize("edges", [PATH2, K4], ids=["path2", "K4"])
    def test_mixed_kernel_peels_and_matches_the_enumeration(self, edges,
                                                            monkeypatch):
        calls = self._count_peels(monkeypatch)
        M = mixed(1)
        got = box(edges, [M] * len(edges), A1, 3, 0)
        assert calls["n"] >= 1
        want, scale = enumerate_box(edges, [M] * len(edges), A1, 3, 0)
        assert_close(got, want, scale)


# ---------------------------------------------------------------------------
# 6. The reach guard on both extrapolated entry points
# ---------------------------------------------------------------------------

#: A purely compact table of Chebyshev radius 2 with dyadic values.
R2 = Interaction.from_table({(0,): 0.5, (1,): 1.0, (-1,): 1.0,
                             (2,): 0.25, (-2,): 0.25})
NU_INF3 = np.full(3, np.inf)


class TestCompactReachGuard:
    def test_error_class(self):
        assert issubclass(InteractionSupportError, ValueError)
        assert issubclass(InteractionSupportError, GraphZetaError)

    def test_p4_at_the_box_root(self):
        """The 4-vertex path with ``root=None``: the box picks the
        max-degree vertex, ``_pick_root`` → 1 (degrees [1, 2, 2, 1]), whose
        eccentricity is 2; with R = 2 the reach is 4.  A ladder whose
        smallest rung is 3 is refused; one starting at 4 is exact."""
        em = ds._collapse_multi_edges(PATH3, NU_INF3)
        root_eff = ds._pick_root(em, 4)
        assert root_eff == 1 and _eccentricity(em, root_eff) == 2
        assert R2.support_radius == 2
        assert ds._compact_reach({k: R2 for k in em}, em, root_eff) == 4
        # an all-compact block has no tail to fit: the degenerate branch
        # evaluates ONE rung lifted to the reach, so a short ladder is
        # lifted rather than refused, and is exact
        got = complex(ds.direct_sum_extrapolated(
            PATH3, NU_INF3, A1, L_list=(4, 5, 6, 7), kernels=[R2] * 3)).real
        exact = enumerate_box(PATH3, [R2] * 3, A1, 7, 1)[0]
        assert got == exact == 27.0
        lifted = complex(ds.direct_sum_extrapolated(
            PATH3, NU_INF3, A1, L_list=(3, 4, 5, 6), kernels=[R2] * 3)).real
        assert lifted == exact
        short = complex(ds.direct_sum_extrapolated(
            PATH3, NU_INF3, A1, L_list=(2, 3), n_correction_terms=1,
            kernels=[R2] * 3)).real
        assert short == exact                           # lifted from 3 to the reach 4
        # root-independent, i.e. the infinite-lattice value ...
        assert enumerate_box(PATH3, [R2] * 3, A1, 8, 0)[0] == exact
        # ... which the refused rung would have clipped (anti-vacuity):
        # the enumeration at L = 3 is a different number, and the single-L
        # engine refuses to return it for an all-compact block
        # (TestFixedLReachGuard) instead of clipping silently
        assert enumerate_box(PATH3, [R2] * 3, A1, 3, 1)[0] != exact
        with pytest.raises(InteractionSupportError, match="reach"):
            box(PATH3, [R2] * 3, A1, 3, 1)

    def test_grid_path_from_root_zero(self):
        """Explicit ``root=0``: eccentricity 3, reach 6.  Passing, the
        one exact rung is the value at EVERY momentum of the grid."""
        kk = np.array([[0.0], [0.25], [0.5]])
        got = ds._direct_sum_extrapolated_grid(PATH3, NU_INF3, A1, kk, root=0,
                                               terminal=3, L_list=(6, 7, 8, 9),
                                               kernels=[R2] * 3)
        assert got.shape == (3,)
        # a ladder below the reach is lifted to it (all-compact block)
        lifted = ds._direct_sum_extrapolated_grid(PATH3, NU_INF3, A1, kk, root=0,
                                                  terminal=3, L_list=(3, 4, 5, 6),
                                                  kernels=[R2] * 3)
        np.testing.assert_allclose(lifted, got, rtol=1e-14, atol=1e-13)   # one rung at 6 vs at 9: same exact sum
        for g, k in zip(got, kk):
            want, scale = enumerate_box(PATH3, [R2] * 3, A1, 9, 0, terminal=3, k=k)
            assert_close(float(g), want, scale, rtol=1e-15)
        assert got[0] == 27.0

    def test_mixed_kernel_is_guarded_too(self):
        """A finite tail does not exempt the compact part: below the reach
        the ladder is refused, at the reach it runs (and fits)."""
        T = Interaction.from_table(dict(R2.compact), b=[1.0], nu=[2.5])
        nu = np.full(3, 2.5)
        with pytest.raises(InteractionSupportError, match="reach"):
            ds.direct_sum_extrapolated(PATH3, nu, A1, L_list=(3, 4, 5, 6),
                                       kernels=[T] * 3)
        v = complex(ds.direct_sum_extrapolated(PATH3, nu, A1, L_list=(4, 5, 6, 7),
                                               kernels=[T] * 3)).real
        assert np.isfinite(v)

    def test_the_message_names_the_compact_subgraph_distance(self):
        """The reach is a radius-weighted shortest path over the COMPACT
        subgraph, not max radius x eccentricity: the two differ exactly
        where the guard matters (a power-law chord: the triangle with two
        R = 2 tables and one power-law edge has reach 4 at roots 0 and 2
        while R x ecc = 2), and the old parenthetical told a user sizing
        ``L_list`` by hand that L >= 2 suffices where the box at L = 2
        was 5.8 % low."""
        T = Interaction.from_table(dict(R2.compact), b=[1.0], nu=[2.5])
        with pytest.raises(InteractionSupportError) as ei:
            ds.direct_sum_extrapolated(PATH3, np.full(3, 2.5), A1,
                                       L_list=(3, 4, 5, 6), kernels=[T] * 3)
        msg = str(ei.value)
        assert "radius-weighted shortest-path distance over the compact subgraph" in msg
        assert "eccentricity" not in msg
        # the case the wording is about: reach 4 where R x ecc would say 2
        tri_nu = np.array([np.inf, np.inf, 2.5])
        em = ds._collapse_multi_edges(TRIANGLE, tri_nu)
        km = ds._collapse_multi_kernels(TRIANGLE, tri_nu,
                                        [R2, R2, Interaction.power_law(2.5)])
        assert ds._compact_reach(km, em, 0) == 4
        assert _eccentricity(em, 0) * R2.support_radius == 2

    def test_no_compact_part_no_guard(self):
        """A multi-term pure power law has reach 0 and takes the ladder as
        the legacy power law does."""
        V = Interaction(b=[1.0, -0.2], nu=[2.5, 4.5])
        em = ds._collapse_multi_edges(K4, np.full(6, 2.5))
        assert ds._compact_reach({k: V for k in em}, em, 0) == 0
        v = complex(ds.direct_sum_extrapolated(K4, np.full(6, 2.5), A1,
                                               L_list=(2, 3, 4, 5), root=0,
                                               kernels=[V] * 6)).real
        assert np.isfinite(v)


class TestFixedLReachGuard:
    """``direct_sum_zero_momentum`` at ONE box half-width, on a block with
    no finite tail at the root.  Measured before the guard: the purely
    compact 2-path with radius-2 tables (root 0, reach 4) returned 0 at
    L = 1 and 2 at L = 2, 3 for an exact 4 — not a truncation of
    anything, a clipped count with no diagnostic — while the extrapolated
    entry points already lifted their one rung to the reach."""
    T2 = Interaction.from_table({(2,): 1.0, (-2,): 1.0})
    NU2 = np.full(2, np.inf)

    def test_below_the_reach_is_refused_at_the_reach_it_is_exact(self):
        em = ds._collapse_multi_edges(PATH2, self.NU2)
        assert ds._compact_reach({k: self.T2 for k in em}, em, 0) == 4
        assert not math.isfinite(1.0 - ds._min_free_cut_nu(em, 0))   # alpha = -inf
        for L, clipped in ((1, 0.0), (2, 2.0), (3, 2.0)):
            with pytest.raises(InteractionSupportError, match="compact reach 4"):
                box(PATH2, [self.T2] * 2, A1, L, 0)
            # what the refused call used to return: the clipped count
            assert enumerate_box(PATH2, [self.T2] * 2, A1, L, 0)[0] == clipped
        assert box(PATH2, [self.T2] * 2, A1, 4, 0) == 4.0
        assert enumerate_box(PATH2, [self.T2] * 2, A1, 4, 0)[0] == 4.0
        assert box(PATH2, [self.T2] * 2, A1, 5, 0) == 4.0
        # the extrapolated entry lifts a short ladder to the same rung
        got = complex(ds.direct_sum_extrapolated(
            PATH2, self.NU2, A1, L_list=(1, 2), root=0, n_correction_terms=1,
            kernels=[self.T2] * 2)).real
        assert got == 4.0

    def test_the_reach_is_the_roots(self):
        """Pinned at the middle vertex the reach is 2: L = 1 is refused,
        L = 2 is exact, and the number is the same 4."""
        em = ds._collapse_multi_edges(PATH2, self.NU2)
        assert ds._compact_reach({k: self.T2 for k in em}, em, 1) == 2
        with pytest.raises(InteractionSupportError, match="compact reach 2"):
            box(PATH2, [self.T2] * 2, A1, 1, 1)
        assert box(PATH2, [self.T2] * 2, A1, 2, 1) == 4.0

    def test_a_finite_tail_keeps_the_fixed_L_truncation_semantics(self):
        """With a power-law tail the single-L call IS a truncation (the
        docstring's contract): below the reach it clips the table-tied
        terms and is not refused — the ladder's smallest-rung guard is
        what protects a fit.  The legacy indicator (no kernels) is
        untouched: the same clipped numbers as before."""
        M = Interaction.from_table({(2,): 1.0, (-2,): 1.0}, b=[1.0], nu=[2.5])
        for L in (1, 2, 4):
            want, scale = enumerate_box(PATH2, [M] * 2, A1, L, 0)
            assert_close(box(PATH2, [M] * 2, A1, L, 0), want, scale)
        legacy = [complex(ds.direct_sum_zero_momentum(
            PATH2, self.NU2, A1, L, root=0)).real for L in (1, 2)]
        assert legacy == [2.0, 4.0]


# ---------------------------------------------------------------------------
# 7. The degenerate basis: no finite tail => one exact rung, no fit
# ---------------------------------------------------------------------------

class TestDegenerateBasis:
    @staticmethod
    def _count_rungs(monkeypatch):
        calls = {"n": 0, "L": []}
        real = ds.direct_sum_zero_momentum

        def counting(edges, nu, A, L, **kw):
            calls["n"] += 1
            calls["L"].append(int(L))
            return real(edges, nu, A, L, **kw)

        monkeypatch.setattr(ds, "direct_sum_zero_momentum", counting)
        return calls

    def test_legacy_inf_three_path_is_exactly_eight(self, monkeypatch):
        """``nu = inf`` on the 3-edge path, ``root=0``, ``L_list=(2, 3, 4)``:
        the rungs are [6, 8, 8] (L = 2 clips the far vertex), the old k = 0
        fit with ``alpha = -inf`` returned their mean, 7.333; the value is
        the one rung at ``max(max(L_list), R·ecc) = max(4, 3) = 4``."""
        rungs = [complex(ds.direct_sum_zero_momentum(
            PATH3, NU_INF3, A1, L, root=0)).real for L in (2, 3, 4)]
        assert rungs == [6.0, 8.0, 8.0]
        calls = self._count_rungs(monkeypatch)
        got = complex(ds.direct_sum_extrapolated(
            PATH3, NU_INF3, A1, L_list=(2, 3, 4), root=0,
            n_correction_terms=2)).real
        assert got == 8.0
        assert calls["n"] == 1 and calls["L"] == [4]

    def test_the_default_correction_terms_do_not_refuse_the_one_rung(self, monkeypatch):
        """No fit is performed on a block with no finite tail, so the
        ladder-length check (K + 1 rungs for a K-term fit) must not run
        first.  Measured before: the 3-edge path at ``nu = inf`` with
        ``L_list = (2, 3, 4)`` and the DEFAULT ``n_correction_terms = 3``
        raised "need at least K+1=4 L values" for a value that is one
        exact rung (8.0)."""
        calls = self._count_rungs(monkeypatch)
        got = complex(ds.direct_sum_extrapolated(
            PATH3, NU_INF3, A1, L_list=(2, 3, 4), root=0)).real
        assert got == 8.0 and calls["L"] == [4]
        NN = Interaction.nearest_neighbour(A1)
        got = complex(ds.direct_sum_extrapolated(
            PATH3, NU_INF3, A1, L_list=(2, 3, 4), root=0, kernels=[NN] * 3)).real
        assert got == 8.0 and calls["L"] == [4, 4]
        # the grid twin follows the same rule
        kk = np.array([[0.0], [0.3]])
        g = ds._direct_sum_extrapolated_grid(PATH3, NU_INF3, A1, kk, root=0,
                                             terminal=3, L_list=(2, 3, 4))
        g2 = ds._direct_sum_extrapolated_grid(PATH3, NU_INF3, A1, kk, root=0,
                                              terminal=3, L_list=(2, 3, 4),
                                              n_correction_terms=2)
        assert g[0] == 8.0 and np.array_equal(g, g2)
        # anti-vacuity: a FINITE tail still needs K + 1 rungs, and an
        # empty ladder is refused on both branches
        with pytest.raises(ValueError, match=r"need at least K\+1=4"):
            ds.direct_sum_extrapolated(PATH3, np.full(3, 2.5), A1,
                                       L_list=(2, 3, 4), root=0)
        with pytest.raises(ValueError, match=r"need K\+1=4"):
            ds._direct_sum_extrapolated_grid(PATH3, np.full(3, 2.5), A1, kk,
                                             root=0, terminal=3, L_list=(2, 3, 4))
        with pytest.raises(ValueError, match="empty"):
            ds.direct_sum_extrapolated(PATH3, NU_INF3, A1, L_list=(), root=0)
        with pytest.raises(ValueError, match="empty"):
            ds._direct_sum_extrapolated_grid(PATH3, NU_INF3, A1, kk, root=0,
                                             terminal=3, L_list=())

    def test_the_reach_lifts_a_short_ladder(self, monkeypatch):
        """``max(L_list) < R·ecc``: the one rung is the REACH, not the
        top rung (which would clip: the raw L = 2 sum is 6)."""
        calls = self._count_rungs(monkeypatch)
        got = complex(ds.direct_sum_extrapolated(
            PATH3, NU_INF3, A1, L_list=(1, 2), root=0,
            n_correction_terms=1)).real
        assert got == 8.0 and calls["L"] == [3]
        assert ds._indicator_radius(A1, 4, 1) == 1
        assert ds._indicator_radius(A_TRI, 3, 2) == 1

    def test_finite_legacy_block_still_runs_the_ladder(self, monkeypatch):
        """Anti-vacuity for the call count: a finite tail fits every rung."""
        calls = self._count_rungs(monkeypatch)
        ds.direct_sum_extrapolated(PATH3, np.full(3, 2.5), A1, L_list=(2, 3, 4),
                                   root=0, n_correction_terms=2)
        assert calls["n"] == 3 and calls["L"] == [2, 3, 4]

    def test_compact_kernels_through_the_ladder(self, monkeypatch):
        """The nearest-neighbour table reproduces the legacy 8; a dyadic
        table is the exact enumeration; one rung each."""
        calls = self._count_rungs(monkeypatch)
        NN = Interaction.nearest_neighbour(A1)
        got = complex(ds.direct_sum_extrapolated(
            PATH3, NU_INF3, A1, L_list=(3, 4, 5), root=0, n_correction_terms=2,
            kernels=[NN] * 3)).real
        assert got == 8.0 and calls["n"] == 1
        D = dyadic(1)
        got = complex(ds.direct_sum_extrapolated(
            PATH3, NU_INF3, A1, L_list=(3, 4, 5, 6), root=0,
            kernels=[D] * 3)).real
        assert got == enumerate_box(PATH3, [D] * 3, A1, 6, 0)[0]
        assert calls["n"] == 2
        # a ladder below the reach is lifted to the reach (one rung, exact)
        got = complex(ds.direct_sum_extrapolated(
            PATH3, NU_INF3, A1, L_list=(2, 3, 4), root=0, n_correction_terms=2,
            kernels=[NN] * 3)).real
        assert got == 8.0 and calls["n"] == 3

    def test_reach_follows_the_compact_subgraph_not_the_whole_graph(self):
        """A power-law chord shortens the all-edge eccentricity but does
        not confine anything: the reach must follow the table edges
        (measured a silent 100 % error before this was fixed)."""
        E = np.array([[0, 1], [0, 2], [1, 2]])
        T3 = Interaction.from_table({(3,): 1.0, (-3,): 1.0})
        P = Interaction.power_law(3.5)
        nu = np.array([3.5, np.inf, np.inf])
        em = ds._collapse_multi_edges(E, nu)
        km = {(0, 1): P, (0, 2): T3, (1, 2): T3}
        assert _eccentricity(em, 0) == 1
        assert ds._compact_reach(km, em, 0) == 6                  # 0 -> 2 (3) -> 1 (3)
        got = complex(ds.direct_sum_extrapolated(
            E, nu, A1, L_list=(3, 4, 5), root=0, n_correction_terms=2,
            kernels=[P, T3, T3])).real
        exact, scale = enumerate_box(E, [P, T3, T3], A1, 6, 0)
        assert_close(got, exact, scale, rtol=1e-15)
        assert got != 0.0
        # and the finite-tail (mixed) guard uses the same reach
        T3m = Interaction.from_table({(3,): 1.0, (-3,): 1.0}, b=[1.0], nu=[3.5])
        with pytest.raises(InteractionSupportError, match="reach"):
            ds.direct_sum_extrapolated(E, np.full(3, 3.5), A1, L_list=(3, 4, 5),
                                       root=0, n_correction_terms=2,
                                       kernels=[P, T3m, T3m])


# ---------------------------------------------------------------------------
# 8. The Richardson basis with kernels present
# ---------------------------------------------------------------------------

class TestAlphaBasisWithKernels:
    def test_check_alpha_basis_never_raises_and_the_ladder_converges(self):
        """Both twins read the same float tails, so a mixed-kernel K4
        cannot desync them.  Ladder accuracy MEASURED against the
        enumeration at L = 32 (its own truncation error is
        ~32^(1-7.5) ~ 1e-10): 1.2e-8 relative for L_list = (4, ..., 8),
        against 1.2e-6 for the raw top rung — the control below."""
        M = mixed(1)
        nu = np.full(6, M.tail_exponent)
        ext = complex(ds.direct_sum_extrapolated(
            K4, nu, A1, L_list=(4, 5, 6, 7, 8), root=0, kernels=[M] * 6)).real
        want, _ = enumerate_box(K4, [M] * 6, A1, 32, 0)
        assert abs(ext - want) <= 1e-7 * abs(want)
        raw = box(K4, [M] * 6, A1, 8, 0)
        assert abs(raw - want) > 10.0 * abs(ext - want)
        # the k = 0 entry of the grid path is the same fit on the same
        # rungs (the residual transform at k = 0 is the plain sum)
        grid = ds._direct_sum_extrapolated_grid(
            K4, nu, A1, np.zeros((1, 1)), root=0, terminal=3,
            L_list=(4, 5, 6, 7, 8), kernels=[M] * 6)
        assert abs(float(grid[0]) - ext) <= 1e-13 * abs(ext)

    def test_d2_ladder(self):
        """Same on the square lattice: measured 9.9e-9 relative against
        the L = 8 enumeration, raw L = 5 rung 6.9e-7."""
        M = mixed(2)
        nu = np.full(6, M.tail_exponent)
        ext = complex(ds.direct_sum_extrapolated(
            K4, nu, A2, L_list=(2, 3, 4, 5), root=0, kernels=[M] * 6)).real
        want, _ = enumerate_box(K4, [M] * 6, A2, 8, 0)
        assert abs(ext - want) <= 1e-7 * abs(want)
        raw = box(K4, [M] * 6, A2, 5, 0)
        assert abs(raw - want) > 10.0 * abs(ext - want)
