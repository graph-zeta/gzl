# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
r"""Engine oracles for general interaction kernels — the TORUS family:
the tensor (:func:`graph_zeta_general` / ``_at_zero``), the hybrid and
the slab, driven through their ``kernels=`` keyword.  The box family
(:mod:`gzl.direct_sum`) is tested in ``tests/test_interaction_box.py``.

Every "equals" in this module has an EXACT oracle and an anti-vacuity
control, because the failure mode of a kernel plumbing is silent — a
kernel argument ignored, a cache entry served across two kernels of the
same tail, a fold over a symmetry the table does not have — all return
smooth, plausible numbers:

* **Multilinearity.**  ``ζ`` is multilinear in the per-edge kernels, so
  ``ζ(e → 0.7 K_{d+1.5} − 0.3 K_{d+2.5} + a) = 0.7 ζ(e → d+1.5)
  − 0.3 ζ(e → d+2.5) + ζ(e → a)`` holds EXACTLY on the same truncation,
  with the two power-law pieces evaluated by the untouched legacy path.
  Tolerances are relative to the SUM OF TERM MAGNITUDES, never to the
  possibly cancelling value.  The control is that the kernel call
  differs from the tail-only legacy value by a macroscopic amount.
* **Exact integer counts.**  A purely compact 0/1 table (``a(0) = 1``,
  a Chebyshev ball) makes ``ζ_G(0)`` a configuration COUNT; the torus
  reproduces the brute-force enumeration in this module exactly (``==``
  on the float) once ``n`` exceeds the wrap threshold, and is bit-equal
  at ``2n``.  The control is a grid below the threshold, where wrapped
  configurations are counted and the number changes.
* **The slab fold.**  ``use_symmetry=True`` equals ``False`` to
  round-off only because the group is filtered on the kernels' tables;
  monkeypatching the filter off is the control, and it is 2–5 % wrong.
* **Cache keys.**  Two kernels with equal tails must never share a
  ``_KERNEL_CACHE`` entry or an SP-bridge entry; the ``_sp``-level test
  shows the collision the old ``("E", nu)`` naming WOULD produce.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from gzl import Interaction, InteractionSupportError
from gzl import _sp
from gzl import slab as slab_mod
from gzl import tensor_network as tn
from gzl.hybrid import hybrid_zeta
from gzl.interaction import _KernelProduct
from gzl.slab import (
    _slab_zeta_finite_k,
    _slab_zeta_finite_k_outer,
    lattice_window_group,
    slab_zeta,
)
from gzl.tensor_network import (
    TorusTruncation,
    _edge_kernel_torus,
    _edge_kernel_torus_uncached,
    _kernel_cache_clear,
    graph_zeta_general,
    graph_zeta_general_at_zero,
)

A1 = np.array([[1.0]])
A2 = np.eye(2)
A_TRI = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])

TRIANGLE = [(0, 1), (1, 2), (0, 2)]
# A theta graph: three internally disjoint 0-1 paths (two direct edges
# and one through vertex 2), so both the parallel merge and the series
# collapse of the SP reduction run on it.
THETA = [(0, 1), (0, 1), (1, 2), (0, 2)]
K4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
C4 = [(0, 1), (1, 2), (2, 3), (0, 3)]
GRAPHS = {"triangle": TRIANGLE, "theta": THETA, "K4": K4}

#: ``d -> (A, n)`` for the multilinearity battery.
CELLS = {1: (A1, 12), 2: (A2, 8)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cross_table(d: int, val: float, origin: float = 0.0) -> dict:
    """``val`` on the 2d axis-neighbours, ``origin`` at 0 (kept when nonzero)."""
    tab = {}
    for i in range(d):
        for s in (-1, 1):
            m = [0] * d
            m[i] = s
            tab[tuple(m)] = val
    if origin != 0.0:
        tab[(0,) * d] = origin
    return tab


def ball_table(d: int, R: int) -> dict:
    """The 0/1 indicator of the Chebyshev ball of radius ``R``, origin
    INCLUDED (``a(0) = 1``), with INTEGER values for the exact oracle."""
    return {m: 1 for m in itertools.product(range(-R, R + 1), repeat=d)}


def brute_force_compact(edges, tables, d: int):
    r"""Exact ``ζ_G(0)`` for purely compact per-edge tables, by enumeration.

    Pins vertex 0 at the origin and enumerates every other vertex over
    the Chebyshev ball of radius ``dist_G(0, v) · R`` (``R`` the largest
    table radius): a configuration with a vertex outside it has, along
    the shortest path from the pin, an edge whose displacement exceeds
    every table's support, so its weight is exactly zero — the box is a
    superset of the support, not an approximation.  Weights multiply as
    the table values are given (Python ints stay ints), so a 0/1 table
    yields the integer configuration count.  Independent of every engine
    and of :class:`Interaction`.
    """
    V = max(max(e) for e in edges) + 1
    adj = {v: set() for v in range(V)}
    for u, w in edges:
        adj[u].add(w)
        adj[w].add(u)
    dist = {0: 0}
    frontier = [0]
    while frontier:
        nxt = []
        for u in frontier:
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    nxt.append(w)
        frontier = nxt
    R = max(max(abs(x) for m in t for x in m) for t in tables)
    ranges = [list(itertools.product(range(-dist[v] * R, dist[v] * R + 1),
                                     repeat=d))
              for v in range(1, V)]
    total = 0
    for pos in itertools.product(*ranges):
        x = [(0,) * d] + list(pos)
        w = 1
        for (u, v), t in zip(edges, tables):
            val = t.get(tuple(a - b for a, b in zip(x[u], x[v])), 0)
            if val == 0:
                w = 0
                break
            w *= val
        total += w
    return total


def _pieces(d: int):
    """The multilinearity ingredients at dimension ``d``: the background
    exponent, the mixed kernel on edge 0, its compact part alone, and
    the background as a genuine Interaction (a kernel list is uniform)."""
    nu0 = d + 1.5
    V = Interaction(b=[0.7, -0.3], nu=[d + 1.5, d + 2.5],
                    compact=cross_table(d, 0.25, origin=0.4))
    a = Interaction.from_table(cross_table(d, 0.25, origin=0.4))
    P0 = Interaction.power_law(nu0)
    return nu0, V, a, P0


def _nu(first, nu0, E):
    return [first] + [nu0] * (len(E) - 1)


def _kern(first, P0, E):
    return [first] + [P0] * (len(E) - 1)


def _assert_multilinear(lhs, terms, control, what, rtol=1e-13):
    """``lhs == Σ terms`` to ``rtol`` relative to ``Σ |terms|`` (a
    value-relative bound is unbounded under cancellation), and the
    anti-vacuity control: ``lhs`` is macroscopically different from
    ``control`` (the tail-only legacy value), so the kernel argument
    demonstrably reached the number."""
    lhs = np.asarray(lhs, dtype=float)
    terms = [np.asarray(t, dtype=float) for t in terms]
    rhs = sum(terms)
    scale = float(np.max(sum(np.abs(t) for t in terms)))
    assert scale > 0.0
    err = float(np.max(np.abs(lhs - rhs)))
    assert err <= rtol * scale, f"{what}: {err:.3e} > {rtol:.0e} * {scale:.3e}"
    gap = float(np.max(np.abs(lhs - np.asarray(control, dtype=float))))
    assert gap > 1e-3 * scale, (
        f"{what}: the kernel call is within {gap:.3e} of the tail-only "
        f"legacy value (scale {scale:.3e}) -- the kernel was ignored"
    )


def _k(d: int) -> np.ndarray:
    """One off-grid momentum in fractional BZ coordinates."""
    return np.array([0.3, 0.1])[:d]


# ---------------------------------------------------------------------------
# (1) The kernel cache
# ---------------------------------------------------------------------------

class TestKernelCache:
    def setup_method(self):
        _kernel_cache_clear()

    def test_legacy_key_is_byte_identical_and_the_object_is_shared(self):
        K = _edge_kernel_torus(2.5, A2, 8)
        key = (2.5, 8, A2.shape, A2.tobytes())
        assert key in tn._KERNEL_CACHE
        assert tn._KERNEL_CACHE[key] is K
        # ``interaction=None`` is the legacy call, not a new branch.
        assert _edge_kernel_torus(2.5, A2, 8, interaction=None) is K
        assert len(tn._KERNEL_CACHE) == 1
        assert np.array_equal(K, _edge_kernel_torus_uncached(2.5, A2, 8))
        assert tn._KERNEL_CACHE_BYTES == K.nbytes

    def test_an_interaction_key_never_collides_with_a_float_key(self):
        K = _edge_kernel_torus(2.5, A2, 8)
        P = _edge_kernel_torus(2.5, A2, 8,
                               interaction=Interaction.power_law(2.5))
        # Same numbers (the sampler is bit-faithful), separate entry.
        assert P is not K
        assert np.array_equal(P, K)
        assert len(tn._KERNEL_CACHE) == 2
        ikey = next(k for k in tn._KERNEL_CACHE if not isinstance(k[0], float))
        assert ikey[0][:2] == ("K", "I")
        assert ikey[1:] == (8, A2.shape, A2.tobytes())
        assert ikey[0] != 2.5

    def test_equal_tails_different_tables_are_different_entries(self):
        K = _edge_kernel_torus(2.5, A2, 8)
        Ia = Interaction.from_table(cross_table(2, 0.3), b=[1.0], nu=[2.5])
        Ib = Interaction.from_table(cross_table(2, 0.6), b=[1.0], nu=[2.5])
        assert Ia.tail_exponent == Ib.tail_exponent == 2.5
        Ka = _edge_kernel_torus(2.5, A2, 8, interaction=Ia)
        Kb = _edge_kernel_torus(2.5, A2, 8, interaction=Ib)
        assert Ka is not Kb
        assert not np.array_equal(Ka, Kb)
        # The float entry is untouched by either, and a repeat is a hit.
        assert _edge_kernel_torus(2.5, A2, 8) is K
        assert _edge_kernel_torus(2.5, A2, 8, interaction=Ia) is Ka
        assert _edge_kernel_torus(2.5, A2, 8, interaction=Ib) is Kb
        assert len(tn._KERNEL_CACHE) == 3
        assert tn._KERNEL_CACHE_BYTES == sum(
            v.nbytes for v in tn._KERNEL_CACHE.values())

    def test_kernel_arrays_are_read_only_float64_on_the_balanced_grid(self):
        Ia = Interaction.from_table(cross_table(2, 0.3, origin=0.7),
                                    b=[1.0], nu=[2.5])
        K = _edge_kernel_torus(2.5, A2, 7, interaction=Ia)
        assert K.dtype == np.float64 and K.shape == (7, 7)
        assert K.flags.writeable is False
        with pytest.raises(ValueError):
            K[0, 0] = 1.0
        z = list(tn._balanced_z_axis(7))
        assert K[z.index(0), z.index(0)] == 0.7
        assert K[z.index(1), z.index(0)] == 1.0 + 0.3
        assert K[z.index(-1), z.index(0)] == 1.0 + 0.3
        assert K[z.index(2), z.index(0)] == 1.0 / (2.0 ** 2.5)

    def test_a_product_has_its_own_key_and_samples_pointwise(self):
        Ia = Interaction.from_table(cross_table(2, 0.3), b=[1.0], nu=[2.5])
        Ib = Interaction(b=[0.5, 0.5], nu=[3.0, 4.0])
        Ka = _edge_kernel_torus(2.5, A2, 8, interaction=Ia)
        Kb = _edge_kernel_torus(3.0, A2, 8, interaction=Ib)
        prod = Ia * Ib
        assert isinstance(prod, _KernelProduct)
        Kp = _edge_kernel_torus(5.5, A2, 8, interaction=prod)
        assert np.array_equal(Kp, Ka * Kb)
        ikey = next(k for k in tn._KERNEL_CACHE if k[0][:2] == ("K", "H"))
        assert ikey[0] == ("K",) + prod.key()

    def test_the_lattice_bytes_stay_in_the_key(self):
        Ia = Interaction.from_table(cross_table(2, 0.3), b=[1.0], nu=[2.5])
        K1 = _edge_kernel_torus(2.5, A2, 8, interaction=Ia)
        K2 = _edge_kernel_torus(2.5, 0.5 * A2, 8, interaction=Ia)
        assert K1 is not K2 and not np.array_equal(K1, K2)
        assert len(tn._KERNEL_CACHE) == 2

    def test_the_window_guard_raises_before_anything_is_cached(self):
        V = Interaction.from_table({(2, 0): 0.1, (-2, 0): 0.1,
                                    (0, 2): 0.1, (0, -2): 0.1},
                                   b=[1.0], nu=[2.5])
        with pytest.raises(InteractionSupportError, match="support"):
            _edge_kernel_torus(2.5, A2, 4, interaction=V)     # holds +2, not -2
        assert len(tn._KERNEL_CACHE) == 0
        K = _edge_kernel_torus(2.5, A2, 5, interaction=V)       # (5-1)//2 = 2
        assert K.shape == (5, 5) and len(tn._KERNEL_CACHE) == 1


# ---------------------------------------------------------------------------
# (2) Multilinearity, engine by engine
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d", [1, 2])
@pytest.mark.parametrize("name", sorted(GRAPHS))
class TestMultilinearityOnTheTorus:
    r"""``ζ(e → 0.7 K_{d+1.5} − 0.3 K_{d+2.5} + a)`` against the legacy
    power-law pieces plus the purely compact piece, at fixed ``n``."""

    def test_tensor_vacuum(self, name, d):
        E, (A, n) = GRAPHS[name], CELLS[d]
        nu0, V, a, P0 = _pieces(d)
        zV = graph_zeta_general_at_zero(E, _nu(d + 1.5, nu0, E), A, n,
                                        kernels=_kern(V, P0, E))
        z1 = graph_zeta_general_at_zero(E, _nu(d + 1.5, nu0, E), A, n)
        z2 = graph_zeta_general_at_zero(E, _nu(d + 2.5, nu0, E), A, n)
        za = graph_zeta_general_at_zero(E, _nu(np.inf, nu0, E), A, n,
                                        kernels=_kern(a, P0, E))
        assert zV.imag == 0.0 and za.imag == 0.0
        _assert_multilinear(zV.real, [0.7 * z1.real, -0.3 * z2.real, za.real],
                            z1.real, f"tensor vacuum {name} d={d}")

    def test_tensor_full_bz_grid(self, name, d):
        E, (A, n) = GRAPHS[name], CELLS[d]
        nu0, V, a, P0 = _pieces(d)

        def grid(nu_first, kern=None):
            return np.real(np.asarray(graph_zeta_general(
                E, _nu(nu_first, nu0, E), A, n, source=0, terminals=(2,),
                space="k", kernels=kern)))
        gV = grid(d + 1.5, _kern(V, P0, E))
        g1, g2 = grid(d + 1.5), grid(d + 2.5)
        ga = grid(np.inf, _kern(a, P0, E))
        assert gV.shape == g1.shape
        _assert_multilinear(gV, [0.7 * g1, -0.3 * g2, ga], g1,
                            f"tensor grid {name} d={d}")

    def test_tensor_single_k(self, name, d):
        E, (A, n) = GRAPHS[name], CELLS[d]
        nu0, V, a, P0 = _pieces(d)

        def single(nu_first, kern=None):
            return float(np.real(np.asarray(graph_zeta_general(
                E, _nu(nu_first, nu0, E), A, n, source=0, terminals=(2,),
                momentum=_k(d), kernels=kern))))
        sV = single(d + 1.5, _kern(V, P0, E))
        s1, s2 = single(d + 1.5), single(d + 2.5)
        sa = single(np.inf, _kern(a, P0, E))
        _assert_multilinear(sV, [0.7 * s1, -0.3 * s2, sa], s1,
                            f"tensor single-k {name} d={d}")

    def test_hybrid_vacuum(self, name, d):
        # The purely compact piece has tail +inf, which the hybrid refuses
        # by design (its SP collapse FFTs); it comes from the tensor,
        # which evaluates the SAME torus truncation.
        E, (A, n) = GRAPHS[name], CELLS[d]
        nu0, V, a, P0 = _pieces(d)
        hV = hybrid_zeta(E, _nu(d + 1.5, nu0, E), A, n, kernels=_kern(V, P0, E))
        h1 = hybrid_zeta(E, _nu(d + 1.5, nu0, E), A, n)
        h2 = hybrid_zeta(E, _nu(d + 2.5, nu0, E), A, n)
        za = graph_zeta_general_at_zero(E, _nu(np.inf, nu0, E), A, n,
                                        kernels=_kern(a, P0, E)).real
        _assert_multilinear(hV, [0.7 * h1, -0.3 * h2, za], h1,
                            f"hybrid vacuum {name} d={d}")

    def test_hybrid_single_k(self, name, d):
        E, (A, n) = GRAPHS[name], CELLS[d]
        nu0, V, a, P0 = _pieces(d)
        k = _k(d)
        hV = hybrid_zeta(E, _nu(d + 1.5, nu0, E), A, n, source=0, terminal=2,
                         momentum=k, kernels=_kern(V, P0, E))
        h1 = hybrid_zeta(E, _nu(d + 1.5, nu0, E), A, n, source=0, terminal=2,
                         momentum=k)
        h2 = hybrid_zeta(E, _nu(d + 2.5, nu0, E), A, n, source=0, terminal=2,
                         momentum=k)
        sa = float(np.real(np.asarray(graph_zeta_general(
            E, _nu(np.inf, nu0, E), A, n, source=0, terminals=(2,),
            momentum=k, kernels=_kern(a, P0, E)))))
        _assert_multilinear(hV, [0.7 * h1, -0.3 * h2, sa], h1,
                            f"hybrid single-k {name} d={d}")

    @pytest.mark.parametrize("sym", [True, False])
    def test_slab(self, name, d, sym):
        E, (A, n) = GRAPHS[name], CELLS[d]
        nu0, V, a, P0 = _pieces(d)

        def slab(nu_first, kern=None):
            return slab_zeta(E, _nu(nu_first, nu0, E), A, n, kernels=kern,
                             use_symmetry=sym)[0]
        sV = slab(d + 1.5, _kern(V, P0, E))
        s1, s2 = slab(d + 1.5), slab(d + 2.5)
        sa = slab(np.inf, _kern(a, P0, E))
        _assert_multilinear(sV, [0.7 * s1, -0.3 * s2, sa], s1,
                            f"slab(sym={sym}) {name} d={d}")


# ---------------------------------------------------------------------------
# (3) Purely compact tables are exact integer counts on the torus
# ---------------------------------------------------------------------------

#: The grid BELOW which wrapped configurations are counted.  A wrap needs
#: a cycle whose displacement sum is a nonzero multiple of ``n`` with
#: every edge inside the ball, and the cycles that matter are a CYCLE
#: BASIS: C4's only cycle has 4 edges (wraps at ``n <= 4R``), while K4's
#: chords make every triangle exact below ``3R`` and the triangles
#: generate its cycle space, so ``n = 3R`` is the largest wrapping grid.
CONTROL_N = {"C4": lambda R: 4 * R - 1, "K4": lambda R: 3 * R}

#: The counts themselves, pinned: exact integers, so platform-free.
PINNED_COUNTS = {
    ("C4", 1, 1): 19, ("C4", 2, 1): 85, ("K4", 1, 1): 15, ("K4", 2, 1): 65,
    ("C4", 1, 2): 361, ("C4", 2, 2): 7225, ("K4", 1, 2): 225,
    ("K4", 2, 2): 4225,
}


class TestCompactTablesAreExactOnTheTorus:
    def test_the_oracle_counts_by_hand(self):
        # A bridge over the ball of radius R: (2R+1)^d configurations.
        for d in (1, 2):
            for R in (1, 2):
                assert brute_force_compact([(0, 1)], [ball_table(d, R)], d) \
                    == (2 * R + 1) ** d
        # A triangle at d = 1, R = 1: |x1| <= 1, |x2| <= 1, |x1 - x2| <= 1
        # has 7 solutions; a non-0/1 value multiplies in.
        assert brute_force_compact(TRIANGLE, [ball_table(1, 1)] * 3, 1) == 7
        weighted = {(-1,): 2, (0,): 3, (1,): 2}
        assert brute_force_compact([(0, 1)], [weighted], 1) == 7

    @pytest.mark.parametrize("d", [1, 2])
    @pytest.mark.parametrize("R", [1, 2])
    @pytest.mark.parametrize("name", ["C4", "K4"])
    def test_ball_table_count_is_exact_and_grid_independent(self, name, R, d):
        E = {"C4": C4, "K4": K4}[name]
        A = np.eye(d)
        tab = ball_table(d, R)
        I = Interaction.from_table({m: float(v) for m, v in tab.items()})
        assert I.tail_exponent == np.inf and I.support_radius == R
        count = brute_force_compact(E, [tab] * len(E), d)
        assert count == PINNED_COUNTS[(name, R, d)]
        nu_inf = [np.inf] * len(E)
        n = 4 * R + 2
        z = graph_zeta_general_at_zero(E, nu_inf, A, n, kernels=[I] * len(E))
        assert z.imag == 0.0
        assert z.real == float(count)                       # EXACT
        z2 = graph_zeta_general_at_zero(E, nu_inf, A, 2 * n,
                                        kernels=[I] * len(E))
        assert z2 == z                                      # bit-equal
        # Anti-vacuity: below the wrap threshold the torus counts wrapped
        # configurations too, and the number is DIFFERENT (larger).
        zc = graph_zeta_general_at_zero(E, nu_inf, A, CONTROL_N[name](R),
                                        kernels=[I] * len(E))
        assert zc != z and zc.real > z.real

    @pytest.mark.parametrize("name,A", [
        ("C4", A1), ("C4", A2), ("C4", 0.5 * A2), ("C4", A_TRI),
        ("C6", A1), ("C6", A2), ("C6", 0.5 * A2), ("C6", A_TRI),
        # A triangle has NN homomorphisms only on a non-bipartite lattice.
        ("triangle", A_TRI),
    ], ids=["C4-chain", "C4-square", "C4-half", "C4-triangular",
            "C6-chain", "C6-square", "C6-half", "C6-triangular",
            "triangle-triangular"])
    def test_nearest_neighbour_table_bit_equals_nu_inf(self, name, A):
        # The free oracle: the NN indicator as a genuine compact table
        # must reproduce the shipped nu = inf path bit for bit.
        E = {"C4": C4, "C6": [(i, (i + 1) % 6) for i in range(6)],
             "triangle": TRIANGLE}[name]
        NN = Interaction.nearest_neighbour(A)
        n = 8
        ref = graph_zeta_general_at_zero(E, [np.inf] * len(E), A, n)
        got = graph_zeta_general_at_zero(E, [np.inf] * len(E), A, n,
                                         kernels=[NN] * len(E))
        assert got == ref
        assert ref.real >= 1.0                               # a real count
        # And the table, not the tail, is what was evaluated: a scaled
        # indicator scales the count by J^E.
        J = 2.0
        got2 = graph_zeta_general_at_zero(E, [np.inf] * len(E), A, n,
                                          kernels=[Interaction.nearest_neighbour(A, J)] * len(E))
        assert got2.real == ref.real * J ** len(E)


# ---------------------------------------------------------------------------
# (4) The slab fold on the table-filtered group
# ---------------------------------------------------------------------------

#: Even but ANISOTROPIC on the square cell: invariant under the sign
#: flips, not under the x <-> y swap (nor the quarter turns).
ANISO = {(1, 0): 0.5, (-1, 0): 0.5, (0, 1): 0.2, (0, -1): 0.2}


class TestSlabTableFilter:
    @pytest.mark.parametrize("n", [5, 6])
    def test_the_group_is_filtered_on_the_square_cell(self, n):
        V = Interaction.from_table(ANISO, b=[1.0], nu=[2.5])
        full = lattice_window_group(A2, n)
        filt = lattice_window_group(A2, n, tables=(V.compact,))
        assert len(full) == 8
        assert len(filt) == 4
        assert all(perm == (0, 1) for perm, _ in filt)
        assert set(filt) <= set(full)
        # tables=() is the legacy call, exactly.
        assert lattice_window_group(A2, n, tables=()) == full

    @pytest.mark.parametrize("n", [5, 6])
    def test_the_filter_is_a_no_op_for_a_radial_table_on_the_triangular_cell(self, n):
        # The NN shell on the hexagonal lattice has every symmetry the
        # metric has; the filter must keep all of them.  Tested on the
        # integer table -- a generator-level array_equal filter would
        # halve this group from ulp noise in the sampled distances.
        NN = Interaction.nearest_neighbour(A_TRI)
        assert len(NN.compact) == 6
        full = lattice_window_group(A_TRI, n)
        assert len(full) == {5: 4, 6: 2}[n]
        assert lattice_window_group(A_TRI, n, tables=(NN.compact,)) == full
        # Two shells, exactly equal values on each: still every symmetry.
        second = {(1, 1): 0.1, (-1, -1): 0.1, (2, -1): 0.1, (-2, 1): 0.1,
                  (1, -2): 0.1, (-1, 2): 0.1}
        radial = Interaction.from_table({**dict(NN.compact), **second},
                                        b=[1.0], nu=[3.0])
        assert len(radial.compact) == 12
        assert lattice_window_group(A_TRI, n, tables=(radial.compact,)) == full

    @pytest.mark.parametrize("n", [5, 6])
    def test_a_from_function_radial_table_keeps_the_full_group(self, n):
        """``from_function`` samples ``V(A m)`` per label, and on the
        triangular cell the six nearest neighbours sit at ``|A m| = 1.0``
        and ``0.9999999999999999``, so ONE shell of ``exp(-|x|^2)`` holds
        two floats 0.68 ulp apart.  An exact value comparison in
        ``_table_invariant`` kept 2 of the cell's 4 window symmetries at
        n = 5 (1 of 2 at n = 6): the fold contracted twice the cells and
        every group-size gate saw half the group the kernel has.  The
        64-ulp band keeps them all, and the fold stays at round-off."""
        V = Interaction.from_function(
            lambda x: np.exp(-np.sum(x * x, axis=-1)), A_TRI, 1.05,
            b=[1.0], nu=[3.0])
        vals = sorted({v for _, v in V.compact})
        # 1.0 at the origin and TWO floats on the one shell: the ulp pair
        # this test is about is real, not hypothetical
        assert len(V.compact) == 7 and len(vals) == 3
        assert 0.0 < vals[1] - vals[0] < 2.0 * np.finfo(float).eps * vals[1]
        NN = Interaction.nearest_neighbour(A_TRI)
        shells = Interaction.from_shells(A_TRI, {1.0: np.exp(-1.0)},
                                         b=[1.0], nu=[3.0])
        full = lattice_window_group(A_TRI, n)
        assert len(full) == {5: 4, 6: 2}[n]
        assert lattice_window_group(A_TRI, n, tables=(NN.compact,)) == full
        assert lattice_window_group(A_TRI, n, tables=(shells.compact,)) == full
        assert lattice_window_group(A_TRI, n, tables=(V.compact,)) == full
        E, nu, kern = TRIANGLE, [3.0] * 3, [V] * 3
        folded, n_folded, _ = slab_zeta(E, nu, A_TRI, n, kernels=kern,
                                        use_symmetry=True)
        unfolded, n_unfolded, _ = slab_zeta(E, nu, A_TRI, n, kernels=kern,
                                            use_symmetry=False)
        assert n_folded < n_unfolded               # the fold does fold
        assert folded == pytest.approx(unfolded, rel=1e-14, abs=0.0)
        ref = graph_zeta_general_at_zero(E, nu, A_TRI, n, kernels=kern)
        assert folded == pytest.approx(ref.real, rel=1e-12, abs=0.0)

    def test_the_table_value_band_is_64_ulp_relative(self):
        """The band admits round-off and nothing else: 8 ulp passes, 128
        ulp is refused, and so is a 1e-12 "near" anisotropy (a table
        that is cubic to 1e-6 is tetragonal, exactly as the metric
        filter treats a cell)."""
        eps = np.finfo(float).eps
        assert slab_mod._TABLE_VALUE_RTOL == 64.0 * eps
        base = dict(Interaction.nearest_neighbour(A2).compact)    # 1.0 on ±e_i
        swap = ((1, 0), (1, 1))                     # x <-> y, a square symmetry
        assert slab_mod._table_invariant(tuple(base.items()), *swap, 6)
        pert = dict(base)
        pert[(0, 1)] = 1.0 + 8.0 * eps
        assert slab_mod._table_invariant(tuple(pert.items()), *swap, 6)
        pert[(0, 1)] = 1.0 + 128.0 * eps
        assert not slab_mod._table_invariant(tuple(pert.items()), *swap, 6)
        pert[(0, 1)] = 1.0 + 1e-12
        assert not slab_mod._table_invariant(tuple(pert.items()), *swap, 6)
        # a label the image misses is refused whatever the values
        assert not slab_mod._table_invariant(
            tuple(dict(ANISO).items()), *swap, 6)
        assert slab_mod._table_invariant(
            tuple(dict(ANISO).items()), (0, 1), (1, -1), 6)   # y -> -y holds

    def test_shell_tables_keep_the_full_group(self):
        # ``from_shells`` subtracts the tail at the SHELL distance, so every
        # label of a shell stores the same float and the exact table
        # filter keeps the whole metric group of the cell.
        shells = Interaction.from_shells(A_TRI, {1.0: 0.4, np.sqrt(3.0): 0.1},
                                         b=[1.0], nu=[3.0])
        vals = {v for _, v in shells.compact}
        assert len(vals) == 2                       # one float per shell
        n = 5
        full = lattice_window_group(A_TRI, n)
        filt = lattice_window_group(A_TRI, n, tables=(shells.compact,))
        assert len(full) == 4 and len(filt) == 4      # from_shells now subtracts at the shell distance
        assert filt == full
        E, nu, kern = TRIANGLE, [3.0] * 3, [shells] * 3
        folded = slab_zeta(E, nu, A_TRI, n, kernels=kern, use_symmetry=True)[0]
        unfolded = slab_zeta(E, nu, A_TRI, n, kernels=kern, use_symmetry=False)[0]
        assert folded == pytest.approx(unfolded, rel=1e-14, abs=0.0)

    @pytest.mark.parametrize("n", [5, 6])
    @pytest.mark.parametrize("name", ["triangle", "K4"])
    def test_folded_equals_unfolded_equals_tensor(self, name, n, monkeypatch):
        E = GRAPHS[name]
        V = Interaction.from_table(ANISO, b=[1.0], nu=[2.5])   # every term > 0
        kern, nu = [V] * len(E), [2.5] * len(E)
        folded = slab_zeta(E, nu, A2, n, kernels=kern, use_symmetry=True)[0]
        unfolded = slab_zeta(E, nu, A2, n, kernels=kern, use_symmetry=False)[0]
        # A sum over n^d cell values computed on symmetry-permuted kernels
        # through separate FFT paths.  Measured margin against a 1e-15
        # band: 1.1e-16 on the triangle (n = 5 and 6), 0.0 on K4 — a 9x
        # margin that np.fft build variation has eaten before (the two
        # algebra pins moved 96–224 ULP on Linux CI, and the py3.11 /
        # py3.12 runners differed by 128 ULP).  1e-14 keeps a 100x
        # margin and stays 1e11 below the percent-level anti-vacuity
        # control at the end of this test.
        assert folded == pytest.approx(unfolded, rel=1e-14, abs=0.0)
        ref = graph_zeta_general_at_zero(E, nu, A2, n, kernels=kern)
        assert ref.imag == 0.0
        assert folded == pytest.approx(ref.real, rel=1e-12, abs=0.0)
        # Anti-vacuity: with the table filter switched off the fold runs
        # on the full metric group and is wrong by percent.
        monkeypatch.setattr(slab_mod, "_table_invariant",
                            lambda *a, **k: True)
        bad = slab_zeta(E, nu, A2, n, kernels=kern, use_symmetry=True)[0]
        assert abs(bad - unfolded) > 1e-3 * abs(unfolded)

    @pytest.mark.parametrize("n", [5, 6])
    def test_finite_k_routes_fold_on_the_same_group(self, n, monkeypatch):
        # Both finite-k associations (terminal enumerated; a third vertex
        # enumerated with the terminal kept open) must use the filtered
        # group -- the outer route's stabiliser weight mult_j / |G| is
        # only right when perms and orbits agree on G.
        E = K4
        V = Interaction.from_table(ANISO, b=[1.0], nu=[2.5])
        kern, nu = [V] * len(E), [2.5] * len(E)
        k = np.array([0.3, 0.1])
        ref = float(np.real(np.asarray(graph_zeta_general(
            E, nu, A2, n, source=0, terminals=(1,), momentum=k, kernels=kern))))
        assert ref != 0.0
        for route in ("terminal", "outer"):
            def run(sym):
                if route == "terminal":
                    return _slab_zeta_finite_k(
                        E, nu, A2, n, source=0, terminal=1, momentum=k,
                        kernels=kern, use_symmetry=sym)[0]
                return _slab_zeta_finite_k_outer(
                    E, nu, A2, n, source=0, terminal=1, outer=2, momentum=k,
                    kernels=kern, use_symmetry=sym)[0]
            folded, unfolded = run(True), run(False)
            assert abs(folded - unfolded) <= 1e-12 * abs(ref), route
            assert abs(folded - ref) <= 1e-12 * abs(ref), route
            with monkeypatch.context() as m:
                m.setattr(slab_mod, "_table_invariant", lambda *a, **kw: True)
                bad = run(True)
            assert abs(bad - ref) > 1e-2 * abs(ref), route


# ---------------------------------------------------------------------------
# (5) The SP bridge cache must not be poisoned across kernels of one tail
# ---------------------------------------------------------------------------

class TestSpCachePoisoning:
    NU = 2.5
    N = 8

    def _mixed(self):
        return Interaction.from_table(cross_table(2, 0.3), b=[1.0], nu=[self.NU])

    def test_both_visit_orders_return_the_uncached_values(self):
        V = self._mixed()
        nu = [self.NU] * len(THETA)
        kern = [V] * len(THETA)
        _sp._sp_cache_clear()
        v_float = hybrid_zeta(THETA, nu, A2, self.N)
        _sp._sp_cache_clear()
        v_mixed = hybrid_zeta(THETA, nu, A2, self.N, kernels=kern)
        # The two answers are far apart, so a served-across entry shows.
        assert abs(v_float - v_mixed) > 0.1 * abs(v_float)
        # float first, then the mixed kernel of the same tail ...
        _sp._sp_cache_clear()
        assert hybrid_zeta(THETA, nu, A2, self.N) == v_float
        assert hybrid_zeta(THETA, nu, A2, self.N, kernels=kern) == v_mixed
        # ... and the opposite order.
        _sp._sp_cache_clear()
        assert hybrid_zeta(THETA, nu, A2, self.N, kernels=kern) == v_mixed
        assert hybrid_zeta(THETA, nu, A2, self.N) == v_float
        # The cache IS live on the kernel path: a repeat call hits.
        before = _sp._SP_CACHE_STATS["hit"]
        assert hybrid_zeta(THETA, nu, A2, self.N, kernels=kern) == v_mixed
        assert _sp._SP_CACHE_STATS["hit"] > before

    def test_the_old_naming_would_have_served_the_stale_bridge(self):
        # Anti-vacuity for the key choice, at the _sp level: naming the
        # mixed kernel's leaves ("E", nu) collides with the float's tree
        # and the reduction returns the FLOAT's bridge (the very object);
        # naming them ("I", key()) does not.
        V = self._mixed()
        n, d = self.N, 2
        trunc = TorusTruncation(n, d, np.eye(d))
        trunc_key = (n, d, A2.tobytes())
        K_float = _edge_kernel_torus(self.NU, A2, n)
        K_mixed = _edge_kernel_torus(self.NU, A2, n, interaction=V)
        assert not np.array_equal(K_float, K_mixed)
        E = len(THETA)
        _sp._sp_cache_clear()
        r_float = _sp.sp_reduce(THETA, [K_float] * E, 0, 0, trunc,
                                leaf_keys=[("E", self.NU)] * E,
                                trunc_key=trunc_key)
        r_poison = _sp.sp_reduce(THETA, [K_mixed] * E, 0, 0, trunc,
                                 leaf_keys=[("E", self.NU)] * E,
                                 trunc_key=trunc_key)
        _sp._sp_cache_clear()
        r_clean = _sp.sp_reduce(THETA, [K_mixed] * E, 0, 0, trunc,
                                leaf_keys=[("I", V.key())] * E,
                                trunc_key=trunc_key)
        assert r_float[1] == r_poison[1] == r_clean[1] == [(0, 2)]
        assert r_poison[2][0] is r_float[2][0]                # the stale array
        assert not np.array_equal(r_clean[2][0], r_float[2][0])
        # And the clean reduction is the uncached one, bit for bit.
        _sp._sp_cache_clear()
        r_none = _sp.sp_reduce(THETA, [K_mixed] * E, 0, 0, trunc)
        assert np.array_equal(r_clean[2][0], r_none[2][0])


# ---------------------------------------------------------------------------
# (6) Parallel edges are the lazy product, bit for bit
# ---------------------------------------------------------------------------

class TestParallelEdgesAreTheLazyProduct:
    I1 = Interaction.from_table(cross_table(2, 0.3, origin=0.2),
                                b=[1.0], nu=[2.5])
    I2 = Interaction(b=[0.5, 0.5], nu=[3.0, 4.0])

    def test_bridge(self):
        two = graph_zeta_general_at_zero([(0, 1), (0, 1)], [2.5, 3.0], A2, 8,
                                         kernels=[self.I1, self.I2])
        one = graph_zeta_general_at_zero([(0, 1)], [5.5], A2, 8,
                                         kernels=[self.I1 * self.I2])
        assert two == one
        assert one.real > 0.0 and one.imag == 0.0
        # The product is not the tail: the legacy K_5.5 bridge differs.
        assert graph_zeta_general_at_zero([(0, 1)], [5.5], A2, 8) != one

    def test_doubled_triangle_and_triple_bridge(self):
        I1, I2 = self.I1, self.I2
        two = graph_zeta_general_at_zero(
            [(0, 1), (0, 1), (1, 2), (0, 2)], [2.5, 3.0, 2.5, 2.5], A2, 8,
            kernels=[I1, I2, I1, I1])
        one = graph_zeta_general_at_zero(
            [(0, 1), (1, 2), (0, 2)], [5.5, 2.5, 2.5], A2, 8,
            kernels=[I1 * I2, I1, I1])
        assert two == one
        three = graph_zeta_general_at_zero([(0, 1)] * 3, [2.5, 3.0, 2.5], A2, 8,
                                           kernels=[I1, I2, I1])
        onep = graph_zeta_general_at_zero([(0, 1)], [8.0], A2, 8,
                                          kernels=[I1 * I2 * I1])
        assert three == onep
        assert (I1 * I2 * I1).tail_exponent == 8.0


class TestPurelyCompactIsExactThroughTheSlab:
    """A bundle with an infinite tail is contracted DENSE by the slab (the
    rule ``hybrid_zeta`` applies to the same core), so an integer table
    returns the tensor's exact count.  Measured before ``peelable`` was
    threaded into the slab's three ``_dense_core`` calls: K4 with
    nearest-neighbour tables on the triangular cell — no K4 clique there,
    so exactly 0 — came back 5.9e-17 (n = 5) and 6.3e-16 (n = 6) through
    the FFT peel while the tensor returned 0.0."""

    @pytest.mark.parametrize("n", [5, 6])
    def test_k4_nearest_neighbour_on_the_triangular_cell_is_exactly_zero(self, n):
        NN = Interaction.nearest_neighbour(A_TRI)
        nu, kern = [np.inf] * 6, [NN] * 6
        assert graph_zeta_general_at_zero(K4, nu, A_TRI, n, kernels=kern) == 0.0
        assert slab_zeta(K4, nu, A_TRI, n, kernels=kern)[0] == 0.0
        assert slab_zeta(K4, nu, A_TRI, n, kernels=kern,
                         use_symmetry=False)[0] == 0.0
        # anti-vacuity: the 0/1 Chebyshev-ball table (a(0) = 1, 9 labels)
        # on the square cell COUNTS — 225 configurations — and the slab
        # returns that integer (nearest-neighbour-only K4 is 0 there too,
        # the cell is bipartite)
        ball = Interaction.from_table(ball_table(2, 1))
        count = graph_zeta_general_at_zero(K4, nu, A2, 6, kernels=[ball] * 6)
        assert count == 225.0
        assert slab_zeta(K4, nu, A2, 6, kernels=[ball] * 6)[0] == 225.0

    def test_the_finite_k_routes_are_exact_too(self):
        NN = Interaction.nearest_neighbour(A2)
        nu, kern = [np.inf] * 4, [NN] * 4
        for k in (np.array([0.0, 0.0]), np.array([0.5, 0.0]), np.array([0.5, 0.5])):
            ref = float(np.real(np.asarray(graph_zeta_general(
                C4, nu, A2, 8, source=0, terminals=(2,), momentum=k,
                kernels=kern))))
            assert ref == int(ref)                          # an integer count
            assert _slab_zeta_finite_k(C4, nu, A2, 8, source=0, terminal=2,
                                       momentum=k, kernels=kern)[0] == ref
            assert _slab_zeta_finite_k_outer(C4, nu, A2, 8, source=0, terminal=2,
                                             outer=1, momentum=k,
                                             kernels=kern)[0] == ref

    def test_a_zero_dimensional_object_array_in_nu_names_kernels(self):
        V = Interaction.nearest_neighbour(A2)
        with pytest.raises(ValueError, match="kernels="):
            graph_zeta_general_at_zero(C4, np.array(V, dtype=object), A2, 6)

    def test_a_non_finite_table_is_refused_by_the_window_group(self):
        with pytest.raises(ValueError, match="non-finite"):
            slab_mod.lattice_window_group(A2, 6, tables=[(((1, 0), float("nan")), ((-1, 0), float("nan")))])

    @pytest.mark.parametrize("n", [5, 6])
    @pytest.mark.parametrize("k", [np.array([0.0, 0.0]), np.array([0.5, 0.0])])
    def test_the_finite_k_routes_hold_a_purely_compact_bundle_dense(self, n, k):
        # the square cell above is exact through the FFT peel anyway; the
        # triangular cell has no K4, so the count is an exact 0.0 only when
        # the purely compact bundle stays dense (peelable=False) on the
        # finite-k slab routes too
        NN = Interaction.nearest_neighbour(A_TRI)
        nu, kern = [np.inf] * 6, [NN] * 6
        assert _slab_zeta_finite_k(K4, nu, A_TRI, n, source=0, terminal=2,
                                   momentum=k, kernels=kern)[0] == 0.0
        assert _slab_zeta_finite_k_outer(K4, nu, A_TRI, n, source=0, terminal=2,
                                         outer=1, momentum=k, kernels=kern)[0] == 0.0


# ---------------------------------------------------------------------------
# Guards: the validator, the hybrid restriction window, the tensor window
# ---------------------------------------------------------------------------

class TestGuards:
    V2 = Interaction.from_table({(2, 0): 0.1, (-2, 0): 0.1,
                                 (0, 2): 0.1, (0, -2): 0.1},
                                b=[1.0], nu=[2.5])

    def test_kernels_must_align_with_the_edges(self):
        with pytest.raises(ValueError, match="one entry per edge"):
            graph_zeta_general_at_zero(TRIANGLE, [2.5] * 3, A2, 8,
                                       kernels=[self.V2] * 2)
        with pytest.raises(ValueError, match="one entry per edge"):
            hybrid_zeta(TRIANGLE, [2.5] * 3, A2, 8, kernels=[self.V2] * 4)
        with pytest.raises(ValueError, match="one entry per edge"):
            slab_zeta(TRIANGLE, [2.5] * 3, A2, 8, kernels=[self.V2] * 2)

    def test_a_float_in_the_kernel_list_is_refused(self):
        with pytest.raises(TypeError, match="not an Interaction"):
            graph_zeta_general_at_zero(TRIANGLE, [2.5] * 3, A2, 8,
                                       kernels=[self.V2, self.V2, 2.5])

    def test_the_tail_vector_must_agree_with_the_kernels_on_infinity(self):
        # A compact kernel under a finite tail would be FFT-peeled and
        # lose its integer exactness; a mixed kernel under an infinite
        # tail would be held dense.  Both are refused.
        compact = Interaction.from_table(cross_table(2, 1.0))
        with pytest.raises(ValueError, match="tail exponent"):
            graph_zeta_general_at_zero(TRIANGLE, [2.5] * 3, A2, 8,
                                       kernels=[compact] * 3)
        with pytest.raises(ValueError, match="tail exponent"):
            graph_zeta_general_at_zero(TRIANGLE, [np.inf] * 3, A2, 8,
                                       kernels=[self.V2] * 3)

    def test_a_finite_tail_must_equal_the_kernels_tail_on_every_torus_engine(self):
        """Measured before: tail 3.5 under ``nu_vec`` 2.5 or 9.0 returned
        the same 5.4399 on tensor, hybrid and slab — the plan read the
        wrong tail and nothing said so, while the SAME vector sets the
        box's Richardson basis and the router's class gates.  Refused at
        the choke point now, naming the edge; a 1e-9-relative round trip
        is accepted and bit-identical (the kernel path never reads the
        float for values)."""
        V = Interaction.from_table(cross_table(2, 0.1), b=[1.0], nu=[3.5])
        kern, good = [V] * 6, [3.5] * 6
        ok = graph_zeta_general_at_zero(K4, good, A2, 6, kernels=kern)
        assert graph_zeta_general_at_zero(
            K4, [3.5 * (1.0 + 5e-10)] * 6, A2, 6, kernels=kern) == ok
        k = np.array([0.3, 0.1])
        engines = {
            "tensor": lambda nu: graph_zeta_general_at_zero(K4, nu, A2, 6, kernels=kern),
            "tensor_k": lambda nu: graph_zeta_general(
                K4, nu, A2, 6, source=0, terminals=(1,), momentum=k, kernels=kern),
            "hybrid": lambda nu: hybrid_zeta(K4, nu, A2, 6, kernels=kern),
            "slab": lambda nu: slab_zeta(K4, nu, A2, 6, kernels=kern),
            "slab_k": lambda nu: _slab_zeta_finite_k(
                K4, nu, A2, 6, source=0, terminal=1, momentum=k, kernels=kern),
            "slab_k_outer": lambda nu: _slab_zeta_finite_k_outer(
                K4, nu, A2, 6, source=0, terminal=1, outer=2, momentum=k,
                kernels=kern),
        }
        for name, call in engines.items():
            for nu in ([2.5] * 6, [9.0] * 6, [float("nan")] * 6):
                with pytest.raises(ValueError, match="tail exponent"):
                    call(nu)
            # the edge is named: index and pair
            with pytest.raises(ValueError,
                               match=r"kernels\[5\] \(edge 5 = \(2, 3\)\)"):
                call([3.5] * 5 + [2.5])
        # a product carries the SUM of its factors' tails: the doubled
        # bridge with two tails 3.5 and the lazy product under 7.0 agree,
        # the product under a single factor's tail is refused
        two = graph_zeta_general_at_zero([(0, 1), (0, 1)], [3.5, 3.5], A2, 6,
                                         kernels=[V, V])
        one = graph_zeta_general_at_zero([(0, 1)], [7.0], A2, 6, kernels=[V * V])
        assert one == two
        with pytest.raises(ValueError, match="SUM"):
            graph_zeta_general_at_zero([(0, 1)], [3.5], A2, 6, kernels=[V * V])
        assert tn._TAIL_RTOL == 1e-9

    def test_hybrid_refuses_a_purely_compact_kernel_like_nu_inf(self):
        compact = Interaction.from_table(cross_table(2, 1.0))
        with pytest.raises(ValueError, match="nu = inf"):
            hybrid_zeta(TRIANGLE, [np.inf] * 3, A2, 8, kernels=[compact] * 3)

    def test_hybrid_refuses_to_restrict_a_table_onto_a_core_that_cannot_hold_it(self):
        with pytest.raises(InteractionSupportError, match="core grid"):
            hybrid_zeta(TRIANGLE, [2.5] * 3, A2, 8, core_n_points=4,
                        kernels=[self.V2] * 3)
        with pytest.raises(InteractionSupportError, match="core grid"):
            hybrid_zeta(K4, [2.5] * 6, A2, 8, sp_n_points=12, core_n_points=4,
                        kernels=[self.V2] * 6)
        # (5 - 1) // 2 = 2 holds the radius-2 table: accepted, finite.
        val = hybrid_zeta(TRIANGLE, [2.5] * 3, A2, 8, core_n_points=5,
                          kernels=[self.V2] * 3)
        assert np.isfinite(val) and val > 0.0

    def test_split_at_the_same_grid_is_the_identity_on_the_kernel_path(self):
        base = hybrid_zeta(K4, [2.5] * 6, A2, 8, kernels=[self.V2] * 6)
        same = hybrid_zeta(K4, [2.5] * 6, A2, 8, sp_n_points=8,
                           kernels=[self.V2] * 6)
        assert same == base

    @pytest.mark.parametrize("name", [
        "graph_zeta_general", "graph_zeta_general_at_zero", "hybrid_zeta",
        "slab_zeta", "_slab_zeta_finite_k", "_slab_zeta_finite_k_outer"])
    def test_an_interaction_in_nu_vec_names_kernels(self, name):
        """The engines accept a general kernel only through ``kernels=``;
        an Interaction in ``nu_vec`` used to die in numpy's float
        coercion (``TypeError: float() argument must be ... not
        'Interaction'``).  One helper, one message, every entry point —
        for a list entry and for a bare Interaction alike."""
        V = self.V2
        k = np.array([0.1, 0.2])
        call = {
            "graph_zeta_general": lambda nu: graph_zeta_general(
                TRIANGLE, nu, A2, 6, source=0, terminals=(1,), momentum=k),
            "graph_zeta_general_at_zero": lambda nu: graph_zeta_general_at_zero(
                TRIANGLE, nu, A2, 6),
            "hybrid_zeta": lambda nu: hybrid_zeta(TRIANGLE, nu, A2, 6),
            "slab_zeta": lambda nu: slab_zeta(TRIANGLE, nu, A2, 6),
            "_slab_zeta_finite_k": lambda nu: _slab_zeta_finite_k(
                TRIANGLE, nu, A2, 6, source=0, terminal=1, momentum=k),
            "_slab_zeta_finite_k_outer": lambda nu: _slab_zeta_finite_k_outer(
                TRIANGLE, nu, A2, 6, source=0, terminal=1, outer=2, momentum=k),
        }[name]
        with pytest.raises(ValueError, match=r"nu_vec\[1\] is an Interaction.*kernels="):
            call([2.5, V, 2.5])
        with pytest.raises(ValueError, match=r"nu_vec is an Interaction.*kernels="):
            call(V)
        with pytest.raises(ValueError, match="kernels="):
            call([V * V] * 3)
        # the spelling the message asks for runs, and so do plain floats
        assert np.all(np.isfinite(np.asarray(call([2.5] * 3))))
        with_kernels = {
            "graph_zeta_general": lambda: graph_zeta_general(
                TRIANGLE, [2.5] * 3, A2, 6, source=0, terminals=(1,), momentum=k,
                kernels=[V] * 3),
            "graph_zeta_general_at_zero": lambda: graph_zeta_general_at_zero(
                TRIANGLE, [2.5] * 3, A2, 6, kernels=[V] * 3),
            "hybrid_zeta": lambda: hybrid_zeta(TRIANGLE, [2.5] * 3, A2, 6,
                                               kernels=[V] * 3),
            "slab_zeta": lambda: slab_zeta(TRIANGLE, [2.5] * 3, A2, 6,
                                           kernels=[V] * 3),
            "_slab_zeta_finite_k": lambda: _slab_zeta_finite_k(
                TRIANGLE, [2.5] * 3, A2, 6, source=0, terminal=1, momentum=k,
                kernels=[V] * 3),
            "_slab_zeta_finite_k_outer": lambda: _slab_zeta_finite_k_outer(
                TRIANGLE, [2.5] * 3, A2, 6, source=0, terminal=1, outer=2,
                momentum=k, kernels=[V] * 3),
        }[name]
        assert np.all(np.isfinite(np.asarray(with_kernels())))

    def test_the_tensor_window_guard_propagates(self):
        with pytest.raises(InteractionSupportError, match="support"):
            graph_zeta_general_at_zero(TRIANGLE, [2.5] * 3, A2, 4,
                                       kernels=[self.V2] * 3)
        with pytest.raises(InteractionSupportError, match="support"):
            slab_zeta(TRIANGLE, [2.5] * 3, A2, 4, kernels=[self.V2] * 3)
        ok = graph_zeta_general_at_zero(TRIANGLE, [2.5] * 3, A2, 5,
                                        kernels=[self.V2] * 3)
        assert ok.real > 0.0
