# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""The slab: a dense core contracted with a second pin.

It is not a third engine beside the torus core (:mod:`gzl.hybrid`) and
the real-space box (:mod:`gzl.direct_sum`).  It evaluates the vacuum
(``k = 0``) value of one dense block.

**The mechanism is one sentence:** the slab is the ordinary dense core
with a SECOND pin.  Pinning the source already turns its incident edges
into one-axis potentials (``hybrid._dense_core`` does this); fixing one
free vertex at an explicit lattice site does the same to *its* incident
edges, removing one axis from every bag it touched.  The post-peel
exponent therefore drops by one — ``n**(3d)`` becomes ``n**(2d)`` — and
an outer sum over that vertex's ``n**d`` sites restores the exact value.

Total work is unchanged to leading order: ``n**d`` outer cells times an
inner contraction a factor ``n**d`` cheaper.  What changes is peak
memory, and on a cubic cell the ``2**d d!``-fold signed-permutation
orbit reduction of the outer loop buys wall-clock on top.

Provenance
----------
This code began as an analysis tool built to give the exponent-3
d = 3 cores a second family of values at grids neither shipped engine
can reach.  Its validation, the slab against
``graph_zeta_general_at_zero`` at d = 1, 2, 3, is a gated test in
``tests/test_slab.py``.

What it is for in the library
-----------------------------
It is not a general-purpose engine and is not routed to on cost.  It
occupies exactly one slot: the k = 0 dense blocks that
``frontend._TORUS_MIN_N_BY_CLASS`` DECLINES, which before it existed
went to the box.  Measured on K5 at d = 3, ν = 3.5, the slab at n = 12
against a box at ``L = (2, 3, 4)``:

    engine        rel error    seconds   peak MiB
    box           6.352e-08     ~85       ~2083
    slab n = 12   1.344e-08      15.2       207

i.e. **4.76x accuracy** (strict resolved bound, band 3.48e-10, so both
sides are >= 37 band-widths out), 5.6x speed and 10.07x memory, all
three in the same direction.

The error column is against a slab reference at n = 16.  The memory
column was measured once and has not been re-measured since.

Two limits are load-bearing and are enforced by the router, not here:

* **The win is a COST CLASS, not every declined block.**  V6E11 -- the
  other d = 3 exponent-3 core in the shipped corpora -- is **0.234x**
  the box on its slab at n = 12 (strict resolved bound; band 1.235e-05,
  box 5.9 band-widths out, slab 20.2, so the loss is resolved).  It
  never reaches this engine, because the SP split acts on it and the
  accuracy gate therefore admits it to the torus; that is the only
  reason the routing is safe, so the arm must stay behind the decline
  rather than becoming a cost route.  At corpus scale the same thing
  happens to 24 of the 30 free-pin exponent-3 shapes.
* **The speed rests on the orbit reduction, and degrades with it.**
  The reduction is a property of the CELL, and
  :func:`lattice_window_group` finds whatever it admits: at n = 12 it
  folds 1728 outer cells onto 84 (cubic), 196 (tetragonal), 343
  (orthorhombic) or 1728 (a sheared cell at even n, i.e. no reduction).
  So the same slab costs 1x, 2.3x, 4.1x or 20.6x the cubic wall-clock
  depending on the cell, while its VALUE is exact on all of them.  The
  router declines the arm below ``frontend._SLAB_MIN_GROUP`` for that
  reason and that reason only -- cost, never correctness.
"""

from __future__ import annotations

import numpy as np

from gzl import _elimination
from gzl._lattices import _resolve_lattice
from gzl.tensor_network import (
    _check_kernels,
    _refuse_interaction_in_nu,
    _edge_kernel_torus,
    _label_distances,
    _min_degree_order,
    _pin_from_generator_at,
)


class SlabTooLargeError(MemoryError):
    r"""The modelled inner peak exceeds the caller's byte ceiling.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    A MemoryError subclass so the frontend's existing box fallbacks
    catch it exactly as they catch ``HybridCoreBudgetError`` — a
    refusal here must land on the incumbent engine, never on an
    uncapped one.
    """


# ---------------------------------------------------------------------------
# Point-group orbits of the outer loop
# ---------------------------------------------------------------------------

#: Relative tolerance on table VALUES in :func:`_table_invariant`: 64 ulp
#: (``64 * np.finfo(float).eps``, absolute tolerance zero).  Exact
#: comparison threw away symmetries a radial table HAS:
#: :meth:`Interaction.from_function` samples ``V(A m)`` per label, and on
#: the triangular cell ``|A (1, 0)| = 1.0`` while ``|A (0, 1)| =
#: 0.9999999999999999``, so one shell of ``exp(-|x|^2)`` stores
#: 0.36787944117144233 at ``(±1, 0)`` and 0.3678794411714424 on its four
#: other sites — 0.68 ulp apart — and the exact filter kept 2 of the
#: cell's 4 window symmetries at n = 5 (1 of 2 at n = 6): the fold
#: contracted twice the cells, and any gate keyed on the group size
#: (``frontend._SLAB_MIN_GROUP``, the fk-outer pricing) saw half the
#: group the kernel actually has.  Under this band the fold's error is
#: bounded by 64 eps times the table's own contribution — round-off, the
#: class the FFT peel already lives in — while a table that is NOT
#: symmetric differs by a macroscopic amount on some site (folding on it
#: anyway was measured 5–55 % wrong), so nothing real is admitted.  64
#: ulp is the repo's own "same number, different build" bound
#: (``tests/test_executor_goldens.py``: 8x above the measured
#: cross-platform np.fft noise, 13x below the smallest real defect).
_TABLE_VALUE_RTOL = 64.0 * float(np.finfo(float).eps)


def _table_invariant(table, perm, signs, n: int,
                     rtol: float = _TABLE_VALUE_RTOL) -> bool:
    r"""Is a compact table mapped onto itself by one signed permutation,
    read as the index action of :func:`lattice_window_group`?

    ``table`` is a sequence of ``((m_1, ..., m_d), value)`` pairs over
    integer lattice labels (``Interaction.compact``).  The candidate acts
    on torus INDICES, ``sigma(w)_i = signs[i] * w[perm[i]] mod n``; on
    the balanced label carried by an index that is
    ``m'_i = bal(signs[i] * m[perm[i]] mod n)`` — the wrap is kept in so
    the test is literally the array action, not a label-space model of
    it.  The condition for the kernel array to satisfy
    ``K[sigma(w)] == K[w]`` on its compact part is that the image of
    every ``(label, value)`` pair is again a pair of the table (sigma is
    a bijection of the finite label set, so mapping the support into
    itself is mapping it onto itself), the labels matched exactly and
    the values to ``rtol`` relative with zero absolute tolerance
    (:data:`_TABLE_VALUE_RTOL`, and the reason a band is right here).

    On the INTEGER table only, never on the sampled float generator:
    an ``array_equal`` filter on the generators halves the group on the
    triangular cell from 2e-16 ulp noise in ``_label_distances``, i.e.
    it throws away symmetries that hold — the opposite failure of the
    metric algebra, and just as silent.  The value band above is the
    same lesson one level up: the noise reaches the table too, through
    ``from_function``'s per-label sampling.
    """
    n = int(n)
    half = n // 2
    rtol = float(rtol)
    entries = {tuple(int(x) for x in m): float(v) for m, v in table}
    for m, v in entries.items():
        img = []
        for pj, sg in zip(perm, signs):
            r = (int(sg) * m[int(pj)]) % n
            img.append(r if r <= half else r - n)
        w = entries.get(tuple(img))
        if w is None:
            return False
        # ``not (... <= ...)`` so a NaN value never passes.
        if not (abs(w - v) <= rtol * max(abs(v), abs(w))):
            return False
    return True


def lattice_window_group(A, n: int, rtol: float = 1e-13, *, tables=()):
    r"""Index maps that leave the torus kernel invariant, as ``(perm, signs)``.

    A candidate is a signed permutation acting on TORUS INDICES,
    ``sigma(w)_i = signs[i] * w[perm[i]] mod n``, and it is admitted
    only if it preserves the array the kernels are literally built
    from — ``tensor_network._label_distances(A, n)``.  Every kernel is
    a function of that array alone, and sigma is linear on residues, so
    ``dist[sigma(w)] == dist[w]`` for every displacement ``w`` is
    exactly the condition under which relabelling every summation
    variable by sigma leaves the contraction unchanged.

    **Derived by testing, not by algebra, and that is not fastidiousness
    — the algebra is wrong.**  The obvious analytic condition is
    ``Pᵀ G P = G`` for ``G = Aᵀ A``, i.e. "sigma preserves the metric",
    which admits ``-I`` on every lattice because ``K`` is even.  It is
    false at even ``n`` on any cell with a cross term.  The balanced
    axis at ``n = 6`` is ``[0, 1, 2, 3, -2, -1]``: label ``3`` is its
    own negative as a residue but has no partner ``-3`` in the window,
    so on the triangular cell

        |A (3, 1)|**2 = 13   while   |A (3, -1)|**2 = 7

    and the two cells the reflection claims to identify are not equal.
    Measured cost of trusting the algebra: the prism at d = 2, n = 6 on
    a triangular cell came out 4.23e-03 wrong — 6 of 15 orbits
    non-constant — while the unreduced sum was exact to 1.7e-16.  A
    smooth, plausible, entirely wrong number, which is the failure mode
    a symmetry argument produces when its window assumption is silent.

    Testing against ``dist`` cannot make that mistake: it sees the
    window, because ``dist`` IS the window.

    Sizes it finds (d = 3, cubic ``A``): 48 at odd ``n``, 48 at even
    ``n`` too, since a diagonal isotropic metric has no cross term for
    the ``n/2`` label to expose.  On a triangular d = 2 cell it finds 4
    at odd ``n`` and 2 at even ``n``, and the missing 2 are exactly the
    reflections the algebra would have wrongly kept.

    ``rtol`` is deliberately tight.  Accepting a NEAR-symmetry returns a
    smooth wrong number of precisely the kind above, so a cell that is
    cubic to 1e-6 is treated as tetragonal, which is what it is.

    ``tables`` (keyword-only, default empty) are the compact tables of
    the general kernels the contraction carries — each a sequence of
    ``((m_1, ..., m_d), value)`` pairs over integer lattice labels, i.e.
    ``Interaction.compact``.  A candidate is kept only if it ALSO maps
    every table onto itself exactly (:func:`_table_invariant`): the
    distance test covers the power-law part of every kernel, but an
    anisotropic table breaks symmetries the metric has, and folding on
    them was measured 5–55 % wrong.  Tested on the integer tables, never
    on the sampled generators (see :func:`_table_invariant` for why).
    With no tables the function is unchanged.
    """
    from itertools import permutations, product

    A = np.asarray(A, dtype=float)
    d = int(A.shape[0])
    n = int(n)
    tabs = [tuple(t) for t in tables]
    for t in tabs:
        if any(not np.isfinite(float(v)) for _, v in t):
            # a NaN fails the invariance test for the identity itself and
            # would leave an EMPTY group to fold over
            raise ValueError("lattice_window_group: a compact table carries a non-finite value")
    dist = _label_distances(A, n)
    coords = np.indices((n,) * d)
    out = []
    for perm in permutations(range(d)):
        for signs in product((1, -1), repeat=d):
            src = tuple((int(sg) * coords[int(pj)]) % n
                        for pj, sg in zip(perm, signs))
            if np.allclose(dist[src], dist, rtol=float(rtol), atol=0.0):
                if all(_table_invariant(t, perm, signs, n) for t in tabs):
                    out.append((tuple(perm), tuple(signs)))
    if not out:
        # The identity always preserves both the metric and every table,
        # so an empty group means a malformed table reached the filter.
        # Refuse here: ``orbit_reps`` would fold over nothing and raise an
        # opaque AttributeError instead, one call further from the cause.
        raise ValueError(
            "lattice_window_group: no symmetry survived the table filter, "
            "not even the identity -- a compact table is malformed"
        )
    return out


def _compact_tables(kernels) -> tuple:
    r"""The DISTINCT compact tables a per-edge kernel list carries, lazy
    products walked down to their factors, in first-seen order; ``()``
    for the legacy path (``kernels is None``) or for pure power laws.
    This is what :func:`lattice_window_group` filters the outer loop's
    group on.
    """
    if kernels is None:
        return ()
    out: list = []
    for kern in kernels:
        for f in getattr(kern, "factors", (kern,)):
            tab = tuple(getattr(f, "compact", ()))
            if tab and tab not in out:
                out.append(tab)
    return tuple(out)


def orbit_reps(n: int, d: int, A=None, return_inverse=False, *, group=None):
    r"""Representatives and multiplicities of the outer loop's orbits.

    The group is :func:`lattice_window_group` acting on torus INDICES.
    ``A = None`` means the cubic cell.  ``group`` (keyword-only) supplies
    a precomputed group instead — the table-filtered one when general
    kernels carry compact parts, so every fold site and the stabiliser
    arithmetic of the finite-k outer route act on ONE group.

    Orbits are computed by CANONICALISATION: each index's key is the
    smallest flat index it maps to under the group.  Because the group
    is closed and acts on indices (not on integer labels), an index and
    its image always both exist — which is the second thing the
    label-space version got wrong, since ``-3`` is not a label at
    ``n = 6`` and its orbit silently split.

    Multiplicities are COUNTED, never derived from a formula, so a wrong
    combinatorial special case cannot slip through.  The reduction
    applies to the OUTER loop only, so it is exact for any inner
    contraction, and ``slab_zeta``'s controls check the reduced sum
    against the unreduced one rather than trusting that.
    """
    n, d = int(n), int(d)
    if group is None:
        group = lattice_window_group(np.eye(d) if A is None else A, n)
    else:
        group = list(group)
    coords = np.indices((n,) * d)
    flat = np.arange(n ** d, dtype=np.int64).reshape((n,) * d)
    best = None
    for perm, signs in group:
        src = tuple((int(sg) * coords[int(pj)]) % n
                    for pj, sg in zip(perm, signs))
        img = flat[src]
        best = img if best is None else np.minimum(best, img)
    _, first, inverse, counts = np.unique(
        best.reshape(-1), return_index=True, return_inverse=True,
        return_counts=True)
    if return_inverse:
        return (first.astype(np.int64), counts.astype(np.float64),
                inverse.astype(np.int64))
    return first.astype(np.int64), counts.astype(np.float64)


# ---------------------------------------------------------------------------
# Symbolic schedule and price
# ---------------------------------------------------------------------------

def _inner_scopes(edges, source: int, slab: int):
    """Factor scopes of the inner contraction, as ``(scope, is_kernel)``.

    Both pinned vertices collapse their incident kernels to one-axis
    potentials, which is the whole mechanism.  Built here rather than
    via ``_elimination.scopes_from_edges`` because that helper knows
    about ONE pin; feeding it an inner vertex instead removes that
    vertex's axis from every bag and under-reports the exponent by
    exactly one — the same size as the effect being priced.
    """
    pinned = {int(source), int(slab)}
    scopes: list[tuple[frozenset, bool]] = []
    for u, v in edges:
        u, v = int(u), int(v)
        au, av = u in pinned, v in pinned
        if au and av:
            continue                       # a scalar, no axis
        if au or av:
            scopes.append((frozenset({v if au else u}), False))
        else:
            scopes.append((frozenset({u, v}), True))
    return scopes


def slab_schedule(edges, source: int, slab: int):
    r"""``(order, steps, exponent_inner)`` for one choice of slab vertex.

    ``exponent_inner`` is in units of ``n**d`` and is one lower than the
    full core's — that is what makes the slab affordable.

    The order is min-degree rather than
    :func:`gzl._elimination.plan`'s exact DP.  Correctness does
    not depend on it (a finite sum reassociated on the torus is exact),
    and min-degree cannot be refused on budget the way the pin-searching
    DP can on a wide core.
    """
    edges = [(int(u), int(v)) for u, v in np.asarray(edges).tolist()]
    pinned = {int(source), int(slab)}
    nodes = sorted({v for e in edges for v in e})
    inner = [v for v in nodes if v not in pinned]
    adj = {v: set() for v in inner}
    for u, v in edges:
        if u in pinned or v in pinned:
            continue
        adj[u].add(v)
        adj[v].add(u)
    order = _min_degree_order(adj, inner, terminals=set())
    steps = _elimination.simulate(_inner_scopes(edges, source, slab), order)
    return order, steps, max((s.exponent for s in steps), default=0)


def choose_slab(edges, d: int, *, source: int = 0):
    r"""Pick the free vertex whose removal drops the exponent furthest.

    Degree is the obvious heuristic and it is WRONG on a shape that
    matters.  V6E11 (K5 with one edge subdivided) has four degree-4 free
    vertices; removing the first of them leaves a K4 on the rest and the
    inner exponent stays at 3, buying nothing.  Removing a different one
    leaves a 4-vertex graph missing an edge, and the exponent drops to 2
    — a factor ``n**d`` in peak memory.

    So the choice is priced, not guessed.  The analysis tool this code
    began as probed each candidate by RUNNING one real cell and
    recording what the executor did; :func:`slab_schedule` reproduces
    that measurement symbolically on all six census cores at
    d = 1, 2, 3 for every candidate slab (0 mismatches), which is what
    lets the router price a slab before allocating one.
    ``tests/test_slab.py::TestThePriceIsTheMeasurement`` re-runs that
    comparison.

    Ties break on the lower vertex index, so the choice is deterministic
    and the block cache cannot see two values for one block.
    """
    nodes = sorted({int(v) for e in np.asarray(edges).tolist() for v in e})
    best, best_exp = None, None
    for cand in [v for v in nodes if v != int(source)]:
        try:
            _, _, ex = slab_schedule(edges, source, cand)
        except Exception:
            continue
        if ex <= 0:
            continue
        if best_exp is None or ex < best_exp:
            best, best_exp = cand, ex
    if best is None:                       # nothing priced; fall back
        deg = {v: sum(1 for e in np.asarray(edges).tolist() if v in e)
               for v in nodes if v != int(source)}
        if not deg:
            raise ValueError("no free vertex to slab")
        return max(deg, key=lambda v: (deg[v], -v))
    return best


def _outer_loop_bytes(n: int, d: int) -> float:
    r"""Bytes the OUTER loop's setup holds, as a floor on the price.

    The inner contraction dominates on any block the router sends here,
    so this is invisible in practice -- but it is what stands between a
    caller and an ignored ``max_bytes`` on a degenerate one.  A block
    whose only vertices are the source and the slab has no inner vertex
    to eliminate: ``slab_schedule`` returns an empty step list,
    ``hybrid._core_peak_bytes`` loops zero times and returns exactly
    0.0, and ``need > max_bytes`` is then False for EVERY budget,
    including 0.  ``slab_zeta`` went on to build ``orbit_reps``' index
    arrays regardless.

    Priced from the arrays that actually exist rather than guessed:
    ``np.indices((n,)*d)`` is ``d`` int64 planes, ``flat`` is one, and
    ``_label_distances`` holds ``z_labels`` and ``phys`` at ``d`` float
    planes each plus the norm -- so ``4d + 3`` planes of 8 bytes per
    site.  MEASURED against ``ru_maxrss`` at d = 3: 112.5 bytes/site at
    n = 60, 91.3 at n = 100, 88.3 at n = 150, against the 120 this
    returns.  It OVER-prices by 7-35%, which is the safe direction: an
    over-priced slab is refused and falls back to the box, while an
    under-priced one is admitted and allocates.
    """
    return 8.0 * float(4 * int(d) + 3) * float(int(n) ** int(d))


def slab_peak_bytes(edges, source: int, slab: int, n: int, d: int) -> float:
    r"""Modelled peak bytes of the INNER contraction.

    Deliberately :func:`gzl.hybrid._core_peak_bytes` on the slab's
    own steps rather than a second cost model — the executor, the
    truncation and the chunked peel are all shared with the dense core,
    so a separate model here could only drift.  Validated against
    measured peak RSS on the K5 d = 3 core at n = 12: 113.9 MiB modelled
    against 104.2 MiB measured (+9% headroom, the same sign and rough
    size as the dense-core rows that model records).

    The OUTER loop is not priced: it holds one cell's factor list and
    a scalar accumulator, both ``O(n**d)``.
    """
    from gzl.hybrid import _core_peak_bytes

    _, steps, _ = slab_schedule(edges, source, slab)
    inner = float(_core_peak_bytes(steps, int(n), int(d)))
    return max(inner, _outer_loop_bytes(int(n), int(d)))


# ---------------------------------------------------------------------------
# The contraction
# ---------------------------------------------------------------------------

def _edge_value_both_pinned(K, ia: int, ib: int) -> float:
    """``K[(x_a - x_b) mod n]`` with both endpoints at explicit sites."""
    return float(_pin_from_generator_at(K, ib, pin_row=False)[ia])


def _edge_kernel_arrays(nu_vec, A, n, kernels=None):
    r"""One torus kernel array per edge.

    Legacy path (``kernels is None``): one build per DISTINCT exponent,
    the dict comprehension every slab entry point carried inline, moved
    here expression for expression.  General kernels: one build per
    distinct ``kernel.key()`` — never per tail, since two kernels sharing
    a tail are different arrays — through the same global cache.
    """
    if kernels is None:
        by_nu = {nu: _edge_kernel_torus(nu, A, n) for nu in set(nu_vec)}
        return [by_nu[nu] for nu in nu_vec]
    by_key: dict = {}
    out = []
    for nu, kern in zip(nu_vec, kernels):
        ck = kern.key()
        if ck not in by_key:
            by_key[ck] = _edge_kernel_torus(float(nu), A, n, interaction=kern)
        out.append(by_key[ck])
    return out


def slab_zeta(edges, nu_vec, A, n, *, source=0, slab=None,
              use_symmetry=True, max_bytes=None, progress=None,
              _max_cells=None, kernels=None):
    r"""Vacuum ``ζ_G(0)`` of one block, with one free vertex slabbed out.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Returns ``(value, n_cells, exponent_inner)``.  The value is the same
    torus-truncated lattice sum ``hybrid_zeta`` and ``graph_zeta_general``
    compute at the same ``n``, merely associated differently, so it is
    NOT an independent truncation — only a cheaper association of the
    same one.

    ``max_bytes`` refuses with :class:`SlabTooLargeError` BEFORE any
    table is allocated, priced by :func:`slab_peak_bytes`.  ``None``
    (the default) is no cap; the router always passes one.

    VERTEX LABELS ARE THE ONES PRESENT IN ``edges``, not ``0..max``:
    the vertex set is the edge support, and since the label-convention
    unification (see gzl/_labels.py) that is the convention of
    EVERY public edge-list entry point, not this engine's private one.
    A label that appears in no edge is not a vertex.  Historically
    ``graph_zeta_general_at_zero`` summed every gap label freely over
    the torus (a factor ``n**d`` each — exactly ``36**6`` on the K4 on
    ``{0, 5, 7, 9}`` at d = 2, n = 6) and the box multiplied in
    ``(2L+1)**d`` per gap; labels are now normalised at every entry,
    and the engines agree bit-for-bit on a sparsely-labelled graph
    because the compression is order-preserving.

    ``use_symmetry`` folds the outer loop onto the orbits of
    :func:`lattice_window_group`, the signed permutations that preserve
    both the cell's metric and the coordinate truncation window.  It
    adapts to the cell rather than requiring one: 48-fold on a cubic
    cell, 16 on a tetragonal one, and 2 or 1 on a fully generic one --
    2 at odd ``n``, but only the identity at even ``n``, where the
    ``n/2`` label has no partner and the reflection stops being a
    symmetry of the window.  It is never
    an approximation — ``use_symmetry=False`` is for controls, not for
    correctness.

    ``kernels`` (default ``None``) supplies GENERAL per-edge kernels —
    :class:`gzl.interaction.Interaction` objects or lazy products
    of them, aligned with ``edges``; ``nu_vec`` then carries each
    kernel's TAIL exponent.  Kernel arrays are built per distinct
    ``kernel.key()``, and the orbit fold uses the group
    :func:`lattice_window_group` finds with the kernels' compact TABLES
    passed in, so an anisotropic table folds only on the symmetries it
    actually has (unfiltered, measured 5–55 % wrong).  A purely compact
    bundle — ``nu_vec`` carrying ``+inf`` on some edge — is contracted
    DENSE (no FFT peel, the rule :func:`~gzl.hybrid.hybrid_zeta`
    applies), so an integer table returns the exact count the tensor
    returns; the finite edges of such a list are held dense too, a cost
    the router never pays because it never sends an infinite tail here.
    ``None`` is the legacy path, byte-identical.

    ``A`` is the lattice matrix or a lattice name -- ``"chain"``,
    ``"square"``, ``"triangular"``, ``"cubic"``.
    """
    edges = [(int(u), int(v)) for u, v in np.asarray(edges).tolist()]
    _refuse_interaction_in_nu(nu_vec, "slab_zeta")
    nu_vec = [float(x) for x in np.asarray(nu_vec).reshape(-1).tolist()]
    if len(nu_vec) != len(edges):
        raise ValueError("nu_vec must have one entry per edge")
    kernels = _check_kernels(kernels, nu_vec, len(edges), edges)
    # A purely compact bundle (tail +inf on some edge) is held dense so
    # the contraction keeps the exact (integer) arithmetic the tensor
    # guarantees it — the rule hybrid_zeta applies to the same core.
    # Measured before this was threaded: K4 with nearest-neighbour
    # tables on the triangular cell (no K4 clique there, so exactly 0)
    # came back 5.9e-17 at n = 5 and 6.3e-16 at n = 6 through the FFT
    # peel, while the tensor returned 0.0.  Cost-only on a mixed list
    # (the finite edges are held dense too, as in hybrid); the router
    # never sends an infinite tail here, so a direct call is the only
    # way to pay it.  The symbolic price (slab_schedule / max_bytes) is
    # the peeled one; a dense inf contraction is under-priced by it.
    peelable = not bool(np.isinf(np.asarray(nu_vec, dtype=float)).any())
    A = np.asarray(_resolve_lattice(A), dtype=float)
    d = int(A.shape[0])
    n = int(n)
    nodes = sorted({v for e in edges for v in e})
    source = int(source)
    if any(u == v for u, v in edges):
        raise ValueError("the slab engine does not accept self-loops")
    if source not in nodes:
        raise ValueError(f"source {source} is not a vertex of the block")

    if slab is None:
        slab = choose_slab(edges, d, source=source)
    slab = int(slab)
    if slab == source:
        raise ValueError("the slab vertex must not be the source")
    if slab not in nodes:
        raise ValueError(f"slab vertex {slab} is not a vertex of the block")

    order, _steps, exponent_inner = slab_schedule(edges, source, slab)
    if max_bytes is not None:
        # `need > nan` is False, so a NaN budget silently disabled the
        # cap entirely.  Reject anything that is not a real, usable
        # number rather than treating it as "no limit" -- the caller
        # passed a budget, so they wanted one enforced.
        _budget = float(max_bytes)
        if not np.isfinite(_budget) or _budget < 0.0:
            raise ValueError(
                f"max_bytes must be a finite, non-negative byte count "
                f"(use None for no limit); got {max_bytes!r}"
            )
        need = slab_peak_bytes(edges, source, slab, n, d)
        if need > _budget:
            raise SlabTooLargeError(
                f"slab inner contraction needs {need / 1024**3:.2f} GiB "
                f"at n={n}, d={d} (exponent {exponent_inner}); "
                f"budget is {_budget / 1024**3:.2f} GiB"
            )

    kernel_list = _edge_kernel_arrays(nu_vec, A, n, kernels)

    if use_symmetry:
        # General kernels fold on the TABLE-filtered group; the legacy
        # call lets orbit_reps find the group itself, as it always has.
        group = (None if kernels is None else
                 lattice_window_group(A, n, tables=_compact_tables(kernels)))
        reps, mult = orbit_reps(n, d, A, group=group)
    else:
        reps = np.arange(n ** d, dtype=np.int64)
        mult = np.ones(n ** d, dtype=float)

    # A COUNT, never a normalisation: a mutated tiling can still sum to
    # 1.0 after rescaling, but it cannot fake the site count.
    if int(round(float(mult.sum()))) != n ** d:
        raise AssertionError(
            f"orbit multiplicities sum to {mult.sum()}, not {n ** d}"
        )

    # THE INNER CONTRACTION IS `hybrid._dense_core` WITH TWO PINS.  This
    # wrapper owns only what the core cannot: the outer sum over the slab
    # vertex's sites, its orbit fold, and the cell bookkeeping (the
    # zero-kernel skip, the contracted-cell count the probe mode reads).
    # `order` is threaded through so the core contracts the exact
    # schedule `slab_schedule` priced, which is what keeps the shipped
    # numbers bit-identical rather than merely equal.
    from gzl.hybrid import _dense_core

    node_list = list(nodes)
    pinset = {source, slab}
    bp_edges = [i for i, (a, b) in enumerate(edges)
                if a in pinset and b in pinset]
    has_free_factor = any(a not in pinset or b not in pinset
                          for a, b in edges)

    total = 0.0
    n_contracted = 0
    for cell, (z, w) in enumerate(zip(reps.tolist(), mult.tolist())):
        # The zero-kernel skip, and the all-pinned fast path, stay OUT of
        # the core: the first avoids contracting a cell whose value is
        # known to be 0 (K(0) = 0 on a coincident pin), the second is a
        # cell with no free factor at all -- and neither counts as a
        # contracted cell for the probe mode.  The scalar is rebuilt
        # inside the core in the same edge order, so the bits agree.
        scalar = 1.0
        for i in bp_edges:
            a, b = edges[i]
            ia = 0 if a == source else z
            ib = 0 if b == source else z
            val = _edge_value_both_pinned(kernel_list[i], ia, ib)
            if val == 0.0:
                scalar = 0.0
                break
            scalar *= val
        if scalar == 0.0:
            continue
        if not has_free_factor:
            total += w * scalar
            continue

        M = _dense_core(
            node_list, edges, kernel_list,
            ((source, 0), (slab, z)), source, n, d, A,
            order=order, peelable=peelable,
        )
        total += w * float(M)
        n_contracted += 1

        if _max_cells is not None and n_contracted >= int(_max_cells):
            # PROBE MODE.  The returned value is a PARTIAL sum and is
            # meaningless; only the schedule metadata is.
            return float("nan"), int(len(reps)), exponent_inner
        if progress and cell % progress == 0:
            print(f"    cell {cell}/{len(reps)}", flush=True)

    return float(total), int(len(reps)), exponent_inner


# ---------------------------------------------------------------------------
# The second pin at finite k
# ---------------------------------------------------------------------------

def _slab_M_dense(edges, nu_vec, A, n, *, source, terminal,
                  use_symmetry=True, max_bytes=None, kernels=None):
    r"""Dense terminal-position function ``M[z]`` via the second pin.

    ``M[z]`` is the vacuum two-pin cell value with the SOURCE at the
    origin and the TERMINAL at cell ``z`` — i.e. exactly the summand of
    :func:`slab_zeta` with the slab vertex FORCED to be the terminal —
    returned as the full ``(n,)*d`` array in the balanced-label layout
    every torus kernel uses.  ``ζ_G(k) = Σ_z cos(2π k·x(z)) M[z]``
    (real by lattice inversion) and the full BZ grid is
    ``np.real(np.fft.fftn(M))`` — the same two consumers
    :func:`gzl.hybrid.hybrid_zeta` applies to its own ``M``,
    validated against it to round-off on shared inputs.

    The orbit fold applies to ``M`` itself: cell values are constant on
    :func:`lattice_window_group` orbits (the same invariance the vacuum
    slab rests on), so representatives are contracted once and broadcast
    through the inverse map.  The k-phase is NOT orbit-constant, which
    is why the fold happens here, on ``M``, and never on the phased sum.

    ``kernels`` as in :func:`slab_zeta`: general per-edge kernels, the
    fold on the table-filtered group.
    """
    from gzl.hybrid import _dense_core

    edges = [tuple(int(x) for x in e) for e in np.asarray(edges, dtype=int)]
    _refuse_interaction_in_nu(nu_vec, "_slab_zeta_finite_k")
    nu_vec = [float(x) for x in np.asarray(nu_vec, dtype=float)]
    kernels = _check_kernels(kernels, nu_vec, len(edges), edges)
    # Dense under an infinite tail, as in slab_zeta (exact arithmetic).
    peelable = not bool(np.isinf(np.asarray(nu_vec, dtype=float)).any())
    A = np.asarray(A, dtype=float)
    n, d = int(n), int(A.shape[0])
    source, terminal = int(source), int(terminal)
    nodes = sorted({v for e in edges for v in e})
    if source == terminal:
        raise ValueError("_slab_M_dense: source == terminal is the vacuum "
                         "case; use slab_zeta")
    if terminal not in nodes or source not in nodes:
        raise ValueError("_slab_M_dense: source and terminal must be "
                         "vertices of the block")

    order, _steps, exponent_inner = slab_schedule(edges, source, terminal)
    if max_bytes is not None:
        need = slab_peak_bytes(edges, source, terminal, n, d)
        if need > float(max_bytes):
            raise SlabTooLargeError(
                f"finite-k slab peak {need:.3e} B exceeds max_bytes "
                f"{float(max_bytes):.3e} B")

    kernel_list = _edge_kernel_arrays(nu_vec, A, n, kernels)

    if use_symmetry:
        group = (None if kernels is None else
                 lattice_window_group(A, n, tables=_compact_tables(kernels)))
        reps, mult, inverse = orbit_reps(n, d, A, return_inverse=True,
                                         group=group)
    else:
        reps = np.arange(n ** d, dtype=np.int64)
        mult = np.ones(n ** d)
        inverse = reps
    total_sites = float(np.asarray(mult, dtype=float).sum())
    if total_sites != float(n ** d):
        raise RuntimeError(
            f"orbit multiplicities sum to {total_sites}, expected {n**d}")

    pinset = {source, terminal}
    bp_edges = [i for i, (a, b) in enumerate(edges)
                if a in pinset and b in pinset]
    has_free_factor = any(a not in pinset or b not in pinset
                          for a, b in edges)

    vals = np.empty(len(reps), dtype=float)
    for j, z in enumerate(reps.tolist()):
        scalar = 1.0
        for i in bp_edges:
            a, b = edges[i]
            ia = 0 if a == source else z
            ib = 0 if b == source else z
            v = _edge_value_both_pinned(kernel_list[i], ia, ib)
            if v == 0.0:
                scalar = 0.0
                break
            scalar *= v
        if scalar == 0.0:
            vals[j] = 0.0
            continue
        if not has_free_factor:
            vals[j] = scalar
            continue
        M = _dense_core(
            list(nodes), [list(e) for e in edges], kernel_list,
            ((source, 0), (terminal, z)), source, n, d, A,
            order=order, peelable=peelable,
        )
        vals[j] = float(M)

    return vals[inverse].reshape((n,) * d), exponent_inner


def _slab_zeta_finite_k(edges, nu_vec, A, n, *, source, terminal,
                        momentum=None, use_symmetry=True, max_bytes=None,
                        kernels=None):
    r"""Finite-k dense-block value via the second pin.

    ``momentum`` of shape ``(d,)`` (fractional BZ coordinates) returns
    the single-k scalar; ``momentum=None`` returns the full BZ grid
    ``(n,)*d``.  Phase conventions follow the shipped finite-k engines
    verbatim: single k uses the SIGNED balanced labels
    (``_balanced_z_axis``) — an unsigned index aliases off-grid — and
    the grid is one real FFT, exact at on-grid k.
    """
    from gzl.tensor_network import _balanced_z_axis

    M, exponent_inner = _slab_M_dense(
        edges, nu_vec, A, n, source=source, terminal=terminal,
        use_symmetry=use_symmetry, max_bytes=max_bytes, kernels=kernels)
    d = int(np.asarray(A, dtype=float).shape[0])
    n = int(n)
    if momentum is None:
        return np.real(np.fft.fftn(M)).astype(float), exponent_inner
    mom = np.asarray(momentum, dtype=float)
    z = _balanced_z_axis(n).astype(float)
    grids = np.meshgrid(*[z] * d, indexing="ij")
    if mom.ndim == 2:
        # Batch of k-vectors, shape (N, d): one matmul, the same cos
        # transform the box's grid path applies to its term_pos.
        zpos = np.stack([g.ravel() for g in grids], axis=1)   # (n^d, d)
        C = np.cos(2.0 * np.pi * (mom @ zpos.T))              # (N, n^d)
        return (C @ M.ravel()).astype(float), exponent_inner
    mom = mom.reshape(d)
    k_dot_z = sum(mom[c] * grids[c] for c in range(d))
    return float(np.sum(np.cos(2.0 * np.pi * k_dot_z) * M)), exponent_inner


def slab_schedule_fk_outer(edges, source: int, terminal: int, outer: int):
    r"""``(order, steps, exponent_inner)`` for the OUTER-PIN finite-k cell.

    Pins ``{source, outer}``; the TERMINAL is the kept open axis and is
    never eliminated, so it rides through every bag it touches and its
    axis is counted in each step's exponent — the honest per-cell
    price of :func:`_slab_M_dense_fk_outer`.  Same min-degree schedule
    family as :func:`slab_schedule`, priced by
    :func:`gzl._elimination.simulate` on the same scopes.
    """
    edges = [(int(u), int(v)) for u, v in np.asarray(edges).tolist()]
    source, terminal, outer = int(source), int(terminal), int(outer)
    pinned = {source, outer}
    nodes = sorted({v for e in edges for v in e})
    free = [v for v in nodes if v not in pinned]
    adj = {v: set() for v in free}
    for u, v in edges:
        if u in pinned or v in pinned:
            continue
        adj[u].add(v)
        adj[v].add(u)
    inner = [v for v in free if v != terminal]
    order = _min_degree_order(
        {v: (nb - {terminal}) for v, nb in adj.items() if v != terminal},
        inner, terminals=set())
    steps = _elimination.simulate(_inner_scopes(edges, source, outer), order)
    return order, steps, max((s.exponent for s in steps), default=0)


def _fk_outer_best(edges, source: int, terminal: int):
    r"""``(best_exponent, best_outer)`` over every candidate outer pin.

    The t-outer baseline is priced by :func:`slab_schedule`; each
    alternative by :func:`slab_schedule_fk_outer`.  Ties keep the
    t-outer (``best_outer is None``), preserving the incumbent route
    bit-for-bit wherever the reassociation buys nothing.
    """
    edges_l = [(int(u), int(v)) for u, v in np.asarray(edges).tolist()]
    source, terminal = int(source), int(terminal)
    if source == terminal:
        raise ValueError("_fk_outer_best: source == terminal is the "
                         "vacuum case and has no finite-k outer to price")
    _, _, e_t = slab_schedule(edges_l, source, terminal)
    best_e, best_u = int(e_t), None
    nodes = sorted({v for e in edges_l for v in e})
    for u in nodes:
        if u in (source, terminal):
            continue
        _, _, e_u = slab_schedule_fk_outer(edges_l, source, terminal, u)
        # The kept terminal axis is a real n^d residual even when no
        # simulated step touches it (a 3-vertex block prices exponent
        # 0), so the outer route's price is floored at 1 -- otherwise
        # the tie contract breaks in exactly the shape where the
        # "counted in each step" claim stops being true.
        if max(int(e_u), 1) < best_e:
            best_e, best_u = max(int(e_u), 1), u
    return best_e, best_u


def _slab_M_dense_fk_outer(edges, nu_vec, A, n, *, source, terminal,
                           outer, use_symmetry=True, max_bytes=None,
                           kernels=None):
    r"""Accumulated terminal field ``T[x_t] = Σ_z M_outer(z; x_t)``.

    The same torus sum as :func:`_slab_M_dense`'s consumer, EXACTLY
    reassociated: the outer sum enumerates ``outer`` (no phase — only
    ``x_t`` carries the momentum), and each cell keeps the terminal as
    the open axis of :func:`gzl.hybrid._dense_core`.  Then
    ``ζ(k) = Σ_x cos(2π k·x(x_t)) T[x_t]`` and the grid is
    ``np.real(np.fft.fftn(T))`` — the identical consumers.  Validated
    against the t-outer route and reduced-vs-unreduced in the tests
    (≤ 1e-12 asserted; measured ≤ 8.8e-15 across d = 1–3 on cubic,
    triangular, sheared, tetragonal and orthorhombic cells at even n,
    with the accumulated field anchored to
    ``graph_zeta_general(space='z')`` at 4.6e-15).

    The orbit fold acts on the FIELD: for a window-group element ``g``
    (it fixes the origin, the metric and the window),
    ``M(gz; x) = M(z; g⁻¹x)``, so
    ``Σ_{z∈orbit} M(z; ·) = (1/|stab|) Σ_{g∈G} M(z_rep; ·) ∘ g`` — one
    representative contraction per orbit, then a group-symmetrised
    gather.  Summing over the whole group makes the ``g`` vs ``g⁻¹``
    orientation of the gather irrelevant (the sum is over a group).

    ``kernels`` as in :func:`slab_zeta`.  With compact tables present
    the group is the table-filtered one, and it is ONE group for the
    price, the permutation gathers and the orbit representatives — the
    stabiliser weight ``mult_j / |G|`` is only right when all three
    agree on ``G``.
    """
    from gzl.hybrid import _dense_core

    edges = [tuple(int(x) for x in e) for e in np.asarray(edges, dtype=int)]
    _refuse_interaction_in_nu(nu_vec, "_slab_zeta_finite_k_outer")
    nu_vec = [float(x) for x in np.asarray(nu_vec, dtype=float)]
    kernels = _check_kernels(kernels, nu_vec, len(edges), edges)
    # Dense under an infinite tail, as in slab_zeta (exact arithmetic).
    peelable = not bool(np.isinf(np.asarray(nu_vec, dtype=float)).any())
    tables = _compact_tables(kernels)
    A = np.asarray(A, dtype=float)
    n, d = int(n), int(A.shape[0])
    source, terminal, outer = int(source), int(terminal), int(outer)
    nodes = sorted({v for e in edges for v in e})
    if len({source, terminal, outer}) != 3:
        raise ValueError("_slab_M_dense_fk_outer: source, terminal and "
                         "outer must be three distinct vertices")
    if not {source, terminal, outer} <= set(nodes):
        raise ValueError("_slab_M_dense_fk_outer: source, terminal and "
                         "outer must be vertices of the block")

    order, steps, exponent_inner = slab_schedule_fk_outer(
        edges, source, terminal, outer)
    if max_bytes is not None:
        # The calibrated dense-core model on the fk schedule's own
        # steps -- the same single cost model slab_peak_bytes insists
        # on, for the same reason: a second model here could only
        # drift (the hand formula this replaced was measured at
        # 0.80x the model against tracemalloc's 10.0 MB real peak).
        # Priced on top: the terminal-field accumulator, the group's
        # index-permutation arrays and the symmetrised gather buffer.
        # A refusal propagates as a MemoryError and the router's tier
        # sends the block to the BOX -- not to the t-outer route,
        # which any budget refusing this one also refuses (measured
        # margin 240x-2187x at n = 8-18 on the census heavies).
        from gzl.hybrid import _core_peak_bytes

        group_n = max(len(lattice_window_group(A, n, tables=tables)), 1)
        need = (float(_core_peak_bytes(steps, int(n), int(d)))
                + 8.0 * float(n) ** d * (group_n + 2))
        if not (need <= float(max_bytes)):
            # `not <=` so a NaN budget refuses instead of silently
            # disabling the cap (the slab_zeta lesson).
            raise SlabTooLargeError(
                f"fk-outer per-cell peak {need:.3e} B exceeds max_bytes "
                f"{float(max_bytes)!r} B")

    kernel_list = _edge_kernel_arrays(nu_vec, A, n, kernels)

    group = lattice_window_group(A, n, tables=tables)
    coords = np.indices((n,) * d)
    flat = np.arange(n ** d, dtype=np.int64).reshape((n,) * d)
    perms = []
    for perm, signs in group:
        src = tuple((int(sg) * coords[int(pj)]) % n
                    for pj, sg in zip(perm, signs))
        perms.append(flat[src].reshape(-1))
    if use_symmetry:
        # The SAME group as ``perms`` — the stabiliser weight below is
        # ``mult_j / |G|`` and is wrong the moment the two disagree.
        reps, mult, _ = orbit_reps(n, d, A, return_inverse=True,
                                   group=group)
    else:
        reps = np.arange(n ** d, dtype=np.int64)
        mult = np.ones(n ** d)
    total_sites = float(np.asarray(mult, dtype=float).sum())
    if total_sites != float(n ** d):
        raise RuntimeError(
            f"orbit multiplicities sum to {total_sites}, expected {n**d}")

    acc = np.zeros(n ** d)
    for j, z in enumerate(reps.tolist()):
        M = np.asarray(_dense_core(
            list(nodes), [list(e) for e in edges], kernel_list,
            ((source, 0), (outer, z)), terminal, n, d, A,
            order=order, peelable=peelable,
        ), dtype=float).reshape(-1)
        if use_symmetry:
            sym = np.zeros(n ** d)
            for P in perms:
                sym += M[P]
            # |stab| = |G| / orbit_size  =>  1/|stab| = mult_j / |G|
            acc += sym * (float(mult[j]) / float(len(perms)))
        else:
            acc += M
    return acc.reshape((n,) * d), exponent_inner


def _slab_zeta_finite_k_outer(edges, nu_vec, A, n, *, source, terminal,
                              outer, momentum=None, use_symmetry=True,
                              max_bytes=None, kernels=None):
    r"""Finite-k dense-block value via the OUTER-PIN association.

    Same contract and phase conventions as
    :func:`_slab_zeta_finite_k`; the value is the identical torus sum,
    reassociated so the enumerated vertex is ``outer`` instead of the
    terminal — chosen by :func:`_fk_outer_best` when its priced inner
    exponent is strictly lower.
    """
    from gzl.tensor_network import _balanced_z_axis

    T, exponent_inner = _slab_M_dense_fk_outer(
        edges, nu_vec, A, n, source=source, terminal=terminal,
        outer=outer, use_symmetry=use_symmetry, max_bytes=max_bytes,
        kernels=kernels)
    d = int(np.asarray(A, dtype=float).shape[0])
    n = int(n)
    if momentum is None:
        return np.real(np.fft.fftn(T)).astype(float), exponent_inner
    mom = np.asarray(momentum, dtype=float)
    z = _balanced_z_axis(n).astype(float)
    grids = np.meshgrid(*[z] * d, indexing="ij")
    if mom.ndim == 2:
        zpos = np.stack([g.ravel() for g in grids], axis=1)
        C = np.cos(2.0 * np.pi * (mom @ zpos.T))
        return (C @ T.ravel()).astype(float), exponent_inner
    mom = mom.reshape(d)
    k_dot_z = sum(mom[c] * grids[c] for c in range(d))
    return float(np.sum(np.cos(2.0 * np.pi * k_dot_z) * T)), exponent_inner
