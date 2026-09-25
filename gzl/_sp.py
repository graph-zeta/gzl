# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Series/parallel graph rewrite, shared across truncations.

The third piece of the truncation-agnostic layer, after the executor
(:mod:`gzl._contract`) and the planner
(:mod:`gzl._elimination`).  Like them it is a small skeleton whose
every arithmetic decision is a :class:`~gzl._contract.Truncation`
method, so the same rewrite serves the periodic torus and the
zero-padded box.

It is a *rewrite*, not an executor: it rebuilds the graph, shrinking it
to its irreducible core, and hands that to the executor.  Two rules and
a scalar:

* **parallel** — edges sharing a pair multiply pointwise (``hadamard``);
* **series** — a degree-2 vertex is composed away (``compose``);
* **loop** — an edge closing onto its own vertex contributes ``trace``,
  the kernel at zero displacement.

Why it is worth doing at all is the truncation RATE, not the cost.  Cost
is untouched — suppressing a degree-2 vertex preserves treewidth for
tw >= 2, and the planner already eliminates such vertices below the peak
bag, so the achieved exponent is identical on the block and on its core.
That is measured, not assumed, and gated by
``tests/test_sp_shared.py::TestReductionDoesNotChangeCost``: over both
shipped corpora the planner's exponent is unchanged on **2636 of 2636**
occurrences (2564 stay at 2, 72 stay at 3, none falls).
What changes is convergence.  Truncation error falls as
``n^{-sigma_eff}`` with ``sigma_eff = (minimum cluster cut) - d``, the
cheapest escape is a degree-2 vertex at cut ``2 nu``, and that is
precisely what this module removes.  A 3-connected core has edge
connectivity >= 3, so its own cut is ``>= 3 nu`` — a full ``nu`` better
in the exponent.

**The two truncations differ in kind here, not in degree.**  The torus
composes cyclically, which is EXACT for its own finite sum.  The box
composes linearly and only approximately: the exact box composition
depends on both endpoints separately rather than on their difference
(``[-L, L]^d`` is not translation-closed, and the product of two
Toeplitz matrices is not Toeplitz), so no difference kernel represents
it. See :func:`gzl._contract.box_compose`.  A box collapse must
therefore be gated on convergence in ``L``, never on agreement with an
uncollapsed box value at fixed ``L``.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np


__all__ = ["sp_reduce"]


# ---------------------------------------------------------------------------
# Bridge cache
# ---------------------------------------------------------------------------
#
# The SP collapse is the same work over and over across the census.  An
# SP part's value is a COMMUTATIVE EXPRESSION TREE -- series is
# convolution, parallel is Hadamard, both associative and commutative --
# so its canonical name is the flattened, SORTED tree with leaves
# ``('E', nu)``.  Sibling order is free; the tree SHAPE is load-bearing:
# a crude "(top kind, sorted nu multiset)" key would conflate
# ``P(2.5, S(2.5, 2.5))`` with ``P(2.5, 2.5, 2.5)``, which are different
# numbers.
#
# ORIENTATION IS PART OF THE KEY.  A series leg is read in the direction
# ``na -> w``, and a leg stored canonically the other way is reversed
# first.  On a sheared cell at even ``n`` the balanced axis holds ``n/2``
# without ``-n/2``, so ``reverse`` is NOT the identity there and two
# otherwise-identical trees with opposite leg orientation are genuinely
# different arrays.  Keying without it would serve one for the other.
#
# MEASURED at d = 2, nu = 2.5, sp_n_points = 1200: the SP collapse was
# 22-49 ms of a 15.6 ms median per-block budget on the blocks that have
# SP structure, and one ``compose`` at that grid costs ~20 ms.
#
# Bounded in BYTES for the same reason the kernel cache is: an entry is
# ``n^d`` float64, trivial at d = 1 and 11 MB at d = 2, n = 1200.
_SP_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_SP_CACHE_BYTES = 0
_SP_CACHE_STATS = {"hit": 0, "miss": 0}
SP_CACHE_MAX_BYTES = int(
    os.environ.get("GZ_SP_CACHE_BYTES", 512 * 1024 * 1024)
)


def _sp_cache_clear() -> None:
    """Drop every cached bridge, and reset the hit/miss counters."""
    global _SP_CACHE_BYTES
    _SP_CACHE.clear()
    _SP_CACHE_BYTES = 0
    _SP_CACHE_STATS.update(hit=0, miss=0)


def _sp_cache_get(key):
    if key is None:
        return None
    hit = _SP_CACHE.get(key)
    if hit is None:
        _SP_CACHE_STATS["miss"] += 1
        return None
    _SP_CACHE.move_to_end(key)
    _SP_CACHE_STATS["hit"] += 1
    return hit


def _sp_cache_put(key, arr):
    global _SP_CACHE_BYTES
    if key is None:
        return arr
    arr = np.asarray(arr)
    nbytes = arr.nbytes
    if nbytes > SP_CACHE_MAX_BYTES:
        return arr
    arr.setflags(write=False)
    while _SP_CACHE and _SP_CACHE_BYTES + nbytes > SP_CACHE_MAX_BYTES:
        _, ev = _SP_CACHE.popitem(last=False)
        _SP_CACHE_BYTES -= ev.nbytes
    _SP_CACHE[key] = arr
    _SP_CACHE_BYTES += nbytes
    return arr


def _oriented(trunc, gen, u, v):
    r"""``gen`` in the canonical ``min -> max`` sense.

    A generator built for the ordered pair ``(u, v)`` represents
    ``K(x_u - x_v)``.  Storing it against an unordered pair needs a
    convention, and reading it back with the opposite one is wrong by
    ``z -> -z``.  That is invisible whenever the kernel is even — which
    it is on every orthogonal lattice, and on the torus at every odd
    ``n`` — but the balanced label axis holds ``n/2`` without ``-n/2``,
    so on a sheared cell at even ``n`` the shell carrying an ``n/2``
    component has no partner and the kernel is genuinely not even there.
    """
    return gen if int(u) < int(v) else trunc.reverse(gen)


def _oriented_key(key, u, v):
    """The symbolic twin of :func:`_oriented`.  ``None`` propagates, so
    a caller that supplies no keys simply gets no caching."""
    if key is None:
        return None
    return key if int(u) < int(v) else ("R", key)


def _add_edge(G, trunc, u, v, gen, key=None):
    r"""Insert ``{u, v}`` carrying its generator canonically oriented.

    ``networkx`` normalises the endpoint order of a ``MultiGraph`` edge,
    so the ordered pair the generator was built for does not survive
    insertion.  Canonicalising on the way in is what makes every later
    read unambiguous.
    """
    u, v = int(u), int(v)
    G.add_edge(u, v, K=_oriented(trunc, gen, u, v),
               key_=_oriented_key(key, u, v))


def sp_reduce(edges, generators, source, terminal, trunc,
              leaf_keys=None, trunc_key=None):
    r"""Collapse series/parallel structure down to the irreducible core.

    ``edges`` is a list of ``[u, v]`` and ``generators[i]`` the kernel
    generator of edge ``i`` on ``trunc``.  Vertices ``source`` and
    ``terminal`` are never eliminated.  Returns
    ``(nodes, residual_edges, residual_generators, scalar)`` — the
    irreducible multigraph plus the accumulated scalar from traced
    loops.

    ``residual_edges`` are sorted pairs and their generators are in the
    canonical ``min -> max`` orientation, which is the convention the
    executor's factor build reads them with.

    ``leaf_keys[i]`` optionally names edge ``i`` symbolically (e.g.
    ``("E", nu)``).  When supplied together with ``trunc_key`` — which
    must identify the truncation completely, since a cached array
    belongs to one grid and one lattice — every intermediate is
    memoised in :data:`_SP_CACHE` under its canonical SP-tree name.
    Omit either and the reduction runs exactly as before, uncached.

    Self-loops in the INPUT are refused rather than traced.  The
    library's other engines refuse them too
    (``direct_sum._collapse_multi_edges``, ``frontend``'s
    ``SelfLoopError``), and the alternative is worse than an exception:
    under the regularised kernel a self-loop contributes ``K(0) = 0``,
    so accepting one would silently return a zero that looks like a
    computed value.
    """
    import networkx as nx

    src, ter = int(source), int(terminal)
    G = nx.MultiGraph()
    keys = ([None] * len(edges) if leaf_keys is None or trunc_key is None
            else [(trunc_key, k) for k in leaf_keys])
    for (u, v), gen, k in zip(edges, generators, keys):
        if int(u) == int(v):
            raise ValueError(
                f"sp_reduce: self-loop at vertex {int(u)}.  Under the "
                f"regularised kernel it contributes K(0) = 0, so the whole "
                f"graph zeta vanishes; that is refused here rather than "
                f"returned as if it were computed."
            )
        _add_edge(G, trunc, u, v, gen, k)
    keep = {src, ter}
    scalar = 1.0

    changed = True
    while changed:
        changed = False

        # --- parallel: merge every multi-edge bundle ------------------
        pairs = {frozenset((a, b)) for a, b, _ in G.edges}
        for pr in pairs:
            ab = tuple(pr)
            if len(ab) == 1:
                # A loop.  Unreachable from a self-loop-free input: the
                # only rule that could create one is the series
                # collapse below, and it traces rather than inserts.
                # Kept correct rather than deleted so the invariant is
                # stated where it would break.
                a = ab[0]
                for x, y, dd in G.edges(keys=False, data=True):
                    if x == a and y == a:
                        scalar *= trunc.trace(dd["K"])
                G.remove_edges_from(
                    [(x, y, key) for x, y, key in list(G.edges(keys=True))
                     if x == a and y == a]
                )
                changed = True
                continue
            # SORTED, not raw frozenset order.  Every stored kernel is
            # canonical min -> max, so their Hadamard product is too —
            # and handing _add_edge a DESCENDING pair makes _oriented
            # reverse an already-canonical kernel.  `tuple(frozenset(...))`
            # is hash-ordered, not sorted: it is descending on 31 of the
            # 120 pairs drawn from 0..15 (e.g. {1,8} -> (8,1)), so this is
            # a common case, not a corner one.  The damage is invisible
            # wherever the kernel is even — every orthogonal lattice, and
            # the torus at every odd n — and is a real error of 5.3e-04
            # on a sheared cell at even n, where the balanced axis holds
            # n/2 without -n/2.
            a, b = sorted(ab)
            bundle = [(dd["K"], dd.get("key_"))
                      for x, y, dd in G.edges(keys=False, data=True)
                      if {x, y} == {a, b}]
            ks = [k for k, _ in bundle]
            if len(ks) > 1:
                kk = [q for _, q in bundle]
                # Parallel is commutative, so the canonical name sorts
                # its children.  ``None`` anywhere disables caching for
                # this subtree and everything above it, which is the
                # right propagation: an unnamed child makes the parent
                # unnameable.
                prod_key = (None if any(q is None for q in kk)
                            else ("P", tuple(sorted(map(repr, kk)))))
                prod = _sp_cache_get(prod_key)
                if prod is None:
                    prod = ks[0]
                    for gen in ks[1:]:
                        prod = trunc.hadamard(prod, gen)
                    prod = _sp_cache_put(prod_key, prod)
                G.remove_edges_from(
                    [(x, y, key) for x, y, key in list(G.edges(keys=True))
                     if {x, y} == {a, b}]
                )
                _add_edge(G, trunc, a, b, prod, prod_key)
                changed = True
        if changed:
            continue

        # --- series: collapse one degree-2 non-terminal vertex --------
        for w in [x for x in G.nodes if x not in keep and G.degree(x) == 2]:
            (na, _, d1), (nb, _, d2) = (
                (e[1], e[2], e[3]) for e in G.edges(w, keys=True, data=True)
            )
            # The chain identity is
            #     K_new(x_na - x_nb) = sum_w K1(x_na - x_w) K2(x_w - x_nb),
            # so the two legs must be read in the directions na -> w and
            # w -> nb.  Both are stored canonically as min -> max, so a
            # leg whose canonical direction runs the other way is
            # reversed here.  Getting this wrong is a no-op on an even
            # kernel and silently wrong otherwise.
            # The key must record the ORIENTATION the legs are read
            # in, not just which kernels they are: ``reverse`` is not
            # the identity on a sheared cell at even n, so two trees
            # with opposite leg orientation are different arrays.
            q1, q2 = d1.get("key_"), d2.get("key_")
            new_key = (None if q1 is None or q2 is None else
                       ("S", tuple(sorted([repr(_oriented_key(q1, na, w)),
                                           repr(_oriented_key(q2, w, nb))]))))
            Knew = _sp_cache_get(new_key)
            if Knew is None:
                K1 = _oriented(trunc, d1["K"], na, w)
                K2 = _oriented(trunc, d2["K"], w, nb)
                Knew = _sp_cache_put(new_key, trunc.compose(K1, K2))
            G.remove_node(w)
            if na == nb:
                # Both legs ran to the same neighbour, so the collapse
                # closes a loop and its contribution is the composed
                # kernel at zero displacement.  Unreachable in practice
                # — the parallel pass above drains every bundle before
                # this one runs, so a degree-2 vertex always has two
                # distinct neighbours (measured: 0 hits over 1447
                # corpus blocks and 18 hand-built shapes) — but stated
                # correctly rather than left as a wrong formula on a
                # branch nobody reads.
                scalar *= trunc.trace(Knew)
            else:
                _add_edge(G, trunc, na, nb, Knew, new_key)
            changed = True
            break

    nodes = list(G.nodes())
    r_edges = [(min(a, b), max(a, b)) for a, b, _ in G.edges]
    r_generators = [dd["K"] for _, _, dd in G.edges(data=True)]
    return nodes, r_edges, r_generators, float(scalar)
