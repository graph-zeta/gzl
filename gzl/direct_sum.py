# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

r"""Direct truncated lattice summation of graph zeta functions at k = 0.

Implements the formal definition

.. math::

   \zeta_G(\boldsymbol{0}) \;=\;
   {\sum_{\boldsymbol{x}^{(1)}, \dots, \boldsymbol{x}^{(N-1)}}}^{\!\!\!\prime}
   \prod_{e = \{u,v\} \in E}
   \frac{1}{\lvert \boldsymbol{x}^{(v)} - \boldsymbol{x}^{(u)} \rvert^{\nu_e}}

by truncating each free vertex's lattice position to a finite box
:math:`[-L, L]^d` and summing exactly via **variable elimination**
(planner-chosen order at the fixed pin, with the min-degree heuristic
as the budget fallback).  The cost is

.. math::

   \mathcal{O}\!\bigl(V \cdot (2L+1)^{d \, (\mathrm{tw}(G) + 1)}\bigr)

for a graph of treewidth :math:`\mathrm{tw}(G)`, and works for any
treewidth — it is therefore the natural reference / fallback for the
graphs that the SP-reduction in :func:`gzl.graph_from_edges`
cannot handle (those containing :math:`K_4` minors).

Convergence of the truncated sum to the infinite-lattice limit is
algebraic, with leading order :math:`L^{d - \nu_{\min}}` where
:math:`\nu_{\min}` is the smallest *effective* per-edge exponent
(after multi-edge collapse).  Richardson extrapolation in :math:`L`
buys back several digits of accuracy at modest extra cost; see
:func:`direct_sum_extrapolated`.

Supports :math:`d \in \{1, 2, 3\}`.  The practical limit is set by
the memory of intermediate tensors, which scales as
:math:`(2L+1)^{d \, (\mathrm{tw}+1)}` — at d=3, tw=2 the bag tensor at
L=3 is :math:`7^9 \approx 4 \times 10^7` floats ≈ 320 MB; ``L \le 4``
fits a desktop, ``L \ge 5`` needs streaming or a different evaluator.

Elimination steps whose bucket isolates one original two-vertex
kernel are contracted by zero-padded real FFT (the kernel is
translation invariant, so that step is a Toeplitz contraction):
identical to the dense sum up to round-off, at
:math:`N^{|C|-1}\log N` instead of :math:`N^{|C|}` per step,
:math:`N = (2L+1)^d`.  See ``_eliminate_all``.

General per-edge kernels (:class:`gzl.interaction.Interaction`,
``V(x) = a(x) + sum_j b_j |x|^{-nu_j}`` with a compact table ``a``)
enter through the keyword ``kernels`` of the public entry points and
replace the power law at the ONE build site, :func:`_conv_kernel_diff`;
everything downstream is array-only.  ``nu`` keeps carrying the
per-edge TAIL exponents, which is what the planner, the cluster cut,
the Richardson basis and the FFT opt-out read.  Without ``kernels``
the legacy power-law path runs unchanged.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.fft import next_fast_len as _next_fast_len

from gzl import _contract
from gzl import _elimination
from gzl._errors import UnsupportedLatticeSumError
from gzl._labels import relabel_to_support as _relabel_to_support
from gzl._lattices import _resolve_lattice
from gzl._real import _as_real
from gzl.interaction import POWER_BOX as _POWER_BOX
from gzl.interaction import InteractionSupportError
from gzl.interaction import is_interaction as _is_interaction
from gzl.tensor_network import _refuse_interaction_in_nu


__all__ = [
    "direct_sum_zero_momentum",
    "direct_sum_extrapolated",
    "PinAlphaDesyncError",
]


class PinAlphaDesyncError(RuntimeError):
    """The Richardson basis exponent disagrees with its independent twin.

    ``direct_sum_extrapolated`` and ``_direct_sum_extrapolated_grid``
    derive ``alpha = d - _min_free_cut_nu(edge_map, root_eff)`` while
    the per-L sums pin ``root_eff``.  A corrupted basis — any drift
    between the cut the engine computes and the quantity it is defined
    to be — fits the wrong power with the least-squares solve still
    succeeding, the value finite and plausible, and every raw per-L
    value individually correct; nothing numerical downstream can catch
    it.  Both extrapolated sites therefore cross-check the engine's cut
    against the planner-side twin
    :func:`gzl._elimination.min_free_cut_nu` at the same root
    and refuse on any mismatch (exact ``!=``: the twins iterate the
    same mapping in the same order, so healthy agreement is bitwise).

    Deliberately a :class:`RuntimeError`: the frontend's tw >= 3 arms
    swallow ``(ValueError, MemoryError)`` into the tensor fallback, and
    this defect must stay loud, not be silently rerouted.
    """

# Private A/B switch for the FFT elimination fast path below.  Read at
# call time inside _eliminate_all so tests can monkeypatch it; with
# False the dense branch reproduces the pre-FFT arithmetic exactly.
_USE_CONV = True

# Force the open-terminal streaming schedule regardless of the memory
# budget (test hook, mirroring tensor_network._FORCE_STREAMING).  Read
# at call time inside _direct_sum_open_terminal.
_FORCE_STREAMING = False

# Per-call diagnostic (the tau*d acceptance test's explicit
# cost-gate exemption): every elimination step that was structurally
# FFT-peelable (an original kernel with an admissible partner sat in
# the bucket) but ran dense because the COST gate declined —
# _conv_possible / _FFT_MIN_BAG, not the dtype guard.  The box gate
# exists because dense really is faster at small L, so a declined peak
# step is a deliberate exponent-vs-constant trade; this record is what
# keeps that trade visible instead of silently degrading the achieved
# exponent.  REPLACED by each _eliminate_all call (never appended
# across calls); entries are dicts {"vertex", "axes", "reason"}.
_LAST_GATE_DECLINED: list = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collapse_multi_edges(edges, nu):
    """Combine parallel edges of the same vertex pair into one effective edge.

    Parameters
    ----------
    edges : ndarray (n, 2) int
    nu : ndarray (n,) float

    Returns
    -------
    edge_map : dict {(u, v) sorted: effective_nu}
        Sum of the per-edge exponents for each unique vertex pair.
    """
    edge_map: dict = {}
    for (u, v), nu_e in zip(edges.tolist(), nu.tolist()):
        u, v = int(u), int(v)
        if u == v:
            raise ValueError(
                f"direct_sum: self-loop at vertex {u}; not allowed."
            )
        key = (min(u, v), max(u, v))
        edge_map[key] = edge_map.get(key, 0.0) + float(nu_e)
    if not edge_map:
        raise ValueError("direct_sum: edge list is empty.")
    return edge_map


def _collapse_multi_kernels(edges, nu, kernels):
    """Per-pair lazy Hadamard product of the kernels, beside the float map.

    :func:`_collapse_multi_edges` keeps summing the TAIL exponents per
    unordered pair — that float is what the planner, the cluster cut,
    the Richardson basis and the FFT opt-out read, and it is untouched
    by this function.  This returns the parallel ``{(min, max):
    kernel}`` map the factor build samples from: the single kernel of
    a simple pair, or the lazy pointwise product of the parallel
    kernels in row order (the box's Hadamard merge, exact on any grid
    — see :class:`gzl.interaction._KernelProduct`).

    Parameters
    ----------
    edges : ndarray (n, 2) int
    nu : ndarray (n,) float
        The per-edge tails, already validated against ``edges``.
    kernels : sequence of Interaction-likes, length ``n``
        One per row of ``edges``, in the same order.

    Raises
    ------
    ValueError
        On a length mismatch, or when a row's ``nu`` and its kernel
        disagree on whether the tail is finite: the engine keys the
        planner's non-kernel edges and the FFT opt-out on ``isinf(nu)``
        and would otherwise peel an integer-exact table or densify a
        peelable bundle.
    TypeError
        On an entry that is not an Interaction-like.
    """
    edges = np.asarray(edges, dtype=int)
    nu = np.asarray(nu, dtype=float)
    kernels = list(kernels)
    if len(kernels) != edges.shape[0]:
        raise ValueError(
            f"direct_sum: kernels has length {len(kernels)} but edges has "
            f"{edges.shape[0]} rows; pass one Interaction per edge, "
            f"aligned with nu."
        )
    kern_map: dict = {}
    for (u, v), nu_e, kern in zip(edges.tolist(), nu.tolist(), kernels):
        u, v = int(u), int(v)
        if not _is_interaction(kern):
            raise TypeError(
                f"direct_sum: kernels entries must be Interaction-likes; "
                f"got {type(kern).__name__} for edge ({u}, {v})."
            )
        tail = float(kern.tail_exponent)
        nu_f = float(nu_e)
        if not (nu_f == tail or (math.isfinite(nu_f) and math.isfinite(tail)
                                 and math.isclose(nu_f, tail, rel_tol=1e-12,
                                                  abs_tol=0.0))):
            # The declared tail sets the Richardson basis exponent; a
            # tail below the kernel's real one was measured to leave the
            # ladder 33 % wrong with no diagnostic.
            raise ValueError(
                f"direct_sum: edge ({u}, {v}) carries nu = {nu_e!r} but its "
                f"kernel's tail exponent is {kern.tail_exponent!r}; nu must "
                f"be the tail the kernel decays with."
            )
        key = (min(u, v), max(u, v))
        prev = kern_map.get(key)
        kern_map[key] = kern if prev is None else prev * kern
    return kern_map


def _kernel_cache_key(kern) -> tuple:
    """``kdiff_cache`` key of an Interaction-like: ``("K",) + kern.key()``.

    A third key space beside the legacy float ``nu`` entries and the
    ``("hat", key, m[, w_half, u_half])`` transform entries: a tuple
    headed by ``"K"`` can never equal a float (a float key is what the
    legacy factor build writes and reads, and must keep meaning the
    power law) nor a ``("hat", ...)`` entry.  Keying on the tail alone
    was measured to serve one kernel's generator to another with the
    same tail (a 200 % error on a two-table path); ``key()`` carries
    the weights and the table, so equal tails with different tables
    are distinct entries.
    """
    return ("K",) + tuple(kern.key())


def _kernel_generator(kdiff_cache, kern, A, L, d):
    """The difference-range generator of an Interaction-like, cached in
    ``kdiff_cache`` under :func:`_kernel_cache_key`.  Returns
    ``(gen, key)``; the key is what the factor's ``conv_nu`` tag and the
    peel's ``("hat", key, m)`` transform entries carry."""
    key = _kernel_cache_key(kern)
    g = kdiff_cache.get(key)
    if g is None:
        g = _conv_kernel_diff(kern.tail_exponent, A, L, d, interaction=kern)
        kdiff_cache[key] = g
    return g, key


def _compact_reach(kern_map, edge_map, root, *, legacy_R=None) -> int:
    """Smallest box half-width that holds every compact part exactly.

    An edge's compact part vanishes beyond its Chebyshev label radius
    ``R_e``, so in any term of ``prod_e (a_e + P_e)`` in which a free
    vertex is tied to the root through table factors only, that vertex
    lies within the ``R_e``-weighted length of that path.  The reach is
    therefore the largest ``R_e``-weighted shortest-path distance from
    the root over the COMPACT SUBGRAPH — the edges whose kernel carries a
    table (weight ``R_e``); a pure power-law edge is not a constraint,
    it is the ladder's power-law tail, and it must NOT shorten the paths
    (an all-edge eccentricity under-estimates the reach as soon as a
    power-law chord exists: measured a silent 100 % error on a triangle
    with one power-law edge at ``L_list = (3, 4, 5)``).  A box of
    half-width ``L >= reach`` holds every table-tied term exactly; one
    below it clips them, a truncation error with no power-law form that
    the Richardson basis cannot fit.  Vertices the root cannot reach
    through table edges impose nothing.  Returns 0 when no kernel has a
    compact part.  On the legacy ``nu = inf`` path (``kern_map is
    None``) the indicator edges are the compact subgraph, of radius
    ``legacy_R`` each.
    """
    weight: dict = {}
    for (u, v) in edge_map:
        if kern_map is not None:
            R = int(kern_map[(u, v)].support_radius)
        else:
            R = (int(legacy_R) if legacy_R and math.isinf(float(edge_map[(u, v)]))
                 else 0)
        if R <= 0:
            continue
        weight.setdefault(u, {})[v] = max(weight.get(u, {}).get(v, 0), R)
        weight.setdefault(v, {})[u] = max(weight.get(v, {}).get(u, 0), R)
    if not weight:
        return 0
    root = int(root)
    dist = {root: 0}
    todo = {root}
    while todo:
        w = min(todo, key=lambda x: dist[x])
        todo.discard(w)
        for y, R in weight.get(w, {}).items():
            cand = dist[w] + R
            if y not in dist or cand < dist[y]:
                dist[y] = cand
                todo.add(y)
    return max(dist.values())


def _check_compact_reach(kern_map, edge_map, root, L_list, where) -> int:
    """Refuse a Richardson ladder whose smallest rung clips a compact part.

    Raises :class:`gzl.interaction.InteractionSupportError` —
    a ``ValueError`` — so the front-end's ``(ValueError, MemoryError)``
    nets fall through to the tensor rather than fitting a clipped
    ladder.  Returns the reach (see :func:`_compact_reach`).
    """
    reach = _compact_reach(kern_map, edge_map, root)
    L_min = min(int(L) for L in L_list)
    if reach and L_min < reach:
        raise InteractionSupportError(
            f"{where}: the compact parts reach Chebyshev label radius "
            f"{reach} from the pinned root {root} (the radius-weighted "
            f"shortest-path distance over the compact subgraph), but the "
            f"smallest Richardson rung is L = {L_min}; the box is exact for "
            f"a compact part only when every rung holds the reach, so use "
            f"L_list >= {reach} or a torus engine."
        )
    return reach


def _indicator_radius(A, L, d) -> int:
    """Chebyshev label radius of the legacy ``nu = inf`` nearest-neighbour
    indicator, read off the generator the engine builds at this ``L``
    (1 on every reduced basis; whatever the window shows otherwise, which
    is the limit the indicator itself lives under)."""
    g = _conv_kernel_diff(np.inf, A, L, d)
    hits = np.argwhere(g != 0.0)
    if hits.size == 0:
        return 0
    return int(np.abs(hits - 2 * L).max())


def _degenerate_rung(edge_map, root, L_list, reach, kern_map, A, d) -> int:
    """The one box half-width a block with no finite tail is summed at.

    When the Richardson basis exponent ``alpha = d - nu_eff_lead`` is
    not finite no free cluster can escape — every edge of every
    escaping cut is purely compact (or the legacy ``nu = inf``
    indicator) — so the truncated sum is EXACT once the box holds the
    reach and there is no tail to fit: the k = 0 power fit has every
    basis column ``L ** (alpha - j)`` identically zero for ``L > 1``
    and degenerates to the arithmetic MEAN of the rungs (measured
    7.333 for an exact 8 on the 3-edge path at ``nu = inf``,
    ``L_list = (2, 3, 4)``, ``root = 0``).  The value is one rung at
    the larger of ``max(L_list)`` and the reach.  Callers key on the
    basis exponent, never on the L values: ``1 ** (-inf) == 1`` makes
    a rung of ``L = 1`` a different degenerate fit, not a non-degenerate
    one.
    """
    if kern_map is None:
        # The legacy indicator: its shell radius (1 on a reduced basis)
        # over the inf-edge subgraph is the reach the kernel guard
        # derives from the tables.
        R = _indicator_radius(A, max(int(L) for L in L_list), d)
        reach = _compact_reach(None, edge_map, root, legacy_R=R)
    return max(max(int(L) for L in L_list), int(reach))


def _adjacency(edge_map, V):
    """Symmetric adjacency dict from the edge map."""
    adj = {v: set() for v in range(V)}
    for (u, v) in edge_map:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _pick_root(edge_map, V):
    """Origin = vertex of maximum degree (graph-centroid heuristic)."""
    deg = np.zeros(V, dtype=int)
    for (u, v) in edge_map:
        deg[u] += 1
        deg[v] += 1
    return int(np.argmax(deg))


def _min_free_incident_nu(edge_map, root):
    r"""Smallest total incident exponent over the *free* (non-root) vertices.

    The box-truncation tail of the multi-vertex sum is governed by the
    slowest-decaying summed vertex: as one free vertex :math:`v` leaves
    the box (its neighbours held near the origin) the integrand decays
    as :math:`\lVert x_v\rVert^{-\sum_{e \ni v}\nu_e}`, so the leading
    Euler-Maclaurin exponent is :math:`d - \min_{v\text{ free}}
    \sum_{e\ni v}\nu_e`.

    For a pendant/degree-1 free vertex this reduces to the single
    edge's :math:`\nu`, i.e. the old ``min(edge_map.values())``; for a
    densely-connected graph (e.g. :math:`K_5`, every summed vertex of
    degree 4) it is :math:`\mathrm{deg}\cdot\nu`, a far steeper — and
    correct — decay that the old shallow basis could not fit.
    """
    inc: dict = {}
    for (u, v), nu_e in edge_map.items():
        inc[u] = inc.get(u, 0.0) + float(nu_e)
        inc[v] = inc.get(v, 0.0) + float(nu_e)
    free = [w for w in inc if w != root]
    if not free:
        return min(edge_map.values())
    return min(inc[w] for w in free)


def _min_free_cut_nu(edge_map, root, externals=(), max_free_exact=16):
    r"""Smallest **cluster cut weight** over escaping free-vertex sets.

    Generalises :func:`_min_free_incident_nu`.  The truncation tail is
    not governed only by a *single* free vertex escaping: a whole
    connected **cluster** ``S`` of free vertices can escape together,
    its internal edges staying short while only the edges cutting
    ``S`` from the rest are stretched.  Such a mode decays as
    ``|X| ** -I_S`` with

    .. math::

       I_S \;=\; \sum_{e \in \mathrm{cut}(S)} \nu_e ,

    so the leading Euler--Maclaurin exponent is :math:`d - \min_S I_S`
    over connected :math:`S`, of which the single-vertex rule is the
    ``|S| = 1`` special case.

    Both ``G[S]`` and ``G[V \ S]`` are required connected: a
    disconnected side has no finite pinned zeta and corresponds to a
    higher-codimension escape, not a leading mode.

    Concretely, for :math:`K_4` pinned at 0 the full free set
    ``{1, 2, 3}`` cuts the same three edges as any single vertex, so it
    is a *leading* mode the single-vertex rule misses entirely.

    Returns the summed ``nu`` (not ``nu - d``), matching the contract
    of :func:`_min_free_incident_nu`.  The result is always ``<=`` that
    function's, since singletons are among the candidates — so the
    exponent can only become more conservative, never more optimistic.

    ``externals`` names further vertices that cannot escape (e.g. an
    open terminal).  Above ``max_free_exact`` free vertices the exact
    ``2 ** |free|`` enumeration is replaced by singletons, connected
    pairs/triples and the full free set; the real corpus caps at 9 free
    vertices, so the exact branch is what runs.
    """
    import itertools

    vertices = set()
    for (u, v) in edge_map:
        vertices.add(u)
        vertices.add(v)
    pinned = {root} | set(externals)
    free = sorted(vertices - pinned)
    if not free:
        return min(edge_map.values())

    adj: dict = {w: set() for w in vertices}
    for (u, v) in edge_map:
        adj[u].add(v)
        adj[v].add(u)

    def _connected(nodes) -> bool:
        nodes = set(nodes)
        if len(nodes) <= 1:
            return True
        start = next(iter(nodes))
        seen, stack = {start}, [start]
        while stack:
            for y in adj[stack.pop()] & nodes:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return len(seen) == len(nodes)

    def _cut(S) -> float:
        return sum(float(nu_e) for (u, v), nu_e in edge_map.items()
                   if (u in S) != (v in S))

    if len(free) <= max_free_exact:
        sizes = range(1, len(free) + 1)
    else:                                   # see docstring
        sizes = list(range(1, 4)) + [len(free)]

    best = None
    for r in sizes:
        for S in itertools.combinations(free, r):
            Sset = set(S)
            if not _connected(Sset):
                continue
            if not _connected(vertices - Sset):
                continue
            w = _cut(Sset)
            if best is None or w < best:
                best = w
    if best is None:                        # no admissible cluster
        return _min_free_incident_nu(edge_map, root)
    return best


def _min_degree_order(adj, free, root):
    """Min-degree elimination order on the free vertices.

    Facade over :func:`gzl._elimination.min_degree_order` with
    ``fill_in=False`` — the box's static-adjacency variant, counting
    edges to remaining-free or pinned-root neighbours.  Deliberately
    NOT the torus's fill-in heuristic.  Since the box adopted the
    exact planner this is the *fallback* schedule — the production
    order comes from :func:`_planned_or_min_degree` — with the
    shipped behaviour preserved verbatim in the merged home.
    """
    return _elimination.min_degree_order(
        adj, free, root=root, fill_in=False)


def _planned_or_min_degree(edge_map, V, root, adj, free, keep=(),
                           extra_scopes=(), non_kernel_edges=()):
    """Box elimination order: the exact planner at the *fixed* pin.

    The pin is NEVER the planner's to move here — a box ``[-L, L]^d``
    is not closed under translation, so re-pinning is not value-exact
    (see ``_elimination.best_order``); ``plan`` is called with
    ``pin=root`` and the echoed pin is asserted, which is the
    ``plan.pin == root_eff`` guard, placed in the one helper every box
    schedule flows through.

    On budget refusal (:class:`_elimination.PlanBudgetExceededError`)
    the shipped min-degree heuristic takes over unchanged.  ``plan`` is
    resolved through the module attribute at call time so the guard
    suite's spy stays live, and its exact-key memoisation makes the
    per-L ladder pay for one DP, not ``len(L_list)``.
    """
    try:
        p = _elimination.plan(edge_map, range(V), pin=root, keep=keep,
                              extra_scopes=extra_scopes,
                              non_kernel_edges=non_kernel_edges)
    except _elimination.PlanBudgetExceededError:
        keep_set = set(keep)
        return _min_degree_order(
            adj, [v for v in free if v not in keep_set], root)
    if p.pin != root:
        raise PinAlphaDesyncError(
            f"direct_sum: planner returned pin {p.pin} for requested "
            f"root {root}; the box pin is not the planner's to move."
        )
    return list(p.order)


def _check_alpha_basis(edge_map, root_eff, nu_eff_lead):
    """Refuse a desynced Richardson basis exponent.

    See :class:`PinAlphaDesyncError`.  Called at both extrapolated
    sites right after the engine derives ``nu_eff_lead``; the twin is
    resolved through :mod:`gzl._elimination`'s module dict so it
    cannot be desynced together with the engine-side function.
    """
    nu_twin = _elimination.min_free_cut_nu(edge_map, root_eff)
    if nu_twin != nu_eff_lead:
        raise PinAlphaDesyncError(
            f"direct_sum: Richardson basis exponent desync at root "
            f"{root_eff}: engine cut {nu_eff_lead!r} != planner-side "
            f"cut {nu_twin!r}; refusing to fit a wrong-power basis."
        )


# ---------------------------------------------------------------------------
# FFT-accelerated elimination step
# ---------------------------------------------------------------------------
#
# An elimination step contracts the bag C = {w} ∪ S at dense cost
# N^|C|, N = (2L+1)^d.  When some u ∈ S lies in the scope of exactly
# one bucket factor and that factor is an original two-vertex kernel
# t[i, j] = K(x_j - x_i) (translation invariant ⇒ Toeplitz in the
# flattened box index), the w-sum
#
#     psi(x_u, rest) = sum_{x_w} Phi(x_w, rest) K(x_u - x_w)
#
# is a linear convolution along the w axis and is done exactly by
# zero-padded real FFT at cost N^(|C|-1) log N — one full power of N
# less, identical to the dense sum up to round-off.  The condition is
# checked per step; steps that fail it (e.g. a previously merged
# factor also carries u) run the dense branch unchanged.

def _axis_extents(L, d, half):
    """Per-axis lengths of a vertex's position grid.

    Index arithmetic lives in :mod:`gzl._contract` (the box half
    of the shared factor-supply layer); these wrappers are the module's
    stable names.
    """
    return _contract.box_axis_extents(L, d, half)


def _axis_origins(L, d, half):
    """Per-axis lattice coordinate of index 0: ``pos[i]_c = i_c + org_c``."""
    return _contract.box_axis_origins(L, d, half)


def _gen_offsets(L, d, org_a, org_b):
    """Per-axis constant in the generator index of a two-vertex factor.

    The generator holds ``K(delta)`` at ``delta + 2L`` per axis, so with
    ``pos[i]_c = i_c + org_c``,

        gen_index_c = (j_c + org_b_c) - (i_c + org_a_c) + 2L
                    = j_c - i_c + (org_b_c - org_a_c + 2L)

    and only that trailing constant depends on which endpoint is the
    half axis.
    """
    return _contract.box_gen_offsets(L, d, org_a, org_b)


def _table_from_generator(gen, ext_a, ext_b, off, d):
    """Two-axis factor read out of the difference-range generator.

    Broadcast index views, so the only ``size_a * size_b`` allocation is
    the result itself — stacking per-axis differences instead would cost
    a further factor ``d`` in peak memory, which is the whole point of
    not materialising these up front.
    """
    return _contract.box_table(gen, ext_a, ext_b, off, d)


def _row_from_generator(gen, ext, off, d):
    """One-axis factor ``K(x_v - 0)``, for a kernel incident on the root.

    The root sits at the origin, so this is the ``i``-indexed slice of
    the same generator — an ``O(n)`` gather instead of an ``O(n^2)``
    table followed by a row pick.
    """
    return _contract.box_row(gen, ext, off, d)


def _box_slice_at(gen, ext_keep, off, d, t_idx, keep_is_b):
    """One row/column of a lazy box kernel at an arbitrary pinned index.

    The two-axis table is ``T[a, b] = gen[(b - a) + off]`` per axis
    (:func:`_contract.box_table`).  Pinning axis ``a`` at multi-index
    ``t`` keeps ``b``: ``gen[b - t + off]`` (``keep_is_b=True``);
    pinning ``b`` keeps ``a``: ``gen[t - a + off]``.  Pure index
    gather of copies — bit-identical to slicing the materialised
    table — used by the open-terminal streaming path to pin the
    terminal per position instead of building ``size^2`` tables.
    """
    idx = []
    for c in range(d):
        if keep_is_b:
            ax = np.arange(ext_keep[c]) + (int(off[c]) - int(t_idx[c]))
        else:
            ax = (int(off[c]) + int(t_idx[c])) - np.arange(ext_keep[c])
        shape = [1] * d
        shape[c] = ext_keep[c]
        idx.append(ax.reshape(shape))
    return gen[tuple(idx)].reshape(-1)


def _factor_array(f):
    """Return a factor's dense array, building it on first use.

    .. warning::
       ``tensor_network`` has a function of the *same name* and a
       different signature (``_factor_array(T, gen, n_points=None)``).
       They do the same job for the two truncations and are intended to
       converge, but until then do not import both into one namespace
       without aliasing — the repo already has one instance of that
       hazard, where two ``_min_degree_order`` variants with different
       signatures are imported side by side.

    Kernel factors are stored as ``{'gen', 'ext', 'off'}`` — the
    difference-range generator plus the index arithmetic that reads a
    table out of it — and materialise only when a consumer actually
    needs the dense form.  Each factor is consumed by exactly one
    elimination step, so nothing is built twice and the tables are never
    all live at once.

    Non-kernel factors (the finite-momentum cos weight, and every
    intermediate produced by elimination) carry ``'tensor'`` directly.
    """
    t = f.get("tensor")
    if t is not None:
        return t
    ext = f["ext"]
    if len(ext) == 1:
        return _row_from_generator(f["gen"], ext[0], f["off"], f["d"])
    return _table_from_generator(f["gen"], ext[0], ext[1], f["off"], f["d"])


def _conv_kernel_diff(nu_e, A, L, d, interaction=None):
    """Kernel sampled on the difference range ``[-2L, 2L]^d``.

    Returns shape ``(2(2L+1)-1,)*d`` in C-order, origin zeroed —
    entry ``delta + 2L`` (per axis) holds ``K(delta)``.

    ``interaction`` (an :class:`gzl.interaction.Interaction` or
    a lazy product of them) replaces the power law wholesale: the same
    integer difference grid is sampled through
    ``interaction.sample(labels, A, power=POWER_BOX)`` — the box's own
    ``dist ** (-nu)`` form for the power-law terms, the compact table
    scattered at its labels with ``a(0)`` kept (a coincident-endpoint
    weight, not zeroed) — and ``nu_e`` is not read.  With
    ``interaction=None``, which is every legacy caller (positional
    ``(nu_e, A, L, d)``), the body below runs unchanged.

    There used to be a d = 1 ``abs(a1·x)`` fast path here, so that the
    generator matched dense tables built the same way.  Both engines now
    read their factors out of this generator, so there are no separate
    tables left to match — and the split had become a hazard rather than
    a saving: ``direct_sum_zero_momentum`` passed ``a1 = A[0, 0]`` while
    ``_direct_sum_open_terminal`` passed ``None``, so the two produced
    different d = 1 generators for the same lattice.  The paths agree
    elementwise over the values a lattice sum uses (``sqrt(x*x)`` and
    ``abs(x)`` diverge only where ``x*x`` overflows or underflows, at
    ``|a1·x|`` outside roughly ``[1e-162, 1e154]``), and every frozen
    box golden is unchanged by the removal.
    """
    if interaction is not None:
        # Integer LABELS (x = A m) on the same grid; a table label
        # outside [-2L, 2L]^d raises InteractionSupportError before a
        # clipped kernel can produce a plausible number.
        m = 2 * L + 1
        axes = [np.arange(-(m - 1), m)] * d
        labels = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        return interaction.sample(labels, A, power=_POWER_BOX)
    m = 2 * L + 1
    axes = [np.arange(-(m - 1), m, dtype=float)] * d
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    dist = np.linalg.norm(grid @ A.T, axis=-1)
    if np.isinf(nu_e):
        # Nearest-neighbour indicator (the ν → ∞ limit), keyed off the
        # minimal nonzero distance exactly as the torus kernel is
        # (tensor_network._edge_kernel_torus) — scale-invariant.  The
        # literal dist ** -inf that previously ran here is the latent
        # defect this branch removes: it evaluates to 1 / nan / 0 for
        # NN distances = 1 / < 1 / > 1, giving 4 / nan / 0 on the
        # 2-path at A = I, I/2, 2I where the correct NN count is 4 for
        # all three.  Unreachable from the frontend today (nn_mode
        # routes every inf block to the tensor) — this makes the box
        # correct before any routing ever exposes it.
        K = np.zeros_like(dist)
        nz = dist > 0.0
        dmin = dist[nz].min()
        K[nz & np.isclose(dist, dmin, rtol=1e-9, atol=0.0)] = 1.0
        return K
    with np.errstate(divide="ignore"):
        t = dist ** (-nu_e)
    t[dist == 0.0] = 0.0
    return t


def _peel_workset_bytes(phi_sizes, w_pos, chunk_pos, m, d):
    """Modelled bytes of one peel chunk's working set, per unit of the
    chunk axis.

    Counts the PEEL, not the bag result: phi (and the copy its axis
    move can force), the padded spectrum, the spectral product (numpy
    allocates the result), the padded inverse, and the output slice
    copy.  A bag-based model under-predicts the true peak by ~2.4x —
    the padded intermediates dominate — which is why this model exists
    and why the test gate requires it to UPPER-bound a measured
    ``ru_maxrss``, not merely correlate with it.
    """
    # The transform runs at next_fast_len(2m - 1) (up to ~15% longer
    # per axis, e.g. 21 -> 24); modelling the minimal length would
    # size chunks against an under-count.  The marker's shorter 3L+1
    # windows are over-predicted by this, which is the safe direction
    # for an upper bound.
    P = int(_next_fast_len(2 * m - 1, True))
    spec_complex = P ** (d - 1) * (P // 2 + 1)     # rfft over d axes
    base = 1
    for i, sz in enumerate(phi_sizes):
        if i not in (w_pos, chunk_pos):
            base *= int(sz)
    m_d = int(phi_sizes[w_pos])
    per_unit = base * (
        8 * m_d              # phi chunk
        + 8 * m_d            # reshape copy after the axis move
        + 16 * spec_complex  # forward spectrum
        + 16 * spec_complex  # spectral product (new allocation)
        + 8 * P ** d         # padded inverse transform
        + 8 * m_d            # output slice copy
    )
    return int(per_unit)


def _pad_lengths(w_ext, u_ext, d):
    """Per-axis FFT lengths of one peel step.

    The minimal alias-free length is ``Ws + Us - 1`` (linear
    convolution), rounded up to the next fast composite for a real
    transform — ``4L + 1`` is prime or large-prime-factored at 11 of
    the 23 rungs L = 2..24 (measured 1.7x penalty at L = 24:
    ``rfft(97)`` 5.46 us vs ``rfft(105)`` 3.22 us).  Any
    ``P >= Ws + Us - 1`` keeps the wanted output window alias-free
    (see :func:`_fft_conv_step`), so the rounding changes only FFT
    round-off, gated at 1e-12 against the frozen goldens.
    """
    return tuple(
        int(_next_fast_len(int(w_ext[c] + u_ext[c] - 1), True))
        for c in range(d)
    )


def _fft_conv_step(phi, kdiff, m, d, kdiff_hat=None,
                   w_ext=None, u_ext=None, gen_start=None):
    """Zero-padded linear convolution along the eliminated axis.

    ``phi`` has the flat ``(m**d,)`` eliminated axis first and any
    number of spectator axes after it; ``kdiff`` is the
    ``(2m-1,)*d`` difference-range kernel.  Returns
    ``psi[u, rest] = sum_v phi[v, rest] kdiff[u - v]`` with the new
    axis (the convolution partner) first, shape ``(m**d,) + rest``.

    Padding each axis to ``2m - 1`` makes the cyclic FFT reproduce
    the linear convolution exactly; the kernel is indexed by
    ``(u - v) + (m - 1)``, so the wanted outputs sit in the slice
    ``[m-1 : 2m-1]`` per axis.

    ``kdiff_hat`` optionally supplies the kernel's padded transform.
    It depends only on ``(nu, A, L, d)`` and not on ``phi``, so a call
    whose steps peel the same kernel repeatedly would otherwise redo an
    identical transform per step.  Supplying it changes nothing
    arithmetically -- it is the same ``rfftn`` of the same array -- so
    the signature keeps ``kdiff`` and the default recomputes.

    ``w_ext`` / ``u_ext`` / ``gen_start`` generalise the step to
    endpoints living on *unequal* per-axis windows — the Z2 marker's
    half axis.  The defaults (all ``None``) are the full-box/full-box
    step: both extents ``(m,)*d``, the whole difference-range kernel,
    output slice ``[m-1 : 2m-1]`` per axis.  In the general form, axis
    ``c`` with w-window ``Ws`` and u-window ``Us`` positions reads the
    kernel sub-range
    ``kdiff[gen_start_c : gen_start_c + Ws + Us - 1]`` (exactly the
    differences the step can reach), transforms at
    ``P_c = next_fast_len(Ws + Us - 1)`` (see :func:`_pad_lengths`:
    the rounding is mathematically free, and 11 of the 23 d = 1 rungs
    L = 2..24 otherwise transform at prime or large-prime lengths —
    the SHIPPED d = 1 ladders stop at L = 8 and are gate-closed, so
    the live wins are the open-gate d >= 2 rungs)
    and takes the outputs in ``[Ws - 1 : Ws - 1 + Us]``.  Alias
    check, valid for ANY ``P_c >= Ws + Us - 1``: the
    linear-convolution support ends at ``Ws + (Ws + Us - 1) - 2``, so
    the cyclic fold maps at most ``[P_c, 2 Ws + Us - 3]`` onto
    ``[0, Ws - 2]``, disjoint from the wanted window — the cyclic
    transform reproduces the linear convolution exactly at the wanted
    outputs.
    """
    if w_ext is None:
        w_ext = (m,) * d
    if u_ext is None:
        u_ext = (m,) * d
    if gen_start is None:
        gen_start = (0,) * d
    K_len = tuple(int(w_ext[c] + u_ext[c] - 1) for c in range(d))
    P = _pad_lengths(w_ext, u_ext, d)
    spect = phi.shape[1:]
    fft_axes = tuple(range(d))
    phi_g = phi.reshape(tuple(w_ext) + spect)
    ks = kdiff[tuple(slice(gen_start[c], gen_start[c] + K_len[c])
                     for c in range(d))]
    fp = np.fft.rfftn(phi_g, s=P, axes=fft_axes)
    fk = (np.fft.rfftn(ks, s=P, axes=fft_axes)
          if kdiff_hat is None else kdiff_hat)
    conv = np.fft.irfftn(
        fp * fk.reshape(fk.shape + (1,) * len(spect)),
        s=P, axes=fft_axes,
    )
    sl = tuple(slice(w_ext[c] - 1, w_ext[c] - 1 + u_ext[c])
               for c in range(d)) + (slice(None),) * len(spect)
    n_out = 1
    for c in range(d):
        n_out *= int(u_ext[c])
    return conv[sl].reshape((n_out,) + spect)


def _fft_step_factor(m, d):
    """Cost of one padded-FFT fibre relative to one dense bag entry.

    Zero-padding each of the ``d`` axes to ``~2m`` (the next fast
    length at or above ``2m - 1``, see :func:`_pad_lengths`) costs a
    factor ``2^d`` in transform length, times the usual log factor.
    The convolutional step therefore beats the dense step on an axis
    of ``n`` points when this factor is (comfortably) below ``n``;
    calibrated against measured per-step timings at d = 1, 2.

    Re-verified after the fast-length change: at the d = 1 rung
    where ``_FFT_MARGIN`` opens the gate (L = 15) the measured
    FFT/dense ratio is 1.11 — inside the model's stated factor-1.5
    accuracy, and better than the 1.47-2.03x the same opening cost
    before fast lengths — with the true crossover at L ~ 20 and the
    FFT decisively ahead from L = 24 (0.82x) on.  The constants
    therefore stay.
    """
    return 2.0 ** d * (d * np.log2(2.0 * m) + 4.0)


# Near the crossover the cost model is only accurate to about a
# factor 1.5, so require that much modelled margin before leaving the
# dense path.
_FFT_MARGIN = 1.5

# …and, independently, never bother below this dense bag size: there
# the step is well under a millisecond either way and the FFT's fixed
# overhead (padding, transform setup) is what dominates.
_FFT_MIN_BAG = 1 << 16

# Memory budget for one peel's working set (phi chunk, padded spectra,
# inverse transform, output slice).  When the modelled working set of a
# full-bag peel exceeds this, the peel runs in chunks along one
# spectator axis — same arithmetic per fibre, so the values are
# bit-identical (gated as such); only the peak allocation changes.
# Mirrors tensor_network._MEM_BUDGET_BYTES.  Read at call time so tests
# can monkeypatch it.
_MEM_BUDGET_BYTES = 256 * 1024 * 1024

# Test hook: None = choose the chunk count from the budget (almost
# always 1, i.e. today's single-shot path); an int forces that chunk
# LENGTH along the spectator axis, which is how the equality gate
# drives chunk sizes 1/3/7 without a budget dance.
_FORCE_CHUNK = None

# Bag-volume threshold above which the "auto" dense strategy runs the
# axis-sliced branch instead of the fused broadcast product.  MEASURED
# on the shipped bucket shapes (bag 3-4, M1 Max, 2026-08): sliced is
# 3-7x slower below ~1e5 bag elements (per-slice overhead dominates),
# breaks even around ~5e5, and wins 1.5-3x wall clock above ~5e6 —
# plus an axis_size(w) factor of peak memory at every size (with the
# peel chunked, the dense full-bag product is the box's memory
# ceiling).  1e5 kept every d=1 rung of the then-shipped L <= 8
# ladder on broadcast (largest d=1 bag was K5's marker-halved
# 17^3 * 9 = 4.4e4 at L=8, where sliced is a measured 3.5-11.5x
# loss; this predates the widened d=1 ladder, L = 20..24) and flips
# the big UNPEELABLE d>=2 bags (K5/prism bag-4 buckets: 3.3e6 at
# d=2 L=3, 1.5e8 at d=3 L=2) that are the memory ceiling;
# peel-covered or marker-halved bags below 1e5 (all of K4's at d=2)
# correctly stay broadcast — the flip follows bag volume, not
# dimension.  Read at call time so tests can monkeypatch.
_DENSE_SLICE_MIN_VOLUME = 100_000


def _conv_possible(n_full, m, d):
    """Whether *any* step of this call could beat the dense path.

    Depends only on the box, so a call that fails it skips both the
    per-step check and the marker pre-pass and behaves exactly as
    before the FFT path existed.
    """
    return _FFT_MARGIN * _fft_step_factor(m, d) <= n_full


def _conv_partner(with_w, w):
    """Find the FFT-peelable factor in the bucket of ``w``.

    Eligible: a factor tagged ``conv_nu`` with scope ``{w, u}`` whose
    other endpoint ``u`` appears in *no other* bucket factor (else
    the remaining product still depends on ``x_u`` and the w-sum is
    not a convolution in ``x_u``).  Returns ``(u, factor)`` with the
    smallest such ``u`` (all candidates cost the same — deterministic
    tie-break), or ``None``.

    The Z₂ marker is *not* excluded (it was before the offset-aware
    peel): a half-box axis is just a window with a different origin
    and extent, and the peel derives its kernel sub-range, transform
    length and output slice from the two endpoint windows — see
    ``_fft_conv_step``.  When ``w`` is the marker, the peel folds the
    Z₂ weights into phi before transforming.
    """
    best = None
    for f in with_w:
        if "conv_nu" not in f:
            continue
        scope = f["scope"]
        if len(scope) != 2 or w not in scope:
            continue
        u = scope[0] if scope[1] == w else scope[1]
        if sum(1 for g in with_w if u in g["scope"]) != 1:
            continue
        if best is None or u < best[0]:
            best = (u, f)
    return best


def _eliminate_all(factors, elimination, marker, n_full, n_half,
                   z2_weights, A, L, d, kdiff_cache=None):
    """Bucket-elimination loop of the two direct-sum engines.

    Adapter over the shared skeleton in :mod:`gzl._contract`:
    the control flow (partition, empty bucket, peel gate, dense
    fallthrough) is :func:`_contract.eliminate`, and every arithmetic
    decision is a :class:`BoxTruncation` method whose body is this
    module's shipped code, moved verbatim.  Factors remain the dicts
    with ``scope``/``tensor`` (or lazy ``gen``/``ext``/``off``) and,
    on original two-vertex kernels, ``conv_nu``.

    The keyword signature is a de-facto public contract (tests call it
    directly); ``n_full``/``n_half``/``z2_weights`` are accepted and
    checked against the truncation's own geometry rather than trusted.
    """
    trunc = BoxTruncation(L, d, A, marker=marker,
                          kdiff_cache=kdiff_cache,
                          z2_weights=z2_weights)
    assert trunc.n_full == n_full and trunc.n_half == n_half, (
        f"inconsistent box geometry: caller says ({n_full}, {n_half}), "
        f"L={L} d={d} gives ({trunc.n_full}, {trunc.n_half})"
    )
    result = _contract.eliminate(factors, elimination, trunc)
    # Per-call diagnostic, REPLACED (never appended) so it cannot
    # accumulate across calls — the exact trap that once inverted a
    # pin measurement.  See _LAST_GATE_DECLINED.
    global _LAST_GATE_DECLINED
    _LAST_GATE_DECLINED = trunc.gate_declined
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _osc_freqs(momentum, terminal=None, root=None):
    """Fractional |k_i| that modulate the finite-k box tail (empty at k=0).

    The box faces sit at *integer* lattice coordinates ``n_i = +/-L``, so
    the modulation ``cos(2*pi*k_i*L)`` sees ``k_i`` only modulo 1 **and**
    only up to reflection: for integer ``L``,

        cos(2*pi*(1-f)*L) = cos(2*pi*f*L),
        sin(2*pi*(1-f)*L) = -sin(2*pi*f*L),

    so ``f`` and ``1-f`` generate the *same* column space.  The returned
    frequency is therefore folded into ``(0, 1/2]``.  Folding is not
    cosmetic: it is what makes the extrapolator respect the exact
    symmetry ``zeta_G(k) = zeta_G(-k) = zeta_G(1-k)``.  Reporting the
    unfolded ``f`` let a slow mode near ``f = 1`` masquerade as a fast
    one, so the adequacy gate below admitted a fit whose period the
    ladder could not span -- the raw box is symmetric to 1e-15 while the
    extrapolated grid differed between ``k`` and ``1-k`` by up to 5e-4.
    An integer component contributes ``cos(2*pi*m*L) = 1`` and does not
    oscillate at all.
    """
    if momentum is None:
        return []
    if (terminal is not None and root is not None
            and int(terminal) == int(root)):
        return []
    kf = np.asarray(momentum, dtype=float).reshape(-1)
    out = set()
    for x in kf:
        f = round(abs(float(x)) % 1.0, 12)
        f = min(f, 1.0 - f)
        if f > 1e-12:
            out.add(round(f, 12))
    return sorted(out)


# The ladder must span this many full periods of the slowest surviving
# mode before any phase-modulated column is admitted.  One period (the
# previous rule) is not enough: measured against an independent
# brute-force lattice sum on 4-vertex tw=3 corpus blocks, a one-period
# gate left the fit worse than the raw box at 40% (d=1) / 49% (d=2) of
# Brillouin-zone momenta, with blow-ups to 187x / 287x.
_TAIL_MIN_PERIODS = 2.0

# A decaying tail cannot need a correction much larger than the last
# observed rung-to-rung increment; a fit that asks for one is
# extrapolating noise, not tail.
_TAIL_CORRECTION_CLAMP = 4.0

# Below this relative rung-to-rung increment the box is already converged
# far past series precision (the surrounding method carries ~1e-4 at
# production n_points) and there is no tail left to remove; fitting one
# only amplifies round-off.  Measured on d=1 ladders of span 16 (L =
# 8..24, raw box accurate to 4.7e-10) and span 32 (L = 8..40, 7.6e-12):
# in both the modulated fit *lost* accuracy on a fifth of the momenta,
# by up to 21x, purely to round-off.  The lesson is that a longer
# finite-k ladder does not rescue the fit -- it makes the fit
# unnecessary, so raise L and skip the extrapolation instead.
_TAIL_CONVERGED_REL = 1e-7


def _has_unphased_free(edge_map, root, terminal):
    """True if some free vertex other than the terminal can escape.

    Only the terminal carries ``cos(2*pi*k.n_terminal)``, so only its
    escape produces an oscillating tail.  Any *other* free vertex
    reaching the box face contributes an un-modulated power tail.  When
    there is no such vertex (a bridge: the single free vertex is the
    terminal) the finite-k tail is purely modulated.
    """
    verts = {v for e in edge_map for v in e}
    free = {v for v in verts if v != int(root)}
    if terminal is not None:
        free.discard(int(terminal))
    return len(free) > 0


def _tail_design(L_arr, alpha, K, freqs, unphased=True):
    """Least-squares columns for the box-truncation tail.

    Column 0 is the constant ``S_inf`` -- the quantity being solved for.

    At finite ``k`` the tail has **two** structurally different parts and
    the basis must carry both:

    * *un-modulated* powers ``L^(alpha-j)``, admitted only when
      ``unphased`` is true.  Only the terminal vertex carries the phase
      ``cos(2*pi*k.n_terminal)``; every *other* free vertex escaping
      through the box face contributes a tail with no phase at all.
      Excluding these columns (the previous basis did) leaves the
      dominant non-oscillating tail unfitted, which is why the fit
      degraded at ``f = 1/2`` where the modulated columns collapse to
      ``(-1)^L``.  A graph whose only free vertex *is* the terminal --
      a bridge, say -- has no such tail, and there ``unphased`` is
      false so the columns are not spent on a mode that cannot exist.
    * *modulated* powers ``L^(alpha-j) cos/sin(2*pi*f*L)`` from the
      terminal escaping.

    At ``f = 1/2`` the sine column is identically zero and is dropped
    rather than handed to the solver as a null column.
    """
    L_arr = np.asarray(L_arr, dtype=float)
    cols = [np.ones_like(L_arr)]
    n_max = max(2, len(L_arr) - 1)      # keep >=1 residual degree of freedom
    j = 0
    while len(cols) < n_max and j < K + 2:
        if unphased:
            cols.append(L_arr ** (alpha - j))
        for f in freqs:
            if len(cols) >= n_max:
                break
            ph = 2.0 * np.pi * f * L_arr
            cols.append(L_arr ** (alpha - j) * np.cos(ph))
            if len(cols) < n_max and abs(f - 0.5) > 1e-12:
                cols.append(L_arr ** (alpha - j) * np.sin(ph))
        j += 1
    return np.column_stack(cols[:n_max])


def _lstsq_const(M, S):
    """Least-squares fit of ``S`` on ``M``; return the constant term."""
    if np.iscomplexobj(S):
        re, *_ = np.linalg.lstsq(M, S.real, rcond=None)
        im, *_ = np.linalg.lstsq(M, S.imag, rcond=None)
        return complex(re[0], im[0])
    sol, *_ = np.linalg.lstsq(M, S, rcond=None)
    return sol[0]


def _tail_extrapolate(L_arr, S, alpha, K, freqs, unphased=True):
    """Fit the box-truncation tail ``S(L)`` and return ``S_inf``.

    At ``k = 0`` the tail is the monotone Euler-Maclaurin series
    ``sum_j C_j L^(alpha-j)`` and the pure-power fit is applied
    unconditionally -- that path is where the extrapolation earns its
    keep (40-60x against exact closed forms).

    At finite ``k`` the summand carries ``cos(2*pi*k . n_terminal)``, so
    the box faces make part of the tail oscillate in ``L``.  Resolving
    such a mode from a handful of rungs is hard, and a fit that does not
    resolve it is *worse than not extrapolating at all*.  Three
    conditions must therefore all hold before a modulated fit is
    accepted; otherwise the raw box value at the largest ``L`` is
    returned unchanged:

    1. the ladder spans at least :data:`_TAIL_MIN_PERIODS` periods of the
       slowest surviving mode;
    2. dropping the top rung and refitting moves ``S_inf`` by no more
       than the correction being applied (the ladder's own consistency
       check -- if the two disagree by more than the correction, the
       correction is not resolved);
    3. the correction is at most :data:`_TAIL_CORRECTION_CLAMP` times the
       last observed rung-to-rung increment;
    4. the ladder has not already converged (:data:`_TAIL_CONVERGED_REL`)
       -- once the box is converged there is no tail to remove and a fit
       only amplifies round-off.

    With these, the finite-k path is never worse than the raw box on the
    measured set (0 of 1152 momenta at d=1, 0 of 768 at d=2, against an
    independent brute-force reference); the previous single-period gate
    was worse at 40% / 49% of momenta respectively.

    Shared by :func:`direct_sum_extrapolated` and
    :func:`_direct_sum_extrapolated_grid` so the two cannot drift apart:
    the grid form needs one design per momentum, since the basis depends
    on ``k``.
    """
    L_arr = np.asarray(L_arr, dtype=float)
    S = np.asarray(S)
    order = np.argsort(L_arr)
    raw = S[order[-1]]

    if not freqs:
        cols = ([np.ones_like(L_arr)]
                + [L_arr ** (alpha - j) for j in range(K)])
        return _lstsq_const(np.column_stack(cols), S)

    # (1) adequacy of the ladder for the slowest mode.
    span = float(L_arr.max() - L_arr.min())
    if span * min(freqs) < _TAIL_MIN_PERIODS:
        return raw
    if L_arr.size < 3:
        return raw

    M = _tail_design(L_arr, alpha, K, freqs, unphased)
    value = _lstsq_const(M, S)
    if not np.all(np.isfinite(np.asarray(value))):
        return raw

    # (2) leave-out-the-top-rung consistency.
    keep = order[:-1]
    M2 = _tail_design(L_arr[keep], alpha, K, freqs, unphased)
    value2 = _lstsq_const(M2, S[keep])
    correction = abs(value - raw)
    if not np.isfinite(abs(value2)) or abs(value2 - value) > correction:
        return raw

    # (3) the correction may not dwarf the last observed increment, and
    #     (4) there must still be a tail worth removing at all.
    last_step = abs(S[order[-1]] - S[order[-2]])
    if correction > _TAIL_CORRECTION_CLAMP * last_step:
        return raw
    if last_step <= _TAIL_CONVERGED_REL * abs(raw):
        return raw

    return value


def direct_sum_zero_momentum(
    edges,
    nu,
    A,
    L: int,
    *,
    root: Optional[int] = None,
    use_symmetry: bool = True,
    momentum: "np.ndarray | None" = None,
    terminal: Optional[int] = None,
    kernels: "Sequence | None" = None,
) -> float:
    r"""Truncated lattice sum of the graph zeta function.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Pins the vertex ``root`` (auto-selected as the maximum-degree vertex
    if not given) at the lattice origin and sums the remaining
    :math:`V - 1` free vertex positions over the box
    :math:`[-L, L]^d` via variable elimination (planner-chosen order
    at the fixed pin; min-degree fallback on budget refusal).

    Finite external momentum (optional).  Passing ``momentum`` (a
    fractional-Brillouin-zone vector, length ``d``) together with
    ``terminal`` and an explicit ``root`` (= the source vertex,
    pinned at the origin) multiplies the summand by the **real**
    weight :math:`\cos\!\bigl(2\pi\,\boldsymbol{k}\cdot
    \boldsymbol{x}_\text{terminal}\bigr)`.  ζ_G(k) is real on any
    Bravais lattice (the global inversion x→−x leaves the kernel
    product invariant and flips the displacement, so the imaginary
    sine part cancels — the same trick as the tensor single-k
    cos-weight), and crucially the cos weight is *even* under that
    inversion, so the Z₂ marker halving below remains valid.  The
    phase enters as one extra 1-axis factor on ``terminal``; the
    leading box-truncation exponent (and hence
    :func:`direct_sum_extrapolated`) is unchanged because ``cos`` is
    bounded.  ``terminal == root`` (or ``momentum is None``) ⇒ the
    k = 0 sum.

    Parameters
    ----------
    edges : array_like (n, 2), int
        Indexed list of undirected edges.  Multigraphs are encoded by
        repeating rows; the routine collapses them internally.
    nu : array_like (n,), float
        Per-edge exponents.  Must satisfy ``nu_e > d`` for the
        infinite-lattice sum to converge.
    A : array_like (d, d), float, or str
        Lattice basis, or the name of a lattice -- ``"chain"``,
        ``"square"``, ``"triangular"``, ``"cubic"``.  Any ``d >= 1`` (the
        elimination, marker and peel machinery are dimension-generic;
        cost, not capability, bounds practical ``d``).
    L : int
        Half-width of the truncation box (free vertex positions range
        over the integer multi-indices :math:`[-L, L]^d`).
    root : int, optional
        Vertex pinned at the origin.  Defaults to the maximum-degree
        vertex of the simple graph (after multi-edge collapse).
    kernels : sequence of Interaction-likes, optional
        One :class:`gzl.interaction.Interaction` (or lazy
        product) per row of ``edges``, replacing the power law on that
        edge; ``nu`` then carries the per-edge TAIL exponents (what the
        planner, the marker choice and the FFT opt-out read; a row
        whose ``nu`` and kernel disagree on the tail's finiteness is
        refused).  Parallel rows multiply pointwise, exactly as the
        exponents add.  A purely compact kernel (infinite tail)
        contracts dense like the ``nu = inf`` indicator, so an integer
        table stays exact; a table label outside ``[-2L, 2L]^d`` raises
        :class:`gzl.interaction.InteractionSupportError`.
        ``None`` (default) is the legacy power-law path, unchanged.

        **Fixed-L semantics with a compact part.**  This is a box
        TRUNCATION: a term in which a free vertex is tied to the root
        through table factors only is held exactly when ``L`` is at
        least the compact reach (:func:`_compact_reach`, the
        radius-weighted shortest-path distance over the compact
        subgraph from ``root``) and CLIPPED below it — a truncation
        error with no power-law form, which is why the Richardson entry
        points refuse a ladder whose smallest rung is short
        (:func:`_check_compact_reach`).  A block with no finite tail at
        ``root`` — every escaping cut carries a purely compact kernel,
        so the Richardson basis exponent is not finite — has no tail to
        truncate: the sum is exact at ``L >= reach`` and simply wrong
        below it (a purely compact 2-path with radius-2 tables returns
        0 at ``L = 1`` and 2 at ``L = 2, 3`` for an exact 4), so
        ``L < reach`` is refused there with
        :class:`~gzl.interaction.InteractionSupportError`;
        :func:`direct_sum_extrapolated` lifts its one rung to the reach
        instead (:func:`_degenerate_rung`).

    Returns
    -------
    float
        :math:`\sum^\prime_{\boldsymbol{x}^{(1)}, \dots, \boldsymbol{x}^{(V-1)} \in [-L, L]^d}
        \prod_e |\boldsymbol{x}^{(v_e)} - \boldsymbol{x}^{(u_e)}|^{-\nu_e}`.

    Raises
    ------
    ValueError
        On self-loops, empty edge lists, a non-square ``A`` or
        ``d < 1``, or finite ``momentum`` without an explicit
        ``root``.

    Notes
    -----
    Convergence to the infinite-lattice limit is algebraic with leading
    order :math:`L^{d - \nu_{\min}}`.  Use
    :func:`direct_sum_extrapolated` for a Richardson-extrapolated value
    that is several digits more accurate at the cost of a few extra
    truncations.

    Algorithm: variable elimination over a tensor representation of
    each edge propagator.  The maximum intermediate-tensor rank is
    ``treewidth(G) + 1``, so the cost is
    :math:`\mathcal{O}\bigl(V \cdot (2L+1)^{d (\mathrm{tw}+1)}\bigr)`.
    For typical perturbative-series graphs (treewidth :math:`\le 4`,
    :math:`V \lesssim 10`) on the chain at :math:`L \le 20` this is a
    sub-second computation.
    """
    _refuse_interaction_in_nu(nu, "direct_sum_zero_momentum")
    edges = np.asarray(edges, dtype=int)
    nu = np.asarray(nu, dtype=float)
    A = np.asarray(_resolve_lattice(A), dtype=float)
    L = int(L)

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(
            f"direct_sum: edges must have shape (n, 2); got {edges.shape}."
        )
    if nu.shape != (edges.shape[0],):
        raise ValueError(
            f"direct_sum: nu must have shape ({edges.shape[0]},); "
            f"got {nu.shape}."
        )
    d = int(A.shape[0])
    if A.ndim != 2 or A.shape[0] != A.shape[1] or d < 1:
        raise ValueError(
            "direct_sum_zero_momentum requires a square lattice basis "
            f"with d >= 1; got A.shape = {A.shape}."
        )
    if L < 1:
        raise ValueError(f"direct_sum: L must be >= 1; got {L}.")

    # The vertex set is the edge support (see gzl/_labels.py):
    # sparse labels compress order-preservingly (bit-identical to the
    # contiguous twin; a contiguous input passes through unchanged), and
    # a root or terminal label that appears in no edge raises.  Without
    # this, every gap label was summed freely over the box — a factor
    # (2L+1)**d each — for a graph whose infinite-lattice sum that
    # factor does not approximate.  Self-loops are named BEFORE the
    # relabel so the diagnostic carries the caller's label, not the
    # compressed one.
    _sl = edges[:, 0] == edges[:, 1] if edges.size else np.zeros(0, bool)
    if _sl.any():
        raise ValueError(
            f"direct_sum: self-loop at vertex "
            f"{int(edges[int(np.argmax(_sl)), 0])}; not allowed."
        )
    edges, _refs = _relabel_to_support(edges, {"root": root,
                                               "terminal": terminal})
    root, terminal = _refs["root"], _refs["terminal"]

    # Lattice positions in d dims.
    #
    # ``pos_full`` is the flattened (2L+1)^d grid over [-L, L]^d.
    # ``pos_half`` is the flattened (L+1)·(2L+1)^(d-1) grid where the
    # *first* coordinate is restricted to [0, L] and the remaining
    # (d - 1) coordinates run over [-L, L].  By -Λ = Λ symmetry of the
    # integrand, scaling the marker vertex by 1 at x1=0 and by 2 at
    # x1 > 0 recovers the full sum from this half-box.
    n_full = (2 * L + 1) ** d
    n_half = (L + 1) * (2 * L + 1) ** (d - 1)
    axes_full = [np.arange(-L, L + 1, dtype=float)] * d
    grid_full = np.stack(np.meshgrid(*axes_full, indexing="ij"), axis=-1)
    pos_full = grid_full.reshape(-1, d)                  # (n_full, d)
    axes_half = [np.arange(0, L + 1, dtype=float)] \
                + [np.arange(-L, L + 1, dtype=float)] * (d - 1)
    grid_half = np.stack(np.meshgrid(*axes_half, indexing="ij"), axis=-1)
    pos_half = grid_half.reshape(-1, d)                  # (n_half, d)

    # Z_2 weight along the marker axis: 1 at first_coord = 0, 2 elsewhere.
    z2_weights = np.where(pos_half[:, 0] == 0.0, 1.0, 2.0)

    # 1. Collapse multi-edges.  The float map keeps carrying the summed
    #    TAIL exponents (what the planner, the marker and the FFT
    #    opt-out read); with ``kernels`` a parallel map of lazy kernel
    #    products supplies what the factor build samples.
    edge_map = _collapse_multi_edges(edges, nu)
    kern_map = (None if kernels is None
                else _collapse_multi_kernels(edges, nu, kernels))

    # 2. Vertex set and adjacency.
    V = max(max(u, v) for u, v in edge_map) + 1
    adj = _adjacency(edge_map, V)

    # 3. Choose origin.
    if momentum is not None and root is None:
        raise ValueError(
            "direct_sum: finite momentum requires an explicit `root` "
            "(= the source vertex pinned at the origin), so the phase "
            "is cos(2π k · x_terminal) with x_source = 0."
        )
    if root is None:
        root = _pick_root(edge_map, V)
    if not (0 <= root < V):
        raise ValueError(
            f"direct_sum: root={root} out of range for V={V}."
        )

    free = [v for v in range(V) if v != root]
    if not free:
        return 1.0 + 0.0j

    # 3b. No finite tail at this root => not a truncation of anything.
    #     With every escaping cut purely compact (the basis exponent
    #     d - _min_free_cut_nu is -inf, the same test the Richardson
    #     entry points key their one-rung branch on) the box holds every
    #     table-tied term exactly at L >= reach and clips them below it:
    #     the purely compact 2-path with radius-2 tables returned 0 at
    #     L = 1 and 2 at L = 2, 3 for an exact 4, with no diagnostic.
    #     Refused in the class _check_compact_reach raises, so the
    #     front-end's (ValueError, MemoryError) nets fall through.  A
    #     block WITH a finite tail keeps the fixed-L truncation semantics
    #     documented above (clipped below the reach; the ladder's
    #     smallest-rung guard is what protects a fit).
    if kern_map is not None:
        if not math.isfinite(d - _min_free_cut_nu(edge_map, root)):
            reach = _compact_reach(kern_map, edge_map, root)
            if L < reach:
                raise InteractionSupportError(
                    f"direct_sum_zero_momentum: the block has no finite "
                    f"tail at root {root} (every escaping cut carries a "
                    f"purely compact kernel), so the box sum is exact only "
                    f"for L >= the compact reach {reach} (the "
                    f"radius-weighted shortest-path distance over the "
                    f"compact subgraph) and clips table-tied terms below "
                    f"it; got L = {L}.  Use L >= {reach}, "
                    f"direct_sum_extrapolated (which sums the one rung at "
                    f"the reach), or a torus engine."
                )

    # 4. Elimination order: the exact planner at the fixed pin, the
    #    shipped min-degree heuristic on budget refusal (the pin itself
    #    is never re-chosen — the box is not translation-closed).  The
    #    finite-momentum cos weight is declared as an extra planner
    #    scope, which keeps the REPORTED plan (its bags and peel
    #    predictions, hence Plan.exponent) honest about the factor the
    #    engine will meet; at a fixed pin the chosen ORDER itself is
    #    unaffected (the DP walks the adjacency, not the scopes, and
    #    with one candidate pin the rank comparison never runs).  The Z₂
    #    marker is the last eliminated vertex, unconditionally; the
    #    offset-aware peel handles marker-touching steps (a half axis
    #    is just a window with a different origin and extent), so the
    #    old symbolic pre-pass that traded the guaranteed factor 2
    #    against an FFT-blocking half axis has nothing left to trade
    #    and was deleted.
    _extra = ([(int(terminal),)]
              if (momentum is not None and terminal is not None
                  and int(terminal) != int(root)) else [])
    _inf_pairs = tuple(k for k, nu_e in edge_map.items()
                       if np.isinf(nu_e))
    elimination = _planned_or_min_degree(edge_map, V, root, adj, free,
                                         extra_scopes=_extra,
                                         non_kernel_edges=_inf_pairs)
    marker = elimination[-1] if (use_symmetry and elimination) else None

    def axis_size(v: int) -> int:
        return n_half if v == marker else n_full

    def positions_for(v: int) -> np.ndarray:
        """Return ``(n_axis, d)`` lattice positions for vertex ``v``."""
        return pos_half if v == marker else pos_full

    def extents_for(v: int) -> tuple:
        return _axis_extents(L, d, v == marker)

    def origins_for(v: int) -> tuple:
        return _axis_origins(L, d, v == marker)

    # 5. Build initial factors.  Kernel factors are stored lazily as
    #    (generator, extents, offsets): the generator is one array over
    #    the difference range [-2L, 2L]^d shared by every factor of the
    #    same nu, and the (size_a, size_b) table is read out of it only
    #    when a consumer needs the dense form.  Each factor is consumed
    #    by exactly one elimination step, so nothing is built twice and
    #    the tables are never all live simultaneously.
    #
    #    kdiff_cache lives at call level so one generator per distinct
    #    nu serves both the factor build and the FFT peel; it also
    #    carries the peel's padded transform, which is otherwise
    #    recomputed at every step that peels the same kernel.
    kdiff_cache: dict = {}

    def generator_for(nu_e):
        g = kdiff_cache.get(nu_e)
        if g is None:
            g = _conv_kernel_diff(nu_e, A, L, d)
            kdiff_cache[nu_e] = g
        return g

    factors: List[dict] = []

    for (u, v), nu_e in edge_map.items():
        # The pair's generator and its cache key: the legacy float path
        # verbatim, or the kernel map's sampled kernel keyed on its
        # key() (see _kernel_cache_key) — never on the tail alone.
        if kern_map is None:
            gen, tag, kern = generator_for(nu_e), float(nu_e), None
        else:
            kern = kern_map[(u, v)]
            gen, tag = _kernel_generator(kdiff_cache, kern, A, L, d)
        if u == root and v == root:
            raise RuntimeError("Unreachable: edge from root to root.")
        elif u == root or v == root:
            other = v if u == root else u
            # K(x_other - x_root) with x_root = 0, so this is the
            # generator's own i-slice: offsets are the origins shifted
            # to the generator's zero at index 2L.
            org = origins_for(other)
            factors.append({
                "scope": (other,),
                "gen": gen,
                "ext": (extents_for(other),),
                "off": tuple(int(org[c] + 2 * L) for c in range(d)),
                "d": d,
            })
        else:
            scope = (min(u, v), max(u, v))
            # conv_nu marks an original kernel eligible for the FFT
            # elimination fast path and names its kdiff_cache entry
            # (the float tail, or the kernel key).  K is even
            # (‖A(−x)‖ = ‖Ax‖), so one difference-range sample serves
            # both peel directions; an orientation flag would be needed
            # only if a non-even kernel were ever introduced.  A ν = ∞
            # indicator bundle never gets the tag: the NN contraction
            # is exact integer arithmetic and an FFT round trip returns
            # 2.0 as 1.9999999999999998 (the torus rule, tensor_network's
            # peelable[key]) — and neither does a purely compact
            # kernel, whose tail is ∞ too, so an integer table stays
            # exact.  A mixed kernel with a finite tail peels like the
            # power law it decays as.
            f = {
                "scope": scope,
                "gen": gen,
                "ext": (extents_for(scope[0]), extents_for(scope[1])),
                "off": _gen_offsets(L, d, origins_for(scope[0]),
                                    origins_for(scope[1])),
                "d": d,
            }
            if not np.isinf(nu_e):
                f["conv_nu"] = tag
                if kern is not None:
                    # What a cache miss inside the peel (a hand-built
                    # factor list) rebuilds the generator from.
                    f["conv_kernel"] = kern
            factors.append(f)

    # 5b. Finite-momentum phase.  Pin = root = source (x_source = 0),
    #     so the displacement is x_terminal and the phase is the real,
    #     even weight cos(2π k · n_terminal) where n_terminal is the
    #     *integer* lattice coordinate (conjugate to the fractional-BZ
    #     momentum, matching the tensor single-k cos-weight
    #     convention).  It enters as one extra 1-axis factor on the
    #     terminal vertex; positions_for() respects the marker
    #     half-axis, and cos is even under the global x→−x inversion so
    #     the Z₂ marker weighting stays exact.
    if (momentum is not None and terminal is not None
            and int(terminal) != int(root)):
        tt = int(terminal)
        if not (0 <= tt < V):
            raise ValueError(
                f"direct_sum: terminal={tt} out of range for V={V}."
            )
        kf = np.asarray(momentum, dtype=float).reshape(-1)
        if kf.size != d:
            raise ValueError(
                f"direct_sum: momentum has length {kf.size}, expected "
                f"d={d}."
            )
        pos_t = positions_for(tt)                     # (n_axis, d) ints
        phase = np.cos(2.0 * np.pi * (pos_t @ kf))    # (n_axis,)
        factors.append({"scope": (tt,), "tensor": phase})

    # 6. Eliminate free vertices in order (FFT fast path per step
    #    where one original kernel can be peeled; dense otherwise).
    factors = _eliminate_all(
        factors, elimination, marker, n_full, n_half, z2_weights,
        A, L, d, kdiff_cache,
    )

    # 7. Multiply remaining scalars (should all have empty scope).
    result = np.array(1.0, dtype=np.float64)
    for f in factors:
        if len(f["scope"]) > 0:
            raise RuntimeError(
                f"direct_sum: non-scalar factor remains, scope={f['scope']}."
            )
        result = result * _factor_array(f)

    return _as_real(result, where="direct_sum_zero_momentum")


# ---------------------------------------------------------------------------
# Single-shot full-BZ direct sum (terminal kept as an open axis)
# ---------------------------------------------------------------------------

def _direct_sum_open_terminal(edge_map, A, L, root, terminal, d,
                              use_symmetry=True, kern_map=None):
    r"""Eliminate every free vertex **except** ``terminal`` and return
    the residual ``M(x_terminal)`` over the terminal's truncation box.

    ``kern_map`` (from :func:`_collapse_multi_kernels`) supplies the
    per-pair kernels the factor build samples; ``edge_map`` keeps the
    float tails either way.

    Same variable elimination as
    :func:`direct_sum_zero_momentum` (planner order at the fixed pin,
    min-degree fallback) with ``root`` pinned at the
    origin, but ``terminal`` is *kept as an open axis* (never summed,
    never the Z₂ marker).  The full-BZ dispersion is then a single
    cosine transform of ``M`` (done by the caller), and that transform
    is real.

    **The returned ``M`` is not itself even in ``x_terminal`` when the
    Z₂ marker is active.**  The fold halves a *free* vertex's axis under
    the global x→−x inversion, and that inversion flips the terminal
    too, so for fixed ``x_terminal`` the fold does not close on itself:
    writing S⁺, S⁻, S⁰ for the parts of the sum with the marker's first
    coordinate positive, negative and zero, the engine returns
    ``2S⁺ + S⁰`` while the true residual is ``S⁺(x_t) + S⁺(−x_t) + S⁰``.
    The two agree only after symmetrisation —
    ``½(M(x_t) + M(−x_t))`` is the true residual, exactly — and the
    caller's cosine transform performs that symmetrisation implicitly
    because ``cos`` is even.  So every value this feeds is correct, but
    ``M`` on its own is a half-folded intermediate: do not compare it
    pointwise against an unfolded reference, and do not "fix" its
    asymmetry.  Pinned by ``TestOpenTerminalSymmetrisation`` in
    ``tests/test_executor_goldens.py`` (measured asymmetry 0.66 at d=1,
    0.50 at d=2; symmetrised agreement 1.4e-16 / 5.7e-16).

    Returns ``(M, term_pos)`` with ``M`` shape ``(n_full,)`` and
    ``term_pos`` shape ``(n_full, d)`` the integer terminal
    coordinates (conjugate to fractional-BZ momentum).
    """
    V = max(max(u, v) for u, v in edge_map) + 1
    adj = _adjacency(edge_map, V)

    n_full = (2 * L + 1) ** d
    n_half = (L + 1) * (2 * L + 1) ** (d - 1)
    axes_full = [np.arange(-L, L + 1, dtype=float)] * d
    pos_full = np.stack(
        np.meshgrid(*axes_full, indexing="ij"), axis=-1,
    ).reshape(-1, d)
    axes_half = [np.arange(0, L + 1, dtype=float)] \
                + [np.arange(-L, L + 1, dtype=float)] * (d - 1)
    pos_half = np.stack(
        np.meshgrid(*axes_half, indexing="ij"), axis=-1,
    ).reshape(-1, d)
    z2_weights = np.where(pos_half[:, 0] == 0.0, 1.0, 2.0)

    free = [v for v in range(V) if v != root]
    _inf_pairs = tuple(k for k, nu_e in edge_map.items()
                       if np.isinf(nu_e))
    elimination = _planned_or_min_degree(edge_map, V, root, adj, free,
                                         keep=(int(terminal),),
                                         non_kernel_edges=_inf_pairs)
    # The Z₂ marker is the last summed-out vertex, unconditionally
    # (never the terminal, which the planner keeps open via ``keep``
    # and the fallback excludes from ``elim_free``); marker-touching
    # steps peel offset-aware, so no pre-pass trades the half axis
    # against the FFT any more.
    marker = elimination[-1] if (use_symmetry and elimination) else None

    def axis_size(v):
        return n_half if v == marker else n_full

    def positions_for(v):
        return pos_half if v == marker else pos_full

    def extents_for(v):
        return _axis_extents(L, d, v == marker)

    def origins_for(v):
        return _axis_origins(L, d, v == marker)

    # Same lazy factor supply as direct_sum_zero_momentum; see there.
    kdiff_cache: dict = {}

    def generator_for(nu_e):
        g = kdiff_cache.get(nu_e)
        if g is None:
            g = _conv_kernel_diff(nu_e, A, L, d)
            kdiff_cache[nu_e] = g
        return g

    factors: List[dict] = []
    for (u, v), nu_e in edge_map.items():
        # Generator and cache key per pair, as in the zero-momentum
        # factor build: legacy float path, or the kernel map's entry.
        if kern_map is None:
            gen, tag, kern = generator_for(nu_e), float(nu_e), None
        else:
            kern = kern_map[(u, v)]
            gen, tag = _kernel_generator(kdiff_cache, kern, A, L, d)
        if u == root and v == root:
            raise RuntimeError("Unreachable: edge from root to root.")
        if u == root or v == root:
            other = v if u == root else u
            org = origins_for(other)
            factors.append({
                "scope": (other,),
                "gen": gen,
                "ext": (extents_for(other),),
                "off": tuple(int(org[c] + 2 * L) for c in range(d)),
                "d": d,
            })
        else:
            scope = (min(u, v), max(u, v))
            f = {
                "scope": scope,
                "gen": gen,
                "ext": (extents_for(scope[0]), extents_for(scope[1])),
                "off": _gen_offsets(L, d, origins_for(scope[0]),
                                    origins_for(scope[1])),
                "d": d,
            }
            # ν = ∞ bundles (and purely compact kernels) never peel —
            # integer-exact contraction (see the zero-momentum factor
            # build).
            if not np.isinf(nu_e):
                f["conv_nu"] = tag
                if kern is not None:
                    f["conv_kernel"] = kern
            factors.append(f)

    # Terminal streaming: batched prefix, per-position suffix.
    # A batched step whose result tensor (bag axes minus the eliminated
    # vertex, terminal axis included) would exceed _MEM_BUDGET_BYTES
    # switches the remaining schedule to pinning the terminal at one
    # position at a time — the suffix bags then drop the n_full
    # terminal axis, exactly the torus _streaming_path strategy.  The
    # split point is found SYMBOLICALLY (scopes depend only on the
    # eliminated set, the same fact the planner's DP rests on), so the
    # fully-batched common case runs the one shipped _eliminate_all
    # call, untouched.
    k_split = len(elimination)
    if _FORCE_STREAMING:
        k_split = 0
    else:
        sym = {s.vertex: s for s in _elimination.simulate(
            _elimination.scopes_from_edges(edge_map, root),
            elimination)}
        for k, vv in enumerate(elimination):
            s = sym.get(vv)
            if s is None:
                continue
            size = 1
            for a in s.bag:
                if a != vv:
                    size *= axis_size(a)
            if 8 * size > _MEM_BUDGET_BYTES:
                k_split = k
                break

    if k_split >= len(elimination):
        factors = _eliminate_all(
            factors, elimination, marker, n_full, n_half, z2_weights,
            A, L, d, kdiff_cache,
        )

        # Remaining factors have scope ⊆ {terminal}; combine into M(x_t).
        M = np.ones(n_full, dtype=np.float64)
        for f in factors:
            sc = f["scope"]
            if sc == ():
                M = M * _factor_array(f)
            elif sc == (terminal,):
                M = M * _factor_array(f)
            else:
                raise RuntimeError(
                    f"direct_sum open-terminal: unexpected residual "
                    f"scope {sc} (terminal={terminal})."
                )
        return M, pos_full

    # --- streamed schedule -------------------------------------------
    global _LAST_GATE_DECLINED
    prefix_declines: list = []
    cur = factors
    if k_split > 0:
        cur = _eliminate_all(
            cur, elimination[:k_split], marker, n_full, n_half,
            z2_weights, A, L, d, kdiff_cache,
        )
        prefix_declines = _LAST_GATE_DECLINED

    suffix = elimination[k_split:]
    t_ext = _axis_extents(L, d, False)   # the terminal is never the marker
    # Materialise 1-axis lazy terminal rows ONCE (O(n_full)); repeating
    # the gather inside the position loop would total a table's worth
    # of work.  Two-axis lazy kernels stay lazy — they are sliced per
    # position straight off the generator below.
    cur = [
        f if (f.get("tensor") is not None
              or terminal not in f["scope"]
              or len(f["scope"]) == 2)
        else {"scope": f["scope"],
              "tensor": np.asarray(_factor_array(f))}
        for f in cur
    ]
    M = np.empty(n_full, dtype=np.float64)
    suffix_declines: list = []
    for t_flat in range(n_full):
        t_idx = np.unravel_index(t_flat, t_ext)
        pinned: list = []
        for f in cur:
            sc = f["scope"]
            if terminal not in sc:
                pinned.append(f)
                continue
            if f.get("tensor") is not None or len(sc) == 1:
                # Dense factor, or a 1-axis lazy row: materialise
                # (O(size) for the row) and take the position.  The
                # residue stays SHAPED — one array axis per scope
                # vertex, the invariant _dense_step_sliced and the
                # chunked peel index by (idx[scope.index(w)]); a
                # flattened multi-vertex residue crashes both exactly
                # in the memory-pressure regime streaming serves.
                arr = np.asarray(_factor_array(f))
                arr = arr.reshape([axis_size(a) for a in sc]) \
                    if len(sc) > 1 else arr
                pos = sc.index(terminal)
                arr = np.take(arr, t_flat, axis=pos)
                new_sc = tuple(a for a in sc if a != terminal)
                pinned.append({"scope": new_sc,
                               "tensor": np.asarray(arr)})
            else:
                # Two-axis lazy kernel: gather the pinned row/column
                # straight off the generator — never build the table.
                pos = sc.index(terminal)
                other = sc[1 - pos]
                ext_keep = f["ext"][1 - pos]
                vec = _box_slice_at(f["gen"], ext_keep, f["off"], d,
                                    t_idx, keep_is_b=(pos == 0))
                pinned.append({"scope": (other,), "tensor": vec})
        res = _eliminate_all(
            pinned, suffix, marker, n_full, n_half, z2_weights,
            A, L, d, kdiff_cache,
        )
        if t_flat == n_full - 1:
            suffix_declines = _LAST_GATE_DECLINED
        s_val = np.array(1.0, dtype=np.float64)
        for f in res:
            if len(f["scope"]) > 0:
                raise RuntimeError(
                    f"direct_sum open-terminal (streamed): unexpected "
                    f"residual scope {f['scope']} "
                    f"(terminal={terminal})."
                )
            s_val = s_val * _factor_array(f)
        M[t_flat] = float(np.asarray(s_val))
    # One aggregated per-call publication: prefix declines plus one
    # representative suffix pass (the suffix decline pattern is
    # structurally identical across terminal positions).
    _LAST_GATE_DECLINED = prefix_declines + suffix_declines
    return M, pos_full


def _direct_sum_extrapolated_grid(
    edges, nu, A, k_grid, *, root, terminal,
    L_list=(4, 5, 6, 7, 8), n_correction_terms=3, kernels=None,
):
    r"""Single-shot full-Brillouin-zone dispersion via the direct sum.

    For each ``L`` the k-independent elimination is run **once**
    (``terminal`` open) → ``M_L(x_t)``; the whole momentum grid is then
    ``Σ_{x_t} cos(2π k·x_t) M_L(x_t)`` (one matmul, real because the cos
    weight is — and, cos being even, the transform also performs the
    symmetrisation that makes the value exact even though ``M`` itself
    is NOT even in ``x_t`` while the Z₂ marker is active; see
    ``_direct_sum_open_terminal``).  The per-L grids are Richardson-extrapolated
    *element-wise* with the degree-aware leading exponent
    ``d − min_{v free} Σ_{e∋v} ν_e`` (3a) via a single multi-RHS
    least-squares solve.

    This is the direct-sum analogue of the tensor ``space='z'`` →
    ``ifftn`` single-shot path: ``len(L_list)`` eliminations + one
    transform, **not** one elimination per k-point.

    Parameters
    ----------
    k_grid : ndarray, shape (n_k, d)
        Fractional-BZ momenta (rows).
    root, terminal : int
        Source (pinned at origin) and terminal vertex; ``root`` is
        required and must differ from ``terminal``.
    kernels : sequence of Interaction-likes, optional
        Per-edge kernels, as in :func:`direct_sum_zero_momentum`.  With
        a compact part every rung must hold the reach
        (:func:`_compact_reach`), else
        :class:`gzl.interaction.InteractionSupportError`; a block
        with no finite tail ships one exact rung instead of a fit
        (:func:`_degenerate_rung`), at every momentum.

    Returns
    -------
    ndarray, shape (n_k,), real
        ``ζ_G(k)`` for every row of ``k_grid``.
    """
    _refuse_interaction_in_nu(nu, "_direct_sum_extrapolated_grid")
    edges = np.asarray(edges, dtype=int)
    nu = np.asarray(nu, dtype=float)
    A = np.asarray(A, dtype=float)
    d = int(A.shape[0])
    k_grid = np.asarray(k_grid, dtype=float).reshape(-1, d)

    L_list = sorted(int(L) for L in L_list)
    if not L_list:
        raise ValueError("_direct_sum_extrapolated_grid: L_list is empty.")
    K = int(n_correction_terms)
    if root is None or terminal is None or int(root) == int(terminal):
        raise ValueError(
            "_direct_sum_extrapolated_grid: distinct explicit root "
            "(source) and terminal are required."
        )
    root, terminal = int(root), int(terminal)

    edge_map = _collapse_multi_edges(edges, nu)
    kern_map = (None if kernels is None
                else _collapse_multi_kernels(edges, nu, kernels))
    nu_min = _checked_nu_min(edge_map, "_direct_sum_extrapolated_grid")
    if not (nu_min > d):
        raise UnsupportedLatticeSumError(
            f"_direct_sum_extrapolated_grid: nu_min={nu_min} <= d={d} is "
            "not supported (see UnsupportedLatticeSumError)."
        )
    nu_eff_lead = _min_free_cut_nu(edge_map, root)
    _check_alpha_basis(edge_map, root, nu_eff_lead)
    alpha = d - nu_eff_lead
    has_unphased = _has_unphased_free(edge_map, root, terminal)

    # Compact parts must fit every rung (_compact_reach), and a block
    # with no finite tail has nothing to fit: one exact rung, no
    # ladder (_degenerate_rung) — keyed on the basis exponent, never
    # on the L values.
    if not np.isfinite(alpha):
        # No rung of L_list is fitted here: the one evaluated rung is
        # lifted to the reach, so neither the smallest-rung check nor
        # the ladder-length check (K + 1 rungs for a K-term fit) applies
        # — no fit is performed.
        reach = (0 if kern_map is None
                 else _compact_reach(kern_map, edge_map, root))
        L_list = [_degenerate_rung(edge_map, root, L_list, reach,
                                   kern_map, A, d)]
    else:
        if K + 1 > len(L_list):
            raise ValueError(
                f"_direct_sum_extrapolated_grid: need K+1={K+1} L values, "
                f"got {len(L_list)}."
            )
        if kern_map is not None:
            _check_compact_reach(kern_map, edge_map, root, L_list,
                                 "_direct_sum_extrapolated_grid")

    grids = []
    for L in L_list:
        M, term_pos = _direct_sum_open_terminal(
            edge_map, A, L, root, terminal, d, kern_map=kern_map,
        )
        # cos(2π k·x_t) is even ⇒ the transform is real and equals the
        # real part of the would-be exp phase (ζ_G(k) ∈ ℝ).
        C = np.cos(2.0 * np.pi * (k_grid @ term_pos.T))   # (n_k, n_full)
        grids.append(C @ M)                               # (n_k,)
    grids = np.asarray(grids)                             # (n_L, n_k)

    if not np.isfinite(alpha):
        # The one rung IS the value at every momentum: at k = 0 the
        # power fit would degenerate to the mean of the rungs, and at
        # finite k the modulated basis has no tail to resolve either.
        return np.asarray(grids[0], dtype=float)

    L_arr = np.array(L_list, dtype=float)
    # The finite-k basis is momentum-dependent (it is modulated at
    # frequency k_i), so unlike the k=0 power basis it cannot be shared
    # across the grid -- one small fit per momentum, negligible beside
    # the eliminations above.
    out = np.empty(k_grid.shape[0], dtype=float)
    for ik in range(k_grid.shape[0]):
        out[ik] = float(np.real(_tail_extrapolate(
            L_arr, grids[:, ik], alpha, K, _osc_freqs(k_grid[ik]),
            unphased=has_unphased,
        )))
    return out


# ---------------------------------------------------------------------------
# Richardson extrapolation
# ---------------------------------------------------------------------------

def _checked_nu_min(edge_map, where: str) -> float:
    """The smallest merged exponent, with a NaN refused as malformed.

    ``min()`` over Python floats returns NaN only when NaN comes first,
    so a guard on its result depends on edge order.
    """
    vals = np.fromiter(edge_map.values(), dtype=float)
    if np.isnan(vals).any():
        raise ValueError(f"{where}: nu contains NaN")
    return float(vals.min())


def direct_sum_extrapolated(
    edges,
    nu,
    A,
    L_list: Sequence[int] = (5, 6, 7, 8, 9),
    *,
    root: Optional[int] = None,
    n_correction_terms: int = 3,
    momentum: "np.ndarray | None" = None,
    terminal: Optional[int] = None,
    kernels: "Sequence | None" = None,
) -> float:
    r"""Richardson-extrapolated truncated sum at :math:`\boldsymbol{k} = 0`.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Computes :func:`direct_sum_zero_momentum` at each ``L`` in
    ``L_list`` and fits the analytical Euler-Maclaurin asymptotic

    .. math::

       S(L) \;\approx\; S_\infty \;+\; \sum_{k=0}^{K-1}
       C_k \, L^{d - \nu_{\min} - k}

    via least squares, returning the extrapolated :math:`S_\infty`.

    Why **integer-shifted** powers (not just every other one):
    Euler-Maclaurin expansion of the boundary lattice sum
    :math:`\sum_{|x|>L} \|x\|^{-\nu}` gives

    .. math::

       \frac{2}{\nu-1}\,L^{1-\nu} \;-\; L^{-\nu}
       \;+\; \frac{\nu}{6}\,L^{-\nu-1} \;-\; \dots

    on the chain — successive integer-step corrections.  Skipping
    every other term aliases the missing-power weight into the
    fitted coefficients, biasing :math:`S_\infty`.

    The leading exponent is :math:`d - \min_{v\text{ free}}
    \sum_{e\ni v}\nu_e` — the smallest total incident exponent over
    the free (non-root) vertices, since the dominant box-truncation
    correction comes from the slowest-decaying summed vertex.  This
    accounts both for multi-edge collapse (an :math:`m`-fold parallel
    bond contributes :math:`m\nu`) and, crucially, for vertex degree:
    a densely-connected graph (:math:`K_5`: every summed vertex of
    degree 4) decays as :math:`\lVert x\rVert^{-4\nu}`, far steeper
    than the old :math:`d-\nu_{\min}` basis could represent.

    Parameters
    ----------
    edges, nu, A, root :
        As in :func:`direct_sum_zero_momentum`.
    L_list : sequence of int, optional
        Box half-widths to sample.  Default ``(5, 6, 7, 8, 9)`` is a
        good compromise for chain :math:`d = 1` graphs at
        :math:`\nu \gtrsim 1.5`.
    n_correction_terms : int, optional
        Number ``K`` of correction terms in the fit.  Default ``3``
        is empirically optimal for ``L_list = (5, 6, 7, 8, 9)``;
        higher ``K`` overfits roundoff at this window size.  Must
        satisfy ``K + 1 <= len(L_list)``.
    kernels : sequence of Interaction-likes, optional
        Per-edge kernels, as in :func:`direct_sum_zero_momentum`;
        ``nu`` carries the tails the Richardson basis is derived from.
        With a compact part every rung must hold the reach
        (:func:`_compact_reach`), else
        :class:`gzl.interaction.InteractionSupportError` (a
        ``ValueError``).  A block with no finite tail — every edge
        purely compact, or the legacy ``nu = inf`` indicator — is summed
        exactly at one rung instead of fitted (:func:`_degenerate_rung`).

    Returns
    -------
    float
        Richardson-extrapolated value of
        :math:`\zeta_G(\boldsymbol{0})`.  Real, as
        :func:`direct_sum_zero_momentum` is.
    """
    L_list = sorted(int(L) for L in L_list)
    if not L_list:
        raise ValueError("direct_sum_extrapolated: L_list is empty.")
    K = int(n_correction_terms)
    if K < 1:
        raise ValueError(
            f"direct_sum_extrapolated: n_correction_terms must be >= 1, "
            f"got {K}."
        )
    # The ladder-length check (K + 1 rungs for a K-term fit) runs AFTER
    # the basis exponent is known, below: a block with no finite tail
    # performs no fit, and refusing it on the ladder length (measured:
    # the 3-edge path at nu = inf with L_list = (2, 3, 4) and the default
    # K = 3 raised "need at least 4 L values" for a value that is one
    # exact rung) refused a value the engine has.

    _refuse_interaction_in_nu(nu, "direct_sum_extrapolated")
    edges = np.asarray(edges, dtype=int)
    nu = np.asarray(nu, dtype=float)
    A = np.asarray(_resolve_lattice(A), dtype=float)
    d = int(A.shape[0])

    # The vertex set is the edge support (see gzl/_labels.py).
    # Normalised HERE as well as in the per-L engine, because this
    # function derives its own root and Richardson basis from the edge
    # map before any per-L call; the per-L calls below then hit the
    # engine's identity path.  Self-loops are named before the relabel
    # so the diagnostic carries the caller's label.
    _sl = edges[:, 0] == edges[:, 1] if edges.size else np.zeros(0, bool)
    if _sl.any():
        raise ValueError(
            f"direct_sum: self-loop at vertex "
            f"{int(edges[int(np.argmax(_sl)), 0])}; not allowed."
        )
    edges, _refs = _relabel_to_support(edges, {"root": root,
                                               "terminal": terminal})
    root, terminal = _refs["root"], _refs["terminal"]

    # Same refusal the per-L engine makes: the finite-k phase is
    # cos(2π k · x_terminal) WITH x_source = 0, so which vertex is the
    # source is caller semantics, not the engine's to guess; the root_eff
    # threading below would otherwise auto-pick a phase source silently.
    if momentum is not None and root is None:
        raise ValueError(
            "direct_sum: finite momentum requires an explicit `root` "
            "(= the source vertex pinned at the origin), so the phase "
            "is cos(2π k · x_terminal) with x_source = 0."
        )

    edge_map = _collapse_multi_edges(edges, nu)
    nu_min = _checked_nu_min(edge_map, "direct_sum_extrapolated")
    if not (nu_min > d):
        raise UnsupportedLatticeSumError(
            f"direct_sum_extrapolated: nu_min = {nu_min} <= d = {d} is not "
            "supported: the Richardson basis of the box is not validated "
            "there (see UnsupportedLatticeSumError)."
        )

    # Leading box-truncation exponent.  ``direct_sum_zero_momentum``
    # pins ``root`` (auto = max-degree vertex when not given) and sums
    # the remaining *free* vertices over [-L, L]^d.  The slowest tail
    # comes from the free vertex of minimum total incident exponent
    # (it decays as |x|^-(sum incident nu)), so the correct leading
    # power is d - min_{v free} sum_{e in v} nu_e — NOT d - min edge nu.
    # The two coincide only for a pendant (degree-1) free vertex; for
    # dense graphs (K5: every summed vertex degree 4) the true decay is
    # far steeper and the old shallow basis floored the fit (~5e-4 for
    # K5).  The effective root is resolved ONCE here and threaded into
    # every per-L call below, and the basis exponent is cross-checked
    # against its planner-side twin before anything is fitted.
    V = max(max(u, v) for (u, v) in edge_map) + 1
    root_eff = root if root is not None else _pick_root(edge_map, V)
    nu_eff_lead = _min_free_cut_nu(edge_map, root_eff)
    _check_alpha_basis(edge_map, root_eff, nu_eff_lead)
    alpha = d - nu_eff_lead

    # Compact parts must fit every rung (_compact_reach), and a block
    # with no finite tail is summed EXACTLY once the box holds the
    # reach — there is no tail to fit, and the k = 0 power fit would
    # degenerate to the mean of the rungs (_degenerate_rung has the
    # measured number).  Keyed on the basis exponent, never on the L
    # values.
    kern_map = (None if kernels is None
                else _collapse_multi_kernels(edges, nu, kernels))
    if not np.isfinite(alpha):
        # No rung of L_list is fitted: the one evaluated rung is lifted
        # to the reach, so the smallest-rung check does not apply.
        reach = (0 if kern_map is None
                 else _compact_reach(kern_map, edge_map, root_eff))
        L_one = _degenerate_rung(edge_map, root_eff, L_list, reach,
                                 kern_map, A, d)
        return _as_real(direct_sum_zero_momentum(
            edges, nu, A, L_one, root=root_eff,
            momentum=momentum, terminal=terminal, kernels=kernels,
        ), where="direct_sum_extrapolated")
    if K + 1 > len(L_list):
        raise ValueError(
            f"direct_sum_extrapolated: need at least K+1={K+1} L values "
            f"to fit S_inf + {K} correction terms; got {len(L_list)}."
        )
    if kern_map is not None:
        _check_compact_reach(kern_map, edge_map, root_eff, L_list,
                             "direct_sum_extrapolated")

    # Compute S(L) for each L, threading the ONE effective root
    # explicitly so the per-L pins and the basis above cannot desync
    # even in principle, rather than letting each per-L call re-derive
    # the pin with consistency resting on `_pick_root` determinism alone.
    # The cos phase (finite k) is bounded but it OSCILLATES, so the
    # tail is not a pure power law -- see the basis construction below.
    S = np.array([
        complex(direct_sum_zero_momentum(
            edges, nu, A, L, root=root_eff,
            momentum=momentum, terminal=terminal, kernels=kernels,
        ))
        for L in L_list
    ], dtype=complex)
    # Complex on purpose: the fit runs one lstsq per part, and the
    # imaginary one (an exactly-zero right-hand side at k = 0) is
    # what the frozen ``box|`` bits were taken with.

    # Fit S(L) -> S_inf.  The basis depends on whether a finite-momentum
    # phase is present; see _tail_extrapolate.
    return _as_real(_tail_extrapolate(
        np.array(L_list, dtype=float), S, alpha, K,
        _osc_freqs(momentum, terminal, root_eff),
        unphased=_has_unphased_free(edge_map, root_eff, terminal),
    ), where="direct_sum_extrapolated")


# ---------------------------------------------------------------------------
# The box truncation, for the shared factor-supply layer
# ---------------------------------------------------------------------------

class BoxTruncation(_contract.Truncation):
    r"""Zero-padded linear truncation: true differences on ``[-L, L]^d``,
    Toeplitz kernels, optional Z2 half-box marker.

    Defined *here*, not in :mod:`gzl._contract`, so that every
    method body resolves this module's tunables and helpers through
    this module's namespace at call time — ``direct_sum._USE_CONV``,
    ``_FFT_MARGIN``, ``_FFT_MIN_BAG``, ``_conv_kernel_diff`` and
    ``_fft_conv_step`` are all monkeypatched by the test suite, and a
    class defined elsewhere (or capturing those names at construction)
    would silently detach the patches.  Each body is the engine's
    current behaviour moved verbatim, not a re-derivation.
    """

    def __init__(self, L: int, d: int, A: np.ndarray, *,
                 marker=None, kdiff_cache=None, z2_weights=None):
        self.L = int(L)
        self.d = int(d)
        self.A = np.asarray(A, dtype=float)
        self.marker = marker
        self.n_full = (2 * self.L + 1) ** self.d
        self.n_half = (self.L + 1) * (2 * self.L + 1) ** (self.d - 1)
        # Cost-gate-declined structural peels, one entry per step; see
        # _LAST_GATE_DECLINED for the per-call publication contract.
        self.gate_declined: list = []
        # Z2 weight along the marker axis: 1 at first-coord = 0, 2
        # elsewhere; sums to n_full, which is what makes the empty-
        # bucket scalar on the marker axis the FULL count.  The engine
        # passes its own array through so the loop multiplies the SAME
        # object it always did; the formula below is the fallback for
        # standalone construction and is value-identical.
        if z2_weights is not None:
            self.z2_weights = z2_weights
        else:
            self.z2_weights = np.repeat(
                np.where(np.arange(self.L + 1) == 0, 1.0, 2.0),
                (2 * self.L + 1) ** (self.d - 1),
            )
        # One generator (and one padded transform) per distinct nu,
        # shared with the engine's own cache when supplied.
        self.kdiff_cache = {} if kdiff_cache is None else kdiff_cache

    def _half(self, v) -> bool:
        return v == self.marker and self.marker is not None

    def axis_size(self, v) -> int:
        # NOT gen.size: the difference-range generator has
        # (2(2L+1)-1)^d entries against (2L+1)^d positions — deriving
        # the axis from it is wrong by 1.89x/3.57x/6.74x at d=1/2/3.
        return self.n_half if self._half(v) else self.n_full

    def coords(self, v) -> np.ndarray:
        L, d = self.L, self.d
        first = np.arange(0, L + 1) if self._half(v) else np.arange(-L, L + 1)
        axes = [first] + [np.arange(-L, L + 1)] * (d - 1)
        grids = np.meshgrid(*axes, indexing="ij")
        return np.stack(grids, axis=-1).reshape(-1, d)

    def generator(self, nu) -> np.ndarray:
        if _is_interaction(nu):
            # An Interaction-like keys on ("K",) + key() — the third
            # key space, see _kernel_cache_key; the float branch below
            # is untouched.
            return _kernel_generator(self.kdiff_cache, nu, self.A,
                                     self.L, self.d)[0]
        # float(nu) keying matches the engines' factor build exactly;
        # the cache also carries ('hat', nu, m) transform entries and
        # ("K", ...) kernel entries, so the key spaces must never be
        # normalised into one.
        nu = float(nu)
        g = self.kdiff_cache.get(nu)
        if g is None:
            g = _conv_kernel_diff(nu, self.A, self.L, self.d)
            self.kdiff_cache[nu] = g
        return g

    def table(self, gen, a, b) -> np.ndarray:
        L, d = self.L, self.d
        return _table_from_generator(
            gen,
            _axis_extents(L, d, self._half(a)),
            _axis_extents(L, d, self._half(b)),
            _gen_offsets(L, d, _axis_origins(L, d, self._half(a)),
                         _axis_origins(L, d, self._half(b))),
            d,
        )

    def pin_slot(self, gen, v, pin_row: bool) -> np.ndarray:
        # The root sits at the origin; K is even on every Bravais
        # lattice (||A(-x)|| = ||Ax||), so row/column orientation is
        # immaterial for the kernels this engine builds — but honour it
        # anyway so the gather is correct for any generator.
        L, d = self.L, self.d
        g = _contract.box_reverse_generator(gen) if pin_row else gen
        org = _axis_origins(L, d, self._half(v))
        return _row_from_generator(
            g, _axis_extents(L, d, self._half(v)),
            tuple(int(org[c] + 2 * L) for c in range(d)), d,
        )

    def reverse(self, gen) -> np.ndarray:
        return _contract.box_reverse_generator(gen)

    def compose(self, g1, g2):
        # A SURROGATE, not an identity — see _contract.box_compose.  The
        # exact box composition is not a difference kernel at all, so a
        # box SP collapse must be gated on convergence in L, never on
        # agreement with an uncollapsed box value at fixed L.
        return _contract.box_compose(g1, g2, self.L, self.d)

    def restrict_to(self, gen, coarse):
        if int(coarse.d) != self.d:
            raise ValueError(
                f"restrict_to: dimension mismatch (fine d={self.d}, "
                f"coarse d={int(coarse.d)})"
            )
        if int(coarse.L) > self.L:
            raise ValueError(
                f"restrict_to: cannot restrict upward (fine L={self.L}, "
                f"coarse L={int(coarse.L)})"
            )
        if not np.array_equal(np.asarray(coarse.A, dtype=float), self.A):
            raise ValueError(
                "restrict_to: the two truncations must share the lattice "
                "matrix A; the difference range is indexed by shared "
                "integers and the physical positions are A z, so a "
                "differing A would reinterpret every entry."
            )
        return _contract.box_restrict(gen, self.L, int(coarse.L), self.d)

    def trace(self, gen) -> float:
        # The difference range holds K(delta) at delta + 2L, so zero
        # displacement is the centre, not index 0.
        return float(np.asarray(gen)[(2 * self.L,) * self.d])

    def weight(self, v):
        return self.z2_weights if self._half(v) else None

    def empty_bucket_scalar(self, v) -> float:
        # On the marker axis the half box is a Z2 folding and the
        # weights are exactly what restores the other half (they sum to
        # n_full).  Returning axis_size(v) here instead scales the
        # whole block by (L+1)/(2L+1) with no exception, no NaN and no
        # shape signal.
        if self._half(v):
            return float(np.sum(self.z2_weights))
        return float(self.n_full)

    def peel_constant(self, gen, u) -> np.ndarray:
        # The box's bucket-of-one peel is the TRUE row sums — genuinely
        # position-dependent (measured spread 20-56% across the box),
        # so the torus's full(N, gen.sum()) shortcut is wrong here.
        #
        # DEFINITIONAL, NOT VERBATIM — the one exception to the class
        # docstring's moved-verbatim rule.  The engine reaches these
        # numbers through its generic phi = ones FFT path, which is
        # equal to this direct table sum only to round-off (~1e-15
        # rel), not bitwise.  No engine path calls this method; any
        # future step that wires it into the loop must gate that
        # substitution against the frozen goldens, not cite this
        # docstring.
        t = self.table(gen, u, None)     # (axis_size(u), n_full)
        return t.sum(axis=1)

    # ---- the shared bucket skeleton's callbacks ----------------------
    #
    # Bodies below are the engine's shipped _eliminate_all code, moved
    # verbatim.  They read _USE_CONV, _FFT_MARGIN, _FFT_MIN_BAG,
    # _conv_partner, _conv_kernel_diff and _fft_conv_step through THIS
    # module's namespace at call time, so every monkeypatch on
    # direct_sum.* keeps working.

    def scope_of(self, f):
        return f["scope"]

    def scalar_factor(self, value):
        return {"scope": (), "tensor": np.array(value)}

    def step_dtype(self, bucket):
        # Per-step dtype from the bucket's own factors (lazy factors
        # carry a float64 generator); a non-floating bucket degrades to
        # the dense branch, whose broadcast product promotes naturally.
        return np.result_type(*[
            (f["tensor"] if f.get("tensor") is not None else f["gen"]).dtype
            for f in bucket
        ])

    def peel_gate(self, bucket, w, union, dtype):
        # Call-level half of the gate (_conv_possible) plus the
        # per-step bag test.  _USE_CONV and the two tunables resolve
        # through the module dict on every step, which is the same
        # observable semantics as the old per-call hoist: no test (and
        # no engine caller) flips them mid-call.
        #
        # A COST decline (not the dtype guard, which is correctness)
        # on a step that had a structural partner is recorded in
        # self.gate_declined — the tau*d acceptance's explicit
        # exemption; see _LAST_GATE_DECLINED.  _conv_partner is pure
        # inspection, so probing it on the declined path adds no
        # arithmetic.
        if not (_USE_CONV and _conv_possible(self.n_full,
                                             2 * self.L + 1, self.d)):
            if _USE_CONV and _conv_partner(bucket, w) is not None:
                self.gate_declined.append(
                    {"vertex": w, "axes": len(union), "reason": "gate"})
            return False
        if not np.issubdtype(dtype, np.floating):
            return False
        bag_size = 1
        for v in union:
            bag_size *= self.axis_size(v)
        if bag_size < _FFT_MIN_BAG:
            if _conv_partner(bucket, w) is not None:
                self.gate_declined.append(
                    {"vertex": w, "axes": len(union),
                     "reason": "min_bag"})
            return False
        return True

    def find_partner(self, bucket, w):
        return _conv_partner(bucket, w)

    def _peel_geometry(self, w, u):
        """Per-axis windows and kernel sub-range of the (w, u) peel.

        Derived from the two endpoints' coordinate windows alone: axis
        ``c`` reaches differences ``x_u - x_w`` in
        ``[u_lo - w_hi, u_hi - w_lo]``, whose generator indices start
        at ``(u_org - (w_org + Ws - 1)) + 2L``.  Full/full gives the
        historic whole-range step; a marker endpoint (half axis 0)
        gives the ``3L + 1`` sub-ranges: ``kdiff[L : 4L+1]`` when the
        *partner* is the marker, ``kdiff[0 : 3L+1]`` when the
        *eliminated* vertex is.
        """
        L, d = self.L, self.d
        w_ext = _axis_extents(L, d, self._half(w))
        u_ext = _axis_extents(L, d, self._half(u))
        w_org = _axis_origins(L, d, self._half(w))
        u_org = _axis_origins(L, d, self._half(u))
        gen_start = tuple(
            int(u_org[c] - (w_org[c] + w_ext[c] - 1) + 2 * L)
            for c in range(d)
        )
        return w_ext, u_ext, gen_start

    def peel_step(self, bucket, token, w, out_axes, dtype):
        u, pf = token
        L, d, m = self.L, self.d, 2 * self.L + 1
        others = [f for f in bucket if f is not pf]
        phi_scope = tuple(sorted(
            {s for f in others for s in f["scope"]} | {w}
        ))
        axes = {var: i for i, var in enumerate(phi_scope)}
        # The tag is the kdiff_cache key: the float tail on the legacy
        # path, ("K",) + kernel.key() for an Interaction-like (which the
        # factor also carries, so a miss can rebuild from the object).
        nu_e = pf["conv_nu"]
        kdiff = self.kdiff_cache.get(nu_e)
        if kdiff is None:
            kern = pf.get("conv_kernel")
            if kern is None:
                kdiff = _conv_kernel_diff(nu_e, self.A, L, d)
            else:
                kdiff = _conv_kernel_diff(kern.tail_exponent, self.A, L, d,
                                          interaction=kern)
            self.kdiff_cache[nu_e] = kdiff
        # Offset-aware geometry: a marker endpoint is a window with a
        # different origin and extent, nothing more.
        w_ext, u_ext, gen_start = self._peel_geometry(w, u)
        # The peeled factor's own table is never built: the peel
        # consumes the generator directly, which is the whole point
        # of carrying it instead of a materialised kernel.  The hat is
        # hoisted per STEP, never per chunk: it depends only on the
        # kernel and the step geometry, and a chunked peel would
        # otherwise redo an identical transform for every chunk.  The
        # full/full key keeps its historic shape; marker-touching
        # steps key on which endpoint is the half axis.
        w_half, u_half = self._half(w), self._half(u)
        hat_key = (("hat", nu_e, m) if not (w_half or u_half)
                   else ("hat", nu_e, m, w_half, u_half))
        kdiff_hat = self.kdiff_cache.get(hat_key)
        if kdiff_hat is None:
            K_len = tuple(int(w_ext[c] + u_ext[c] - 1) for c in range(d))
            P = _pad_lengths(w_ext, u_ext, d)
            ks = kdiff[tuple(slice(gen_start[c], gen_start[c] + K_len[c])
                             for c in range(d))]
            kdiff_hat = np.fft.rfftn(ks, s=P, axes=tuple(range(d)))
            self.kdiff_cache[hat_key] = kdiff_hat

        # ---- chunk decision --------------------------------------------
        # The peel's working set (phi, its padded spectrum, the
        # spectral product, the inverse transform) scales with the
        # product of the SPECTATOR axis sizes; the transform itself
        # only ever runs along w's d position axes.  When the modelled
        # working set exceeds the budget and a spectator axis exists,
        # the peel runs chunk by chunk along the last spectator: phi is
        # built per chunk (slice-then-multiply is elementwise-identical
        # to multiply-then-slice), each chunk is transformed by the
        # unchanged _fft_conv_step, and the slices land in a
        # preallocated psi.  Per-fibre FFTs are independent, so the
        # values are bit-identical to the single-shot path — gated as
        # np.array_equal, not argued.
        spectators = [v for v in phi_scope if v != w]
        chunk_axis = spectators[-1] if spectators else None
        chunk_len = None
        if chunk_axis is not None:
            ax_len = self.axis_size(chunk_axis)
            if _FORCE_CHUNK is not None:
                chunk_len = max(1, min(int(_FORCE_CHUNK), ax_len))
            else:
                per_slice = _peel_workset_bytes(
                    [self.axis_size(v) for v in phi_scope],
                    axes[w], axes[chunk_axis], m, d)
                if per_slice * ax_len > _MEM_BUDGET_BYTES:
                    chunk_len = max(1, int(_MEM_BUDGET_BYTES // per_slice))
                    chunk_len = min(chunk_len, ax_len)

        if chunk_len is None or chunk_len >= self.axis_size(chunk_axis):
            # Single-shot path, byte for byte the shipped arithmetic on
            # full/full steps.
            phi = np.ones(tuple(self.axis_size(v) for v in phi_scope),
                          dtype=np.float64)
            for f in others:
                shape = [1] * len(phi_scope)
                for var in f["scope"]:
                    shape[axes[var]] = self.axis_size(var)
                phi = phi * _factor_array(f).reshape(shape)
            if w_half:
                # Eliminating the marker by peel: fold the Z2 weights
                # into phi — exactly what the dense branch applies at
                # this step, moved ahead of the transform.  Omitting
                # them would undercount every x_1 > 0 slice by 2x with
                # no exception and no shape signal.
                wshape = [1] * len(phi_scope)
                wshape[axes[w]] = self.n_half
                phi = phi * self.z2_weights.reshape(wshape)
            psi = _fft_conv_step(np.moveaxis(phi, axes[w], 0), kdiff, m, d,
                                 kdiff_hat, w_ext=w_ext, u_ext=u_ext,
                                 gen_start=gen_start)
        else:
            psi = self._chunked_peel(others, phi_scope, axes, w,
                                     chunk_axis, chunk_len,
                                     kdiff, kdiff_hat, m, d,
                                     w_ext, u_ext, gen_start)

        new_scope = tuple(sorted(
            [u] + [v for v in phi_scope if v != w]
        ))
        # Materialise the axis move: a strided view would slow the
        # next step's broadcast product by more than this copy costs.
        new_tensor = np.ascontiguousarray(
            np.moveaxis(psi, 0, new_scope.index(u))
        )
        return {"scope": new_scope, "tensor": new_tensor}

    def _chunked_peel(self, others, phi_scope, axes, w, chunk_axis,
                      chunk_len, kdiff, kdiff_hat, m, d,
                      w_ext, u_ext, gen_start):
        """The peel, chunk by chunk along one spectator axis.

        Each factor's dense form is materialised ONCE (not per chunk)
        and sliced along the chunk axis where it carries it; the
        broadcast product, the padded transform and the inverse run per
        chunk.  Peak memory holds one chunk's working set plus the
        full psi accumulator, instead of the full-bag working set.

        The chunk axis is a spectator, never ``w``, so the Z2 fold on
        a marker elimination multiplies the same weight vector into
        every chunk's ``w`` axis — slice-then-weight is elementwise
        identical to weight-then-slice.
        """
        ax_len = self.axis_size(chunk_axis)
        n_out = 1
        for c in range(d):
            n_out *= int(u_ext[c])
        # psi in the same axis order the single-shot path produces:
        # w's axis (relabelled to u's positions, u's axis length)
        # first, then the spectators in phi_scope order.
        spect_shape = tuple(self.axis_size(v) for v in phi_scope
                            if v != w)
        psi = np.empty((n_out,) + spect_shape, dtype=np.float64)
        # Position of the chunk axis inside psi: 1 + its rank among
        # the spectators (axis 0 is the convolved one).
        spectators = [v for v in phi_scope if v != w]
        psi_c_pos = 1 + spectators.index(chunk_axis)
        w_half = self._half(w)

        arrays = [(_factor_array(f), f["scope"]) for f in others]
        for start in range(0, ax_len, chunk_len):
            sel = slice(start, min(start + chunk_len, ax_len))
            n_sel = sel.stop - sel.start
            chunk_shape = tuple(
                n_sel if v == chunk_axis else self.axis_size(v)
                for v in phi_scope
            )
            phi_c = np.ones(chunk_shape, dtype=np.float64)
            for arr, scope in arrays:
                a = arr
                if chunk_axis in scope:
                    # A basic slice is a VIEW; np.take with an index
                    # array would copy the chunk of every factor per
                    # chunk, a cost the memory model does not count.
                    idx = [slice(None)] * a.ndim
                    idx[scope.index(chunk_axis)] = sel
                    a = a[tuple(idx)]
                shape = [1] * len(phi_scope)
                for var in scope:
                    shape[axes[var]] = (n_sel if var == chunk_axis
                                        else self.axis_size(var))
                phi_c = phi_c * a.reshape(shape)
            if w_half:
                wshape = [1] * len(phi_scope)
                wshape[axes[w]] = self.n_half
                phi_c = phi_c * self.z2_weights.reshape(wshape)
            psi_c = _fft_conv_step(np.moveaxis(phi_c, axes[w], 0),
                                   kdiff, m, d, kdiff_hat,
                                   w_ext=w_ext, u_ext=u_ext,
                                   gen_start=gen_start)
            dest = [slice(None)] * psi.ndim
            dest[psi_c_pos] = sel
            psi[tuple(dest)] = psi_c
        return psi

    def dense_strategy(self, bag_volume: int) -> str:
        # "auto": sliced at or above _DENSE_SLICE_MIN_VOLUME bag
        # elements (measured threshold — see the tunable's comment).
        # In practice: every shipped d = 1 rung stays on the fused
        # broadcast product; at d >= 2 the big unpeelable bags slice
        # while peel-covered or marker-halved bags below the threshold
        # stay broadcast.  The torus is not flipped.
        return "auto"

    def dense_step(self, bucket, w, out_axes, dtype):
        union_scope = tuple(sorted(set(out_axes) | {w}))
        axes = {var: i for i, var in enumerate(union_scope)}

        bag_volume = 1
        for v in union_scope:
            bag_volume *= self.axis_size(v)
        strategy = _contract.resolve_dense_strategy(
            self.dense_strategy(bag_volume), bag_volume,
            _DENSE_SLICE_MIN_VOLUME)
        if strategy == "sliced":
            return self._dense_step_sliced(bucket, w, union_scope)

        prod_shape = tuple(self.axis_size(v) for v in union_scope)
        prod = np.ones(prod_shape, dtype=np.float64)
        for f in bucket:
            shape = [1] * len(union_scope)
            for var in f["scope"]:
                shape[axes[var]] = self.axis_size(var)
            prod = prod * _factor_array(f).reshape(shape)

        if w == self.marker and self.marker is not None:
            # Apply Z_2 weights along the marker axis: first-coord = 0
            # -> 1, first-coord > 0 -> 2.  In d = 1 this reduces to the
            # original [1, 2, 2, ...] weighting; in d >= 2 the half-box
            # pattern is encoded into z2_weights at construction time.
            wshape = [1] * len(union_scope)
            wshape[axes[w]] = self.n_half
            prod = prod * self.z2_weights.reshape(wshape)

        new_tensor = prod.sum(axis=axes[w])
        new_scope = tuple(v for v in union_scope if v != w)
        return {"scope": new_scope, "tensor": new_tensor}

    def _dense_step_sliced(self, bucket, w, union_scope):
        r"""Axis-sliced dense contraction: accumulate over ``w``'s
        positions instead of materialising the full-bag product.

        Identical multiply-adds to the broadcast branch, re-associated:
        the broadcast product holds *two* full-bag arrays at the peak
        (each ``prod * factor`` allocates the next full-bag array while
        the previous is still live), while this branch's peak is the
        out-scope volume plus one slice product — a factor
        ``2 * axis_size(w)`` of peak memory.  With the peel chunked,
        the dense full-bag product is the box's memory ceiling (K5's
        d = 3 L = 4 dense fallback: full-bag product tens of GB
        against an out-scope in the tens of MB), which is what this
        branch removes at d >= 2.

        Values move only by summation re-association (a loop of
        in-place adds versus ``np.sum``'s pairwise reduction over one
        axis), gated at 1e-12 relative against the frozen goldens plus
        the Richardson fit residual.  At golden sizes the flip
        reproduced every frozen record bit-identically: the flipped
        steps eliminate the bag product's LEADING axis, where numpy's
        reduction accumulates sequentially — the same association as
        this loop (a trailing-axis reduction goes pairwise and would
        differ in the last bits; verified both ways).
        ``box_raw_k5_d2_L3`` pins the class.

        The Z2 marker weight becomes a per-slice scalar; dtype
        promotion matches the broadcast branch exactly (float64 ones
        times the factor slices, accumulated in the promoted dtype).
        """
        out_scope = tuple(v for v in union_scope if v != w)
        out_shape = tuple(self.axis_size(v) for v in out_scope)
        out_pos = {var: i for i, var in enumerate(out_scope)}
        weights = (self.z2_weights
                   if (w == self.marker and self.marker is not None)
                   else None)

        arrays = [(_factor_array(f), f["scope"]) for f in bucket]
        res_dtype = np.result_type(np.float64,
                                   *[a.dtype for a, _ in arrays])
        res = np.zeros(out_shape, dtype=res_dtype)
        for i in range(self.axis_size(w)):
            sprod = np.ones(out_shape, dtype=np.float64)
            for a, scope in arrays:
                if w in scope:
                    # Basic integer index: a view, not a copy.
                    idx = [slice(None)] * a.ndim
                    idx[scope.index(w)] = i
                    a = a[tuple(idx)]
                shape = [1] * len(out_shape)
                for var in scope:
                    if var != w:
                        shape[out_pos[var]] = self.axis_size(var)
                sprod = sprod * a.reshape(shape)
            if weights is not None:
                sprod = sprod * weights[i]
            res += sprod
        return {"scope": out_scope, "tensor": res}
