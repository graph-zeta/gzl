# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Truncation-independent planning for bucket elimination.

Both treewidth-agnostic engines — the torus/cyclic one in
:mod:`gzl.tensor_network` and the box/zero-padded one in
:mod:`gzl.direct_sum` — contract the same object: a graph lattice
sum, eliminated vertex by vertex, with an FFT fast path on any step that
isolates one original edge kernel.  Only the *truncation* differs
(differences mod ``n`` against true differences on ``[-L, L]^d``), and
the truncation does not enter the schedule at all.

This module owns the schedule.  It answers, without evaluating anything:

* which vertex to pin,
* in which order to eliminate,
* how large each bag is,
* which steps the FFT peel can fire on,
* and hence the cost exponent the contraction will actually achieve.

The cost model is the one both engines realise::

    dense step   ~  n^(|bag| * d)
    peeled step  ~  n^((|bag| - 1) * d) * log n

so a treewidth-:math:`\tau` block contracts at :math:`n^{\tau d}` — *not*
at the :math:`n^{(\tau + 1 + N_t) d}` dense bound, which is an upper bound
on an unpeeled contraction and badly understates both engines.

The pinned vertex is a size-1 axis: it is taken at index 0 and drops out
of every scope, so it never contributes to a bag.  Which vertex carries
that privilege is therefore a real choice, and
:func:`plan` makes it by trying all of them.

Scope evolution is simulated symbolically: the factor scopes after
eliminating a set ``S`` depend on ``S`` alone and not on the order
within it, which is what makes the subset DP below valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

__all__ = [
    "sp_reduced_cut_nu",
    "Step",
    "Plan",
    "PlanBudgetExceededError",
    "scopes_from_edges",
    "simulate",
    "reachable_through",
    "bag_axes",
    "peel_partners",
    "plan",
    "plan_for_order",
    "best_order",
    "min_degree_order",
    "min_free_cut_nu",
    "MAX_EXACT_FREE",
    "MAX_PLAN_FREE",
]


class PlanBudgetExceededError(RuntimeError):
    """The pin-searching exact DP was refused on size.

    :func:`plan` multiplies the ``2^|free|`` subset DP by up to ``|V|``
    pin candidates, which is 8.97 s of pure Python on a 16-cycle against
    a sub-second corpus pass.  Above :data:`MAX_PLAN_FREE` eliminable
    vertices it refuses outright rather than degrading silently; an
    engine that adopts :func:`plan` must catch this and fall back to its
    own heuristic schedule.  Contrast :func:`best_order`, which searches
    no pin and therefore *falls back* above :data:`MAX_EXACT_FREE`
    instead of refusing.
    """


# --------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    """One elimination step, as planned.

    Attributes
    ----------
    vertex
        The vertex summed out.
    bag
        Sorted axes touched by the step — the union of the scopes of
        every factor containing ``vertex``, with the pin already
        removed.  Includes ``vertex`` itself.
    partners
        Vertices ``u`` for which the step is FFT-peelable: the bucket
        holds an original two-vertex kernel ``{vertex, u}`` and ``u``
        occurs in no other factor of the bucket.  Empty when the step
        must run dense.
    """

    vertex: int
    bag: tuple[int, ...]
    partners: tuple[int, ...]

    @property
    def peelable(self) -> bool:
        """Whether the FFT peel is structurally available here."""
        return bool(self.partners)

    @property
    def exponent(self) -> int:
        """Cost exponent in units of ``n^d``, peel included."""
        return len(self.bag) - (1 if self.partners else 0)


@dataclass(frozen=True)
class Plan:
    """A complete elimination schedule and the exponent it achieves."""

    pin: int
    order: tuple[int, ...]
    keep: tuple[int, ...]
    steps: tuple[Step, ...]

    @property
    def max_bag(self) -> int:
        """Largest bag over all steps, in axes (pin excluded)."""
        return max((len(s.bag) for s in self.steps), default=0)

    @property
    def exponent(self) -> int:
        """Achieved cost exponent in units of ``n^d``.

        This is the number to compare against the block's treewidth:
        the invariant the engines must hold is ``exponent == tw``.
        """
        return max((s.exponent for s in self.steps), default=0)

    @property
    def peak_steps(self) -> tuple[Step, ...]:
        """Steps sitting at the maximum bag — the ones that dominate."""
        top = self.max_bag
        return tuple(s for s in self.steps if len(s.bag) == top)

    @property
    def peak_peel_coverage(self) -> int:
        """How many peak steps peel.  The DP's tie-break objective."""
        return sum(1 for s in self.peak_steps if s.partners)

    def work(self, n_points: int, d: int) -> float:
        """Modelled contraction work, summed over steps.

        Deterministic, so it is a usable cost instrument on a machine
        that is not quiet.  It ignores Python overhead, FFT constants
        and memory traffic, and therefore always reads optimistic —
        confirm direction with wall-clock, but do not rank on it.
        """
        return sum(float(n_points) ** (s.exponent * d) for s in self.steps)


# --------------------------------------------------------------------
# Symbolic scope machinery
# --------------------------------------------------------------------

def scopes_from_edges(
    edge_map: Iterable,
    pin: int,
    *,
    extra_scopes: Iterable[Iterable[int]] = (),
    non_kernel_edges: Iterable = (),
) -> list[tuple[frozenset, bool]]:
    """Factor scopes the engines will build, as ``(scope, is_kernel)``.

    ``edge_map`` is any iterable of ``(u, v)`` vertex pairs — the
    Hadamard-merged simple graph, one entry per bundle.  Pin-incident
    kernels collapse to one-axis factors (the pin is taken at index 0)
    and are not peelable; free-free kernels are the two-axis originals
    that carry a generator and hence are.  ``extra_scopes`` covers
    non-kernel factors such as the momentum cosine weight.

    ``non_kernel_edges`` names bundles that must NOT be modelled as
    peelable kernels even though they are free-free — the ν = ∞
    nearest-neighbour indicators, which both engines deliberately
    contract dense (integer-exact; an FFT round trip returns 2.0 as
    1.9999999999999998).  Without this the plan claims peels the
    engine will never fire, its exponent reads optimistic, and the
    τ·d acceptance would see a silent degradation with no gate
    diagnostic to exempt it.

    Works for any pin; the box and the torus core both build their
    scopes here.
    """
    nk = _normalise_pairs(non_kernel_edges)
    scopes: list[tuple[frozenset, bool]] = []
    for u, v in edge_map:
        u, v = int(u), int(v)
        if u == pin or v == pin:
            other = v if u == pin else u
            if other != pin:                     # self-loops are rejected
                scopes.append((frozenset({other}), False))
        else:
            key = (u, v) if u <= v else (v, u)
            scopes.append((frozenset({u, v}), key not in nk))
    for s in extra_scopes:
        scopes.append((frozenset(int(x) for x in s) - {pin}, False))
    return scopes


def _normalise_pairs(pairs: Iterable) -> frozenset:
    """Orientation-normalised ``(min, max)`` pair set."""
    return frozenset(
        (u, v) if u <= v else (v, u)
        for u, v in ((int(a), int(b)) for a, b in pairs)
    )


def simulate(
    scopes: Sequence[tuple[frozenset, bool]],
    order: Sequence[int],
) -> tuple[Step, ...]:
    """Replay an elimination symbolically and record every step.

    Returns one :class:`Step` per vertex of ``order`` that has a
    non-empty bucket.  A vertex with no incident factor contributes
    only a scalar axis count and no contraction, exactly as both
    engines handle it, so it produces no step.

    The replay does not depend on the truncation (torus or box).
    """
    facs = [(frozenset(s), bool(k)) for s, k in scopes]
    steps: list[Step] = []
    for w in order:
        w = int(w)
        bucket = [(s, k) for s, k in facs if w in s]
        rest = [(s, k) for s, k in facs if w not in s]
        if not bucket:
            facs = rest
            continue
        union = frozenset().union(*[s for s, _ in bucket])
        partners = set()
        for s, is_kernel in bucket:
            if not is_kernel or len(s) != 2:
                continue
            (u,) = s - {w}
            # The partner must not survive in the rest of the bucket,
            # or the remaining product still depends on its position
            # and the w-sum is not a convolution in it.
            if sum(1 for t, _ in bucket if u in t) == 1:
                partners.add(u)
        steps.append(
            Step(w, tuple(sorted(union)), tuple(sorted(partners)))
        )
        facs = rest + [(union - {w}, False)]
    return tuple(steps)


def reachable_through(
    adj: Mapping[int, set], eliminated: frozenset, v: int
) -> set:
    """``Q(S, v)``: vertices outside ``S ∪ {v}`` reachable from ``v``
    through vertices of ``S``.

    Together with ``v`` this is the bag of the step, because the
    factors merged by eliminating ``S`` have exactly the boundaries of
    ``S``'s connected components as their scopes.
    """
    out: set = set()
    seen = {v}
    stack = [v]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y == v or y in seen:
                continue
            if y in eliminated:
                seen.add(y)
                stack.append(y)
            else:
                out.add(y)
    return out


def merged_boundary(
    adj: Mapping[int, set], eliminated: frozenset, v: int
) -> set:
    """Vertices reaching ``v`` **through** at least one eliminated vertex.

    These are the axes contributed by the *merged* factors in ``v``'s
    bucket, as opposed to the direct neighbours, which still sit on
    their own original two-vertex kernels.  The distinction is what
    decides peelability, so it cannot be folded into
    :func:`reachable_through`.
    """
    out: set = set()
    seen = {v}
    stack: list = []
    for y in adj[v]:
        if y in eliminated and y not in seen:
            seen.add(y)
            stack.append(y)
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y == v or y in seen:
                continue
            if y in eliminated:
                seen.add(y)
                stack.append(y)
            else:
                out.add(y)
    return out


def bag_axes(
    adj: Mapping[int, set], eliminated: frozenset, v: int, pin: int
) -> frozenset:
    """Axes of the step that eliminates ``v``, pin excluded."""
    return frozenset({v} | reachable_through(adj, eliminated, v)) - {pin}


def peel_partners(
    adj: Mapping[int, set],
    eliminated: frozenset,
    v: int,
    pin: int,
    non_kernel: frozenset = frozenset(),
) -> frozenset:
    """Partners ``u`` making the step that eliminates ``v`` peelable.

    ``u`` must sit on an original, still-unconsumed two-vertex kernel
    ``{v, u}`` — so ``u`` is a neighbour of ``v``, neither endpoint is
    the pin, and ``u`` has not been eliminated — and must not occur in
    any *other* factor of the bucket, which for a neighbour means it
    must not also arrive through a merged factor.  A bundle listed in
    ``non_kernel`` (normalised ``(min, max)`` pairs; the ν = ∞
    indicators) is no kernel and never a partner.

    Note the qualifier: a plain direct neighbour is precisely the
    peelable case, so testing against :func:`reachable_through` (which
    includes direct neighbours, correctly, because they are bag axes)
    would reject every partner there is.
    """
    if v == pin:
        return frozenset()
    through = merged_boundary(adj, eliminated, v)
    return frozenset(
        u for u in adj[v]
        if u != pin and u not in eliminated and u not in through
        and ((v, u) if v <= u else (u, v)) not in non_kernel
    )


# --------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------

def plan_for_order(
    edge_map: Iterable,
    order: Sequence[int],
    pin: int,
    *,
    keep: Iterable[int] = (),
    extra_scopes: Iterable[Iterable[int]] = (),
    non_kernel_edges: Iterable = (),
) -> Plan:
    """Build the :class:`Plan` a given pin and order would realise.

    Use this to describe what an engine does today; use :func:`plan` to
    choose what it should do.
    """
    scopes = scopes_from_edges(edge_map, pin, extra_scopes=extra_scopes,
                               non_kernel_edges=non_kernel_edges)
    steps = simulate(scopes, order)
    return Plan(int(pin), tuple(int(v) for v in order),
                tuple(sorted(int(k) for k in keep)), steps)


def _adjacency(edge_map: Iterable, vertices: Iterable[int]) -> dict:
    adj: dict = {int(v): set() for v in vertices}
    for u, v in edge_map:
        u, v = int(u), int(v)
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def _optimal_order(
    adj: Mapping[int, set],
    elim: Sequence[int],
    pin: int,
    non_kernel: frozenset = frozenset(),
) -> tuple[list[int], int]:
    """Exact min-max-*exponent* order by subset DP.

    Blocks in the corpus have ``|V| <= 10``, so ``2^|elim| <= 512``
    states is microseconds and exact — a min-degree heuristic is not
    needed and is not always optimal.

    The quantity minimised is the **post-peel** step exponent,
    ``|bag| - 1`` on a peelable step and ``|bag|`` otherwise, because
    that is what the contraction actually pays.  Minimising the max
    *bag* instead is subtly wrong and measurably so: on a real 9-vertex
    corpus block both the min-degree order and the bag-optimal order
    reach ``max_bag = 3``, but every bag-3 step of the former peels
    (exponent 2) while the latter strands one that cannot (exponent 3).
    Treating peel coverage as a tie-break after the bag cannot see
    that, since the two orders are tied on the primary key and the
    coverage of a *future* step is not yet known.

    The min-max recursion is exact because the step cost depends only
    on the eliminated set, not on the order within it.  ``max_bag`` is
    carried as a secondary preference — it drives peak memory even when
    a step peels — but only the exponent is guaranteed optimal.
    """
    idx = {v: i for i, v in enumerate(elim)}
    full = (1 << len(elim)) - 1
    if full < 0:                                   # nothing to eliminate
        return [], 0

    INF = 10 ** 9
    # dp[mask] = (max exponent, max bag) — both minimised, in order.
    dp: list[tuple[int, int]] = [(INF, INF)] * (full + 1)
    par: dict = {0: None}
    dp[0] = (0, 0)

    for mask in range(full + 1):
        cur = dp[mask]
        if cur[0] >= INF:
            continue
        S = frozenset(elim[i] for i in range(len(elim)) if mask >> i & 1)
        for v in elim:
            if mask >> idx[v] & 1:
                continue
            size = len(bag_axes(adj, S, v, pin))
            expo = size - (1 if peel_partners(adj, S, v, pin,
                                              non_kernel) else 0)
            cand = (max(cur[0], expo), max(cur[1], size))
            nxt = mask | (1 << idx[v])
            if cand < dp[nxt]:
                dp[nxt] = cand
                par[nxt] = (mask, v)

    order: list[int] = []
    m = full
    while par.get(m):
        m, v = par[m]
        order.append(v)
    order.reverse()
    return order, dp[full][0]


#: Hard ceiling on the number of eliminable vertices :func:`plan` will
#: accept.  Above it the pin search refuses with
#: :class:`PlanBudgetExceededError` — it does not degrade to a
#: heuristic, because a silent quality cliff inside an adopted planner
#: is exactly the failure mode the planner-adoption guards forbid.
#: Real corpus blocks have ``|free| <= 9``.
MAX_PLAN_FREE = 12


def plan(
    edge_map: Iterable,
    vertices: Iterable[int],
    *,
    pin: int | None = None,
    keep: Iterable[int] = (),
    extra_scopes: Iterable[Iterable[int]] = (),
    non_kernel_edges: Iterable = (),
) -> Plan:
    """Choose pin and elimination order, and return the resulting plan.

    ``edge_map`` is the Hadamard-merged simple graph as ``(u, v)``
    pairs, ``vertices`` its vertex set, ``keep`` the vertices that stay
    open (free terminals) and are never eliminated.  Passing ``pin``
    fixes the pin; leaving it ``None`` searches every candidate.

    The pin is free by translation invariance — the value is identical
    whichever vertex is held at the origin — but it is a size-1 axis,
    so it decides which end of ``tw(G) - 1 <= tw(G - p) <= tw(G)`` the
    contraction pays.  A maximum-degree vertex does not reliably attain
    the lower end, so every candidate is tried and the best kept.

    Refuses with :class:`PlanBudgetExceededError` above
    :data:`MAX_PLAN_FREE` eliminable vertices — adopting engines catch
    it and keep their heuristic schedule.

    Results are memoised on the exact call (edge multiset with
    orientation normalised, vertex set, pin, keep, extra scopes).  The
    cache is deliberately **not** keyed on the isomorphism class: the
    DP's tie-breaks are label-dependent, so transferring a plan across
    an isomorphic relabelling can return a schedule that is equally
    optimal but different from what an uncached call would produce,
    which moves engine round-off and breaks the frozen-golden
    bit-identity the suite gates on.  The exact key makes the cache
    value-transparent: hit or miss, the returned ``Plan`` is
    byte-for-byte what the uncached search yields.  Tests may reset it
    via ``_plan_cached.cache_clear()``.
    """
    verts_key = tuple(sorted(int(v) for v in vertices))
    keep_key = tuple(sorted(int(k) for k in keep))
    keep_set = frozenset(keep_key)
    pin_key = None if pin is None else int(pin)

    if pin_key is not None:
        n_elim = sum(1 for v in verts_key
                     if v != pin_key and v not in keep_set)
    else:
        n_free = sum(1 for v in verts_key if v not in keep_set)
        n_elim = max(n_free - 1, 0)
    if n_elim > MAX_PLAN_FREE:
        raise PlanBudgetExceededError(
            f"plan: {n_elim} eliminable vertices exceed the "
            f"MAX_PLAN_FREE = {MAX_PLAN_FREE} budget for the "
            f"pin-searching exact DP; use best_order (fixed pin, "
            f"heuristic fallback) or the engine's own schedule."
        )

    edges_key = tuple(sorted(
        (u, v) if u <= v else (v, u)
        for u, v in ((int(a), int(b)) for a, b in edge_map)
    ))
    extra_key = tuple(sorted(
        tuple(sorted({int(x) for x in s})) for s in extra_scopes
    ))
    nk_key = tuple(sorted(_normalise_pairs(non_kernel_edges)))
    return _plan_cached(edges_key, verts_key, pin_key, keep_key,
                        extra_key, nk_key)


@lru_cache(maxsize=4096)
def _plan_cached(
    edges_key: tuple,
    verts_key: tuple,
    pin_key: int | None,
    keep_key: tuple,
    extra_key: tuple,
    nk_key: tuple = (),
) -> Plan:
    """The exact search behind :func:`plan`, on normalised keys.

    Every :class:`Step` is content-invariant to factor-list order (bags,
    partners and unions are set-built), so planning on the normalised
    edge multiset returns the identical ``Plan`` the raw argument order
    would — that is what makes the memoisation value-transparent.
    """
    verts = list(verts_key)
    keep_set = frozenset(keep_key)
    nk = frozenset(nk_key)
    adj = _adjacency(edges_key, verts)

    candidates = [pin_key] if pin_key is not None else [
        v for v in verts if v not in keep_set
    ]
    if not candidates:                            # every vertex kept open
        candidates = verts[:1]

    best: Plan | None = None
    for p in candidates:
        elim = [v for v in verts if v != p and v not in keep_set]
        order, _ = _optimal_order(adj, elim, p, nk)
        cand = plan_for_order(edges_key, order, p, keep=keep_set,
                              extra_scopes=extra_key,
                              non_kernel_edges=nk_key)
        if best is None or _rank(cand) < _rank(best):
            best = cand
    assert best is not None
    return best


def _rank(p: Plan) -> tuple:
    """Order plans: cheapest exponent, then fewest dense peak steps."""
    return (p.exponent, p.max_bag,
            len(p.peak_steps) - p.peak_peel_coverage)


# --------------------------------------------------------------------
# Order selection for the engines
# --------------------------------------------------------------------

#: Above this many eliminable vertices the exact subset DP (``2^|free|``
#: states) stops being free and the engines fall back to their
#: heuristic.  Real corpus blocks have ``|free| <= 9``; the ceiling is
#: set well above that so the DP always runs where it matters, and the
#: fallback exists only so a hand-built graph cannot hang.
MAX_EXACT_FREE = 14


def best_order(
    edge_map: Iterable,
    vertices: Iterable[int],
    pin: int,
    *,
    keep: Iterable[int] = (),
    fallback=None,
    non_kernel_edges: Iterable = (),
) -> list:
    """Elimination order minimising the achieved cost exponent.

    **The pin is not changed.**  Re-ordering is value-exact on either
    truncation — it is the same finite sum contracted in a different
    order — whereas re-*pinning* is value-exact only on the periodic
    scheme: a box ``[-L, L]^d`` is not closed under translation, so
    which configurations fall inside depends on which vertex sits at the
    origin.  Keeping the pin fixed is what makes this safe to apply to
    both engines.

    ``fallback`` is called as ``fallback()`` when the graph is too large
    for the exact DP; it should return the caller's heuristic order.
    """
    verts = sorted(int(v) for v in vertices)
    keep_set = frozenset(int(k) for k in keep) | {int(pin)}
    n_free = sum(1 for v in verts if v not in keep_set)
    if n_free > MAX_EXACT_FREE:
        return list(fallback()) if fallback is not None else verts
    adj = _adjacency(edge_map, verts)
    elim = [v for v in verts if v not in keep_set]
    order, _ = _optimal_order(adj, elim, int(pin),
                              _normalise_pairs(non_kernel_edges))
    return order


# --------------------------------------------------------------------
# The two shipped min-degree heuristics, one home
# --------------------------------------------------------------------

def min_degree_order(adj, vertices, *, terminals=None, root=None,
                     fill_in: bool) -> list:
    """Greedy min-degree elimination order — both shipped variants.

    The engines historically carried two same-named heuristics with
    different signatures and genuinely different behaviour: the torus
    one (``fill_in=True``, ``terminals`` kept) adds fill-in edges after
    each elimination, the box one (``fill_in=False``, degree counted
    toward remaining-free neighbours and the pinned ``root``) walks the
    static adjacency.  They are NOT interchangeable, and each sits at
    its own distance from the exact planner, so the merge keeps both
    bodies verbatim behind the flag rather than unifying the semantics.
    The engine facades preserve their historic names and signatures and
    delegate here.
    """
    if fill_in:
        if terminals is None:
            raise TypeError("fill_in=True requires `terminals`")
        nb = {v: set(adj[v]) for v in vertices}
        remaining = set(vertices) - set(terminals)
        order: list = []
        active = remaining | set(terminals)
        while remaining:
            v = min(
                remaining,
                key=lambda u: len(nb[u] & active),
            )
            order.append(v)
            # fill-in: connect all active neighbours pairwise
            active_nb = list(nb[v] & active)
            for a in active_nb:
                for b in active_nb:
                    if a != b:
                        nb[a].add(b)
            active.discard(v)
            remaining.discard(v)
        return order

    if root is None:
        raise TypeError("fill_in=False requires `root`")
    elimination: list = []
    remaining = set(vertices)
    while remaining:
        next_v = min(
            remaining,
            key=lambda v: sum(
                1 for u in adj[v] if u in remaining or u == root
            ),
        )
        elimination.append(next_v)
        remaining.remove(next_v)
    return elimination


# --------------------------------------------------------------------
# Pin/alpha cross-check
# --------------------------------------------------------------------

def sp_reduced_cut_nu(edge_map: Mapping, root: int, externals=(),
                      max_free_exact: int = 16) -> float:
    r"""``min_free_cut_nu`` of the SP-REDUCED core, not of the block.

    This is the truncation rate a SPLIT evaluation converges at, and it
    is not the block's.  The block rate is set by its cheapest escape,
    which is a degree-2 free vertex at cut :math:`2\nu` --- and a
    degree-2 free vertex is precisely what SP reduction suppresses.  So
    collapsing the series/parallel part on a fine grid does not merely
    reduce that mode, it relocates it off the coarse grid entirely, and
    the residual decays at the core's own cut.

    A 3-connected core has edge connectivity :math:`\ge 3`, so this is
    :math:`\ge 3\nu` against the block's :math:`2\nu` --- the split buys
    a full :math:`\nu` in the exponent, which is the whole mechanism.

    Reduction rules, and getting the second one backwards inverts every
    prediction: parallel edges **SUM** their :math:`\nu`; suppressing a
    degree-2 vertex leaves the **MIN** of its two.  A cut through a
    2-path is limited by its weaker edge, not helped by its stronger.

    ``root`` and ``externals`` are never suppressed --- they are pinned,
    so they are not free to escape and the reduction must not remove
    them.  Self-loops that would be created by suppressing a vertex
    whose two neighbours coincide are left alone for the same reason
    ``_sp`` does: the loop is a different object, not a series edge.

    Returns the block's own cut when the core is empty or nothing
    reduces, so it is always safe to call --- on a 3-connected block it
    IS ``min_free_cut_nu``, and on a fully SP-reducible one there is no
    core and the caller should not have been splitting anyway.
    """
    pinned = {int(root)} | {int(x) for x in externals}
    adj: dict = {}
    for (u, v), w in dict(edge_map).items():
        u, v = int(u), int(v)
        a, b = (u, v) if u < v else (v, u)
        adj.setdefault(a, {})
        adj.setdefault(b, {})
        adj[a][b] = adj[a].get(b, 0.0) + float(w)   # parallel: SUM
        adj[b][a] = adj[b].get(a, 0.0) + float(w)

    changed = True
    while changed:
        changed = False
        # Re-check the degree INSIDE the loop, not just when building the
        # candidate list: suppressing one vertex changes its neighbours'
        # degrees, so a vertex that was degree-2 when the list was built
        # can be degree-1 by the time it is reached.  Trusting the
        # snapshot raises "not enough values to unpack" on any block with
        # two adjacent degree-2 vertices -- i.e. on a plain cycle.
        for x in [v for v in list(adj) if v not in pinned]:
            if x not in adj or len(adj[x]) != 2:
                continue
            (a, wa), (b, wb) = list(adj[x].items())
            if a == b:
                continue                             # would make a loop
            del adj[x]
            adj[a].pop(x, None)
            adj[b].pop(x, None)
            m = min(wa, wb)                          # series: MIN
            adj[a][b] = adj[a].get(b, 0.0) + m       # ...then parallel: SUM
            adj[b][a] = adj[b].get(a, 0.0) + m
            changed = True

    core = {(min(u, v), max(u, v)): w
            for u, nb in adj.items() for v, w in nb.items()}
    if not core:
        return min_free_cut_nu(edge_map, root, externals, max_free_exact)
    return min_free_cut_nu(core, root, externals, max_free_exact)


def min_free_cut_nu(edge_map: Mapping, root: int, externals=(),
                    max_free_exact: int = 16) -> float:
    r"""Smallest cluster cut weight over escaping free-vertex sets.

    Planner-side twin of ``direct_sum._min_free_cut_nu``, kept
    **deliberately independent**: the box's Richardson basis exponent is
    ``alpha = d - _min_free_cut_nu(edge_map, root_eff)``, and a
    pin/alpha desync — the per-L sums pinned at one root while the
    basis is derived from another — fits a wrong-power basis with the
    least-squares solve still succeeding and every raw per-L value
    individually correct.  Nothing numerical can catch that, so the
    extrapolated call sites cross-check the engine's cut against this
    implementation at the same root and refuse on any mismatch.  Both
    functions iterate the same ``edge_map`` mapping in the same order,
    so agreement is exact (``==``), not approximate — asserted by the
    equality battery in ``tests/test_elimination.py``.

    Semantics (mirroring the engine): minimise
    :math:`I_S = \sum_{e \in \mathrm{cut}(S)} \nu_e` over connected
    free-vertex clusters ``S`` with connected complement; with no free
    vertex, the minimum edge weight; with no admissible cluster, the
    minimum total incident weight over free vertices.  ``externals``
    names further vertices that cannot escape.  Above
    ``max_free_exact`` free vertices the exact ``2^|free|`` enumeration
    is replaced by singletons, connected pairs/triples and the full
    free set, exactly as the engine does.
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
    else:
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
    if best is not None:
        return best

    # No admissible cluster: minimum total incident weight over the
    # free vertices (the engine's `_min_free_incident_nu` fallback).
    inc: dict = {}
    for (u, v), nu_e in edge_map.items():
        inc[u] = inc.get(u, 0.0) + float(nu_e)
        inc[v] = inc.get(v, 0.0) + float(nu_e)
    free_inc = [w for w in inc if w != root]
    if not free_inc:
        return min(edge_map.values())
    return min(inc[w] for w in free_inc)
