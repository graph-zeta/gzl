# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Hybrid SP-reduction + dense-core graph-zeta evaluation.

Best of both worlds between the real-space tensor
(:mod:`gzl.tensor_network`) and the cycle-space algebra: every
**series / parallel** part of a block is collapsed by the convolution
theorem (an FFT each), and only the irreducible 3-connected **core** is
contracted densely.  Translation invariance makes every edge kernel
circulant, so:

* **series** — a degree-2 vertex ``a — w — b`` is the convolution
  ``(K_1 * K_2)`` of its two edge kernels: one ``ifftn(fftn·fftn)``.
* **parallel** — two edges on the same pair multiply pointwise in real
  space (Hadamard merge), ``K_1 · K_2``.
* **loop** — an edge closing onto its own vertex contributes the
  kernel at ZERO displacement, which the regularised kernel makes 0.
  A self-loop in the input is therefore refused rather than traced.

A cycle collapses entirely → one FFT (``n log n``) instead of a dense
contraction; a dense core like ``K_4`` does not reduce and is contracted
in position space, to the **terminal position** ``M(x_t)``, the whole
Brillouin-zone grid read off by a *single* ``fftn`` at the end.  That is
also what the tensor engine does — one transform closing an all-real
elimination, on an order from the same ``_best_order(..., keep=...)``
call that holds the terminal out — so the achieved exponent is identical.
The gain is topological, not asymptotic: after SP reduction the core is a
**minor** of the block, run by the same executor on the same torus, so
hybrid never costs more than the tensor it defers to.  The constant factor
that remains is earned only where there is SP structure to collapse; on a
pure dense core hybrid is marginally *slower* than the tensor, never
cheaper by a power of ``n``.

Equal to round-off to :func:`gzl.tensor_network.graph_zeta_general`
(same truncated torus kernel), at any ``k`` — on- or off-grid.  Modes
mirror that engine: vacuum (``terminal`` is ``None`` or ``== source``),
single ``k`` (``momentum`` given), full BZ grid (``terminal != source``
and ``momentum is None``).
"""

from __future__ import annotations

import string

from collections import OrderedDict
from math import comb

import numpy as np

from gzl import _contract
from gzl import _elimination
from gzl import _sp
from gzl._labels import relabel_to_support as _relabel_to_support
from gzl._lattices import _resolve_lattice
from gzl._elimination import best_order as _best_order
from gzl._elimination import sp_reduced_cut_nu as _sp_reduced_cut_nu
from gzl.interaction import InteractionSupportError
from gzl.tensor_network import (
    TorusTruncation,
    _balanced_z_axis,
    _check_kernels,
    _refuse_interaction_in_nu,
    _edge_difference_table,
    _edge_kernel_torus,
    _factor_array,
    _min_degree_order,
    _pin_from_generator,
    _pin_from_generator_at,
)


__all__ = ["hybrid_zeta", "HybridCoreTooLargeError"]


class HybridCoreTooLargeError(MemoryError):
    """Raised when the irreducible core cannot be contracted here — the
    caller should fall back to a real-space method (e.g.
    ``graph_zeta_general`` / ``direct_sum_extrapolated``).

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Structural refusals (too many einsum letters or bucket axes) use this
    class directly: the tensor engine genuinely handles more vertices, so
    falling through to it is the right answer.
    """


class HybridCoreBudgetError(HybridCoreTooLargeError):
    """Raised when the core contraction would exceed ``max_core_bytes``.

    Distinct from its parent because the correct response is the
    OPPOSITE.  A byte-budget refusal says the contraction is too big,
    and ``graph_zeta_general`` runs the SAME contraction on the raw
    block rather than its SP minor, with no byte cap and an unchunked
    peel — so it is never smaller and is usually larger.  At d = 3 that
    is how a refused 2 GB core became a 512 GiB ``np.ones``.  A caller
    that set a budget must route this to a BOUNDED engine, not to the
    tensor.
    """


def _sp_reduce(edges, kernels, source, terminal, n, d,
               leaf_keys=None, trunc_key=None):
    r"""Collapse series/parallel structure on the torus.

    Thin adapter over :func:`gzl._sp.sp_reduce`, the shared
    truncation-agnostic rewrite, factored out of the engines as the
    executor (:mod:`gzl._contract`) and the planner
    (:mod:`gzl._elimination`) were.  The arithmetic is unchanged:
    ``TorusTruncation.compose`` is the same ``ifftn(fftn . fftn).real``
    this function used to inline, applied in the same order, so values
    are bit-identical across the relocation.

    ``edges`` is a list of ``[u, v]``; ``kernels[i]`` the real-space
    torus kernel (shape ``(n,)*d``) of edge ``i``.  Returns
    ``(nodes, residual_edges, residual_kernels, scalar)``.  Vertices
    ``source`` and ``terminal`` are never eliminated.

    ``n`` and ``d`` are accepted for call-site compatibility and to
    document the resolution the kernels carry; the rewrite reads it off
    ``kernels[i].shape``.
    """
    # The rewrite touches only compose / hadamard / trace / reverse,
    # every one of which is lattice-independent index arithmetic on
    # the generator — A never enters, so the identity is a faithful
    # placeholder rather than a silent assumption about the cell.
    #
    # The bridge CACHE is a different matter: the arrays it stores were
    # built from kernels that do depend on A, so ``trunc_key`` must
    # carry the real lattice even though the truncation object here does
    # not.  Two cells sharing (nu, n) would otherwise collide and one
    # would be served the other's bridge.
    return _sp.sp_reduce(edges, kernels, int(source), int(terminal),
                         TorusTruncation(n, d, np.eye(int(d))),
                         leaf_keys=leaf_keys, trunc_key=trunc_key)


#: Live-array retention factor of the shared executor, for the optional
#: memory cap below.  The post-peel step volume is not the whole
#: working set: an FFT-peeled step holds the padded phi, the padded
#: kernel spectrum and their product, and the previous step's result is
#: still live while the next is allocated (a cross-step factor of 2).
#: CALIBRATED, not guessed: on the d=2 tw=4 core at n=24 the predicted
#: post-peel volume is 1.424 GB against a measured 6.03 GB peak RSS
#: (4.23x); the tensor engine on the same block measures 7.46 GB
#: (4.9x).  Rounded UP so the cap errs toward refusing.
_CORE_RETENTION = 5.0

#: Retention of a peel step whose working set is CHUNKED.  The chunked
#: peel holds the ``psi`` accumulator plus at most one chunk's working
#: set, so it is priced as ``result + budget`` rather than as a multiple
#: of the result; this factor covers the factor tables and the
#: cross-step term on top of that.
#:
#: MEASURED against ``ru_maxrss`` deltas in clean subprocesses, uniform
#: nu = d + 0.5, comparing the guard's own peak with the chunked peel
#: live:
#:
#:     core     d   n     model     measured    ratio
#:     K5       3   8    1.25 GiB   1.546 GiB   1.24
#:     V6E11    3   8    1.25 GiB   1.546 GiB   1.24
#:     K5       2  20    0.75 GiB   0.846 GiB   1.13
#:
#: 1.5 clears the worst of those by 21%.  Note the same three configs
#: measure 3.75 / 3.75 / 3.20 against the UNCHUNKED peel, which is what
#: `_CORE_RETENTION = 5.0` was calibrated to bound and still does.
_CHUNKED_PEEL_RETENTION = 1.5

#: Per-axis bound on the number of two-axis factors a dense bucket over
#: ``bag`` can hold, for the table term below.  A bucket over ``k`` axes
#: can carry at most ``C(k,2)`` of them, and the term charges
#: ``min(k, C(k,2))`` tables: NONE at k = 1 -- the bucket is one-axis
#: factors only and ``TorusTruncation.dense_step`` sums it in its scalar
#: branch, materialising no table -- one at k = 2, and ``k`` from k = 3
#: up.  ``k`` is what the shipped cores actually reach and is what the
#: measurement below is calibrated against, so above k = 2 this is a
#: bound on the SHAPES THAT SHIP, not a proof.  Charging ``k`` at k = 1
#: priced a 2-node core -- a 131 kB contraction -- at ``8 N^2``, which is
#: exactly 2 GiB at d = 2, n = 128, and refused it against the cap.
_DENSE_TABLE_FACTORS = 1.0


def _core_peak_bytes(steps, n, d):
    r"""Modelled peak bytes of the dense-core contraction.

    Replaces a single ``max(n ** (d * exponent)) * _CORE_RETENTION``.
    Two corrections, both forced by measurement, and the SECOND is the
    one that matters:

    1. **A chunked peel is priced as ``result + budget``.**  Without
       chunking the peel holds phi, its spectrum and the inverse, all at
       the result's size; chunked, it holds the accumulator plus one
       chunk.  Measured 3.75x -> 1.55x on the d = 3 K5 core at n = 8.

    2. **A DENSE step allocates its RESULT, not its work volume.**  The
       torus dense step is axis-sliced (``TorusTruncation.dense_step``),
       so a bucket over ``bag`` axes allocates ``n ** (d * (|bag| - 1))``
       plus its factor tables — not ``n ** (d * |bag|)``.  For a PEEL
       step the two coincide exactly, because ``exponent = |bag| - 1``
       and the peeled output has ``|bag| - 1`` axes; that is why the old
       single-``max`` model was well calibrated despite pricing dense
       steps by work.  It is also why chunking the peel ALONE changes
       nothing: on the d = 3 K5 core the dense step at bag 3 was priced
       1.0 GiB while allocating 2 MB, and it tied the max.

    The table term is the cost the old model never carried at all: a
    dense step materialises an ``n**d x n**d`` table per two-axis
    factor.  At d = 3, n = 16 that is 134 MB apiece and it is what makes
    the K4 core measure 5.37x its modelled peak — above the 5.0 the
    guard assumed.

    It is charged per two-axis factor the bucket CAN hold,
    ``min(|bag|, C(|bag|, 2))``: none at bag 1 — a one-axis bucket is
    summed by ``dense_step``'s scalar branch and builds no table at all
    — one at bag 2, and ``|bag|`` from bag 3 up, where the calibration
    below was taken.  Charging ``|bag|`` at bag 1 refused a 2-node core
    at d = 2, n = 128 (``8 N^2`` is exactly 2 GiB there), and with it
    every d = 2 pass at ``n_points >= 128`` that met one.

    Validation (measured peak RSS delta vs this model):

        core     d   n   chunk   model      measured   headroom
        K5       3   8   on      1.88 GiB   1.546 GiB    +21%
        V6E11    3   8   on      1.88 GiB   1.546 GiB    +21%
        K5       2  20   on      1.15 GiB   0.846 GiB    +36%
        K4       3  16   off     0.65 GiB   0.672 GiB     -3%
        V6E11    2  16   off     0.65 GiB   0.413 GiB    +58%

    The K4 row's shortfall is not new: the single-``max`` model this one
    replaced priced that same configuration at 0.625 GiB against the
    same 0.672 GiB measurement, 7% under rather than 3%.  It is recorded
    rather than fixed because fixing it means raising the cap.

    Re-derived when the table count became ``min(|bag|, C(|bag|, 2))``
    (macOS/arm64, same method — ``ru_maxrss`` delta in a clean
    subprocess, ``max_core_bytes`` lifted so the guard prices but never
    refuses).  The priced maximum is a bag ≥ 3 step on every row, so the
    correction changes NO row's model; the measured column reproduces
    the one above within noise:

        core     d   n   chunk   steps (bag, peel)            model      measured   headroom
        K5       3   8   on      4p 3d 2d 1d                  1.875 GiB  1.539 GiB    +22%
        V6E11    3   8   on      4p 3d 2d 1d                  1.875 GiB  1.532 GiB    +22%
        K5       2  20   on      4p 3d 2d 1d                  1.090 GiB  0.846 GiB    +29%
        K4       3  16   off     3p 2d 1d                      0.625 GiB  0.672 GiB     -7%
        V6E11    2  16   off     4p 3d 2d 1d                  0.625 GiB  0.415 GiB    +51%

    The peel branches carry no table term although a peel densifies
    every OTHER kernel in its bucket (``_peel_convolution``); that
    under-charge is what the K4 row's negative headroom is, it predates
    the count correction (old model == new model there), and it is
    deferred with this table as its gate: if a row ever loses headroom
    to a change on the dense side, the peel-side term is the fix.
    """
    from gzl import tensor_network as _tn

    N = int(n) ** int(d)
    table_bytes = 8.0 * float(N) ** 2
    peak = 0.0
    for st in steps:
        bag = len(st.bag)
        result = 8.0 * float(n) ** (int(d) * max(bag - 1, 0))
        if st.partners:
            chunk = _tn._peel_chunk_length(max(bag - 1, 1), N, int(n), N)
            if chunk is not None:
                step = ((result + float(_tn._PEEL_BUDGET_BYTES))
                        * _CHUNKED_PEEL_RETENTION)
            else:
                step = result * _CORE_RETENTION
        else:
            # A bucket over k axes holds at most C(k,2) two-axis factors:
            # none at bag 1 (dense_step's scalar branch), one at bag 2.
            # bag >= 3 is unchanged (bag <= C(bag,2)), so every row of
            # the validation table above keeps its number.
            n_tables = min(_DENSE_TABLE_FACTORS * bag, float(comb(bag, 2)))
            step = result * _CORE_RETENTION + n_tables * table_bytes
        peak = max(peak, step)
    return peak


def _dense_core(nodes, edges, kernels, pins, terminal, n, d, A,
                max_core_bytes=None, peelable=True, *, order=None):
    r"""Contract the (irreducible) residual graph in position space.

    ``pins`` fixes vertices at explicit torus sites.  A bare ``int`` is
    the historical calling convention and means that vertex pinned at
    the ORIGIN — every caller that passes one, positional included,
    keeps its exact behaviour.  The general form is a sequence of
    ``(vertex, flat_site)`` pairs: each vertex is held at that flat
    torus index, its incident kernels collapse to one-axis potentials
    (or to scalars where both endpoints are pinned), and its axis never
    enters a bag.  ``pins=((source, 0),)`` IS the historical core;
    ``pins=((source, 0), (w, z))`` is one cell of the slab
    (:mod:`gzl.slab`), whose inner loop is this function — the
    slab is not a second engine, it is this core with one more pin and
    an outer sum over ``z`` restoring the exact value.

    All non-pinned, non-terminal vertices are summed; the terminal
    position is kept open.  Returns ``M`` of shape ``(n,)*d`` = ``ζ`` as
    a function of the terminal lattice position (or a 0-d array for the
    vacuum case ``terminal == pins[0].vertex``).  A terminal equal to a
    NON-first pinned vertex is refused: its position is fixed, so
    "kept open" is unsatisfiable and the historical vacuum test would
    silently mis-shape the result.

    ``order`` overrides the elimination order (a sequence of the free
    vertices).  ``None`` keeps the historical choice: the
    achieved-exponent DP at a single pin, min-degree above one pin —
    matching :func:`gzl.slab.slab_schedule`, so a direct
    multi-pin call and the slab wrapper contract on the same schedule.
    Re-ordering is value-exact on the torus; it is threaded through so
    the slab's shipped numbers stay BIT-identical, not merely exact.

    The contraction itself is the shared bucket-elimination executor —
    :func:`gzl._contract.eliminate` on a
    :class:`gzl.tensor_network.TorusTruncation` — i.e. the same
    code path, FFT peel included, that ``graph_zeta_general`` runs on
    the same factors.

    **There is no cost-based refusal, deliberately.**  An earlier
    version of this function ran a pre-flight ``np.einsum_path`` greedy
    search and refused the core on that path's FLOP total — but the
    execution loop moved onto the shared executor on the ``best_order``
    schedule, so the guard was pricing a path the engine no longer
    took.  It was measurably arbitrary (four census cores of *identical*
    planner cost scored 6.6e11 and one scored 5.07e14, a 767x spread,
    and the passing ones cleared the 1e12 budget by only 1.5x) and it
    refused work the engine does well: the 5.07e14 core contracts in
    9.45 s / 6.03 GB, while the tensor engine the frontend falls back
    to on refusal takes 12.48 s / 7.46 GB for the same value to ~1 ulp.

    That is the structural reason a cost refusal cannot pay here: after
    SP reduction the core is a topological *minor* of the block, run by
    the *same* executor on the *same* torus, so hybrid never costs more
    than the fallback it defers to — refusing only picks the slower,
    hungrier engine.  Two **structural** refusals remain, both real
    limits rather than cost heuristics: the einsum letter budget, and a
    bucket whose output-axis count exceeds the 26 lowercase letters the
    dense step spells with.

    ``max_core_bytes`` is an OPTIONAL cap (default ``None`` = no cap)
    for callers that must bound an allocation — CI, or a batch job
    sharing a machine.  It is priced on what the executor actually
    pays: the planner's post-peel peak step volume times
    :data:`_CORE_RETENTION`.  Note it does NOT model the raw bag volume
    (820 GB where the measured peak is 6 GB, because the peel means the
    bag is never materialised) — a bag-volume cap would over-refuse far
    more brutally than the guard this replaced.

    Memory discipline: factors are handed to the executor lazily, as
    circulant generators in the torus engine's ``(axes, table, gen)``
    representation; a dense ``n^{2d}`` table is materialised only by
    the step that consumes it and released with the bucket, FFT-peeled
    steps never build one, and source-incident edges never do either
    (their pinned slice is read straight off the generator).

    ``peelable=False`` keeps every factor dense so the whole
    contraction stays in exact (integer, for the ν = ∞ nearest-
    neighbour indicator) arithmetic — an FFT round trip would return
    2.0 as 1.9999999999999998.  The densification happens *after* the
    structural refusals and the optional cap have vetted the core, and
    the cap prices it correctly: with no peel available the post-peel
    volume degenerates to the raw bag volume, which is what that
    contraction really pays.
    """
    letters = string.ascii_letters
    idx = {v: i for i, v in enumerate(nodes)}
    if isinstance(pins, (int, np.integer)):
        pins = ((int(pins), 0),)
    pins = tuple((int(v), int(z)) for v, z in pins)
    pin_site = dict(pins)
    src_raw, ter_raw = pins[0][0], int(terminal)
    for v_pin, _ in pins:
        idx[v_pin]                       # KeyError on a pin outside nodes
    idx[ter_raw]
    if len(pin_site) != len(pins):
        raise ValueError(f"duplicate pinned vertex in pins={pins}")
    if any(v == ter_raw for v, _ in pins[1:]):
        raise ValueError(
            f"terminal {ter_raw} is pinned at an explicit site; a pinned "
            f"position cannot be kept open"
        )
    if len(nodes) > len(letters):
        raise HybridCoreTooLargeError(
            f"hybrid core needs {len(nodes)} einsum letters "
            f"(> {len(letters)}) — fall back to a real-space method."
        )

    # Factor list in the torus engine's representation: axes ascending
    # (edges are sorted pairs carrying canonically oriented kernels, so
    # T[i, j] = K[(i - j) mod n] on the sorted axes), lazy generators
    # for vertex-vertex edges, pinned slices read off the generator for
    # source-incident ones.
    factors = []
    scalar = 1.0
    for (a, b), K in zip(edges, kernels):
        a, b = int(a), int(b)
        sa, sb = pin_site.get(a), pin_site.get(b)
        if sa is not None and sb is not None:
            # Both endpoints at explicit sites: the edge is the scalar
            # K[(site_a - site_b) mod n].  At (0, 0) this is bitwise the
            # historical K[(0,) * d] (verified: the roll-by-0 gather is a
            # copy), so the single-pin path is unchanged; under a second
            # pin it is the hot branch every source–slab edge takes.
            scalar *= float(
                _pin_from_generator_at(K, sb, pin_row=False)[sa])
            continue
        if sa is not None:            # fix x_a = site: row K(site - x_b)
            factors.append(
                ([b], _pin_from_generator_at(K, sa, pin_row=True), None))
            continue
        if sb is not None:            # fix x_b = site: row K(x_a - site)
            factors.append(
                ([a], _pin_from_generator_at(K, sb, pin_row=False), None))
            continue
        factors.append(([a, b], None, K))

    vacuum = ter_raw == src_raw
    if not factors:
        M = np.array(1.0) if vacuum else np.ones((n,) * d)
        return np.asarray(M) * scalar

    # Vertex elimination order, chosen exactly as graph_zeta_general
    # chooses it: achieved-exponent DP with the min-degree fallback,
    # source pinned, terminal kept.  Re-ordering is value-exact on the
    # torus (same finite sum, different association).
    adj = {int(v): set() for v in nodes}
    for a, b in edges:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    keep_set = set() if vacuum else {ter_raw}
    if order is not None:
        order = [int(v) for v in order]
        pinned_set = set(pin_site)
        if any(v in pinned_set or v in keep_set for v in order):
            raise ValueError(
                "order must contain only free, non-terminal vertices"
            )
    elif len(pins) == 1:
        order = _best_order(
            [tuple(int(x) for x in e) for e in edges],
            [int(v) for v in nodes],
            src_raw,
            keep=keep_set,
            fallback=lambda: _min_degree_order(
                adj, [int(v) for v in nodes],
                terminals=keep_set | {src_raw}),
        )
    else:
        # More than one pin and no caller order: min-degree over the
        # free vertices — the same choice `slab_schedule` makes, so a
        # direct call and the slab wrapper contract the same schedule.
        # (`_best_order`'s DP excludes exactly ONE pin from the order;
        # handing it a multi-pin core would eliminate a pinned vertex.)
        order = _min_degree_order(
            adj, [int(v) for v in nodes],
            terminals=keep_set | set(pin_site))

    # Symbolic pass over the schedule: the executor's dense step spells
    # a bucket's output axes in lowercase einsum letters, so a bucket
    # with more than 26 output axes must be refused HERE — as a budget
    # error the frontend can fall back on — not mid-contraction.
    scopes = [tuple(ax) for ax, _, _ in factors]
    for v in order:
        in_bucket = [s for s in scopes if v in s]
        scopes = [s for s in scopes if v not in s]
        union_ax = sorted({x for s in in_bucket for x in s})
        out_axes = tuple(x for x in union_ax if x != v)
        if len(out_axes) > len(string.ascii_lowercase):
            raise HybridCoreTooLargeError(
                f"hybrid core bucket has {len(out_axes)} output axes "
                f"(> {len(string.ascii_lowercase)} einsum letters) at "
                f"vertex {v} — fall back to a real-space method."
            )
        scopes.append(out_axes)

    # Optional memory cap, priced on the schedule the executor will
    # actually run (see the docstring for why there is no cost-based
    # refusal by default).  Off unless the caller asks for it.
    if max_core_bytes is not None:
        pairs = sorted({(min(int(u), int(v)), max(int(u), int(v)))
                        for u, v in edges})
        # Densified (nu = inf) cores never peel, so the post-peel
        # volume correctly degenerates to the raw bag volume.
        if len(pins) == 1:
            p_plan = _elimination.plan_for_order(
                pairs, [int(v) for v in order], src_raw, keep=keep_set,
                non_kernel_edges=pairs if not peelable else (),
            )
            steps = p_plan.steps
        else:
            # The single-pin plan cannot see the extra pins' removed
            # axes and would OVER-price the core (conservative, but
            # wrong).  Build the scopes at the first pin, drop every
            # other pinned vertex from every scope — exactly what the
            # factor assembly above did — and simulate the real order.
            extra = set(pin_site) - {src_raw}
            scopes = [
                (frozenset(sc) - extra, is_k and not (frozenset(sc) & extra))
                for sc, is_k in _elimination.scopes_from_edges(
                    pairs, src_raw,
                    non_kernel_edges=pairs if not peelable else ())
                if frozenset(sc) - extra
            ]
            steps = _elimination.simulate(scopes, [int(v) for v in order])
        est = _core_peak_bytes(steps, n, d)
        if est > float(max_core_bytes):
            peak = max((float(n) ** (d * st.exponent)
                        for st in steps), default=1.0)
            raise HybridCoreBudgetError(
                f"hybrid core peak ~{est:.2e} B (post-peel step volume "
                f"{peak:.2e} elems; see _core_peak_bytes for the "
                f"per-step model) exceeds "
                f"max_core_bytes={float(max_core_bytes):.2e} "
                f"(n={n}, d={d}) — fall back to a real-space method."
            )

    if not peelable:
        # ν = ∞ bundles: keep the exact integer arithmetic of the dense
        # contraction, exactly like graph_zeta_general's eager tables.
        # Only now — the budget guard above has already vetted the core.
        factors = [(ax, _factor_array(T, g, n), None)
                   for ax, T, g in factors]

    # Execute on the shared bucket-elimination executor.  One truncation
    # object per call: its methods read tensor_network's tunables
    # (_USE_CONV etc.) late-bound, preserving per-call kill-switch
    # semantics.
    trunc = TorusTruncation(n, d, A)
    residual = _contract.eliminate(factors, order, trunc)

    r_scalar = np.array(1.0)
    vec = None
    for ax, T, g in residual:
        arr = _factor_array(T, g, n)
        if len(ax) == 0:
            r_scalar = r_scalar * arr
        else:
            if list(ax) != [ter_raw]:
                raise RuntimeError(
                    f"unexpected leftover axes after elimination: {ax}"
                )
            vec = arr if vec is None else vec * arr
    if vacuum:
        M = np.asarray(r_scalar)
    else:
        vec = np.ones(n ** d) if vec is None else vec
        M = (r_scalar * vec).reshape((n,) * d)
    return np.asarray(M) * scalar


#: Let the planner choose the pin on the vacuum path.  Set False to pin
#: at the caller's ``source``, which is what earlier versions did.  Not
#: bit-identical -- a different pin re-associates the same sum -- so it
#: keeps a switch, but it is value-exact by translation invariance and
#: was measured at 1.55e-16, with several grids agreeing BITWISE.
USE_PLANNER_PIN = True


#: Memo for :func:`_planner_pin`.  The choice is a pure function of the
#: block's topology and exponents, and a corpus pass re-evaluates the same
#: few hundred shapes thousands of times — measured 1075 hybrid calls over
#: a small number of distinct blocks at d = 2 order 11.  Without this the
#: per-pin scan below was 7.9 s of a 13 s pass, the largest single line
#: item, which is a poor trade for a decision that never changes.
_PIN_CACHE: "OrderedDict[tuple, int]" = OrderedDict()
_PIN_CACHE_MAX = 4096

#: Number of times the exponent tie-break below has run.  A silent
#: no-op here is indistinguishable from "no ties occurred", so the
#: counter is what a test asserts on.
_PIN_TIEBREAK_COUNT = 0


def _pin_remember(key, pin: int) -> int:
    if key is not None:
        _PIN_CACHE[key] = int(pin)
        if len(_PIN_CACHE) > _PIN_CACHE_MAX:
            _PIN_CACHE.popitem(last=False)
    return int(pin)


def _planner_pin_key(edges, nu_vec):
    """Canonical, order-independent identity for the pin decision."""
    return tuple(sorted(
        (min(int(u), int(v)), max(int(u), int(v)), float(w))
        for (u, v), w in zip(np.asarray(edges)[:, :2], np.asarray(nu_vec))
    ))


def _planner_pin(edges, nu_vec, source: int) -> int:
    r"""The pin to eliminate around, on a block whose value does not care.

    On the torus the pin is FREE: translation invariance makes
    ``zeta_B`` independent of which vertex is held at the origin, so the
    only thing it changes is cost.  And it changes it by a full power of
    ``n^d``.

    ``hybrid_zeta`` had always pinned the caller's ``source``, whose
    default is ``0``, and never asked ``_elimination.plan`` -- which
    computes the cost-optimal pin, and was already computing it for the
    ORDER.  On K5-minus-an-edge with the terminals at the missing edge's
    endpoints (a real corpus block) that is contraction exponent 3 at
    vertex 0 against 2 at vertex 1, i.e. at d = 2, n_points = 32:

        pinned at source=0     33.002 s   27.47 GB
        pinned by the planner   0.065 s    0.20 GB    -> 508x, 137x memory

    agreeing to 1.55e-16, and BITWISE at n = 24.  Over the d = 2 order-11
    census, 31 of 576 distinct tw >= 3 blocks drop from exponent 3 to 2
    this way; the 4 that do not are treewidth 4, where 3 IS the width and
    ``e(pi) >= w(pi) >= theta`` forbids any pin from helping.

    ``plan`` with no ``pin`` already returns the argmin-exponent pin --
    measured on 63 of 63 random connected graphs -- so this costs one
    plan call, not one per vertex.

    ONLY on the vacuum path.  With both terminals kept the exponent is 3
    at every pin (measured), and more to the point the pin is no longer
    free: the kept indices are what the caller asked for.
    """
    if not USE_PLANNER_PIN:
        return int(source)
    try:
        key = _planner_pin_key(edges, nu_vec)
    except Exception:
        key = None
    if key is not None:
        hit = _PIN_CACHE.get(key)
        if hit is not None:
            _PIN_CACHE.move_to_end(key)
            return int(hit)
    try:
        vertices = sorted({int(v) for e in edges for v in e[:2]})
        # No nu = inf case to worry about: `hybrid_zeta` refuses it, so
        # every factor here peels and the peel model is the right one.
        exps = {p: _elimination.plan(edges, vertices, pin=p).exponent
                for p in vertices}
        best = min(exps.values())
        ties = [p for p in vertices if exps[p] == best]
        if len(ties) == 1:
            return _pin_remember(key, int(ties[0]))
        # TIE-BREAK ON THE SP-REDUCED CUT, and it is not cosmetic.
        #
        # The pin is never eliminated, so pinning a DEGREE-2 vertex keeps
        # its escape alive and leaves sigma_core at the block's own rate —
        # which is the pass floor, so the core-sizing rule then correctly
        # concludes there is no gap to spend and declines.  Pinning
        # anywhere else lets that vertex reduce away and the core's cut
        # jumps.
        #
        # Measured on the d = 2 order-11 block that dominated the pass
        # (V6E11, tw 4, degrees [2,4,4,4,4,4], exponent 3 at EVERY pin so
        # the exponent alone cannot choose):
        #
        #     root = 0 (lowest label)   sigma_core = 3.00   grading declines
        #     root = 1..5               sigma_core = 8.00   n_core = 12
        #
        # i.e. 17.2 GB against 48 MB, from a tie-break.  15 of 576 d = 2
        # blocks were declining the grading for exactly this reason, and
        # all 15 had sigma_core pinned down to the floor.
        emap: dict = {}
        for (u, v), w in zip(np.asarray(edges), np.asarray(nu_vec)):
            ekey = (min(int(u), int(v)), max(int(u), int(v)))
            emap[ekey] = emap.get(ekey, 0.0) + float(w)
        # Lowest label wins a further tie, so the choice stays
        # deterministic and the block cache cannot serve two answers.
        global _PIN_TIEBREAK_COUNT
        _PIN_TIEBREAK_COUNT += 1
        return _pin_remember(
            key, int(max(ties, key=lambda q: (_sp_reduced_cut_nu(emap, q), -q))))
    except Exception:
        # A planner refusal is not a reason to fail the evaluation; the
        # caller's pin is always valid, just possibly dearer.
        return int(source)


def _momentum_is_zero(momentum) -> bool:
    r"""True when ``momentum`` is EXACTLY the zero vector, either sign.

    The one definition of k = 0 shared by :func:`hybrid_zeta`'s vacuum
    rewrite, the frontend's block-cache key
    (``frontend._momentum_cache_key``) and the finite-k arm's exemption
    of an explicit zero from treewidth-2 core grading
    (``frontend._block_at_finite_k``), so that a cache entry cannot hold
    one truncation under another's name.  ``None`` is not zero: here it
    is the full-BZ grid request.

    No tolerance, deliberately: 1e-16 is not k = 0 to every evaluator.
    Below, a kept terminal on a graded core is a different truncation
    from the vacuum rewrite, and the frontend's closed forms resolve the
    |k|^σ cusp of the lattice sum (an on-spine bridge at k = 1e-16 is
    3.0e-02 off its k = 0 value at σ = 0.1).  The measured cost of the
    key's old 1e-15 tolerance is in ``frontend._momentum_cache_key``.
    Signed zeros are one momentum: measured bit-identical on every
    finite-k route (bridge, algebra, nearest-neighbour, hybrid on a
    treewidth-2 and a dense block, the pinned slab, the box, the tensor).
    """
    return momentum is not None and not np.any(
        np.asarray(momentum, dtype=np.float64))


def hybrid_zeta(
    edges,
    nu_vec,
    A,
    n_points: int,
    *,
    source: "int | None" = None,
    terminal: int | None = None,
    momentum: np.ndarray | None = None,
    max_core_bytes: "float | None" = None,
    sp_n_points: "int | None" = None,
    core_n_points: "int | None" = None,
    kernels: "Sequence | None" = None,
):
    r"""Graph zeta via SP-reduction (FFT) + dense-core contraction.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Agrees with :func:`gzl.tensor_network.graph_zeta_general` to
    round-off (same truncated torus kernel), but collapses every
    series/parallel part with one FFT and only contracts the irreducible
    core densely.

    Agreement is to round-off, not bit-identity: the FFT collapse
    re-associates the sum, and an irreducible core — though it runs on
    the same shared bucket-elimination executor as the tensor engine —
    contracts an SP-reduced factor list rather than the raw edge list.
    Measured before the core moved onto the shared executor, over 48
    topology/dimension/ν/mode configurations on orthogonal lattices, 45
    agreed exactly and 3 differed by at most 7.4e-16; the executor
    delegation moves core values by at most a few ulp of scale (measured
    per case in ``tests/test_hybrid_s7_executor.py``).  Do not assert
    exact equality between the engines.

    Edge orientation is tracked explicitly through the reduction (see
    :func:`_oriented`).  It matters only on a sheared lattice at even
    ``n_points``, where ``_balanced_z_axis`` holds ``n/2`` without
    ``-n/2`` so the kernel is genuinely not even; before that was
    handled this function was wrong by 4.0e-05 at ``n = 8`` on a
    triangular lattice while :func:`graph_zeta_general` sat at 1e-16.

    ``sp_n_points`` (default ``None`` = off) turns on **split
    resolution**: the series/parallel part is collapsed on a torus of
    ``sp_n_points`` while the irreducible core is contracted on
    ``n_points``, the residual kernels being carried across by
    :meth:`~gzl.tensor_network.TorusTruncation.restrict_to`.

    The point is the truncation RATE, not the cost.  Truncation error
    falls as ``n^{-sigma_eff}`` with ``sigma_eff = (minimum cluster
    cut) - d``, and the cheapest escape is a degree-2 vertex — cut
    ``2 nu`` — which is exactly what the SP reduction removes.  A
    3-connected core has edge connectivity >= 3, so its own cut is
    ``>= 3 nu``: collapsing the SP part finely and contracting only the
    core coarsely buys a full ``nu`` in the exponent.

    Measured on a subdivided ``K_4`` at ``nu = 2.5``, and gated in
    ``tests/test_hybrid_split.py`` so these stay honest:

    * d = 1 (``sigma = 1.5``), TRUE errors against a converged
      ``graph_zeta_general`` at n = 4096 — the uniform family fits
      ``-3.98`` against its predicted ``2 nu - d = 4.0``, which is what
      validates the estimator, and only then does the split's ``-6.36``
      against ``3 nu - d = 6.5`` mean anything.  The gap ``-2.38`` is
      the predicted ``-nu``.  In absolute terms the split is 25x closer
      at ``n_points = 24`` and 669x at 96.
    * d = 2 (``sigma = 0.5``), where a converged reference costs
      ``n^{tau d} = n^4`` and is not affordable: successive differences
      on a geometric ladder give a gain of ``-2.00``.  That is SHORT of
      the predicted ``-nu = -2.5``, on a ladder topping out at n = 64;
      read it as confirming the direction and most of the magnitude, not
      as attaining the exponent.
    * finite ``k``: the gain holds at every ``k`` sampled across the
      Brillouin zone, the zone boundary included.

    Scope: ``nu > d`` throughout.  These are two block shapes on
    orthogonal and triangular cells, not a corpus-wide constant.

    ``sp_n_points == n_points`` is a no-op *numerically* — the gather
    is an exact identity there — but it still takes the split code
    path, which is what makes it a usable self-check.  ``None`` skips
    the branch outright, so the default is today's arithmetic
    unchanged rather than merely equal to it.

    Refused rather than silently degraded: ``sp_n_points < n_points``
    (there is nothing to restrict *up* to), and any infinite ``nu``
    (the kernel is then a nearest-neighbour indicator whose exactness
    an FFT collapse would destroy — ``2.0`` comes back as
    ``1.9999999999999998``).

    ``core_n_points`` (default ``None`` = off) is the **downward half
    of the same axis**: where ``sp_n_points`` raises the SP grid above
    ``n_points``, this lowers the irreducible core below it.  Both open
    a fine/coarse gap and the gap is the whole point — the coarse side
    converges at the SP-reduced core's cut (``>= 3ν − d``) rather than
    the block's (``2ν − d``), so it reaches a given accuracy at a much
    smaller ``n``.  It is also the expensive side, costing
    ``n^(τ·d)`` against the fine grid's ``n^d``, which is why moving it
    down buys more than moving the other one up.

    The three grids are ordered ``sp_n_points >= n_points >=
    core_n_points``, and only ``n_points`` is tied to the answer: a
    vacuum call returns a scalar, a single-``k`` call evaluates the
    trigonometric polynomial straight off the coarse array, and only
    the full-BZ-grid call has to come back up — by
    :meth:`TorusTruncation.embed_into`, which is exact for the
    approximant rather than an interpolation.

    Refused, again rather than degraded: ``core_n_points > n_points``
    (that is ``sp_n_points``'s job), and infinite ``nu`` (the
    nearest-neighbour indicator needs ``n`` above the block's longest
    cycle for the constraint to close, so a coarser core changes which
    walks exist rather than approximating the answer).

    ``max_core_bytes`` does NOT bound the fine grid.  It prices the
    core contraction, which runs at ``core_n_points`` (``n_points``
    when that is off) — so a capped call can
    still allocate the SP grid's ``sp_n_points^d`` kernels and the
    transform buffers of the series convolution on top of the cap.  At
    d = 3 that is the dominant term by far (measured: 3.80 GB peak at
    ``sp_n_points = 256`` against a core costing 0.01 GB), so a caller
    bounding total memory must bound ``sp_n_points`` itself.

    Scope of the evidence.  CORRECTNESS is checked at d = 1, 2 AND 3 at
    physical ``nu > d`` -- vacuum, single ``k`` and full BZ grid, uniform
    and non-uniform ``nu``, cubic and sheared cells, even and odd ``n``,
    0/1/2 terminals -- against ``graph_zeta_general`` and an independent
    lattice-sum oracle.  ``nu = inf`` is NOT exact here: the SP collapse
    FFTs, so a densified core can return 5.9999999999999964 for 6.  The
    frontend never routes ``nu = inf`` to this engine.  The box
    truncation is untested.

    COST is gated only at d <= 2.  ``_CORE_RETENTION`` is a d = 2
    calibration; at d = 3 the measured retention reaches ~6.6x on tw = 3
    cores, so ``max_core_bytes`` is a SOFT ceiling there -- expect
    ~25-30% overshoot -- not a hard one.  It also does not bound the
    fine SP grid (see above).

    A REFUSAL IS NOT FREE, and this function does not choose what it
    costs.  A BUDGET refusal raises ``HybridCoreBudgetError``, which the
    frontend re-raises to a BOUNDED engine, because
    ``graph_zeta_general`` runs the same contraction on the raw block
    rather than its SP minor, with no byte cap -- at d = 3 that is how a
    refused 2 GB core became a 512 GiB allocation.  A STRUCTURAL refusal
    (einsum letters, bucket axes) raises the plain parent class and does
    fall through to the tensor, which genuinely handles more vertices.

    ``kernels`` (default ``None``) supplies GENERAL per-edge kernels —
    :class:`gzl.interaction.Interaction` objects or lazy products
    of them, aligned with ``edges``.  ``nu_vec`` then carries each
    kernel's TAIL exponent (``Interaction.tail_exponent``), which is
    what the refusals above, the planner pin and the peel decision
    read, unchanged; only the kernel build and the cache names consult
    the kernel itself.  A purely compact kernel has tail ``+inf`` and
    is refused exactly like ``nu = inf`` (same reason: the SP collapse
    FFTs).  The bridge cache names a general edge ``("I", key())``,
    never ``("E", nu)``: a mixed kernel reusing the float's name was
    served a stale bridge with a sign-flipped 2.0 relative error.  A
    compact part that the CORE grid cannot hold is refused with
    :class:`gzl.interaction.InteractionSupportError` before the
    restriction could clip it.  ``None`` is the legacy power-law path,
    byte-identical.

    Returns a real ``float`` (vacuum or single ``k``) or a real ``ndarray``
    of shape ``(n_points,)^d`` (full BZ grid: ``terminal != source`` and
    ``momentum is None``).

    ``A`` is the lattice matrix or a lattice name -- ``"chain"``,
    ``"square"``, ``"triangular"``, ``"cubic"``.
    """
    A = np.asarray(_resolve_lattice(A), dtype=float)
    d = int(A.shape[0])
    n = int(n_points)
    # The vertex set is the edge support (see gzl/_labels.py):
    # sparse labels compress order-preservingly (the engine was already
    # support-based and bit-agrees on a relabelled graph); a source or
    # terminal label appearing in no edge raises instead of the bare
    # KeyError (terminal) / silent planner re-pin (source) it used to.
    _edges_arr = np.asarray(edges, dtype=int)
    _edges_arr, _refs = _relabel_to_support(
        _edges_arr,
        {"source": None if source is None else int(source),
         "terminal": None if terminal is None else int(terminal)})
    source = 0 if _refs["source"] is None else int(_refs["source"])
    terminal = _refs["terminal"]
    edges = [tuple(int(x) for x in e) for e in _edges_arr]
    _refuse_interaction_in_nu(nu_vec, "hybrid_zeta")
    nu_vec = np.asarray(nu_vec, dtype=float)
    kernels = _check_kernels(kernels, nu_vec, len(edges), edges)

    # nu = inf IS NOT SUPPORTED HERE, and refusing beats optimising it.
    #
    # At nu = inf the kernel is the NN indicator and the value is an
    # INTEGER homomorphism count, but this engine collapses the
    # series/parallel part by FFT, so the answer comes back off the
    # integer: measured 99.99999999999996 for 100 on a theta graph at
    # d = 2, and negative counts on other shapes.  `graph_zeta_general`
    # is exact there because it never leaves the integer arithmetic.
    #
    # Nothing loses a capability: `frontend._block_general` already
    # gates on `not np.any(np.isinf(nu_arr))`, so the router has never
    # sent nu = inf here.  This makes that skip explicit instead of
    # incidental -- and stops the internal machinery (the `peelable`
    # arm, the pin's cost model) being tuned for a path whose answers
    # are wrong anyway.
    if bool(np.isinf(nu_vec).any()):
        raise ValueError(
            "hybrid_zeta: nu = inf (the nearest-neighbour indicator limit) "
            "is not supported -- the SP collapse is an FFT, so the integer "
            "homomorphism count comes back inexact (measured "
            "99.99999999999996 for 100).  Use graph_zeta_general, which is "
            "exact at nu = inf; gzl.evaluate_graph already routes "
            "there automatically."
        )

    # Grid the SP collapse runs on.  ``None`` is not "the same number"
    # — it is a different branch, so the default path is today's
    # arithmetic itself rather than something equal to it.
    if sp_n_points is None:
        n_sp = n
    else:
        n_sp = int(sp_n_points)
        if n_sp < n:
            raise ValueError(
                f"hybrid_zeta: sp_n_points={n_sp} is below n_points={n}; "
                f"the SP grid is restricted DOWN onto the core grid, so it "
                f"cannot be the coarser of the two."
            )
        if bool(np.isinf(nu_vec).any()):
            raise ValueError(
                "hybrid_zeta: sp_n_points is not supported for infinite nu. "
                "The kernel is then the nearest-neighbour indicator and the "
                "contraction is exact integer arithmetic; an FFT collapse "
                "returns 2.0 as 1.9999999999999998.  There is also nothing "
                "to buy — an indicator kernel has no power-law tail to "
                "truncate."
            )

    # Grid the irreducible CORE is contracted on.  ``None`` is a separate
    # branch for the same reason ``sp_n_points`` is: the default path
    # stays today's arithmetic itself rather than something equal to it.
    #
    # This is the downward half of the same axis.  ``sp_n_points`` raises
    # the SP grid above ``n_points``; ``core_n_points`` lowers the core
    # below it.  Both open a fine/coarse gap, and the gap is the point:
    # the truncation rate on the coarse side is the SP-reduced CORE's
    # minimum cluster cut (>= 3ν − d for a 3-connected core) rather than
    # the block's (2ν − d).  Because the core converges a full ν faster,
    # it reaches any given accuracy at a much smaller n — and it is the
    # expensive half, costing n^(τ·d) against the fine grid's n^d.
    #
    # MEASURED, d = 2, σ = 0.5, top census block (tw 3, V5 E7, occ 53),
    # per-block relative error against a two-family reference:
    #
    #     n_core      raw        split (sp = 1536)
    #          8   6.32e-03          3.48e-05
    #         16   9.63e-04          3.36e-07
    #         24   2.90e-04            <floor
    #
    # i.e. a core at n = 8 with the split is 8x MORE accurate than a core
    # at n = 24 without it, for 1/81 of the contraction.  That gap is what
    # this parameter exists to spend.
    if core_n_points is None:
        n_core = n
    else:
        n_core = int(core_n_points)
        if n_core > n:
            raise ValueError(
                f"hybrid_zeta: core_n_points={n_core} exceeds "
                f"n_points={n}; the core grid is what the SP grid is "
                f"restricted DOWN onto, so it cannot be the finer of the "
                f"two.  To raise the SP grid instead, use sp_n_points."
            )
        if bool(np.isinf(nu_vec).any()):
            raise ValueError(
                "hybrid_zeta: core_n_points is not supported for infinite "
                "nu.  The nearest-neighbour indicator needs n > the "
                "block's longest cycle for the constraint to close, so a "
                "coarser core does not approximate the answer — it changes "
                "which walks exist.  There is also nothing to buy: an "
                "indicator kernel has no power-law tail to truncate."
            )

    if kernels is not None and (sp_n_points is not None or n_core != n):
        # The SP grid is gathered DOWN onto the core grid below
        # (``restrict_to``), and a gather cannot see a table: a compact
        # part whose support does not fit the core window would be
        # clipped silently into a plausible wrong number.  The balanced
        # axis holds both +R and -R only for R <= (n_core - 1) // 2.
        # Belt and braces on the INPUT radii: the router floors a graded
        # core at ``frontend._compact_core_floor`` -- the SP-reduced
        # residual radii (series chains add, parallel merges take the
        # max) and the block's winding-safe size -- which implies this
        # check; a direct call can still get it wrong.
        half = (int(n_core) - 1) // 2
        for i, kern in enumerate(kernels):
            if kern.has_compact and int(kern.support_radius) > half:
                raise InteractionSupportError(
                    f"hybrid_zeta: kernels[{i}] has a compact part of "
                    f"Chebyshev radius {int(kern.support_radius)}, which the "
                    f"core grid core_n_points={int(n_core)} cannot hold "
                    f"(needs n >= {2 * int(kern.support_radius) + 1}); "
                    f"restricting the SP grid onto it would clip the table.  "
                    f"The router floors a graded core at the winding-safe "
                    f"size of the residual tables; evaluate on one grid instead."
                )

    # One array per DISTINCT exponent, not per edge.  The kernels are
    # read-only inputs to the reduction — the parallel merge, the series
    # convolution and the orientation flip all return new arrays, and
    # the executor builds its own tables — so sharing is safe.
    #
    # Measured over both shipped corpora the duplication factor is 4.22x
    # (23277 edges carrying 5510 distinct exponents; the modal block has
    # 8-10 edges and just 2 distinct exponents).  It matters at d = 3,
    # where a 7-edge uniform-nu block at sp_n_points = 384 costs 2.95 GB
    # one-array-per-edge against 0.42 GB deduplicated.  It is NOT the
    # binding term, though: the series convolution's complex transform
    # buffers are.
    _kernel_cache: dict = {}
    gens = []
    if kernels is None:
        for i in range(len(edges)):
            nu_i = float(nu_vec[i])
            K = _kernel_cache.get(nu_i)
            if K is None:
                K = _kernel_cache[nu_i] = _edge_kernel_torus(nu_i, A, n_sp)
            gens.append(K)
    else:
        # General kernels: the per-call dict keys on the kernel's own
        # fingerprint, never on its tail — two kernels sharing a tail
        # are different arrays.  The global cache dedupes the build.
        for i in range(len(edges)):
            ck = kernels[i].key()
            K = _kernel_cache.get(ck)
            if K is None:
                K = _kernel_cache[ck] = _edge_kernel_torus(
                    float(nu_vec[i]), A, n_sp, interaction=kernels[i])
            gens.append(K)

    # ZERO MOMENTUM IS THE VACUUM CASE, identically: zeta_B(k=0) sums the
    # terminal position with phase 1, which is what the vacuum branch
    # computes.  Recognising it here is not a shortcut -- it is what frees
    # the pin below, because a held terminal is a kept index and a kept
    # index is what forbids re-pinning.  Verified bit-identical against
    # the single-k branch on the corpus block that motivated this.  An
    # EXACT zero only (``_momentum_is_zero``), and the block-cache key
    # collapses exactly the same set.
    vacuum = (terminal is None or int(terminal) == int(source)
              or _momentum_is_zero(momentum))
    source = (_planner_pin(edges, nu_vec, int(source))
              if vacuum else int(source))
    term = int(source) if vacuum else int(terminal)

    # Leaf names for the bridge cache: an edge is fully identified by
    # its exponent, and the truncation by (grid, dimension, lattice).
    # The SP collapse is then memoised by canonical SP-tree, which is
    # what makes it cheap across a corpus where the same bridges recur.
    #
    # A general kernel is named by its OWN fingerprint, never by its
    # tail: ``("E", nu)`` for a mixed kernel of tail ``nu`` would make
    # the parallel-merge and series keys collide with the float's, and
    # the cache would serve the float's bridge (measured: a sign-flipped
    # 2.0 relative error on a theta graph).
    if kernels is None:
        leaf_keys = [("E", float(v)) for v in nu_vec]
    else:
        leaf_keys = [("I", kern.key()) for kern in kernels]
    trunc_key = (int(n_sp), int(d), A.tobytes())
    nodes, r_edges, r_kernels, scalar = _sp_reduce(
        edges, gens, int(source), term, n_sp, d,
        leaf_keys=leaf_keys, trunc_key=trunc_key,
    )

    if sp_n_points is not None or n_core != n:
        # Restrict AFTER the whole reduction, with every kernel already
        # in the canonical min -> max orientation: restriction does not
        # commute with reversal at even n_points on a sheared cell (the
        # balanced axis holds n/2 without -n/2), so the operation needs
        # one fixed place in the pipeline.  _dense_core does its own
        # reversals downstream of here, on the coarse grid.
        #
        # One gather serves both knobs: whatever the SP part was
        # collapsed on (``n_sp``) lands directly on whatever the core is
        # contracted on (``n_core``), with no intermediate stop at
        # ``n_points``.  Chaining two gathers through n_points would be
        # value-identical but would pay a second pass and, at even n on a
        # sheared cell, would need its own orientation argument.
        fine = TorusTruncation(n_sp, d, A)
        coarse = TorusTruncation(n_core, d, A)
        r_kernels = [fine.restrict_to(K, coarse) for K in r_kernels]

    M = _dense_core(
        nodes, r_edges, r_kernels, int(source), term, n_core, d, A,
        max_core_bytes,
        peelable=not bool(np.isinf(nu_vec).any()),
    ) * scalar

    if vacuum:
        return float(np.real(M))

    if momentum is None:
        # full BZ grid:  ζ_G(k) = Σ_{x_t} exp(-2πi k·x_t) M(x_t) = fftn(M)
        #
        # With a coarse core, ``M`` lives on ``n_core`` labels but the
        # caller asked for the value on the ``n_points`` grid.  Embedding
        # first is exact for the approximant, not an interpolation: ζ_B(k)
        # from a coarse core IS a trigonometric polynomial with those
        # coefficients, and the padded transform evaluates that same
        # polynomial at the fine nodes.  Verified against hybrid's own
        # single-k branch below — which reads the polynomial off the
        # coarse array directly — to 5.3e-16 over the full grid.
        if n_core != n:
            M = TorusTruncation(n_core, d, A).embed_into(
                M, TorusTruncation(n, d, A),
            )
        return np.real(np.fft.fftn(M)).astype(float)

    # single k:  ζ_G(k) = Σ_{x_t} exp(-2πi k·x_t) M(x_t)
    #
    # The flat index of ``M`` is a position on the **balanced** z-axis
    # (``_balanced_z_axis``: negative labels sit on the upper half of the
    # index range), the same layout ``_edge_kernel_torus`` builds — not an
    # unsigned 0..n-1 coordinate.  The phase must therefore use the signed
    # labels: an unsigned index puts ``x_t = -m`` at ``n - m``, which costs
    # a spurious factor ``exp(-2πi k n)``.  That factor is exactly 1 only
    # at on-grid ``k = j / n_points``, so unsigned indices happen to agree
    # there and alias badly in between — the returned value then depends on
    # where k sits between grid points rather than on k, and does not
    # converge with ``n_points``.  The grid branch above is immune because
    # ``fftn`` evaluates only at the on-grid nodes.
    #
    # ζ_G is real by lattice inversion symmetry, so the sum is taken with a
    # real cos weight, matching ``graph_zeta_general``'s single-k branch.
    # ``n_core``, not ``n``: the labels are the ones ``M`` actually
    # carries.  With a coarse core this branch needs no embedding at all
    # — summing cos(2π k·z) against the coarse array already evaluates
    # the trigonometric polynomial at an arbitrary k, on or off the fine
    # grid.  Using ``n`` here would index a longer axis than M has.
    mom = np.asarray(momentum, dtype=float).reshape(d)
    z = _balanced_z_axis(n_core).astype(float)
    grids = np.meshgrid(*[z] * d, indexing="ij")
    k_dot_z = sum(mom[c] * grids[c] for c in range(d))
    return float(np.sum(np.cos(2.0 * np.pi * k_dot_z) * np.real(M)))
