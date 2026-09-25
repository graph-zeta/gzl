# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Public front-end: :func:`evaluate_graph` evaluates ζ_G at zero external
momentum via the topology-first router.

The router Hadamard-merges parallel edges, block-cut decomposes the
resulting simple graph, and picks the cheapest correct evaluator for
each biconnected block.  Let σ ≡ ν − d, let tw be the block's
treewidth (min-fill-in heuristic), and let |external| count the
block's cut vertices in the full graph plus any source / terminal
vertices that lie inside it:

* **Bridge** — exactly two vertices and one bundled edge of exponent
  ``ν_B``: closed-form ``epstein_zeta(ν_B, A, 0, 0)``, the
  ``d``-dimensional Epstein zeta of lattice ``A`` at zero
  displacement and zero momentum.
* **Simple cycle** — ``n_v ≥ 3``, ``n_e = n_v``: closed-form
  :func:`gzl.zeta_circle`.  Block-cut decomposition only ever
  joins blocks at single cut vertices (never along edges), so the
  closed-form scalar is correct at zero external momentum
  irrespective of how many cut vertices the cycle has.
* **σ_max=4 algebra** — ``tw == 2`` and ``σ < 1.49`` and
  ``|external| ≤ 2`` and the block is SP-reducible at the chosen
  terminal pair: :func:`gzl.graph_from_edges` followed by
  :func:`gzl.graph_zero` on the block's induced subgraph.
  Each Hadamard-merged bundle is kept analytically as an Epstein
  zeta of exponent ``Σ_e ν_e``; only cross-bundle algebraic
  operations touch the ``n_points^d`` grid.
* **Tensor (bucket-elimination)** — every other block:
  :func:`gzl.graph_zeta_general` on the block's induced
  subgraph at ``σ_max = 0``, with optional 3-point Richardson
  extrapolation.

The whole-graph value is the product of the per-block values, since
at zero external momentum every cut vertex carries zero momentum
too.

ζ_G is real at zero source position and any external momentum k by
the lattice's inversion symmetry; the imaginary residue from the
FFT path is discarded at every block evaluator and
:func:`evaluate_graph` returns a :class:`float`.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict

import numpy as np

from gzl import _elimination
from gzl._labels import edge_array, vertex_label
from gzl._lattices import lattice_matrix

# networkx and epsteinlib are dependencies of gzl, and the router cannot
# run without them.  The import is guarded all the same, so that a broken
# installation meets TopologyEvaluatorUnavailableError, chained to the
# ImportError kept here, when the router is called rather than at
# `import gzl`.  Only networkx can be missing at this point: gzl.core
# imports epsteinlib unguarded, and `import gzl` fails first without it.
try:
    import networkx as _nx
    from epsteinlib import epstein_zeta as _epstein_zeta
    _TOPO_AVAILABLE = True
    _TOPO_IMPORT_ERROR = None
except ImportError as _exc:                          # pragma: no cover
    _TOPO_AVAILABLE = False
    _TOPO_IMPORT_ERROR = _exc

from gzl.circle import zeta_circle, CycleQuadratureError, _CLOSED_FORM_DIMS
from gzl.interaction import (
    InteractionSupportError,
    coerce_nu,
    is_interaction,
)
from gzl.construction import (
    NotTreewidthTwoError,
    graph_from_edges,
)
from gzl.core import (
    _epstein_zeta_k,
    graph_sample,
    graph_sample_at,
    graph_zero,
    graph_zero_conditioned,
)
from gzl.direct_sum import (  # noqa: F401
    _min_free_cut_nu,
    direct_sum_extrapolated,
    _direct_sum_extrapolated_grid,
)
from gzl._elimination import sp_reduced_cut_nu as _sp_reduced_cut_nu
from gzl.tensor_network import (
    _balanced_z_axis,
    graph_zeta_general,
    graph_zeta_general_at_zero,
)
from gzl.slab import (
    slab_zeta as _slab_zeta,
    lattice_window_group as _slab_window_group,
    choose_slab as _choose_slab,
    slab_schedule as _slab_schedule,
    _slab_zeta_finite_k,
    _slab_zeta_finite_k_outer,
    _fk_outer_best,
)
from gzl.hybrid import (
    hybrid_zeta,
    HybridCoreBudgetError,
    HybridCoreTooLargeError,
    _momentum_is_zero,
    _planner_pin,
)


# Engines that evaluate a general (finite-ν) on-spine / vacuum block.
# "hybrid" (default) collapses series/parallel structure by FFT and
# contracts only the irreducible core, reading the BZ grid with a single
# fftn — equal to round-off to "tensor" (graph_zeta_general), typically far
# faster.  It falls back to the tensor for ν = ∞ (the NN-indicator path),
# for ≥ 2 free terminals (not yet supported), and on any MemoryError —
# HybridCoreTooLargeError (budget guards, einsum-letters overflow on
# pathological high-treewidth cores) or a genuine allocation failure.
_BLOCK_ENGINES = ("hybrid", "tensor")


#: Number of times `_block_general` dropped a requested split/graded
#: core because it had to degrade to the tensor engine.  The tensor takes
#: neither `sp_n_points` nor `core_n_points`, so such a block is evaluated
#: on `n_points` instead of the fine grid the caller asked for -- measured
#: 5.4e-03 relative on a subdivided K4.  Surfaced as
#: `n_block_split_dropped` so the diagnostics cannot go on reporting a
#: split that did not happen.  Same shape as
#: `direct_sum._LAST_GATE_DECLINED`, which `test_tau_d_acceptance` reads.
_SPLIT_DROPPED = 0


def _block_general(
    local_edges, nu_vec, A, n_points, *,
    source, terminal, mode, momentum=None, engine="hybrid",
    sp_n_points=None, core_n_points=None, kernels=None,
):
    r"""Dispatch a single-terminal general-ν block evaluation to the
    hybrid engine (default) or the tensor engine.

    ``mode`` is ``"grid"`` (full BZ grid → ``(n,)^d``), ``"single"``
    (one ``momentum`` → scalar), or ``"vacuum"`` (k = 0 → scalar).
    Returns the raw engine output; callers apply their own
    ``reshape`` / ``.real`` exactly as for ``graph_zeta_general``.

    ``sp_n_points`` turns on hybrid's split-resolution SP collapse: the
    series/parallel part is collapsed on a torus of ``sp_n_points`` and
    only the irreducible core is contracted on ``n_points``.  It is the
    caller's job to hand over an already-resolved value — see
    ``_resolve_dense_routing``.  It is silently dropped on the tensor
    fallback, which has no such mode; that is a rate loss, not a
    correctness one, and only reachable when hybrid has already refused.

    ``core_n_points`` is the downward half of the same axis: the
    irreducible core is contracted on a grid COARSER than ``n_points``,
    sized per block by :func:`_core_n_for_block`, and the grid readout
    comes back up by an exact band-limited embedding.  Same contract —
    already resolved by the caller, and silently dropped on the tensor
    fallback for the same reason.
    """
    nu_arr = np.asarray(nu_vec, dtype=float)
    # The kernels channel (general interactions): passed only when
    # present, so the legacy call forms below are literally unchanged.
    kern_kw = {} if kernels is None else {"kernels": list(kernels)}
    if engine == "hybrid" and not np.any(np.isinf(nu_arr)):
        # Not a ValueError-catching site on purpose.  hybrid raises plain
        # ValueError for a coarser-than-core SP grid, for ν = inf and for
        # self-loops; those are caller bugs, and swallowing them here
        # would turn a wrong call into a silently different engine.  The
        # resolver upstream is what guarantees they cannot happen.
        sp_kw = {} if sp_n_points is None else {"sp_n_points": int(sp_n_points)}
        if core_n_points is not None:
            sp_kw["core_n_points"] = int(core_n_points)
        # Bound the core contraction.  hybrid's own cap defaults to None,
        # so before this it was OFF on every routed call and the only
        # thing standing between a mis-sized core and the machine was a
        # genuine MemoryError.  The ceiling is priced from the symbolic
        # schedule, so it refuses before the tables are allocated; a
        # refusal is a HybridCoreTooLargeError, which is re-raised for
        # the box fallbacks below to catch -- NOT downgraded to the
        # uncapped tensor.
        sp_kw["max_core_bytes"] = _DENSE_CORE_MAX_BYTES
        try:
            if mode == "grid":
                return hybrid_zeta(
                    local_edges, nu_vec, A, n_points,
                    source=source, terminal=terminal, **sp_kw, **kern_kw,
                )
            if mode == "single":
                return hybrid_zeta(
                    local_edges, nu_vec, A, n_points,
                    source=source, terminal=terminal, momentum=momentum,
                    **sp_kw, **kern_kw,
                )
            return hybrid_zeta(
                local_edges, nu_vec, A, n_points, **sp_kw, **kern_kw,   # vacuum
            )
        except HybridCoreBudgetError:
            # A BYTE-BUDGET refusal is not an allocation failure, and
            # must not be downgraded into one.
            #
            # Falling through to graph_zeta_general hands the SAME block
            # at the SAME n_points to an engine with no byte cap and an
            # unchunked peel.  At d = 3 that turns a refused 2 GB core
            # into a 512 GiB ``np.ones`` on a shipped-corpus block (K5,
            # first at order 10), and numpy does NOT raise for it -- a
            # 1 TB ``np.empty`` succeeds here -- so the MemoryError nets
            # below never arm and the OS ends the process instead.
            #
            # Re-raise.  ``HybridCoreBudgetError`` is a MemoryError
            # subclass, so the box fallbacks in `_block_at_finite_k` and
            # `_evaluate_via_topology` catch it for blocks the router
            # diverted to the torus -- which is what the comments here
            # always claimed happened.  For any other block it surfaces,
            # which is correct: the caller asked for a bounded core and
            # cannot have one.
            #
            # STRUCTURAL refusals (letters / bucket axes) are the plain
            # parent class and still fall through below: there the tensor
            # really does handle more vertices than hybrid, so it is the
            # right destination.  That distinction is what keeps the
            # nu <= d circulant in tests/test_hybrid.py working.
            raise
        except MemoryError:
            # A genuine allocation failure, or a STRUCTURAL refusal
            # (einsum letters / bucket axes) -- fall through to the
            # real-space tensor engine instead of leaking the exception.
            #
            # NOTE the tensor has no split and no graded core: it takes
            # neither `sp_n_points` nor `core_n_points`.  So a block that
            # lands here is evaluated with its SP part collapsed on
            # `n_points` rather than the fine grid the caller asked for
            # -- measured 5.4e-03 relative on a subdivided K4 at d = 1,
            # nu = 1.5, n_points = 64, sp_n_points = 4096.  The caller's
            # diagnostics must not go on claiming the split was applied;
            # `_SPLIT_DROPPED` is how the counters are told.
            global _SPLIT_DROPPED
            if sp_n_points is not None or core_n_points is not None:
                _SPLIT_DROPPED += 1
            pass

    if mode == "grid":
        return graph_zeta_general(
            local_edges, nu_vec, A, n_points,
            source=source, terminals=(terminal,), space="k", **kern_kw,
        )
    if mode == "single":
        return graph_zeta_general(
            local_edges, nu_vec, A, n_points,
            source=source, terminals=(terminal,), momentum=momentum,
            **kern_kw,
        )
    return graph_zeta_general_at_zero(
        local_edges, nu_vec, A, n_points, pinned_vertex=source, **kern_kw,
    )


# Small, d-aware box-width ladders for the dense-block direct-sum route
# (3b).  Dense (tw≥3) blocks have a steep ``|x|^-(deg·ν)`` boundary
# tail, so a tiny L already reaches series precision after the
# degree-aware Richardson extrapolation in ``direct_sum_extrapolated``.
# Memory of an intermediate is ``(2L+1)^(d·tw)``, so the ladder
# shrinks with d; at d=3 the dense direct sum is only feasible for the
# small cores at small L (documented caveat — falls back to tensor on
# MemoryError).
#
# Lattice-anisotropy note.  The truncation box is ``{-L..L}^d`` in
# *integer* lattice coordinates; the kernel uses the *physical*
# distance ``‖A·n‖``.  For a skewed ``A`` the box is a sheared
# parallelepiped, so the effective physical truncation radius along
# the thin direction is ``~ L·σ_min(A)``.  The convergence *rate* is
# unchanged (it is set by the graph vertex degree, not the lattice —
# higher lattice coordination number does not matter), only the
# constant.  This ladder is therefore deliberately keyed on ``d``
# only, not on ``A``: for every standard lattice
# (chain/square/triangular/hexagonal/cubic, cond(A) ≲ 1.7) the
# measured K₅ accuracy is ~1e-10 at these L — far inside the
# corpus/MC tolerance — and it degrades only gracefully with cond(A)
# (~7e-7 even at the pathological cond≈4.4), with the tensor path as
# the safety net beyond that.  Anisotropy-aware L-scaling
# (``L ∝ 1/σ_min(A)``) is intentionally NOT added: it would inflate
# the ``(2L+1)^(d·tw)`` cost to "fix" a regime no physical lattice in
# scope reaches.  Revisit only if a strongly-sheared lattice becomes
# a real target.
#
# The d = 1 ladder was WIDENED from (4,5,6,7,8).  Measured at the shipped
# 1qp config (nu = 3.0, n_points = 64), L2 distance of the whole BZ
# coefficient to the converged value, and the wall cost of the pass:
#
#     L = 4..8     O9 6.69e-04   O10 6.30e-03   O11 3.93e-02   10.0 s
#     L = 10..14   O9 4.08e-05   O10 3.67e-04   O11 2.06e-03   10.2 s
#     L = 20..24   O9 2.87e-06   O10 2.52e-05   O11 1.36e-04   10.9 s
#     L = 40..44   O9 1.43e-07   O10 1.23e-06   O11 6.49e-06   13.4 s
#
# i.e. the shipped ladder was leaving the order-11 1qp coefficient ~4%
# wrong, and reach was the whole cause: the box converges monotonically
# onto the independently-computed torus value, ~15x per reach doubling.
# 20..24 buys three orders of magnitude for 9% more wall time and is the
# knee; 40..44 buys one more order for 34%.
#
# This matters even though tw >= 3 blocks now route to the torus by
# default at every d: the box remains the ladder for nu <= d
# (where it is the only engine that refuses a divergent sum rather than
# returning a confident wrong number), and for the MemoryError fallback.
# A route that only runs in the hard cases should not be the least
# accurate one.
_DENSE_DSUM_L_LIST = {
    1: (20, 21, 22, 23, 24),
    2: (3, 4, 5, 6),
    3: (2, 3, 4),
}

# --------------------------------------------------------------------------
# Dense (tw ≥ 3) block routing.
#
# ``"direct_sum"`` is the historical route: the box ladder above.
# ``"torus"`` skips it and lets the block fall through to the ordinary
# tensor/hybrid path, which at k = 0 also carries the σ_eff Richardson
# ladder and, when ``sp_n_points`` is set, hybrid's split-resolution SP
# collapse.
#
# Measured per arm and per dimension, median
# gain over the box route against a two-family reference whose gap is
# reported as a floor:
#
#             d = 1                          d = 2 (n_points = 16)
#   k = 0     routing alone   2.8x           routing alone  0.0x  (a LOSS)
#             + ladder     1223x             + ladder       3.0x
#   fin. k    routing alone  21x (s=0.5)     not measured
#                        1577x (s=2.0)
#
# At k = 0 the lever is the sigma_eff Richardson ladder, not reach and
# not the split: it lives on this module's k = 0 path and a dense block
# has never been able to reach it.  At finite k there is no ladder --- the
# tail carries a cos(2 pi k.x) factor and stops being a power law --- and
# the split is the only lever there.
#
# d = 2 is included on a thinner margin than d = 1: at the shipped
# n_points = 16 the box's L = 6 ladder is on equal terms with a
# half-width-8 torus, so routing alone loses and the ladder recovers only
# 3x.  It reaches 84x at n = 24 and below the reference floor at n = 32,
# so the d = 2 gain is a resolution question that this table does not
# settle.
#
# d = 3 now routes to the torus too.  It stayed on the box because an
# earlier measurement found direct_sum more accurate on 16/16 d = 3 tw = 3
# blocks and because no d = 3 config shipped -- both of which have changed.
#
# Measured on the cubic 1qp k = 0 pass (n = 8, orders <= 9), each arm
# from a COLD cache and with the torus arm run first so the ordering
# cannot flatter it:
#
#     route        wall     peak RSS
#     direct_sum   54.2 s    3.44 GB
#     torus         5.5 s    0.90 GB     9.9x faster, 3.8x leaner
#
# and the two agree to 1.35e-09 at order 9, with BOTH 9/9 orders inside
# the Fey cubic MC noise.  It scales: n = 12 costs 8.5 s / 1.10 GB and
# n = 16 costs 29.5 s / 1.56 GB, bounded by the byte ceiling rather than
# by luck.
#
# HISTORICAL, and no longer describes what ships: when this route first
# shipped, the split and the graded core stayed OFF at d = 3 because
# `_SPLIT_SP_N_POINTS` and `_CORE_KAPPA` had no d = 3 entry, so it
# changed the ROUTE only.  Both entries have since been measured and
# shipped -- see `_SPLIT_SP_N_POINTS` and `_CORE_KAPPA` themselves --
# and the paragraph is kept because the measurement above was taken in
# that configuration and is not a measurement of today's.
#
# The box stays the default for any d missing from the table below
# (d >= 4), takes a declined dense block the slab cannot, and is the
# MemoryError fallback.
_DENSE_ENGINES = ("direct_sum", "torus")

_DENSE_ENGINE_BY_D = {1: "torus", 2: "torus", 3: "torus"}

# --------------------------------------------------------------------------
# The ACCURACY gate on the torus route, beside the byte-budget gate.
#
# ``hybrid``'s ``max_core_bytes`` answers "does this block FIT on the
# torus".  It does not answer "is this block BETTER there", and the two
# are not the same question: a dense block that fits only at a small
# ``n`` can fit and still be worse than the real-space box it displaces.
# Measured at d = 3 against a slab reference (the same lattice sum with
# a second pin, which reaches n = 16 where the full contraction needs
# 550 GB), the two exponent-3 cores a byte-budget change moves are worse
# on an n = 8 torus than in the box by 5.0x (K5) and 10.6x (V6E11).  This
# gate is the second question.
#
# THE KEY IS THE PLAN EXPONENT, NEVER THE TREEWIDTH.  A census of 3684
# random dense shapes splits 333 as (tw 3, exponent 2), 21 as
# (tw 4, exponent 3) and 7 as (tw 3, exponent 3): treewidth does not
# determine the cost class, and a rule keyed on it prices the seven
# (tw 3, exponent 3) shapes as if they were the 333.
#
# ``_TORUS_MIN_N_BY_CLASS`` maps ``(d, post-peel plan exponent)`` to the
# smallest torus grid at which the block is at least as accurate as the
# box.  A class with no entry is UNCONSTRAINED and keeps today's
# routing, so this table can only send a block back to the engine it
# used before any budget change — it is monotone toward the incumbent by
# construction and cannot regress a value that ships today.
#
# HOW MUCH OF THE CORPUS THIS GOVERNS: 1.44%.  Of the 2636 dense-block
# occurrences in the two TFIM corpora, 2598 are plan exponent 2 and only
# 38 are exponent 3 (K5 itself occurs 5 times, V6E11 7).  The entry is
# still the right call for those 38 -- an exponent-3 core is the one that
# costs n**9 and the one a byte-budget change moves -- but it is not a
# statement about d = 3 dense routing at large.  The exponent-2 class has
# its own unanswered question; see _CORE_KAPPA's absence at d = 3.
#
# THE ONE ENTRY.  Relative error against the slab reference, d = 3,
# nu = 3.5, box at the router's own L = (2, 3, 4).  "declined" is the
# shipped ladder refused by its own guards, which is the right call
# everywhere it happens here:
#
#         n      K5 raw    K5 ladder |   V6E11 raw   V6E11 ladder
#         8    3.08e-07     declined |    1.44e-03       declined
#        10    6.49e-08     declined |    6.31e-04       declined
#        12    1.45e-08     declined |    3.12e-04       1.64e-05
#        14    3.76e-09     declined |    1.71e-04       3.60e-06
#        16    1.54e-09     declined |    1.01e-04       5.72e-07
#       box    6.20e-08              |    1.37e-04
#
# (Reference band 1.3e-09 on K5 and 2.0e-06 on V6E11, from the spread of
# the two parity families' own ladders; K5's n >= 14 rows sit inside it
# and are not resolved, which does not affect the threshold.)
#
# n = 12 is the first grid at which BOTH clear the box — K5 on its raw
# value (4.3x), V6E11 on the ladder its guards accept there (8.3x).  At
# n = 10 the best either core reaches, raw or extrapolated, is 0.95x
# (K5) and 1.05x (V6E11, and its guards decline that one), so n = 10 is
# a wash; at n = 8 both are far worse.  Note V6E11's n = 12 entry needs
# ``richardson=True``: its RAW value there is still 2.3x worse than the
# box, which is why the ladder is part of the gate rather than a bonus
# on top of it.
#
# AND IT DOES NOT MOVE WITH nu -- now MEASURED across the whole
# production range on BOTH emblems, not extrapolated from one.  The second
# pin makes n = 17, 18 reference rungs affordable, tightening the
# per-parity Richardson references by 12x - 1000x, and every cell below
# is stated against those bands; a cell within 2 band-widths does not
# adjudicate.  First grid at which the block (raw value, or the ladder
# its own guards accept) clears the box, per coupling:
#
#      nu        3.25   3.50   4.00   5.00
#      K5          12     12     12     10     (raw clears)
#      V6E11       12     12     12     12     (accepted ladder clears)
#
#   * K5's row is raw-borne: n = 12 clears the box 3.7x / 4.7x / 6.2x /
#     22.5x, every cell resolved; at nu = 5 even n = 10 clears (1.7x),
#     so 12 is conservative by one even rung at the very top.
#   * V6E11's row is LADDER-borne at every nu: its raw value does not
#     clear the box until n = 13..16 (only 1.04x at n = 16, nu = 3.25),
#     while the guard-accepted ladder at n = 12 clears 7.9x / 8.8x /
#     11.1x / 18.7x, resolved.  The ladder is part of the gate, not a
#     bonus on top of it -- and since the sigma_eff ladder is
#     deliberately absent at finite k, a finite-k torus arm for this
#     class must key on RAW-clearing grids or measure its own
#     extrapolation; n = 12 raw is NOT enough there.
#   * Two of this comment's original claims are hereby retired by
#     measurement: "n = 10 first clears it at nu = 4" (a lead from two
#     marginal cells) is REFUTED -- n = 10 is 0.85x there, solidly
#     resolved -- and "conservative by one rung at the top" understated
#     the entry: 12 is exactly right at nu = 3.25 .. 4.00 and
#     conservative only at nu = 5.
#
# WHAT IT COSTS: free at n_points >= 10, where the byte budget already
# refuses this class at d = 3 (11.6 GiB at n = 10, 768 GiB at n = 16)
# and the block reaches the box either way.  NOT free at n_points <= 8:
# the gate closes the window the budget would admit, and the box costs
# ~1 - 3 min per distinct exponent-3 shape -- measured 163.8 s for the
# single such block in the order-9 cubic 1qp pass at n = 8, which is
# 166.9 of that pass's 177.8 s.  That cost is the price of not
# shipping a 5 - 58x-worse value at the pass grid; the slab arm below
# (``_SLAB_N_BY_CLASS``) replaces the box with the second pin at the
# measured minimum grid on cells with enough symmetry.  What the entry
# buys is that the n <= 8 window stays shut when the budget moves --
# ``_DENSE_CORE_MAX_BYTES`` is an environment variable, and raising it
# is exactly the change that would silently trade 5x of accuracy for
# wall clock.
_TORUS_MIN_N_BY_CLASS = {(3, 3): 12}


def _dense_torus_min_n(edges, vertices, pin, keep, d):
    r"""Minimum torus grid for this block's cost class, or ``None``.

    ``None`` means "no measurement for this class", which is read as
    unconstrained.  The pin and ``keep`` set must be the ones the core
    is actually contracted at -- a block classified at one pin and
    contracted at another is classified wrong, the same desync
    :func:`_core_n_capped` is careful about.
    """
    try:
        exponent = _elimination.plan(
            edges, vertices, pin=pin, keep=tuple(keep)).exponent
    except Exception:
        return None
    return _TORUS_MIN_N_BY_CLASS.get((int(d), int(exponent)))


# ---------------------------------------------------------------------
# The slab arm, behind the accuracy gate's decline
# ---------------------------------------------------------------------
#
# A DECLINED block is one the gate says needs n >= _TORUS_MIN_N_BY_CLASS
# to beat the box, and which cannot afford that grid on the dense core.
# Before the slab existed there was nothing to do but pay the box.  The
# slab (:mod:`gzl.slab`) is the SAME torus truncation associated
# with a second pin, which drops the inner post-peel exponent by one and
# therefore CAN afford the grid the gate asks for.
#
# KEYED ON THE FREE-PIN EXPONENT, WHICH IS NOT THE GATE'S KEY.  The gate
# classifies at the router's root because it is asking about the graded
# torus CORE, which is pinned where it is pinned.  This arm asks a
# different question -- slab or box -- and both of those engines track
# the free-pin class: the slab's inner exponent is one below it BY
# CONSTRUCTION, and the box's cost tracks it EMPIRICALLY (below).
#
# What is NOT true, and was asserted here until it was checked: that
# `direct_sum._pick_root` "independently finds the same pin".  It agrees
# with the planner's free pin on 1680 of 2636 dense-block occurrences --
# 63.7% -- and differs on 3 of the 13 shapes the cost separation below
# was measured on.  The separation holds on all 13 anyway, including
# those 3, so the routing key stands; the mechanism offered for it did
# not, and is not needed.
#
# WHAT THE ENTRY IS MEASURED ON.  A census of the shipped TFIM corpora
# (0qp + 1qp) finds 2636 DENSE (treewidth >= 3) block occurrences --
# not 36 866, which is every biconnected block with >= 3 vertices and is
# 93% cycles.  Of those, 38 occurrences in 30 distinct shapes are
# free-pin exponent 3, and routing every one of the 30 through
# `evaluate_graph` at d = 3 shows what actually reaches this arm:
#
#     13 occurrences in  6 distinct shapes  -> the slab   (0.49% of dense)
#     25 occurrences in 24 distinct shapes  -> the torus  (the split acts)
#
# THE ENTRY IS MEASURED ON 5 OF THOSE 6 SHAPES.  The sixth is a V6E11
# shape that reaches the arm and was never measured; it inherits the
# class entry, which is what a class-keyed rule is for, but it is not
# covered evidence and saying "all of them were measured" would be
# false.
#
# THE FREE-PIN EXPONENT SEPARATES BOX COST EXACTLY, on the 13 shapes
# where box wall-clock was measured:
#
#   free-pin exp 3 (5 shapes)      box  77 - 178 s
#   free-pin exp 2 (8 shapes)      box 0.2 - 1.7 s
#
# with no overlap and a 45x gap.  Nothing else tried does: not
# treewidth, V, E, min/max degree, triangle count, connectivity or
# clique number, and not the plan at a FIXED pin -- two of these shapes
# have byte-identical fixed-pin plans and 565x different box wall-clock.
#
# ON ACCURACY THE TWO CLASSES OVERLAP, and that is why the rule is
# stated on cost.  Strict resolved bounds for slab@12 against the box:
#
#   exp 3    1.61x  4.76x  5.47x  9.43x  30.38x     -- wins on all five
#   exp 2    0.46x  0.72x  0.91x  1.25x  1.30x
#            1.54x  1.70x  4.94x                    -- loses on three
#
# So the (3, 3) entry ships and there is deliberately no (3, 2) entry.
# At exponent 3 the slab is better on BOTH axes on every shape measured
# (3.2x - 7.4x faster through the router, 1.61x - 30.38x more accurate).
# At exponent 2 it is 15-100x SLOWER for a verdict that is worse on
# three of eight -- a large certain cost for an uncertain gain.  A
# cost-blind slab route would have taken all 13; the first calibration
# attempted here was exactly that, and the measurement refuted it.
#
# WHY THE ARM MUST STAY BEHIND THE DECLINE.  V6E11 is free-pin exponent
# 3 and its slab at n = 12 is 0.234x the box -- strict resolved bound,
# band 1.235e-05, box 5.9 band-widths out and slab@12 20.2, so the LOSS
# is resolved and not a band artefact.  (Two earlier point estimates of
# this same quantity, 0.29x and 0.44x, were quoted against different
# references; 0.234x is the band-resolved one.)  It never reaches here,
# because the SP split acts on it and the gate therefore admits it to
# the torus.  That exclusion is load-bearing, not incidental -- and the
# census above says the same thing at scale: 24 of the 30 free-pin
# exponent-3 shapes are diverted the same way.
_SLAB_N_BY_CLASS = {(3, 3): 12}

#: Smallest orbit group the slab arm will run on, per dimension.  The
#: d = 3 entry is 16 -- tetragonal or better -- because the measured
#: 1.61x-30.38x speed-and-accuracy win rests on a 48-fold cubic
#: reduction.  Priced, not estimated: at n = 12 the cubic group folds
#: 1728 outer cells onto 84 and an orthorhombic one (|G| = 8) onto 343,
#: so such a cell costs 343/84 = 4.08x the measured wall-clock for the
#: same value.  Not a correctness condition: see `_slab_grid`.
_SLAB_MIN_GROUP = {3: 16}

# ---------------------------------------------------------------------
# The PREFERRED slab route: the split-no-op (3, 2) sub-class rides the
# second pin
# ---------------------------------------------------------------------
#
# Corpus census (0qp + 1qp), the free-pin exponent-2 dense class, 2598
# occurrences in 1296 shapes, keyed on the two symbolic prices this
# route reads:
#
#     slab inner exp 1, split NO-OP :   463 occ (17.8%),  113 shapes
#     slab inner exp 1, split ACTS  :  2127 occ (81.9%), 1175 shapes
#     slab inner exp 2              :     8 occ ( 0.3%),    8 shapes
#
# ONLY THE FIRST ROW IS TAKEN.  It is the sub-class that carried the
# shipped defect: with no split to relocate its escape, a 3-connected
# exponent-2 block ran the torus at the pass grid, where n_points = 8 is
# 3.6x - 58x worse than the box (K4 the emblem).  For these shapes the
# second pin makes the inner contraction O(n^d), so n = 18 costs LESS
# than the box (measured on the row's three census-core members: K4
# 0.3 s, prism 0.8 s, K33 0.9 s, against boxes of 0.6 - 1.7 s) and,
# band-resolved against slab n = 20 references at nu = 3.5 on the
# cubic cell, beats it on accuracy on every resolved shape; the box is
# 5.6 - 230 reference bands out.
#
# THE SPLIT-ACTING ROWS ARE EXCLUDED, AND THE FIRST DRAFT OF THIS ROUTE
# DID NOT EXCLUDE THEM -- the full suite caught it.  A split-acting
# block's incumbent route already relocates its slow escape onto the
# fine grid, and that is measured far better than a raw n = 18
# truncation: K4SUB runs at 1.94e-08 from a resolved reference through
# the split (2442x the box, see _SPLIT_SP_N_POINTS) while its raw slab
# at n = 18 is 4.3e-05 out -- the degree-2 escape that makes the split
# act is the same mode that makes the raw truncation slow.  Hijacking
# those blocks would have been a ~2000x accuracy regression delivered
# by the change that claimed to fix accuracy.  Same lesson as the
# class's gate history: the bare-core table does not transfer to
# split-acting blocks.
#
# TWO RUNGS, NOT ONE, and the pair is the safety: the block is
# evaluated at both entries of the tuple and shipped at the top rung
# ONLY if the two agree to `_SLAB_SELF_BAND_MAX` relative -- a
# per-block self-consistency band that needs no shape census.  A
# disagreement means this block's truncation has not settled by n = 18
# (nothing measured does this; the tripwire exists for the shape nobody
# measured) and the block falls back to the box, the one engine that is
# never confidently wrong here.  The inner-exponent-2 remainder (8
# occurrences, 8 shapes) is MEASURED and needs no entry: through the
# router, at both pass grids n = 8 and 16, every one of the 8 shapes
# sits 0.5 - 0.7 band-widths from its pair-extrapolated slab reference
# while the box is 1.1 - 1.9 band-widths out on 7 of them (the eighth
# is statistically tied at 0.3 vs 0.6), worst case over both candidate
# basis exponents.  The incumbent route is at least as accurate as the
# engine an entry would send it to, so an entry could only regress it;
# deliberately none, measured rather than pending.
_SLAB_PREFERRED_RUNGS = {(3, 2): (16, 18)}
#: Finite-k pair rungs BY CLASS (first entry whose p-threshold the
#: block clears wins) and the per-k correction tripwire (see the arm
#: in _block_at_finite_k).  The BALANCED policy: each class gets the
#: cheapest rung pair that stays at least box-accurate, not the most
#: accurate pair we can license -- no single block may dominate a
#: pass.  Measured (512 pass k-points vs the n = 18 pinned reference;
#: costs are the seconds to compute the block's M field, all on one
#: machine):
#:
#:   p >= 10 class           med rel   worse-than-box  max dev/box  cost
#:    K5 emblem (0 -> 1 sector):
#:     pair (8, 10)  SHIPPED  4.5e-08      1 / 512         1.57x     7 s
#:     pair (10, 12)          2.9e-09      0 / 512         0.09x    29 s
#:     2term (6,8,10)         3.1e-08      1 / 512         1.04x     8 s
#:    K5-e corpus emblem, hop across the missing edge (0 -> 4,
#:    both-pins basis p = 11; reference band 1.6e-10):
#:     pair (8, 10)  SHIPPED  1.3e-07     46 / 512         1.86x     6 s
#:     pair (10, 12)          1.8e-08      0 / 512         0.40x    21 s
#:     box                    7.9e-07    (max rel 4.8e-05)          74 s
#:   The shipped pair is NOT pointwise-dominant over the box in the
#:   second sector: it is 6x better in median and 10x better in
#:   grid-max (4.5e-06 vs 4.8e-05), and cedes <= 1.86x at 9% of
#:   points -- the accepted BALANCED trade (4x cheaper than the
#:   pointwise-dominant (10, 12)).  (6, 8) is refused for the class:
#:   49x the box on the K5 emblem.  Two-term buys nothing at equal
#:   cost.
#:
#:   7 <= p < 10 class (K5-e with a deg-3 vertex free, p = 7.5)
#:     pair (10, 12) SHIPPED  8.0e-08      0 / 512         0.74x    28 s
#:     pair (8, 10)  refused  7.6e-07     69 / 512         3.73x     6 s
#:     2term (6,8,10) refused 9.9e-07     69 / 512         7.51x     6 s
#:   Every top-rung-10 scheme fails this class: its p = 7.5
#:   asymptotics are not clean below n = 12, so (10, 12) is the
#:   licensed floor, not gold-plating.
#:
#: The tripwire ceiling is set from the shipped pairs' measured per-k
#: corrections (max |pair - hi|/|value|: 5.0e-07 on K5 at (8, 10),
#: 9.1e-06 on K5-e at (10, 12)) with >= 110x headroom -- a correction
#: beyond it says the pair is outside its measured regime, and the
#: box keeps the block.
_FK_PAIR_BY_P = ((10.0, (8, 10)), (7.0, (10, 12)))
_FK_PAIR_BAND_MAX = 1e-3
_SLAB_SELF_BAND_MAX = 1e-3


def _slab_preferred(edges, nu_vec, d, A, nn_mode=False):
    r"""``(rungs, pin)`` for the preferred slab route, or ``None``.

    ``None`` keeps the incumbent route (torus ladder / box), so every
    refusal here is safe by construction.  Mirrors `_slab_grid`'s
    guards -- nn mode, non-finite or divergent nu, the orbit group --
    and adds the two that define this route: the free-pin plan
    exponent must key `_SLAB_PREFERRED_RUNGS`, and the slab inner
    exponent must be exactly 1, which is what makes the n = 18 grid
    cost less than the box.  Both are symbolic prices; nothing is
    contracted to decide.
    """
    if bool(nn_mode):
        return None
    nu_arr = np.asarray(nu_vec, dtype=float)
    if nu_arr.size == 0 or not np.all(np.isfinite(nu_arr)):
        return None
    if float(nu_arr.min()) <= float(d):
        return None
    try:
        pairs = [(int(u), int(v)) for u, v in np.asarray(edges)[:, :2].tolist()]
        verts = sorted({int(x) for x in np.asarray(edges)[:, :2].reshape(-1)})
        plan = _elimination.plan(pairs, verts)
    except Exception:
        return None
    rungs = _SLAB_PREFERRED_RUNGS.get((int(d), int(plan.exponent)))
    if rungs is None:
        return None
    try:
        sl = _choose_slab(pairs, int(d), source=int(plan.pin))
        inner = _slab_schedule(pairs, int(plan.pin), sl)[2]
    except Exception:
        return None
    if int(inner) != 1:
        return None
    # THE SPLIT MUST BE A STRUCTURAL NO-OP on this block.  Where it
    # acts, the incumbent route relocates the block's slow escape onto
    # the fine grid and is measured ~2000x better than a raw n = 18
    # truncation (K4SUB: 1.94e-08 through the split against 4.3e-05
    # raw) -- the degree-2 escape that makes the split act is the same
    # mode that makes the raw truncation slow.  Read at the planner
    # pin, like every other predicate on this path.
    try:
        emap: dict = {}
        for (u, v), w in zip(pairs, np.asarray(nu_vec, float).tolist()):
            k = (min(u, v), max(u, v))
            emap[k] = emap.get(k, 0.0) + float(w)
        sb = _min_free_cut_nu(emap, int(plan.pin)) - float(d)
        sc = _sp_reduced_cut_nu(emap, int(plan.pin)) - float(d)
    except Exception:
        return None
    if sc > sb + 1e-12:
        return None
    # Cost guard only -- the value is exact on every cell.  Measured on
    # the cubic group (48); an O(n^d) inner is cheap enough that the
    # orthorhombic 8 still prices under the box, and below that the
    # unreduced outer loop starts to matter.
    try:
        group = _slab_window_group(np.asarray(A, dtype=float), int(rungs[-1]))
    except Exception:
        return None
    if len(group) < 8:
        return None
    return rungs, int(plan.pin)



def _slab_grid(edges, nu_vec, d, A, nn_mode=False):
    r"""``(grid, pin)`` to run the slab at for this block, or ``None``.

    ``None`` means "not this block" and sends it on to the box, which is
    where a declined block went before this arm existed -- so every
    refusal here restores the incumbent route rather than inventing a
    third one.

    THE PIN IS THE PLANNER'S, AND NOTHING IS KEPT.  This arm serves the
    k = 0 dense route, where a block contributes the scalar ``zeta_B(0)``
    however many cut vertices it carries -- the box arm beside it calls
    ``direct_sum_extrapolated`` with no ``root`` and no ``terminal`` for
    exactly that reason, and lets ``_pick_root`` choose.  So the slab is
    classified and run the same way: free pin, empty ``keep``.  Reading
    the router's root and the block's externals instead would classify a
    block at a pin it is not contracted at, and on the census shapes that
    is the difference between exponent 3 and exponent 2 -- i.e. between
    this arm and the box.

    Four conditions, each of which has a way of being silently wrong:

    * **A measured entry for the cost class.**  Keyed on the plan
      exponent, never on treewidth: a census of 3684 random dense shapes
      splits 333 as (tw 3, exponent 2) and 7 as (tw 3, exponent 3), and
      a treewidth-keyed rule prices the seven as if they were the 333.
    * **Enough of an orbit reduction to be affordable.**  The slab's
      outer loop runs one cell per orbit of
      :func:`gzl.slab.lattice_window_group`, and that group is a
      property of the CELL: 48 on a cubic one, 16 tetragonal, 8
      orthorhombic, 2 or 1 on a sheared one.  The entry is measured on
      a cubic cell, so a cell offering an order-of-magnitude weaker
      reduction is outside it and is declined rather than silently
      costing 4.08x (orthorhombic) to 20.6x (no reduction at all) the
      measured wall-clock.  This is a COST condition only --
      :func:`gzl.slab.slab_zeta` is exact on every cell
      (``tests/test_slab.py`` checks it against the tensor engine on
      four cell types per dimension at d = 2, 3 and two at d = 1, at
      both parities of ``n``), and the group is derived by testing
      candidates against the kernel's own distance array rather than
      from a symmetry argument, because the symmetry argument is WRONG
      at even ``n`` on a cell with a cross term.
    * **min nu > d.**  A divergent lattice sum has no value for a torus
      engine to approximate, and the box is the only engine that refuses
      one instead of returning a confident wrong number.  The dense arm
      already raises ``DivergentLatticeSumError`` upstream of here; this
      is defence in depth and keeps the predicate independently
      testable.
    * **Not nearest-neighbour mode.**  At ``nu = inf`` the kernel is an
      integer-exact indicator, the truncation question does not arise,
      and no entry is measured there.
    """
    if bool(nn_mode):
        return None
    nu_arr = np.asarray(nu_vec, dtype=float)
    if nu_arr.size == 0 or not np.all(np.isfinite(nu_arr)):
        return None
    if float(nu_arr.min()) <= float(d):
        return None
    try:
        plan = _elimination.plan(
            [(int(u), int(v)) for u, v in np.asarray(edges)[:, :2].tolist()],
            sorted({int(x) for x in np.asarray(edges)[:, :2].reshape(-1)}),
        )
    except Exception:
        return None
    n_slab = _SLAB_N_BY_CLASS.get((int(d), int(plan.exponent)))
    if n_slab is None:
        return None
    # The group is n-dependent (it shrinks at even n on a cell with a
    # cross term), so it is asked at the grid the block would actually
    # run at -- not at a probe size that could report a group the run
    # does not get.
    try:
        group = _slab_window_group(np.asarray(A, dtype=float), int(n_slab))
    except Exception:
        return None
    if len(group) < _SLAB_MIN_GROUP.get(int(d), 1):
        return None
    return int(n_slab), int(plan.pin)


def _dense_torus_is_accurate(edges, vertices, pin, keep, d, n, ladder,
                             split_acts=False):
    r"""May this dense block go to the torus on ACCURACY grounds?

    ``ladder`` is whether the k = 0 sigma_eff Richardson ladder will be
    attempted on it.  It is part of the gate rather than a bonus: the
    measured n = 12 entry for ``(3, 3)`` clears the box on V6E11 only
    with the ladder running, so a caller with ``richardson=False`` -- and
    every finite-k caller, where the ladder is deliberately absent
    because the tail carries a ``cos(2 pi k.x)`` factor -- does not
    inherit it.

    ``split_acts`` says the split-resolution SP collapse is switched on
    AND has something to reduce on this block.  It releases the grid
    requirement outright, because the requirement is a statement about
    how long the block's OWN slow mode takes to die on the coarse grid,
    and an acting split removes that mode from the coarse grid entirely:
    what is left decays at the SP-reduced cut instead.

    WHAT THE SPLIT IS WORTH WHEN IT ACTS.  Quoted on the shape where it
    is RESOLVED rather than on the shape this table governs, because the
    two are not the same shape and saying so is the point.

    Cleanly measured, K4SUB at d = 3, n_points = 16, against a slab
    reference at resolution 4.96e-11: 2.033e-06 without the split and
    1.940e-08 with it, i.e. **23.3x the box becomes 2442x** — 40 988 and
    391 reference band-widths, so both cells are solid.

    On the class this table actually governs — plan exponent 3 — the
    evidence is weaker and that is a fact about the class, not about the
    mechanism.  Its only two exemplars are K5, on which the split is an
    exact no-op, and V6E11, whose reference band is 1.955e-06: there
    1.44e-03 without the split is 737 band-widths (so "10.6x WORSE than
    the box" is solid) but 3.6e-07 with it is **0.18 band-widths and not
    resolved**.  What is resolved there is a bound: from 10.6x worse than
    the box to **at least 57.7x better**.

    THAT BOUND IS ``(box_dev - band) / (split_dev + band)``, not
    ``box_dev / band``.  The second form treats the band as an
    uncertainty on the split's side while treating the box's deviation as
    exact, and it over-states: it reads 69.8x here against the strict
    57.7x.  Measured at every nu, saturated sp, each against its own
    reference (band | box deviation | strict bound):

        3.25   1.331e-05   3.476e-04   >= 20.6x
        3.50   1.955e-06   1.365e-04   >= 57.7x
        4.00   4.361e-07   2.289e-05   >= 40.6x
        5.00   5.417e-09   7.630e-07   >= 109.0x

    so the direction is resolved across the whole d = 3 alpha range, and
    only the size is not.

    And that bound is what is available: extending the raw-torus ladder
    cannot fix it, because V6E11 converges at sigma_eff = 4 and two more
    rungs buy well under the 11x tightening the point estimate would
    need.  Resolving it needs a second d = 3 family for exponent-3 cores
    — the split's OWN n-ladder would do it, since it converges at
    sigma_core = 11, but V6E11's full contraction is exponent 3 and out
    of budget above n = 8.  The RELEASE does not need any of this: the
    sign of the effect is resolved many times over.

    WHETHER IT ACTS DEPENDS ON THE BOUNDARY, and V6E11 is the example of
    that rather than of a guaranteed release.  The cuts are evaluated at
    the router's root -- ``min(external)`` where the block has a
    boundary, else the SP-cut-maximising vertex -- and on V6E11 they read
    (4.0, 11.0) at roots 1..5 but (4.0, 4.0) at root 0, because pinning
    the degree-2 vertex suppresses the very escape the split would
    relocate.  So on a standalone single-block call, where the source IS
    the boundary, ``source=0`` DECLINES to the box and ``source=1``
    releases to the torus -- both measured through this router.  The
    corpus case is the second: a dense block reached through a cut vertex
    is rarely entered at its degree-2 vertex.

    This is the conservative side of a known one-directional disagreement
    (see the note above on ``hybrid._planner_pin``): 0 blocks are
    released that the evaluation would not act on, 172 of 2636 are
    denied that it would.  The predicate is exactly the one that makes
    the split a no-op elsewhere, so it cannot release a block the split
    does not help.

    WHICH PIN THE PREDICATE IS EVALUATED AT, because the two available
    conventions disagree.  The caller computes the cuts at the
    router's own root -- ``min(external)`` where the block has a
    boundary, else the SP-cut-maximising vertex -- while the k = 0
    vacuum evaluation re-pins through ``hybrid._planner_pin``, which
    protects a different vertex.  Measured over all 2636 dense-block
    occurrences in the two TFIM corpora, the two conventions disagree on
    172 (6.5%), and **strictly one-directionally**: 0 occurrences are
    released here that the evaluation would not act on, and 172 are
    denied that it would (2 of the 38 exponent-3 occurrences the table
    constrains).  So this is safe as written and leaves ~6.5% of the
    release unclaimed; a future reader should not assume the two agree.
    """
    need = _dense_torus_min_n(edges, vertices, pin, keep, d)
    if need is None:
        return True
    if bool(split_acts):
        return True
    return bool(ladder) and int(n) >= int(need)


# Fine SP-collapse grid for the torus route, per d.  ``sp_n_points`` must
# be ≥ ``n_points`` (the SP grid is restricted *down* onto the core grid),
# and it costs ``sp_n_points^d`` per distinct ν — trivial at d ≤ 2
# (512² × 8 B = 2 MB), which is why d = 3 is deliberately absent: a
# missing entry means "no split", not "split at n_points".
#
# NOTE hybrid's ``max_core_bytes`` prices the CORE contraction at
# ``n_points`` and does NOT bound this grid (hybrid.py's own docstring
# says so).  A caller bounding total memory must bound this itself.
# Values are MEASURED, end-to-end through evaluate_graph against a box
# reference (the box family is independent of the torus, so it cannot
# flatter the split).  Per-block relative error, sigma = 0.5, acting
# blocks only — blocks with no suppressible degree-2 vertex are an exact
# no-op and reproduce the split-off column digit for digit, which is the
# control that makes the rest of the table readable:
#
#   d = 1, n_points = 64        OFF      1024      4096     16384
#     tw3 V5E7               7.2e-07   2.2e-06   1.5e-07   2.8e-08
#     tw3 V5E8               1.9e-06   3.6e-06   3.0e-07   9.0e-08
#     tw3 V5E7               3.2e-06   2.2e-06   4.7e-08   8.7e-08
#     tw3 V6E9               2.4e-06   5.8e-06   4.1e-07   7.6e-08
#
#   d = 2, n_points = 16        OFF       512      2048      8192
#     tw3 V5E7               1.7e-05   3.5e-07   3.2e-07   3.2e-07
#     tw3 V5E8               1.5e-05   5.7e-07   5.4e-07   5.4e-07
#     tw3 V5E7               2.3e-05   1.1e-07   1.5e-07   1.5e-07
#
# Two things to read off, and they point opposite ways.
#
# d = 1 needed RAISING: at 1024 the split was WORSE than no split on 3 of
# 4 acting blocks.  1024 sat in the trough where the fine grid is too
# coarse to have moved the block-cut mode much, while already being fine
# enough to invalidate the block-cut Richardson basis.  4096 clears it.
# 16384 is better again per block (median 8.4e-08 vs 2.3e-07), but the
# operating point is a PASS-level question, not a block-level argmin:
# 4096 captures ~98% of the converged coefficient move for +30% wall,
# where 65536 costs +345% for another ~2%.
#
# d = 2 was already right, and SATURATES at 512 — 2048 and 8192 buy
# nothing (identical to three digits).  It saturates early because the
# residual is then the core term A_core * n^-sigma_core, and n = 16 makes
# that term large; there is no point resolving the SP part far below the
# core's own floor.  Raising it would cost fine-grid memory
# (2048^2 x 8 B = 33 MB per distinct nu against 512^2 = 2 MB) for nothing.
#
# The mechanism behind both rows: err(n, sp) ~ A_blk*sp^-sigma_blk +
# A_core*n^-sigma_core.  The split shrinks the first term, the Richardson
# ladder (on the CORE basis — see the p_used site) removes the second.
# They are complements; sizing either while ignoring the other is what
# produced the trough at d = 1.
#
# d = 3 WAS DELIBERATELY ABSENT, AND THE REASON DID NOT SURVIVE
# MEASUREMENT.  The change that routed dense blocks to the torus at
# d = 3 scoped itself to the route alone -- "``_SPLIT_SP_N_POINTS`` and
# ``_CORE_KAPPA`` still have no d = 3 entry, so the split and the graded
# core stay off, and a new test pins that so adding one has to be
# deliberate" -- on the stated grounds that the fine grid costs
# ``sp_n_points**d``.  At d = 3, sp = 128 costs 128**3 x 8 B = 17 MB per
# distinct nu, against the 2 MB the d = 2 entry's own comment calls
# trivial.  This is that deliberate addition.
#
# THE MEASUREMENT.  d = 3, nu = 3.5, k = 0, relative error against a slab
# reference, and the multiple of the real-space box's own error.  K4SUB
# (K4 with one edge subdivided) is the canonical member of the class that
# dominates the corpus -- see the census below:
#
#     K4SUB, n_points = 16 (the shipped d = 3 grid)
#       sp        rel err     vs box     wall
#       off      2.033e-06     23.3x     1.7 s
#        64      4.934e-07     96.0x     1.9 s
#       128      1.940e-08   2442  x     2.6 s
#       256      1.028e-08   4607  x     1.7 s
#       box      4.736e-05      1  x
#
# 64 is still in the trough the d = 1 entry's own history warns about
# and 128 clears it; that much is resolved -- on K4SUB the two sit at
# 9948 and 391 reference band-widths, i.e. 96x and 2442x the box.  128 AGAINST 256 is NOT decided on accuracy:
# 256 is a resolved 1.9x better on K4SUB, reads 1.4x worse on a second
# acting shape (V6E9a, 5.17e-07 against 7.03e-07) where the difference is
# 1.9e-07 against that shape's 3.2e-07 band and so is not resolved at
# all, and a 24-shape census found the two agreeing to <= 5% everywhere.
# 128 is taken on COST, which is resolved: 17 MB against 134 MB per
# distinct nu, and cold on one exponent-2 block 0.57-0.89 s against
# 7.1-8.1 s with a 2.07 GB peak.  A second acting shape at 128, V6E9a:
# 3.37e-06 -> 5.17e-07, i.e. from 212x the box to at least 850x (that
# row is 1.6 band-widths, so the point value 1379x is not resolved).
#
# The reference is resolved: an n >= 20 slab ladder here and an
# independent 12-rung ladder to n = 96 agree to 3.79e-09 relative, sharing
# no rung above n = 24, and the comparisons above are quoted against the
# latter (resolution 4.96e-11).
#
# WHY IT IS WORTH A DEFAULT: the census.  Over both TFIM corpora (31 080
# graphs, 2636 tw >= 3 block occurrences in 1447 distinct shapes) the
# split acts on 2091 = 79.3%, of which the router reaches 2029 = 77.0%
# (93 occurrences have |external| >= 3 and go to graph_zeta_general,
# which has no split).
#
# AND THE GAIN IS BIMODAL -- do not read K4SUB's 2442x as typical.  What
# predicts it is the SP-REDUCED core cut, not whether the split acts:
#
#   sigma_core   share of all dense occurrences   gain over the box
#   11.0 (4 nu)              52.6%                34.8x-1159x, median
#                                                 115x at n_points = 8
#    7.5 (3 nu)              26.6%                0.96x-6.9x at
#                                                 n_points = 8;
#                                                 139x-2442x at 16
#   no-op                    20.7%                bit-identical; the box
#                                                 is 5x-50x better at
#                                                 n_points = 8
#
# K4SUB is in the SECOND class, which is why its number is quoted at
# n_points = 16 and not at 8: at n_points = 8 that class is a wash
# (one of three measured shapes reads 0.96x).  The cheap fix for it is
# n_points, not sp -- at n_points = 12 it reaches 15x-82x the box in
# 0.14-0.20 s, still under the box's own 0.31-0.41 s -- and that is
# affordable because 98.6% of d = 3 dense-block occurrences are plan
# exponent 2 (n**6), not 3 (n**9).
#
# COST, and it is not a uniform win either.  Cold, on ONE exponent-2
# block in a fresh process, sp = 128 is 1.4x-2.8x SLOWER than the box
# (0.57-0.89 s against 0.31-0.41 s); sp = 256 is 20x slower and needs
# 2.07 GB.  Warm -- repeat calls with the SP-collapse memo hot, which is
# the corpus case the block cache creates -- every sp is 4-13x faster
# than the box.
#
# WHY IT CANNOT DISTURB THE REST.  The split acts only where the block's
# own cut is below its SP-reduced cut; where they coincide there is
# nothing to reduce and it is an exact no-op.  The three exponent-2 cores
# that ship at d = 3 are all in that second group, and are BIT-IDENTICAL
# at sp = 64, 128 and 256 (K4 30.825196658828744, prism
# 1020.4186113579578, K33 1887.5284456250074 at n_points = 16).
#
# WHAT IT DOES NOT FIX.  An exponent-3 core at n_points = 16 needs
# 768 GiB and is byte-refused before the split matters; the fine grid
# rescues the block's RATE, not the size of its core.  Reaching those
# blocks needs the graded core (``_CORE_KAPPA``, still absent at d = 3)
# or the slab.
_SPLIT_SP_N_POINTS = {1: 4096, 2: 512, 3: 128}

#: Dimensions whose split entry is switched on at FINITE k as well as at
#: k = 0.
#:
#: d = 3 is deliberately absent, and it is absent because it was
#: measured.  The k = 0 case is unambiguous (above).  At finite k there
#: is no third family to rank against -- the k = 0 sigma_eff ladder does
#: not run there, and the slab is a k = 0 instrument -- so all that can
#: be measured is whether the split moves the torus toward the box or
#: away from it.  It moves it AWAY.  K4SUB at d = 3, n_points = 16,
#: |torus - box| / box, split off against sp = 128:
#:
#:      k                     off        sp = 128    change
#:      (0,     0,     0)     7.621e-05  4.639e-05    1.64x
#:      (0.125, 0,     0)     1.504e-03  1.627e-03    0.92x
#:      (0.25,  0.25,  0)     1.626e-03  1.766e-03    0.92x
#:      (0.5,   0.5,   0.5)   4.052e-03  4.372e-03    0.93x
#:      (0.375, 0.125, 0.25)  1.260e-03  1.366e-03    0.92x
#:
#: and K4, on which the split cannot act, is 1.00x at every k -- the
#: control that says the 0.92x is the split and not the harness.
#:
#: An 8% widening of a gap that is already 1e-3 is not evidence that the
#: split is WRONG at finite k; it may be the box moving.  It is evidence
#: that nothing here can tell, and a default should not be set on that.
#: The cheapest experiment that would settle it is a second finite-k
#: family at d = 3 -- the slab does not do finite k today.
_SPLIT_FINITE_K_DIMS = frozenset({1, 2})

# Budget for a single dense-core contraction, in bytes.  Above it the
# block takes the real-space box instead, via the ``MemoryError``
# re-raise in ``_block_general`` (``HybridCoreBudgetError`` is a
# ``MemoryError`` subclass, and ``_dense_core`` prices the contraction
# from ``_elimination.plan_for_order`` BEFORE allocating).
#
# The point is DETERMINISM first and speed second.  With no budget the
# route was decided by whether an 8.6 GB allocation happened to fit, so
# the same pass took 13.4 s or 45.6 s depending on free RAM -- and the
# SLOW branch was the one where the allocation SUCCEEDED, because the
# box is cheaper on exactly the blocks that are too big for the torus.
#
# 2 GB is chosen to sit above every block the torus should keep and
# below the p_cost = 3 class at the resolutions production reaches
# (32^6 x 8 B = 8.6 GB, 48^6 = 98 GB).  It is a ceiling on ONE
# contraction, not on the process: hybrid's own docstring notes the cap
# does not bound the SP grid, and the executor holds several live arrays
# per step (``_CORE_RETENTION``), both of which this figure already
# accounts for by being well under the machine.
_DENSE_CORE_MAX_BYTES = float(
    os.environ.get("GZ_DENSE_CORE_MAX_BYTES", 2.0 * 1024**3)
)

# ---------------------------------------------------------------------------
# Connectivity-graded core resolution — the DOWNWARD half of the split axis.
#
# ``sp_n_points`` raises the SP grid above ``n_points``.  This lowers the
# irreducible CORE below it, per block, and it is the half that carries the
# cost: a core contracts ``n^(τ·d)`` entries against the fine grid's ``n^d``.
#
# THE RULE.  Both rates are already on the routing path (see the
# ``sigma_eff_core`` site): the block converges at σ_blk = min cluster cut
# − d = 2ν − d (a degree-2 escape), and once the SP part is collapsed on the
# fine grid what is left decays at the CORE's own cut, σ_core ≥ 3ν − d.
# Sizing the core so its residual sits at or below the error the block would
# contribute at the delivery grid gives
#
#     n_core = κ · n_points^(σ_blk / σ_core)
#
# with nothing else to fit.  Two properties make it safe to apply blindly:
# on a block with nothing to reduce the two cuts are EQUAL by construction,
# the exponent is 1, and the rule returns ``n_points`` — it self-disables on
# the 13.5% of occurrences that are already 3-connected.  And the exponent is
# < 1 exactly when the split has something to do, so the saving GROWS with
# ``n_points``: dense-core cost scales as ``n^(σ_blk/σ_core · τ·d)`` instead
# of ``n^(τ·d)`` — at d = 2, ν = 2.5 that is n^2.18 against n^4.
#
# κ IS MEASURED, and κ = 1 is not good enough — it under-sizes by a factor
# 1.7–2.8.  Calibrated operationally (no amplitude fit): the smallest core
# whose SPLIT error is at or below the error the RAW block carries at the
# delivery grid, over the top census blocks at σ = 0.5, against a
# two-family-resolved reference.  Acting blocks only; the no-op blocks
# return κ = 1.00 and n_core = n_points exactly, which is the control that
# makes the rest readable.
#
#   d = 1: κ = 1.70, 2.00, 2.13, 2.24, 2.21, 2.50, 2.60, 2.63, 2.80
#   d = 2: κ = 1.76, 1.77, 1.95, 2.43, 2.43, 2.60, 2.60, 2.83, 2.83
#
# median 2.24 (d=1) / 2.43 (d=2), max 2.83.  κ = 3.0 clears every measured
# block.  The margin is thinner than it looks in error terms rather than
# thicker: the calibration target (the block's own raw error at n_points) is
# itself ~17x BELOW the pass-level floor measured for the same
# configuration (1.64e-02 at d = 2 for n_points 16 → 32), so a κ that
# merely matches it is already conservative for the quantity that ships.
#
# RECALIBRATED for the pass-floor numerator (see ``_pass_floor_sigma``).
# The values above were measured for the OLD numerator, ``sigma_blk``, and
# do not transfer: the pass floor is a smaller numerator, so the rule
# reaches further down and needs more margin.  d = 1 needed raising;
# d = 2 did not.
#
# d = 2 — measured over 12 gradeable census blocks at nu = 2.5,
# n_points = 32, target 1e-3 against the UNGRADED value at the same
# n_points (so it isolates what grading costs and nothing else):
# kappa needed is median 1.21, max 1.51.  The shipped 3.0 is a 2x margin.
# Checked separately on K5 itself (4-connected, tw = 4, the shape that
# broke d = 1): grading error 3.2e-06 at n_points = 16 and 2.6e-06 at 32.
#
# d = 1 — 3.0 was NOT enough.  Worst grading error over
# {K5, W4, K5-e} x {nu 1.5, 2.105, 2.5} x {n_points 16, 64}:
#
#     kappa_d1   worst grading error
#          3.0              3.73e-02      <- fails; caught by
#                                            test_k5_routed_to_the_torus
#          4.0              1.72e-04
#          4.5              1.31e-04      but stops grading W4 / K5-e at all
#          5.0              7.22e-05
#
# 4.0 clears the 1e-3 target with ~6x margin and still grades.  The
# blocks that forced it are the 4-CONNECTED ones: a larger sigma_core
# drives n_core down faster while the amplitude does not follow, and a
# census sampling only sigma_core = 3.5 blocks misses them entirely --
# which is exactly how the first calibration attempt here went wrong.
#
# CAVEAT, stated because the sample is small: 12 census blocks at d = 2
# and 3 hand-picked shapes x 4 configs at d = 1, all at k = 0, square /
# chain only.  A wider campaign could move these.
#
# d = 3 IS PRESENT, AND ITS VALUE IS MASKED.  It was absent because the
# d = 3 torus routing change scoped itself to the route, and because
# grading an exponent-3 core sends it to a SMALLER torus, which measured
# on its own is a LOSS: V6E11 graded to n_core = 8 with the split off
# reads 1.44e-03 against the box's 1.37e-04.  Two things changed.  The
# split now ships at d = 3, and the rung-conditional stability tolerance
# now accepts the exponent-2 ladders that grading exposes -- K4 at
# n_top = 12 and 14 went from `unstable` (raw 0.15x and 0.43x the box)
# to accepted (8.01x and 13.82x).  Together those remove both objections.
#
# MEASURED, d = 3, nu = 3.5, k = 0, slab-referenced, kappa absent vs
# present (relative error, and the multiple of the box's own error):
#
#   core     n_points   without kappa3        with kappa3
#   K4          16      torus     121.71x     identical (not graded)
#   K4          24      BOX         1.00x     graded torus  121.71x
#   K4          32      BOX         1.00x     graded torus  121.71x
#   prism       24      BOX         1.00x     graded torus   56.37x
#   K33         24      BOX         1.00x     graded torus   25.15x
#   V6E11       16      BOX  1.00x  72.5 s    graded torus  381.82x  6.9 s
#   V6E11       24      BOX  1.00x  78.0 s    graded torus  381.82x 10.4 s
#   V6E11       32      BOX  1.00x  75.3 s    graded torus  381.82x  7.1 s
#
# Two properties worth stating because they are what make it safe.  The
# exponent-2 class is UNTOUCHED at n_points = 16 -- the minimum-shrink
# gate declines there -- so the shipped d = 3 grid moves not at all for
# 98.6% of dense-block occurrences; grading only reaches them at
# n_points >= 24, where it replaces the box with a torus that is 25x-122x
# better.  And the exponent-3 class stops reaching the box at ALL: its
# core is byte-refused at every n_points, so before this it took the box
# unconditionally.
#
# THE VALUE IS NOT LOAD-BEARING: kappa = 3.0 and kappa = 4.0 give
# byte-for-byte identical results in every row above, because at d = 3
# `_core_n_capped`'s byte ceiling binds before kappa does.  3.0 is taken
# to match d = 2 rather than because it was fitted.  (V6E11's 381.82x is
# a point estimate inside its reference band; the resolved bound is
# >= 57.7x -- see _dense_torus_is_accurate.)
_CORE_KAPPA = {1: 4.0, 2: 3.0, 3: 3.0}

# Below this the torus is pre-asymptotic and the rule stops meaning
# anything: `_richardson_rungs`' own docstring records the sequence
# overshooting and coming back below n ~ 8, and two d = 3 ladders were
# measured disagreeing by 2.1 in the exponent at n_c ≤ 16.  The rule
# reaches this floor only at small ``n_points``, where there was little
# to save anyway.
_CORE_N_FLOOR = 8

# A graded core also GIVES UP the k = 0 Richardson ladder, so grading is
# only worth taking when it buys enough to pay for that.  Measured, d = 2,
# sigma = 0.5, n_points = 16, census blocks, against a two-family
# reference, separating the two effects (A = shipped, B = ladder off
# only, C = graded):
#
#   n_core   cost saving   ladder was worth   coarsening costs   TOTAL paid
#     10         6.6x            1.0x               7.7x            7.7x
#     14         1.7x          163.5x               2.0x          318.3x
#
# At n_core = 10 the ladder was already declining on its own guards, so
# grading costs only the coarsening and buys 6.6x.  At n_core = 14 the
# ladder was WORKING and worth 163x, while the core reduction saves just
# 1.7x — paying 318x for 1.7x.  The coarsening itself is cheap there
# (2.0x); almost all of that loss is the ladder.
#
# So: grade only when the core actually shrinks enough to matter, and
# otherwise leave the block alone entirely — which also leaves its ladder
# running.  A ratio rather than a cost factor because the router applies
# it before it needs `p_cost`; at d = 2, p_cost = 2 this ratio is a
# 3.2x floor on the entries saved.
_CORE_MIN_SHRINK = 0.75

# How far the core term must exceed the SP floor before extrapolating is
# worth it.  See :func:`_core_term_below_sp_floor` for the measurement.
_CORE_LADDER_MARGIN = 3.0


#: Hard ceiling on ONE dense-core contraction, in bytes.  This is a
#: CEILING, not a target: the accuracy rule picks a core size and this
#: caps it, so cost is bounded by construction rather than by hoping the
#: rule was generous.
#:
#: The contraction holds ``(n_core^d)^p`` complex entries, where ``p`` is
#: the post-peel exponent, so the cap is ``n <= (bytes/16)^(1/(p d))``:
#:
#:     d   p   n_core cap at 256 MB     entries        what it is
#:     2   2                     53   7.9e+06          tw 3, one terminal
#:     2   3                     14   7.5e+06          tw 3 + 2 terminals, tw 4
#:     2   4                      8   1.6e+07          tw 5
#:     3   3                      5         —          below the floor -> box
#:
#: Without it a d = 2 exponent-3 block at n_points = 32 allocates
#: **17.2 GB** and one such block took 78% of an order-11 pass.  The
#: exponent grows with treewidth and with d, so an uncapped core is a
#: memory wall waiting for the first tw = 5 shape or the first d = 3 run;
#: capping makes the worst case a function of the budget alone.
#:
#: Where even ``_CORE_N_FLOOR`` does not fit, the block is left to the
#: existing ``HybridCoreBudgetError`` -> box fallback, which is the
#: right answer there: the box's cost does not scale with ``n_points``.
_CORE_MAX_BYTES = 256 * 1024 * 1024


def _core_n_for_budget(exponent, d, max_bytes=None, itemsize=16):
    r"""Largest even grid whose core contraction fits the byte budget.

    Returns ``None`` when the exponent is unknown or non-positive (no
    constraint expressible), and may return a value below
    ``_CORE_N_FLOOR``, which the caller should read as "this block does
    not belong on the torus at all".
    """
    if exponent is None:
        return None
    p = int(exponent) * int(d)
    if p <= 0:
        return None
    budget = _CORE_MAX_BYTES if max_bytes is None else float(max_bytes)
    n = int(2 * math.floor(((float(budget) / float(itemsize)) ** (1.0 / p)) / 2.0))
    # The float root lands just under an exact power at the boundary, so
    # settle it in integer arithmetic rather than trusting it.
    while n >= 2 and (n ** int(d)) ** int(exponent) * itemsize > budget:
        n -= 2
    while ((n + 2) ** int(d)) ** int(exponent) * itemsize <= budget:
        n += 2
    return max(n, 2)


def _core_n_capped(n_rate, edges, vertices, root, keep, d):
    r"""Apply the byte ceiling to an accuracy-chosen core size.

    ``n_rate`` is what :func:`_core_n_for_block` asked for.  This never
    RAISES it — a budget is not a licence to refine — it only lowers it.
    """
    try:
        exponent = _elimination.plan(
            edges, vertices, pin=root, keep=tuple(keep)).exponent
    except Exception:
        return n_rate
    cap = _core_n_for_budget(exponent, d)
    if cap is None:
        return n_rate
    return min(int(n_rate), max(int(cap), _CORE_N_FLOOR))


def _block_plan_exponent(edges, root, keep):
    r"""Post-peel cost exponent of a block's core, or ``None``.

    The same memoised plan :func:`_core_n_capped` prices, at the same
    pin and kept set, so the two cannot disagree about which core they
    are describing.  ``None`` when the planner refuses, which the
    callers read as "no opinion" — never as an exponent.
    """
    try:
        return int(_elimination.plan(
            edges, sorted({int(v) for e in edges for v in e[:2]}),
            pin=int(root), keep=tuple(keep),
        ).exponent)
    except Exception:
        return None


def _pass_floor_sigma(nu, d):
    r"""The rate at which the PASS's ``n_points`` error decays.

    Not the rate of any particular block — the rate of the SLOWEST thing
    on the grid, which is what ``n_points`` is sized for and therefore
    the level a dense core only has to reach.

    It is the degree-2 escape, ``2 nu_base - d``.  MEASURED on the d = 2
    1qp corpus, over every block that reaches an ``n_points``-dependent
    engine (the closed forms — bridges, off-spine cycles — carry no grid
    error and are excluded):

        cut/nu   rate = cut - d   distinct blocks   occurrences
          2.00             3.00               357          1001   85.0% / 83.5%
          4.00             8.00                55           179
          6.00            13.00                 6            17
          8.00            18.00                 2             2

    and ``cut/nu = 3`` never occurs as a block minimum at an
    n_points-dependent engine, at order <= 8, <= 9 or <= 11.

    CORRECTION, and it matters for how this is justified rather than for
    the number.  The table above is the distribution of BLOCK CUT rates.
    It is NOT the pass's convergence exponent, and a direct fit of
    ``c_O(n_points)`` says the pass floor is **4.3-4.5**, not 3.0:

        two ratio-2 rung families, 8/16/32/64 and 12/24/48/96,
        s = log2(|d_i| / |d_{i+1}|) with no fitting, orders 4/6/9:
            family A   4.59 4.53 | 4.23 4.32 | 4.39 4.40
            family B   4.55 4.51 | 4.29 4.35 | 4.39 4.40
        control: orders 1-3 are BIT-IDENTICAL across n = 12..96
        (bridges and off-spine cycles are closed form), so the harness
        demonstrably separates n-dependent from n-independent orders.

    The reason the cut census and the fit disagree: 83-85% of
    n_points-dependent occurrences take the sigma_max = 4 algebra path,
    which does not carry a ``2 nu - d`` term at all — at
    ``2 nu - d = 3.0 <= d + sigma_max = 6.0`` the degree-2 chain exponent
    is produced ANALYTICALLY and never enters the windowed Fourier part.
    So the slowest block cut is not the slowest thing on the grid.

    WHY THIS FUNCTION STILL RETURNS 2 nu - d.  It is deliberately BELOW
    the measured floor, which makes the cores SMALLER, not larger — the
    conservative direction for cost and the aggressive one for accuracy.
    That is safe here only because it is measured end to end: at d = 2,
    n_points = 32, order 11, the whole pass agrees with the box to
    1.7e-09 (k = 0) and 1.5e-06 (full BZ grid), against a pass floor of
    ~1.9e-06.  Raising this to the fitted 4.4 would take ``n_core`` from
    12 to 20 at ``n_points = 32`` and cost roughly ``(20/12)^(tau d)``
    for accuracy the pass cannot see.

    ``nu`` IS ONE BLOCK'S OWN PER-EDGE EXPONENTS.  Everything above was
    measured on uniform-nu passes, where every block reads the same
    ``2 nu - d``, so there the floor of the pass and the floor of each
    block are one number.  With per-edge nu they are not, and reading
    the floor across blocks was measured wrong twice.  A pendant bridge
    set it: K5 at nu = 2.5 on the chain, n_points = 32, beside a bridge
    at nu = 1.2 was graded to a core of 8 and came out 1.10e-01 off its
    value without the bridge.  With the closed forms left out, a grid
    block still set it, because a low-nu neighbour's actual error lies
    far below its ``2 nu - d`` level (the CORRECTION above).  K5 beside
    a pendant diamond at nu_x, the shift of the K5 factor against the
    diamond's own error, both at n_points = 32:

        nu_x    K5 shift    diamond's error
        2.2     8.4e-06     3.6e-05
        2.0     5.7e-05     9.6e-05
        1.8     4.7e-04     2.8e-04
        1.5     5.5e-03     5.4e-05
        1.2     1.1e-01     6.5e-04

    So the router passes each block its own per-edge exponents.  A
    dense core then reads the floor it would read alone, whatever sits
    beside it, which is the regime ``_CORE_KAPPA`` was calibrated in,
    and a uniform-nu call reads the same number as before.  (Where the
    block attaches still decides its pin, and with it ``sigma_core``;
    that is unchanged.)  The price is the extra saving a lower-nu
    neighbour used to buy, which a uniform pass never had.

    Per edge, not per bundle, because that is what the calibration and
    the pass measurements above read.  It leaves one dependence in
    place: splitting a bundle into parallel edges lowers the floor.  K5
    at nu = 2.5 reads 4.0 and is graded to 20; the same K5 with one edge
    as two parallel edges of 1.25 reads 1.5 and is graded to 8, 1.10e-01
    away for the same lattice sum.  Reading bundles would move uniform-nu
    passes wherever a dense block has no single edge.

    Returns ``None`` for an empty ``nu``, which the router never passes.
    """
    nu = np.asarray(nu, dtype=float)
    if nu.size == 0:
        return None
    return 2.0 * float(np.min(nu)) - float(d)


def _core_n_for_block(n_points, sigma_blk, sigma_core, d, sigma_ref=None):
    r"""Coarse core grid for one block, or ``n_points`` to leave it alone.

    Even-valued: the balanced label axis is symmetric at odd ``n`` and
    asymmetric at even ``n`` (label ``n/2`` has no partner ``-n/2``),
    which splits the truncation coefficient into two families, and a
    core whose parity floats with the arithmetic drifts between them.
    """
    kappa = _CORE_KAPPA.get(int(d))
    n = int(n_points)
    if kappa is None or n <= _CORE_N_FLOOR:
        return n
    # THE NUMERATOR IS THE PASS'S RATE, NOT THE BLOCK'S.
    #
    # Sizing against ``sigma_blk`` — the block's own pre-reduction rate —
    # is circular: it targets the error THIS block would have at the
    # delivery grid, so it can never do better than the delivery grid.
    # On a 3-connected block the two cuts coincide by construction, the
    # exponent is exactly 1, and the rule self-disables on precisely the
    # blocks that cost the most.  ``sigma_ref`` (see
    # :func:`_pass_floor_sigma`) is the level the core actually has to
    # reach: what the REST of the pass contributes at ``n_points``.
    #
    # ``None`` falls back to the block's own rate, which is the legacy
    # behaviour and is correct only where the two coincide — i.e. on a
    # block whose cheapest escape is the same degree-2 mode that sets the
    # pass floor.
    ref = float(sigma_blk) if sigma_ref is None else float(sigma_ref)
    if not sigma_core or ref <= 0.0 or float(sigma_core) <= ref:
        # A core that converges no faster than the floor it is being
        # sized against: there is no gap to spend.
        return n
    raw = kappa * (float(n) ** (ref / float(sigma_core)))
    n_core = int(2 * math.ceil(raw / 2.0))              # snap up to even
    n_core = max(_CORE_N_FLOOR, min(n_core, n))
    if n_core > _CORE_MIN_SHRINK * n:
        # Marginal shrink: not worth the ladder it would cost us.
        return n
    return n_core


def _core_term_below_sp_floor(n_core, sigma_core, sp_n, sigma_blk):
    r"""Has the core truncation term fallen below the SP-collapse floor?

    The split's error model, measured and validated to 1-4% over three
    decades::

        err(n_core, sp) ~= A_blk * sp^-sigma_blk + A_core * n_core^-sigma_core

    The first term is what the SP collapse leaves behind.  It does not
    depend on ``n_core`` at all, so it is a CONSTANT across any ladder in
    ``n_core`` and no extrapolation can remove it, which was measured
    directly.  The ladder can only take the second term.  Once that has
    fallen below the first, extrapolating removes nothing and amplifies
    whatever noise is left.

    Both sides are known before any evaluation: ``sigma_core`` is exact
    in the physical regime ``nu > d`` (brute-force verified over the
    corpus) and ``sigma_blk`` is the block's own cluster cut.
    Amplitudes are dropped deliberately — they are O(1) and comparing
    them would need the per-block fit this gate exists to avoid.

    ``_CORE_LADDER_MARGIN`` is why this is not a bare inequality: the
    amplitudes are dropped, so the two sides are known only up to O(1)
    factors and the crossing has to be cleared by a margin, not merely
    reached.  It is MEASURED — d = 2, sigma = 0.5, exact-exponent
    two-point elimination on graded cores, ratio = core term / SP floor:

        sigma_core  n_core   ratio    measured gain over raw
           5.5        20      9.4     13.4-15.5x   (3/3 win)
           8.0        10      1.3      0.0- 2.5x   (4/6 LOSE)
           8.0        12      0.3      0.8- 4.9x   (3/5 lose)

    Any margin between 1.3 and 9.4 separates the winning population from
    the losing one; 3.0 sits in the middle of that gap.

    With no split (``sp_n is None``) there is no SP floor, so the answer
    is always False and the ladder runs as it always has.
    """
    if sp_n is None or not sigma_core or not sigma_blk:
        return False
    return (float(n_core) ** (-float(sigma_core))
            <= _CORE_LADDER_MARGIN * float(sp_n) ** (-float(sigma_blk)))


def _block_cut_rates(edges, nu_vec, d, root, externals=()):
    r"""``(sigma_blk, sigma_core)`` for one block, from its edge map.

    Shared by the k = 0 and the finite-k dispatch so the two cannot
    drift: the same block reached through different entry points must be
    graded identically, or the block cache would serve one entry's
    answer to the other's question.
    """
    emap: dict = {}
    for (u, v), w in zip(np.asarray(edges), np.asarray(nu_vec)):
        key = (min(int(u), int(v)), max(int(u), int(v)))
        emap[key] = emap.get(key, 0.0) + float(w)
    ext = tuple(int(x) for x in externals)
    return (_min_free_cut_nu(emap, int(root), externals=ext) - float(d),
            _sp_reduced_cut_nu(emap, int(root), externals=ext) - float(d))


def _compact_core_floor(edges, kernels) -> int:
    r"""Smallest even core grid on which a graded core keeps every table
    of a block exact; ``0`` when no bundle carries a compact part.

    A graded core is contracted on a grid COARSER than the one the
    series/parallel part was collapsed on: hybrid gathers the residual
    kernels down onto the core window (``_contract.restrict_to``), and a
    gather cannot see a table.  Two conditions, both on the SP-REDUCED
    residuals rather than on the input kernels:

    * the core window must hold every residual table.  A series chain
      SUMS the radii of its edges (the all-table term of ``a_1 * a_2``
      reaches ``r_1 + r_2``), a parallel merge takes the MAX (the
      compact part of ``(a_1 + t_1)(a_2 + t_2)`` is supported on the
      union), and the balanced axis holds ``±r`` only for
      ``r <= (n - 1) // 2`` -- ``n >= 2 r + 2`` at the even sizes cores
      take.  A table clipped by the window is an O(a^m) error, not a
      truncation: it never decays with ``n``;
    * no core cycle may wind.  The all-table term of a cycle whose
      residual radii sum to ``S`` closes around the torus once
      ``n <= S``, and ``S`` is bounded by the block's own winding-safe
      size ``n_v R + 2``: each residual edge realises the radius of one
      original simple path, and internally disjoint paths around a core
      cycle form a simple cycle of at most ``n_v`` original edges.

    The reduction here keeps NO vertex, which suppresses at least what
    any engine and any pin suppresses; radii only grow under suppression
    (a sum exceeds each part, a max exceeds each branch), so the bound
    holds whichever vertices hybrid keeps.  The tail terms of a residual
    (``a * t``, ``t * t``) are power laws and are what the kappa rule
    truncates, exactly as for a pure kernel.
    """
    if kernels is None:
        return 0
    radii = [int(k.support_radius) if k.has_compact else 0 for k in kernels]
    if not any(radii):
        return 0
    pairs = np.asarray(edges, dtype=int).reshape(-1, 2).tolist()
    n_v = len({int(v) for e in pairs for v in e})
    adj: dict = {}
    for (u, v), r in zip(pairs, radii):
        u, v = int(u), int(v)
        if u == v:
            continue
        adj.setdefault(u, {})
        adj.setdefault(v, {})
        adj[u][v] = max(adj[u].get(v, 0), r)      # parallel: MAX
        adj[v][u] = max(adj[v].get(u, 0), r)
    r_max = max((r for nb in adj.values() for r in nb.values()), default=0)
    changed = True
    while changed:
        changed = False
        for x in list(adj):
            if x not in adj or len(adj[x]) != 2:
                continue
            (a, ra), (b, rb) = list(adj[x].items())
            if a == b:
                continue                             # would make a loop
            del adj[x]
            adj[a].pop(x, None)
            adj[b].pop(x, None)
            r = int(ra) + int(rb)                    # series: SUM
            adj[a][b] = max(adj[a].get(b, 0), r)     # ...then parallel: MAX
            adj[b][a] = max(adj[b].get(a, 0), r)
            r_max = max(r_max, r)
            changed = True
    floor = max(n_v * max(radii) + 2, 2 * r_max + 2)
    return int(2 * math.ceil(floor / 2.0))


def _graded_core_with_tables(n_graded, n_ref, edges, kernels) -> int:
    r"""Lift a rule-chosen core size to :func:`_compact_core_floor` and
    re-apply the marginal-shrink rule against ``n_ref`` (the grid the
    block would otherwise run at).  Returns ``n_ref`` to leave the block
    alone.  A no-op for a block without tables.
    """
    n_ref = int(n_ref)
    n_core = int(n_graded)
    floor = _compact_core_floor(edges, kernels)
    if floor:
        n_core = max(n_core, floor)
    if n_core >= n_ref or n_core > _CORE_MIN_SHRINK * n_ref:
        return n_ref
    return n_core


def _compact_exact_grid(edges, R, source=None, terminal=None) -> int:
    r"""Smallest torus on which a PURELY compact block is exact.

    A purely compact bundle (every kernel a table of Chebyshev radius
    ``<= R``, tail ``+inf``) has finitely many configurations, and the
    torus sum equals the lattice sum exactly as soon as every
    configuration on the torus lifts uniquely to the lattice.  Fix a
    spanning tree of the block and lift positions along it: a
    configuration fails to lift only when the representatives of some
    FUNDAMENTAL cycle -- each in ``[-R, R]^d`` -- sum to a non-zero
    multiple of ``n``, which a cycle of ``L`` edges cannot do once
    ``n > L R``.  The bound is therefore the longest fundamental cycle
    of the best spanning tree, not the vertex count: the shipped rule
    ``n_v R + 2`` (a simple cycle has at most ``n_v`` edges) is the
    same argument with the worst tree.  Breadth-first trees are tried
    from every root; a K5 at ``R = 1`` then closes at ``n = 4``
    (every fundamental cycle a triangle) against ``n_v R + 2 = 7``, and
    the dense contraction pays that ratio to the power of its
    exponent.

    At finite momentum the form factor is read off the balanced label
    axis, so the source-terminal displacement -- at most ``ell R`` for
    the graph distance ``ell`` of the two -- must also sit inside the
    window: ``n >= 2 ell R + 1``.  Both bounds are met at every odd
    ``n`` this returns (``max(L R, 2 max(ell, 1) R) + 1``), and the
    table itself needs ``n >= 2 R + 1``, the ``ell = 1`` case.  The
    value is then the SAME finite lattice sum on every admissible
    torus -- identical bit for bit for an integer table (the sparse
    peel keeps its arithmetic exact) and to round-off otherwise --
    which is what lets the router choose the cheapest grid freely.
    ``R = 0`` (a table on the origin alone) is treated as ``1``.
    """
    R = max(int(R), 1)
    pairs = np.asarray(edges, dtype=int).reshape(-1, 2)[:, :2].tolist()
    adj: dict = {}
    for u, v in pairs:
        u, v = int(u), int(v)
        if u == v:
            continue
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    for v in (source, terminal):
        if v is not None:
            adj.setdefault(int(v), set())
    verts = sorted(adj)
    simple = {(min(int(u), int(v)), max(int(u), int(v)))
              for u, v in pairs if int(u) != int(v)}

    def _bfs(root):
        depth = {root: 0}
        parent = {root: None}
        frontier = [root]
        while frontier:
            nxt = []
            for x in frontier:
                for y in sorted(adj[x]):
                    if y not in depth:
                        depth[y] = depth[x] + 1
                        parent[y] = x
                        nxt.append(y)
            frontier = nxt
        return depth, parent

    L = 0
    if len(simple) >= len(verts):            # at least one cycle
        best = None
        for r in verts:
            depth, parent = _bfs(r)
            L_r = 0
            for u, v in simple:
                if u not in depth or v not in depth:
                    continue                 # another component: not ours
                if parent.get(v) == u or parent.get(u) == v:
                    continue                 # tree edge
                a, b = u, v
                while a != b:                # lowest common ancestor
                    if depth[a] >= depth[b]:
                        a = parent[a]
                    else:
                        b = parent[b]
                L_r = max(L_r, depth[u] + depth[v] + 1 - 2 * depth[a])
            best = L_r if best is None else min(best, L_r)
        L = int(best or 0)
    ell = 0
    if (source is not None and terminal is not None
            and int(source) != int(terminal)):
        depth, _ = _bfs(int(source))
        ell = int(depth.get(int(terminal), 0))
    return int(max(L * R, 2 * max(ell, 1) * R) + 1)


def _resolve_dense_routing(d, dense_engine, sp_n_points, n_points, nn_mode,
                           core_grading=True):
    r"""Resolve the dense-block route, its SP grid and its core grading.

    Returns ``(engine, sp_n, core_grade)`` where ``engine`` is one of
    ``_DENSE_ENGINES``, ``sp_n`` is either ``None`` (no split) or an
    ``int`` strictly greater than ``n_points``, and ``core_grade`` is a
    bool saying whether dense cores are sized down per block by
    :func:`_core_n_for_block`.

    Core grading is NOT tied to the dense route: it is offered to every
    block that reaches hybrid, tw ≤ 2 included.  A tw ≤ 2 block at
    σ ≥ 1.49 skips the σ_max algebra and hybrid is its only engine, so
    the dense-engine choice says nothing about it — and those blocks
    were 16.6 s of an 18.2 s order-9 d = 2 pass at n_points = 64 while
    the rule already sized their cores at 18.  The dense arms add their
    own ``dense_on_torus`` gate on top.  Grading IS suppressed at
    ``ν = inf`` (a coarser core changes which walks exist rather than
    approximating the answer) and when ``d`` has no measured ``κ``: a
    missing entry means "not calibrated here", not "grade with some
    default", exactly as a missing ``_SPLIT_SP_N_POINTS`` entry means
    "no split".

    Normalising here rather than at the call sites is what keeps the
    block-cache key honest: the key stores exactly this pair, so two
    callers whose *requests* differ but whose *resolved* routing agrees
    share cache entries, and two whose resolved routing differs never do.

    ``sp_n_points == n_points`` is folded to ``None``.  hybrid treats the
    two as bit-identical (its split branch is an exact no-op there), so
    collapsing them keeps the default path structurally untouched and
    stops a redundant second cache entry for the same numbers.
    """
    eng = (_DENSE_ENGINE_BY_D.get(int(d), "direct_sum")
           if dense_engine is None else str(dense_engine))
    if eng not in _DENSE_ENGINES:
        raise ValueError(
            f"dense_engine must be one of {_DENSE_ENGINES}, got {eng!r}"
        )
    grade = bool(core_grading) and not nn_mode and int(d) in _CORE_KAPPA
    # The split is meaningless off the torus route, and hybrid refuses it
    # outright at ν = inf (the kernel is then an indicator with no
    # power-law tail to truncate — an FFT collapse would return 2.0 as
    # 1.9999999999999998).  Suppress it here rather than let hybrid raise
    # a ValueError that no caller in this module catches.  Grading is
    # NOT suppressed with it: a tw ≤ 2 block still runs hybrid whatever
    # the dense engine is, and ``grade`` already excludes nn_mode.
    if eng != "torus" or nn_mode:
        return eng, None, grade
    sp = (_SPLIT_SP_N_POINTS.get(int(d)) if sp_n_points is None
          else int(sp_n_points))
    if sp is None:
        return eng, None, grade
    sp = max(int(sp), int(n_points))
    return eng, (None if sp == int(n_points) else sp), grade


__all__ = [
    "evaluate_graph",
    "GraphZetaError",
    "SelfLoopError",
    "DisconnectedGraphError",
    "VertexOutOfRangeError",
    "NPointsRequiredError",
    "TopologyEvaluatorUnavailableError",
    "UnsupportedLatticeSumError",
    "UnsupportedRequestError",
]


# ---------------------------------------------------------------------------
# Spine tagging on the block-cut tree
# ---------------------------------------------------------------------------

def _spine_path(
    blocks_v,
    cuts,
    s: int,
    t: int,
) -> "list[tuple[frozenset, int, int]]":
    """Return a list of ``(block, s_b, t_b)`` triples for each block on
    the s-t spine of the block-cut tree, in order from ``s`` to ``t``.

    ``s_b`` is the spine-entry vertex of the block (the cut vertex
    where the spine enters, or ``s`` itself for the first block);
    ``t_b`` is the spine-exit vertex (the cut vertex where the spine
    leaves, or ``t`` for the last block).  Off-spine blocks (1-sum
    decorations) are not in the result.

    Returns the empty list when ``s == t`` (no spine — every block
    contributes its k = 0 scalar regardless).  When ``s`` and ``t``
    share a block, returns a one-element list ``[(block, s, t)]``.

    Implementation uses an auxiliary bipartite "block-cut graph": the
    blocks and the articulation points are nodes, with an edge
    whenever a cut vertex belongs to a block.  An s-t path on that
    tree alternates blocks and cut vertices; reading off the cuts on
    either side of each block gives ``(s_b, t_b)``.
    """
    if int(s) == int(t):
        return []

    s_blocks = [b for b in blocks_v if s in b]
    t_blocks = [b for b in blocks_v if t in b]
    common = set(s_blocks) & set(t_blocks)
    if common:
        return [(next(iter(common)), int(s), int(t))]

    bc = _nx.Graph()
    block_node: dict = {}
    for i, b in enumerate(blocks_v):
        node = ("block", i)
        block_node[b] = node
        bc.add_node(node)
    cut_node = {c: ("cut", c) for c in cuts}
    for node in cut_node.values():
        bc.add_node(node)
    for b in blocks_v:
        for v in b:
            if v in cut_node:
                bc.add_edge(block_node[b], cut_node[v])

    if s in cut_node:
        s_anchor = cut_node[s]
    else:
        s_anchor = block_node[s_blocks[0]]
    if t in cut_node:
        t_anchor = cut_node[t]
    else:
        t_anchor = block_node[t_blocks[0]]

    try:
        path = _nx.shortest_path(bc, s_anchor, t_anchor)
    except _nx.NetworkXNoPath:
        return []

    out: list = []
    for i, node in enumerate(path):
        if node[0] != "block":
            continue
        b = blocks_v[node[1]]
        if i == 0:
            s_b = int(s)
        else:
            prev = path[i - 1]
            s_b = int(prev[1]) if prev[0] == "cut" else int(s)
        if i == len(path) - 1:
            t_b = int(t)
        else:
            nxt = path[i + 1]
            t_b = int(nxt[1]) if nxt[0] == "cut" else int(t)
        out.append((b, s_b, t_b))
    return out


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

from gzl._errors import (  # noqa: E402  (re-exported; see _errors.py)
    GraphZetaError,
    UnsupportedLatticeSumError,
    UnsupportedRequestError,
)


class SelfLoopError(GraphZetaError):
    """The input edge list contains a self-loop ``(v, v)``.

    The topology-first router does not support self-loops; they would
    require a block that is a single vertex with a multi-edge to itself,
    which falls outside the bridge / cycle / algebra / tensor taxonomy.
    """


class DisconnectedGraphError(GraphZetaError):
    """The simple graph (after Hadamard-merging parallel edges) is disconnected.

    ζ_G factorises across connected components, but the router refuses to
    silently make that choice for the caller.  Evaluate each component
    separately.
    """


class VertexOutOfRangeError(GraphZetaError):
    """A vertex reference (``source`` / ``terminal``) or an edge label
    is not a vertex of the graph.  The vertex set is the edge support
    (see gzl/_labels.py): labels are non-negative integer NAMES,
    and a label that appears in no edge is not a vertex — there is no
    ``[0, V)`` range requirement any more.
    """


#: Deprecated alias, kept because callers may catch it by name.  The old
#: name asserted a divergence the code never checked and which is false
#: for the very example its docstring cited -- see
#: :class:`UnsupportedLatticeSumError`.
DivergentLatticeSumError = UnsupportedLatticeSumError


class NPointsRequiredError(GraphZetaError):
    """The request needs a grid ``n_points`` that it does not give.

    Raised when ``n_points`` is 0 and a block needs a grid (the algebra,
    the torus or the box), or a 1qp graph is asked for on the full
    Brillouin-zone grid, and when the ``n_points ** d`` sites of the
    torus cannot hold such a block, i.e. its vertices cannot be placed on
    them with the two ends of every edge on distinct sites.  The kernel
    vanishes at the origin, so every term of the sum vanishes on such a
    torus and the value is 0 up to round-off (the diamond on the chain
    returned 0.0 at ``n_points`` = 1 and 2, K4 1.5e-16 at 3).  Graphs of
    bridges and cycles evaluate in closed form and need no grid, and a
    block whose edges are all compact runs on its own exact grid.
    """


class TopologyEvaluatorUnavailableError(GraphZetaError):
    """A dependency of the topology-first router cannot be imported.

    ``networkx`` and ``epsteinlib`` are dependencies of gzl, so this
    means a broken installation, and reinstalling gzl repairs it.  In
    practice the missing package is ``networkx``: without ``epsteinlib``
    ``import gzl`` fails before the router exists.  The ``ImportError``
    is chained as ``__cause__``.  Nothing else raises this error; a
    request no evaluator serves raises :class:`UnsupportedRequestError`.
    """


def _torus_holds(sub, m: int) -> bool:
    """Whether the block ``sub`` can be placed on ``m`` torus sites with
    the two ends of every edge on distinct sites, i.e. coloured properly
    with ``m`` colours.

    The kernel vanishes at the origin, so every term of the torus sum
    in which an edge has both ends on one site is 0.  At ``k = 0`` the
    other terms are positive, so the torus value is 0, up to round-off,
    precisely when this is False.  Decided by a bipartite test for ``m = 2``, and
    otherwise by DSatur and, only where DSatur needs more than ``m``
    colours, an exact backtracking search, which the tiny grids that
    reach it keep small.
    """
    n = sub.number_of_nodes()
    if m >= n:
        return True
    if m < 2:
        return sub.number_of_edges() == 0
    if m == 2:
        return _nx.is_bipartite(sub)
    greedy = _nx.greedy_color(sub, strategy="DSATUR")
    if max(greedy.values(), default=-1) + 1 <= m:
        return True
    order = sorted(sub.nodes, key=lambda v: (-sub.degree(v), v))
    colour: dict = {}

    def place(i, used):
        if i == len(order):
            return True
        v = order[i]
        taken = {colour[u] for u in sub[v] if u in colour}
        # Colours are interchangeable: a vertex takes a colour already in
        # use, or the next new one.
        for c in range(min(used + 1, m)):
            if c not in taken:
                colour[v] = c
                if place(i + 1, max(used, c + 1)):
                    return True
                del colour[v]
        return False

    return place(0, 0)


def _require_grid(n_points: int, sub, d: int, *,
                  lifted: bool = False) -> None:
    """Refuse a grid a non-closed-form block cannot be evaluated on.

    See :class:`NPointsRequiredError`.  ``n_points = 0`` is refused for
    every block that needs a grid.  A positive ``n_points`` is refused
    when the ``n_points ** d`` sites of the torus cannot hold the block
    (:func:`_torus_holds`), where every term of its sum vanishes.  A
    coarse grid that can hold it gives a truncation, however poor, and is
    left to the caller.  ``lifted`` marks a block with a compact part,
    which runs at or above its winding-safe grid ``n_v R + 2`` whatever
    ``n_points`` is, so only ``n_points = 0`` is refused there.
    """
    n_v = sub.number_of_nodes()
    if n_points <= 0:
        raise NPointsRequiredError(
            "n_points > 0 is required to evaluate at least one "
            f"σ-routed block (this graph has a block of {n_v} vertices "
            "that is neither a bridge nor a cycle evaluated in closed "
            "form)."
        )
    if lifted or n_points ** d >= n_v or _torus_holds(sub, n_points ** d):
        return
    n_min = n_points + 1
    while n_min ** d < n_v and not _torus_holds(sub, n_min ** d):
        n_min += 1
    sites = n_points ** d
    raise NPointsRequiredError(
        f"n_points = {n_points} gives a torus of {sites} "
        f"site{'' if sites == 1 else 's'} at d = {d}, and a block of {n_v} "
        f"vertices cannot be placed on it with the two ends of every edge "
        f"on distinct sites.  Every term of its sum vanishes there, so the "
        f"value would be 0.  Use n_points >= {n_min}."
    )


# ---------------------------------------------------------------------------
# Corpus-wide per-block memoization
# ---------------------------------------------------------------------------
#
# Block-cut already factors ζ_G = Π ζ_block.  Across a corpus the same
# biconnected block (same topology, same per-bundle ν, same lattice /
# discretisation / momentum) recurs in thousands of distinct graphs —
# every bridge is the *same* epstein_zeta value, cycles collapse to one
# per (length, ν), and even the expensive σ-routed blocks show only a
# few dozen distinct signatures at order ≥ 9.  An optional cache keyed
# on a canonical block signature evaluates each distinct block once per
# corpus pass; the WL-hash is collision-guarded by an exact
# ν-/role-matched isomorphism check, so no non-equivalent block is ever
# conflated.  The reused value is that of an *isomorphic representative*
# block — at k = 0 the exact ζ_block is vertex-labelling-independent,
# so the representative and a fresh evaluation are approximations of
# the *same* exact number.  How much they differ depends on the
# per-block evaluator:
#
#   * bridges (epstein_zeta), simple cycles (zeta_circle), and the
#     tensor path (graph_zeta_general) are labelling-invariant up to
#     floating-point reassociation only — ~1e-13 relative, the same
#     magnitude as changing a summation order.
#   * the σ_max=4 algebra path (graph_from_edges → graph_zero, used
#     for σ < 1.49 tw-2 blocks) carries a *finite-n_points*
#     discretisation error whose value depends on the SP-reduction
#     terminal / ordering, which differs between a block's own
#     labelling and the representative's.  The two therefore differ
#     at the scale of that per-block discretisation residual (e.g.
#     ~1e-4 at n_points = 32, σ ≈ 1), not at the ULP level.  Both are
#     equally valid O(n^-(2-σ)) approximants of the identical exact
#     value, so series accuracy stays within the method's own error
#     bar — the cache merely selects which within-residual approximant
#     is used, it does not add error beyond the discretisation the
#     caller has already accepted.
#
# Either way the cached pass is fully deterministic and reproducible
# run to run.

_NU_KEY_DECIMALS = 12        # ν quantisation for the cache key (momentum is
                             # keyed exactly, see _momentum_cache_key)

# A simple cycle whose smallest per-bundle σ = ν − d exceeds this is routed
# through the truncated-Fourier tensor instead of the closed-form
# ``zeta_circle`` quadrature: at large σ the latter is both slow (its
# scipy quadrature cost grows with cycle length), while the tensor is
# exact and fast.  This auto-enables the ``fast_cycles`` behaviour for
# that block.  It is NOT the guard against the closed form's large-exponent
# overflow: that is driven by the LARGEST bundle exponent, while this gate
# keys on the smallest, and it is handled inside ``zeta_circle`` itself
# (at d = 1 a direct Epstein sum on retry, else CycleQuadratureError).
# Conditioning ceiling for the sigma_max=4 algebra at k = 0.  The
# evaluation's cancellation ratio kappa (see
# gzl.core.graph_zero_conditioned) bounds the achievable relative
# accuracy at ~eps*kappa; a block whose kappa exceeds this is refused and
# re-routed to the tensor path.  Measured separation is wide: healthy
# blocks top out at kappa ~ 6e10 (13-cycle at sigma = 0.05, error 2.5e-6,
# grid-limited), while the failures start at 1.2e14 (theta(12,12,1) at
# sigma = 0.2) and reach 8e16, where the algebra returns a NEGATIVE value
# for a positive lattice sum.  1e12 sits in that gap with 16x margin
# below the worst healthy case and 100x above the mildest failure.
#
# The mechanism is not the Gamma prefactor: extended-precision prefactors
# buy only 1.4x, and lowering sigma_max is 10-12 orders WORSE (it pushes
# nu ~ d terms into a barely-convergent real-space sum).  The loss is
# carried by the float64 coefficient and Epstein arrays themselves, so
# detect-and-reroute is the available remedy rather than repair.
#
# A kernel block with a compact TABLE has a second cancellation source:
# signed table entries that cancel inside the composed real-space
# window.  ``graph_zero_conditioned`` folds a kernel-built object's
# table magnitude (sum |a|, propagated through the compositions) into
# kappa as an upper bound, so such a block can exceed this ceiling and
# be re-routed too; the power-law path's kappa is byte-identical to
# before (measured: two cancelling tables of size 1e6 in series give a
# value 7e-3 wrong and kappa 8e14, refused; realistic J1 + J2-style
# tables on a sigma = 1.2 tail stay at kappa <= 2.3e3).
_ALGEBRA_MAX_CONDITION: float = 1e12

_FAST_CYCLE_SIGMA = 10.0


def _large_sigma_cycle(b_edges, d) -> bool:
    """True if this cycle block's smallest bundled σ = ν − d exceeds the
    ``_FAST_CYCLE_SIGMA`` threshold (so it should bypass ``zeta_circle``)."""
    return (min(float(e[2]) for e in b_edges) - float(d)) > _FAST_CYCLE_SIGMA


# Smallest torus a Richardson rung may use.  Below this the balanced
# label set is too coarse for the asymptotic tail model to hold.
_RICHARDSON_MIN_N = 8

# Smallest rung a d >= 2 ladder may use.  A d >= 2 torus is memory-bound,
# so its top rung is small (16 at d=2, 8-10 at d=3) and a floor of 16 left
# fewer than two usable rungs -- the ladder never ran.  Coarse rungs are
# safe now that rungs share parity; the four accuracy guards below still
# reject a fit that does not behave.
_RICHARDSON_MIN_N_HIGH_D = 4

# Ladder step ratio, by dimension.  The rungs have to stay inside the
# ASYMPTOTIC WINDOW -- the range of n over which the truncation really
# does behave as n^-sigma_eff.  That window has a floor set by the
# block, not by the dimension: measured on K_4 at d = 3, nu = 3.75, the
# torus sequence T(n) is still pre-asymptotic (it overshoots and comes
# back, so the successive differences change sign) below n ~ 8, and only
# from n >= 8 is it monotone with the modelled exponent.
#
# What differs by dimension is the *top* of the window: a d = 1 torus can
# afford n_top = 64-256, so stepping by 3/4 keeps four rungs above the
# floor, while a memory-bound d >= 2 torus tops out at n = 8-16 and the
# same 3/4 step lands the third rung at 6 or 4 -- below the floor and on
# the wrong side of the sign change.  The accuracy guards then (rightly)
# reject the fit, so the d >= 2 ladder stayed dead even after the rung
# floor was lowered.  A shallower step keeps every rung asymptotic.
#
# Measured on K_4 at d = 3, nu = 3.75, against a two-geometry reference
# (box and torus extrapolations agreeing to 2.2e-10 relative):
#     rungs [16,12,10,6] (3/4)   -> guard REJECTS, falls back to raw n=16
#     rungs [16,14,12]   (7/8)   -> 2.7e-9 rel, 61x better than raw n=16
#     rungs [12,10,8]    (7/8)   -> 4.4e-8 rel at ~1/3 the cost of
#                                   direct_sum's shipped L_list, which
#                                   gives 1.7e-7.
_RICHARDSON_RATIO = {1: 0.75}
_RICHARDSON_RATIO_HIGH_D = 0.875


class _RichardsonNotNeeded(Exception):
    """Internal: T(n) is already converged, skip the ladder."""


def _richardson_rungs_at(n_top: int, floor: int, ratio: float,
                         n_rungs: int) -> list:
    """Descending same-parity rungs for one fixed step ``ratio``."""
    out: list = []
    parity = int(n_top) % 2
    for jj in range(n_rungs):
        target = float(n_top) * (ratio ** jj)
        n_j = int(round(target))
        if n_j % 2 != parity:
            # Snap to the nearest rung of the SAME parity as n_top.  The
            # balanced-z label axis is symmetric for odd n and asymmetric
            # for even n (label n/2 has no partner -n/2), which splits the
            # truncation coefficient into two families; a ladder mixing
            # both fits a single power law through two different constants
            # and degrades the answer.  Ties go to the larger rung.
            lo, hi = n_j - 1, n_j + 1
            n_j = hi if abs(target - hi) <= abs(target - lo) else lo
        if n_j < floor:
            break
        if not out or n_j < out[-1]:
            out.append(n_j)
    return out


def _richardson_rungs(n_top: int, d: int = 1, n_rungs: int = 4) -> list:
    """Descending torus sizes for the extrapolation ladder.

    Geometric in ``n`` so the rungs probe distinct decades of the tail
    while costing a small fraction of the top rung (a rung at ``n/2``
    costs ``2^-(bag·d)`` of it).  Returns ``[n_top, …]``, descending
    and strictly decreasing, dropping anything below
    :data:`_RICHARDSON_MIN_N`.

    The step ratio is dimension-dependent (see
    :data:`_RICHARDSON_RATIO_HIGH_D`): at d >= 2 the affordable ``n_top``
    is small, so a 3/4 step would drop straight out of the asymptotic
    window and the fit would be rejected.

    A gentle ratio can however collide with the same-parity snapping and
    yield too *few* distinct rungs: at d = 3, ``n_top = 8`` the 7/8 step
    gives ``[8, 6]`` because the third target (6.125) snaps back onto 6.
    A two-rung ladder determines the fit exactly and leaves every
    acceptance guard vacuous (see :data:`_RICHARDSON_MIN_RUNGS`), so when
    the preferred ratio cannot reach :data:`_RICHARDSON_MIN_RUNGS` rungs
    we retry with tighter steps and keep the longest ladder found —
    ``[8, 6, 4]`` at d = 3, which is every rung the floor admits.
    """
    floor = _RICHARDSON_MIN_N if d <= 1 else _RICHARDSON_MIN_N_HIGH_D
    ratio = _RICHARDSON_RATIO.get(int(d), _RICHARDSON_RATIO_HIGH_D)
    best = _richardson_rungs_at(n_top, floor, ratio, n_rungs)
    if len(best) >= _RICHARDSON_MIN_RUNGS:
        return best
    for alt in (0.75, 0.625, 0.5):
        if alt >= ratio:
            continue
        cand = _richardson_rungs_at(n_top, floor, alt, n_rungs)
        if len(cand) > len(best):
            best = cand
        if len(best) >= _RICHARDSON_MIN_RUNGS:
            break
    return best


def _richardson_extrapolate(ns, vals, p, n_terms: int = 3):
    """Fit ``v(n) = v_inf + Σ_j a_j n^-(p+j)`` and return ``v_inf``.

    ``p`` is the derived torus-truncation exponent σ_eff (see the
    per-block derivation in :func:`_evaluate_via_topology`); the
    integer-spaced corrections are the usual Euler–Maclaurin tail.
    Returns ``None`` when the ladder is too short or the fit is not
    finite.
    """
    ns = np.asarray(ns, dtype=float)
    vals = np.asarray(vals, dtype=float)
    if ns.size < 2 or not np.all(np.isfinite(vals)):
        return None
    k = max(1, min(int(n_terms), ns.size - 1))
    design = np.column_stack(
        [np.ones(ns.size)] + [ns ** (-(p + j)) for j in range(k)]
    )
    try:
        sol, *_ = np.linalg.lstsq(design, vals, rcond=None)
    except np.linalg.LinAlgError:
        return None
    out = float(sol[0])
    return out if np.isfinite(out) else None


# --- Richardson ladder acceptance guards -------------------------------
#
# The ladder fit is always *exactly determined* (``_richardson_extrapolate``
# uses ``n_terms = len(rungs) - 1``), so it has no residual of its own to
# report and every guard has to be an external consistency test.
#
# ``_RICHARDSON_AMP_RATIO``
#     For a clean tail ``v(n) = v_inf - a n^-p`` the amplitude implied by
#     each consecutive rung pair,
#         a_j = (v_j - v_{j+1}) / (n_{j+1}^-p - n_j^-p),
#     is the *same* number ``a`` for every pair.  Drift in ``a_j`` measures
#     how far the ladder is from its asymptotic regime.  Replaces the old
#     "successive differences must strictly shrink" test, which compared
#     RAW differences and therefore failed whenever parity snapping made
#     the rung spacing uneven (rungs [32,24,18,14]: the 18->14 step spans
#     ratio 1.286 against 1.333 for the others, so at small p its
#     difference is legitimately the largest and the old test rejected a
#     perfectly good ladder).
#
# ``_RICHARDSON_MAX_CORRECTION``
#     Bound on the accepted correction as a multiple of the tail the
#     ladder itself predicts, ``|a_0| n_top^-p``.  The old form of this
#     guard bounded the correction by ``2.0 * last_gap`` instead.  That
#     constant is *not* a sanity bound: for a pure power law
#         correction / last_gap = 1 / ((n_top/n_1)^p - 1)
#     exactly, so the ratio is a deterministic function of p alone
#     (measured against the closed forms: 34.261 at p=0.1 vs 34.263
#     predicted, 13.409 vs 13.410 at p=0.25, 6.464 vs 6.464 at p=0.5,
#     4.152 vs 4.153 at p=0.75, 1.285 vs 1.286 at p=2).  A fixed cap of
#     2.0 therefore admits only p >~ 1.7 and rejected every slowly
#     converging tail -- exactly the regime the ladder exists to fix.  It
#     cost a measured 1e5-4e8x on the d=1 bridge at sigma <= 0.75.
#
# ``_RICHARDSON_STABILITY``
#     Refit without the lowest rung; the two extrapolants must agree to a
#     fraction of the correction being applied.  This is what actually
#     catches the coarse high-p ladders at d >= 2 where the bottom rung is
#     nowhere near asymptotic (a d=2 6-cycle at nu=d+2, n=16, rungs
#     [16,12,10,6]: the n=6 value is 7% off and the fit lands 2.6x worse
#     than the raw torus).  Costs one extra least-squares solve and no
#     extra engine evaluations.
#
# ``_RICHARDSON_CONVERGED``
#     Pre-check, before any rung is evaluated, that skips the ladder when
#     the predicted tail is already below what double precision can carry.
#
# ``_RICHARDSON_MIN_RUNGS``
#     Minimum ladder length to accept at all.  ``_richardson_extrapolate``
#     uses ``n_terms = len(rungs) - 1``, so a TWO-rung ladder determines
#     its two coefficients exactly: the implied amplitude has a single
#     entry (nothing to compare), the correction equals the predicted
#     tail *identically* (so the bounded-correction test can never fire),
#     and there is no rung to drop (so the stability test is skipped).
#     Every guard is then vacuous and the ladder is accepted
#     unconditionally -- measured on the d=3 traces, a 2-rung ladder
#     accepted 100% of cases and cost up to 18.8x (a 5-cycle at nu=d+0.75,
#     n=6, rungs [6,4]).  Three rungs is the shortest ladder that carries
#     any redundancy, and is what d=3 can afford at n=8-9.
_RICHARDSON_AMP_RATIO = 3.0
_RICHARDSON_MAX_CORRECTION = 4.0
_RICHARDSON_STABILITY = 0.1
_RICHARDSON_CONVERGED = 1e-13
_RICHARDSON_MIN_RUNGS = 3

#: Stability tolerance for a ladder whose LOWEST rung is at least
#: :data:`_RICHARDSON_STABILITY_RUNG`.
#:
#: WHY THE FLAT 0.1 IS WRONG, and it is a property of the test rather
#: than of the data.  On a synthetic sequence lying EXACTLY in the fitted
#: basis, ``v(n) = 1 - a n^-p - b n^-(p+1)`` on rungs [12, 10, 8] at
#: p = 11, the drop-one refit gap grows with the subleading amplitude
#: while the fit stays exact to 2.2e-16 throughout:
#:
#:     b/a      0    1      5      22     100    -> inf
#:     ratio  0.0007 0.0175 0.0677 0.1493 0.2063  0.2311 (ceiling)
#:
#: so 0.1 rejects any exactly-fitted two-term tail with ``b/a >~ 8``.
#: That is the regime a d = 3 ladder lives in, and it is why the guard
#: declined every K5 ladder measured and both of K4's affordable ones.
#:
#: MEASURED OVER 596 LADDERS (d = 1, 2, 3; K4, prism, K33, K5, K5-e,
#: V6E11 and bridges; sigma 0.25-2.0), classified against a slab
#: reference as good / mild / harmful by what accepting them costs
#: against the raw torus:
#:
#:     rule                       accept g/m/HARM   reject g/m/h   worst
#:     shipped, 0.10 flat            273/3/0        228/24/68      1.85x
#:     THIS, rung-conditional        342/5/0        159/22/68      1.98x
#:     0.50 flat, no condition       351/8/1        150/19/67     18.86x
#:     all guards off                501/27/68           --       70.18x
#:
#: +69 good accepts, ZERO harmful, worst accepted loss 1.98x against
#: 1.85x.  Per dimension: d = 1 174/0/0, d = 2 150/5/0, d = 3 18/0/0.
#:
#: THE RUNG CONDITION IS LOAD-BEARING.  Without it, 0.5 admits K4 at
#: d = 2, sigma = 1.0, n = 8 -- rungs [8, 6, 4], stability ratio 0.2214 --
#: at 18.86x WORSE than the raw torus.  A ladder that reaches below rung
#: 6 is not asymptotic and its drop-one refit is measuring that, not a
#: subleading amplitude.  Margin: the first harmful ladder this admits
#: only through stability sits at ratio 1.4900, 3.0x above 0.5.
#:
#: WHAT IT UNBLOCKS.  K4 at d = 3 is the exponent-2 core the graded
#: core would send to n = 14, where the shipped guard declines and the
#: raw value is 0.43x the box; accepted, the ladder is 13.82x the box.
#: At n = 12 it goes 0.15x -> 8.01x.  That is what makes ``_CORE_KAPPA``
#: safe at d = 3.
#:
#: NOT RELAXED: the amplitude sign test.  Rescuing K5 needs that removed
#: AND this above 0.5546, a window only 2.69x wide, and the sign-flip
#: population over the same 596 is 87 good against 15 bad with a worst
#: case of 69.26x.
_RICHARDSON_STABILITY_DEEP = 0.5

#: Lowest rung at or above which :data:`_RICHARDSON_STABILITY_DEEP`
#: applies.  Below it the tight :data:`_RICHARDSON_STABILITY` is kept.
_RICHARDSON_STABILITY_RUNG = 6


def _richardson_amplitudes(rungs, vals, p):
    """Tail amplitude implied by each consecutive rung pair.

    For ``v(n) = v_inf - a n^-p`` every entry equals ``a``; the spread
    across the ladder is the departure from the asymptotic regime.
    ``rungs`` is strictly decreasing, so the denominator is positive.
    """
    n = np.asarray(rungs, dtype=float)
    v = np.asarray(vals, dtype=float)
    return (v[:-1] - v[1:]) / (n[1:] ** (-p) - n[:-1] ** (-p))


def _richardson_ladder(eval_at, n_top, d, p, v_top):
    """Run the torus-size extrapolation ladder and apply the guards.

    ``eval_at(n)`` returns the block value on an ``n``-torus, ``v_top``
    is its already-computed value at ``n_top``, ``p`` the leading
    truncation exponent (σ_eff, from the cluster-cut rule) and ``d`` the
    lattice dimension.

    Returns ``(value, reason)``.  ``value`` is the extrapolated scalar
    when every guard passes and ``None`` otherwise; ``reason`` names the
    guard that declined, and is ``"ok"`` on acceptance.  Kept separate
    from :func:`_evaluate_via_topology` so the guards can be swept
    against exact closed forms without going through the router.
    """
    # Nothing to remove when the tail n^-p is already below what double
    # precision carries: a fit could only re-inject noise from the
    # coarse lower rungs.
    if float(n_top) ** (-p) < _RICHARDSON_CONVERGED:
        return None, "converged"
    rungs = _richardson_rungs(n_top, int(d))
    if len(rungs) < _RICHARDSON_MIN_RUNGS:
        # Fewer than three rungs leaves the fit exactly determined with
        # no redundancy for any guard to test -- see the note above.
        return None, "short_ladder"
    try:
        vals = [float(v_top)] + [float(eval_at(n_j)) for n_j in rungs[1:]]
    except Exception:
        return None, "error"
    v_inf = _richardson_extrapolate(rungs, vals, p)
    if v_inf is None:
        return None, "fit_failed"

    # (1) Tail-like: the implied amplitudes must keep one sign and must
    #     not drift by more than _RICHARDSON_AMP_RATIO across the ladder.
    amps = _richardson_amplitudes(rungs, vals, p)
    if amps.size >= 2:
        if not np.all(amps * amps[0] > 0.0):
            return None, "not_tail_like"
        mag = np.abs(amps)
        if mag.min() <= 0.0 or mag.max() > _RICHARDSON_AMP_RATIO * mag.min():
            return None, "not_tail_like"

    # (2) Bounded correction, measured against the tail the ladder itself
    #     predicts at the top rung rather than against a p-blind constant.
    correction = abs(v_inf - float(v_top))
    tail_pred = abs(amps[0]) * float(n_top) ** (-p)
    if correction > _RICHARDSON_MAX_CORRECTION * tail_pred:
        return None, "unbounded"

    # (3) Stability: the extrapolant must not depend on the lowest,
    #     least asymptotic rung.  The tolerance is RUNG-CONDITIONAL --
    #     see _RICHARDSON_STABILITY_DEEP for the 596-ladder confusion
    #     table.  A ladder reaching below rung 6 keeps the tight bound
    #     because its drop-one gap is measuring non-asymptoticity; one
    #     that does not is measuring a subleading amplitude, which the
    #     fit already carries.
    if len(rungs) >= 3:
        v_drop = _richardson_extrapolate(rungs[:-1], vals[:-1], p)
        if v_drop is None:
            return None, "fit_failed"
        _stab_tol = (_RICHARDSON_STABILITY
                     if min(rungs) < _RICHARDSON_STABILITY_RUNG
                     else _RICHARDSON_STABILITY_DEEP)
        if abs(v_inf - v_drop) > _stab_tol * correction:
            return None, "unstable"

    return float(v_inf), "ok"


def _momentum_cache_key(momentum):
    """Hashable, k=0-collapsing momentum descriptor for the block key.

    ``None`` and an EXACTLY zero vector (either sign) are both mapped to
    ``None`` because every block value is boundary- and
    momentum-independent at :math:`\\boldsymbol{k} = \\boldsymbol{0}`
    (the k=0 cell of the terminal-resolved tensor marginalises the
    boundary axes).  ``"grid"`` and any other vector keep their identity,
    and force the boundary-aware signature path.

    The zero test is ``hybrid._momentum_is_zero``, the one hybrid's
    vacuum rewrite uses, and it has no tolerance.  This key used to
    collapse ``max |k| <= 1e-15`` as well while hybrid rewrote only an
    exact zero, so ``0`` and ``1e-16`` were two truncations under one key
    and a shared cache served whichever came first.  Measured, k = 0
    against k = 1e-16 without a cache: a treewidth-2 corpus block on the
    chain at n = 512, 2.4e-05 (the exact zero ungraded as the vacuum
    problem, 1e-16 graded with the terminal kept; 1.6e-03 in an earlier
    version that graded both); K4 with a subdivided edge and the terminal
    on the subdivision vertex, square lattice at n = 32, 5.7e-04 (graded
    at both, vacuum rewrite against kept terminal); and an on-spine
    bridge at σ = 0.1, 3.0e-02 -- its closed form resolves the |k|^σ
    cusp, and at k = 1e-16 it was stored under the bridge key that every
    VACUUM graph with that exponent reads.  A key change only: no value
    evaluated without a cache moved.

    Every other momentum is keyed EXACTLY, by its float values, for the
    same reason one level up.  It used to be rounded to
    ``_NU_KEY_DECIMALS``, so momenta within ~5e-13 of each other shared
    an entry.  Where ζ is smooth that serves a neighbour's value, off by
    ζ'·δk (measured 2e-12 at k = 0.3).  At a reciprocal-lattice point
    (k integer in these fractional coordinates) the bridge closed form
    resolves the |k|^σ cusp, so rounding-equal momenta are genuinely
    different numbers there: at σ = 0.1 on the chain, 2e-15 then 1e-13
    was served 2.1e-02 off and 1 − 1e-16 then 1.0 was served 3.0e-02 off
    (6.5e-07 and 2.5e-08 at σ = 0.5).  All that rounding ever shared was
    bitwise-different momenta, which a corpus pass never produces: it
    parses its momentum once and hands every graph the same array
    (measured: identical hit counts and bit-identical coefficients).
    Signed zero components are one momentum here (``-0.0 == 0.0`` in
    the tuple), which is sound because every finite-k route evaluates
    them bit-identically (measured on square and triangular cells).
    """
    if momentum is None:
        return None
    if isinstance(momentum, str):           # "grid"
        return momentum
    arr = np.asarray(momentum, dtype=float).ravel()
    if _momentum_is_zero(arr):
        return None                          # exactly what hybrid rewrites
    return tuple(arr.tolist())


#: Memo for :func:`_block_signature`.  The WL hash is a pure function of
#: the labelled block, and a corpus pass asks for the same one many times
#: over — measured 41,302 calls over 7,264 distinct inputs at d = 1
#: n = 512 (5.7x), one input alone accounting for 16,863 of them.  Bounded
#: in entries: a value is a small networkx graph, not a grid.
_SIGNATURE_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_SIGNATURE_CACHE_MAX = 16384


def _bundle_key(x):
    """The cache-key token of a bundle: the rounded exponent for a float
    (the legacy key, byte-identical), or ``key()`` for an Interaction-like
    -- a tuple that never compares equal to a float.  A call carries
    either floats on every bundle or Interaction-likes on every bundle
    (``coerce_nu`` guarantees the uniformity), so the sorted key tuples
    below never compare a float with a tuple."""
    if is_interaction(x):
        _plain = x.as_plain_float()
        if _plain is not None:
            # the demotion the boundary promises: a plain power law in a
            # per-edge mix keys like the float it is, so a pure block shares
            # its cache entry with the same block in an all-float graph.
            # Read it through ``as_plain_float`` rather than ``x.nu[0]``:
            # a bundle of PARALLEL plain power laws arrives here as a
            # ``_KernelProduct``, which has no ``nu`` and carried the
            # Hadamard SUM the float path forms (``bundles[key] += nu_e``)
            # only on that method.  Indexing ``nu`` raised AttributeError
            # on exactly that bundle, and only when a ``block_cache`` was
            # supplied -- which every corpus pass supplies.
            return round(float(_plain), _NU_KEY_DECIMALS)
        return x.key()
    return round(float(x), _NU_KEY_DECIMALS)


def _sorted_keys(vals) -> tuple:
    """``sorted`` for bundle keys: floats or tuples alone sort as they always
    did; a mix (a plain power law beside a table in one cycle) sorts by
    type then repr, deterministically."""
    vals = list(vals)
    if all(isinstance(v, tuple) for v in vals) or not any(isinstance(v, tuple) for v in vals):
        return tuple(sorted(vals))
    return tuple(sorted(vals, key=lambda v: (isinstance(v, tuple), repr(v))))


def _signature_key(b_edges_local, role_map):
    """Exact (not isomorphism-invariant) identity of a signature request.

    This memoises the *derivation* of the cache key, so it must be exact:
    two inputs sharing this key produce the same ``(hash, graph)`` by
    construction.  Isomorphic-but-differently-labelled blocks get
    separate entries and are still collapsed downstream by the WL hash
    itself, which is what the block cache keys on.
    """
    return (
        tuple(sorted((int(u), int(v), _bundle_key(nu_b))
                     for u, v, nu_b in b_edges_local)),
        tuple(sorted((int(v), int(r)) for v, r in role_map.items())),
    )


def _block_signature(b_edges_local, role_map):
    """Canonical (WL-hash, verifier-graph) for a σ-routed block.

    ``b_edges_local`` is a list of ``(u, v, nu_bundle)`` on locally
    relabelled vertices; ``role_map`` assigns each local vertex an
    integer role (0 internal, 1 external/cut, 2 spine-source,
    3 spine-terminal) so two structurally identical blocks with
    different boundary roles never share a cache entry at finite k.

    Returns ``(wl_hash, G)`` where ``G`` carries the rounded ν as edge
    attribute ``w`` and the role as node attribute ``role``; ``G`` is
    kept for the exact ``is_isomorphic`` collision guard on a hash hit.
    """
    ckey = _signature_key(b_edges_local, role_map)
    hit = _SIGNATURE_CACHE.get(ckey)
    if hit is not None:
        _SIGNATURE_CACHE.move_to_end(ckey)
        return hit

    G = _nx.Graph()
    for v, r in role_map.items():
        G.add_node(int(v), role=int(r))
    for u, v, nu_b in b_edges_local:
        # Parallel edges are Hadamard-merged upstream, so the induced
        # block is simple; sum defensively if a duplicate slips through.
        w = _bundle_key(nu_b)
        if G.has_edge(int(u), int(v)):
            if is_interaction(nu_b) or not isinstance(G[int(u)][int(v)]["w"], float):
                # A general-kernel bundle is a lazy product; adding its
                # keys would be meaningless and the merge is done
                # upstream, so a duplicate here is a router bug.
                raise RuntimeError(
                    "_block_signature: duplicate bundle on a general-"
                    "kernel block; parallel edges are merged upstream."
                )
            G[int(u)][int(v)]["w"] = round(
                G[int(u)][int(v)]["w"] + w, _NU_KEY_DECIMALS
            )
        else:
            G.add_edge(int(u), int(v), w=w)
    h = _nx.weisfeiler_lehman_graph_hash(
        G, edge_attr="w", node_attr="role", iterations=4,
    )
    # ``G`` is read-only downstream (an ``is_isomorphic`` argument and a
    # stored reference), so one shared object may serve every hit.
    _SIGNATURE_CACHE[ckey] = (h, G)
    if len(_SIGNATURE_CACHE) > _SIGNATURE_CACHE_MAX:
        _SIGNATURE_CACHE.popitem(last=False)
    return h, G


def _cache_lookup(block_cache, primary_key, verify_G):
    """Return the cached block value or ``None``.

    ``block_cache`` maps ``primary_key`` → list of
    ``(verifier_graph_or_None, value)``.  ``verifier_graph`` is ``None``
    for closed-form bridge/cycle entries (their ``primary_key`` is
    already exact); for σ-routed entries it is the block graph and the
    WL-hash hit is confirmed with an exact ν-/role-matched
    ``is_isomorphic`` before reuse.
    """
    bucket = block_cache.get(primary_key)
    if not bucket:
        return None
    for vf, val in bucket:
        if vf is None or verify_G is None:
            if vf is None and verify_G is None:
                return val
            continue
        if vf is verify_G:
            # `_block_signature` memoises, so one shared graph object
            # serves every occurrence of a given labelled block: identity
            # is the common case here, and it implies isomorphism.  The
            # full check below stays for genuine WL-hash collisions and
            # for differently-labelled isomorphic blocks.
            return val
        if _nx.is_isomorphic(
            vf, verify_G,
            edge_match=lambda a, b: a.get("w") == b.get("w"),
            node_match=lambda a, b: a.get("role") == b.get("role"),
        ):
            return val
    return None


def _cache_store(block_cache, primary_key, verify_G, value):
    block_cache.setdefault(primary_key, []).append((verify_G, value))


# ---------------------------------------------------------------------------
# Per-block finite-k evaluator
# ---------------------------------------------------------------------------

def _reduce_to_cell(k_frac) -> np.ndarray:
    """``k_frac`` with every component outside ``[0, 1)`` shifted by the
    nearest integer, which is the same momentum.

    The closed forms take the momentum as ``inv(A.T) @ k_frac``, and on a
    non-orthogonal cell that product is not exactly a reciprocal-lattice
    vector for an integer ``k_frac``: on the triangular cell
    ``(1, 0)`` lands about 1e-17 off it, where epsteinlib evaluates the
    ``|k|^(nu - d)`` cusp instead of the value at ``k = 0`` (measured:
    relative errors up to 1e27 below ``nu = d``, 6.8e-9 at ``nu = 2.5``,
    and a finite number in place of the pole at ``nu = d``).  The
    subtraction is exact (Sterbenz), and a component already in
    ``[0, 1)`` -- every cell of a Brillouin-zone grid -- is left as it
    is, bit for bit.
    """
    k = np.array(k_frac, dtype=float)
    out = (k < 0.0) | (k >= 1.0)
    k[out] -= np.rint(k[out])
    return k


def _is_low_bridge(n_v, n_e, b_edges, b_kern, d) -> bool:
    """A power-law bridge whose finite exponent is ``<= d``."""
    if not (n_v == 2 and n_e == 1) or b_kern is not None:
        return False
    nu_b = float(b_edges[0][2])
    return bool(np.isfinite(nu_b)) and not (nu_b > d)


#: epsteinlib returns NaN within this distance of the pole ``nu = d``
#: (measured by bisection on every standard lattice: finite from
#: 9.3132e-10 = 2**-30 on).
_POLE_WINDOW = 2.0 ** -30


def _finite_bridge(value, nus, d):
    """The bridge closed form ``value``, refused where it is not finite.

    ``nus`` holds the exponents of the bridge's power-law terms: the
    bundle exponent, or every term of an Interaction's kernel (a zero-
    argument callable is accepted, and only called on failure).

    A bridge keeps the meromorphic continuation of its Epstein zeta
    function below ``nu = d``, but not the pole: ``nu = d`` at a momentum
    on the reciprocal lattice (``k = 0`` included).  epsteinlib also
    returns NaN within ``2**-30`` of the pole, where the value is finite
    but not evaluable, and a term of an Interaction's kernel can sit at
    the pole below its tail exponent.  Testing the VALUE catches all of
    these, where a test on ``nu`` would not; ``nus`` only decides how
    the failure is reported.  Far from the pole a non-finite value is
    epsteinlib's float64 range (it returns NaN on the unit chain from
    ``nu = 344`` on, where ``2 zeta(nu)`` is about 2), which is a
    ``GraphZetaError`` but not an unsupported exponent.  The check must
    run before the value is stored in a block cache, which would
    otherwise serve the NaN to later calls.
    """
    if np.all(np.isfinite(np.asarray(value, dtype=float))):
        return value
    nus = [float(x) for x in (nus() if callable(nus) else nus)]
    if any(abs(x - d) <= _POLE_WINDOW for x in nus):
        raise UnsupportedLatticeSumError(
            f"bridge with exponents {nus}: its Epstein zeta function is "
            f"not finite here.  nu = d = {int(d)} is its pole at a "
            f"momentum on the reciprocal lattice (k = 0 included), and it "
            f"cannot be evaluated within 2**-30 of the pole."
        )
    raise GraphZetaError(
        f"bridge with exponents {nus}: epsteinlib returned a non-finite "
        f"value away from the pole nu = d = {int(d)}, which is outside "
        f"the range it evaluates in float64."
    )


def _nn_block_finite_k(local_edges, s_alg, t_alg, A, n_internal,
                       n_points, momentum, kernels=None):
    r"""Nearest-neighbour on-spine block at finite momentum.

    With the NN-indicator kernel ζ_block(k) is an *exact finite
    trigonometric polynomial*, so we compute the real-space form factor
    ``M(x_t) = Σ_internal Π K_NN`` exactly on a torus just large enough to
    hold its finite support (``n_internal = n_v + 2``, no wraparound),
    then evaluate ``ζ(k) = Σ_{x_t} M(x_t) e^{2πi x_t·k}`` at the requested
    momenta.  The internal computation carries no discretisation error;
    ``n_points`` sets only the output BZ-grid resolution.
    """
    d = int(A.shape[0])
    if kernels is None:
        nu_vec = np.full(len(local_edges), np.inf)
        M = np.asarray(
            graph_zeta_general(
                local_edges, nu_vec, A, n_internal,
                source=s_alg, terminals=(t_alg,), space='z',
            )
        ).reshape((n_internal,) * d)
    else:
        # A purely compact interaction (tail +inf on some edge): the same
        # exact form factor, on a torus sized to the block's support.
        nu_vec = np.asarray([k.tail_exponent for k in kernels], dtype=float)
        M = np.asarray(
            graph_zeta_general(
                local_edges, nu_vec, A, n_internal,
                source=s_alg, terminals=(t_alg,), space='z',
                kernels=list(kernels),
            )
        ).reshape((n_internal,) * d)
    z = _balanced_z_axis(n_internal).astype(float)   # label → balanced integer

    if isinstance(momentum, str) and momentum == "grid":
        n = int(n_points)
        j = np.arange(n, dtype=float)
        # P[j, idx] = exp(2πi z[idx] · j/n): the exact trig-polynomial
        # sampling matrix; output resolution n is independent of n_internal.
        P = np.exp(2j * np.pi * np.outer(j, z) / n)          # (n, n_internal)
        out = M.astype(complex)
        for _ in range(d):                                   # one axis per dim
            out = np.tensordot(out, P, axes=([0], [1]))
        return out.real.astype(float)                        # (n,) * d

    k_frac = np.asarray(momentum, dtype=float).reshape(d)
    out = M.astype(complex)
    for a in range(d):
        out = np.tensordot(out, np.exp(2j * np.pi * z * k_frac[a]),
                           axes=([0], [0]))
    return float(np.asarray(out).real)


def _block_at_finite_k(
    sub,
    b_set: set,
    b_edges: list,
    bundle_orig_nus: dict,
    s_b: int,
    t_b: int,
    A: np.ndarray,
    n_points: int,
    momentum,
    nn_mode: bool = False,
    engine: str = "hybrid",
    dense_engine: "str | None" = None,
    sp_n_points: "int | None" = None,
    core_grading: bool = True,
    sigma_ref: "float | None" = None,
    info: "dict | None" = None,
    accuracy: str = "strict",
    b_kern=None,
):
    """Evaluate one on-path block at the given external momentum.

    ``b_kern`` (general interactions) is the per-bundle list of
    Interaction-likes aligned with ``b_edges``, or ``None`` on the legacy
    power-law path; ``b_edges`` keeps carrying the float tail exponents
    every routing decision reads.

    ``sigma_ref`` is the block's core-grading floor
    (:func:`_pass_floor_sigma` of its own per-edge exponents), which the
    router computes once per block for the cache key and passes here.
    ``None`` leaves the core sized against the block's own rate, the
    legacy behaviour.

    ``momentum`` is either an ``ndarray`` of shape ``(d,)`` (single-k,
    returns ``float``) or the string ``"grid"`` (full BZ grid, returns
    ``np.ndarray`` of shape ``(n_points,) ** d``, real).

    The block has spine endpoints ``(s_b, t_b)`` (the cut vertices
    where the spine enters / leaves this block).  All other vertices
    of the block — including any *off-spine* cut vertices that
    happen to lie inside the block — are eliminated as internal
    vertices, since they carry zero momentum at the full-graph level
    by the 1-sum-attachment theorem.

    ``info``, when given, is the caller's diagnostics dict and is
    updated by whichever branch actually runs.  The caller cannot infer
    this from the block's shape: it used to tally on-spine blocks by
    guessing from ``n_v``/``n_e``, which billed every finite-k
    direct-sum evaluation as ``n_block_tensor`` and never touched
    ``n_block_direct_sum`` or ``n_block_algebra`` at all.  Any
    before/after built on ``return_diagnostics`` was wrong for on-spine
    blocks as a result.
    """
    def _bump(key, by=1):
        if info is not None:
            info[key] = info.get(key, 0) + by

    def _note_tw(tw):
        if info is not None and tw > info.get("max_tensor_block_tw", 0):
            info["max_tensor_block_tw"] = int(tw)
    n_v, n_e = len(b_set), len(b_edges)
    d = int(A.shape[0])
    block_orig_nus = []
    for (u, v) in sub.edges():
        key = (u, v) if u < v else (v, u)
        block_orig_nus.extend(bundle_orig_nus[key])
    sigma_block_min = min(block_orig_nus) - float(d)
    use_algebra_low_sigma = sigma_block_min < (1.49 - 1e-12)
    SIGMA_MAX_ALG = 4.0

    is_grid = isinstance(momentum, str) and momentum == "grid"
    if not is_grid:
        k_frac = np.asarray(momentum, dtype=float).reshape(d)
    has_compact_block = (b_kern is not None
                         and any(k.has_compact for k in b_kern))
    kern_kw = {} if b_kern is None else {"kernels": list(b_kern)}
    # (n_block_mixed_kernel is counted per block by the caller.)
    # THE WINDING-SAFE GRID of a block with a compact part.  Below
    # n_v * R + 2 the all-table term of a cycle of up to n_v edges winds
    # around the torus and contributes a spurious O(a^E) term (measured:
    # the NN indicator on C4 at n = 4 is 33 % high; a mixed triangle at
    # n = 3 carries a spurious J^3), so every torus grid a compact block
    # runs on -- the pass grid, the pinned tier, the pair rungs, the form
    # factor -- is floored at it.  The algebra route composes tables by
    # cyclic convolution and needs n > sum of the radii instead.
    _R_fk = (max(int(k.support_radius) for k in b_kern)
             if has_compact_block else 0)
    _lift_fk = (n_v * max(_R_fk, 1) + 2) if has_compact_block else 0
    _sumR_fk = (sum(int(k.support_radius) for k in b_kern if k.has_compact)
                if has_compact_block else 0)

    # ------------------------------------------------------------------
    # Nearest-neighbour (ν = inf): the closed-form / direct-sum
    # evaluators are undefined at ν = inf, so every on-spine block routes
    # to the exact NN form-factor path; the dispersion is exact at every
    # output momentum.  A purely compact interaction carries the same
    # tail (+inf) and the same exactness argument.  The internal torus
    # is the block's own smallest exact grid (``_compact_exact_grid``:
    # the longest fundamental cycle of its best spanning tree times R,
    # or twice the source-terminal distance times R, plus one) -- not
    # the winding-safe ``n_v R + 2`` the mixed-kernel floors use, which
    # is the same lifting argument with the worst tree.  Measured on the
    # 1qp corpus at R = 2: the grids 12-20 become 7-13, and the dense
    # contraction pays the ratio to the power of its exponent.
    # ------------------------------------------------------------------
    if nn_mode:
        sorted_v = sorted(b_set)
        local = {v: i for i, v in enumerate(sorted_v)}
        local_edges = np.array(
            [[local[u], local[v]] for u, v, _ in b_edges], dtype=int,
        )
        _bump("n_block_tensor")
        _bump("n_block_nn")
        _R = (max(k.support_radius for k in b_kern) if b_kern is not None
              else 1)
        return _nn_block_finite_k(
            local_edges, local[s_b], local[t_b], A,
            _compact_exact_grid(local_edges, _R, local[s_b], local[t_b]),
            n_points, momentum, **kern_kw,
        )

    # ------------------------------------------------------------------
    # Bridge (closed-form)
    # ------------------------------------------------------------------
    # A block that is PART nearest-neighbour and part power law has no
    # evaluator at finite momentum.  The arm above treats every edge as
    # the NN indicator, so before this guard such a block silently
    # returned its all-``inf`` value -- a mixed [inf, 2.5] path graph gave
    # exactly the dispersion of the all-NN one -- while the routes below
    # are undefined at ``nu = inf``.  Refuse instead of choosing one of
    # two wrong answers.  The refusal is of the request, so it is an
    # UnsupportedRequestError, a GraphZetaError:
    # TopologyEvaluatorUnavailableError, which this used to raise, means
    # that a dependency is missing.
    if b_kern is None and not nn_mode:
        _nu_b_all = [float(nu_b) for _, _, nu_b in b_edges]
        if any(np.isinf(v) for v in _nu_b_all):
            raise UnsupportedRequestError(
                "no finite-momentum evaluator for a block that mixes "
                f"nu = inf with finite exponents (this block: {_nu_b_all}). "
                "The nearest-neighbour form factor treats every edge as the "
                "NN indicator and the closed forms are undefined at nu = inf. "
                "Such a graph is evaluated as a vacuum graph (no terminal), "
                "and at any momentum once the nearest-neighbour edge is "
                "expressed as an Interaction "
                "(Interaction.nearest_neighbour(A)) and passed through the "
                "kernels channel, which carries a finite tail per edge."
            )

    if n_v == 2 and n_e == 1:
        _bump("n_bridges")
        if b_kern is not None:
            # General interaction: the exact lattice sum of the bundle
            # kernel at k (Epstein zetas for the power-law terms plus the
            # finite trigonometric sum of the compact part).
            _nus = lambda: [nu for _, nu in b_kern[0].fourier_terms(A)[0]]  # noqa: E731
            if not is_grid:
                return float(_finite_bridge(
                    b_kern[0].lattice_sum(A, k_frac), _nus, d))
            n = int(n_points)
            axes = [np.arange(n, dtype=float) / n for _ in range(d)]
            kk = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d)
            return _finite_bridge(
                np.asarray(b_kern[0].lattice_sum(A, kk), dtype=float),
                _nus, d).reshape((n,) * d)
        nu_b = float(b_edges[0][2])
        Astar = np.linalg.inv(A.T)
        zero = np.zeros(d, dtype=float)
        if not is_grid:
            # _epstein_zeta_k: epsteinlib's value, with the cusp it drops
            # below |k| ~ 1e-32 added back.  The grid below never gets
            # that close to 0 (its nodes are j / n_points).
            k_lat = Astar @ _reduce_to_cell(k_frac)
            return float(_finite_bridge(
                _epstein_zeta_k(nu_b, A, k_lat).real, [nu_b], d))
        # Grid: the closed form at every BZ-grid node, one node per
        # INVERSION orbit.  ζ(-k) = ζ(k) on any Bravais lattice (the
        # kernel is even), and the unsigned node grid j ∈ [0, n)^d is
        # closed under j -> -j mod n at every n -- unlike the balanced
        # position window that bit the slab at even n -- so the partner
        # of each evaluated node copies it.  epsteinlib is scalar-only,
        # and these grids were the cost once the cores were graded: at
        # d = 2, n = 512 the five on-spine bridge ν of the order-9 1qp
        # pass cost 1.31e6 calls at ~15 µs, 21.3 s of a 33 s pass.
        # This halves the calls on every cell; the full point group
        # would give 8x on the square lattice but only 2x on a sheared
        # one, for machinery a stress-test grid does not justify.
        #
        # The evaluated node is computed EXACTLY as before (same j / n
        # floats, same call), so those values are bit-identical to the
        # per-node loop.  A copied partner is the same closed form at
        # the mirrored argument: ζ(k) = ζ(-k) exactly, and epsteinlib
        # returns the two to round-off -- measured over every census
        # cell at both parities, |ζ(k) - ζ(-k)| <= 1.1e-14 absolute, at
        # most 4.1 ulp of max|ζ| on the grid.  (A RELATIVE deviation is
        # large only where ζ itself nearly vanishes: 5e-12 at a node
        # with |ζ| = 1.6e-3 is that same 8e-15.)  Reproduced with
        # epsteinlib alone in tests/test_bridge_grid_inversion.py.  That
        # round-off is the only value change the inversion fold makes.
        n = int(n_points)
        j = np.stack(
            np.meshgrid(*[np.arange(n)] * d, indexing="ij"), axis=-1,
        ).reshape(-1, d)
        mirror = np.ravel_multi_index(tuple(((-j) % n).T), (n,) * d)
        flat = np.arange(j.shape[0])
        reps = flat[flat <= mirror]           # one node per orbit; fixed points once
        kk = j[reps].astype(float) / n
        out = np.empty(j.shape[0], dtype=float)
        for i, r in enumerate(reps):
            out[r] = float(_epstein_zeta(nu_b, A, zero, Astar @ kk[i]).real)
        out[mirror[reps]] = out[reps]
        return _finite_bridge(out, [nu_b], d).reshape((n,) * d)

    # ------------------------------------------------------------------
    # A MIXED block with a compact part (finite tail) takes the SAME
    # dispatch as a power-law block below -- the dense refusal, the
    # accuracy gate, the pinned tier, the pair, the box, the algebra and
    # the split -- with two changes only: every torus grid is floored at
    # the winding-safe grid ``_lift_fk`` (a grid the pass cannot hold
    # goes through the exact form factor at that grid), and a graded
    # core is floored at ``_compact_core_floor`` so the residual tables
    # stay exact on it (see the grading site below; the measurement that
    # once declined it -- a core of 6 on a lift-6 block, 20x worse than
    # the raw grid -- sat below ``_CORE_N_FLOOR``, where the rule never
    # grades).  An earlier version returned the form factor for every
    # compact block ABOVE the dense refusal, so a tail <= d on a dense
    # spine block returned a truncation artefact at finite k while
    # raising at k = 0.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Simple cycle on-path: zeta_circle is k = 0 only — fall through to
    # the algebra / tensor path below by NOT taking the closed-form
    # branch.  This degrades to the σ-routed evaluator at finite k.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # σ-routed path: algebra (SP-reducible at the spine endpoints) or
    # tensor.  Every route below needs a grid, as at k = 0; without this
    # a single-k request at n_points = 0 failed inside an engine
    # ("cannot reshape array of size 1 into shape (0,)").
    # ------------------------------------------------------------------
    _require_grid(int(n_points), sub, d, lifted=has_compact_block)
    sorted_v = sorted(b_set)
    local = {v: i for i, v in enumerate(sorted_v)}
    s_alg = local[s_b]
    t_alg = local[t_b]
    local_edges = np.array(
        [[local[u], local[v]] for u, v, _ in b_edges], dtype=int,
    )
    local_nu_vec = np.array(
        [float(nu_b) for _, _, nu_b in b_edges], dtype=float,
    )

    from networkx.algorithms.approximation import treewidth_min_fill_in
    tw_block, _ = treewidth_min_fill_in(sub)

    # (3d) Dense on-spine block at finite k.  The tensor path costs
    # n^((tw+1+N_t)·d) — n^12 for a K_5 at d=2 with one terminal,
    # infeasible.  Route to the real-space direct sum with the source
    # pinned and a real cos(2π k·x_terminal) phase (ζ_G(k) is real by
    # inversion symmetry).  The same high connectivity that makes tw
    # large makes the |x|^-(deg·ν) box tail steep, so a tiny L reaches
    # series precision after the degree-aware Richardson exponent (3a).
    # Single-k: one phased extrapolation.  Grid: a genuine *single
    # shot* — the k-independent elimination is run once per L with the
    # terminal kept open (→ M_L(x_t)), then the whole BZ is one cosine
    # transform `Σ_x cos(2π k·x) M_L(x)`, Richardson-combined over L.
    # This is the direct-sum analogue of the tensor `space='z'`→ifftn
    # path (len(L_list) eliminations + one transform, NOT one
    # elimination per k-point), so the headline single-shot-BZ
    # guarantee holds for dense on-spine blocks at any d.  Falls back
    # to the tensor path on any failure.
    #
    # The route above is now a choice, not a law.  ``"torus"``
    # skips this arm entirely and lets the block reach ``_block_general``
    # below — which at finite k means hybrid (with the split, when a
    # fine grid is configured) rather than the box.  Note the k = 0
    # Richardson ladder is NOT part of that: at finite k the tail carries
    # a cos(2π k·x) factor and stops being a clean power law, so the
    # σ_eff basis does not describe it and the ladder is deliberately
    # absent on this path.  The split is therefore the only accuracy
    # lever here.
    # ``dense_engine`` / ``sp_n_points`` arrive ALREADY RESOLVED from
    # ``_evaluate_via_topology`` — do not re-run ``_resolve_dense_routing``
    # on them.  ``sp_n_points=None`` is resolved output meaning "no
    # split"; re-resolving would read it as "unset" and reinstate the
    # per-d default, silently turning the split on where the caller had
    # switched it off.  ``None`` for the engine means the historical
    # route, so a direct test caller gets today's behaviour.
    dense_route = "direct_sum" if dense_engine is None else str(dense_engine)
    dense_sp_n = sp_n_points
    # ν ≤ d on a dense block is refused OUTRIGHT, below, on either engine
    # setting -- the raise happens before the box branch is consulted, so
    # the ``ν_min > d`` conjunct here is belt-and-braces rather than the
    # thing that keeps the guard alive.  (It is kept deliberately: it costs
    # nothing and it means reordering the raise cannot silently open the
    # torus to a case neither engine can judge.)
    #
    # This is a SUPPORT limit, not a divergence verdict -- see
    # ``UnsupportedLatticeSumError``.  Unreachable from TFIM
    # (ν = d + σ with σ > 0), which is exactly why it must be explicit.
    dense_is_torus = (
        dense_route == "torus"
        and float(local_nu_vec.min()) > float(d)
    )
    # The accuracy gate, same table as the k = 0 arm.  ``ladder=False``
    # unconditionally here, so a gated class does not ride the torus at
    # the pass grid on this path.  That is not bluntness: the k = 0
    # sigma_eff ladder is deliberately absent at finite k (the tail
    # carries a cos(2 pi k.x) factor and stops being a clean power law),
    # and the table's one entry is earned partly BY that ladder --
    # V6E11's raw value at the threshold grid n = 12 is still 2.3x worse
    # than the box, and only the accepted ladder takes it to 8.3x
    # better.  A declined block falls to the FINITE-K SECOND PIN below
    # when its class is measured raw-clearing, else to the box.
    if dense_is_torus and tw_block >= 3:
        if not _dense_torus_is_accurate(
            local_edges, list(range(n_v)), int(s_alg), (int(t_alg),),
            int(d), int(n_points), False,
        ):
            dense_is_torus = False
            _bump("n_block_dense_torus_declined")
    if tw_block >= 3 and float(local_nu_vec.min()) <= float(d):
        raise UnsupportedLatticeSumError(
            f"dense block (treewidth {int(tw_block)}) has min nu = "
            f"{float(local_nu_vec.min())} <= d = {int(d)}, which is not "
            f"supported: some such sums diverge and this engine cannot "
            f"tell which, while a torus would return a truncation "
            f"artefact rather than a value.  (Convergence is governed by "
            f"the cluster cut, not by min nu, so this refusal is "
            f"conservative -- it declines cases that do converge.)"
        )
    # THE FINITE-K SECOND PIN.  A gate-declined block whose class is
    # measured raw-clearing rides M(z) via the two-pin cells -- the
    # same torus truncation the k = 0 entry licenses, made affordable
    # at finite k by the second pin -- as a PER-K TWO-RUNG PAIR at the
    # class-keyed rungs of ``_FK_PAIR_BY_P`` with the exact basis
    # ``p = min_free_cut_nu - d`` at the block's own source pin.  The
    # rungs follow the BALANCED policy (measured table at the
    # constant): the cheapest pair that stays at least box-accurate,
    # so no single block dominates a pass -- p >= 10 ships (8, 10)
    # (4x cheaper, noise-level loss only at exactly k = 0), while
    # 7 <= p < 10 keeps (10, 12) because every cheaper scheme fails
    # that class (69/512 points up to 9.5x the box).  The finite-k box
    # grid degrades off k = 0 while the pin stays uniform, which is
    # why even the lower rung alone clears it away from k = 0.  The
    # eligibility floor p >= 7 is where raw-clearing is measured
    # (K5 p = 10..17, K5-e p = 7.5, V6E11 p = 7 at nu = 5); below it
    # the class is LADDER-borne at k = 0 (V6E11 at production nu) with
    # no finite-k extrapolation measured, so it keeps the box.  The
    # per-k pair correction is guarded by ``_FK_PAIR_BAND_MAX``: a
    # correction beyond it means the pair is extrapolating outside its
    # measured regime, and the block falls back to the box.
    # Cell-symmetry and byte eligibility mirror the k = 0 slab arm;
    # any refusal falls through to the box.
    if (tw_block >= 3 and not dense_is_torus
            and float(local_nu_vec.min()) > float(d)):
        _emap: dict = {}
        for (_u, _v), _nu_b in zip(local_edges.tolist(), local_nu_vec):
            _kk = (min(_u, _v), max(_u, _v))
            _emap[_kk] = _emap.get(_kk, 0.0) + float(_nu_b)
        # The basis excludes BOTH pins: the outer z-sum over the
        # terminal is exact on the torus, so the terminal is not a
        # truncation-error source -- the decay is governed by the min
        # cut over the vertices that stay FREE inside a cell.  On the
        # corpus emblem K5-e with the hop across the missing edge
        # (both deg-3 vertices pinned) this is p = 11, not the
        # source-only 7.5: measured 4.6e-09 median raw-12 in that
        # sector vs 8.0e-07 in the (deg-3 free) sector, which is the
        # sector the measurement at _FK_PAIR_BY_P licensed.  Source-only
        # p under-weights the pair and mis-classes such blocks into rungs
        # they do not need.
        _p = float(_elimination.min_free_cut_nu(
            _emap, int(s_alg), (int(t_alg),))) - d
        _rungs = next((r for _pmin, r in _FK_PAIR_BY_P if _p >= _pmin),
                      None)
        if not np.isfinite(_p):
            _rungs = None                 # every free cut through an inf bundle
        if _rungs is not None and int(_rungs[0]) < _lift_fk:
            # The class pair's lower rung would wind the block's tables:
            # lift the pair (same spacing, same parity) so the pinned
            # two-rung Richardson still runs at the exact basis instead of
            # falling to a box ladder that cannot hold the compact reach.
            _lo = int(_lift_fk) + (int(_lift_fk) + int(_rungs[0])) % 2
            _rungs = (_lo, _lo + int(_rungs[1]) - int(_rungs[0]))
        # FLOOR-MATCHED GRID TIER.  A grid request evaluates every
        # block of the pass on the same n_points torus, so the pass's
        # accuracy floor is the ordinary blocks' raw truncation at
        # that n -- measured on the order-9 census neighbours at n = 8:
        # e2 (5,8) median 1.4e-05 rel, e2 K4 median 3.3e-05 (vs the
        # n = 16 grid).  The declined block PINNED at the same n sits
        # an order of magnitude below that floor (K5-e corpus emblem:
        # median 1.3e-06, max 2.5e-05, k = 0 2.5e-05 -- level with
        # the neighbours' k = 0), so precision beyond raw-at-n is
        # invisible in the summed coefficient, and the block stops
        # being special: ~1 s like its neighbours.  Above the class
        # lo rung, raw-at-n costs more than the pair while the pair
        # is already below the (smaller) floor at that n, so the
        # pair takes over.  Single-k requests have no neighbouring
        # floor and keep the box-dominance standard (the class pair)
        # -- which is also what the frozen reference keys pin.
        # FLOOR MODE widens the tier to every request shape and every
        # p (the p >= 7 predicate is a BOX-dominance statement; in a
        # pass the standard is the pass floor, which the pinned
        # truncation meets by construction -- same family, same n,
        # basis at or above the neighbours' own cut class.  Measured on
        # the order-11 population that motivated it: ordinary p = 4 e2
        # blocks raw-8 med 2.3e-03/5.6e-03 vs their own n = 16 grids;
        # the box-bound (7,11) p = 4 sector pinned-8 med 2.7e-03 --
        # level -- at 1.6 s against an hours-class box grid).
        if accuracy == "floor":
            _floor_n = int(n_points) if int(n_points) > 0 else None
            _eligible = _floor_n is not None
        else:
            _floor_n = (int(n_points)
                        if (_rungs is not None and is_grid
                            and int(n_points) <= _rungs[0]) else None)
            _eligible = _rungs is not None
        if _floor_n is not None:
            # A compact block's pinned grid is floored at its winding-safe
            # grid; the fold is exact with tables (the orbit group is
            # filtered by table invariance), so the tier's licence holds.
            if _lift_fk > _floor_n:
                _bump("n_block_compact_lift")
            _floor_n = max(_floor_n, _lift_fk)
        if (_eligible
                and len(_slab_window_group(
                    A, _floor_n if _floor_n is not None else _rungs[1]))
                >= _SLAB_MIN_GROUP.get(int(d), 10 ** 9)):
            _lo, _hi = _rungs if _rungs is not None else (0, 0)
            try:
                if is_grid:
                    n = int(n_points)
                    axes = [np.arange(n, dtype=float) / n for _ in range(d)]
                    _ks = np.stack(
                        np.meshgrid(*axes, indexing="ij"), axis=-1,
                    ).reshape(-1, d)
                else:
                    _ks = np.asarray(k_frac, dtype=float).reshape(1, d)
                if _floor_n is not None:
                    # THE OUTER PIN IS PRICED, NOT FIXED.  The k-phase
                    # attaches to the terminal, but the ENUMERATED
                    # vertex of the second pin is an association
                    # choice: pin any other vertex as the outer sum
                    # and keep the terminal as the open axis, and the
                    # value is the SAME torus sum exactly reassociated
                    # (measured <= 8.8e-15 vs the t-outer route across
                    # every cell family at even n, with an independent
                    # graph_zeta_general(space='z') anchor).  A profile
                    # of the orders <= 11 floor pass put 87% of its
                    # 795.7 s into 27 pinned calls; standalone timings
                    # (on a contended machine) attributed the bulk to
                    # two sectors whose t-outer schedule prices inner
                    # exponent 3, and with the priced outer both ship in
                    # ~1-1.5 s through the router (795.7 s -> 141.5 s).
                    # Ties keep the t-outer route bit-for-bit.  This tier
                    # is shared by floor mode and STRICT grid requests at
                    # n_points <= the class lo rung, so strict-grid values
                    # on lever sectors move at reassociation round-off --
                    # no frozen key covers that tier (all router_* keys
                    # are vacuum or single-k strict), verified by the
                    # slow gate.
                    _, _bu = _fk_outer_best(
                        local_edges, int(s_alg), int(t_alg))
                    if _bu is not None:
                        vals, _ = _slab_zeta_finite_k_outer(
                            local_edges, local_nu_vec, A, _floor_n,
                            source=int(s_alg), terminal=int(t_alg),
                            outer=int(_bu),
                            momentum=_ks,
                            max_bytes=_DENSE_CORE_MAX_BYTES, **kern_kw)
                        _bump("n_block_fk_outer_pin")
                    else:
                        vals, _ = _slab_zeta_finite_k(
                            local_edges, local_nu_vec, A, _floor_n,
                            source=int(s_alg), terminal=int(t_alg),
                            momentum=_ks,
                            max_bytes=_DENSE_CORE_MAX_BYTES, **kern_kw)
                    _bump("n_block_slab_fk_floor")
                    _note_tw(tw_block)
                    if is_grid:
                        return np.asarray(vals, dtype=float).reshape(
                            (int(n_points),) * d)
                    return float(np.asarray(vals).ravel()[0])
                v_lo, _ = _slab_zeta_finite_k(
                    local_edges, local_nu_vec, A, _lo,
                    source=int(s_alg), terminal=int(t_alg),
                    momentum=_ks, max_bytes=_DENSE_CORE_MAX_BYTES, **kern_kw)
                v_hi, _ = _slab_zeta_finite_k(
                    local_edges, local_nu_vec, A, _hi,
                    source=int(s_alg), terminal=int(t_alg),
                    momentum=_ks, max_bytes=_DENSE_CORE_MAX_BYTES, **kern_kw)
                # Per-k one-term Richardson at the exact basis p: the
                # closed form of _richardson_extrapolate for two rungs,
                # vectorised over the k-set.
                w_lo = float(_lo) ** (-_p)
                w_hi = float(_hi) ** (-_p)
                vals = (v_hi * w_lo - v_lo * w_hi) / (w_lo - w_hi)
                corr = np.abs(vals - v_hi)
                scale = np.maximum(np.abs(vals), 1e-300)
                if float(np.max(corr / scale)) <= _FK_PAIR_BAND_MAX:
                    _bump("n_block_slab_finite_k")
                    _note_tw(tw_block)
                    if is_grid:
                        return np.asarray(vals, dtype=float).reshape(
                            (int(n_points),) * d)
                    return float(vals[0])
                _bump("n_block_fk_pair_band_declined")
            except (ValueError, MemoryError):
                pass                       # any refusal keeps the box

    _box_refusal = None
    if tw_block >= 3 and not dense_is_torus:
        L_list = _DENSE_DSUM_L_LIST.get(int(d), (2, 3, 4))
        K_corr = min(3, len(L_list) - 1)
        try:
            if not is_grid:
                v = direct_sum_extrapolated(
                    local_edges, local_nu_vec, A,
                    L_list=L_list, n_correction_terms=K_corr,
                    root=s_alg, terminal=t_alg, momentum=k_frac, **kern_kw,
                )
                _bump("n_block_direct_sum")
                _note_tw(tw_block)
                return float(np.asarray(v).real)
            n = int(n_points)
            axes = [np.arange(n, dtype=float) / n for _ in range(d)]
            kk = np.stack(
                np.meshgrid(*axes, indexing="ij"), axis=-1,
            ).reshape(-1, d)
            grid = _direct_sum_extrapolated_grid(
                local_edges, local_nu_vec, A, kk,
                root=s_alg, terminal=t_alg,
                L_list=L_list, n_correction_terms=K_corr, **kern_kw,
            )
            _bump("n_block_direct_sum")
            _note_tw(tw_block)
            return grid.reshape((n,) * d).astype(float)
        except UnsupportedLatticeSumError:
            raise                # never hand an unsupported sum to the torus
        except (ValueError, MemoryError) as exc:
            _box_refusal = exc   # fall through to the tensor path below

    algebra_eligible = tw_block == 2 and use_algebra_low_sigma

    if algebra_eligible:
        # The algebra composes compact tables by cyclic convolution, so
        # its grid must exceed the sum of the radii; a lifted grid still
        # delivers the caller's grid exactly (the regular part is a
        # trigonometric polynomial sampled at any k by graph_sample_at).
        _n_alg = (max(int(n_points), _sumR_fk + 1) if has_compact_block
                  else int(n_points))
        try:
            g_block = graph_from_edges(
                local_edges, local_nu_vec, A, _n_alg,
                s=s_alg, t=t_alg, sigma_max=SIGMA_MAX_ALG, **kern_kw,
            )
        except (NotTreewidthTwoError, InteractionSupportError):
            g_block = None
        if g_block is not None:
            _bump("n_block_algebra")
            if is_grid and _n_alg == int(n_points):
                return graph_sample(g_block).astype(float)
            if is_grid:
                n = int(n_points)
                axes = [np.arange(n, dtype=float) / n for _ in range(d)]
                kk = np.stack(
                    np.meshgrid(*axes, indexing="ij"), axis=-1,
                ).reshape(-1, d)
                out = np.array([graph_sample_at(g_block, k) for k in kk],
                               dtype=float)
                return out.reshape((n,) * d)
            return graph_sample_at(g_block, k_frac)

    # Hybrid (default) or tensor.
    #
    # The split is scoped to DENSE blocks on the torus route, deliberately.
    # A tw ≤ 2 block is fully SP-reducible, so collapsing it at the fine
    # grid would compute essentially the whole block at ``sp_n_points``
    # and restrict down — a large, unmeasured change to the tw ≤ 2 numbers
    # that this routing work has no evidence for and no mandate to make.
    dense_on_torus = tw_block >= 3 and dense_is_torus
    block_sp_n = dense_sp_n if dense_on_torus else None
    # Connectivity-graded core, for EVERY block that reaches hybrid.  This
    # arm is where the grading matters most: there is NO Richardson ladder
    # at finite k (the tail carries cos(2π k·x) and stops being a power
    # law), so nothing else is removing the core term and the split plus a
    # right-sized core is the whole accuracy story.  It is not a
    # dense-block privilege: a tw ≤ 2 block at σ ≥ 1.49 skips the σ_max
    # algebra and lands here at the raw delivery grid, and at n_points = 64
    # those blocks were 16.6 s of an 18.2 s order-9 d = 2 pass (every tw-3
    # block combined: 0.61 s) while the rule already sized their cores at
    # 18 — the pass then scales as n^3.9 in the delivery grid instead of
    # ~n^1.2.  The block's two terminals are pinned, so they are the
    # externals the cut rule must not suppress.
    #
    # A tw ≤ 2 block is graded only when its plan exponent, both pins
    # kept, is ≥ 2.  Census of the tw ≤ 2 occurrences the rule would
    # otherwise touch (d = 2, σ = 1.5, orders ≤ 9, n = 64): 673 of 823
    # are exponent 1 — an O(n^d) core with nothing to save — so grading
    # them moves a value for no time; the gate keeps them bit-identical
    # at no cost (A/B on the same tree, medians of 3 alternated runs:
    # gated / ungated = 1.00, 1.02, 0.96 at n = 32, 64, 128).  The
    # dense arm is not gated:
    # its population is exponent ≥ 2 by construction and it is the
    # calibrated, shipped route.
    #
    # AN EXPLICIT ZERO MOMENTUM IS NOT OFFERED THIS EXTENSION.  hybrid
    # contracts ``momentum = 0`` as the vacuum problem: it drops the
    # terminal, re-pins at the planner pin and SP-reduces around that
    # one vertex.  The core below is sized from the cut rates with BOTH
    # endpoints pinned -- for a contraction hybrid does not run at
    # k = 0 -- and the vacuum residual on it converges at the block
    # rate instead.  Measured on the chain at n = 512, per block against
    # its ungraded value: the order-11 corpus block with a 5-fold bundle
    # between the hopping endpoints (ν = 3) is sized for rate 20 at a
    # core of 20, converges at rate 5.0 and was 2.6e-06 off (the grid
    # node, terminal kept, is exact there); a block with a genuine
    # two-terminal core (ν = 2) is sized for rate 11 at 22, converges at
    # rate 3.0 and was 1.6e-03 off.  Ungraded, the vacuum residual is
    # the whole n-point truncation and costs no more than the graded one
    # (0.26 ms against 0.29 ms on the second block).  This is the k = 0
    # arm's own argument, see ``TestVacuum`` in
    # tests/test_tw2_core_grading.py, applied to the one finite-k
    # request that is contracted as the vacuum problem.  The exemption
    # covers everything this extension reaches, which includes a dense
    # block the box refused above: at an explicit zero every such block
    # is back on the ungraded core it had before the extension.  A grid
    # request and a non-zero k keep the terminal and are graded as
    # before; the dense-on-torus branch is unchanged.
    #
    # The zero test is hybrid's own, ``_momentum_is_zero``: exact, and
    # the set the block-cache key collapses.  Do not give it a tolerance
    # here alone.  A near-zero k exempted here but not rewritten by
    # hybrid runs UNGRADED with the terminal kept, and that core does
    # not fit: the two-terminal-core block above, on the square lattice
    # at n = 128 (measured at ν = 3), prices at 3.6e9 B against the
    # 2.15e9 B ceiling and raises HybridCoreBudgetError.
    k0_vacuum = (not is_grid) and _momentum_is_zero(k_frac)
    block_core_n = None
    # A block with a compact part is graded too, with its core floored
    # at ``_compact_core_floor`` so the residual tables stay exact and
    # only the tail is truncated -- see the k = 0 grading site.
    if core_grading and (dense_on_torus or (
            not k0_vacuum
            and (_block_plan_exponent(local_edges, s_alg, (s_alg, t_alg))
                 or 2) >= 2)):
        s_blk, s_core = _block_cut_rates(
            local_edges, local_nu_vec, d, s_alg, externals=(s_alg, t_alg),
        )
        n_graded = _core_n_for_block(
            int(n_points), s_blk, s_core, int(d), sigma_ref=sigma_ref,
        )
        # The accuracy rule may decline (it returns n_points) or may ask
        # for more than the budget allows.  The ceiling is unconditional.
        n_graded = _core_n_capped(
            n_graded, local_edges,
            sorted({int(v) for e in local_edges for v in e[:2]}),
            s_alg, (s_alg, t_alg), int(d),
        )
        if has_compact_block:
            n_graded = _graded_core_with_tables(
                n_graded, int(n_points), local_edges, b_kern,
            )
        if n_graded < int(n_points):
            block_core_n = n_graded
            if has_compact_block:
                _bump("n_block_core_graded_compact")
    core_kw = {} if block_core_n is None else {"core_n_points": block_core_n}
    if dense_on_torus:
        _bump("n_block_dense_torus")
        if block_sp_n is not None:
            _bump("n_block_split")
        if block_core_n is not None:
            _bump("n_block_core_graded")
    elif block_core_n is not None:
        _bump("n_block_core_graded_tw2")
    _bump("n_block_tensor")
    _note_tw(tw_block)
    try:
        if has_compact_block and _lift_fk > int(n_points):
            _bump("n_block_compact_lift")
            if not is_grid:
                # A single momentum: the SAME evaluator as the vacuum arm at
                # the same lifted grid (hybrid, split kept), so a block that
                # sits on the spine in one graph and off it in another
                # shares one cache entry AND one number at k = 0.  An
                # earlier form-factor path here was 1.7e-05 apart from the
                # vacuum arm under the same key -- a 1qp pass at k = 0
                # became graph-order dependent.
                val = _block_general(
                    local_edges, local_nu_vec, A, int(_lift_fk),
                    source=s_alg, terminal=t_alg, mode="single",
                    momentum=k_frac, engine=engine, sp_n_points=block_sp_n,
                    **core_kw, **kern_kw,
                )
                return float(np.asarray(val).real)
            # A grid request (its own cache key): the exact form factor on
            # the lifted torus, transformed to the caller's grid; no split
            # on this path, the caller's grid is only the output resolution.
            if block_sp_n is not None:
                _bump("n_block_split", -1)
            return _nn_block_finite_k(
                local_edges, s_alg, t_alg, A, int(_lift_fk),
                n_points, momentum, **kern_kw,
            )
        if is_grid:
            arr = _block_general(
                local_edges, local_nu_vec, A, n_points,
                source=s_alg, terminal=t_alg, mode="grid", engine=engine,
                sp_n_points=block_sp_n, **core_kw, **kern_kw,
            )
            return (np.asarray(arr)
                    .reshape((int(n_points),) * d).real.astype(float))
        val = _block_general(
            local_edges, local_nu_vec, A, n_points,
            source=s_alg, terminal=t_alg, mode="single",
            momentum=k_frac, engine=engine, sp_n_points=block_sp_n,
            **core_kw, **kern_kw,
        )
        return float(np.asarray(val).real)
    except MemoryError as exc:
        # Last resort, and only for a dense block we diverted here
        # ourselves: hybrid has already degraded to the tensor inside
        # ``_block_general`` and the tensor has run out too.  The box is
        # the one engine whose cost does not grow with n_points, so it
        # keeps the d = 3 / tw ≥ 4 capability that the torus cannot hold.
        # For every other block this is a genuine allocation failure and
        # must propagate -- naming the box's earlier refusal when there
        # was one (a compact reach the ladder could not hold).
        if not dense_on_torus:
            if _box_refusal is not None:
                raise MemoryError(
                    f"the torus ran out of memory on this block after the "
                    f"box had refused it: {_box_refusal}"
                ) from exc
            raise
        _bump("n_block_dense_torus", -1)
        if block_sp_n is not None:
            _bump("n_block_split", -1)
        _bump("n_block_tensor", -1)
        L_list = _DENSE_DSUM_L_LIST.get(int(d), (2, 3, 4))
        K_corr = min(3, len(L_list) - 1)
        _bump("n_block_direct_sum")
        if not is_grid:
            v = direct_sum_extrapolated(
                local_edges, local_nu_vec, A,
                L_list=L_list, n_correction_terms=K_corr,
                root=s_alg, terminal=t_alg, momentum=k_frac, **kern_kw,
            )
            return float(np.asarray(v).real)
        n = int(n_points)
        axes = [np.arange(n, dtype=float) / n for _ in range(d)]
        kk = np.stack(
            np.meshgrid(*axes, indexing="ij"), axis=-1,
        ).reshape(-1, d)
        grid = _direct_sum_extrapolated_grid(
            local_edges, local_nu_vec, A, kk,
            root=s_alg, terminal=t_alg,
            L_list=L_list, n_correction_terms=K_corr, **kern_kw,
        )
        return grid.reshape((n,) * d).astype(float)


# ---------------------------------------------------------------------------
# Private router (raise-on-error contract)
# ---------------------------------------------------------------------------

def _evaluate_via_topology(
    edges_flat: np.ndarray,
    nu: float | np.ndarray,
    A: np.ndarray,
    s: int,
    t: int,
    n_points: int = 0,
    return_info: bool = False,
    richardson: bool = False,
    p_richardson: float | None = None,
    momentum=None,
    fast_cycles: bool = False,
    block_cache: "dict | None" = None,
    engine: str = "hybrid",
    dense_engine: "str | None" = None,
    sp_n_points: "int | None" = None,
    core_grading: bool = True,
    accuracy: str = "strict",
) -> "float | np.ndarray | tuple":
    r"""Evaluate ζ_G at zero external momentum via block-cut decomposition.

    The full-graph zeta at :math:`k_\text{external}=0` factorises over
    biconnected blocks: every cut vertex carries zero momentum too, so
    each block contributes its own at-zero scalar and the whole-graph
    value is the product.  For each block (after Hadamard-merging
    parallel edges into a single bundled edge with combined exponent)
    the cheapest correct evaluator is selected:

    * **Bridge** — exactly two vertices with one bundled edge of
      exponent :math:`\nu_B`: analytic
      :math:`\mathrm{epstein\_zeta}(\nu_B, A, \mathbf{0}, \mathbf{0})`,
      the :math:`d`-dimensional Epstein zeta of lattice :math:`A` at
      zero displacement and zero momentum.
    * **Simple cycle** — :math:`n_v \ge 3`, :math:`n_e = n_v`:
      analytic :func:`gzl.zeta_circle`.  At zero external
      momentum every cut vertex of the cycle carries zero momentum
      too, so the closed-form scalar is correct regardless of how
      many cut vertices the cycle has.
    * **σ_max=4 algebra** via :func:`graph_from_edges` on the block
      when ``tw(block) == 2``, the block's smallest per-edge σ is
      ``< 1.49``, ``|external| ≤ 2``, and SP reduction succeeds at
      the chosen terminal pair.  Each Hadamard-merged bundle is
      handed to ``graph_from_edges`` as a single edge with the
      bundle's summed ν, so non-uniform per-edge ν across the block
      is handled natively.
    * **Tensor (bucket-elimination)** via
      :func:`graph_zeta_general` for every other block — i.e. when
      ``tw ≥ 3``, or σ ≥ 1.49, or ``|external| ≥ 3``, or the algebra
      path raised :class:`NotTreewidthTwoError` (SP reduction failed
      at the chosen terminals).

    Real-projection.  ζ_G is real at zero source position and any
    external momentum k by the lattice's inversion symmetry; the
    imaginary residue from the FFT path is discarded at every block
    evaluator and the running ``value`` accumulator stays a Python
    :class:`float`.

    Parameters
    ----------
    edges_flat
        Already-expanded multi-edge list (rows ``(u, v)`` repeated per
        multiplicity).  Self-loops raise :class:`SelfLoopError`.
    nu
        Per-edge exponent.  Either a scalar (uniform across all rows) or
        a 1-D array of length ``len(edges_flat)``.  Bundles inherit the
        sum :math:`\sum \nu_e` over the edges they merge.  Both the
        algebra path and the tensor path accept non-uniform per-edge ν
        natively; non-uniform input does not force tensor.
    A
        Lattice matrix.
    s, t
        Source / terminal vertex labels of the 1qp graph.  Equal
        ``s == t`` is the 0qp / vacuum convention.  Out-of-range labels
        raise :class:`VertexOutOfRangeError`.
    n_points
        Discretisation passed to :func:`graph_zeta_general` for any
        block that needs it.  ``0`` (default) is fine for graphs that
        decompose entirely into bridges + off-path cycles — those
        evaluate analytically and never touch the tensor evaluator.
        Otherwise raises :class:`NPointsRequiredError`.
    return_info
        If ``True``, return ``(value, info_dict)`` where ``info_dict``
        reports per-block route counts: ``n_bridges``, ``n_simple_cycles``,
        ``n_block_algebra``, ``n_block_tensor``, ``max_algebra_block_tw``,
        ``max_tensor_block_tw``, and (when ``richardson=True``)
        ``n_block_richardson_theory`` / ``n_block_richardson_skipped``
        (and ``n_block_richardson_self``, retained but always 0 — see
        below).
    richardson
        If ``True``, extrapolate every tensor-routed block along a
        ladder of torus sizes to remove the leading truncation tail.
        The basis exponent is the block's **cluster cut**
        ``σ_cut`` (:func:`direct_sum._min_free_cut_nu`), not the
        ``p = 2 − σ`` trapezoid guess an earlier version used — that
        guess was unrelated to the true rate and where it fired it could
        be worse than not extrapolating at all.  Rungs come from
        :func:`_richardson_rungs` (geometric, parity-matched) and the
        fit is accepted only if all five guards in
        :func:`_richardson_ladder` pass.

        ``_evaluate_via_topology`` defaults this to ``False``;
        :func:`evaluate_graph` defaults it to ``True``.

        ``n_block_richardson_self`` counted a self-tuning empirical
        exponent that no longer exists — the ladder always uses σ_cut
        (or ``p_richardson``).  The key is kept so existing readers do
        not break, but it is permanently 0 and must not be read as
        "self-tuning did not fire".
    p_richardson
        Hard override of the Richardson exponent.
        Debug knob — not exposed via :func:`evaluate_graph`'s public
        signature; used by the regression suite to pin ``p`` to a known
        value.  Ignored when ``richardson=False``.

    Returns
    -------
    float
        The graph zeta at :math:`k_\text{external}=0`, real-valued.
    tuple[float, dict]
        With ``return_info=True``, returns the value plus the diagnostics
        dict described above.

    Raises
    ------
    TopologyEvaluatorUnavailableError
        ``networkx`` cannot be imported (a broken installation).
    SelfLoopError
        ``edges_flat`` contains a self-loop.
    DisconnectedGraphError
        The simple graph after Hadamard-merging is disconnected.
    VertexOutOfRangeError
        ``s`` or ``t`` is not a vertex.
    NPointsRequiredError
        At least one block needs the σ-routed evaluator and
        ``n_points <= 0``.
    """
    info = {
        "n_bridges":           0,   # bridge blocks evaluated via epstein_zeta
        "n_simple_cycles":    0,   # cycle blocks evaluated via zeta_circle
        "n_block_algebra":     0,   # tw=2 blocks at σ<1.49: σ_max=4 algebra path
        # Algebra results refused on conditioning grounds and re-routed to
        # the tensor path — a deep chain at small σ crowds the retained
        # exponents toward ν → d, and the final convolution then has to
        # cancel float64 coefficients over ~1e14 of dynamic range.  See
        # _ALGEBRA_MAX_CONDITION.
        "n_block_algebra_refused": 0,
        "n_block_direct_sum":  0,   # dense tw≥3 blocks: small-L direct sum (3b)
        "n_block_tensor":      0,   # everything else: graph_zeta_general
        # Dense routing.  ``n_block_dense_torus`` counts tw≥3
        # blocks diverted off the box onto the torus; they are ALSO
        # counted in n_block_tensor (they really do run that path), so
        # the two are not disjoint and must not be summed.  Without this
        # counter the migration would be invisible — a tw≥3 block moving
        # from the box to the torus looks exactly like an ordinary tensor
        # block in the tally.
        "n_block_dense_torus": 0,   # tw≥3 blocks routed to the torus
        "n_block_split":       0,   # ...of which ran hybrid's SP split
        # ...of which then LOST it to the tensor fallback, which has no
        # split mode.  n_block_split counts what was ASKED; a non-zero
        # value here means that many of them did not get it.
        "n_block_split_dropped": 0,
        # ...and of which had the core sized DOWN by the grading rule.
        # A block whose cuts coincide (nothing SP-reducible) gets
        # n_core == n_points and is deliberately NOT counted here, so
        # this reads as "how much of the corpus the rule actually acts
        # on" rather than "how often it was switched on".
        "n_block_core_graded": 0,
        "n_block_core_graded_tw2": 0,
        # ...of which carried a compact part (core floored at the
        # winding-safe size of its residual tables).
        "n_block_core_graded_compact": 0,
        # tw≥3 blocks with ≥3 externals: they take graph_zeta_general
        # directly (hybrid supports at most one free terminal), so the
        # split never reaches them.  Logged because the occurrence share
        # is otherwise unmeasurable from outside.
        "n_block_dense_multi_ext": 0,
        # Dense blocks the torus route WOULD have taken and the accuracy
        # gate sent back to the box (see _TORUS_MIN_N_BY_CLASS).  Counted
        # separately from a byte-budget refusal because the two decline
        # for different reasons and only one of them is fixed by a bigger
        # machine.
        "n_block_dense_torus_declined": 0,
        "n_block_slab_finite_k": 0,
        "n_block_slab_fk_floor": 0,
        # Subset of n_block_slab_fk_floor: floor-tier blocks whose
        # priced outer pin is NOT the terminal (see _fk_outer_best).
        "n_block_fk_outer_pin": 0,
        "n_block_fk_pair_band_declined": 0,
        # Declined blocks the SLAB then took, rather than the box (see
        # _SLAB_N_BY_CLASS).  A subset of the declines above: every slab
        # block was first declined by the accuracy gate, which is what
        # keeps the arm off the shapes the slab loses on.
        "n_block_slab": 0,
        # Admitted blocks the PREFERRED slab route took instead of the
        # torus ladder (see _SLAB_PREFERRED_RUNGS), and the ones its
        # two-rung self-band tripwire sent to the box instead.
        "n_block_slab_preferred": 0,
        "n_block_slab_selfband_declined": 0,
        "n_block_nn":          0,   # NN-kernel (ν=inf) blocks (subset of tensor)
        "nn_mode":         False,   # any edge ν = inf → nearest-neighbour kernel
        "max_algebra_block_tw": 0,  # always 2 by construction; logged for symmetry
        "max_tensor_block_tw": 0,   # max treewidth of a tensor-evaluated block
                                    # (min-fill-in heuristic upper bound; tight
                                    # for the small blocks we hit in practice).
        # Richardson-on-tensor diagnostics (only nonzero when richardson=True):
        "n_block_richardson_self":    0,  # DEAD: the self-tuning branch
                                    # was removed; always 0.  Kept so
                                    # existing readers do not break.
        "n_block_richardson_theory":  0,  # ladder accepted (basis σ_cut)
        "n_block_richardson_skipped": 0,  # ladder declined by a guard,
                                    # or block_n below the rung floor
        # Ladder deliberately NOT attempted because the core was graded
        # down: the term it removes is already below the delivery grid's
        # floor, and the rungs beneath a graded core are pre-asymptotic.
        # Separate from ..._skipped so "declined by a guard" keeps
        # meaning that, and the two causes stay countable apart.
        "n_block_richardson_skipped_graded": 0,
        # Ladder not attempted because a rung would sit below the block's
        # winding-safe grid n_v * R + 2 (a compact part; see the lift).
        "n_block_richardson_skipped_window": 0,
        "n_block_compact_lift":       0,  # compact block run at its lifted grid
        "n_block_cache_hits":         0,  # blocks served from block_cache
    }

    _split_dropped_at_entry = _SPLIT_DROPPED

    def _ret(value):
        # `_block_general` degrades to the tensor without seeing `info`,
        # so the count is read back as a delta over this call.
        info["n_block_split_dropped"] = _SPLIT_DROPPED - _split_dropped_at_entry
        return (value, info) if return_info else value

    if not _TOPO_AVAILABLE:
        raise TopologyEvaluatorUnavailableError(
            f"the topology-first router needs networkx and epsteinlib, "
            f"and importing them failed ({_TOPO_IMPORT_ERROR}).  Both are "
            f"dependencies of gzl, so the installation is broken: "
            f"reinstall gzl."
        ) from _TOPO_IMPORT_ERROR

    edges_flat = np.asarray(edges_flat, dtype=int)
    if edges_flat.ndim == 1:
        edges_flat = edges_flat.reshape(-1, 2)

    # Coerce nu to a 1-D ndarray of length E (broadcast scalar internally).
    # ``coerce_nu`` is the historical coercion verbatim on the power-law
    # path; with general interactions it also returns the per-edge
    # kernel list, and ``nu_arr`` then carries the TAIL exponents every
    # routing decision below reads, the pass floor included (it reads
    # them per edge, through ``bundle_orig_nus``).
    nu_arr, kernels = coerce_nu(nu, len(edges_flat))
    _nu_f = np.asarray(nu_arr, dtype=float)
    if np.isnan(_nu_f).any() or np.isneginf(_nu_f).any():
        # A malformed argument, not an unsupported exponent: refuse it
        # before the nu <= d test below, which would read it as one.
        # -inf too: legacy nearest-neighbour mode keys on isinf, so a
        # -inf bridge returned the nu = +inf count (2.0 on the chain).
        raise ValueError(
            f"nu must be a number or +inf, got {np.asarray(nu_arr)!r}")
    # NOTE: deliberately no call-wide "has a compact part" flag here.  Such
    # a flag must never reach a routing decision: keying the split on
    # kernels OUTSIDE a block made a pure block's value depend on its
    # neighbours (measured; see the split resolution below), and gating an
    # exactness shortcut on a call-wide flag is what froze unrelated blocks
    # at a tiny grid.  Compactness is a per-BLOCK property; read it from
    # ``b_kern`` inside the block loop.
    info["interaction_mode"] = kernels is not None
    info["n_block_mixed_kernel"] = 0
    info["n_block_cycle_fallback"] = 0

    # Nearest-neighbour mode: any ν = inf requests the exact NN-indicator
    # kernel (the ν → ∞ limit).  In this mode every block routes to the
    # truncated-Fourier tensor — bridges/cycles/dense blocks must NOT reach
    # the closed-form (epstein_zeta / zeta_circle) or real-space direct-sum
    # evaluators, which are undefined at ν = inf.
    # Legacy: any nu = inf edge puts the whole CALL in nearest-neighbour
    # mode.  Kernel path: the call is in nn mode only when EVERY kernel is
    # purely compact.  A graph that mixes a purely compact bundle with
    # finite-tail bundles keeps the power-law routes: a block whose bundles
    # are all purely compact is still evaluated exactly on its lifted torus
    # (``nn_blk`` below), a block that mixes the two takes the compact-block
    # rules (lift floors, closed forms, ladders on the finite tails).
    # Measured: a triangle with one nearest-neighbour table beside nu = 2.3
    # tails shipped the raw n = 8 algebra, 1.5e-2 off, under the call-wide
    # flag; the closed form is exact there.
    nn_mode = (bool(np.isinf(nu_arr).any()) if kernels is None
               else bool(np.isinf(nu_arr).all()))
    info["nn_mode"] = nn_mode

    # Resolve the dense-block route ONCE per call, here, so that the
    # block-cache key and every routing decision below read the same
    # triple.  Resolving per block would let the key and the route drift
    # apart, which is the one failure mode the cache cannot detect: its
    # lookup guard validates topology, ν and role, never numerics.
    # A compact part does NOT enter here: the split is value-correct with
    # tables (hybrid samples the table on the fine SP grid; measured on
    # K4 with a subdivided edge, R = 1 + nu = 3 on the triangular cell,
    # n = 8: raw 1.5e-04, split 1.5e-05 against the box ladder, the same
    # 10x it buys a pure power law), and keying the split on kernels
    # OUTSIDE a block made a pure block's value depend on its neighbours
    # (measured 12x worse for a K4 beside a pendant bridge carrying the
    # only table).  The graded core is declined PER compact block below.
    dense_route, dense_sp_n, dense_core_grade = _resolve_dense_routing(
        int(A.shape[0]), dense_engine, sp_n_points, n_points,
        nn_mode, core_grading,
    )
    info["dense_engine"] = dense_route
    info["sp_n_points"] = dense_sp_n

    # THE SPLIT GRID AN ON-SPINE BLOCK ACTUALLY RUNS AT.  The finite-k
    # arm suppresses the split where no finite-k measurement backs it
    # (see _SPLIT_FINITE_K_DIMS), while the k = 0 blocks of the SAME
    # graph keep it.  Hoisted here from the call site because the BLOCK
    # CACHE KEY has to see it: a block evaluated with the split and one
    # evaluated without it are different numbers, and without this the
    # two share a key.
    #
    # Measured on the shape that found it -- two isomorphic K4SUB blocks
    # sharing a cut vertex, d = 3, nu = 3.5, n_points = 8, at
    # momentum = 0 with terminal = 1.  One block is on the spine (split
    # suppressed) and the other is off it (split on); before this they
    # keyed identically and `evaluate_graph` returned 150231.4152957815
    # with no cache against 150036.5470942466 with one -- 1.30e-03
    # apart, with the winner set by block iteration order.  Passing
    # `sp_n_points == n_points` (which folds the split off everywhere)
    # made them agree bitwise, isolating the cause.
    _fk_sp_pass = dense_sp_n
    if sp_n_points is None and int(A.shape[0]) not in _SPLIT_FINITE_K_DIMS:
        _fk_sp_pass = None
    info["core_grading"] = dense_core_grade

    is_grid = isinstance(momentum, str) and momentum == "grid"
    finite_k = momentum is not None and int(s) != int(t)

    if len(edges_flat) == 0:
        if is_grid:
            d = int(A.shape[0])
            n = int(n_points)
            return _ret(np.ones((n,) * d, dtype=float))
        return _ret(1.0)

    # Step 1: Hadamard-merge parallel edges into bundles, tracking the
    # original per-edge ν's that contributed to each bundle (needed for
    # the per-block uniformity check that gates algebra eligibility).
    bundles: dict = {}
    bundle_orig_nus: dict = {}
    # General interactions: the bundle kernel is the lazy POINTWISE
    # product of the parallel kernels (row order), while ``bundles``
    # keeps the float tail sum every planner consumer reads.
    bundle_kernels: dict = {}
    for i in range(len(edges_flat)):
        u, v = int(edges_flat[i, 0]), int(edges_flat[i, 1])
        if u == v:
            raise SelfLoopError(
                f"Self-loop at vertex {u} in row {i} of edges_flat; the "
                f"topology-first router does not support self-loops."
            )
        key = (u, v) if u < v else (v, u)
        nu_e = float(nu_arr[i])
        bundles[key] = bundles.get(key, 0.0) + nu_e
        bundle_orig_nus.setdefault(key, []).append(nu_e)
        if kernels is not None:
            bundle_kernels[key] = (kernels[i] if key not in bundle_kernels
                                   else bundle_kernels[key] * kernels[i])

    # Step 2: build the simple graph (one edge per bundle).  The vertex
    # set is the edge support (see gzl/_labels.py): a label that
    # appears in no edge is not a vertex, so labels need not be
    # contiguous — every block is relabelled to ``0..n_v-1`` before
    # dispatch anyway.  Building over ``range(max + 1)`` here used to
    # turn each gap label into a phantom isolated node and refuse a
    # connected graph as "disconnected".
    # SORTED insertion is load-bearing: NetworkX traversal (biconnected
    # components, subgraph edge iteration, min-fill tie-breaks) follows
    # node insertion order, and for a contiguous input sorted() IS the
    # old range(V) order — bundle-key order is not (measured: a 1-ULP
    # router shift on a triangle whose vertices are first mentioned out
    # of order, via a permuted nu_vec into zeta_circle).
    G = _nx.Graph()
    G.add_nodes_from(sorted({w for e in bundles for w in e}))
    for (u, v), nu_b in bundles.items():
        G.add_edge(u, v, nu=nu_b)
        if kernels is not None:
            G[u][v]["kern"] = bundle_kernels[(u, v)]

    if int(s) not in G:
        raise VertexOutOfRangeError(
            f"source vertex s={s} is not a vertex of this multigraph: "
            f"the label appears in no edge (the vertex set is the edge "
            f"support)."
        )
    if int(t) not in G:
        raise VertexOutOfRangeError(
            f"terminal vertex t={t} is not a vertex of this multigraph: "
            f"the label appears in no edge (the vertex set is the edge "
            f"support)."
        )
    if not _nx.is_connected(G):
        raise DisconnectedGraphError(
            "The simple multigraph (after Hadamard-merging parallel edges) "
            "is disconnected; ζ_G factorises across components, but you "
            "must evaluate each component separately."
        )

    # Step 3: block-cut decomposition.
    blocks_v = [frozenset(b) for b in _nx.biconnected_components(G)]
    cuts = set(_nx.articulation_points(G))

    # Step 4: per-block evaluation.  At k_external = 0, every cut vertex
    # carries zero momentum too, so each block contributes its at-zero
    # scalar with all its boundary vertices pinned at the origin (in the
    # source axis) or integrated at k = 0 (in the terminal axes).  The
    # full-graph value is the product.
    #
    # At finite k the blocks on the (s, t)-spine of the block-cut tree
    # carry the external momentum and go to ``_block_at_finite_k``;
    # every other block evaluates at zero momentum.
    d = A.shape[0]
    zero_d = np.zeros(d, dtype=float)
    SIGMA_MAX_ALG = 4.0

    # nu <= d, refused for every block but a bridge (see
    # UnsupportedLatticeSumError).  Checked HERE, after the Hadamard merge
    # and the block-cut decomposition and before any block is cached,
    # routed or evaluated, so that every momentum mode, every engine
    # setting, ``n_points = 0`` and the legacy nearest-neighbour mode
    # refuse the same blocks.  Before, the refusal lived in the engines,
    # and routes around them returned numbers: a treewidth-2 block with
    # three attachments went to the torus (292.5 / 867.5 / 886.5 at
    # n = 8 / 16 / 32 for a diamond at 0.9 with pendants at 2.5), and one
    # nu = inf edge anywhere in the call skipped the dense guard.  The
    # BUNDLE exponent is what the engines see, so two parallel edges of
    # 0.6 on the chain form one bundle of 1.2 and pass.  ``not (w > d)``
    # so that nothing unordered can pass; +inf, a purely compact bundle,
    # passes.  A bridge keeps its value, and its pole is refused where
    # the closed form is evaluated (``_finite_bridge``).
    for b_vset in blocks_v:
        b_ws = [float(w) for _, _, w in G.subgraph(b_vset).edges(data="nu")]
        if len(b_ws) == 1:
            continue
        low = sorted(w for w in b_ws if not (w > d))
        if low:
            raise UnsupportedLatticeSumError(
                f"a block with {len(b_vset)} vertices has a bundle exponent "
                f"nu = {low[0]} <= d = {int(d)}.  Exponents nu <= d are "
                f"supported on bridges only, where the Epstein zeta function "
                f"continues the sum.  The sum over this block may well "
                f"converge (that is governed by its cluster cut, not by the "
                f"smallest exponent), but no evaluator in gzl is validated "
                f"there."
            )

    # Spine tagging: which blocks lie on the s-t path of the
    # block-cut tree.  Off-spine blocks contribute their k = 0 scalar
    # regardless of the requested ``momentum``; only on-spine blocks
    # see the external momentum.  When ``s == t`` (vacuum) or
    # ``momentum is None`` (the k = 0 path), no block is on-path.
    spine_endpoints: dict = {}
    if finite_k:
        for b, s_b, t_b in _spine_path(blocks_v, cuts, int(s), int(t)):
            spine_endpoints[b] = (s_b, t_b)

    # ---- per-block cache key ------------------------------------------
    # Off-spine blocks always contribute ζ_block(0) (attachment
    # theorem), so they are momentum-independent; only spine blocks see
    # the requested k.  At k = 0 the exact value is also
    # boundary-independent (the k=0 cell marginalises the boundary
    # axes), so bridges/cycles get an exact structural key and the
    # σ-routed blocks of the k = 0 arm (vacuum and off-spine) an
    # all-roles-0 WL signature — maximising reuse exactly where the
    # corpus redundancy concentrates.  Spine blocks fold the
    # spine-endpoint roles into the signature at EVERY k, k = 0
    # included, and genuine finite-k spine blocks the momentum as well,
    # so non-equivalent blocks — and equivalent blocks evaluated by
    # different routines — never collide.  See ``_make_key``.
    if block_cache is not None:
        _A_key = (
            tuple(A.shape),
            np.asarray(A, dtype=float).round(_NU_KEY_DECIMALS).tobytes(),
        )
        # ``engine`` belongs here: hybrid and tensor are only equal to
        # round-off, not bit-identical, so a cache shared across a
        # hybrid call and a tensor call would serve one engine's value
        # to the other.  Nothing else in the key distinguishes them.
        #
        # The same argument applies with far more force to the dense
        # route: box-at-L and torus-at-n are different numbers, not
        # different roundings, and the split changes the value again.
        # Both are the RESOLVED values, so callers that spell the same
        # routing differently still share entries.  A corpus pass shares
        # one dict across thousands of graphs, so getting this wrong
        # would serve one setting's value to another silently.
        _flags = (
            bool(richardson),
            None if p_richardson is None else round(float(p_richardson),
                                                    _NU_KEY_DECIMALS),
            bool(fast_cycles),
            str(engine),
            str(dense_route),
            None if dense_sp_n is None else int(dense_sp_n),
            bool(dense_core_grade),
            # Floor-mode values are a different truncation choice for
            # the same block; a mixed-mode session must never share.
            str(accuracy),
        )
        _nkey = int(n_points)
        def _make_key(b_vset, b_set, sub, b_edges, n_v, n_e, b_kern=None,
                      floor=None):
            is_spine = b_vset in spine_endpoints
            nn_here = (nn_mode if b_kern is None
                       else all(not np.isfinite(k.tail_exponent) for k in b_kern))
            mkey = _momentum_cache_key(momentum) if is_spine else None
            # In nn_mode every block is tensor-routed, so bridges/cycles key
            # via the σ-WL branch (consistent with how they are evaluated).
            #
            # The key must describe the EVALUATOR that produced the value,
            # not merely the mathematical object.  ``mkey`` collapses k = 0
            # to ``None`` because the exact ζ_block is momentum-independent
            # there — but an on-spine block is dispatched through
            # ``_block_at_finite_k`` regardless, and that routine uses the
            # closed form only for BRIDGES (``epstein_zeta`` at any k,
            # which at k = 0 reproduces the off-spine value exactly).  For a
            # CYCLE it deliberately falls through to the σ-routed
            # algebra/tensor evaluator, since ``zeta_circle`` is k = 0
            # only.  Letting that discretisation-carrying value be stored
            # under the closed-form ``("cycle", …)`` key poisoned the entry
            # for every later vacuum / off-spine reuse of the same cycle,
            # making a shared-cache corpus pass depend on graph ORDER
            # (measured: 2.5e-4 at n_points = 16, 4.7e-3 at n_points = 8,
            # reachable from the documented ``momentum=np.zeros(d)`` call).
            # An on-spine cycle therefore takes the σ-WL branch.
            is_bridge = ((n_v == 2 and n_e == 1)
                         and (not nn_here
                              or _is_low_bridge(n_v, n_e, b_edges, b_kern, d)))
            is_cycle = (
                n_v >= 3 and n_v == n_e and d in _CLOSED_FORM_DIMS
                and not (fast_cycles or nn_here or is_spine
                         or _large_sigma_cycle(b_edges, d))
            )
            # General interactions key on the bundle kernels' fingerprints
            # (uniform per call, see _bundle_key); the legacy path keeps
            # its rounded floats byte-identical.
            _slots = b_edges if b_kern is None else [
                (u, v, k) for (u, v, _), k in zip(b_edges, b_kern)
            ]
            if mkey is None and is_bridge:
                nu_b = _bundle_key(_slots[0][2])
                return ("bridge", nu_b, _A_key, _nkey, _flags), None
            if mkey is None and is_cycle:
                nus = _sorted_keys(_bundle_key(e[2]) for e in _slots)
                return ("cycle", n_v, nus, _A_key, _nkey, _flags), None
            sorted_v = sorted(b_set)
            local = {v: i for i, v in enumerate(sorted_v)}
            role_map = {i: 0 for i in range(len(sorted_v))}
            # The spine roles are folded in whenever the block is ON the
            # spine, not only when ``mkey`` is set.  The cycle argument
            # above, one level down: ``mkey`` collapses k = 0 to ``None``
            # because the exact ζ_block is momentum-independent there,
            # but an on-spine block at k = 0 is still evaluated by
            # ``_block_at_finite_k`` at (s_b, t_b) -- where the σ_eff
            # Richardson ladder is deliberately absent -- while an
            # off-spine block of the same shape takes the k = 0 arm and
            # gets the ladder.  Keying both on the all-roles-0 signature
            # served one route's value to the other, so a shared-cache
            # pass depended on block order.  Measured on two isomorphic
            # K_{2,3} blocks sharing a cut vertex at d = 2, ν = 3.5,
            # ``momentum=np.zeros(2)``: cache vs no cache 9.2e-07 at
            # n = 32 and 2.9e-08 at n = 64, gone with ``richardson=False``
            # (which names the ladder) and present with
            # ``core_grading=False`` (which acquits the grading).
            # Reachable in the shipped default (``richardson=True``)
            # whenever a pass's on-spine block is isomorphic to another
            # graph's off-spine block.  The ENDPOINTS go in, not just a
            # spine flag: the finite-k evaluator's dense arms (the
            # accuracy gate, the second-pin basis, the split's SP
            # reduction) are keyed on which vertices are pinned, so the
            # same shape entered at different endpoints is a different
            # truncation even at k = 0.  Bridges return above and stay
            # shared: their k = 0 closed form IS the off-spine value.
            if is_spine:
                s_b, t_b = spine_endpoints[b_vset]
                role_map[local[s_b]] = 2
                role_map[local[t_b]] = 3
            b_edges_local = [
                (local[u], local[v], nu_b) for u, v, nu_b in _slots
            ]
            h, vg = _block_signature(b_edges_local, role_map)
            # The EFFECTIVE split grid, not the pass-level one.  See
            # `_fk_sp_pass` above: on-spine and off-spine blocks of the
            # same shape can run at different grids in the same call, and
            # ``mkey`` cannot separate them because it collapses k = 0 to
            # ``None``.  Bridges and cycles return above and are
            # split-insensitive (closed forms), so this is the only
            # branch that needs it.
            _eff_sp = _fk_sp_pass if is_spine else dense_sp_n
            _eff_sp = None if _eff_sp is None else int(_eff_sp)
            # The block's own core-grading floor (see _pass_floor_sigma).
            # It is read from the PER-EDGE exponents, and the signature
            # above sees only the merged bundles: K5 at nu = 2.5 and K5
            # with one edge split into two parallel edges of 1.25 have
            # the same signature, but floors 4.0 and 1.5, so on the chain
            # at n_points = 32 their cores are graded to 20 and 8 and
            # their values differ by 11%.  ``bool(dense_core_grade)`` in
            # ``_flags`` records only that grading is on, never what it
            # decided.  Only included when grading is live, so the cache
            # does not fragment on a number nothing reads, and only on
            # this key: the closed forms never read it.
            _ref_key = (round(float(floor), _NU_KEY_DECIMALS)
                        if dense_core_grade and floor is not None else None)
            return ("sigma", h, _A_key, _nkey, _flags, _ref_key, mkey,
                    _eff_sp), vg

    if is_grid:
        n = int(n_points)
        value = np.ones((n,) * d, dtype=float)
    else:
        value = 1.0     # ζ_G is real at x = 0 by lattice symmetry.

    for b_vset in blocks_v:
        b_set = set(b_vset)
        sub = G.subgraph(b_vset)
        b_edges = [(u, v, sub[u][v]['nu']) for u, v in sub.edges()]
        n_v, n_e = len(b_set), len(b_edges)
        # THIS block's core-grading floor, from its own per-edge
        # exponents and nothing outside it (see _pass_floor_sigma).
        blk_floor = _pass_floor_sigma(
            [nu_e for u, v in sub.edges()
             for nu_e in bundle_orig_nus[(min(u, v), max(u, v))]], d)
        # The per-bundle kernels, in the SAME order as ``b_edges`` (both
        # iterate ``sub.edges()``); ``None`` on the legacy path.
        b_kern = (None if kernels is None
                  else [sub[u][v]["kern"] for u, v in sub.edges()])
        has_compact_block = (b_kern is not None
                             and any(k.has_compact for k in b_kern))
        kern_kw = {} if b_kern is None else {"kernels": list(b_kern)}
        # THIS block's nearest-neighbour flag: the call-wide legacy flag on
        # the float path; on the kernel path, every bundle purely compact.
        nn_blk = (nn_mode if b_kern is None
                  else all(not np.isfinite(k.tail_exponent) for k in b_kern))
        # A power-law bridge at a finite exponent <= d: it keeps its
        # closed form even in legacy nearest-neighbour mode (see the
        # bridge arm and ``_make_key``).
        low_bridge = _is_low_bridge(n_v, n_e, b_edges, b_kern, d)
        # ``nn_blk`` is a routing HINT -- steer off the closed forms, which
        # are undefined at ν = inf, and raise the grid.  The call-wide
        # legacy flag is safe in that role: it only ever costs accuracy.
        # ``nn_exact_blk`` is a PROOF OBLIGATION.  It alone may gate
        # ``_compact_exact_grid``, whose exactness premise is that EVERY
        # bundle of THIS block is compact, and which therefore drops
        # ``n_points`` from the answer.  The call-wide flag cannot carry
        # that premise: one ν = inf edge anywhere in the call would pin an
        # unrelated power-law block to a tiny torus and freeze its value
        # against n_points, so the only convergence check a caller has
        # reports perfect stability on a wrong number (measured before
        # this gate: a ν = 2.5 K4 beside one ν = inf pendant bridge, 3.6x
        # high and bit-identical at n = 8, 16, 32 and 64; a mixed
        # [inf, 3.5] path graph 11.25 % low against the exact 4ζ(3.5)).
        # ``bool(b_edges)`` because a vacuous ``all()`` would claim the
        # licence for an edgeless block -- the same shape of mistake.
        nn_exact_blk = ((bool(b_edges)
                         and all(bool(np.isinf(float(_nu_b))) for _, _, _nu_b in b_edges))
                        if b_kern is None else nn_blk)
        # The block's winding-safe torus grid (see the lift at the tensor
        # arm) and the algebra's composed-support grid, both 0 without a
        # compact part.
        _R_blk = (max(int(k.support_radius) for k in b_kern)
                  if has_compact_block else (1 if nn_blk else 0))
        _lift_blk = ((n_v * max(_R_blk, 1) + 2)
                     if (has_compact_block or nn_blk) else 0)
        _sumR_blk = (sum(int(k.support_radius) for k in b_kern
                         if k.has_compact) if has_compact_block else 0)
        if b_kern is not None and any(
                k.has_compact and np.isfinite(k.tail_exponent)
                for k in b_kern):
            # Genuinely MIXED: a compact part beside a power-law tail.
            # (A purely compact block is counted under n_block_nn; a
            # multi-term pure power law takes every legacy route.)
            info["n_block_mixed_kernel"] += 1

        # Cache short-circuit: identical block already evaluated this
        # corpus pass?  (Exact — WL-hash hits are isomorphism-verified.)
        _pk = _vg = None
        if block_cache is not None:
            _pk, _vg = _make_key(b_vset, b_set, sub, b_edges, n_v, n_e,
                                 b_kern, floor=blk_floor)
            _cached = _cache_lookup(block_cache, _pk, _vg)
            if _cached is not None:
                info["n_block_cache_hits"] += 1
                value = value * _cached
                continue

        def _store(bv):
            if block_cache is not None and _pk is not None:
                _cache_store(block_cache, _pk, _vg, bv)
            return bv

        # On-path block at finite k.  Dispatch into the finite-k
        # per-block evaluator and skip the k = 0 logic.
        # Bridge blocks update n_bridges; everything else is counted
        # under n_block_tensor / n_block_algebra by the underlying
        # evaluator's choice (we tally based on the dispatch path
        # below).
        if b_vset in spine_endpoints:
            s_b, t_b = spine_endpoints[b_vset]
            # The evaluator tallies its own branch into ``info``: which
            # route an on-spine block takes cannot be inferred from
            # n_v / n_e here, and guessing it billed every finite-k
            # direct-sum call as n_block_tensor.
            # The split is scoped per dimension at finite k (see
            # _SPLIT_FINITE_K_DIMS).  d = 3's entry is a k = 0
            # measurement and does not carry to this arm; the k = 0
            # blocks of the SAME graph -- every off-spine block -- keep
            # it, which is why this is suppressed here rather than at the
            # resolver.  An explicit caller-supplied grid is honoured:
            # silently ignoring it would make the parameter inert on the
            # one path a finite-k measurement campaign has to drive.
            _fk_sp = _fk_sp_pass
            block_value = _block_at_finite_k(
                sub, b_set, b_edges, bundle_orig_nus,
                s_b, t_b, A, n_points, momentum, nn_mode=nn_exact_blk,
                engine=engine, dense_engine=dense_route,
                sp_n_points=_fk_sp, info=info,
                core_grading=dense_core_grade,
                sigma_ref=blk_floor,
                accuracy=accuracy, b_kern=b_kern,
            )
            value = value * _store(block_value)
            continue

        # External boundary of this block: cut vertices in it, plus s
        # and t if they lie in it.  These are the vertices whose
        # position couples the block to the rest of the graph.
        external = (b_set & cuts) | (b_set & {int(s), int(t)})

        # Original per-edge ν's that landed in this block (across all
        # bundles); drives the per-block σ test.  After Hadamard merging
        # each bundle has one summed ν, so the algebra path can take a
        # per-bundle ν vector via graph_from_edges (no uniform-ν gate).
        block_orig_nus = []
        for (u, v) in sub.edges():
            key = (u, v) if u < v else (v, u)
            block_orig_nus.extend(bundle_orig_nus[key])
        sigma_block_min = min(block_orig_nus) - float(d)
        # Leading torus-truncation exponent of this block.  Letting one
        # free vertex escape the torus leaves a tail
        # ∫_{R} r^{d-1} r^{-I_v} dr ~ R^{d-I_v} with R ~ n/2 and
        # I_v = Σ_{e ∋ v} ν_e, so the error decays as n^-σ_eff with
        #     σ_eff = min over free vertices of I_v − d
        # — governed by the smallest *total incident* exponent, not by
        # the smallest per-edge ν.  (Measured: a bridge at ν = d + 0.5
        # decays with p = 0.5, a triangle at the same ν with p = 2ν−d.)
        _emap: dict = {}
        _inc: dict = {}
        for (u, v) in sub.edges():
            key = (u, v) if u < v else (v, u)
            w = float(sum(bundle_orig_nus[key]))
            _emap[key] = w
            _inc[u] = _inc.get(u, 0.0) + w
            _inc[v] = _inc.get(v, 0.0) + w
        _free = [k for k in _inc if k not in external]
        if _free:
            # Pinned set = the block's external boundary (cut vertices /
            # terminals); those cannot escape.  With no boundary the
            # engine still pins one vertex, so pin the lowest label.
            # THE ROOT IS THE PIN, AND IT DECIDES WHETHER GRADING HAPPENS.
            #
            # The pin is never eliminated, so pinning a degree-2 vertex
            # keeps its escape alive and holds sigma_core down at the
            # block's own rate — whereupon the sizing rule correctly
            # concludes there is no gap to spend and declines.  With no
            # external boundary the pin is FREE (translation invariance on
            # the torus), so choose it to maximise the SP-reduced cut,
            # which is the same criterion `hybrid._planner_pin` breaks its
            # exponent ties on.  The two MUST agree: this site decides the
            # core size, that one does the contraction, and a block sized
            # for one pin and contracted at another is sized wrong.
            #
            # Measured on the d = 2 order-11 block that dominated the pass
            # (V6E11, degrees [2,4,4,4,4,4]): root 0 gives sigma_core 3.00
            # and no grading, any other root gives 8.00 and n_core = 12 —
            # 17.2 GB against 48 MB.
            #
            # With an external boundary the pin is NOT free; keep the
            # historical choice there.
            if external:
                _root = min(external)
            else:
                _root = max(sorted(_inc),
                            key=lambda r: (_sp_reduced_cut_nu(_emap, r), -r))
            sigma_eff_block = _min_free_cut_nu(
                _emap, _root, externals=external) - float(d)
            # The rate a SPLIT evaluation converges at, which is not the
            # block's.  The block rate is set by its cheapest escape, a
            # degree-2 free vertex at cut 2ν — and that is exactly what
            # SP reduction suppresses, so the split does not reduce that
            # mode, it relocates it onto the fine grid.  What is left on
            # the coarse grid decays at the CORE's cut (≥ 3ν, from the
            # edge connectivity of a 3-connected graph).
            sigma_eff_core = _sp_reduced_cut_nu(
                _emap, _root, externals=external) - float(d)
        else:
            sigma_eff_block = min(_inc.values()) - float(d)
            sigma_eff_core = sigma_eff_block
        # σ < 1.49 (per-block) routes tw=2 blocks through the σ_max=4
        # algebra path (analytic Epstein retained for the slow tail)
        # rather than graph_zeta_general (pure improved-Fourier).
        # The threshold is empirical.
        use_algebra_low_sigma = sigma_block_min < (1.49 - 1e-12)

        # Closed-form fast paths.  In nn_mode (ν = inf) these are skipped:
        # epstein_zeta / zeta_circle are undefined at ν = inf, so the block
        # falls through to the NN-kernel tensor path below.
        if n_v == 2 and n_e == 1 and (not nn_blk or low_bridge):
            # Bridge.  In legacy nearest-neighbour mode a bridge whose
            # finite exponent is <= d takes the closed form as well: the
            # torus would return a truncation of the divergent sum
            # (measured 8.21 / -37.51 / -37.72 at n = 8 / 16 / 32 for
            # [inf, 0.9], exact -37.7205), and at the pole a finite
            # number.  A bridge above d keeps the torus there, unchanged.
            info["n_bridges"] += 1
            if b_kern is not None:
                # General interaction: the exact lattice sum of the
                # bundle kernel (Epstein zetas plus the finite table sum).
                value *= _store(float(_finite_bridge(
                    b_kern[0].lattice_sum(A),
                    lambda: [nu for _, nu in b_kern[0].fourier_terms(A)[0]],
                    d)))
                continue
            nu_b = float(b_edges[0][2])
            value *= _store(float(_finite_bridge(
                _epstein_zeta(nu_b, A, zero_d, zero_d).real, [nu_b], d)))
            continue
        _cycle_route = (
            n_v >= 3 and n_v == n_e
            and not (fast_cycles or nn_blk or _large_sigma_cycle(b_edges, d))
        )
        if _cycle_route and d not in _CLOSED_FORM_DIMS:
            # zeta_circle is implemented for d = 1, 2, 3 only.  Any other
            # d takes the sigma-router below, as fast_cycles=True does and
            # as a CycleQuadratureError does.  _make_key applies the same
            # rule, so the value is stored under the sigma key.
            info["n_block_cycle_fallback"] += 1
            _cycle_route = False
        if _cycle_route:
            # Simple cycle.  At zero external momentum every cut vertex
            # carries zero momentum, so the cycle's at-zero scalar is
            # zeta_circle(ν_vec, A) regardless of how many cut vertices
            # the cycle attaches to (block-cut decomposition only
            # connects blocks at single vertices, never at edges).
            # We skip this closed-form quadrature and let the σ-router
            # below handle the cycle as a tw=2 (tensor) block when any of:
            #   * ``fast_cycles=True`` (caller opt-in — useful at d ≥ 3
            #     where the 3D pyramidal Duffy quadrature can dominate);
            #   * σ = min(bundled ν) − d > _FAST_CYCLE_SIGMA (large-σ:
            #     ``zeta_circle`` is slow there);
            #   * ``nn_mode`` (ν = inf: ``zeta_circle`` is undefined).
            # On-spine cycles never reach this branch at finite k:
            # they go to ``_block_at_finite_k`` above, since this
            # scalar form holds only at zero momentum.
            if b_kern is not None:
                # General interactions: the generalised closed form (the
                # per-edge transforms carry the compact parts as exact
                # trigonometric polynomials; see circle.py).  Its
                # self-check ladder may decline an unconverged kernel; the
                # block then takes the sigma-router exactly as
                # ``fast_cycles=True`` would, and is NOT stored under the
                # closed-form key (a discretised value there would poison
                # every later reuse -- the on-spine cycle lesson above).
                try:
                    cyc = float(zeta_circle(list(b_kern), A))
                except CycleQuadratureError:
                    cyc = None
                if cyc is not None:
                    info["n_simple_cycles"] += 1
                    value *= _store(cyc)
                    continue
                info["n_block_cycle_fallback"] += 1
                _pk = None
            else:
                # The plain closed form declines the same way when its
                # d = 1 rule stays non-finite (circle.py,
                # ``_int_full_1d_finite``): same fallback, so the closed
                # form never hands back a NaN.  A cycle whose value lies
                # outside the float64 range still comes back non-finite
                # from the fallback.
                nu_vec = np.array([e[2] for e in b_edges], dtype=float)
                try:
                    cyc = float(zeta_circle(nu_vec, A))
                except CycleQuadratureError:
                    cyc = None
                if cyc is not None:
                    info["n_simple_cycles"] += 1
                    value *= _store(cyc)
                    continue
                info["n_block_cycle_fallback"] += 1
                _pk = None

        # General per-block evaluation.  Two paths:
        #   (a) σ_max=4 algebra (graph_from_edges with per-bundle ν)
        #       when:
        #       * tw(block) == 2,
        #       * σ_block_min < 1.49,
        #       * |external| ≤ 2.
        #     Better at small σ because the slow 1/r^ν tail is kept
        #     analytically as Epstein zetas.  Non-uniform per-edge ν
        #     across the block is fine: Hadamard merging produces one
        #     bundle per simple-graph edge with the summed exponent,
        #     and graph_from_edges accepts that bundle ν vector
        #     directly.
        #   (b) Tensor (graph_zeta_general) otherwise.
        #     Improved-Fourier representation of the block on the
        #     n_points^d torus; identical to (a) at σ_max=0 and
        #     fp-precision at large σ.
        if not nn_blk:
            _require_grid(int(n_points), sub, int(d),
                          lifted=has_compact_block)

        sorted_v = sorted(b_set)
        local = {v: i for i, v in enumerate(sorted_v)}
        if not external:
            external_local = [0]                  # purely internal block
        else:
            external_local = sorted(local[v] for v in external)

        # Treewidth via the min-fill-in heuristic on the block's
        # induced subgraph.  Upper bound; exact for the V ≤ ~12 blocks
        # we encounter in the TFIM-softcore corpora.  Drives both the
        # algebra-vs-tensor decision below and the cost-relevant
        # tw exponent reported in info.
        from networkx.algorithms.approximation import treewidth_min_fill_in
        tw_block, _ = treewidth_min_fill_in(sub)

        scalar = None

        # ----------------------------------------------------------------
        # (3b) Dense-block route.  Treewidth-≥3 blocks (K_4/K_5 minors)
        # cost O(n^((tw+1)·d)) on the tensor path — n^12 for a K_5 at
        # d=2, infeasible.  But the same high connectivity makes their
        # box-truncation tail decay as |x|^-(deg·ν): with the
        # degree-aware Richardson exponent (3a) a *tiny* L already
        # reaches series precision.  Route them to the real-space
        # direct sum instead.  Exact (no n_points grid); falls back to
        # the tensor path on any failure (e.g. ν≤d divergence guard, or
        # MemoryError at d=3 where (2L+1)^(d·tw) can still be large).
        # ----------------------------------------------------------------
        #
        # ``dense_route == "torus"`` skips this arm and leaves
        # ``scalar is None``, so the block reaches the ordinary tensor /
        # hybrid path below — which at k = 0 also carries the σ_eff
        # Richardson ladder, and hybrid's split when a fine grid is
        # configured.  The torus route is a skip, not a new code path.
        dense_torus_block = False
        if tw_block >= 3 and not nn_blk:
            _blk_nu_min = float(min(float(nu_b) for _, _, nu_b in b_edges))
            if _blk_nu_min <= float(d):
                raise UnsupportedLatticeSumError(
                    f"dense block (treewidth {int(tw_block)}) has "
                    f"min nu = {_blk_nu_min} <= d = {int(d)}, which is not "
                    f"supported: some such sums diverge and this engine "
                    f"cannot tell which, while a torus would return a "
                    f"truncation artefact rather than a value.  "
                    f"(Convergence is governed by the cluster cut, not by "
                    f"min nu, so this refusal is conservative -- it "
                    f"declines cases that do converge.)"
                )
            ds_edges = np.array(
                [[local[u], local[v]] for u, v, _ in b_edges], dtype=int,
            )
            ds_nu = np.array(
                [float(nb) for _, _, nb in b_edges], dtype=float,
            )
            # The ν ≤ d case never reaches here -- the raise above takes
            # it on either engine setting -- so this conjunct is
            # belt-and-braces, kept so that reordering the raise cannot
            # silently open the torus to a case neither engine can judge.
            # See the same reasoning in _block_at_finite_k.
            dense_torus_block = (
                dense_route == "torus" and float(ds_nu.min()) > float(d)
            )
            # ...and fitting is not the same as being better there.  The
            # byte budget answers the first question; this answers the
            # second (see _TORUS_MIN_N_BY_CLASS).  Declining sends the
            # block down the box arm immediately below -- or, since the
            # slab arm landed, to the slab.  ("Can only restore the
            # incumbent route" was true when the gate shipped and is not
            # any more: a declined block now has the slab behind it.
            # What remains true is that declining never sends the block
            # somewhere the gate has not measured.)
            if dense_torus_block:
                _pin_local = (int(local[_root]) if _free
                              else (min(external_local) if external else 0))
                # THE EVALUATION'S OWN PIN.  On the vacuum path
                # `hybrid._planner_pin` re-pins the block to the
                # argmin-exponent vertex, and the two can disagree.  Used
                # below by the split predicate and by the slab arm;
                # deliberately NOT by the accuracy gate, which is the
                # subtlety recorded at the gate call itself.
                _split_pin = _pin_local
                if len(external) <= 1:
                    try:
                        _split_pin = int(_planner_pin(
                            ds_edges, ds_nu, _pin_local))
                    except Exception:
                        _split_pin = _pin_local
                # THE GATE READS THE GRADED GRID, NOT n_points.  The
                # connectivity-graded core can only LOWER the grid a block
                # actually runs at, so gating on n_points would admit a
                # block and then evaluate it below its own threshold.
                # Measured once _CORE_KAPPA gained its d = 3 entry: K5 at
                # n_points = 12, 16 and 32 is graded to 8 and lands at
                # 3.075e-07, 0.20x the box -- the exact regression this
                # table exists to prevent.  The two calls below are the
                # ones the router makes later at the grading site, in the
                # same order and at the same pin, so the number gated on
                # is the number evaluated at.
                _n_eff = int(n_points)
                if dense_core_grade and not nn_blk:
                    # A compact block runs at the lifted grid, so the rule
                    # is fed that grid here exactly as at the grading site.
                    _g = _core_n_for_block(
                        (max(int(n_points), int(_lift_blk))
                         if has_compact_block else _n_eff),
                        sigma_eff_block, sigma_eff_core, int(d),
                        sigma_ref=blk_floor,
                    )
                    _g = _core_n_capped(
                        _g, ds_edges, list(range(len(local))), _pin_local,
                        [int(local[v]) for v in external] if external else (),
                        int(d),
                    )
                    if has_compact_block:
                        # The same floor the grading site applies below,
                        # against the same reference grid.
                        _g = _graded_core_with_tables(
                            _g, max(int(n_points), int(_lift_blk)),
                            ds_edges, b_kern,
                        )
                    _n_eff = min(_n_eff, int(_g))
                # The split is ON for this call and has something to
                # reduce on THIS block.
                #
                # EVALUATED AT THE PIN THE EVALUATION USES, not at the
                # router's root.  ``sigma_eff_core`` / ``sigma_eff_block``
                # above are computed at ``_root = min(external)``, while a
                # vacuum block is re-pinned by ``hybrid._planner_pin``, and
                # the two disagree on 172 of 2636 dense-block occurrences
                # in the TFIM corpora -- always in the same direction: 0
                # released that the evaluation does not act on, 172 denied
                # that it does.  V6E11 standalone is one of the 172: its
                # cuts are (4.0, 4.0) at vertex 0, the degree-2 vertex the
                # source pins, and (4.0, 11.0) everywhere else, so reading
                # them at the router's root refuses a block the split takes
                # from 10.6x worse than the box to >= 57.7x better.
                # Re-pinning here claims that back and cannot release
                # anything new, because the census direction is one-sided.
                _emap_at_pin: dict = {}
                for (_u, _v), _w in zip(np.asarray(ds_edges).tolist(),
                                        np.asarray(ds_nu).tolist()):
                    _k = (min(int(_u), int(_v)), max(int(_u), int(_v)))
                    _emap_at_pin[_k] = _emap_at_pin.get(_k, 0.0) + float(_w)
                try:
                    _sb = _min_free_cut_nu(_emap_at_pin, _split_pin) - float(d)
                    _sc = _sp_reduced_cut_nu(_emap_at_pin, _split_pin) - float(d)
                except Exception:
                    _sb, _sc = float(sigma_eff_block), float(sigma_eff_core)
                _split_acts = (dense_sp_n is not None and _sc > _sb + 1e-12)
                # ...AND THE GATE IS KEYED AT THE ROUTER'S ROOT, NOT AT
                # `_split_pin`.  This looks like the pin desync the split
                # predicate fixes and is the opposite case; the two
                # questions have different pins because they are about
                # different engines.
                #
                # The gate asks "would the graded torus CORE beat the
                # box", and the core is pinned where it is pinned.
                # Reclassifying at the planner pin releases 8 of the 13
                # d = 3 shapes that reach here (all 13 are exponent 3 at
                # the root; only 5 are at the planner pin) -- and
                # MEASURED, that release is a REGRESSION, because those
                # blocks then run their graded core at n = 8.  Two of
                # them, against a slab reference:
                #
                #   V5 E9  (K5 less an edge)  1.43e-07 vs the box 1.78e-08
                #   V5 E9  (another)          2.51e-05 vs the box 1.15e-06
                #
                # i.e. 8x and 22x WORSE.  The exponent-2 class at d = 3
                # has no measured minimum grid yet, and until it has one
                # the gate's conservative reading is what stands between
                # those blocks and a graded n = 8 torus.
                #
                # `_slab_grid` below classifies at the PLANNER'S OWN free
                # pin, and that is not an inconsistency: its question is
                # "slab or box", and both of those engines are governed
                # by the free-pin class -- the slab's inner exponent is
                # one below it by construction, and the box's own
                # `_pick_root` independently finds the same pin.
                # Measured, that key separates the 13 shapes' BOX COST
                # exactly: 77-178 s at free-pin exponent 3 against
                # 0.2-1.7 s at exponent 2, no overlap.
                if not _dense_torus_is_accurate(
                    ds_edges, list(range(len(local))), _pin_local,
                    [int(local[v]) for v in external] if external else (),
                    int(d), _n_eff, bool(richardson),
                    split_acts=_split_acts,
                ):
                    dense_torus_block = False
                    info["n_block_dense_torus_declined"] += 1
                    # THE SLAB ARM.  A declined block needs a grid it
                    # cannot afford on the dense core; the slab is the
                    # same truncation associated with a second pin, and
                    # can.  Refusing here -- no measured entry, a
                    # non-cubic cell, a byte-budget refusal, anything --
                    # falls through to the box arm below, which is where
                    # a declined block went before this existed.
                    # A compact part rides the slab like a power law: the
                    # fold is exact with tables (the orbit group is
                    # filtered by table invariance) and the slab grid is
                    # floored at the block's winding-safe grid below.
                    # Declining it sent every d = 3 (3, 3) block with a
                    # table to the box ladder, in floor mode too.
                    _slab = _slab_grid(ds_edges, ds_nu, int(d), A,
                                       nn_mode=nn_blk)
                    if _slab is not None and accuracy == "floor":
                        # FLOOR MODE: the declined block ships the
                        # PASS-GRID truncation like every neighbour --
                        # slab@n_eff is the exact fold of the torus at
                        # the pass grid (same value as raw-at-n, at
                        # n^(2d) memory), so the pass stays linear in
                        # its graph count.  Licensed: K5-class vacuum
                        # raw-8 is 3.1e-07 rel vs the n = 18 ladder
                        # (sigma = 0.5) -- 45x below even the tight
                        # order-9 floor of 1.4e-05.  The class n
                        # (_SLAB_N_BY_CLASS) is the box-dominance
                        # standard and stays the strict-mode route.
                        _slab = (int(_n_eff), _slab[1])
                    if _slab is not None and has_compact_block:
                        if int(_lift_blk) > int(_slab[0]):
                            info["n_block_compact_lift"] += 1
                        _slab = (max(int(_slab[0]), int(_lift_blk)), _slab[1])
                    if _slab is not None:
                        _slab_n, _slab_source = _slab
                        try:
                            scalar = float(_slab_zeta(
                                ds_edges, ds_nu, A, int(_slab_n),
                                source=int(_slab_source),
                                max_bytes=_DENSE_CORE_MAX_BYTES, **kern_kw,
                            )[0])
                            info["n_block_slab"] += 1
                            if tw_block > info["max_tensor_block_tw"]:
                                info["max_tensor_block_tw"] = int(tw_block)
                        except (ValueError, MemoryError):
                            # SlabTooLargeError is a MemoryError, so a
                            # priced refusal lands here with the rest.
                            scalar = None
                # THE PREFERRED SLAB ROUTE (see _SLAB_PREFERRED_RUNGS).
                # An ADMITTED block of a keyed class with slab inner
                # exponent 1 is evaluated at both rungs of the tuple and
                # shipped at the top rung if the pair self-agrees; a
                # disagreement -- the truncation not settled, on a shape
                # nobody measured -- falls to the box, and any refusal
                # (byte budget, a poor cell, a failed price) keeps the
                # incumbent torus route.  This is the fix for the
                # class's shipped defect: these blocks ran the torus at
                # the PASS grid, where n_points = 8 is 3.6x - 58x worse
                # than the box; n = 18 costs less than the box does.
                if dense_torus_block:
                    _pref = _slab_preferred(ds_edges, ds_nu, int(d), A,
                                            nn_mode=nn_blk)
                    if (_pref is not None and has_compact_block
                            and int(_pref[0][0]) < int(_lift_blk)):
                        # The lower rung would wind the block's tables:
                        # the incumbent torus route keeps the block.
                        _pref = None
                    if _pref is not None:
                        (_r_lo, _r_hi), _pref_pin = _pref
                        try:
                            _s_lo = float(_slab_zeta(
                                ds_edges, ds_nu, A, int(_r_lo),
                                source=_pref_pin,
                                max_bytes=_DENSE_CORE_MAX_BYTES, **kern_kw)[0])
                            _s_hi = float(_slab_zeta(
                                ds_edges, ds_nu, A, int(_r_hi),
                                source=_pref_pin,
                                max_bytes=_DENSE_CORE_MAX_BYTES, **kern_kw)[0])
                        except (ValueError, MemoryError):
                            _s_lo = _s_hi = None
                        if _s_hi is not None:
                            _self_band = (abs(_s_hi - _s_lo)
                                          / max(abs(_s_hi), 1e-300))
                            dense_torus_block = False
                            if _self_band <= _SLAB_SELF_BAND_MAX:
                                # Ship the PAIR, not the raw top rung:
                                # both rungs are already computed, so the
                                # one-term Richardson at the exact basis
                                # p = min_free_cut_nu - d is free, and it
                                # is measured 3.4x - 84x more accurate
                                # than raw n = 18 on every rung-complete
                                # shape (8 distinct topologies + V6E11,
                                # all resolved against slab-20-anchored
                                # references; K4 12.2x, prism 3.4x,
                                # K33 6.2x, K5-e-class 46x, three further
                                # corpus shapes 19x, 10x, 10x, V6E11 84x).
                                # The self-band tripwire above still guards
                                # the unmeasured remainder per block.
                                _pmap: dict = {}
                                for (_u2, _v2), _nb in zip(
                                        np.asarray(ds_edges).tolist(),
                                        np.asarray(ds_nu).ravel()):
                                    _k2 = (min(_u2, _v2), max(_u2, _v2))
                                    _pmap[_k2] = (
                                        _pmap.get(_k2, 0.0) + float(_nb))
                                _pp = float(_elimination.min_free_cut_nu(
                                    _pmap, int(_pref_pin), ())) - int(d)
                                _wl = float(_r_lo) ** (-_pp)
                                _wh = float(_r_hi) ** (-_pp)
                                scalar = ((_s_hi * _wl - _s_lo * _wh)
                                          / (_wl - _wh))
                                info["n_block_slab_preferred"] += 1
                                if tw_block > info["max_tensor_block_tw"]:
                                    info["max_tensor_block_tw"] = int(tw_block)
                            else:
                                # Tripwire fired: scalar stays None and
                                # the box arm below takes the block.
                                info["n_block_slab_selfband_declined"] += 1
            L_list = _DENSE_DSUM_L_LIST.get(int(d), (2, 3, 4))
            K_corr = min(3, len(L_list) - 1)
            if not dense_torus_block and scalar is None:
                try:
                    scalar = float(
                        direct_sum_extrapolated(
                            ds_edges, ds_nu, A,
                            L_list=L_list, n_correction_terms=K_corr,
                            **kern_kw,
                        ).real
                    )
                    info["n_block_direct_sum"] += 1
                    if tw_block > info["max_tensor_block_tw"]:
                        info["max_tensor_block_tw"] = int(tw_block)
                except UnsupportedLatticeSumError:
                    raise    # never hand an unsupported sum to the torus
                except (ValueError, MemoryError):
                    # d=3 memory wall -> tensor.  (nu <= d never gets
                    # here: it is refused before routing.)
                    scalar = None

        algebra_eligible = (
            scalar is None
            and tw_block == 2
            and use_algebra_low_sigma
            and len(external) <= 2
        )

        if algebra_eligible:
            # Hand each bundle to graph_from_edges as a single edge with
            # the bundle's summed ν.  No re-expansion needed; per-edge ν
            # within the block is fine because graph_from_edges accepts
            # per-edge ν.
            try:
                local_edges_flat = np.array(
                    [[local[u], local[v]] for u, v, _ in b_edges],
                    dtype=int,
                )
                local_nu_vec = np.array(
                    [float(nu_b) for _, _, nu_b in b_edges], dtype=float,
                )
                if len(external_local) >= 2:
                    s_alg, t_alg = external_local[0], external_local[1]
                else:
                    # 0 or 1 external → vacuum-like at the cut vertex
                    s_alg = t_alg = external_local[0]
                # The algebra composes compact tables by cyclic
                # convolution: its grid must exceed the sum of the radii
                # (it costs n^d, so the lift is free), and it keeps the
                # singular tail analytic where the tensor below would
                # truncate it.
                _n_alg = (max(int(n_points), _sumR_blk + 1)
                          if has_compact_block else int(n_points))
                g_block = graph_from_edges(
                    local_edges_flat, local_nu_vec, A, _n_alg,
                    s=s_alg, t=t_alg, sigma_max=SIGMA_MAX_ALG, **kern_kw,
                )
                cand, condition = graph_zero_conditioned(g_block)
                # Two refusals, both cheap and both measured.  ζ_B(0) is a
                # sum of non-negative kernel products, so a non-positive
                # value is proof the float64 cancellation destroyed the
                # answer — this fires on deep chains at small σ, where the
                # algebra otherwise returns a confidently negative number
                # for a manifestly positive lattice sum.  The conditioning
                # bound catches the same failure before it flips sign.
                if cand <= 0.0 or condition > _ALGEBRA_MAX_CONDITION:
                    info["n_block_algebra_refused"] += 1
                    scalar = None
                else:
                    scalar = float(cand)
                    info["n_block_algebra"] += 1
                    if tw_block > info["max_algebra_block_tw"]:
                        info["max_algebra_block_tw"] = int(tw_block)
            except NotTreewidthTwoError:
                # Min-fill heuristic gave tw=2 but SP-reduction fails:
                # fall through to tensor.
                scalar = None
            except InteractionSupportError:
                # The block's compact parts compose past the pass grid's
                # window (or n_points cannot hold a table): the tensor arm
                # below lifts the block grid to n_v * R + 2 instead.
                scalar = None

        if scalar is None:
            # Tensor path (graph_zeta_general on the block subgraph).
            local_edges = np.array(
                [[local[u], local[v]] for u, v, _ in b_edges], dtype=int,
            )
            nu_vec = np.array([nu_b for _, _, nu_b in b_edges], dtype=float)
            source = external_local[0]
            terms = tuple(external_local[1:])

            # The split is scoped to DENSE blocks on the torus route.  A
            # tw ≤ 2 block is fully SP-reducible, so collapsing it at the
            # fine grid would evaluate essentially the whole block at
            # sp_n_points and restrict down — a large, unmeasured change
            # to the tw ≤ 2 numbers this work has no evidence for.
            #
            # ≥ 2 kept terminals go to graph_zeta_general, which has no
            # split; count them so the uncovered share is visible rather
            # than inferred.
            block_sp_n = dense_sp_n if dense_torus_block else None
            if dense_torus_block:
                info["n_block_dense_torus"] += 1
                if len(terms) > 1:
                    info["n_block_dense_multi_ext"] += 1
                    block_sp_n = None
                elif block_sp_n is not None:
                    info["n_block_split"] += 1

            # Helper: evaluate this tensor block at an arbitrary
            # discretisation count.  Used twice in the Richardson
            # path (n_points and n_points // 2) and three times when
            # self-tuning is enabled (also n_points // 4).  Returns
            # the real part of the (0,…,0)-momentum cell.
            def _eval_tensor_at(npts):
                # 0 or 1 external terminal: the k = 0 cell is the vacuum
                # scalar (∑ over all positions with the source pinned), so
                # the hybrid vacuum engine gives it directly.  ≥ 2
                # terminals: not yet supported by the hybrid → tensor.
                # The vacuum scalar ζ_B(0) is pin-independent by torus
                # translation invariance — external_local[0] was only
                # ever a determinate choice, not a semantic one (for a
                # purely internal block it is the fabricated label 0)
                # — so the engine gets source=None and the planner
                # chooses the pin.  Finite-k and kept-terminal arms
                # keep their explicit pins: there the layout encodes
                # x_source = 0.
                #
                # ``block_sp_n`` is held FIXED across Richardson rungs
                # rather than scaled with ``npts``.  The ladder is then
                # extrapolating the CORE's convergence at a frozen SP
                # accuracy, which is the intent: the SP part is already
                # far more converged than the core.  It also keeps the
                # ``sp_n_points >= n_points`` invariant automatically,
                # since the rungs only ever go down.  Fixed-vs-scaled is
                # an open measurement question and is why this is stated
                # rather than left implicit.
                if len(terms) <= 1:
                    arr = _block_general(
                        local_edges, nu_vec, A, npts,
                        source=None,
                        terminal=source,
                        mode="vacuum", engine=engine,
                        sp_n_points=block_sp_n, **kern_kw,
                    )
                else:
                    arr = graph_zeta_general(
                        local_edges, nu_vec, A, npts,
                        source=source, terminals=terms, space='k',
                        **kern_kw,
                    )
                    arr = np.asarray(arr).reshape(-1)[0]
                return float(np.asarray(arr).real)

            # In nn_mode use a per-block n, and ONLY the block's own: a
            # purely compact block (the NN indicator, R = 1, or any
            # table of radius R) is exact on its smallest exact torus
            # ``_compact_exact_grid`` -- the longest fundamental cycle of
            # its best spanning tree times R, plus one -- whatever the
            # pass grid asks for, so it runs there, below n_points as
            # readily as above it (the value is the same finite sum on
            # every admissible torus; a pass grid above it bought
            # nothing but n^(tau d) work).  Per-*block* n means a long
            # cycle and a dense block never multiply their grid sizes
            # (no n^((tw+1)·d) blow-up), and it lets nn_mode run with
            # the caller's n_points = 0.  A MIXED block with a compact
            # part is lifted to the winding-safe grid n_v * R + 2
            # instead: below it the all-table term of every cycle of
            # length up to n_v winds around the torus and contributes a
            # spurious O(a^E) term that no power-law tail model removes
            # (measured on the NN indicator: C4 33 % high at n = 4), and
            # the algebra route's composed support wraps likewise.  At
            # k = 0 the block value is a scalar, so a per-block grid costs
            # nothing in bookkeeping — exactly the nn_mode arrangement.
            if nn_exact_blk:
                block_n = _compact_exact_grid(local_edges, _R_blk)
            elif has_compact_block or nn_blk:
                # Not provably exact on any fixed torus, so keep the rule
                # that only ever RAISES the grid and leaves the value
                # convergent in n_points.  For a legacy block this is
                # ``max(n_points, n_v + 2)`` -- ``_lift_blk`` reduces to
                # exactly that when the block carries no table -- which is
                # the pre-existing behaviour for every mixed block.
                block_n = max(int(n_points), int(_lift_blk))
            else:
                block_n = int(n_points)
            if has_compact_block and block_n > int(n_points):
                info["n_block_compact_lift"] += 1

            # Connectivity-graded core.  At k = 0 the block value is a
            # SCALAR, so there is no delivery grid to match and the core
            # simply runs at its own size — no embedding, and hybrid's
            # ``core_n_points`` is not needed here at all: handing
            # ``_eval_tensor_at`` a smaller n with ``block_sp_n`` held
            # fixed above it IS the coarse core.  (The finite-k arm does
            # need the parameter, because there the value has to come
            # back on the caller's BZ grid.)
            #
            # DENSE BLOCKS ONLY here, on measurement -- unlike the
            # finite-k arm, which offers the graded core to every
            # hybrid-bound block.  A tw ≤ 2 block at k = 0 is
            # SP-collapsed against a FREE pin with nothing kept (hybrid
            # re-pins the vacuum call), so its residual is an O(n^d)
            # core with nothing to save, and the rates above -- taken at
            # the external pin -- do not describe that contraction.
            # Grading it anyway moved the order-9 d = 2 pass by 2.2e-06
            # at n = 64, above the pass floor, for 1.00x speed.
            graded_core = False
            # A block with a compact part is graded like a pure one, with
            # its core floored at ``_compact_core_floor``: the residual
            # tables then stay exact on the core window and no core cycle
            # winds, so what the coarse core truncates is the tail alone
            # -- the same power-law term the kappa rule was calibrated
            # on.  (An earlier measurement that declined every compact
            # block -- a core of 6 on a K4 at n = 8, 21x worse than the
            # raw grid -- sat BELOW that floor, where the rule never
            # grades: ``_CORE_N_FLOOR`` is 8.)
            if dense_core_grade and dense_torus_block and not nn_blk:
                n_graded = _core_n_for_block(
                    block_n, sigma_eff_block, sigma_eff_core, int(d),
                    sigma_ref=blk_floor,
                )
                # local_edges / local labels: `plan` needs (u, v) pairs on
                # a contiguous 0..n_v-1 labelling, not the block's original
                # vertex ids nor b_edges' (u, v, nu) triples.
                # THE SAME GUARD `_pin_local` USES.  `_root` is assigned
                # only inside `if _free:` above, so on a block whose every
                # vertex is external -- reachable: a K5 with a pendant
                # triangle on each vertex makes all five cut vertices --
                # it is left over from a PREVIOUS block of the same graph.
                # That does not raise (the stale label is often still a
                # vertex here, so `local[_root]` resolves) and instead
                # sizes the core at a pin the contraction does not use,
                # which is the exact desync `_core_n_capped`'s own
                # docstring warns about.  Latent before, because d = 3 had
                # no `_CORE_KAPPA` entry and so never reached this site;
                # shipping one is what makes it reachable.
                _grade_pin = (int(local[_root]) if _free
                              else (min(int(local[v]) for v in external)
                                    if external else 0))
                n_graded = _core_n_capped(
                    n_graded, local_edges, list(range(len(local))),
                    _grade_pin,
                    [int(local[v]) for v in external] if external else (),
                    int(d),
                )
                if has_compact_block:
                    n_graded = _graded_core_with_tables(
                        n_graded, block_n, ds_edges, b_kern,
                    )
                if n_graded < block_n:
                    block_n = n_graded
                    graded_core = True
                    info["n_block_core_graded"] += 1
                    if has_compact_block:
                        info["n_block_core_graded_compact"] += 1
            try:
                T_full = _eval_tensor_at(block_n)
            except MemoryError:
                # Last resort, and only for a dense block WE diverted
                # here.  hybrid has already degraded to the tensor inside
                # _block_general and the tensor has run out too.  The box
                # is the one engine whose cost does not grow with
                # n_points, so it keeps the d = 3 / tw ≥ 4 capability the
                # torus cannot hold in memory.  Everywhere else a
                # MemoryError is a real allocation failure and must
                # propagate rather than become a quietly different value.
                #
                # ds_edges / ds_nu / L_list / K_corr are in scope: the
                # tw ≥ 3 arm above is the only thing that can set
                # dense_torus_block, and it defines all four.
                if not dense_torus_block:
                    raise
                info["n_block_dense_torus"] -= 1
                if block_sp_n is not None:
                    info["n_block_split"] -= 1
                if len(terms) > 1:
                    info["n_block_dense_multi_ext"] -= 1
                if has_compact_block and not nn_blk:
                    # A compact block keeps its full grid (no graded core), so
                    # the dense core can run out where the legacy path's graded
                    # core would not.  The slab is the same truncation with a
                    # second pin at n^(2d) memory and is exact with tables:
                    # ship it at the pass grid floored at the lift before
                    # reaching for the box (measured: a compact K5 on the cubic
                    # cell at n = 16 went to the box ladder here).
                    _sl = _slab_grid(ds_edges, ds_nu, int(d), A, nn_mode=nn_blk)
                    if _sl is not None:
                        try:
                            scalar = float(_slab_zeta(
                                ds_edges, ds_nu, A,
                                max(int(n_points), int(_lift_blk)),
                                source=int(_sl[1]),
                                max_bytes=_DENSE_CORE_MAX_BYTES, **kern_kw,
                            )[0])
                            info["n_block_slab"] += 1
                            if tw_block > info["max_tensor_block_tw"]:
                                info["max_tensor_block_tw"] = int(tw_block)
                            value *= _store(scalar)
                            continue
                        except (ValueError, MemoryError):
                            pass
                try:
                    scalar = float(
                        direct_sum_extrapolated(
                            ds_edges, ds_nu, A,
                            L_list=L_list, n_correction_terms=K_corr,
                            **kern_kw,
                        ).real
                    )
                except InteractionSupportError as exc:
                    raise MemoryError(
                        f"the torus ran out of memory on a dense block and "
                        f"the box ladder {L_list} cannot hold the block's "
                        f"compact reach: {exc}"
                    ) from exc
                info["n_block_direct_sum"] += 1
                if tw_block > info["max_tensor_block_tw"]:
                    info["max_tensor_block_tw"] = int(tw_block)
                value *= _store(scalar)
                continue
            scalar = T_full

            # Richardson on the tensor block.  Off by default (kwarg
            # richardson=False).  When on, the block is evaluated on a
            # descending ladder of torus sizes and the sequence is
            # extrapolated to n -> infinity with the *derived* leading
            # exponent σ_eff (see its definition above): the torus
            # truncation decays as n^-σ_eff, so that basis — plus the
            # integer-spaced Euler-Maclaurin corrections — removes the
            # tail rather than merely reducing it.
            #
            # This matters most exactly where the raw torus is weakest:
            # at small σ_eff the raw value converges only as n^-σ_eff
            # (a degree-1 free vertex at ν = d + 0.5 is still ~3e-2
            # off at n = 256), while the extrapolated value reaches
            # ~1e-11 on the same ladder.
            #
            # p_richardson overrides the exponent.  The lower rungs
            # cost a small fraction of the top one (a rung at n/2 is
            # 2^-(bag·d) of it), so the ladder is roughly 2x the
            # single evaluation.
            #
            # k = 0 only, deliberately.  At finite k the tail carries
            # an oscillatory cos(2π k·x) factor, so it is no longer a
            # clean power law and the σ_eff basis stops describing it:
            # fitting the exponent of a 5-cycle at ν = 1.5 across
            # k = 0, 0.05, 0.17, 0.31, 0.5 gives 1.79, 4.64, 2.02,
            # 0.79, 1.91 against σ_eff = 2.  Finite-k blocks are routed
            # through _block_at_finite_k and must stay unextrapolated
            # unless that k-dependence is modelled.
            # Two usable rungs is the minimum for a fit; the ladder
            # steps by 3/4, so that is reached well below the n a d = 2
            # or d = 3 block can afford.  The gate must admit the n a
            # d = 3 block can actually pay for (8-10), or the ladder is
            # dead in exactly the dimension that needs it most.
            _min_top = (2 * _RICHARDSON_MIN_N if d <= 1
                        else 2 * _RICHARDSON_MIN_N_HIGH_D)
            if graded_core and _core_term_below_sp_floor(
                    block_n, sigma_eff_core, block_sp_n, sigma_eff_block):
                # The ladder removes A_core·n^-σ_core and NOTHING ELSE.
                # Measured: with sp held above every rung the SP
                # residual is a CONSTANT offset across the ladder, and no
                # Richardson fit removes a constant (it is an
                # n_c-independent one-signed additive offset, so it
                # becomes the floor).  So once grading has pushed the
                # core term below that floor there is nothing left to
                # take, and eliminating a term that is not there injects
                # noise instead.
                #
                # MEASURED here, d = 2, σ = 0.5, exact-exponent two-point
                # elimination on graded cores, split by which term
                # dominates (SP floor sp^-σ_blk = 7.5e-09 at sp = 512):
                #
                #   core term n^-σ_core   gain over raw   n
                #     7.0e-08  (ABOVE)      13.4-15.5x    3/3 win
                #     1.0e-08  (below)       0.0-2.5x     4/6 LOSE
                #     2.3e-09  (below)       0.8-4.9x     3/5 lose
                #
                # Both sides of the comparison are ANALYTIC — σ_core is
                # exact for ν > d (brute-force verified) and σ_blk is the
                # block cut — so this is a gate, not a fit.
                info["n_block_richardson_skipped_graded"] += 1
            elif (has_compact_block and (
                    nn_blk or min(_richardson_rungs(block_n, int(d)),
                                   default=block_n) < int(_lift_blk))):
                # A purely compact block is EXACT on its torus (no
                # power-law tail to extrapolate), and a mixed block's
                # rungs must all sit at or above its winding-safe grid:
                # a rung below it carries the all-table winding term of
                # a cycle, which is not a power-law tail and must not be
                # fitted (measured on K4 with a subdivided edge, R = 1,
                # n = 8: the rungs [8, 6, 4] against a lift of 7 gave an
                # accepted ladder 6x WORSE than the raw grid).  Either way
                # the ladder has nothing valid to fit; counted apart from
                # a guard refusal.
                if richardson:
                    info["n_block_richardson_skipped_window"] += 1
            elif richardson and block_n >= _min_top:
                # WHICH BASIS depends on whether the split is on, and
                # getting it wrong is not a small loss: the ladder fits
                # an n^-σ tail, and with the split that tail is no longer
                # there.  Measured at the shipped config (d = 1, σ = 0.5,
                # sp = 1024), the block basis makes the guards refuse on
                # 4 of 4 blocks where the split acts — correctly, since
                # they are being asked to fit a mode that was relocated
                # onto the fine grid.  The core basis fits all of them.
                #
                # The two are complements, not substitutes:
                #     err(n, sp) ≈ A_blk·sp^-σ_blk + A_core·n^-σ_core
                # the split shrinks the first term, the ladder removes
                # the second.  On an acting block at d = 1, σ = 0.5,
                # n = 64, against a box reference: ladder alone 7.2e-07,
                # split alone (sp = 16384) 4.5e-07, both 2.8e-08.
                #
                # On a block with nothing to reduce the two cuts are
                # equal by construction, so this is a no-op there.
                _p_basis = (sigma_eff_core if block_sp_n is not None
                            else sigma_eff_block)
                p_used = (float(p_richardson) if p_richardson is not None
                          and p_richardson > 0 else float(_p_basis))
                if np.isfinite(p_used):
                    v_inf, _why = _richardson_ladder(
                        _eval_tensor_at, block_n, int(d), p_used, T_full,
                    )
                else:
                    v_inf = None          # a mixed block whose every cut is compact
                if v_inf is not None:
                    scalar = v_inf
                    info["n_block_richardson_theory"] += 1
                else:
                    info["n_block_richardson_skipped"] += 1
            elif richardson:
                # User asked for Richardson but the σ or n_points
                # guards rejected this block.  Logged as skipped.
                info["n_block_richardson_skipped"] += 1

            info["n_block_tensor"] += 1
            if nn_blk:
                info["n_block_nn"] += 1
            if tw_block > info["max_tensor_block_tw"]:
                info["max_tensor_block_tw"] = int(tw_block)

        value *= _store(scalar)

    return _ret(value)


# ---------------------------------------------------------------------------
# Public front-end
# ---------------------------------------------------------------------------

def _multigraph_to_arrays(mg) -> "tuple[np.ndarray, np.ndarray]":
    """Convert a NetworkX (Multi)Graph with integer node labels in
    ``[0, V)`` and ``'nu'`` edge attributes to ``(edges_flat, nu_arr)``.

    Raises ``ValueError`` if the labels are not integers in ``[0, V)``
    or if any edge is missing the ``'nu'`` attribute.
    """
    # Labels are NAMES (the vertex set is the edge support, see
    # gzl/_labels.py), so the only label requirement is
    # non-negative integers — the old ``[0, n_nodes)`` contiguity gate
    # refused sparse labellings the edge-list path accepts, and fired
    # before the isolated-node refusal below whenever the isolated
    # node's label fell outside the range.
    bad = [v for v in mg.nodes()
           if not isinstance(v, (int, np.integer)) or int(v) < 0]
    if bad:
        sample = bad[:3]
        more = "..." if len(bad) > 3 else ""
        raise ValueError(
            f"evaluate_graph: MultiGraph node labels must be non-negative "
            f"integers; got non-conforming labels {sample!r}{more}.  "
            f"Use networkx.convert_node_labels_to_integers(mg) to relabel."
        )

    # A NetworkX graph, unlike an edge list, CAN express an isolated
    # vertex — and an isolated vertex makes the infinite-lattice zeta
    # diverge (its escaping cluster cut is 0), so it is refused rather
    # than silently dropped in the conversion to an edge list.  Before
    # this check, an isolated node with an interior label was refused
    # with a misleading "disconnected" message while one carrying the
    # largest label vanished from the result entirely.
    isolated = [v for v in mg.nodes() if mg.degree(v) == 0]
    if isolated:
        raise DisconnectedGraphError(
            f"evaluate_graph: MultiGraph has isolated node(s) "
            f"{isolated[:3]!r}{'...' if len(isolated) > 3 else ''}: an "
            f"isolated vertex is its own component and its lattice-sum "
            f"factor diverges on the infinite lattice.  Remove it, or "
            f"evaluate the components separately."
        )

    edges_flat = []
    nu_list = []
    for u, v, data in mg.edges(data=True):
        if "nu" not in data:
            raise ValueError(
                f"evaluate_graph: edge ({u!r}, {v!r}) is missing the "
                f"required 'nu' edge attribute."
            )
        edges_flat.append([int(u), int(v)])
        # An Interaction-like edge attribute is kept as the object; the
        # float path is unchanged.
        nu_list.append(data["nu"] if is_interaction(data["nu"])
                       else float(data["nu"]))

    if edges_flat:
        if any(is_interaction(x) for x in nu_list):
            return np.asarray(edges_flat, dtype=int), nu_list
        return (
            np.asarray(edges_flat, dtype=int),
            np.asarray(nu_list, dtype=float),
        )
    return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=float)


def _is_multigraph(obj) -> bool:
    """True if ``obj`` is a NetworkX (Multi)Graph; False otherwise.

    Returns False if NetworkX is not installed (the array path then
    handles the input)."""
    if not _TOPO_AVAILABLE:
        return False
    return isinstance(obj, (_nx.MultiGraph, _nx.Graph))


def _normalise_terminal(terminal, source: int) -> "int | None":
    """Parse the public ``terminal`` argument into a single int or
    ``None``.

    * ``None`` → vacuum (``None``).
    * ``int`` → that vertex label.
    * length-0 array-like → vacuum (``None``).
    * length-1 array-like → unwrap to int.
    * length ≥ 2 → :class:`UnsupportedRequestError`, a
      ``NotImplementedError``.

    A label must be a whole number (see :func:`gzl._labels.vertex_label`).
    """
    if terminal is None:
        return None
    if isinstance(terminal, (int, np.integer, float, np.floating)):
        return vertex_label(terminal, "terminal", "evaluate_graph")
    arr = np.asarray(terminal).reshape(-1)
    if arr.size == 0:
        return None
    if arr.size == 1:
        return vertex_label(arr[0], "terminal", "evaluate_graph")
    raise UnsupportedRequestError(
        "evaluate_graph: multi-terminal evaluation "
        f"(len(terminal) = {arr.size}) is not implemented; pass a single "
        "vertex label or an array of length ≤ 1."
    )


def _finite_momentum(arr: np.ndarray, momentum, where: str) -> np.ndarray:
    """``arr``, refused with ``ValueError`` if an entry is NaN or infinite.

    A non-finite momentum is malformed.  Unchecked, it gave NaN from the
    torus engines, a GraphZetaError claiming an epsteinlib range problem
    from the bridge closed form, and a number from a vacuum graph, which
    does not read the momentum at all.
    """
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{where}: momentum must be finite; got {momentum!r}"
        )
    return arr


def _grid_size(n_points, where: str) -> int:
    """``n_points`` as a non-negative ``int``.

    A whole number only: ``int(2.5)`` used to evaluate at ``n_points = 2``
    without a word.  An integral float is accepted.  Whether the grid is
    large enough for a block is decided per block (:func:`_require_grid`).
    """
    if isinstance(n_points, np.ndarray) and n_points.ndim == 0:
        n_points = n_points[()]
    if isinstance(n_points, (bool, np.bool_)) or not isinstance(
            n_points, (int, np.integer, float, np.floating)):
        raise TypeError(
            f"{where}: n_points must be an integer; got {n_points!r}"
        )
    if not (np.isfinite(n_points) and float(n_points).is_integer()):
        raise ValueError(
            f"{where}: n_points must be a whole number; got {n_points!r}"
        )
    if n_points < 0:
        raise ValueError(
            f"{where}: n_points must be >= 0 (0 means no grid); got "
            f"{n_points!r}"
        )
    return int(n_points)


def _normalise_momentum(momentum, d: int) -> "tuple[str, np.ndarray | None]":
    """Parse the public ``momentum`` argument into a ``(mode, k)`` pair.

    Modes:

    * ``"none"`` — `momentum is None`; vacuum / k = 0 path.
    * ``"single"`` — single k-vector; ``k`` has shape ``(d,)``.
    * ``"batch"`` — batch of k-vectors; ``k`` has shape ``(N, d)``.

    A scalar input is interpreted as a single k-vector in ``d = 1``;
    a 1-D array of length ``N`` (with ``d = 1``) is a batch.  In
    higher dimensions, length-``d`` 1-D arrays are single, and
    ``(N, d)`` 2-D arrays are batches.
    """
    if momentum is None:
        return "none", None
    if isinstance(momentum, (int, float)):
        if d != 1:
            raise ValueError(
                f"evaluate_graph: scalar momentum is accepted only "
                f"in d = 1; got d = {d}."
            )
        return "single", _finite_momentum(
            np.asarray([float(momentum)], dtype=float), momentum,
            "evaluate_graph")
    arr = _finite_momentum(np.asarray(momentum, dtype=float), momentum,
                           "evaluate_graph")
    if arr.ndim == 1:
        if arr.size == d:
            return "single", arr
        if d == 1:
            return "batch", arr.reshape(-1, 1)
        raise ValueError(
            f"evaluate_graph: 1-D momentum of length {arr.size} is "
            f"ambiguous for d = {d}; pass shape ({d},) for a single "
            f"k or shape (N, {d}) for a batch."
        )
    if arr.ndim == 2 and arr.shape[1] == d:
        return "batch", arr
    raise ValueError(
        f"evaluate_graph: momentum has unsupported shape {arr.shape} "
        f"for d = {d}; expected None, ({d},), or (N, {d})."
    )


def evaluate_graph(
    graph,
    nu=None,
    A=None,
    *,
    source: "int | None" = None,
    terminal=None,
    momentum=None,
    n_points: int = 0,
    richardson: bool = True,
    fast_cycles: bool = False,
    block_cache: "dict | None" = None,
    return_diagnostics: bool = False,
    engine: str = "hybrid",
    dense_engine: "str | None" = None,
    sp_n_points: "int | None" = None,
    core_grading: bool = True,
    accuracy: "str | None" = None,
) -> "float | np.ndarray | tuple":
    r"""Evaluate ζ_G via the topology-first router, at zero or at finite
    external momentum.

    Two input forms are supported:

    * **Flat arrays** — ``evaluate_graph(edges_flat, nu, A, *, ...)``.
    * **NetworkX MultiGraph** — ``evaluate_graph(multigraph, A, *, ...)``.

    The result is real-valued at any momentum (Bravais lattices are
    closed under x → -x, so ζ_G(k) = ζ_G(-k) ∈ ℝ).

    ``graph``, ``nu``, ``A``, ``source``, ``terminal``, ``momentum``,
    ``n_points`` and ``return_diagnostics`` are stable API.  The other
    parameters, ``richardson``, ``fast_cycles``, ``block_cache``,
    ``engine``, ``dense_engine``, ``sp_n_points``, ``core_grading`` and
    ``accuracy``, select or tune the numerical method and are
    provisional: they may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Parameters
    ----------
    graph, nu, A
        Either flat arrays or a MultiGraph
        with `nu` edge attributes.  ``A`` is the lattice matrix, whose
        columns are the primitive vectors, or the name of a lattice --
        ``"chain"``, ``"square"``, ``"triangular"``, ``"cubic"``.
        ``nu`` is "nu-like": a float (every edge), a per-edge float
        array, an :class:`~gzl.Interaction` (every edge), or a
        per-edge sequence mixing floats and Interactions; a MultiGraph's
        ``nu`` attributes may be Interactions too.  A plain
        ``Interaction(b=[1], nu=[nu])`` is demoted to the float ``nu``
        (the legacy path, byte for byte).  Routing reads each
        bundle's tail exponent ``min_j nu_j`` (``+inf`` for a purely
        compact kernel, which takes the nearest-neighbour routes); a block
        carrying a compact part of support radius ``R`` runs every torus
        grid at or above its winding-safe size ``n_v * R + 2`` (counted in
        ``n_block_compact_lift`` when that exceeds ``n_points``), keeps the
        split, the pinned tiers and the box, takes the graded core with its
        core grid floored at the size that holds every SP-reduced residual
        table without winding (counted in ``n_block_core_graded_compact``),
        and is counted in ``n_block_mixed_kernel`` when a power-law tail
        sits beside the table.
    source
        Pinned source vertex.  Default ``None``: the smallest vertex
        label present in the graph (0 for a contiguous labelling — the
        historical default; the vacuum value is pin-independent).
    terminal
        Free terminal vertex (1qp).  ``None`` (default) or
        ``terminal == source`` ⇒ vacuum (k-independent).
        Accepts an ``int`` or an array-like of length ≤ 1; arrays of
        length ≥ 2 raise :class:`UnsupportedRequestError`, which is a
        ``NotImplementedError``.
    momentum
        External momentum k in fractional Brillouin-zone coordinates
        (same convention as :func:`graph_sample` / `graph_zeta_general(...,
        space='k')`).  Modes:

        * ``None`` (default) — a vacuum graph returns its single
          ``float``; a 1qp graph (``terminal != source``) returns the
          full Brillouin-zone grid, a real array of shape
          ``(n_points,) * d`` whose cell ``i`` holds
          ``k = i / n_points`` (per axis).
        * ``scalar`` (d = 1) or ``(d,)`` ndarray — single k-vector;
          returns ``float`` ζ_G(k).
        * 1-D array (d = 1) or ``(N, d)`` ndarray — batch of N
          k-vectors; returns shape-``(N,)`` real array.
    n_points
        Discretisation for σ-routed blocks (algebra / tensor), a whole
        number.  ``0`` (default) is fine for graphs that evaluate
        analytically.  The ``n_points ** d`` sites of the torus must
        hold every block that is not evaluated in closed form, with the
        two ends of every edge on distinct sites.
    return_diagnostics
        If ``True``, return ``(value, info)``.  The keys and values of
        the ``info`` dict (route counts per block) are meant for
        inspection and are not part of the stable API.
    richardson
        If ``True`` (default), extrapolate torus-routed blocks evaluated
        at ``k = 0`` (vacuum graphs and blocks off the spine) along a
        ladder of grid sizes to remove the leading truncation tail, at
        the block's cluster-cut exponent; the fit is kept only when every
        guard of ``_richardson_ladder`` passes.  Blocks that see a
        momentum, an explicit ``k = 0`` on the spine included, are not
        extrapolated, nor are graded cores and compact blocks whose
        ladder would reach below their winding-safe grid.
    fast_cycles
        If ``True``, simple-cycle blocks are routed through the
        σ-router (algebra at σ < 1.49 / tensor otherwise) instead of
        the closed-form :func:`zeta_circle` quadrature.  Useful at
        d ≥ 3 where the 3-D pyramidal Duffy quadrature in
        :func:`zeta_circle` can dominate runtime; the trade-off is
        the algebra/tensor path's ``n_points^d`` discretisation
        residual replacing the quadrature residual.  Default
        ``False`` (most accurate).
    block_cache
        Optional mutable ``dict`` used to memoize per-biconnected-block
        values across calls.  Block-cut already factors
        ζ_G = Π ζ_block; the same block (topology + per-bundle ν +
        lattice + discretisation + momentum) recurs across thousands of
        corpus graphs, so passing one shared dict makes each distinct
        block evaluate once per corpus pass.  WL-hash keys are
        isomorphism-verified (ν- and boundary-role matched), so no
        non-equivalent block is ever conflated and a cached pass is
        deterministic and reproducible.  The reused value is an
        isomorphic representative's; at k = 0 the exact ζ_block is
        labelling-independent, so it and a fresh evaluation approximate
        the *same* number.  For bridge / cycle / tensor blocks they
        differ only by floating-point reassociation (~1e-13 rel).  For
        σ_max=4 algebra blocks (σ < 1.49) the difference is the
        labelling-dependent part of the finite-``n_points``
        discretisation error (e.g. ~1e-4 at n=32, σ≈1) — both are
        equally valid O(n^-(2-σ)) approximants, so series accuracy
        stays within the method's own error bar; the cache does not add
        error beyond the discretisation already accepted.  Default
        ``None`` (no caching — single-call behaviour and performance
        are unchanged).
        :func:`gzl.series.compute_series_coefficients` allocates
        one per corpus pass automatically.
    engine : {"hybrid", "tensor"}, optional
        Engine for general-ν on-spine / vacuum blocks.  ``"hybrid"``
        (default) collapses series/parallel structure by FFT and
        contracts only the irreducible core, reading the BZ grid with a
        single ``fftn`` — equal to round-off to ``"tensor"``
        (:func:`gzl.graph_zeta_general`) but typically much
        faster.  It transparently falls back to the tensor for ν = ∞,
        for ≥ 2 free terminals, and on
        :class:`gzl.HybridCoreTooLargeError`.
    dense_engine : {"direct_sum", "torus"}, optional
        Route for dense (treewidth ≥ 3) blocks.  ``"direct_sum"`` is the
        historical real-space box ladder (``_DENSE_DSUM_L_LIST``);
        ``"torus"`` sends the block down the ordinary tensor / hybrid
        path instead, which at k = 0 also picks up the σ_eff Richardson
        ladder and, with ``sp_n_points``, hybrid's split resolution.
        Default ``None`` resolves per ``d`` from ``_DENSE_ENGINE_BY_D``.

        A ``MemoryError`` from the torus path keeps the box regardless
        of this flag, since the box is the one engine whose cost does not
        grow with ``n_points``.  ν ≤ d never reaches either engine: it is
        refused before routing (see Raises).
    sp_n_points : int, optional
        Fine SP-collapse grid for dense blocks on the ``"torus"`` route.
        Must be ≥ ``n_points``; values below it are raised to it, and a
        value equal to it means no split.  Ignored off that route, at
        ν = ∞, and for blocks with ≥ 2 kept terminals (hybrid supports
        at most one).  Default ``None`` resolves per ``d`` from
        ``_SPLIT_SP_N_POINTS``; a dimension absent from that table gets
        no split.  It is NOT applied to treewidth ≤ 2 blocks, which are
        fully SP-reducible and would effectively be evaluated on the
        fine grid outright.
    core_grading : bool, optional
        Size each dense block's irreducible CORE *down* from
        ``n_points``, per block, by
        ``n_core = κ · n_points^(σ_blk/σ_core)`` — the downward half of
        the ``sp_n_points`` axis, and the half that carries the cost
        (a core contracts ``n^(τ·d)`` entries against the fine grid's
        ``n^d``).  Both rates are already computed on the routing path.

        The point is that the core converges a full ν faster than its
        block, so at a shared grid it is hugely over-resolved: spending
        ``n^(τ·d)`` on the half that converged first is what makes
        raising ``n_points`` expensive.  Grading makes dense-core cost
        scale as ``n^(σ_blk/σ_core · τ·d)`` instead — n^2.18 rather
        than n^4 at d = 2, ν = 2.5 — so the saving GROWS with
        ``n_points``.  Measured per block at d = 2: 0.203 s → 0.031 s
        at ``n_points = 32``, and the graded time is flat in
        ``n_points``.

        It self-disables where it should: a block with nothing
        SP-reducible has σ_core = σ_blk by construction, so the
        exponent is 1 and ``n_core`` comes back as ``n_points``
        untouched — 13.5% of census occurrences, and the control that
        makes the rest legible.  Suppressed wherever the split is (off
        the torus route, at ν = ∞) and for any ``d`` absent from
        ``_CORE_KAPPA``, which means "not calibrated here" rather than
        "grade with a default".

        Turning this on also suppresses the k = 0 Richardson ladder on
        the graded blocks, deliberately — see the ``graded_core`` site.
        Default ``True``.
    accuracy : str | None, optional
        Default ``None`` resolves by REQUEST SHAPE: a full-BZ grid
        request defaults to ``"floor"`` (a grid is a pass context —
        every block shares its n_points, so the request's own floor is
        that grid), while vacuum scalars and explicit-k requests
        default to ``"strict"``.
        ``"strict"`` keeps the accuracy gate and its measured
        detours — the box-dominance standard every frozen reference key
        pins, right for standalone calls whose answer IS the block.
        ``"floor"`` is the CORPUS-PASS standard: the accuracy gate
        stands down and every dense block ships the pass-grid torus
        truncation (through the affordable pinned / slab associations,
        so byte limits still hold), which keeps the pass's numerical
        work approximately LINEAR in the number of graphs — no block
        may buy precision the pass cannot see.  Licensed by
        measurement: an n = 8 pass's floor is its ordinary blocks' own
        raw truncation (p = 7.5 class med 1.4e-05–3.3e-05; the p = 4
        class that dominates order ≥ 10 med 2.3e-03–5.6e-03), and every
        declined block pinned at the same n lands in the same error
        class or below it (K5 vacuum raw-8 3.1e-07; a p = 4 sector at
        order 11, pinned at n = 8: 2.7e-03 — level with its neighbours
        at 1.6 s instead of an hours-class box grid).
        ``compute_series_coefficients`` passes ``"floor"``.

    Returns
    -------
    float | np.ndarray | tuple
        ``float`` for vacuum / single-k, ``ndarray`` for a batch or the
        full grid.  With ``return_diagnostics=True``, returns
        ``(value, info_dict)``.

    Raises
    ------
    ValueError, TypeError
        A malformed argument: an edge list that is not of shape
        ``(E, 2)`` or has labels that are not whole numbers, a fractional
        ``source``, ``terminal`` or ``n_points``, a negative
        ``n_points``, a momentum that is not finite, or an ``A`` that is
        not a finite, non-singular square matrix or a lattice name.
        These errors are not ``GraphZetaError`` subclasses.
    UnsupportedLatticeSumError
        A block other than a bridge has a bundle exponent ``ν ≤ d``
        (after the Hadamard merge), or a bridge is evaluated at its pole.
        A bridge at ``ν ≤ d`` otherwise returns the meromorphic
        continuation of its Epstein zeta function.
    SelfLoopError
        The edge list contains a self-loop.
    DisconnectedGraphError
        The graph is disconnected.
    VertexOutOfRangeError
        ``source`` or ``terminal`` is not a vertex, or a label of an edge
        list is negative.
    NPointsRequiredError
        A block needs a grid, or a 1qp graph is asked for on the full
        Brillouin-zone grid (``momentum=None``), and ``n_points`` is 0;
        or the ``n_points ** d`` sites of the torus cannot hold a block
        that needs a grid, where its value would be 0.
    UnsupportedRequestError
        No evaluator exists for the request: two or more free terminals,
        or a 1qp graph whose path from ``source`` to ``terminal`` runs
        through a block that mixes ``ν = inf`` with finite exponents.
    TopologyEvaluatorUnavailableError
        ``networkx`` cannot be imported.  It is a dependency of gzl, so
        this means a broken installation.
    """
    if engine not in _BLOCK_ENGINES:
        raise ValueError(
            f"evaluate_graph: engine must be one of {_BLOCK_ENGINES}; "
            f"got {engine!r}."
        )
    # Validate eagerly, at the public boundary, rather than letting an
    # unknown value reach the resolver deep inside a corpus pass — a
    # typo'd dense_engine should fail on the first call, not after an
    # hour of blocks have already been cached under a wrong key.
    if dense_engine is not None and dense_engine not in _DENSE_ENGINES:
        raise ValueError(
            f"evaluate_graph: dense_engine must be None or one of "
            f"{_DENSE_ENGINES}; got {dense_engine!r}."
        )
    if sp_n_points is not None and int(sp_n_points) <= 0:
        raise ValueError(
            f"evaluate_graph: sp_n_points must be a positive int or None; "
            f"got {sp_n_points!r}."
        )
    if accuracy not in (None, "strict", "floor"):
        raise ValueError(
            f"evaluate_graph: accuracy must be None, 'strict' or "
            f"'floor'; got {accuracy!r}."
        )
    if _is_multigraph(graph):
        if A is None and nu is not None:
            A, nu = nu, None
        if A is None:
            raise TypeError(
                "evaluate_graph(multigraph, A, ...): A is required."
            )
        if nu is not None:
            raise TypeError(
                "evaluate_graph: nu is taken from the MultiGraph's edge "
                "attributes; do not pass it explicitly."
            )
        edges_arr, nu_arr = _multigraph_to_arrays(graph)
    else:
        if nu is None or A is None:
            raise TypeError(
                "evaluate_graph(edges_flat, nu, A, ...): nu and A are "
                "required for array input."
            )
        edges_arr = edge_array(graph, "evaluate_graph")
        nu_arr = nu

    # After the MultiGraph swap above: evaluate_graph(mg, "square") puts
    # the name in nu, and the swap moves it to A.
    A_arr = lattice_matrix(A, "evaluate_graph")
    d = int(A_arr.shape[0])
    n_points = _grid_size(n_points, "evaluate_graph")
    if edges_arr.size and int(edges_arr.min()) < 0:
        raise VertexOutOfRangeError(
            f"evaluate_graph: vertex labels must be non-negative "
            f"integers; found {int(edges_arr.min())}."
        )
    # ``source=None`` resolves to the smallest vertex label present —
    # for a contiguous graph that is 0, the old default; for a sparse
    # one it is the natural pin the old default made unreachable.  The
    # vacuum value is pin-independent by translation invariance.
    s_int = (int(edges_arr.min()) if source is None and edges_arr.size
             else 0 if source is None
             else vertex_label(source, "source", "evaluate_graph"))
    t_int_norm = _normalise_terminal(terminal, s_int)
    is_vacuum = t_int_norm is None or t_int_norm == s_int

    mode, k = _normalise_momentum(momentum, d)

    # The default accuracy is keyed on the REQUEST SHAPE.  A full-BZ
    # grid request is a pass context by construction — every block is
    # evaluated on the same n_points torus, so the request's own
    # accuracy floor is that grid, whoever makes the call — and
    # defaults to "floor".  A vacuum scalar or an explicit-k request
    # is a reference ask (one number, best licensed value — the
    # semantics every frozen reference key pins) and defaults to
    # "strict".  Both are overridable; compute_series_coefficients
    # passes "floor" explicitly for ALL its calls, because a corpus
    # pass is a pass whatever momentum it asks at.
    if accuracy is None:
        accuracy = "floor" if (not is_vacuum and mode == "none") else "strict"

    def _scalar_result(value, info=None):
        if return_diagnostics:
            info = info or {}
            info["version"] = 1
            return float(value), info
        return float(value)

    def _array_result(arr, info=None):
        arr = np.asarray(arr, dtype=float)
        if return_diagnostics:
            info = info or {}
            info["version"] = 1
            return arr, info
        return arr

    # ----------------------------------------------------------------
    # Vacuum path: k-independent.  The result is the k = 0 scalar;
    # broadcast across `momentum`'s requested shape.
    # ----------------------------------------------------------------
    if is_vacuum:
        if return_diagnostics:
            scalar, info = _evaluate_via_topology(
                edges_arr, nu_arr, A_arr, s_int, s_int,
                n_points=n_points, return_info=True, richardson=richardson,
                fast_cycles=fast_cycles, block_cache=block_cache, engine=engine,
                dense_engine=dense_engine, sp_n_points=sp_n_points,
                core_grading=core_grading,
                accuracy=accuracy,
            )
        else:
            scalar = _evaluate_via_topology(
                edges_arr, nu_arr, A_arr, s_int, s_int,
                n_points=n_points, return_info=False, richardson=richardson,
                fast_cycles=fast_cycles, block_cache=block_cache, engine=engine,
                dense_engine=dense_engine, sp_n_points=sp_n_points,
                core_grading=core_grading,
                accuracy=accuracy,
            )
            info = None
        scalar = float(scalar)
        if mode == "none" or mode == "single":
            return _scalar_result(scalar, info)
        if mode == "batch":
            return _array_result(np.full(k.shape[0], scalar), info)
        raise AssertionError(f"unreachable mode: {mode}")

    # ----------------------------------------------------------------
    # 1qp path: terminal != source.  Dispatch on momentum mode.
    # ----------------------------------------------------------------
    t_int = t_int_norm
    if mode == "none":
        # Default for 1qp: full BZ grid.  By the cost analysis the
        # all-k path is essentially as cheap as a single-k call when
        # the terminal is a path-end vertex (the typical 1qp case),
        # so returning the whole grid by default gives users the most
        # useful object — a complete dispersion ready to plot or
        # accumulate over a corpus.
        if int(n_points) <= 0:
            raise NPointsRequiredError(
                "evaluate_graph: full-BZ grid mode requires "
                "n_points > 0; pass momentum=np.zeros(d) for the "
                "single-point at k = 0 instead."
            )
        if return_diagnostics:
            value, info = _evaluate_via_topology(
                edges_arr, nu_arr, A_arr, s_int, t_int,
                n_points=n_points, return_info=True, richardson=richardson,
                momentum="grid", fast_cycles=fast_cycles, block_cache=block_cache, engine=engine,
                dense_engine=dense_engine, sp_n_points=sp_n_points,
                core_grading=core_grading,
                accuracy=accuracy,
            )
            return _array_result(value, info)
        value = _evaluate_via_topology(
            edges_arr, nu_arr, A_arr, s_int, t_int,
            n_points=n_points, return_info=False, richardson=richardson,
            momentum="grid", fast_cycles=fast_cycles, block_cache=block_cache, engine=engine,
            dense_engine=dense_engine, sp_n_points=sp_n_points,
            core_grading=core_grading,
                accuracy=accuracy,
        )
        return _array_result(value)

    if mode == "single":
        if return_diagnostics:
            value, info = _evaluate_via_topology(
                edges_arr, nu_arr, A_arr, s_int, t_int,
                n_points=n_points, return_info=True, richardson=richardson,
                momentum=k, fast_cycles=fast_cycles, block_cache=block_cache, engine=engine,
                dense_engine=dense_engine, sp_n_points=sp_n_points,
                core_grading=core_grading,
                accuracy=accuracy,
            )
            return _scalar_result(value, info)
        value = _evaluate_via_topology(
            edges_arr, nu_arr, A_arr, s_int, t_int,
            n_points=n_points, return_info=False, richardson=richardson,
            momentum=k, fast_cycles=fast_cycles, block_cache=block_cache, engine=engine,
            dense_engine=dense_engine, sp_n_points=sp_n_points,
            core_grading=core_grading,
                accuracy=accuracy,
        )
        return _scalar_result(value)

    if mode == "batch":
        out = np.empty(k.shape[0], dtype=float)
        last_info = None
        for i, k_i in enumerate(k):
            if return_diagnostics and i == 0:
                v_i, last_info = _evaluate_via_topology(
                    edges_arr, nu_arr, A_arr, s_int, t_int,
                    n_points=n_points, return_info=True,
                    richardson=richardson, momentum=k_i,
                    fast_cycles=fast_cycles, block_cache=block_cache, engine=engine,
                    dense_engine=dense_engine, sp_n_points=sp_n_points,
                    core_grading=core_grading,
                accuracy=accuracy,
                )
            else:
                v_i = _evaluate_via_topology(
                    edges_arr, nu_arr, A_arr, s_int, t_int,
                    n_points=n_points, return_info=False,
                    richardson=richardson, momentum=k_i,
                    fast_cycles=fast_cycles, block_cache=block_cache, engine=engine,
                    dense_engine=dense_engine, sp_n_points=sp_n_points,
                    core_grading=core_grading,
                accuracy=accuracy,
                )
            out[i] = float(v_i)
        return _array_result(out, last_info)

    raise AssertionError(f"unreachable mode: {mode}")
