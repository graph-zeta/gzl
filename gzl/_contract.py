# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Shared factor-supply layer for the two lattice truncations.

There is one discrete model in this library — a graph lattice sum
contracted by bucket elimination — evaluated under two truncations:

* **torus / cyclic** (`tensor_network`): differences taken mod ``n``,
  circulant kernel, plain FFT peel;
* **box / zero-padded linear** (`direct_sum`): true differences,
  Toeplitz kernel, padded FFT peel.

Both engines carry each original edge kernel as a small *generator*
array and read dense tables out of it only where a consumer needs one.
This module holds the pure index arithmetic those reads share — the
gathers are functions of the generator and the index geometry alone,
with no reference to any module-level tunable — plus the
:class:`Factor` record and the :class:`Truncation` interface that the
shared bucket loop (:func:`eliminate` and its step
:func:`eliminate_one`, below) consumes.

**Construction rule (deliberate, non-negotiable):** the two
``Truncation`` subclasses are *not* defined here.  ``TorusTruncation``
lives in ``tensor_network.py`` and ``BoxTruncation`` in
``direct_sum.py``, and their method bodies resolve their engine's
tunables and peel functions through **their own module's namespace at
call time**.  The test suite patches ``tensor_network._USE_CONV``,
``direct_sum._fft_conv_step`` and a dozen similar names on the engine
modules; a Truncation defined here, or one that captured those names at
construction, would silently detach every one of those patches — and
several of the suite's anti-vacuity tests exist precisely to fail
loudly when that happens.

Nothing in this module is public API; it is re-exported from nowhere.
"""

from __future__ import annotations

from dataclasses import dataclass

from collections import OrderedDict

import numpy as np


# ---------------------------------------------------------------------------
# The factor record
# ---------------------------------------------------------------------------

@dataclass
class Factor:
    r"""One multiplicative factor of the lattice sum.

    A factor is either *lazy* — an original edge kernel carried as its
    generator plus the index arithmetic that reads a table out of it —
    or *dense* (``arr`` set), which covers weights, pinned tensors and
    every intermediate produced by elimination.

    ``peelable`` is an explicit field rather than a property derived
    from ``gen is not None``: the torus engine encodes "never FFT this
    bundle" (nu = inf homomorphism counts, where an FFT round trip
    returns 2.0 as 1.9999999999999998) by eagerly densifying, and that
    trick only works while laziness and peelability happen to
    coincide.  A box kernel that must stay off the FFT path but should
    still be stored lazily needs the two concepts separated.
    """

    scope: tuple            # sorted vertex labels naming each axis
    arr: "np.ndarray | None" = None      # dense form; None => still lazy
    gen: "np.ndarray | None" = None      # kernel generator
    off: "tuple | None" = None           # per-axis gather offsets
    nu: "float | None" = None            # peel tag (the box's conv_nu)
    peelable: bool = True


# ---------------------------------------------------------------------------
# Torus (cyclic) gathers — index map (i - j) mod n
# ---------------------------------------------------------------------------

#: Memo for the (i - j) mod n index map used by :func:`circulant_table`.
#: The map depends only on ``(n, d)``, never on the kernel, so ONE entry
#: serves every edge exponent in a pass -- measured 2,835 table builds
#: over 114 distinct kernels at d = 1 n = 512, i.e. the map was rebuilt
#: 2,835 times to be used once each.  Caching the finished tables instead
#: would cost 239 MB there; this costs one ``(n, n)`` array.
_CIRC_IDX_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_CIRC_IDX_CACHE_MAX = 8


def _circulant_index(n: int, d: int):
    """Cached ``(i - j) mod n`` gather index for :func:`circulant_table`.

    Returned arrays are marked read-only: they are shared across every
    caller in the process, and a mutation would corrupt every kernel
    built afterwards rather than fail where it happened.
    """
    key = (int(n), int(d))
    hit = _CIRC_IDX_CACHE.get(key)
    if hit is not None:
        _CIRC_IDX_CACHE.move_to_end(key)
        return hit

    if d == 1:
        i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        idx = (i_idx - j_idx) % n
        idx.flags.writeable = False
    else:
        base = np.arange(n)
        ax = (base[:, None] - base[None, :]) % n
        ax.flags.writeable = False
        parts = []
        for c in range(d):
            shape = [1] * (2 * d)
            shape[c] = shape[d + c] = n
            parts.append(ax.reshape(shape))
        idx = tuple(parts)

    _CIRC_IDX_CACHE[key] = idx
    if len(_CIRC_IDX_CACHE) > _CIRC_IDX_CACHE_MAX:
        _CIRC_IDX_CACHE.popitem(last=False)
    return idx


def circulant_table(K: np.ndarray, n_points: int) -> np.ndarray:
    r"""Two-axis edge tensor ``T[i, j] = K[(i - j) mod n]`` over flat
    ``n^d`` torus indices.

    Advanced indexing with broadcast index views, so the only
    ``n^(2d)`` allocation is the result itself — stacking per-component
    differences instead costs a further factor ``d`` in peak memory
    (4.6 GB at d = 3, n = 24, against a 1.5 GB table).

    Returns an array of ``K``'s own dtype (float64 for every kernel
    this library builds), shape ``(n^d, n^d)``.
    """
    n = int(n_points)
    d = int(K.ndim)
    n_total = n ** d

    if d == 1:
        return K[_circulant_index(n, 1)]

    return K[_circulant_index(n, d)].reshape(n_total, n_total)


def reverse_generator(K: np.ndarray) -> np.ndarray:
    r"""Per-axis cyclic index reversal: ``K_rev[m] = K[(-m) mod n]``.

    The transpose of a circulant difference table is circulant with the
    reversed generating vector.
    """
    out = K
    for a in range(K.ndim):
        out = np.roll(np.flip(out, axis=a), 1, axis=a)
    return out


def circulant_pin(gen: np.ndarray, pin_row: bool) -> np.ndarray:
    r"""One row/column of a circulant table, without building the table.

    With ``T[i, j] = gen[(i - j) mod n]`` per axis, pinning the row at
    index 0 leaves ``gen[(-j) mod n]`` and pinning the column leaves
    ``gen[i]`` — ``O(n^d)`` instead of ``O(n^(2d))``.
    """
    g = reverse_generator(gen) if pin_row else gen
    return g.reshape(-1)


# ---------------------------------------------------------------------------
# Box (zero-padded linear) gathers — index map (i - j) + offset
# ---------------------------------------------------------------------------

def box_axis_extents(L: int, d: int, half: bool) -> tuple:
    """Per-axis lengths of a box vertex's position grid.

    The grid is a C-order product, which is what lets a flat position
    index decompose per axis and a factor be read out of the
    difference-range generator without forming the
    ``(size_a, size_b, d)`` difference array.
    """
    return ((L + 1,) + (2 * L + 1,) * (d - 1)) if half else (2 * L + 1,) * d


def box_axis_origins(L: int, d: int, half: bool) -> tuple:
    """Per-axis lattice coordinate of index 0: ``pos[i]_c = i_c + org_c``."""
    return ((0,) + (-L,) * (d - 1)) if half else (-L,) * d


def box_gen_offsets(L: int, d: int, org_a: tuple, org_b: tuple) -> tuple:
    """Per-axis constant in the generator index of a two-vertex factor.

    The generator holds ``K(delta)`` at ``delta + 2L`` per axis, so with
    ``pos[i]_c = i_c + org_c``,

        gen_index_c = (j_c + org_b_c) - (i_c + org_a_c) + 2L
                    = j_c - i_c + (org_b_c - org_a_c + 2L)

    and only that trailing constant depends on which endpoint is the
    half axis.
    """
    return tuple(int(org_b[c] - org_a[c] + 2 * L) for c in range(d))


def box_table(gen: np.ndarray, ext_a: tuple, ext_b: tuple, off: tuple,
              d: int) -> np.ndarray:
    """Two-axis box factor read out of the difference-range generator.

    Broadcast index views again: the only ``size_a * size_b``
    allocation is the result itself.
    """
    idx = []
    for c in range(d):
        ax = (np.arange(ext_b[c])[None, :]
              - np.arange(ext_a[c])[:, None] + off[c])
        shape = [1] * (2 * d)
        shape[c] = ext_a[c]
        shape[d + c] = ext_b[c]
        idx.append(ax.reshape(shape))
    size_a = int(np.prod(ext_a))
    size_b = int(np.prod(ext_b))
    return gen[tuple(idx)].reshape(size_a, size_b)


def box_row(gen: np.ndarray, ext: tuple, off: tuple, d: int) -> np.ndarray:
    """One-axis box factor ``K(x_v - 0)`` for a kernel incident on the
    root, which sits at the origin — an ``O(n)`` gather instead of an
    ``O(n^2)`` table followed by a row pick.
    """
    idx = []
    for c in range(d):
        ax = np.arange(ext[c]) + off[c]
        shape = [1] * d
        shape[c] = ext[c]
        idx.append(ax.reshape(shape))
    return gen[tuple(idx)].reshape(-1)


def box_compose(g1: np.ndarray, g2: np.ndarray, L: int,
                d: int) -> np.ndarray:
    r"""Series composition of two box generators — a zero-padded LINEAR
    convolution on the difference range ``[-2L, 2L]^d``, cropped back.

    **This is a translation-invariant surrogate, not an identity.**  The
    exact box composition is

        K_new(x_a, x_b) = sum_{x_w in [-L,L]^d} K1(x_a - x_w) K2(x_w - x_b)

    which depends on ``x_a`` and ``x_b`` SEPARATELY, not on their
    difference: ``[-L, L]^d`` is not closed under translation, and the
    product of two Toeplitz matrices is not Toeplitz.  So no difference
    kernel represents it exactly, and the collapse cannot be
    value-preserving at finite ``L`` the way the torus's cyclic
    convolution is.

    What this returns instead is the full linear convolution — the
    ``x_w`` sum extended to the whole difference range with zero padding
    — restricted to ``[-2L, 2L]^d``.

    Its error against the exact box contraction is ONE-SIGNED (the
    surrogate over-counts) and decays with ``L``.  The observed rate is
    NOT a general law — it depends on the shape and on which endpoints
    are pinned.  The figure gated by
    ``tests/test_sp_shared.py::TestBoxCollapse`` is a 4-cycle at d=1,
    nu=2.5 reduced through :func:`gzl._sp.sp_reduce` (two series
    collapses), where the relative excess over
    ``direct_sum_zero_momentum`` runs +6.41e-03 at L=4 monotonically down
    to +2.63e-07 at L=48, fitted rate -4.07 against ``2 nu - d = 4.0``.
    A single collapse on the same cycle gives +3.20e-03 to +9.45e-08 at
    the same rate; quote whichever experiment you mean, since they differ
    by a factor 2 in amplitude.

    Both families converge to the same infinite-lattice value, so a
    caller may use this to trade a fine collapse against a coarse
    contraction — but a box SP collapse must be gated on convergence in
    ``L``, never on agreement with an uncollapsed box value at fixed
    ``L``.
    """
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    m = 4 * L + 1                      # difference-range length per axis
    if g1.shape != (m,) * d or g2.shape != (m,) * d:
        raise ValueError(
            f"box_compose: generators must be ({m},)*{d} on the "
            f"[-2L, 2L] difference range; got {g1.shape} and {g2.shape}"
        )
    full = (2 * m - 1,) * d            # linear-convolution support
    ax = tuple(range(d))               # explicit: numpy 2.0 deprecates None
    out = np.fft.irfftn(np.fft.rfftn(g1, full, axes=ax)
                        * np.fft.rfftn(g2, full, axes=ax), full, axes=ax)
    # Crop the centre: the convolution's origin sits at index 2*(2L) per
    # axis, and we keep 2L either side of it.
    lo = 2 * (2 * L) - 2 * L
    return out[tuple(slice(lo, lo + m) for _ in range(d))]


def box_restrict(gen: np.ndarray, L_fine: int, L_coarse: int,
                 d: int) -> np.ndarray:
    r"""Centred crop of a box generator from ``L_fine`` to ``L_coarse``.

    Both ranges are symmetric about the origin, so the restriction is a
    plain sub-array — no wrapping, no modular map, no balanced-label
    convention.  At ``L_coarse == L_fine`` it is the whole array.
    """
    m_f, m_c = 4 * L_fine + 1, 4 * L_coarse + 1
    gen = np.asarray(gen)
    if gen.shape != (m_f,) * d:
        raise ValueError(
            f"box_restrict: expected ({m_f},)*{d} on the fine difference "
            f"range; got {gen.shape}"
        )
    lo = 2 * (L_fine - L_coarse)
    return gen[tuple(slice(lo, lo + m_c) for _ in range(d))]


def box_reverse_generator(K: np.ndarray) -> np.ndarray:
    r"""Index reversal on the symmetric difference range ``[-2L, 2L]^d``:
    ``K_rev[delta] = K[-delta]`` is a plain per-axis flip — no roll,
    because the range is symmetric about its centre, unlike the cyclic
    ``0..n-1`` labels of :func:`reverse_generator`.
    """
    out = K
    for a in range(K.ndim):
        out = np.flip(out, axis=a)
    return out


# ---------------------------------------------------------------------------
# Dense-strategy resolution
# ---------------------------------------------------------------------------

def resolve_dense_strategy(mode: str, bag_volume: int,
                           threshold: int) -> str:
    r"""Resolve a dense-strategy mode to a concrete branch name.

    ``"broadcast"`` (fused full-bag product, peak ``2x`` the bag
    volume) and ``"sliced"`` (accumulate slice by slice along the
    eliminated axis, peak the *out*-scope volume) pass through
    unchanged.  ``"auto"`` picks ``"sliced"`` exactly when
    ``bag_volume >= threshold``.

    The threshold is a *measured* machine constant owned by the engine
    module (this module keeps zero tunables): slicing trades a
    per-slice Python/allocation overhead against working sets that
    exceed cache.  Measured on the shipped bucket shapes (bag 3-4),
    sliced is 3-7x slower below ~1e5 bag elements, breaks even around
    ~5e5, and wins 1.5-3x (plus an axis_size(w) factor of peak memory)
    above ~5e6.
    """
    if mode == "auto":
        return "sliced" if bag_volume >= threshold else "broadcast"
    if mode not in ("broadcast", "sliced"):
        raise ValueError(f"unknown dense strategy mode: {mode!r}")
    return mode


# ---------------------------------------------------------------------------
# The truncation interface
# ---------------------------------------------------------------------------

class Truncation:
    r"""What the shared bucket loop needs to know about one truncation.

    Abstract: every method raises here.  The two implementations —
    ``tensor_network.TorusTruncation`` and ``direct_sum.BoxTruncation``
    — live in their engine's module (see the module docstring for why),
    and each method body is that engine's *current* behaviour moved
    verbatim, not a re-derivation.

    Two members are **methods, not capability booleans**, on purpose:

    ``empty_bucket_scalar``
        What a vertex with no incident factor contributes.  The box
        returns the full position count (``z2_weights.sum()`` on the
        marker axis — the Z2 fold restores the other half); a
        ``constant_row_sum``-style flag nothing reads could silently
        skip it, and the failure mode is a value scaled by
        ``(L+1)/(2L+1)`` with no exception, no NaN and no shape signal.

    ``peel_constant``
        The bucket-of-one peel result.  On the torus the row sums of a
        circulant table are constant, so ``full(N, gen.sum())`` is
        exact; on the box the row sums genuinely vary (measured spread
        20-56%), and a shared code path that assumed constancy would be
        wrong on every box call that reaches it.
    """

    d: int

    def axis_size(self, v) -> int:
        """Length of vertex ``v``'s position axis.  EXPLICIT — never
        derive it from ``gen.size``: that is a circulant-only identity
        (torus generators have ``n^d`` entries, one per position), and
        it is wrong by ``(2 - 1/(2L+1))^d`` — 1.89x/3.57x/6.74x at
        d = 1/2/3 — on the box's ``(2(2L+1)-1)^d`` difference range."""
        raise NotImplementedError

    def coords(self, v) -> np.ndarray:
        """``(axis_size(v), d)`` integer lattice coordinates."""
        raise NotImplementedError

    def generator(self, nu) -> np.ndarray:
        """The kernel generator for exponent ``nu`` (cached per call)."""
        raise NotImplementedError

    def table(self, gen, a, b) -> np.ndarray:
        """Dense two-axis factor for vertices ``(a, b)``."""
        raise NotImplementedError

    def pin_slot(self, gen, v, pin_row: bool) -> np.ndarray:
        """One-axis factor of a kernel with one endpoint pinned at 0."""
        raise NotImplementedError

    def reverse(self, gen) -> np.ndarray:
        """The generator of the transposed table."""
        raise NotImplementedError

    def restrict_to(self, gen, coarse) -> np.ndarray:
        """Re-express ``gen`` — a generator on *this* truncation — on the
        coarser ``coarse`` truncation of the same lattice and dimension.

        A pure index gather: no arithmetic, so a restriction to an
        equal-sized truncation is the identity bit for bit.  That is
        what lets a caller collapse the series/parallel part of a graph
        on a fine grid and contract the irreducible core on a coarse
        one — the split-resolution scheme, whose payoff is that the
        truncation rate is set by the *core's* minimum cluster cut
        (``>= 3 nu - d`` for a 3-connected core) rather than the
        block's (``2 nu - d``), a gap of exactly ``nu``.

        The two truncations must describe the same lattice: the labels
        are shared integers and the physical positions are ``A z``, so
        restricting across different ``A`` would silently reinterpret
        every entry.
        """
        raise NotImplementedError

    def embed_into(self, arr, fine) -> np.ndarray:
        """The upward mirror of :meth:`restrict_to`: re-express ``arr``,
        an array on *this* truncation, on the finer ``fine`` truncation
        of the same lattice, padding the labels ``fine`` has and this
        one does not with zero.

        Where ``restrict_to`` lets the SP part run finer than the core,
        this lets the CORE run *coarser* than the grid its value has to
        be delivered on.  A core contracted on a coarse torus produces a
        terminal-position amplitude whose Brillouin-zone transform is a
        trigonometric polynomial; embedding and transforming evaluates
        that same polynomial on the fine grid, so the step is exact for
        the approximant and adds no error of its own.  Both directions
        are pure index moves, and both are the identity bit for bit when
        the two truncations have equal size.

        Only the torus implements this.  The box's labels are a centred
        window rather than a residue system, and a zero pad outside it
        asserts a decay the linear composition has not established, so
        ``BoxTruncation`` deliberately leaves it unimplemented rather
        than offering a plausible wrong answer.
        """
        raise NotImplementedError

    # ---- the SP graph rewrite (gzl._sp) -----------------------
    #
    # Three operations, one per reduction rule.  They are methods rather
    # than a capability flag because the two truncations differ in kind,
    # not in degree: the torus composes CYCLICALLY (exact for its own
    # finite sum) while the box composes LINEARLY and only
    # approximately, so a shared implementation guarded by a boolean
    # would be silently wrong on one of them.

    def compose(self, g1, g2):
        """Series rule: the kernel of a collapsed degree-2 vertex.

        ``g1`` runs ``a -> w`` and ``g2`` runs ``w -> b``, so the result
        generates ``K(x_a - x_b) = sum_w K1(x_a - x_w) K2(x_w - x_b)``.
        Callers must orient both legs before calling — getting that
        wrong is invisible on an even kernel and silently wrong
        otherwise.
        """
        raise NotImplementedError

    def hadamard(self, g1, g2):
        """Parallel rule: two edges on the same pair multiply pointwise,
        ``K_1 K_2``.  Elementwise on the generator for both truncations,
        hence a working default."""
        return np.asarray(g1) * np.asarray(g2)

    def trace(self, gen) -> float:
        """The kernel at ZERO displacement — what a loop contributes.

        NOT ``gen.sum()``.  A loop closes its two endpoints onto the
        same vertex, so the configuration it contributes is
        ``K(x - x) = K(0)``, a single entry; ``gen.sum()`` is the
        disconnected answer ``(sum K_1)(sum K_2)`` and is larger by
        orders of magnitude.  For the regularised kernel
        (``K(0) = 0``) this is exactly the zero that drops every
        coincident-endpoint configuration.
        """
        raise NotImplementedError

    def weight(self, v):
        """Per-position weight on ``v``'s axis (None if uniform)."""
        raise NotImplementedError

    def empty_bucket_scalar(self, v) -> float:
        """Contribution of a vertex with no incident factor."""
        raise NotImplementedError

    def peel_constant(self, gen, u) -> np.ndarray:
        """Row sums of ``gen``'s table over ``u``'s axis — the peel of
        a bucket whose only factor is the kernel itself."""
        raise NotImplementedError

    # ---- consumed by the shared bucket skeleton (eliminate_one) -------

    def scope_of(self, f) -> tuple:
        """The vertex labels a native factor's axes carry.  The
        skeleton treats factors as opaque; this is the one accessor it
        needs, and it is what lets each engine keep its shipped factor
        representation through the extraction."""
        raise NotImplementedError

    def scalar_factor(self, value):
        """Wrap a scalar as a native empty-scope factor."""
        raise NotImplementedError

    def pre_step(self, bucket, v, out_axes) -> None:
        """Validation hook, called after the empty-bucket dispatch and
        before anything arithmetic.  Default: nothing.  The torus
        raises its einsum-letters limit here, preserving the shipped
        behaviour that an oversized bucket errors even when it could
        have peeled."""

    def step_dtype(self, bucket):
        """The step's result dtype, from the bucket's own factors."""
        raise NotImplementedError

    def peel_gate(self, bucket, v, union, dtype) -> bool:
        """Whether this step may attempt the FFT peel at all."""
        raise NotImplementedError

    def find_partner(self, bucket, v):
        """Locate the peelable factor; an opaque token, or None.  The
        tie-break is engine-specific (torus: first match in list
        order; box: smallest partner label) and both are pinned by the
        frozen goldens, so neither may adopt the other's."""
        raise NotImplementedError

    def peel_step(self, bucket, token, v, out_axes, dtype):
        """Contract the bucket by FFT peel; returns one native factor."""
        raise NotImplementedError

    def dense_step(self, bucket, v, out_axes, dtype):
        """Contract the bucket densely; returns one native factor."""
        raise NotImplementedError

    def dense_strategy(self, bag_volume: int) -> str:
        """Which dense branch this truncation runs at the given bag
        volume: ``"broadcast"``, ``"sliced"`` or ``"auto"`` (resolved
        through :func:`resolve_dense_strategy` with the engine's own
        threshold).  A method rather than a flag so the pinned state
        is explicit and testable per truncation — the torus's dense
        step *is* its axis-sliced einsum loop and is not flipped."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# The shared bucket-elimination skeleton
# ---------------------------------------------------------------------------
#
# What is shared here is CONTROL FLOW only: partition the factor list by
# the eliminated vertex, dispatch the empty bucket, gate and attempt the
# FFT peel, fall through to the dense contraction.  Every arithmetic
# decision — the step dtype, the peel gate, the partner tie-break, the
# peel itself, the dense strategy, the empty-bucket scalar — is a
# Truncation method whose body is its engine's shipped code, moved
# verbatim.  Factors are OPAQUE to the skeleton: each engine keeps its
# native representation (the torus's (axes, arr, gen) triples, the box's
# scope/gen/off dicts) and the Truncation adapts, which is what keeps
# the extraction a pure move instead of a rewrite.
#
# Branch-order note, so nobody "restores" it: the torus engine used to
# check its scalar branch (no output axes) BEFORE attempting the peel.
# The skeleton attempts the peel first — which cannot change which
# branch runs, because a peelable factor has two axes {v, u} with
# u != v, so its bucket's output-axis set contains u and is never
# empty; find_partner on a no-output-axes bucket must therefore return
# None, and the scalar branch (now the head of the torus dense_step) is
# reached exactly as before.  The partner SEARCH runs where it
# previously did not, but it is pure inspection — no arithmetic, and
# the suite's peel sentinels count the peel executors, not the search.

def eliminate_one(factors: list, v, trunc) -> list:
    """Eliminate vertex ``v`` from ``factors``; returns the new list."""
    bucket = []
    rest = []
    for f in factors:
        (bucket if v in trunc.scope_of(f) else rest).append(f)

    if not bucket:
        return rest + [trunc.scalar_factor(trunc.empty_bucket_scalar(v))]

    union = tuple(sorted({a for f in bucket for a in trunc.scope_of(f)}))
    out_axes = tuple(a for a in union if a != v)

    trunc.pre_step(bucket, v, out_axes)

    dtype = trunc.step_dtype(bucket)
    if trunc.peel_gate(bucket, v, union, dtype):
        token = trunc.find_partner(bucket, v)
        if token is not None:
            return rest + [trunc.peel_step(bucket, token, v,
                                           out_axes, dtype)]
    return rest + [trunc.dense_step(bucket, v, out_axes, dtype)]


def eliminate(factors: list, order, trunc) -> list:
    """Eliminate the vertices in ``order``, one bucket at a time."""
    for v in order:
        factors = eliminate_one(factors, v, trunc)
    return factors
