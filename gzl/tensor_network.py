# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

r"""Treewidth-agnostic Fourier-grid graph zeta evaluation.

A bucket-elimination algorithm that takes an arbitrary multigraph
(edge list + per-edge exponents + lattice basis) and returns
:math:`\zeta_g(\boldsymbol{0}, \dots, \boldsymbol{0})` — the graph
zeta at all-zero external momenta — by sequentially summing out
non-source vertex positions on a torus of ``n_points^d`` sites per
vertex axis.

The representation is a pure Fourier grid (equivalent to the
``sigma_max = 0`` improved-Fourier method for treewidth-≤2 graphs),
with no analytic-Epstein algebra.  Cost is

.. math::

    \mathcal{O}\!\bigl(V \cdot n_\text{points}^{(\tau + 1)\,d}\bigr)

where :math:`\tau` is the treewidth of the underlying simple graph.
For tw = 2 this reproduces ``graph_from_edges_uniform`` at
``sigma_max = 0`` to floating-point precision; for tw > 2 it is a
new code path that doesn't require SP reduction.

Not implemented: when the result is being
projected to zero momentum *and* one operand of a contraction is a
single Epstein-zeta edge or a tw-≤2 ``GraphZeta`` with analytic
``bVec``/``nuVec``, route through analytic convolution to avoid
edge-side aliasing.
"""

from __future__ import annotations

import os
import string
from collections import OrderedDict

import numpy as np

from gzl import _contract
from gzl import _elimination
from gzl._elimination import best_order as _best_order
from gzl._labels import relabel_to_support as _relabel_to_support
from gzl._lattices import _resolve_lattice
from gzl._real import _as_real
from gzl.interaction import POWER_TORUS


__all__ = [
    "graph_zeta_general",
    "graph_zeta_general_at_zero",
]

# Private A/B switch for the FFT elimination fast path: read at call
# time inside _eliminate_vertex so tests can monkeypatch it; with
# False the dense slice-einsum reproduces the pre-FFT arithmetic
# exactly.
_USE_CONV = True

# Private A/B switch for the real-transform convolution in
# TorusTruncation.compose.  With False the complex fftn path runs, which
# is the historical arithmetic exactly.  Read at call time so tests can
# monkeypatch it.
_USE_RFFT = True

# Private A/B switch for the EXACT sparse peel of compact bundles
# (nu = inf: the nearest-neighbour indicator and every purely compact
# Interaction).  Such a bundle never takes the FFT peel -- an FFT round
# trip returns 2.0 as 1.9999999999999998 and the value is a
# homomorphism count -- and used to be contracted densely, at a full
# power of n^d more per step.  With True the bundle stays lazy and a
# step that isolates it runs ``_peel_sparse``: the same cyclic
# convolution, summed over the table's non-zero labels only, so an
# integer table stays exactly integral.  With False the bundle is
# densified eagerly and the planner is told it cannot peel, which is
# the historical arithmetic exactly.  Read at call time so tests can
# monkeypatch it.
_USE_SPARSE_PEEL = True

# Test hook: force the terminal-streaming schedule (normally reached
# only via a genuine MemoryError in the batched path).
_FORCE_STREAMING = False

# Per-intermediate-tensor budget of the streaming path's batched
# prefix (module level so tests can monkeypatch it).
_MEM_BUDGET_BYTES = 256 * 1024 * 1024

# Memory budget for ONE peel's working set (the chunk's phi product,
# its half-spectrum and the inverse transform).  When the modelled
# working set of a full-bag peel exceeds this, the peel runs in chunks
# along one spectator axis — same arithmetic per fibre, so the values
# are bit-identical (gated as such); only the peak allocation changes.
# Mirrors ``direct_sum._MEM_BUDGET_BYTES``.
#
# Deliberately a SEPARATE name from ``_MEM_BUDGET_BYTES`` above, which
# sizes the streaming TERMINAL split.  Each is monkeypatched by its own
# gate, and one name would make each test perturb the other subsystem.
#
# Read at call time so tests can monkeypatch it.
_PEEL_BUDGET_BYTES = 256 * 1024 * 1024

# Test hook: None = choose the chunk count from the budget (the shipped
# sizes almost always stay single-shot); an int forces that chunk
# LENGTH along the spectator axis, which is how the equality gate drives
# chunk sizes 1/3/7 without a budget dance.  Mirrors
# ``direct_sum._FORCE_CHUNK``.
_FORCE_CHUNK = None


# ---------------------------------------------------------------------------
# Edge kernel on the torus
# ---------------------------------------------------------------------------

def _balanced_z_axis(n: int) -> np.ndarray:
    r"""Balanced lattice-label axis used by :func:`graph_compress`.

    Returns ``[0, 1, …, n//2, -((n + 1) // 2) + 1, …, -1]``, matching
    the layout that ``graph_compress`` produces from
    ``z_axis = np.concatenate([pos, neg])``.  This is the discrete
    representation of "label space", with negative-equivalent labels
    on the upper half of the index range so that real-space sums are
    centred around the origin.
    """
    pos = np.arange(0, n // 2 + 1, dtype=int)
    neg = np.arange(-((n + 1) // 2) + 1, 0, dtype=int)
    return np.concatenate([pos, neg])


def _label_distances(A: np.ndarray, n_points: int) -> np.ndarray:
    r"""Physical distance from the origin to each torus label.

    Uses the balanced z-axis convention so the (d-dimensional)
    label vector at index ``[i_1, …, i_d]`` is
    ``(z_axis[i_1], …, z_axis[i_d])`` and the physical position is
    ``A @ z_label``.  Returns ``||A @ z_label||`` as an array of
    shape ``(n_points,) * d``.

    This is the convention used by :func:`gzl.graph_compress`
    at ``sigma_max = 0`` (via ``_vint_array``).  For diagonal ``A``
    (e.g. the integer 1D chain) this coincides with shortest-image
    distance; for non-orthogonal lattices (e.g. the 2D triangular
    lattice) it can differ from shortest-image at boundary labels — the
    lib's choice favours coordinate-aligned periodisation rather than
    nearest physical image.
    """
    d = int(A.shape[0])
    n = int(n_points)

    z = _balanced_z_axis(n)
    grids    = np.meshgrid(*[z for _ in range(d)], indexing="ij")
    z_labels = np.stack(grids, axis=-1).astype(float)    # (n,)*d, (d,)
    phys     = np.einsum("ij,...j->...i", A, z_labels)   # (n,)*d, (d,)
    return np.linalg.norm(phys, axis=-1)


#: Cache of built edge kernels, keyed on ``(nu, n_points, A bytes)``.
#: MEASURED: at d = 2, nu = 2.5 the whole shipped corpus carries just
#: **6 distinct edge exponents** (nu, 2nu, ... 6nu, from bundle
#: multiplicities) across 13800 edge instances — a 2300x reuse — yet
#: ``hybrid_zeta`` built them afresh for every block because its
#: ``_kernel_cache`` is created per call.  On the 1447-block tw>=3
#: census at ``sp_n_points = 1200`` that was **~85 ms of the 113 ms**
#: spent per block, against 246 ms to build all six ONCE.
#:
#: Bounded in BYTES, not entries: a kernel is ``n_points^d`` float64, so
#: the same count means 0.03 MB at d = 1 n = 4096 and 134 MB at d = 3
#: n = 256.  Least-recently-used eviction; a single kernel larger than
#: the budget is returned uncached rather than evicting everything.
_KERNEL_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_KERNEL_CACHE_BYTES = 0
#: Budget, overridable for tests and for memory-tight runs.  512 MB
#: holds every kernel a d <= 2 pass needs; a d = 3 run at large
#: ``sp_n_points`` will evict, which is the intended degradation.
KERNEL_CACHE_MAX_BYTES = int(
    os.environ.get("GZ_KERNEL_CACHE_BYTES", 512 * 1024 * 1024)
)


def _kernel_cache_clear() -> None:
    """Drop every cached kernel.  For tests and for callers that need
    the memory back between passes."""
    global _KERNEL_CACHE_BYTES
    _KERNEL_CACHE.clear()
    _KERNEL_CACHE_BYTES = 0


def _edge_kernel_torus(nu: float, A: np.ndarray, n_points: int,
                       interaction=None) -> np.ndarray:
    r"""Real-space edge kernel on the torus.

    **Cached across calls** (see :data:`_KERNEL_CACHE`).  The returned
    array is the cache's own and is treated as READ-ONLY by every
    consumer: the parallel merge, the series convolution and the
    orientation flip all return new arrays, and the executor builds its
    own tables.  It is marked non-writeable so a future consumer that
    forgets cannot corrupt every other block silently.

    Returns ``K`` of shape ``(n_points,) * d`` **real (float64)** with
    ``K[idx] = 1 / dist(idx) ** nu`` where ``dist`` is the
    balanced-z-axis label distance, and ``K[0] = 0`` (regularised
    self-energy).  Matches :func:`gzl.graph_compress` at
    ``sigma_max = 0`` exactly.

    When ``nu`` is ``np.inf`` the kernel is the **nearest-neighbour
    indicator** (the ν → ∞ limit): ``K = 1`` on the minimal nonzero
    lattice shell and 0 elsewhere.

    The kernel is real by construction; the bucket-elimination engine
    relies on numpy's standard dtype promotion to upcast to complex only
    when a finite-momentum phase tensor enters the contraction.

    ``interaction`` (default ``None``) selects the GENERAL-kernel build:
    an :class:`gzl.interaction.Interaction`, or a lazy product of
    them, sampled on the same balanced label grid in this engine's own
    power form (``1.0 / dist ** nu``, see
    :func:`_edge_kernel_torus_interaction`).  ``nu`` is then the
    kernel's TAIL exponent and is not read here — it stays in the
    signature so every positional call site keeps working and every
    consumer that reads tails keeps reading one vector.  The cache key
    is ``("K",) + interaction.key()`` in place of the float: a tuple,
    so it can never collide with a float key, and distinct for two
    kernels that share a tail but not a table — a stale entry served
    across two such kernels was measured at a 200 % error.  With
    ``interaction=None`` the body and the key are the legacy ones,
    expression for expression.
    """
    global _KERNEL_CACHE_BYTES
    A = np.asarray(A, dtype=float)
    if interaction is None:
        key = (float(nu), int(n_points), A.shape, A.tobytes())
    else:
        key = ((("K",) + tuple(interaction.key())),
               int(n_points), A.shape, A.tobytes())
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        _KERNEL_CACHE.move_to_end(key)
        return hit
    if interaction is None:
        K = _edge_kernel_torus_uncached(nu, A, n_points)
    else:
        K = _edge_kernel_torus_interaction(interaction, A, n_points)
    K.setflags(write=False)
    nbytes = K.nbytes
    if nbytes <= KERNEL_CACHE_MAX_BYTES:
        while _KERNEL_CACHE and _KERNEL_CACHE_BYTES + nbytes > KERNEL_CACHE_MAX_BYTES:
            _, ev = _KERNEL_CACHE.popitem(last=False)
            _KERNEL_CACHE_BYTES -= ev.nbytes
        _KERNEL_CACHE[key] = K
        _KERNEL_CACHE_BYTES += nbytes
    return K


def _edge_kernel_torus_uncached(nu, A, n_points) -> np.ndarray:
    """The build itself.  Split out so the cache wrapper stays readable
    and so tests can compare cached against freshly built."""
    dist = _label_distances(A, n_points)
    K = np.zeros_like(dist, dtype=float)
    nz = dist > 0.0
    if np.isinf(nu):
        # Nearest-neighbour indicator (the ν → ∞ limit): K = 1 on the
        # minimal nonzero shell, 0 elsewhere.  Keyed off the *minimal*
        # nonzero distance (not |x| == 1), so it is scale-invariant —
        # lattices whose NN distance ≠ 1 (e.g. A = 0.5·I, BCC, FCC) work
        # unchanged, whereas the literal 1/dist**inf would diverge for
        # |x| < 1.  The relative tolerance selects exactly the minimal
        # shell (the coordination-number sites).
        dmin = dist[nz].min()
        K[nz & np.isclose(dist, dmin, rtol=1e-9, atol=0.0)] = 1.0
        return K
    K[nz] = 1.0 / (dist[nz] ** float(nu))
    return K


def _edge_kernel_torus_interaction(interaction, A, n_points) -> np.ndarray:
    r"""The general-kernel build: ``interaction.sample`` on the balanced
    label grid, in this engine's power form.

    The grid is the integer twin of :func:`_label_distances` — the
    balanced axis in every dimension, ``(n,)*d + (d,)`` — so the
    power-law part lands bit-for-bit where ``_edge_kernel_torus_uncached``
    puts it (``tests/test_interaction.py`` pins that) and the compact
    table is scattered at its own labels, ``a(0)`` included: the origin
    is not special for the compact part, it weights coincident
    endpoints.  A table label outside the window raises
    :class:`gzl.interaction.InteractionSupportError` inside
    ``sample`` rather than aliasing onto the torus.
    """
    d = int(A.shape[0])
    n = int(n_points)
    z = _balanced_z_axis(n)
    labels = np.stack(np.meshgrid(*[z for _ in range(d)], indexing="ij"),
                      axis=-1)
    K = np.ascontiguousarray(
        interaction.sample(labels, A, power=POWER_TORUS), dtype=np.float64)
    if K.shape != (n,) * d:
        raise RuntimeError(
            f"interaction.sample returned shape {K.shape}, expected {(n,) * d}"
        )
    return K


#: Relative tolerance between a FINITE ``nu_vec`` entry and its kernel's
#: tail exponent in :func:`_check_kernels`.  The box's own check
#: (``direct_sum._collapse_multi_kernels``) holds 1e-12; the torus reads
#: the tail for cost (planner pin and order, peel decision, the slab's
#: inner exponent) rather than for a Richardson basis, so 1e-9 leaves
#: room for a caller that rounds (``Interaction.key`` keeps 12 decimals)
#: while refusing every disagreement that means anything: a tail 3.5
#: declared as 2.5 or 9.0 was accepted silently by every torus engine
#: (same value, the wrong plan), and the same vector is what the box's
#: basis and the router's gates read, where it is value-relevant.
_TAIL_RTOL = 1e-9


def _is_interaction_like(x) -> bool:
    """Duck-typed twin of :func:`gzl.interaction.is_interaction`:
    anything that samples on a label grid, has a cache key and carries a
    tail exponent — an :class:`~gzl.interaction.Interaction` or a
    lazy product of them.  Duck-typed so this module keeps importing
    nothing from :mod:`gzl.interaction` beyond its constants."""
    return (hasattr(x, "sample") and hasattr(x, "key")
            and hasattr(x, "tail_exponent"))


def _refuse_interaction_in_nu(nu_vec, where: str) -> None:
    r"""Raise a ``ValueError`` naming ``kernels=`` when an Interaction-like
    sits in ``nu_vec`` (or IS ``nu_vec``).

    Every engine coerces ``nu_vec`` with ``np.asarray(..., dtype=float)``,
    which turned a misplaced Interaction into numpy's opaque
    ``TypeError: float() argument must be a string or a real number,
    not 'Interaction'``.  The engines accept a general kernel only
    through the keyword ``kernels=``, with ``nu_vec`` carrying its tail
    exponent; this is the one message every torus and box entry point
    raises instead.  A float array — the legacy path — returns at the
    dtype check without touching an element, and a plain scalar returns
    at the ``list()``; neither changes what the engine then computes.
    """
    if isinstance(nu_vec, np.ndarray) and nu_vec.dtype != object:
        return
    if _is_interaction_like(nu_vec):
        items, bare = [nu_vec], True
    elif isinstance(nu_vec, np.ndarray):
        items, bare = list(nu_vec.flat), nu_vec.ndim == 0    # a 0-d object array too
    else:
        try:
            items, bare = list(nu_vec), False
        except TypeError:
            return
    for i, x in enumerate(items):
        if _is_interaction_like(x):
            slot = "nu_vec" if bare else f"nu_vec[{i}]"
            raise ValueError(
                f"{where}: {slot} is an Interaction "
                f"({type(x).__name__}); a general kernel goes through the "
                f"keyword kernels= (one Interaction per edge), and nu_vec "
                f"then carries each kernel's tail exponent "
                f"(Interaction.tail_exponent)."
            )


def _check_kernels(kernels, nu_vec, n_edges: int, edges=None):
    r"""Validate a per-edge ``kernels`` list against its tail vector.

    Returns ``None`` (the legacy path, untouched) or a list of exactly
    ``n_edges`` Interaction-likes.  ``nu_vec`` keeps carrying the per-edge
    TAIL exponents that every planner, cut, peel and refusal consumer
    reads, and it must be the kernels' own: ``nu_vec[i]`` equals
    ``kernels[i].tail_exponent`` — ``+inf`` for a purely compact kernel,
    ``min_j nu_j`` for a mixed one, the SUM of the factors' tails for a
    lazy product (parallel edges multiply, exponents add) — to
    :data:`_TAIL_RTOL` relative, or the call is refused naming the edge
    (``edges``, when given, supplies the ``(u, v)`` pair for the
    message).  An infinite tail is not interchangeable with a finite
    one: a purely compact kernel FFT-peeled under a finite tail would
    lose the integer exactness the dense contraction guarantees it, and
    a mixed kernel held dense under an infinite tail would silently pay
    a full power of ``n^d``.  A finite tail declared at the wrong value
    costs nothing on the torus but the plan (measured: tail 3.5 under
    ``nu_vec`` 2.5 or 9.0 returned the same 9.8433 on tensor, hybrid and
    slab), yet the same vector sets the box's Richardson basis and the
    router's class gates, where it is value-relevant; the contract is
    enforced here, at the one place every torus engine passes through.
    """
    if kernels is None:
        return None
    kernels = list(kernels)
    if len(kernels) != int(n_edges):
        raise ValueError(
            f"kernels must have one entry per edge; got {len(kernels)} "
            f"for {int(n_edges)} edges"
        )
    for i, k in enumerate(kernels):
        if not _is_interaction_like(k):
            raise TypeError(
                f"kernels[{i}] is not an Interaction (got "
                f"{type(k).__name__}); pass floats through nu_vec"
            )
        tail = float(k.tail_exponent)
        nu_i = float(nu_vec[i])
        if edges is None:
            where = f"edge {i}"
        else:
            where = f"edge {i} = {tuple(int(x) for x in edges[i])}"
        if bool(np.isinf(nu_i)) != bool(np.isinf(tail)):
            raise ValueError(
                f"kernels[{i}] ({where}) has tail exponent {tail!r} but "
                f"nu_vec[{i}] = {nu_i!r}; nu_vec must carry each kernel's "
                f"tail exponent (Interaction.tail_exponent), which is what "
                f"the planner, the peel and the refusals read"
            )
        # ``not (... <= ...)`` so a NaN entry is refused too.
        if (np.isfinite(tail)
                and not (abs(nu_i - tail) <= _TAIL_RTOL * max(1.0, abs(tail)))):
            raise ValueError(
                f"kernels[{i}] ({where}) has tail exponent {tail!r} but "
                f"nu_vec[{i}] = {nu_i!r}; nu_vec must carry each kernel's "
                f"tail exponent (Interaction.tail_exponent — a product "
                f"carries the SUM of its factors' tails) to {_TAIL_RTOL:g} "
                f"relative: it is what the planner, the peel, the box's "
                f"Richardson basis and the router's gates read"
            )
    return kernels


def _edge_difference_table(K: np.ndarray, n_points: int) -> np.ndarray:
    r"""Build the 2-axis edge-tensor ``T[i, j] = K[diff(i, j)]`` where
    ``i, j`` are flat indices over the ``n_points^d`` torus and
    ``diff(i, j)`` is the per-component label difference modulo
    ``n_points``.

    Returns ``T`` of shape ``(n_points^d, n_points^d)`` in ``K``'s own
    dtype (float64 for every kernel this module builds).

    The index arithmetic lives in :mod:`gzl._contract` — it is
    the torus half of the shared factor-supply layer.  This wrapper is
    the module's patch point: internal consumers (``_factor_array``,
    the bundle build) call it through this name, so a test that
    monkeypatches it still intercepts every table build.
    """
    return _contract.circulant_table(K, n_points)


def _factor_array(T, gen, n_points=None):
    r"""Return a factor's dense array, building it on first use.

    Initial edge factors are stored as their generating vector alone
    (``T is None``); the ``(n^d, n^d)`` difference table is
    reconstructed only when a consumer actually needs the dense form.
    The generator determines the table exactly — including
    orientation, since the reversed generator generates the transpose,
    and Hadamard-merged bundles, since the elementwise product of
    generators generates the elementwise product of tables.

    ``n_points`` defaults to the generator's own axis length.
    """
    if T is not None:
        return T
    return _edge_difference_table(
        gen, gen.shape[0] if n_points is None else n_points
    )


def _pin_from_generator(gen, pin_row: bool) -> np.ndarray:
    r"""One row/column of a circulant difference table, from its
    generator, without building the table.

    With ``T[i, j] = gen[(i - j) mod n]`` per axis, pinning the row at
    index 0 leaves ``gen[(-j) mod n]`` and pinning the column leaves
    ``gen[i]`` — an ``O(n^d)`` operation instead of ``O(n^(2d))``.
    """
    g = _reverse_generator(gen) if pin_row else gen
    return g.reshape(-1)


def _pin_from_generator_at(gen, flat_index: int,
                           pin_row: bool) -> np.ndarray:
    r"""One row/column of a circulant difference table at an
    **arbitrary** index, from its generator, without building the table.

    Generalises :func:`_pin_from_generator` (its index-0 special case):
    with ``T[i, j] = gen[(i - j) mod n]`` per axis, row ``t`` is
    ``gen[(t - j) mod n]`` — the reversed generator rolled by ``t`` —
    and column ``t`` is ``gen[(i - t) mod n]`` — the generator rolled
    by ``t``.  ``gen`` is the shaped ``(n,)*d`` generator (the form
    every kernel is stored in); the result is the flat ``n^d`` vector.
    Pure per-axis ``np.roll`` of copies, no arithmetic, so the gather
    is bit-identical to slicing the materialised table; this is what
    the streaming path uses to pin a terminal at each index instead of
    rebuilding the ``(n^d, n^d)`` table per index.
    """
    g = _reverse_generator(gen) if pin_row else gen
    t_idx = np.unravel_index(int(flat_index), gen.shape)
    return np.roll(g, t_idx, axis=tuple(range(gen.ndim))).reshape(-1)


def _reverse_generator(K: np.ndarray) -> np.ndarray:
    r"""Per-axis cyclic index reversal: ``K_rev[m] = K[(-m) mod n]``.

    The transpose of a circulant difference table is circulant with
    the reversed generating vector.  For the kernels built here the
    reversal is a numerical no-op (``‖A(−z)‖ = ‖Az‖`` and the label
    set maps onto itself), but it is applied explicitly so the FFT
    peel below is correct for *any* generating vector.
    """
    return _contract.reverse_generator(K)


# ---------------------------------------------------------------------------
# Min-degree elimination order
# ---------------------------------------------------------------------------

def _min_degree_order(
    adj:       dict,
    vertices:  list,
    terminals: set,
) -> list:
    r"""Pick a heuristic min-degree elimination order over non-terminal
    vertices.  Terminals are kept (not eliminated).

    Facade over :func:`gzl._elimination.min_degree_order` with
    ``fill_in=True`` — the torus's variant, which is deliberately NOT
    the same heuristic as the box's fill-in-free
    :func:`gzl.direct_sum._min_degree_order`.  Historic name and
    signature preserved for its importers (hybrid, slab and the
    tests).
    """
    return _elimination.min_degree_order(
        adj, vertices, terminals=terminals, fill_in=True)


# ---------------------------------------------------------------------------
# Bucket-elimination engine
# ---------------------------------------------------------------------------
#
# FFT fast path.  Eliminating ``v`` contracts the bag C = {v} ∪ S at
# dense cost N^|C|, N = n^d.  When some u ∈ S lies in the scope of
# exactly one bucket factor and that factor is an original circulant
# edge tensor T[i, j] = gen[(i - j) mod n], the v-sum
#
#     psi(z_u, rest) = sum_{z_v} Phi(z_v, rest) gen[(z_u - z_v) mod n]
#
# is a cyclic convolution along the v axis, done exactly by FFT at
# N^(|C|-1) log N — one full power of N less, identical to the dense
# slice-einsum up to round-off.  Steps failing the condition (e.g. a
# previously merged factor also carries u) run the dense branch
# unchanged.

def _find_conv_peel(relevant: list, v: int):
    """Locate an FFT-peelable factor in the bucket of ``v``.

    A candidate carries a generating vector, has exactly the two axes
    ``{v, u}``, and its partner ``u`` appears in no other bucket
    factor (else the remaining product still depends on ``z_u`` and
    the v-sum is not a convolution in ``z_u``).  Factors outside the
    bucket may carry ``u`` freely.  First match in list order wins
    (all candidates cost the same).  Returns ``(index, u)`` or
    ``None``.
    """
    for idx, (ax, _, gen) in enumerate(relevant):
        if gen is None or len(ax) != 2 or v not in ax:
            continue
        u = ax[0] if ax[1] == v else ax[1]
        if sum(1 for ax2, _, _ in relevant if u in ax2) == 1:
            return idx, u
    return None


def _peel_workset_bytes(n_phi_axes: int, N: int, n: int, itemsize: int = 8):
    r"""Modelled bytes of one peel chunk's working set, per unit of the
    chunk axis.

    With ``k = n_phi_axes`` axes each of length ``N = n**d``, one unit of
    the chunk axis carries ``N**(k-1)`` elements, and the peel holds
    three arrays over that index set (see :func:`_peel_convolution`):

        phi        real       ``8 * N**(k-1)``
        spectrum   complex    ``16 * N**(k-1) * (n//2 + 1) / n``
        psi        real       ``8 * N**(k-1)``

    Only the LAST of the ``d`` transformed axes is halved by ``rfftn``,
    which is where the ``(n//2 + 1) / n`` comes from.  ``spec *= g_spec``
    is in place, so unlike the box there is no separately-allocated
    spectral product to charge for — copying the box's model here is
    what makes it over-predict and chunk when it should not.

    Scope, stated exactly, because the box's twin makes a stronger claim
    and copying that claim here would be false.  This prices ONE CHUNK'S
    WORKING SET and nothing else.  It is deliberately blind to:

    * the full ``psi`` accumulator (``8 * N**k``) — allocated once and
      the same size chunked or not, so it is a floor this routine cannot
      lower.  Charging the DECISION for it would make the model refuse
      to chunk exactly when chunking is the only thing that helps;
    * the densified factor tables, the transform workspace, and every
      other subsystem in the call.

    So it does **not** bound process ``ru_maxrss``, and the tests must
    not assert that it does.  What it does over-predict is the working
    set itself: the live set peaks at ``phi + spectrum`` and again at
    ``spectrum + psi``, i.e. ``8E + 16E*half``, where this returns
    ``16E + 16E*half`` — one full real array of margin.

    Measured process-level calibration, K5 vacuum, 256 MB budget, for
    whoever prices the CORE budget later (peak against
    ``model + 8*N**k``, baseline subtracted):

        n = 8, d = 3   1.544 GiB measured   1.248 GiB modelled   1.24x
        n = 6, d = 3   0.454 GiB measured   0.324 GiB modelled   1.40x

    A peak estimate therefore needs the accumulator AND a slack factor
    of at least 1.5 — it is not this function's job to carry it, but it
    is this docstring's job to say so, because a caller that adds only
    the accumulator will under-price by ~40%.
    """
    per_unit_elems = int(N) ** (int(n_phi_axes) - 1)
    half = (int(n) // 2 + 1) / float(n)
    return int(per_unit_elems * (2.0 * itemsize + 2.0 * itemsize * half))


def _peel_chunk_length(n_phi_axes: int, N: int, n: int, axis_len: int):
    """Chunk length along the spectator axis, or ``None`` for one shot.

    ``None`` means single-shot, for either of two reasons: the modelled
    full-bag working set already fits :data:`_PEEL_BUDGET_BYTES`, or
    chunking would not buy enough to pay for itself.

    The second gate is not decoration.  A budget overshot by a hair
    yields a chunk length just under the axis — measured at d = 3,
    n = 6, where a 256.3 MB working set against a 256 MB budget split
    216 into 215 + 1.  That runs the loop twice, sets up an extra
    transform for a single fibre, and lowers the peak by 0.5 %.  So
    require the split to at least HALVE the working set; below that,
    take the sliver of overage instead.

    Both module globals are read here, at call time, so a test can
    monkeypatch either.
    """
    if axis_len <= 0:
        return None
    if _FORCE_CHUNK is not None:
        return max(1, min(int(_FORCE_CHUNK), int(axis_len)))
    per_unit = _peel_workset_bytes(n_phi_axes, N, n)
    if per_unit <= 0 or per_unit * int(axis_len) <= _PEEL_BUDGET_BYTES:
        return None
    c = max(1, min(int(_PEEL_BUDGET_BYTES // per_unit), int(axis_len)))
    if 2 * c > int(axis_len):
        return None
    return c


def _chunked_peel_convolution(remaining, phi_axes, v, u, chunk_axis,
                              chunk_len, g_use, out_axes_sorted,
                              res_dtype, n, d, N):
    """The peel, chunk by chunk along one spectator axis.

    Elementwise identical to the single-shot body below: each fibre of
    the chunk axis is multiplied and transformed exactly as it would be
    in the full-bag product, so slice-then-convolve equals
    convolve-then-slice bit for bit.

    Three deliberate differences from ``direct_sum``'s chunked peel,
    each of which would be a bug if copied across:

    1. The transform is CYCLIC — ``rfftn``/``irfftn`` at length ``n``
       per axis.  No zero padding, no ``next_fast_len``, no window
       extraction, no per-vertex extents or origins.
    2. ``psi`` is preallocated in ``out_axes_sorted`` order and each
       chunk's transpose is written straight into its slice.  That
       removes the single-shot path's trailing
       ``ascontiguousarray(transpose(...))`` — a second full copy of the
       result — rather than reproducing it per chunk.
    3. There is no Z2 marker fold: ``TorusTruncation.weight`` is
       ``None``, so the box's half-axis weighting has no analogue here.

    ``g_spec`` is transformed ONCE per step, never per chunk: it depends
    only on the kernel, and rebuilding it inside the loop would redo an
    identical transform for every chunk.
    """
    k = len(phi_axes)
    p = phi_axes.index(v)
    c_pos = phi_axes.index(chunk_axis)

    # Output axis order, exactly as the single-shot path produces it.
    res_axes = [u if a == v else a for a in phi_axes]
    perm = [res_axes.index(a) for a in out_axes_sorted]
    q = list(out_axes_sorted).index(chunk_axis)

    psi_full = np.empty((N,) * k, dtype=res_dtype)

    # Densify each factor ONCE, not per chunk, and hoist the kernel
    # transform out of the loop.
    arrays = [(_factor_array(T, g2), ax2) for ax2, T, g2 in remaining]
    g_spec = np.fft.rfftn(g_use)
    g_bshape = (1,) * p + g_spec.shape + (1,) * (k - 1 - p)

    for start in range(0, N, chunk_len):
        sel = slice(start, min(start + chunk_len, N))
        n_sel = sel.stop - sel.start

        Phi = np.ones(
            tuple(n_sel if i == c_pos else N for i in range(k)),
            dtype=res_dtype,
        )
        for arr, ax2 in arrays:
            a = arr
            if chunk_axis in ax2:
                # A basic slice is a VIEW; np.take with an index array
                # would copy the chunk of every factor per chunk, a cost
                # the memory model does not count.
                idx = [slice(None)] * a.ndim
                idx[list(ax2).index(chunk_axis)] = sel
                a = a[tuple(idx)]
            bshape = tuple(
                ((n_sel if ax == chunk_axis else N) if ax in ax2 else 1)
                for ax in phi_axes
            )
            Phi *= a.reshape(bshape)

        flat_shape = Phi.shape
        Phi = Phi.reshape(flat_shape[:p] + (n,) * d + flat_shape[p + 1:])
        fft_axes = tuple(range(p, p + d))
        spec = np.fft.rfftn(Phi, axes=fft_axes)
        del Phi
        spec *= g_spec.reshape(g_bshape)
        psi_c = np.fft.irfftn(spec, s=(n,) * d, axes=fft_axes)
        del spec

        dest = [slice(None)] * k
        dest[q] = sel
        psi_full[tuple(dest)] = np.transpose(
            psi_c.reshape(flat_shape), perm,
        )

    return (out_axes_sorted, psi_full, None)


def _peel_convolution(relevant: list, peel_idx: int, v: int, u: int,
                      out_axes_sorted: list, res_dtype) -> tuple:
    """Contract the bucket of ``v`` by cyclic FFT convolution.

    The peeled factor's generator follows the row-minus-column
    convention ``T[i, j] = gen[(i - j) mod n]`` on its sorted axes:
    with ``v`` the column axis the v-sum convolves with ``gen``
    directly, with ``v`` the row axis it convolves with the reversed
    generator.  Returns the replacement ``(axes, tensor, None)``
    triple for the whole bucket.
    """
    ax_p, _, gen = relevant[peel_idx]
    d = gen.ndim
    n = int(gen.shape[0])
    N = n ** d
    g_use = gen if ax_p[1] == v else _reverse_generator(gen)

    remaining = relevant[:peel_idx] + relevant[peel_idx + 1:]
    if not remaining:
        # psi(z_u) = sum_v gen(...) is constant in z_u: O(N) instead
        # of the dense O(N^2).
        return (out_axes_sorted,
                np.full(N, g_use.sum(), dtype=res_dtype), None)

    # Phi over sorted(all axes - {u}) — contains v; every factor's
    # axis list is ascending, hence an ordered subsequence of
    # phi_axes, so a reshape embedding broadcasts correctly.
    phi_axes = sorted({a for ax2, _, _ in remaining for a in ax2})

    # Chunk along the LAST spectator, never along v: v is the axis the
    # convolution runs over, and a partial fibre of it would wrap a
    # partial sum around the torus.  Last, because it is the fastest
    # varying in C order, so the per-chunk factor slices below are views
    # rather than copies.
    spectators = [a for a in phi_axes if a != v]
    chunk_len = (
        _peel_chunk_length(len(phi_axes), N, n, N) if spectators else None
    )
    if chunk_len is not None and chunk_len < N:
        return _chunked_peel_convolution(
            remaining, phi_axes, v, u, spectators[-1], chunk_len,
            g_use, out_axes_sorted, res_dtype, n, d, N,
        )

    Phi = np.ones((N,) * len(phi_axes), dtype=res_dtype)
    for ax2, T, g2 in remaining:
        bshape = tuple(N if a in ax2 else 1 for a in phi_axes)
        Phi *= _factor_array(T, g2).reshape(bshape)

    # Cyclic convolution along v's (n,)*d position axes (real FFT —
    # the exact convolution of real operands is real).
    p = phi_axes.index(v)
    flat_shape = Phi.shape
    Phi = Phi.reshape(flat_shape[:p] + (n,) * d + flat_shape[p + 1:])
    fft_axes = tuple(range(p, p + d))
    spec = np.fft.rfftn(Phi, axes=fft_axes)
    del Phi
    g_spec = np.fft.rfftn(g_use)
    spec *= g_spec.reshape(
        (1,) * p + g_spec.shape + (1,) * (len(phi_axes) - 1 - p)
    )
    psi = np.fft.irfftn(spec, s=(n,) * d, axes=fft_axes)
    del spec
    psi = psi.reshape(flat_shape)

    # Relabel the convolved axis v -> u and order axes as
    # out_axes_sorted expects.
    res_axes = [u if a == v else a for a in phi_axes]
    perm = [res_axes.index(a) for a in out_axes_sorted]
    # Materialise the permutation: a strided view here would be paid
    # back many times over by the next dense step, whose per-slice
    # np.take gathers along the transposed axis (measured 5x on the
    # step, cancelling this step's saving).  The copy is one pass over
    # a tensor the step already holds.
    psi = np.ascontiguousarray(np.transpose(psi, perm), dtype=res_dtype)
    return (out_axes_sorted, psi, None)


def _peel_sparse(relevant: list, peel_idx: int, v: int, u: int,
                 out_axes_sorted: list, res_dtype) -> tuple:
    r"""Contract the bucket of ``v`` by an EXACT sparse convolution.

    The compact twin of :func:`_peel_convolution`, for a bundle whose
    generator is a table (a purely compact kernel, ``nu = inf``): the
    v-sum is the same cyclic convolution, written out over the table's
    non-zero labels,

        psi(z_u, rest) = sum_{y : g(y) != 0} g(y) Phi(z_u - y, rest),

    one rolled copy of ``Phi`` per label.  Every term is a product and
    a sum the dense contraction also forms -- the zeros it skips
    contribute exactly nothing -- so an integer table (the
    nearest-neighbour indicator, a homomorphism count) stays exactly
    integral where an FFT round trip would return 2.0 as
    1.9999999999999998, and a real table agrees with the dense step to
    round-off.  Cost ``|S| N^k`` for ``|S|`` non-zero labels and ``k``
    remaining axes, against the dense ``N^(k+1)``: the same power of
    ``N = n^d`` the FFT peel removes, at ``|S|`` in place of ``log N``.
    Not chunked: the working set is ``Phi``, ``psi`` and one rolled
    copy.  Same generator orientation and output layout as the FFT
    peel, so the two are interchangeable to the executor.
    """
    ax_p, _, gen = relevant[peel_idx]
    d = gen.ndim
    n = int(gen.shape[0])
    N = n ** d
    g_use = gen if ax_p[1] == v else _reverse_generator(gen)

    remaining = relevant[:peel_idx] + relevant[peel_idx + 1:]
    if not remaining:
        # psi(z_u) = sum_v gen(...) is constant in z_u.
        return (out_axes_sorted,
                np.full(N, g_use.sum(), dtype=res_dtype), None)

    phi_axes = sorted({a for ax2, _, _ in remaining for a in ax2})
    Phi = np.ones((N,) * len(phi_axes), dtype=res_dtype)
    for ax2, T, g2 in remaining:
        bshape = tuple(N if a in ax2 else 1 for a in phi_axes)
        Phi *= _factor_array(T, g2).reshape(bshape)

    p = phi_axes.index(v)
    flat_shape = Phi.shape
    Phi = Phi.reshape(flat_shape[:p] + (n,) * d + flat_shape[p + 1:])
    roll_axes = tuple(range(p, p + d))
    psi = np.zeros_like(Phi)
    # np.roll(Phi, y)[i] = Phi[i - y]: the shift IS the table label, on
    # the same index arithmetic mod n the circulant table uses.
    for label in zip(*np.nonzero(g_use)):
        rolled = np.roll(Phi, shift=tuple(int(x) for x in label),
                         axis=roll_axes)
        rolled *= g_use[label]
        psi += rolled
    del Phi
    psi = psi.reshape(flat_shape)

    res_axes = [u if a == v else a for a in phi_axes]
    perm = [res_axes.index(a) for a in out_axes_sorted]
    psi = np.ascontiguousarray(np.transpose(psi, perm), dtype=res_dtype)
    return (out_axes_sorted, psi, None)


def _eliminate_vertex(
    tensors:  list,
    v:        int,
    trunc:    "TorusTruncation",
) -> list:
    r"""Eliminate ``v`` from the active tensor list.

    Adapter over the shared skeleton in :mod:`gzl._contract`:
    the control flow (partition, empty bucket, peel gate, dense
    fallthrough) is :func:`_contract.eliminate_one`, and every
    arithmetic decision is a :class:`TorusTruncation` method whose body
    is this module's shipped code, moved verbatim — see the class for
    the memory profile and branch semantics that used to be documented
    here.  Factors remain ``(axis_order, ndarray, gen)`` triples.
    """
    return _contract.eliminate_one(tensors, v, trunc)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def graph_zeta_general(
    edges_flat:    np.ndarray,
    nu_vec:        np.ndarray,
    A:             "np.ndarray | str",
    n_points:      int,
    *,
    source:        "int | None" = None,
    terminals:     "Sequence[int]" = (),
    space:         str = "z",
    momentum:      "np.ndarray | None" = None,
    kernels:       "Sequence | None" = None,
) -> np.ndarray:
    r"""Treewidth-agnostic Fourier-grid evaluation of a graph zeta
    function via bucket elimination on a torus of ``n_points^d`` sites
    per vertex axis.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    The ``source`` vertex is pinned at the origin (label index 0 in
    each dimension).  Every vertex listed in ``terminals`` is **kept
    as a free axis** in the output tensor, so the result depends on
    those terminal positions (or their conjugate momenta).  All other
    vertices are summed out along the bucket-elimination order.

    Parameters
    ----------
    edges_flat : ndarray, shape (E, 2), int
        Already-expanded multigraph.  Each row ``(u, v)`` is a single
        edge.  Vertex labels are non-negative integer NAMES: the vertex
        set is the labels present in ``edges_flat``, compressed
        order-preservingly to ``0..V-1`` on entry (a gap label is not a
        vertex; see gzl/_labels.py).  Multiple parallel edges are
        encoded as multiple rows.
    nu_vec : ndarray, shape (E,), float
        Per-edge exponents.
    A : ndarray, shape (d, d), float, or str
        Lattice basis, or the name of a lattice -- ``"chain"``,
        ``"square"``, ``"triangular"``, ``"cubic"``.  Supports ``d ≥ 1``;
        at ``d ≥ 3`` the edge difference table is ``(n^d) × (n^d)`` so
        practical ``n_points`` is bounded (≤ 16 at d=3 with tw ≤ 2 fits
        comfortably in RAM).
    n_points : int
        Discretisation points per dimension on the torus.
    source : int or None, optional
        Vertex pinned at the origin.  Default ``None``: on the vacuum
        path (no terminals, no momentum) the planner chooses the pin —
        value-exact by translation invariance on the torus, it only
        decides which end of ``tw(G) - 1 <= tw(G - p) <= tw(G)`` the
        contraction pays — while terminal/momentum calls pin vertex 0
        as before (their kept-axis layout and cos weight encode
        ``x_source = 0``).  Pass an explicit vertex to fix the pin
        everywhere.
    terminals : sequence of int, optional
        Vertices kept as free axes.  Empty (default) reproduces the
        :func:`graph_zeta_general_at_zero` scalar result.  When
        non-empty, the result is a tensor with one axis per terminal
        of size ``n_points^d`` (flat index over the d-dimensional
        torus, balanced-z convention).
    space : ``'z'`` or ``'k'``, optional
        Output representation.  ``'z'`` (default) returns the
        bucket-elimination output directly: a tensor over terminal
        *positions* in label space, matching the lib's
        :class:`gzl.GraphZeta` ``aMat`` convention.  ``'k'``
        applies an inverse-FFT-with-`n^d`-rescaling along each
        terminal axis to convert to *momentum* space, matching what
        :func:`gzl.graph_sample` returns at each k-grid point.
    kernels : sequence of Interaction, optional
        General per-edge kernels
        (:class:`gzl.interaction.Interaction` or lazy products),
        aligned with ``edges_flat``.  ``nu_vec`` then carries each
        kernel's TAIL exponent (``Interaction.tail_exponent``; ``+inf``
        for a purely compact kernel), which is what the planner, the
        FFT peel and the ν = ∞ bookkeeping read — so a purely compact
        bundle takes the exact sparse peel (``_peel_sparse``: rolls
        over the table's labels, integer-exact) exactly like
        ``nu = inf``, while a mixed kernel FFT-peels.  Only the kernel
        BUILD consults this argument; ``None`` (default) is the legacy
        power-law path, byte-identical.

    Returns
    -------
    ndarray
        - ``terminals = ()``: 0-d array containing the scalar
          :math:`\zeta_g(\boldsymbol{0}, \dots, \boldsymbol{0})`.
        - otherwise: shape ``(n_points ** d,) * len(terminals)``, one
          flat axis of ``n_points ** d`` positions or momenta per
          terminal (the edge-tensor difference-table layout), ordered
          by sorted terminal vertex label.  Real ``float64`` for
          ``space='z'``, ``complex128`` for ``space='k'``.

    Notes
    -----
    :math:`\mathcal{O}\!\bigl(V \cdot n_\text{points}^{(\tau + 1 + N_t)\,d}\bigr)`
    (:math:`\tau` the treewidth, :math:`N_t = \mathrm{len(terminals)}`;
    the pinned source axis doesn't count) is the **dense upper bound**,
    not the cost: the FFT peel takes one power of :math:`n^d` off every
    step that isolates an original kernel, and the planner-chosen
    order (plus pin, on the vacuum path) realises the post-peel
    exponent :math:`\tau \cdot d` — the invariant
    ``tests/test_tau_d_acceptance.py`` enforces per instrumented step.
    Never quote the dense bound as the cost.
    """
    _refuse_interaction_in_nu(nu_vec, "graph_zeta_general")
    edges_flat = np.asarray(edges_flat, dtype=int)
    nu_vec     = np.asarray(nu_vec, dtype=float)
    A          = np.asarray(_resolve_lattice(A), dtype=float)

    d = int(A.shape[0])
    if d < 1:
        raise ValueError(f"graph_zeta_general requires d ≥ 1; got d = {d}.")
    if space not in ("z", "k"):
        raise ValueError(f"space must be 'z' or 'k'; got {space!r}.")

    # `momentum` mode: weight the single free terminal vertex by the
    # real cos kernel cos(2π k_frac · x_term) (see the weight-tensor
    # construction below for the inversion-symmetry argument) and
    # eliminate it as a regular vertex, returning a real scalar at
    # cost O(V · n^((tw+1) d)) — saves the kept-terminal-axis factor
    # n^d compared with the default `terminals=(t,), space='k'` path
    # (which produces the full grid and applies one cheap final FFT).
    if momentum is not None:
        if len(terminals) != 1:
            raise NotImplementedError(
                "graph_zeta_general(momentum=...) requires exactly one "
                f"terminal vertex; got terminals={list(terminals)}."
            )
        momentum = np.asarray(momentum, dtype=float).reshape(-1)
        if momentum.size != d:
            raise ValueError(
                f"momentum has wrong shape: expected length {d}, "
                f"got size {momentum.size}."
            )

    if edges_flat.size == 0:
        # Empty edge set: a graph with no edges contributes 1 (vacuum).
        # Result is real (k = 0 path) or complex (finite k); the finite-k
        # path is currently a degenerate scalar carrying no phase, so a
        # real 1.0 is correct in both cases.
        if not terminals or momentum is not None:
            return np.array(1.0, dtype=float)
        # No edges and free terminals: indeterminate (vertex positions
        # uncoupled).  Treat as 1 broadcast over all terminal axes.
        n = int(n_points)
        return np.ones((n ** d,) * len(terminals), dtype=float)
    if edges_flat.ndim != 2 or edges_flat.shape[1] != 2:
        raise ValueError("edges_flat must have shape (E, 2)")
    if len(nu_vec) != len(edges_flat):
        raise ValueError("nu_vec must have one entry per edge")
    kernels = _check_kernels(kernels, nu_vec, len(edges_flat), edges_flat)

    # The vertex set is the edge support (see gzl/_labels.py):
    # sparse labels are compressed order-preservingly, so a relabelled
    # graph is bit-identical to its contiguous twin, and a contiguous
    # input passes through as the same object.  A source or terminal
    # label that appears in no edge is not a vertex and raises.
    _refs = {"source": None if source is None else int(source)}
    _refs.update({f"terminals[{i}]": int(t) for i, t in enumerate(terminals)})
    edges_flat, _refs = _relabel_to_support(edges_flat, _refs)

    V         = int(edges_flat.max()) + 1
    n         = int(n_points)
    src       = 0 if source is None else int(_refs["source"])

    free_terminals = [int(_refs[f"terminals[{i}]"])
                      for i in range(len(terminals))]
    if src in free_terminals:
        raise ValueError("source must not appear in terminals")
    if len(set(free_terminals)) != len(free_terminals):
        raise ValueError("terminals must be a list of distinct vertices")

    # In momentum mode the terminal is weighted and eliminated rather
    # than kept as a free axis; remember the vertex label and switch
    # to the no-free-terminals contraction schedule.
    momentum_term: int | None = None
    if momentum is not None:
        momentum_term = int(free_terminals[0])
        free_terminals = []

    # 1. underlying simple-graph adjacency (for the elimination heuristic) ---
    adj = {v: set() for v in range(V)}
    for u, w in edges_flat:
        u, w = int(u), int(w)
        if u != w:
            adj[u].add(w)
            adj[w].add(u)

    # 1b. Planner pin + order on the vacuum path.
    #     The torus is translation-closed, so re-pinning a vacuum call
    #     is value-exact and the pin is a free optimisation; a caller
    #     passing an explicit ``source`` keeps it (the repin
    #     value-exactness test relies on that to stay meaningful), and
    #     terminal/momentum calls keep pin 0 because the kept-axis
    #     layout and the cos weight encode x_source = 0.  ``plan`` is
    #     resolved through the module attribute at call time (the guard
    #     suite's spy patches it); a budget refusal falls back to the
    #     shipped fixed-pin best_order path at step 6.
    _bundle_pairs = sorted({(min(a, b), max(a, b))
                            for a in adj for b in adj[a]})
    # Bundles carrying any nu = inf edge never take the FFT peel
    # (integer-exact NN contraction, see peelable[] below).  With the
    # sparse peel on they still peel -- by rolls, under the same
    # structural condition -- so the planner prices them as kernel
    # edges; with it off they are dense and the planner must know, or
    # its exponent reads optimistic and the tau*d acceptance would see
    # a silent degradation.
    _inf_pairs = sorted({
        (min(int(edges_flat[i, 0]), int(edges_flat[i, 1])),
         max(int(edges_flat[i, 0]), int(edges_flat[i, 1])))
        for i in range(len(edges_flat)) if np.isinf(float(nu_vec[i]))
    })
    _non_kernel_pairs = () if _USE_SPARSE_PEEL else _inf_pairs
    planned_order: "list | None" = None
    if source is None and not free_terminals and momentum is None:
        try:
            _p = _elimination.plan(_bundle_pairs, range(V),
                                   non_kernel_edges=_non_kernel_pairs)
        except _elimination.PlanBudgetExceededError:
            _p = None
        if _p is not None:
            src = _p.pin
            planned_order = list(_p.order)

    # 2. one cached real-space kernel per unique exponent --------------------
    nu_to_K = {}
    edge_K: "list | None" = None
    if kernels is None:
        for nu in np.unique(nu_vec):
            nu_to_K[float(nu)] = _edge_kernel_torus(float(nu), A, n)
    else:
        # General kernels: one build per EDGE through the same cache,
        # which dedupes on the kernel's own key rather than on its
        # tail — two kernels sharing a tail must never share an array.
        edge_K = [
            _edge_kernel_torus(float(nu_vec[i]), A, n, interaction=kernels[i])
            for i in range(len(edges_flat))
        ]

    # 3. build initial tensor list, one per *unique* edge pair --------------
    #    Parallel edges (multi-edges) between the same pair of vertices are
    #    pre-merged via Hadamard product into a single tensor.  This is
    #    mathematically transparent — each edge contributes an independent
    #    factor of the propagator to the integrand, so two parallel edges
    #    with exponents nu1 and nu2 between u and v contribute
    #    K_{nu1}(z_u - z_v) * K_{nu2}(z_u - z_v) at every position pair.
    #    Pre-merging keeps each bucket-elimination step from inflating
    #    pairwise einsum intermediates by the multi-edge count, which can
    #    push memory through the roof for high-multiplicity high-order
    #    1qp graphs at large n_points (e.g. tw=3 graphs at n=128 in d=1
    #    would otherwise materialise (n^2, n^2) = 4 GiB matmul intermediates).
    #    A bundle is represented by its circulant generating vector
    #    alone (row-minus-column convention: T[i, j] = gen[(i - j) mod
    #    n] per axis, so the u > w orientation stores the index-
    #    reversed kernel, and merged parallel edges multiply their
    #    generators elementwise).  That vector determines the table
    #    exactly and is n^d rather than n^(2d), so the table is built
    #    only where a consumer needs the dense form: source-incident
    #    edges never need one (the pinning step below reads a single
    #    row straight off the generator), and neither do edges the FFT
    #    fast path peels.
    gen_dict: dict[tuple[int, int], np.ndarray] = {}
    peelable: dict[tuple[int, int], bool] = {}
    for e_idx in range(len(edges_flat)):
        u, w = int(edges_flat[e_idx, 0]), int(edges_flat[e_idx, 1])
        if u == w:
            raise NotImplementedError("self-loops are not supported")
        nu_e = float(nu_vec[e_idx])
        key  = (u, w) if u < w else (w, u)
        K_e  = nu_to_K[nu_e] if edge_K is None else edge_K[e_idx]
        g    = K_e if u < w else _reverse_generator(K_e)
        gen_dict[key] = gen_dict[key] * g if key in gen_dict else g.copy()
        # Nearest-neighbour indicator: the dense contraction is exact
        # integer arithmetic (these are homomorphism counts), and an
        # FFT round trip would return 2.0 as 1.9999999999999998.  Such
        # bundles never take the FFT fast path.  They take the EXACT
        # sparse peel instead (``_peel_sparse``, the same convolution
        # over the table's non-zero labels): the bundle stays lazy and
        # the truncation is told which pairs are compact, so the step
        # that isolates one peels by rolls rather than densifying.
        # With ``_USE_SPARSE_PEEL`` off the bundle keeps its eager
        # table, as it always did.
        peelable[key] = peelable.get(key, True) and not np.isinf(nu_e)
    compact_pairs = (frozenset(k for k, ok in peelable.items() if not ok)
                     if _USE_SPARSE_PEEL else frozenset())
    tensors: list = [
        (list(k),
         None if (peelable[k] or k in compact_pairs)
         else _edge_difference_table(g, n),
         g if (peelable[k] or k in compact_pairs) else None)
        for k, g in gen_dict.items()
    ]

    # Momentum-weight tensor for the single-k path: a 1-axis tensor
    # attached to the terminal vertex, so that bucket-eliminating
    # ``x_term`` along with the rest folds the Fourier kernel of the
    # terminal-source displacement into the contraction.
    #
    # Real cos weight (not the complex phase exp(2πi k·z)).  ζ_G(k) is
    # real on any Bravais lattice: pairing each configuration X with
    # its inversion −X leaves every kernel norm invariant and flips
    # the sign of the terminal-source displacement, so
    #
    #     ζ_G(k) = Σ_configs cos(2π k·(x_t − x_s)) · Π_e 1/‖·‖^ν_e.
    #
    # Every kernel factor is real, hence
    #     Re[ Σ exp(−2πi k·x_t) Π K_e ] = Σ cos(2π k·x_t) Π K_e
    # exactly (linearity).  The caller already keeps only the real
    # part of the single-k result, so the cos weight is the value the
    # pipeline returns — computed in real arithmetic from the start.
    # With an all-real tensor pool ``res_dtype`` stays float64, so the
    # whole single-k bucket elimination runs real (≈2-4× faster BLAS,
    # half the memory) at the same asymptotic cost.  cos is even, so
    # the exponent-sign convention is irrelevant; at k = 0 this
    # reduces exactly to the real zero-momentum path.
    #
    # The flat index in the kernel tensors corresponds to a position
    # in the balanced-z-axis layout (``_balanced_z_axis(n)``), not to
    # an unsigned integer 0..n-1 — so the weight has to use the same
    # signed positions, otherwise the discrete Fourier extrapolation
    # at off-grid k is biased and loses lattice inversion symmetry.
    if momentum_term is not None:
        z = _balanced_z_axis(n).astype(float)
        z_grid = np.meshgrid(*[z for _ in range(d)], indexing="ij")
        k_dot_z = sum(z_grid[i] * float(momentum[i]) for i in range(d))
        weight = np.cos(2.0 * np.pi * k_dot_z).reshape(n ** d)
        tensors.append(([momentum_term], weight, None))

    # 5. apply pin to source: select flat index 0 on every tensor that
    #    contains the source (label index 0 corresponds to the multi-dim
    #    origin (0, …, 0)).  Free terminals remain as full axes for the
    #    fast batched path; if that path runs out of memory we retry below
    #    with the terminal-streaming path.  Pinning breaks translation
    #    invariance, so the generating vector is dropped.
    src_pinned: list = []
    for axes, T, gen in tensors:
        if src in axes:
            ax_idx = axes.index(src)
            if T is None and len(axes) == 2:
                # Still lazy: read the pinned row/column straight off
                # the generator instead of building the n^(2d) table
                # only to take one slice of it.  This is what keeps a
                # source-incident edge from ever allocating a table.
                T = _pin_from_generator(gen, pin_row=(ax_idx == 0))
            else:
                T = np.take(_factor_array(T, gen, n), 0, axis=ax_idx)
            axes = [a for a in axes if a != src]
            gen = None
        src_pinned.append((axes, T, gen))

    sorted_terminals = sorted(free_terminals)
    n_term = len(sorted_terminals)

    # Dtype the final result will carry.  Float64 on every internal
    # path: the kernels are real and the single-k weight is the real
    # cos tensor (see the momentum-weight construction above), so the
    # whole bucket elimination runs in real arithmetic.  Kept as a
    # ``result_type`` over the pool so a caller passing complex kernels
    # directly still promotes correctly.
    res_dtype = np.result_type(
        *[(T if T is not None else g).dtype for _, T, g in src_pinned]
    ) if src_pinned else np.dtype(float)

    # 6. eliminate non-source, non-terminal vertices ------------------------
    # The order is chosen to minimise the achieved cost exponent
    # (|bag| minus one on a peelable step), not merely the maximum bag:
    # a bag-optimal order can strand a peak step that cannot peel while
    # another order of the same width can, which costs a full power of
    # n^d on the step that dominates.  Measured over the shipped corpus,
    # the min-degree order pays that extra power on 17.7% of distinct
    # tw>=3 blocks.  On the vacuum path with ``source=None`` the PIN
    # comes from the planner too (step 1b — value-exact on the periodic
    # scheme, and only there: a box [-L, L]^d is not closed under
    # translation, see ``_elimination.best_order``); every other call
    # keeps its fixed pin and the fixed-pin exact DP below.
    # One truncation object per call: its methods read this module's
    # tunables late-bound, so constructing it here (not caching it)
    # matches the per-call semantics every kill-switch test relies on.
    trunc = TorusTruncation(n, d, A, compact_pairs=compact_pairs)

    do_not_eliminate = {src, *free_terminals}
    if planned_order is not None:
        order = planned_order
    else:
        order = _best_order(
            _bundle_pairs, range(V), src,
            keep=free_terminals,
            fallback=lambda: _min_degree_order(
                adj, list(range(V)), terminals=do_not_eliminate),
            non_kernel_edges=_non_kernel_pairs,
        )

    # ---------------------------------------------------------------------
    # Fast path: keep free terminals as broadcast axes through bucket
    # elimination, then einsum-combine the residual tensors over the
    # terminal axes.  Peak memory is :math:`n^{(\tau+1+|terminals|)\,d}`
    # in the worst bag.  This is the cheaper schedule (fewer total Python
    # iterations of the elimination loop) when memory permits.
    # ---------------------------------------------------------------------
    def _batched_path() -> np.ndarray:
        cur = [(list(ax), T, g) for ax, T, g in src_pinned]
        for vv in order:
            cur = _eliminate_vertex(cur, vv, trunc)
        if not cur:
            if not free_terminals:
                return np.array(1.0, dtype=res_dtype)
            return np.ones((n ** d,) * n_term, dtype=res_dtype)

        if not free_terminals:
            r = np.array(1.0, dtype=res_dtype)
            for ax, T, g in cur:
                T = _factor_array(T, g)
                if T.ndim > 0:
                    T = T.flatten().sum() if len(ax) > 0 else T
                r = r * T
            return np.asarray(r, dtype=res_dtype)

        constant_factor = np.array(1.0, dtype=res_dtype)
        bucket: list = []
        for ax, T, g in cur:
            T = _factor_array(T, g)
            if not ax:
                constant_factor = constant_factor * T
            else:
                bucket.append((ax, T))

        if not bucket:
            return constant_factor * np.ones(
                (n ** d,) * n_term, dtype=res_dtype,
            )

        all_axes_set = set()
        for ax, _ in bucket:
            all_axes_set.update(ax)
        if not all_axes_set.issubset(sorted_terminals):
            raise RuntimeError(
                f"unexpected leftover axes after elimination: "
                f"{sorted(all_axes_set - set(sorted_terminals))}"
            )
        if n_term > len(string.ascii_lowercase):
            raise NotImplementedError(
                f"too many free terminals ({n_term}); einsum letters exhausted."
            )
        letter = {a: string.ascii_lowercase[i] for i, a in enumerate(sorted_terminals)}
        in_specs = [''.join(letter[a] for a in ax) for ax, _ in bucket]
        out_spec = ''.join(letter[a] for a in sorted_terminals)
        eq       = ','.join(in_specs) + '->' + out_spec
        arrays   = [T for _, T in bucket]
        return constant_factor * np.einsum(eq, *arrays, optimize=True)

    # ---------------------------------------------------------------------
    # Memory-safe partial-pinning path.  Used as a fallback when
    # :func:`_batched_path` raises a memory error.
    #
    # Strategy: run as many "cheap" elimination steps as possible
    # batched (terminals carried as free axes), then switch to per-
    # terminal-index streaming once the next batched step would produce
    # a result tensor exceeding ``mem_budget_bytes``.  The cheap prefix
    # is computed only ONCE; only the expensive suffix is repeated
    # ``n^(d·|terminals|)`` times.  This avoids the wasted work of the
    # naive "always-stream" schedule, which redoes every elimination
    # for every terminal value.
    #
    # Mathematically identical to the batched path; total ops unchanged.
    # ---------------------------------------------------------------------
    def _bag_result_size_bytes(cur: list, vv: int) -> int:
        """Estimated size of the result tensor of eliminating ``vv``
        from the current tensor list, in bytes (using ``res_dtype``
        itemsize so the budget tracks real-vs-complex correctly)."""
        all_ax: set = set()
        for ax, _, _ in cur:
            if vv in ax:
                all_ax.update(a for a in ax if a != vv)
        return int(res_dtype.itemsize) * (n ** d) ** len(all_ax)

    def _streaming_path() -> np.ndarray:
        # Phase 1: run as many batched eliminations as possible.
        cur = [(list(ax), T, g) for ax, T, g in src_pinned]
        k_split = len(order)
        for k, vv in enumerate(order):
            if _bag_result_size_bytes(cur, vv) > _MEM_BUDGET_BYTES:
                k_split = k
                break
            cur = _eliminate_vertex(cur, vv, trunc)
        suffix_order = order[k_split:]

        # If all eliminations fit, finalise in batched mode.
        if not suffix_order:
            if n_term == 0:
                r = np.array(1.0, dtype=res_dtype)
                for ax, T, g in cur:
                    T = _factor_array(T, g)
                    if ax:
                        T = np.asarray(T).sum()
                    r = r * T
                return np.asarray(np.asarray(r).item(), dtype=res_dtype)
            # Combine remaining tensors over terminal axes.
            constant_factor = np.array(1.0, dtype=res_dtype)
            bucket: list = []
            for ax, T, g in cur:
                T = _factor_array(T, g)
                if not ax:
                    constant_factor = constant_factor * T
                else:
                    bucket.append((ax, T))
            if not bucket:
                return constant_factor * np.ones(
                    (n ** d,) * n_term, dtype=res_dtype,
                )
            letter = {a: string.ascii_lowercase[i]
                      for i, a in enumerate(sorted_terminals)}
            in_specs = [''.join(letter[a] for a in ax) for ax, _ in bucket]
            out_spec = ''.join(letter[a] for a in sorted_terminals)
            eq = ','.join(in_specs) + '->' + out_spec
            return constant_factor * np.einsum(eq, *bucket and [T for _, T in bucket],
                                               optimize=True)

        # Phase 2: per-terminal-index streaming over the remaining
        # eliminations.  Pin each terminal to one value at a time.
        if n_term == 0:
            # No free terminals: run the suffix once.
            r = np.array(1.0, dtype=res_dtype)
            cur2 = [(list(ax), T, g) for ax, T, g in cur]
            for vv in suffix_order:
                cur2 = _eliminate_vertex(cur2, vv, trunc)
            for ax, T, g in cur2:
                T = _factor_array(T, g)
                if ax:
                    T = np.asarray(T).sum()
                r = r * T
            return np.asarray(np.asarray(r).item(), dtype=res_dtype)

        out_shape = (n ** d,) * n_term
        out_z = np.empty(out_shape, dtype=res_dtype)
        for flat in range(int(np.prod(out_shape))):
            term_idx = np.unravel_index(flat, out_shape)
            pinned: list = []
            for ax, T, gen in cur:
                new_ax = list(ax)
                arr    = T
                for k_term, term_v in enumerate(sorted_terminals):
                    if term_v in new_ax:
                        pos = new_ax.index(term_v)
                        if arr is None and len(new_ax) == 2:
                            # Still-lazy circulant kernel: gather the
                            # pinned row/column straight off the
                            # generator (bit-identical, O(n^d)).
                            # Materialising the full (n^d, n^d) table
                            # here would build it once per terminal
                            # index only to keep one slice of it —
                            # n^(d·|terminals|) table builds per call.
                            arr = _pin_from_generator_at(
                                gen, term_idx[k_term],
                                pin_row=(pos == 0))
                        else:
                            arr = np.take(_factor_array(arr, gen),
                                          term_idx[k_term], axis=pos)
                        new_ax = [a for a in new_ax if a != term_v]
                        gen = None      # pinning breaks circulance
                pinned.append((new_ax, arr, gen))
            cur2 = pinned
            for vv in suffix_order:
                cur2 = _eliminate_vertex(cur2, vv, trunc)
            s = np.array(1.0, dtype=res_dtype)
            for ax, T, g in cur2:
                T = _factor_array(T, g)
                if ax:
                    T = np.asarray(T).sum()
                s = s * T
            out_z.flat[flat] = np.asarray(s).item()
        return out_z

    # 7. try batched first, fall back to streaming on memory pressure -----
    if _FORCE_STREAMING:
        out_z = _streaming_path()
    else:
        try:
            out_z = _batched_path()
        except MemoryError:
            out_z = _streaming_path()

    if not free_terminals:
        return out_z

    if space == "z":
        return out_z

    # space == "k": apply inverse-FFT-style transform along each terminal
    # axis with the n^d rescaling.  Matches graph_sample's convention.
    #
    # Symmetry with the single-k path: ``out_z`` here is the result of
    # an all-real bucket elimination (no momentum-weight tensor was
    # attached — the terminal was kept as a free axis), so the entire
    # contraction ran in float64.  The only complex step is this one
    # cheap final FFT, O(n_term·d · n^d log n), which the caller
    # real-projects.  The single-k path achieves the same "real
    # bucket-elim, defer the Fourier kernel to the very end" structure
    # by folding a real cos weight in and skipping the FFT entirely.
    if d == 1:
        out_k = np.fft.ifftn(out_z, axes=tuple(range(n_term))) * (n ** n_term)
    else:
        out_full = out_z.reshape((n,) * (n_term * d))
        out_k    = np.fft.ifftn(out_full, axes=tuple(range(n_term * d))) \
                    * (n ** (n_term * d))
        out_k    = out_k.reshape((n ** d,) * n_term)
    return out_k


def graph_zeta_general_at_zero(
    edges_flat:    np.ndarray,
    nu_vec:        np.ndarray,
    A:             "np.ndarray | str",
    n_points:      int,
    *,
    pinned_vertex: "int | None" = None,
    kernels:       "Sequence | None" = None,
) -> float:
    r"""Compute :math:`\zeta_g(\boldsymbol{0}, \dots, \boldsymbol{0})`
    via Fourier-grid bucket elimination on a torus of ``n_points^d``
    sites per vertex axis.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Treewidth-agnostic — works for arbitrary multigraph topology
    including K_4 minors and beyond.  Dense bound
    :math:`\mathcal{O}(V \cdot n_\text{points}^{(\tau+1)\,d})`; the
    peel plus the planner realise :math:`n^{\tau d}` on the dominant
    step (see :func:`graph_zeta_general`, Notes).

    This is a thin wrapper over :func:`graph_zeta_general` with no
    free terminals; see that function for the general case where
    one or more terminal vertex positions are kept open and the
    output is a tensor over terminal positions or momenta.

    Parameters
    ----------
    edges_flat : ndarray, shape (E, 2), int
        Already-expanded multigraph.  Each row ``(u, v)`` is a single
        edge.  Vertices are 0-indexed.  Multiple edges between the
        same pair of vertices are encoded as multiple rows.
    nu_vec : ndarray, shape (E,), float
        Per-edge exponents.
    A : ndarray, shape (d, d), float, or str
        Lattice basis or lattice name (``d ≥ 1``; see
        :func:`graph_zeta_general` for the practical ``n_points``
        ceiling at d ≥ 3).
    n_points : int
        Discretisation points per dimension on the torus.
    pinned_vertex : int or None, optional
        Vertex pinned at the origin (the "source" :math:`s`).  All
        other vertex positions are summed over the torus.  The choice
        of pin is immaterial — translation invariance of the kernel
        guarantees the same scalar regardless of which vertex is
        pinned.  Default ``None``: the planner chooses the pin (which
        moves nothing but round-off, and which end of the bag-width
        interval the contraction pays); pass a vertex to fix it.
    kernels : sequence of Interaction, optional
        General per-edge kernels, forwarded to
        :func:`graph_zeta_general` (``nu_vec`` then carries the tails).

    Returns
    -------
    float
        Graph zeta at all-zero external momenta.  Real: the sum equals
        its own conjugate under the lattice inversion (see
        :mod:`gzl._real`), and an imaginary part above round-off
        raises rather than being dropped.
    """
    out = graph_zeta_general(
        edges_flat, nu_vec, A, n_points,
        source=pinned_vertex, terminals=(), space="z",
        kernels=kernels,
    )
    return _as_real(np.asarray(out).item(),
                    where="graph_zeta_general_at_zero")


# ---------------------------------------------------------------------------
# The torus truncation, for the shared factor-supply layer
# ---------------------------------------------------------------------------

class TorusTruncation(_contract.Truncation):
    r"""Cyclic truncation: differences mod ``n``, circulant kernels.

    Defined *here*, not in :mod:`gzl._contract`, so that every
    method body resolves this module's tunables and helpers through
    this module's namespace at call time — ``tensor_network._USE_CONV``
    and the peel functions are monkeypatched by the test suite, and a
    class defined elsewhere (or capturing those names at construction)
    would silently detach the patches.  Each body is the engine's
    current behaviour moved verbatim, not a re-derivation.
    """

    def __init__(self, n: int, d: int, A: np.ndarray, compact_pairs=()):
        self.n = int(n)
        self.d = int(d)
        self.A = np.asarray(A, dtype=float)
        # Sorted vertex pairs whose ORIGINAL bundle is a compact table
        # (nu = inf) and must peel by ``_peel_sparse``, never by FFT.
        # Only an original bundle carries a generator, so a pair here
        # names exactly one lazy factor; every intermediate is dense.
        self.compact_pairs = frozenset(
            (min(int(a), int(b)), max(int(a), int(b)))
            for a, b in compact_pairs)

    def axis_size(self, v) -> int:
        # Uniform: every torus vertex ranges over the same n^d labels.
        # Equal to gen.size for a circulant generator — but that is a
        # circulant-only identity, which is why this is an explicit
        # method rather than a derivation (see Truncation.axis_size).
        return self.n ** self.d

    def coords(self, v) -> np.ndarray:
        z = _balanced_z_axis(self.n)
        grids = np.meshgrid(*[z] * self.d, indexing="ij")
        return np.stack(grids, axis=-1).reshape(-1, self.d)

    def generator(self, nu) -> np.ndarray:
        return _edge_kernel_torus(float(nu), self.A, self.n)

    def table(self, gen, a, b) -> np.ndarray:
        # Module-scope call: the patch point stays live.
        return _edge_difference_table(gen, self.n)

    def pin_slot(self, gen, v, pin_row: bool) -> np.ndarray:
        return _pin_from_generator(gen, pin_row)

    def reverse(self, gen) -> np.ndarray:
        return _reverse_generator(gen)

    def restrict_to(self, gen, coarse) -> np.ndarray:
        r"""Gather ``gen`` onto a coarser torus of the same lattice.

        Both tori index their generators by the balanced label axis
        (:func:`_balanced_z_axis`), whose defining property is
        ``z[j] == j (mod n)``.  So the coarse index ``j`` carries label
        ``z_c[j]``, which sits at fine index ``z_c[j] % n_fine`` — one
        per-axis gather, no arithmetic.  At ``n_coarse == n_fine`` the
        index vector is ``arange(n)`` and the gather is the identity
        bit for bit, which is the property the split-resolution gate
        rests on.

        The gather is legal because every coarse label is a fine label:
        the coarse axis spans ``[-((n_c+1)//2)+1, n_c//2]``, inside the
        fine axis's range whenever ``n_coarse <= n_fine``.

        NOTE this does **not** commute with :meth:`reverse` at even
        ``n_coarse`` on a sheared cell: the balanced axis holds
        ``n_c/2`` without ``-n_c/2``, so reversing first lands on the
        fine label ``-n_c/2`` while restricting first lands on
        ``+n_c/2``, and those are genuinely different kernel values
        there (the same asymmetry :func:`gzl.hybrid._oriented`
        exists to handle).  Callers must therefore restrict at one
        fixed point in the pipeline, with the kernels already in their
        canonical orientation.
        """
        if int(coarse.d) != self.d:
            raise ValueError(
                f"restrict_to: dimension mismatch (fine d={self.d}, "
                f"coarse d={int(coarse.d)})"
            )
        n_c = int(coarse.n)
        if n_c > self.n:
            raise ValueError(
                f"restrict_to: cannot restrict upward (fine n={self.n}, "
                f"coarse n={n_c})"
            )
        if not np.array_equal(np.asarray(coarse.A, dtype=float), self.A):
            raise ValueError(
                "restrict_to: the two truncations must share the lattice "
                "matrix A; the labels are shared integers and the physical "
                "positions are A z, so a differing A would reinterpret "
                "every entry."
            )
        gen = np.asarray(gen)
        if gen.shape != (self.n,) * self.d:
            # Without this the gather silently succeeds on a generator
            # belonging to a different grid — e.g. fine and coarse passed
            # the wrong way round — and returns plausible finite numbers.
            raise ValueError(
                f"restrict_to: generator has shape {gen.shape}, expected "
                f"{(self.n,) * self.d} for the fine truncation it is being "
                f"restricted FROM"
            )
        idx = _balanced_z_axis(n_c) % self.n
        return gen[np.ix_(*[idx] * self.d)]

    def embed_into(self, arr, fine) -> np.ndarray:
        r"""Scatter ``arr`` — an array on *this* truncation — onto the
        finer ``fine`` truncation of the same lattice, padding with zero.

        The exact mirror of :meth:`restrict_to`: same index vector
        ``_balanced_z_axis(n_coarse) % n_fine``, used to scatter instead
        of to gather.  No arithmetic, so at ``n_fine == n_coarse`` the
        index vector is ``arange(n)`` and this is the identity bit for
        bit — the property the split-resolution gate rests on, here in
        its upward direction.

        **What it means physically, and why it is not interpolation.**
        A core contracted on a coarse torus returns the terminal-position
        amplitude ``M`` on ``n_coarse`` labels; its Brillouin-zone
        transform is a trigonometric polynomial in ``k`` with exactly
        those coefficients.  Padding with zero and transforming on the
        fine grid *evaluates that same polynomial* at the fine nodes —
        it agrees with the coarse transform on every shared node and
        interpolates between them in the band-limited sense.  So the
        operation adds no error beyond the coarse core's own truncation,
        which the caller has already chosen to accept.

        MEASURED, against hybrid's independent single-k branch (which
        evaluates the same polynomial at an arbitrary ``k`` directly from
        the coarse array): max relative disagreement over the fine grid
        was 5.3e-16 (d=1, 8→32), 3.8e-16 (d=1, 10→40) and 6.3e-16
        (d=2, 6→24), i.e. round-off.

        Zero is the right pad value and not merely a convenient one: the
        labels outside the coarse torus carry no amplitude in the
        approximant being evaluated, so any other filling would assert
        information the coarse contraction never computed.
        """
        if int(fine.d) != self.d:
            raise ValueError(
                f"embed_into: dimension mismatch (coarse d={self.d}, "
                f"fine d={int(fine.d)})"
            )
        n_f = int(fine.n)
        if n_f < self.n:
            raise ValueError(
                f"embed_into: cannot embed downward (coarse n={self.n}, "
                f"fine n={n_f}); use restrict_to for that direction"
            )
        if not np.array_equal(np.asarray(fine.A, dtype=float), self.A):
            raise ValueError(
                "embed_into: the two truncations must share the lattice "
                "matrix A; the labels are shared integers and the physical "
                "positions are A z, so a differing A would reinterpret "
                "every entry."
            )
        arr = np.asarray(arr)
        if arr.shape != (self.n,) * self.d:
            # Without this the scatter silently succeeds on an array
            # belonging to a different grid — e.g. coarse and fine passed
            # the wrong way round — and returns plausible finite numbers.
            raise ValueError(
                f"embed_into: array has shape {arr.shape}, expected "
                f"{(self.n,) * self.d} for the coarse truncation it is "
                f"being embedded FROM"
            )
        idx = _balanced_z_axis(self.n) % n_f
        out = np.zeros((n_f,) * self.d, dtype=arr.dtype)
        out[np.ix_(*[idx] * self.d)] = arr
        return out

    def compose(self, g1, g2):
        # Cyclic convolution: on the torus the free vertex's sum IS the
        # full index range, so this is EXACT for the truncated problem —
        # unlike the box, whose linear composition is a surrogate.
        #
        # REAL transform.  Both generators are real (every edge kernel
        # this module builds is float64) and so is the convolution, so
        # the complex ``fftn`` was carrying a zero imaginary part
        # through three transforms and discarding it at the end.
        # ``rfftn``/``irfftn`` store only the non-redundant half of the
        # spectrum: ~2x less time and ~2x less memory in the transform
        # buffers, which is where the SP collapse's cost lives at large
        # ``sp_n_points``.
        #
        # NOT bit-identical to the complex path — a different transform
        # decomposition reassociates the sum — which is why it is gated
        # by ``_USE_RFFT`` rather than swapped in silently, and why the
        # goldens are compared against a freshly measured bound rather
        # than assumed unchanged.  ``s=`` is mandatory: ``irfftn``
        # cannot infer an odd final axis length from the half-spectrum
        # and would silently return an even-length axis.
        g1 = np.asarray(g1)
        g2 = np.asarray(g2)
        if _USE_RFFT and not (np.iscomplexobj(g1) or np.iscomplexobj(g2)):
            # ``axes`` must be given alongside ``s``: numpy 2.0
            # deprecated the ``s``-without-``axes`` form, and the
            # replacement semantics differ (``s[i]`` would come to mean
            # the size along ``axes[i]``).  Spelling both out keeps this
            # correct on both sides of that change.
            axes = tuple(range(g1.ndim))
            return np.fft.irfftn(np.fft.rfftn(g1, axes=axes)
                                 * np.fft.rfftn(g2, axes=axes),
                                 s=g1.shape, axes=axes)
        return np.fft.ifftn(np.fft.fftn(g1) * np.fft.fftn(g2)).real

    def trace(self, gen) -> float:
        # Zero displacement sits at index 0 on the balanced label axis.
        return float(np.asarray(gen)[(0,) * self.d])

    def weight(self, v):
        return None                      # uniform measure on the torus

    def empty_bucket_scalar(self, v) -> float:
        # A vertex with no incident factor contributes one term per
        # position it ranges over — n^d on the torus.  That is the
        # correct value FOR AN ISOLATED VERTEX, and since the
        # label-convention unification (gzl/_labels.py) no
        # public entry point can produce one: the vertex set is the
        # edge support, so every vertex reaching the elimination has at
        # least one incident kernel.  Kept total (not an assertion)
        # because the shared skeleton's contract requires it, and
        # because the value is the mathematically right one should an
        # internal caller ever construct such a bucket deliberately.
        return float(self.n ** self.d)

    def peel_constant(self, gen, u) -> np.ndarray:
        # Circulant row sums are constant — each row of the table is a
        # permutation of the generator — so the bucket-of-one peel is
        # exactly gen.sum() at every position.  This shortcut is
        # torus-only; the box's row sums genuinely vary (20-56%
        # measured spread), which is why this is a method and not a
        # shared code path with a capability flag.
        #
        # DEFINITIONAL, NOT VERBATIM — the one exception to the class
        # docstring's moved-verbatim rule.  The engine's bucket-of-one
        # branch sums the ORIENTED generator (g_use, reversed when the
        # eliminated vertex is the row axis), and np.sum over the
        # permuted element order can differ in the last ulp on a skew
        # cell at even n.  No engine path calls this method; wiring it
        # into the loop is a gated substitution, not a free move.
        return np.full(self.n ** self.d, gen.sum(), dtype=gen.dtype)

    # ---- the shared bucket skeleton's callbacks ----------------------
    #
    # Bodies below are the engine's shipped _eliminate_vertex code,
    # moved verbatim.  They read _USE_CONV, _find_conv_peel and
    # _peel_convolution through THIS module's namespace at call time,
    # so every monkeypatch on tensor_network.* keeps working.

    def scope_of(self, f):
        return f[0]

    def scalar_factor(self, value):
        # The empty-bucket contribution.  NOTE this is a deliberate
        # semantic definition on a branch unreachable from the
        # frontend: the pre-extraction loop SKIPPED a vertex with no
        # incident factor (contributing 1), which is wrong by n^d per
        # isolated vertex and disagreed with the box engine on the
        # same input.  Both engines now count the vertex's positions.
        return ([], np.asarray(float(value)), None)

    def pre_step(self, bucket, v, out_axes):
        # Preserved shipped behaviour: an oversized bucket errors even
        # when it could have peeled (the check preceded the peel gate).
        if len(out_axes) > len(string.ascii_lowercase):
            raise NotImplementedError(
                f"tensor_network: bucket has > 26 axes, einsum letters "
                f"exhausted (vertex {v})."
            )

    def step_dtype(self, bucket):
        # Inherit dtype from the inputs.  Both the k=0 and the
        # single-k (real cos-weight) paths are all-float64, so this
        # stays real; the promotion to complex only triggers if a
        # caller supplies complex kernels directly.
        return np.result_type(
            *[(T if T is not None else g).dtype for _, T, g in bucket]
        )

    def peel_gate(self, bucket, v, union, dtype):
        # The floating guard keeps caller-supplied complex kernels on
        # the dense branch (rfftn would discard their imaginary part).
        return _USE_CONV and np.issubdtype(dtype, np.floating)

    def find_partner(self, bucket, v):
        return _find_conv_peel(bucket, v)

    def peel_step(self, bucket, token, v, out_axes, dtype):
        peel_idx, u = token
        ax = bucket[peel_idx][0]
        if (self.compact_pairs
                and (min(ax), max(ax)) in self.compact_pairs):
            # A compact bundle: the exact roll-sum, never the FFT.
            return _peel_sparse(bucket, peel_idx, v, u,
                                list(out_axes), dtype)
        return _peel_convolution(bucket, peel_idx, v, u,
                                 list(out_axes), dtype)

    def dense_strategy(self, bag_volume: int) -> str:
        # The torus dense step IS the axis-sliced einsum loop below;
        # there is no broadcast variant and the torus is deliberately
        # never flipped (its values are the bit-identity oracle for
        # two other subsystems).  Pinned, not "auto".
        return "sliced"

    def dense_step(self, bucket, v, out_axes, dtype):
        r"""Axis-sliced einsum contraction — the engine's dense branch.

        Memory profile: a naive all-at-once einsum needs an
        intermediate of size n^(|bag| d); slicing along the ``v`` axis
        (np.take per input, einsum per slice, accumulate) drops peak
        memory to n^((|bag|-1) d) plus one slice, with the identical
        floating-point result — same multiply-adds, one big BLAS call
        traded for n^d smaller ones.
        """
        out_axes_sorted = list(out_axes)
        relevant = bucket
        res_dtype = dtype

        # v-axis length, from the first factor exactly as shipped —
        # with the identity it rests on now ASSERTED, because deriving
        # an axis from gen.size is circulant-only and silently wrong
        # on any other truncation.
        first_ax, first_T, first_g = relevant[0]
        n_v = int(first_T.shape[first_ax.index(v)]) if first_T is not None             else int(first_g.size)
        assert n_v == self.axis_size(v), (
            f"axis derivation broke: factor says {n_v}, truncation "
            f"says {self.axis_size(v)} for vertex {v}"
        )

        if not out_axes_sorted:
            # All input tensors are 1-axis (just [v]):
            #     result = sum_v  prod_k T_k[v]
            # Hadamard-multiply the 1-D tensors elementwise, then sum.
            arrs = [_factor_array(T, g, None) for _, T, g in relevant]
            prod = np.ones_like(arrs[0])
            for T in arrs:
                prod = prod * np.asarray(T)
            return (out_axes_sorted,
                    np.asarray(prod.sum(), dtype=res_dtype), None)

        letter = {a: string.ascii_lowercase[i]
                  for i, a in enumerate(out_axes_sorted)}

        # Sliced einsum spec — drop v from each input axis list.
        in_specs = [''.join(letter[a] for a in ax if a != v)
                    for ax, _, _ in relevant]
        out_spec = ''.join(letter[a] for a in out_axes_sorted)
        eq       = ','.join(in_specs) + '->' + out_spec

        # Pre-cache, for each relevant tensor, the position of the v
        # axis so the inner loop avoids repeated .index() calls.
        v_positions = [ax.index(v) for ax, _, _ in relevant]
        arrays_full = [_factor_array(T, g, None) for _, T, g in relevant]

        # Pre-compute the einsum optimisation path once on a
        # representative slice; numpy reuses it for all slices.
        sample_sliced = [
            np.take(T, 0, axis=v_positions[k])
            for k, T in enumerate(arrays_full)
        ]
        path, _ = np.einsum_path(eq, *sample_sliced, optimize="optimal")

        # ...and then EXECUTE that path ourselves.
        #
        # ``np.einsum(eq, *ops, optimize=path)`` does NOT reuse the path:
        # whenever ``optimize`` is not False, einsum re-enters
        # ``einsum_path`` to rebuild the contraction list, every call.
        # This loop runs once per slice of the v axis, so on a d = 2
        # order-11 full-BZ pass that was 579,842 calls to ``einsum_path``
        # for 12.6 s — deriving the same two-step plan a hundred thousand
        # times over.
        #
        # ``einsum_call=True`` hands back the pairwise contraction list
        # that einsum would have built.  Each step is then a TWO-operand
        # einsum, whose optimal path is trivial, so it runs with
        # ``optimize=False`` and reaches BLAS directly with no path work
        # at all.  Same operations in the same order — this is numpy's
        # own execution loop, hoisted out of the slice iteration.
        # ``einsum_call=True`` is SEMI-PRIVATE and its tuple arity is not
        # stable across numpy versions — it returned 5-tuples locally and
        # 3-tuples on CI, which crashed with "not enough values to
        # unpack".  So index defensively rather than destructure, and
        # then PROVE the hoisted execution reproduces plain einsum on the
        # sample slice before trusting it for the other n_v - 1.  A
        # fast path that cannot be checked is not worth having.
        contraction_list = None
        try:
            _, raw = np.einsum_path(
                eq, *sample_sliced, optimize="optimal", einsum_call=True,
            )
            steps = [(tuple(step[0]), str(step[2])) for step in raw]

            def _run(ops, steps=steps):
                ops = list(ops)
                for inds, es in steps:
                    tmp = [ops.pop(x) for x in inds]
                    ops.append(np.einsum(es, *tmp, optimize=False))
                return ops[0]

            check = _run(sample_sliced)
            want = np.einsum(eq, *sample_sliced, optimize=path)
            if (np.shape(check) == np.shape(want)
                    and np.array_equal(np.asarray(check), np.asarray(want))):
                contraction_list = steps
        except Exception:
            contraction_list = None       # fall back to plain einsum below

        def _contract(ops):
            if contraction_list is None:
                return np.einsum(eq, *ops, optimize=path)
            ops = list(ops)
            for inds, es in contraction_list:
                tmp = [ops.pop(x) for x in inds]
                ops.append(np.einsum(es, *tmp, optimize=False))
            return ops[0]

        # Output shape: dimension k is the size of the axis labelled by
        # out_axes_sorted[k] in any input tensor that has it.
        axis_size: dict = {}
        for ax, T in zip((a for a, _, _ in relevant), arrays_full):
            for j, a in enumerate(ax):
                axis_size[a] = T.shape[j]
        out_shape = tuple(axis_size[a] for a in out_axes_sorted)

        result = np.zeros(out_shape, dtype=res_dtype)
        sliced: list = [None] * len(relevant)
        for i in range(n_v):
            for k, T in enumerate(arrays_full):
                sliced[k] = np.take(T, i, axis=v_positions[k])
            result += _contract(sliced)

        return (out_axes_sorted, result, None)
