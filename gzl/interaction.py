# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
r"""General interaction kernels for graph zeta functions.

An :class:`Interaction` is the per-edge kernel

.. math::

    K(\boldsymbol{x}) = a(\boldsymbol{x})
        + \sum_j b_j K_{\nu_j}(\boldsymbol{x}),
    \qquad K_\nu(\boldsymbol{x}) = |\boldsymbol{x}|^{-\nu}\ (\boldsymbol{x} \ne 0),
    \quad K_\nu(0) = 0,

with ``a`` a real, EVEN, compactly supported function on the Bravais
lattice ``Λ = A·ℤ^d`` (columns of ``A`` are the primitive vectors), stored
as a finite table over integer lattice LABELS ``m`` (``x = A m``).  The
power-law part is the regularised kernel the rest of the library uses;
the compact part is new.

The contract is two-tier.  The FRONT ENDS -- :func:`evaluate_graph`,
:func:`compute_series_coefficients`, :func:`evaluate_corpus`,
:func:`zeta_circle`, and a NetworkX edge attribute -- accept an
:class:`Interaction` in the ``nu`` slot itself (see :func:`coerce_nu`).
The BUILDERS and ENGINES -- ``graph_from_edges``, ``graph_zeta_general``,
``hybrid_zeta``, ``slab_zeta``, ``direct_sum_*`` -- keep ``nu_vec``
carrying the float tail exponents every planner, cut, peel and refusal
consumer reads, and take kernels only through their keyword-only
``kernels=`` channel; each refuses an :class:`Interaction` in ``nu_vec``
with a message naming that channel.  A plain ``Interaction(b=[1],
nu=[ν])`` is demoted to the float ``ν`` at the front-end boundary, so the
legacy power-law path is byte-identical.

**The origin is not special for the compact part.**  ``a(0)`` is allowed
and simply weights every configuration in which the two endpoints of the
edge coincide (the power-law part is zero there only because
``|0|^{-ν}`` has to be regularised).  A physical pair interaction has
``a(0) = 0``; a table or function that gives ``a(0) ≠ 0`` is honoured, not
silently zeroed.

**Compact parts must be even.**  The router discards edge orientation
(parallel edges are keyed on the unordered pair) and the finite-momentum
engines use a real ``cos`` weight, both of which assume ``K(-x) = K(x)``.
:meth:`Interaction.from_function` samples a function of the real-space
displacement VECTOR ``x`` (it may be anisotropic), checks ``K(-x) = K(x)``
at every sampled point to ``1e-12`` RELATIVE TO THE LARGEST ``|K|`` IN THE
SAMPLED TABLE (``|K(x) - K(-x)| <= 1e-12 · max_m |K(A m)|`` — a meV-scale
table gets a meV-scale guard, never an absolute one) and stores the
symmetrised value; :meth:`Interaction.from_table` requires an exactly even
table.

**Products are lazy.**  Parallel edges multiply pointwise (the Hadamard
merge), so ``I1 * I2`` and ``I ** m`` return a :class:`_KernelProduct`
that samples as the pointwise product of its factors — exact on any grid
— and expands the analytic power-law tail only where a closed form needs
it (:meth:`_KernelProduct.lattice_sum`).  A symbolic product would need
the lattice for its cross terms and loses digits on reconstruction.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from gzl._errors import GraphZetaError
from gzl._lattices import _resolve_lattice, lattice_matrix
from gzl.core import _epstein_zeta_k

__all__ = [
    "Interaction",
    "InteractionSupportError",
    "coerce_nu",
    "is_interaction",
    "KEY_DECIMALS",
]

#: Quantisation of exponents in :meth:`Interaction.key`.  Must equal
#: ``gzl.frontend._NU_KEY_DECIMALS`` (a test pins the two).
KEY_DECIMALS = 12

#: Relative tolerance with which a lattice shell is identified by its
#: distance -- the rule ``tensor_network._edge_kernel_torus`` uses for the
#: nearest-neighbour indicator at ``nu = inf``.
_SHELL_RTOL = 1e-9

#: The two power forms the engines use.  They differ bitwise on ~30 % of
#: entries, and the frozen reference harness pins each engine's form, so
#: the form is the ENGINE's choice and is passed into :meth:`sample`.
POWER_TORUS = "torus"   # 1.0 / (dist ** nu)   -- tensor_network.py
POWER_BOX = "box"       # dist ** (-nu)         -- direct_sum.py


class InteractionSupportError(GraphZetaError):
    """The compact part of an interaction does not fit the window an
    engine evaluates on (a torus window, a Richardson rung, a coarse core
    grid, or a real-space box).  Raised BEFORE a clipped or aliased
    kernel can produce a plausible wrong number.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _power(dist: np.ndarray, nu: float, form) -> np.ndarray:
    """``|x|^{-nu}`` on the nonzero entries of ``dist`` in the engine's form."""
    if callable(form):
        return form(dist, nu)
    if form == POWER_TORUS:
        return 1.0 / (dist ** float(nu))
    if form == POWER_BOX:
        return dist ** (-float(nu))
    raise ValueError(f"unknown power form {form!r}; use 'torus', 'box' or a callable")


def _lattice_fingerprint(A) -> bytes:
    A = np.asarray(_resolve_lattice(A), dtype=float)
    # ``+ 0.0`` turns a rounded ``-0.0`` (an entry such as ``-1e-17`` from a
    # rotation) into ``+0.0``: different bytes, the same cell.
    return np.ascontiguousarray(np.round(A, KEY_DECIMALS) + 0.0).tobytes()


def _as_matrix(A) -> np.ndarray:
    # The one choke point for every public ``A``: the Interaction and
    # _KernelProduct methods, which forward the raw value onwards.
    return lattice_matrix(A, "Interaction")


def _labels_in_ball(A: np.ndarray, radius: float):
    """All integer labels ``m`` with ``|A m| <= radius (1 + 1e-12)``.

    Returns ``(labels (N, d) int, dist (N,) float)`` in lexicographic
    label order; ``m = 0`` is included.
    """
    d = A.shape[0]
    radius = float(radius)
    if not (radius >= 0.0) or not math.isfinite(radius):
        raise ValueError(f"radius must be a finite non-negative number; got {radius!r}")
    smin = float(np.linalg.svd(A, compute_uv=False).min())
    R = int(math.ceil(radius / smin)) + 1 if radius > 0 else 0
    axis = np.arange(-R, R + 1)
    grids = np.meshgrid(*([axis] * d), indexing="ij")
    labels = np.stack(grids, axis=-1).reshape(-1, d)
    dist = np.linalg.norm(labels @ A.T, axis=1)
    keep = dist <= radius * (1.0 + 1e-12)
    labels, dist = labels[keep], dist[keep]
    order = np.lexsort(labels.T[::-1])
    return labels[order], dist[order]


#: Tolerance of the evenness check in :meth:`Interaction.from_function` /
#: :meth:`Interaction.from_total`, RELATIVE TO THE SAMPLED TABLE'S LARGEST
#: ``|K|``: ``|K(x) - K(-x)| <= _EVEN_RTOL * max(max_m |K(A m)|, tiny)``.
#: The stored value is the symmetrised one, so the table is then exactly
#: even.
_EVEN_RTOL = 1e-12


def _refuse_complex(arr: np.ndarray, where: str) -> np.ndarray:
    """Refuse a complex-valued result of ``K`` BEFORE any float cast --
    ``np.asarray(z, dtype=float)`` would drop the imaginary part with
    nothing but a ComplexWarning."""
    if np.iscomplexobj(arr):
        raise ValueError(
            f"K(x) must be real-valued; it returned complex values {where}.  "
            f"An Interaction is a real kernel: pass the real part explicitly "
            f"if the imaginary part is known to vanish."
        )
    return arr


def _agrees_pointwise(got, K: Callable, x: np.ndarray) -> bool:
    """Does the batch result at one point match calling ``K`` on that point?

    Used to confirm a batch call really was a batch call.  Any failure to
    produce one finite-or-NaN scalar counts as disagreement, so the caller
    falls back to the per-point loop, which raises the precise error.
    """
    try:
        v = np.asarray(K(x), dtype=float).reshape(-1)
    except Exception:
        return False
    if v.size != 1:
        return False
    a, b = float(got), float(v[0])
    if np.isnan(a) and np.isnan(b):
        return True
    return a == b or bool(np.isclose(a, b, rtol=1e-12, atol=0.0))


def _call_vector(K: Callable, pts: np.ndarray) -> np.ndarray:
    """Evaluate ``K`` on physical points of shape ``(N, d)`` (the
    ``_vint_array`` convention), tolerating a callable that only accepts
    one point at a time: ANY exception from the batch call (a TypeError
    from ``float(array)``, a KeyError from a per-point table lookup, an
    AttributeError, ...) falls back to the per-point loop, which re-raises
    whatever is genuinely wrong with a single-point context.  A complex
    result is refused on either path."""
    pts = np.asarray(pts, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        try:
            batch = np.asarray(K(pts))
        except Exception:
            batch = None
        if batch is not None:
            _refuse_complex(batch, "on the (N, d) point batch")
            try:
                out = np.asarray(batch, dtype=float)
            except Exception:
                out = None
            if out is not None and out.shape == pts.shape[:-1]:
                # A matching SHAPE is not proof that ``K`` understood the
                # batch.  A per-point callable handed the whole ``(N, d)``
                # array commonly returns something of length ``d`` -- it
                # indexes ``x[0]``, which on a batch is the first ROW --
                # and when ``N == d`` that shape matches, so a WRONG table
                # would be stored with no exception and no warning.  Spot
                # check the ends against the per-point contract, which is
                # what the table is finally meant to mean, before trusting
                # the batch.  Two extra scalar calls per table build.
                _probe = (0,) if len(pts) < 2 else (0, len(pts) - 1)
                if all(_agrees_pointwise(out[i], K, pts[i]) for i in _probe):
                    return out
        out = []
        for x in pts:
            v = _refuse_complex(np.asarray(K(x)), f"at the point x = {x.tolist()}")
            v = np.asarray(v, dtype=float).reshape(-1)
            if v.size != 1:
                raise ValueError(
                    "K(x) must return one value per point: on a single point "
                    f"of shape {x.shape} it returned shape {v.shape}"
                )
            out.append(float(v[0]))
        return np.asarray(out, dtype=float)


def _symmetrised(labels: np.ndarray, vals: np.ndarray, what: str) -> np.ndarray:
    """Check ``K(-x) = K(x)`` on a mirror-closed label set to
    ``_EVEN_RTOL`` relative to the table's largest ``|K|`` OFF THE ORIGIN
    (the on-site value ``K(0)`` is trivially even and is not a
    pair-coupling scale: a soft core with ``K(0) = 1e6`` must not license
    an odd part of 1e-7 on the couplings) and return the symmetrised
    values (exactly even by construction)."""
    vals = np.asarray(vals, dtype=float)
    labels = np.asarray(labels)
    off = np.any(labels != 0, axis=1) if labels.ndim == 2 else np.ones(len(vals), bool)
    ref = vals[off] if np.any(off) else vals
    scale = max(float(np.max(np.abs(ref))) if ref.size else 0.0,
                float(np.finfo(float).tiny))
    tol = _EVEN_RTOL * scale
    index = {tuple(int(x) for x in m): i for i, m in enumerate(labels)}
    out = np.empty_like(vals)
    for m, i in index.items():
        j = index[tuple(-x for x in m)]
        a, b = float(vals[i]), float(vals[j])
        if abs(a - b) > tol:
            raise ValueError(
                f"{what} must be even, K(-x) = K(x): at lattice label m = {m} "
                f"K = {a!r} but at -m K = {b!r} (|difference| = {abs(a - b):.3e}, "
                f"tolerance {_EVEN_RTOL:g} x the table's largest off-origin |K| = "
                f"{scale:.6g}).  The router discards edge orientation and the "
                f"finite-k engines use a real cos weight, both of which assume "
                f"an even kernel."
            )
        out[i] = 0.5 * (a + b) if a != b else a
    return out


def _canonical_table(table) -> tuple:
    """Normalise a table to a sorted tuple of ``((m_1..m_d), value)`` with
    python ints / floats, exact zeros dropped, and check exact evenness."""
    if table is None:
        return ()
    items = table.items() if isinstance(table, Mapping) else table
    out: dict = {}
    dim = None
    for m, v in items:
        raw = np.asarray(m).ravel()
        if raw.size == 0:
            raise ValueError("a compact-table label must have at least one component")
        if not np.all(raw == np.round(raw)):
            raise ValueError(f"compact-table label {tuple(raw.tolist())!r} is not integer")
        m = tuple(int(x) for x in raw)
        if dim is None:
            dim = len(m)
        elif len(m) != dim:
            raise ValueError(f"compact-table labels mix dimensions {dim} and {len(m)}")
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"compact-table value at {m!r} is not finite: {v!r}")
        if m in out:
            raise ValueError(f"compact-table label {m!r} given twice")
        out[m] = v
    out = {m: v for m, v in out.items() if v != 0.0}
    for m, v in out.items():
        mirror = tuple(-x for x in m)
        if mirror not in out or out[mirror] != v:
            raise ValueError(
                f"the compact part must be even: a({m}) = {v!r} but "
                f"a({mirror}) = {out.get(mirror, 0.0)!r}.  The router discards "
                f"edge orientation and the finite-k engines use a real cos "
                f"weight, both of which assume K(-x) = K(x)."
            )
    return tuple(sorted(out.items()))


def _canonical_terms(b, nu) -> tuple:
    """Power-law terms as a tuple of ``(b, nu)`` python floats, sorted by
    ``nu``, exact-duplicate exponents merged, zero weights dropped.  An
    EMPTY ``b`` with a non-empty ``nu`` means unit weights
    (``Interaction(nu=[3])`` is the plain power law ``|x|^{-3}``); a
    non-empty ``b`` must match ``nu`` in length."""
    b = [] if b is None else [float(x) for x in np.asarray(b, dtype=float).ravel()]
    nu = [] if nu is None else [float(x) for x in np.asarray(nu, dtype=float).ravel()]
    if not b and nu:
        b = [1.0] * len(nu)
    if len(b) != len(nu):
        raise ValueError(f"b and nu must have the same length; got {len(b)} and {len(nu)}")
    merged: dict = {}
    for bj, nj in zip(b, nu):
        if not math.isfinite(bj):
            raise ValueError(f"power-law weight b = {bj!r} is not finite")
        if not math.isfinite(nj):
            raise ValueError(
                f"power-law exponent nu = {nj!r} is not finite.  The "
                f"nearest-neighbour indicator (the nu -> inf limit) is a compact "
                f"table: use Interaction.nearest_neighbour(A) or pass the float "
                f"np.inf on its own."
            )
        if nj <= 0.0:
            raise ValueError(f"power-law exponent nu = {nj!r} must be positive")
        merged[nj] = merged.get(nj, 0.0) + bj
    return tuple(sorted(((bj, nj) for nj, bj in merged.items() if bj != 0.0),
                        key=lambda t: t[1]))


def _chebyshev_radius(compact: tuple) -> int:
    return max((max(abs(x) for x in m) for m, _ in compact), default=0)


def _nearest_shells(A: np.ndarray, r: float, dist: np.ndarray) -> tuple:
    """The shell distances just below and just above ``r`` as ``%g``
    strings (``'none'`` below the first shell), for the
    :meth:`Interaction.from_shells` error.  ``dist`` is the ball already
    enumerated; it is widened (doubling) until a shell above ``r`` is in
    it -- a lattice vector of length in ``(r, r + |a_1|]`` always exists."""
    r = float(r)
    R = r
    dd = np.asarray(dist, dtype=float)
    for _ in range(64):
        above = dd[dd > r * (1.0 + _SHELL_RTOL)]
        if above.size:
            break
        R *= 2.0
        _, dd = _labels_in_ball(A, R)
    nz = dd[dd > 0.0]
    below = nz[nz < r * (1.0 - _SHELL_RTOL)]
    return ((f"{below.max():.10g}" if below.size else "none"),
            (f"{above.min():.10g}" if above.size else "none"))


def _scatter_table(compact: tuple, flat: np.ndarray, radius: int) -> np.ndarray:
    """``a(m)`` from a canonical table on a flat ``(N, d)`` label array,
    guarding the window: every label of the table must appear EXACTLY
    once in ``flat`` (absent: clipped; twice: aliased)."""
    out = np.zeros(flat.shape[0], dtype=np.float64)
    for m, v in compact:
        hits = np.flatnonzero(np.all(flat == np.asarray(m), axis=1))
        if hits.size != 1:
            raise InteractionSupportError(
                f"the compact part has support at label {m} (Chebyshev radius "
                f"{radius}), which is "
                f"{'absent from' if hits.size == 0 else 'duplicated in'} the "
                f"evaluation window of {flat.shape[0]} labels; use a larger grid "
                f"(torus: n_points >= 2 R + 1 so the balanced window holds both "
                f"+R and -R) or a smaller support."
            )
        out[hits[0]] += v
    return out


def _lattice_sum_from_terms(fourier, A, k_frac):
    """Evaluate ``Σ_j c_j Z_{ν_j}(k) + Σ_m v_m cos(2π m·k)`` at ``k`` in
    fractional BZ coordinates; see :meth:`Interaction.fourier_terms` for
    the data and :meth:`Interaction.lattice_sum` for the accepted shapes
    (``None`` / ``(d,)`` -> float, ``(N, d)`` -> ``(N,)``; in d = 1 also a
    scalar -> float and a 1-D array of length ``N != 1`` -> ``(N,)``)."""
    terms, labels, values = fourier
    A = _as_matrix(A)
    d = A.shape[0]
    if k_frac is None:
        kk, scalar = np.zeros((1, d)), True
    else:
        k = np.asarray(k_frac, dtype=float)
        if k.ndim == 0 and d == 1:
            kk, scalar = k.reshape(1, 1), True
        elif k.ndim == 1 and k.shape[0] == d:
            kk, scalar = k.reshape(1, d), True
        elif k.ndim == 1 and d == 1:
            kk, scalar = k.reshape(-1, 1), False       # d = 1: a batch of N momenta
        elif k.ndim == 2 and k.shape[1] == d:
            kk, scalar = k, False
        else:
            raise ValueError(
                f"k must be one fractional momentum of shape (d,) = ({d},) or a "
                f"batch of shape (N, d) = (N, {d})"
                + (" (in d = 1 also a scalar, or a 1-D array of length N != 1 "
                   "as a batch)" if d == 1 else "")
                + f"; got shape {k.shape}"
            )
    out = np.zeros(kk.shape[0], dtype=float)
    if len(values):
        out += np.cos(2.0 * np.pi * (kk @ labels.T.astype(float))) @ values
    if terms:
        Astar = np.linalg.inv(A.T)
        vol = abs(float(np.linalg.det(A)))
        for i in range(kk.shape[0]):
            # The same momentum with every component outside [0, 1)
            # shifted by the nearest integer, exactly as
            # frontend._reduce_to_cell does, since ``Astar @ k`` misses
            # the reciprocal lattice at an integer ``k`` on a
            # non-orthogonal cell.  Components in [0, 1) are unchanged.
            ki = np.array(kk[i], dtype=float)
            out_of_cell = (ki < 0.0) | (ki >= 1.0)
            ki[out_of_cell] -= np.rint(ki[out_of_cell])
            y = Astar @ ki
            for coef, nu in terms:
                out[i] += coef * float(np.real(_epstein_zeta_k(nu, A, y, vol)))
    return float(out[0]) if scalar else out


# ---------------------------------------------------------------------------
# The public object
# ---------------------------------------------------------------------------

@dataclass(frozen=True, init=False, repr=False)
class Interaction:
    r"""A general edge kernel ``K(x) = a(x) + Σ_j b_j |x|^{-ν_j}``.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Attributes
    ----------
    b, nu : tuple[float, ...]
        The power-law terms, sorted by exponent, exact-duplicate exponents
        merged, zero weights dropped.  ``nu_j`` must be finite and
        positive (``np.inf`` is the nearest-neighbour indicator, which is
        a compact TABLE here — see :meth:`nearest_neighbour`).
    compact : tuple[tuple[tuple[int, ...], float], ...]
        The compact part ``a`` as sorted ``((m_1, ..., m_d), a(m))`` pairs
        over integer lattice labels (``x = A m``); exact zeros are dropped;
        the origin may be present.  Even by construction/check.
    label : str | None
        A human-readable name (used in CSV output); see
        :attr:`default_label`.
    lattice : bytes | None
        Fingerprint of the lattice matrix the table was sampled on (set
        by the constructors that take ``A``; the constructor itself takes
        either the matrix — fingerprinted, as :meth:`from_table` does —
        or a lattice name, ``"chain"`` / ``"square"`` / ``"triangular"``
        / ``"cubic"``, or the fingerprint bytes); an engine evaluating on
        a different lattice raises.

    Construct with the classmethods :meth:`power_law`, :meth:`from_table`,
    :meth:`from_function`, :meth:`from_total`, :meth:`from_shells`,
    :meth:`nearest_neighbour`, or directly ``Interaction(b=..., nu=...,
    compact=...)``.

    Every ``A`` below — the constructors' and :meth:`sample`,
    :meth:`fourier_terms`, :meth:`lattice_sum` — is the lattice matrix,
    whose columns are the primitive vectors, or the name of a lattice:
    ``"chain"``, ``"square"``, ``"triangular"``, ``"cubic"``.
    """

    b: tuple
    nu: tuple
    compact: tuple
    label: "str | None"
    lattice: "bytes | None"

    __array_ufunc__ = None      # numpy scalars defer to __rmul__ / __radd__

    def __init__(self, b=(), nu=(), compact=None, *, label=None, lattice=None):
        terms = _canonical_terms(b, nu)
        table = _canonical_table(compact)
        if not terms and not table:
            raise ValueError(
                "an Interaction needs at least one power-law term or a "
                "non-zero compact table (the zero kernel is refused so a "
                "graph zeta cannot silently vanish)"
            )
        if label is not None and not isinstance(label, str):
            raise TypeError(f"label must be a str or None; got {type(label).__name__}")
        if isinstance(lattice, str):
            # A lattice NAME.  Resolved OUTSIDE the try below, so its
            # ValueError keeps the list of names instead of being
            # swallowed into the generic TypeError.
            lattice = _resolve_lattice(lattice)
        if lattice is not None and not isinstance(lattice, (bytes, bytearray)):
            # A lattice MATRIX, as from_table takes it: fingerprint it here.
            try:
                mat = np.asarray(lattice, dtype=float)
            except (TypeError, ValueError) as exc:
                raise TypeError("lattice must be a (d, d) lattice matrix, its "
                                "fingerprint bytes, or None") from exc
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                raise TypeError("lattice must be a (d, d) lattice matrix, its "
                                f"fingerprint bytes, or None; got shape {mat.shape}")
            lattice = _lattice_fingerprint(mat)
        object.__setattr__(self, "b", tuple(t[0] for t in terms))
        object.__setattr__(self, "nu", tuple(t[1] for t in terms))
        object.__setattr__(self, "compact", table)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "lattice", None if lattice is None else bytes(lattice))

    # -- constructors ------------------------------------------------------

    @classmethod
    def power_law(cls, nu: float, b: float = 1.0, *, label=None) -> "Interaction":
        """The regularised power law ``b |x|^{-nu}``.  With ``b = 1`` this is
        demoted to the float ``nu`` at every public boundary."""
        return cls(b=[b], nu=[nu], label=label)

    @classmethod
    def from_table(cls, table, *, b=(), nu=(), label=None, lattice=None) -> "Interaction":
        """An explicit compact part ``{(m_1, ..., m_d): a(m)}`` over integer
        lattice labels, plus optional power-law terms.  The table must be
        exactly even; the origin ``(0, ..., 0)`` may be present."""
        return cls(b=b, nu=nu, compact=table, label=label,
                   lattice=None if lattice is None else _lattice_fingerprint(lattice))

    @classmethod
    def from_function(cls, K: Callable, A, radius: float, *, b=(), nu=(),
                      label=None) -> "Interaction":
        r"""Sample a short-range function of the real-space displacement
        VECTOR, ``K(x)`` with ``x = A m``, onto the lattice: ``a(m) = K(A m)``
        for every label with ``|A m| <= radius`` (the origin INCLUDED —
        ``a(0) = K(0)`` weights coincident endpoints; return 0 there for a
        pair interaction), zero beyond.  ``K`` receives a float array of
        points of shape ``(N, d)`` and returns ``(N,)`` (the ``_vint_array``
        convention; a callable taking one point at a time is accepted, at
        a cost; a complex-valued ``K`` is refused).  ``K`` may be
        anisotropic but must be even, ``K(-x) = K(x)``: the check is
        ``|K(x) - K(-x)| <= 1e-12 · max_m |K(A m)|`` — relative to the
        LARGEST ``|K|`` in the sampled table, never absolute — and the
        stored value is the symmetrised one.  A table that is entirely zero
        (``K`` vanishes on every lattice point inside the radius) raises: it
        is not silently a plain power law.  The optional ``b, nu`` add
        power-law terms ON TOP of ``a``; use :meth:`from_total` when ``K``
        is the whole interaction.  For a radial potential write
        ``lambda x: f(np.linalg.norm(x, axis=-1))``."""
        A = _as_matrix(A)
        labels, dist = _labels_in_ball(A, radius)
        vals = _call_vector(K, labels @ A.T)
        if not np.all(np.isfinite(vals)):
            bad = labels[~np.isfinite(vals)][0]
            raise ValueError(
                f"K(x) is not finite at the lattice point m = {tuple(int(x) for x in bad)} "
                f"(|x| = {float(np.linalg.norm(A @ bad))!r}); K must be finite at "
                f"every lattice point inside the radius, x = 0 included (return 0 "
                f"there for a pair interaction)."
            )
        vals = _symmetrised(labels, vals, "the short-range function")
        if not np.any(vals != 0.0):
            raise ValueError(
                f"K vanishes on every lattice point inside radius {float(radius)!r} "
                f"({len(labels)} labels; the symmetrised table is entirely zero), "
                f"so there is no compact part to store.  Check the radius and the "
                f"function; a plain power law is Interaction.power_law(nu)."
            )
        table = {tuple(int(x) for x in m): float(v) for m, v in zip(labels, vals)}
        return cls(b=b, nu=nu, compact=table, label=label, lattice=_lattice_fingerprint(A))

    @classmethod
    def from_total(cls, K: Callable, A, radius: float, b, nu, *, label=None) -> "Interaction":
        r"""``K(x)`` (a function of the displacement vector, as in
        :meth:`from_function`) is the WHOLE interaction inside the radius
        and equals its power-law tail ``Σ_j b_j |x|^{-ν_j}`` beyond it.  The
        table stores the deviation ``K(A m) - Σ_j b_j |A m|^{-ν_j}``
        (``K(0)`` at the origin).  Note that a soft core with ``K ≈ 0`` at
        short range makes the deviation cancel the tail nearly exactly, and
        reconstructing ``K`` from the two parts then costs digits — the
        representation is exact, the arithmetic is not.  There is no
        threshold on the deviation: a ``K`` one ulp from its tail keeps a
        ``~1e-15`` table entry (not an exact zero, so not dropped) and the
        kernel keeps its compact part and compact routing; only an EXACT
        equality stores an exact zero, which is dropped."""
        A = _as_matrix(A)
        terms = _canonical_terms(b, nu)
        labels, dist = _labels_in_ball(A, radius)
        vals = _call_vector(K, labels @ A.T)
        if not np.all(np.isfinite(vals)):
            raise ValueError("K(x) must be finite at every lattice point inside the radius")
        vals = _symmetrised(labels, vals, "the total interaction")
        tail = np.zeros_like(dist)
        nz = dist > 0.0
        for bj, nj in terms:
            tail[nz] += bj * _power(dist[nz], nj, POWER_BOX)
        table = {tuple(int(x) for x in m): float(v) for m, v in zip(labels, vals - tail)}
        return cls(b=[t[0] for t in terms], nu=[t[1] for t in terms], compact=table,
                   label=label, lattice=_lattice_fingerprint(A))

    @classmethod
    def from_shells(cls, A, shells: Mapping[float, float], *, b=(), nu=(),
                    total: bool = True, label=None) -> "Interaction":
        r"""Couplings on lattice SHELLS keyed by physical distance, e.g.
        ``{1.0: J1, sqrt(3): J2}`` on the triangular lattice.  A shell is the
        set of lattice points whose distance matches the key to
        ``_SHELL_RTOL``; a key matching no shell raises and names the
        nearest shells below and above it.  With ``total=True`` (default)
        the value is the TOTAL coupling on that shell — the table stores
        ``J - Σ_j b_j r^{-ν_j}`` so that ``K = J`` there exactly — with
        ``total=False`` it is added on top of the power-law tail.  There
        is no threshold on that deviation: a ``J`` one ulp from the tail
        value keeps a ``~1e-15`` table entry (not an exact zero, so not
        dropped) and the kernel keeps its compact part and compact
        routing; only ``J`` EXACTLY equal to the tail stores an exact
        zero, which is dropped."""
        A = _as_matrix(A)
        terms = _canonical_terms(b, nu)
        if not shells:
            raise ValueError("shells must map at least one distance to a coupling")
        radii = [float(r) for r in shells]
        if any(not (r > 0.0) or not math.isfinite(r) for r in radii):
            raise ValueError("shell distances must be finite and positive")
        labels, dist = _labels_in_ball(A, max(radii) * (1.0 + 1e-6))
        table: dict = {}
        for r, J in shells.items():
            r = float(r)
            mask = np.isclose(dist, r, rtol=_SHELL_RTOL, atol=0.0)
            if not mask.any():
                below, above = _nearest_shells(A, r, dist)
                raise ValueError(
                    f"no lattice shell at distance {r!r} on this lattice; the "
                    f"nearest shells are {below} (below) and {above} (above); "
                    f"a key must match a shell to {_SHELL_RTOL:g} relative"
                )
            for m in labels[mask]:
                val = float(J)
                if total:
                    # Subtract the tail at the SHELL distance, so every
                    # label of a shell stores the same float (the shell
                    # is one symmetry orbit; equal entries keep the slab
                    # fold's exact table test honest).
                    for bj, nj in terms:
                        val -= bj * float(_power(np.asarray(r), nj, POWER_BOX))
                key = tuple(int(x) for x in m)
                if key in table:
                    raise ValueError(f"shell distances {radii} overlap at label {key}")
                table[key] = val
        return cls(b=[t[0] for t in terms], nu=[t[1] for t in terms], compact=table,
                   label=label, lattice=_lattice_fingerprint(A))

    @classmethod
    def nearest_neighbour(cls, A, J: float = 1.0, *, label=None) -> "Interaction":
        """The nearest-neighbour indicator: ``J`` on the minimal non-zero
        lattice shell, zero elsewhere — the ``nu -> inf`` limit, identical
        to passing ``np.inf`` (same shell rule as the engines)."""
        A = _as_matrix(A)
        d = A.shape[0]
        dmin_guess = float(min(np.linalg.norm(A[:, i]) for i in range(d)))
        # The ball carries the engines' shell slack: a vector within
        # _SHELL_RTOL of the minimum IS on the nu = inf shell, so it must be
        # enumerated -- a ball of radius exactly dmin_guess (1e-12 slack)
        # dropped the second axis of diag(1, 1 + 1e-10), giving a 2-label
        # table where both engines put 4 ones.  dmin_guess is an upper
        # bound on dmin, so widening loses nothing.
        labels, dist = _labels_in_ball(A, dmin_guess * (1.0 + 2.0 * _SHELL_RTOL))
        nz = dist > 0.0
        dmin = float(dist[nz].min())
        shell = nz & np.isclose(dist, dmin, rtol=_SHELL_RTOL, atol=0.0)
        table = {tuple(int(x) for x in m): float(J) for m in labels[shell]}
        return cls(compact=table, label=label, lattice=_lattice_fingerprint(A))

    @classmethod
    def from_config(cls, cfg: Mapping, A) -> "Interaction":
        """Build from a TOML/JSON-style mapping (the CLI's ``[interaction]``
        table): keys ``label``, ``b``, ``nu``, and at most one of

        - ``compact``: rows ``[m_1, ..., m_d, value]``;
        - ``compact_npz``: an ``.npz`` file with arrays ``labels`` of shape
          ``(N, d)`` (a 1-D ``(N,)`` array is accepted in d = 1 only) and
          ``values`` of shape ``(N,)``;
        - ``shells``: rows ``[distance, coupling]`` or a mapping
          ``{distance: coupling}`` (TOML inline-table keys are strings and
          are parsed as floats), with ``total`` — a bool, default ``True``
          — as in :meth:`from_shells`.  ``total`` is refused with any
          other form.

        Labels are NOT cast to int here: they go through the same
        integrality guard as :meth:`from_table`, so ``1.5`` or
        ``0.9999999999`` raise instead of being truncated onto a
        neighbouring site."""
        A = _as_matrix(A)
        d = A.shape[0]
        allowed = {"label", "b", "nu", "compact", "compact_npz", "shells", "total"}
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(f"unknown [interaction] keys {sorted(unknown)}; allowed: {sorted(allowed)}")
        forms = [k for k in ("compact", "compact_npz", "shells") if k in cfg]
        if len(forms) > 1:
            raise ValueError(f"[interaction] may use only one of compact / compact_npz / shells; got {forms}")
        b, nu, label = cfg.get("b", ()), cfg.get("nu", ()), cfg.get("label")
        if "total" in cfg:
            total = cfg["total"]
            if not isinstance(total, (bool, np.bool_)):
                raise TypeError(
                    f"[interaction] total must be a bool (true / false); got {total!r}"
                )
            if forms != ["shells"]:
                raise ValueError(
                    "[interaction] total applies to the shells form only (it says "
                    "whether a shell coupling is the TOTAL coupling or is added on "
                    "top of the tail); with "
                    + (f"{forms[0]!r}" if forms else "no table form")
                    + " it has no meaning"
                )
        if not forms:
            return cls(b=b, nu=nu, label=label)
        if forms[0] == "shells":
            rows = cfg["shells"]
            if isinstance(rows, Mapping):
                items = list(rows.items())
                if any(isinstance(v, Mapping) for _, v in items):
                    raise ValueError(
                        "shells: a bare decimal key such as 1.0 = J is a TOML DOTTED key "
                        "(a nested table); quote the distance, {'1.0' = J}, or use rows "
                        "[[1.0, J]]"
                    )
            else:
                items = []
                for row in rows:
                    if isinstance(row, (str, bytes)) or not hasattr(row, "__len__") or len(row) != 2:
                        raise ValueError(
                            f"shells rows must be [distance, coupling] pairs; got {row!r}"
                        )
                    items.append((row[0], row[1]))
            shells: dict = {}
            for r, J in items:
                r, J = float(r), float(J)
                if r in shells:
                    raise ValueError(f"shell distance {r!r} given twice")
                shells[r] = J
            return cls.from_shells(A, shells, b=b, nu=nu, total=bool(cfg.get("total", True)), label=label)
        if forms[0] == "compact":
            table = []
            for row in cfg["compact"]:
                if isinstance(row, (str, bytes)) or not hasattr(row, "__len__"):
                    raise ValueError(
                        f"compact rows must be [m_1, ..., m_d, value] lists; got {row!r}"
                    )
                row = list(row)
                if len(row) != d + 1:
                    raise ValueError(f"compact rows must have d + 1 = {d + 1} entries; got {row}")
                try:
                    lab, val = tuple(float(x) for x in row[:d]), float(row[d])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"compact rows must be numbers [m_1, ..., m_d, value]; got {row!r}"
                    ) from exc
                # The raw (float) label: _canonical_table's integrality
                # guard decides, exactly as for from_table.
                table.append((lab, val))
        else:
            path = str(cfg["compact_npz"])
            data = np.load(path)
            if not hasattr(data, "files"):
                raise ValueError(
                    f"compact_npz: {path} is not an .npz archive (need the arrays "
                    f"'labels' (N, d) and 'values' (N,))"
                )
            try:
                missing = [k for k in ("labels", "values") if k not in data.files]
                if missing:
                    raise ValueError(
                        f"compact_npz: {path} lacks the array"
                        f"{'s' if len(missing) > 1 else ''} {missing}; need "
                        f"'labels' (N, d) and 'values' (N,)"
                    )
                labels = np.asarray(data["labels"])
                values = np.asarray(data["values"], dtype=float)
            finally:
                data.close()
            if d == 1 and labels.ndim == 1:
                labels = labels.reshape(-1, 1)
            if labels.ndim != 2 or labels.shape[1] != d:
                raise ValueError(
                    f"compact_npz: labels must have shape (N, d) = (N, {d}); got "
                    f"{labels.shape}"
                )
            if not (np.issubdtype(labels.dtype, np.integer)
                    or np.issubdtype(labels.dtype, np.floating)):
                raise ValueError(
                    f"compact_npz: labels must be an integer (or integral float) "
                    f"array; got dtype {labels.dtype}"
                )
            if values.ndim != 1 or values.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"compact_npz: values must have shape (N,) = ({labels.shape[0]},) "
                    f"to match labels; got {values.shape}"
                )
            # Raw labels (ints, or floats that must be integral): the
            # integrality guard of _canonical_table decides, never int().
            table = [(tuple(m.tolist()), float(v)) for m, v in zip(labels, values)]
        return cls.from_table(table, b=b, nu=nu, label=label, lattice=A)

    # -- derived quantities ------------------------------------------------

    @property
    def terms(self) -> tuple:
        """The power-law terms as ``((b_j, nu_j), ...)``."""
        return tuple(zip(self.b, self.nu))

    @property
    def dim(self) -> "int | None":
        """Label dimension of the compact table, ``None`` without one."""
        return len(self.compact[0][0]) if self.compact else None

    @property
    def tail_exponent(self) -> float:
        """``min_j nu_j`` — the slowest decay, the one number every planner,
        cut and routing consumer reads; ``+inf`` for a purely compact kernel."""
        return min(self.nu) if self.nu else math.inf

    @property
    def has_compact(self) -> bool:
        return bool(self.compact)

    @property
    def is_pure_power_law(self) -> bool:
        """No compact part (``zeta_circle``-eligible after factoring ``Π b``)."""
        return not self.compact

    @property
    def is_plain_power_law(self) -> bool:
        """Exactly ``|x|^{-nu}`` — demoted to the float ``nu`` at the boundary."""
        return not self.compact and len(self.nu) == 1 and self.b == (1.0,)

    def as_plain_float(self) -> "float | None":
        return self.nu[0] if self.is_plain_power_law else None

    @property
    def support_radius(self) -> int:
        """Chebyshev radius of the compact support in label space (0 if none)."""
        return _chebyshev_radius(self.compact)

    def key(self) -> tuple:
        """Hashable, sortable, exactly comparable fingerprint (label-free):
        ``("I", rounded exponents, weights, table, lattice fingerprint)``;
        never equal to a float.  The lattice fingerprint (``b""`` when the
        table was not sampled on a lattice) keeps two equal tables sampled
        on different cells in different cache entries, so the lattice
        guard of :meth:`sample` is never skipped by a cache hit.

        Exponents are quantised to ``KEY_DECIMALS`` = 12 decimals — the
        rounding the frontend block cache applies to every exponent — so
        the engines' kernel caches treat two Interactions whose exponents
        differ by less than ``1e-12`` as one kernel (measured ``1.4e-13``
        relative on the kernel at d = 1, n = 8; ``1e-11`` apart stays
        distinct), whereas the legacy float kernel key is exact."""
        return ("I", tuple(round(n, KEY_DECIMALS) for n in self.nu), tuple(self.b),
                self.compact, self.lattice or b"")

    @property
    def default_label(self) -> str:
        if self.label is not None:
            return self.label
        if self.is_plain_power_law:
            return f"nu={self.nu[0]:g}"
        digest = hashlib.sha1(repr(self.key()).encode()).hexdigest()[:10]
        return f"K#{digest}"

    # -- sampling and closed forms -------------------------------------------

    def _check_lattice(self, A: np.ndarray) -> None:
        if self.dim is not None and self.dim != A.shape[0]:
            raise ValueError(
                f"the compact table is {self.dim}-dimensional but the lattice is "
                f"{A.shape[0]}-dimensional"
            )
        if self.lattice is not None and self.lattice != _lattice_fingerprint(A):
            raise ValueError(
                "this Interaction was sampled on a different lattice matrix A; "
                "rebuild it with the lattice you evaluate on"
            )

    def sample(self, labels, A, *, power=POWER_TORUS) -> np.ndarray:
        r"""The kernel on an integer label array ``labels`` of shape
        ``(..., d)``: ``Σ_j b_j |A m|^{-ν_j}`` (zero where ``|A m| = 0``, in
        the ENGINE's power form) plus the compact table scattered at its
        labels (``a(0)`` kept).  Returns float64 of shape ``labels.shape[:-1]``.

        Raises :class:`InteractionSupportError` if a table label is absent
        from ``labels`` — the window would clip or alias the compact part.
        """
        A = _as_matrix(A)
        self._check_lattice(A)
        labels = self._check_labels(labels, A.shape[0])
        shape = labels.shape[:-1]
        flat = labels.reshape(-1, A.shape[0])
        dist = np.linalg.norm(flat @ A.T, axis=1)
        out = np.zeros(flat.shape[0], dtype=np.float64)
        nz = dist > 0.0
        for bj, nj in self.terms:
            out[nz] += bj * _power(dist[nz], nj, power)
        out += self._scatter(flat)
        return out.reshape(shape)

    def compact_array(self, labels) -> np.ndarray:
        r"""The compact part ALONE, ``a(m)`` scattered onto an integer label
        array of shape ``(..., d)`` (zero elsewhere) — what the semi-analytic
        algebra puts into ``aMat`` while it keeps the power-law terms in
        ``(bVec, nuVec)``.  Same window guard as :meth:`sample`."""
        d = self.dim
        labels = np.asarray(labels)
        if d is None:
            if labels.ndim < 1:
                raise ValueError("labels must have shape (..., d)")
            return np.zeros(labels.shape[:-1], dtype=np.float64)
        labels = self._check_labels(labels, d)
        return self._scatter(labels.reshape(-1, d)).reshape(labels.shape[:-1])

    @staticmethod
    def _check_labels(labels, d: int) -> np.ndarray:
        labels = np.asarray(labels)
        if labels.ndim < 1 or labels.shape[-1] != d:
            raise ValueError(f"labels must have shape (..., {d}); got {labels.shape}")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must be an integer array")
        return labels

    def _scatter(self, flat: np.ndarray) -> np.ndarray:
        """``a(m)`` on a flat ``(N, d)`` label array, guarding the window."""
        return _scatter_table(self.compact, flat, self.support_radius)

    def fourier_terms(self, A):
        r"""The lattice Fourier transform of the kernel as analytic data:
        ``K̂(k) = Σ_j c_j Z_{ν_j}(k) + Σ_m v_m cos(2π m·k)`` with ``Z_ν`` the
        Epstein zeta ``Σ_{x≠0} |x|^{-ν} e^{-2πi x·k}``.  Returns
        ``(terms, labels, values)``: ``terms`` a tuple of ``(c_j, ν_j)``,
        ``labels`` an ``(N, d)`` int array and ``values`` ``(N,)`` — the
        compact part, whose transform is a finite trigonometric polynomial
        (even table ⇒ a cosine sum).  This is what the bridge closed form
        and the cycle quadrature consume."""
        A = _as_matrix(A)
        self._check_lattice(A)
        d = A.shape[0]
        if self.compact:
            labels = np.asarray([m for m, _ in self.compact], dtype=int)
            values = np.asarray([v for _, v in self.compact], dtype=float)
        else:
            labels = np.zeros((0, d), dtype=int)
            values = np.zeros(0, dtype=float)
        return tuple(self.terms), labels, values

    def lattice_sum(self, A, k_frac=None):
        r"""``Σ_x K(x) e^{-2πi x·k}`` over the whole lattice — the bridge
        closed form — with ``k`` in fractional Brillouin-zone coordinates
        (``None`` or zeros: ``k = 0``).  Exact: the compact part is a finite
        sum, the power-law part is ``Σ_j b_j Z_{ν_j}(k)`` (Epstein zeta).
        A ``(d,)`` ``k`` returns a float; a ``(N, d)`` array returns ``(N,)``.
        In d = 1 a scalar ``k`` is one momentum (float) and a 1-D array of
        length ``N != 1`` is a BATCH of ``N`` momenta (returns ``(N,)`` —
        the shape ``evaluate_graph`` reads the same way); in d >= 2 a 1-D
        array whose length is not ``d`` raises ``ValueError``.
        """
        return _lattice_sum_from_terms(self.fourier_terms(_as_matrix(A)), A, k_frac)

    # -- algebra ---------------------------------------------------------------

    def _scaled(self, c: float) -> "Interaction":
        c = float(c)
        if c == 0.0:
            raise ValueError("scaling an Interaction by 0 gives the zero kernel, which is refused")
        return Interaction(b=[c * bj for bj in self.b], nu=self.nu,
                           compact={m: c * v for m, v in self.compact},
                           label=self.label, lattice=self.lattice)

    def __mul__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)) and not isinstance(other, bool):
            return self._scaled(other)
        if isinstance(other, Interaction):
            return _KernelProduct((self, other))
        if isinstance(other, _KernelProduct):
            return _KernelProduct((self,) + other.factors)
        return NotImplemented

    __rmul__ = __mul__

    def __pow__(self, m):
        if isinstance(m, bool) or float(m) != int(m) or int(m) < 1:
            raise ValueError("the Hadamard power needs a positive integer multiplicity")
        m = int(m)
        return self if m == 1 else _KernelProduct((self,) * m)

    def __add__(self, other):
        if not isinstance(other, Interaction):
            return NotImplemented
        if (self.lattice is not None and other.lattice is not None
                and self.lattice != other.lattice):
            raise ValueError("cannot add Interactions sampled on different lattices")
        table = dict(self.compact)
        for m, v in other.compact:
            table[m] = table.get(m, 0.0) + v
        return Interaction(b=self.b + other.b, nu=self.nu + other.nu, compact=table,
                           lattice=self.lattice if self.lattice is not None else other.lattice)

    def __radd__(self, other):
        # ``sum([V1, V2, ...])`` starts from the int 0, which is the natural
        # way to build a multi-term kernel from a list of shell
        # contributions.  Without this it fails with a TypeError naming
        # ``int``, which points at the wrong thing.  Only the additive
        # identity is accepted: a bare scalar plus a kernel is meaningless,
        # because the sum would not decay.
        if isinstance(other, (int, float)) and other == 0:
            return self
        return NotImplemented

    def __neg__(self):
        return self._scaled(-1.0)

    def __sub__(self, other):
        if not isinstance(other, Interaction):
            return NotImplemented
        return self + (-other)

    def __repr__(self) -> str:
        parts = []
        if self.nu:
            parts.append("b=" + repr(list(self.b)) + ", nu=" + repr(list(self.nu)))
        if self.compact:
            parts.append(f"compact={len(self.compact)} labels (R={self.support_radius})")
        if self.label is not None:
            parts.append(f"label={self.label!r}")
        return "Interaction(" + ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Lazy Hadamard products
# ---------------------------------------------------------------------------

@dataclass(frozen=True, init=False, repr=False)
class _KernelProduct:
    """The pointwise product of several :class:`Interaction` factors —
    what parallel edges (and a corpus multiplicity ``m``) evaluate to.
    Samples as the exact pointwise product of the factor samples; the
    analytic tail is expanded only in :meth:`lattice_sum`."""

    factors: tuple

    __array_ufunc__ = None

    def __init__(self, factors: Iterable):
        flat: list = []
        for f in factors:
            if isinstance(f, _KernelProduct):
                flat.extend(f.factors)
            elif isinstance(f, Interaction):
                flat.append(f)
            else:
                raise TypeError(f"product factors must be Interactions; got {type(f).__name__}")
        if len(flat) < 2:
            raise ValueError("a _KernelProduct needs at least two factors; use the Interaction itself")
        dims = {f.dim for f in flat if f.dim is not None}
        if len(dims) > 1:
            raise ValueError(f"product factors mix table dimensions {sorted(dims)}")
        lat = {f.lattice for f in flat if f.lattice is not None}
        if len(lat) > 1:
            raise ValueError("product factors were sampled on different lattices")
        object.__setattr__(self, "factors", tuple(flat))
        if self._purely_compact and not self._collapsed_compact():
            # mirrors the constructor's refusal of the zero kernel
            raise ValueError(
                "the Hadamard product of purely compact tables with disjoint "
                "supports is the zero kernel (a parallel bundle whose tables "
                "share no label); it is refused so a graph zeta cannot silently "
                "vanish"
            )

    @property
    def label(self) -> "str | None":
        return None

    @property
    def default_label(self) -> str:
        return "*".join(f.default_label for f in self.factors)

    @property
    def dim(self) -> "int | None":
        return next((f.dim for f in self.factors if f.dim is not None), None)

    @property
    def lattice(self) -> "bytes | None":
        return next((f.lattice for f in self.factors if f.lattice is not None), None)

    @property
    def tail_exponent(self) -> float:
        return float(sum(f.tail_exponent for f in self.factors))

    @property
    def has_compact(self) -> bool:
        return any(f.has_compact for f in self.factors)

    @property
    def is_pure_power_law(self) -> bool:
        return all(f.is_pure_power_law for f in self.factors)

    @property
    def is_plain_power_law(self) -> bool:
        return all(f.is_plain_power_law for f in self.factors)

    def as_plain_float(self) -> "float | None":
        if not self.is_plain_power_law:
            return None
        # The same float sum the frontend's Hadamard merge forms for
        # repeated legacy rows (bundles[key] += nu_e, in order).
        total = 0.0
        for f in self.factors:
            total = total + f.nu[0]
        return total

    @property
    def _purely_compact(self) -> bool:
        """Every factor is a table without power-law terms."""
        return all(not f.terms for f in self.factors)

    def _collapsed_compact(self) -> tuple:
        """A product of purely compact factors AS ONE TABLE over the
        INTERSECTION of their supports, each value the factor values
        multiplied in factor order — bit-identical to the pointwise
        product of the factor samples there (and that product is an exact
        zero everywhere else).  Exact zeros are kept."""
        tabs = [dict(f.compact) for f in self.factors]
        common = set(tabs[0])
        for t in tabs[1:]:
            common &= set(t)
        out = []
        for m in sorted(common):
            v = tabs[0][m]
            for t in tabs[1:]:
                v = v * t[m]
            out.append((m, v))
        return tuple(out)

    @property
    def support_radius(self) -> int:
        """Chebyshev radius of the window the product needs.  When EVERY
        factor is purely compact the product's support is the INTERSECTION
        of the factor supports (it can even be empty), bounded by the
        smallest factor radius — and :meth:`sample` scatters the collapsed
        intersection table, so the window need only hold that.  With any
        power-law factor the product is non-zero wherever any table is, so
        the largest factor radius."""
        radii = [f.support_radius for f in self.factors]
        return min(radii) if self._purely_compact else max(radii)

    def key(self) -> tuple:
        return ("H", tuple(sorted(f.key() for f in self.factors)))

    def sample(self, labels, A, *, power=POWER_TORUS) -> np.ndarray:
        """The pointwise product of the factor samples (exact on any grid).
        A product of purely compact factors is scattered from its collapsed
        intersection table instead, so the window guard follows
        :attr:`support_radius` (the intersection), not the largest factor."""
        if self._purely_compact:
            A = _as_matrix(A)
            for f in self.factors:
                f._check_lattice(A)
            d = A.shape[0]
            labels = Interaction._check_labels(labels, d)
            return _scatter_table(self._collapsed_compact(), labels.reshape(-1, d),
                                  self.support_radius).reshape(labels.shape[:-1])
        out = None
        for f in self.factors:
            s = f.sample(labels, A, power=power)
            out = s if out is None else out * s
        return out

    def _tail_expansion(self) -> list:
        """``Π_e (Σ_j b_ej K_{ν_ej})`` as ``[(coefficient, ν_sum), ...]`` over
        exponent MULTISETS — canonical, so equal sums merge exactly."""
        states: dict = {(): 1.0}
        for f in self.factors:
            if not f.terms:
                return []          # a purely compact factor kills the tail
            nxt: dict = {}
            for ms, coef in states.items():
                for bj, nj in f.terms:
                    key = tuple(sorted(ms + (nj,)))
                    nxt[key] = nxt.get(key, 0.0) + coef * bj
            states = nxt
        out: dict = {}
        for ms, coef in states.items():
            if coef == 0.0:
                continue
            nu_sum = math.fsum(ms)
            out[nu_sum] = out.get(nu_sum, 0.0) + coef
        return [(c, n) for n, c in sorted(out.items()) if c != 0.0]

    def fourier_terms(self, A):
        r"""As :meth:`Interaction.fourier_terms`, for the product kernel
        ``W = Π_e V_e``: the compact part is ``Π_e V_e - Π_e P_e`` on the
        union of the factor supports (a finite table), and the power-law
        part is the multiset expansion of ``Π_e P_e`` in Epstein zetas."""
        A = _as_matrix(A)
        for f in self.factors:
            f._check_lattice(A)
        d = A.shape[0]
        support = sorted({m for f in self.factors for m, _ in f.compact})
        if support:
            labels = np.asarray(support, dtype=int)
            dist = np.linalg.norm(labels @ A.T, axis=1)
            nz = dist > 0.0
            full = np.ones(len(support), dtype=float)
            tail = np.ones(len(support), dtype=float)
            for f in self.factors:
                P = np.zeros(len(support), dtype=float)
                for bj, nj in f.terms:
                    P[nz] += bj * _power(dist[nz], nj, POWER_BOX)
                tab = dict(f.compact)
                a = np.asarray([tab.get(m, 0.0) for m in support], dtype=float)
                full *= a + P
                tail *= P
            values = full - tail
        else:
            labels = np.zeros((0, d), dtype=int)
            values = np.zeros(0, dtype=float)
        return tuple(self._tail_expansion()), labels, values

    def lattice_sum(self, A, k_frac=None):
        r"""``Σ_x Π_e V_e(x) e^{-2πi x·k}``: the finite sum over the union of
        the compact supports of ``Π_e V_e - Π_e P_e`` plus the multiset
        expansion of the pure power-law product in Epstein zetas."""
        return _lattice_sum_from_terms(self.fourier_terms(_as_matrix(A)), A, k_frac)

    def __mul__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)) and not isinstance(other, bool):
            return _KernelProduct((self.factors[0]._scaled(other),) + self.factors[1:])
        if isinstance(other, Interaction):
            return _KernelProduct(self.factors + (other,))
        if isinstance(other, _KernelProduct):
            return _KernelProduct(self.factors + other.factors)
        return NotImplemented

    __rmul__ = __mul__

    def __pow__(self, m):
        if isinstance(m, bool) or float(m) != int(m) or int(m) < 1:
            raise ValueError("the Hadamard power needs a positive integer multiplicity")
        m = int(m)
        return self if m == 1 else _KernelProduct(self.factors * m)

    def __repr__(self) -> str:
        return "(" + " * ".join(repr(f) for f in self.factors) + ")"


InteractionLike = (Interaction, _KernelProduct)


def is_interaction(x) -> bool:
    return isinstance(x, InteractionLike)


# ---------------------------------------------------------------------------
# The boundary normaliser
# ---------------------------------------------------------------------------

def coerce_nu(nu, n_edges: int):
    r"""Normalise the ν-like argument of every public entry point.

    Returns ``(nu_tail, kernels)`` where ``nu_tail`` is the float64 array
    of per-edge TAIL exponents (``min_j ν_j``, ``+inf`` for a purely compact
    kernel) that every planner / cut / routing consumer reads, and
    ``kernels`` is either ``None`` — the legacy power-law path, taken when
    no genuine Interaction is present (a plain ``Interaction(b=[1],
    nu=[ν])`` is demoted to ``ν``) — or a full per-edge list of
    Interaction-likes (floats promoted to :meth:`Interaction.power_law`),
    never a mix, so cache keys stay type-uniform per call.

    The legacy branch is the front-end's historical coercion verbatim
    (scalar broadcast; 1-D float array with the same length ``ValueError``).
    """
    n_edges = int(n_edges)
    if isinstance(nu, np.ndarray) and nu.dtype == object:
        # np.array([K, K, K]) / np.array([K, 2.5], dtype=object): a per-edge
        # SEQUENCE, never the float branch (whose cast dies on an
        # Interaction with an opaque TypeError).
        nu = list(nu.ravel())
    if isinstance(nu, InteractionLike):
        kernels = [nu] * n_edges
    elif isinstance(nu, np.ndarray) or np.isscalar(nu):
        if np.isscalar(nu):
            return np.full(n_edges, float(nu), dtype=float), None
        nu_arr = np.asarray(nu, dtype=float).ravel()
        if len(nu_arr) != n_edges:
            raise ValueError(
                f"nu has length {len(nu_arr)} but edges_flat has {n_edges} edges."
            )
        return nu_arr, None
    else:
        seq = list(nu)
        if not any(isinstance(x, InteractionLike) for x in seq):
            nu_arr = np.asarray(seq, dtype=float).ravel()
            if len(nu_arr) != n_edges:
                raise ValueError(
                    f"nu has length {len(nu_arr)} but edges_flat has {n_edges} edges."
                )
            return nu_arr, None
        if len(seq) != n_edges:
            raise ValueError(
                f"nu has length {len(seq)} but edges_flat has {n_edges} edges."
            )
        kernels = []
        for x in seq:
            if isinstance(x, InteractionLike):
                kernels.append(x)
            else:
                xf = float(x)
                if not math.isfinite(xf):
                    raise ValueError(
                        "a per-edge list that mixes Interactions with nu = inf is "
                        "ambiguous; use Interaction.nearest_neighbour(A) for that edge"
                    )
                kernels.append(Interaction.power_law(xf))
    plain = [k.as_plain_float() for k in kernels]
    if all(p is not None for p in plain):
        return np.asarray(plain, dtype=float), None
    nu_tail = np.asarray([k.tail_exponent for k in kernels], dtype=float)
    return nu_tail, kernels
