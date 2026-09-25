# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

"""
Graph Zeta Library for the efficient and precise evaluation of graph zeta functions, based on a semi-analytic algebra.

Implements the :class:`GraphZeta` data structure (Fourier part ``aMat``,
Epstein-zeta coefficients ``bVec``/``nuVec``, lattice matrix ``A``) together
with the core operations :func:`graph_multiply`, :func:`graph_convolve`,
:func:`graph_compress`, :func:`graph_sample`, and :func:`graph_zero`.

Author: Andreas A. Buchheit, 2026
"""

from __future__ import annotations

import functools
from math import factorial

import numpy as np
from dataclasses import dataclass
from scipy.fft import fftn, ifftn
from scipy.special import gamma, gammaln, gammasgn, rgamma

from epsteinlib import epstein_zeta, epstein_zeta_reg

from gzl._errors import GraphZetaError
from gzl._lattices import _resolve_lattice


__all__ = [
    "GraphZeta",
    "cNu",
    "make_graph_obj",
    "make_epstein_graph",
    "graph_sample",
    "graph_zero",
    "graph_zero_conditioned",
    "graph_compress",
    "graph_multiply",
    "graph_multiply_power",
    "graph_convolve",
    "graph_convolve_power",
    "graph_attach",
    "periodic_convolve_nd",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Below this value of ``|y|**2 * vol**(2/d)``, :func:`_epstein_zeta_k`
#: adds the cusp itself.  epsteinlib replaces ``y`` by ``0`` in the zero
#: term of its Fourier sum when that product, in its own scaled units, is
#: below ``1e-64``, and returns the value at ``k = 0``.  The threshold
#: sits two decades of ``|y|`` above that, where the two agree to
#: round-off (measured on the chain, the square, triangular, sheared and
#: scaled cells, the cubic and the fcc cell: at most 7e-15 relative).
_CUSP_SCALED_Y2 = 1e-60

#: epsteinlib's window around ``nu = d`` and around the zeros of
#: ``1 / Gamma(nu / 2)`` at ``nu = 0, -2, -4, ...`` (``EPS`` in its
#: ``zeta.c``).
_EPSTEIN_NU_WINDOW = 2.0 ** -30


def _epstein_zeta_k(nu, A, y, vol=None) -> complex:
    r"""``epstein_zeta(nu, A, 0, y)`` at any Cartesian momentum ``y``.

    epsteinlib returns the value at ``y = 0`` for ``0 < |y| < 1e-32``
    (scaled by the cell), which drops the cusp
    :math:`\hat s_\nu(y) / V` of

    .. math::

        Z_\nu(y) = Z^\mathrm{reg}_\nu(y) + \frac{1}{V}\,
        \pi^{\nu - d/2}\,
        \frac{\Gamma\bigl((d - \nu)/2\bigr)}{\Gamma(\nu/2)}\,
        |y|^{\nu - d},

    with :math:`Z^\mathrm{reg}` epsteinlib's regularised function, which
    is analytic at ``y = 0``, and ``V = |det A|``.  For ``nu < d`` the
    cusp diverges as ``y -> 0``, and just above ``d`` it vanishes slowly:
    on the chain at ``k = 1e-35`` epsteinlib gave -18.86 for the exact
    49431.86 at ``nu = 0.9``, and was 83 % off at ``nu = 1.01``.  Below
    :data:`_CUSP_SCALED_Y2` this function returns the right-hand side
    above, which reproduces the exact Lindelöf expansion of the chain to
    round-off down to ``|k| = 1e-60``.

    Everywhere else, and for an exact zero ``y``, the value is
    epsteinlib's, bit for bit.  It is also epsteinlib's where the cusp
    is not the missing piece: for ``nu >= d + 1`` the dropped term is
    below ``1e-30`` of the scale, and within :data:`_EPSTEIN_NU_WINDOW`
    of ``nu = d`` (a logarithmic cusp, where epsteinlib returns NaN and
    the caller refuses) or of a zero of ``1 / Gamma(nu / 2)`` (where
    epsteinlib returns the ``y``-independent value at every momentum).
    ``vol`` may be passed when the caller already has ``|det A|``.
    """
    d = A.shape[0]
    zero = np.zeros(d, dtype=float)
    value = epstein_zeta(nu, A, zero, y)
    nu = float(nu)
    if not nu < d + 1 or not np.any(y):
        return value
    m = float(np.max(np.abs(y)))
    ynorm = m * float(np.sqrt(np.sum((np.asarray(y) / m) ** 2)))
    if vol is None:
        vol = abs(float(np.linalg.det(A)))
    if not (ynorm * vol ** (1.0 / d)) ** 2 < _CUSP_SCALED_Y2:
        return value
    if abs(nu - d) <= _EPSTEIN_NU_WINDOW:
        return value
    if nu < 1 and abs(nu / 2 - round(nu / 2)) < _EPSTEIN_NU_WINDOW:
        return value
    with np.errstate(over="ignore", invalid="ignore"):
        cusp = (np.pi ** (nu - d / 2) * gamma((d - nu) / 2) * rgamma(nu / 2)
                * np.exp((nu - d) * np.log(ynorm)) / vol)
    return epstein_zeta_reg(nu, A, zero, y) + cusp

def _vint_array(Az: np.ndarray, nu: float, eps: float = 1e-14) -> np.ndarray:
    """Vectorised interaction potential 1/|x|^nu over a grid of points.

    Given an array ``Az`` of shape ``(..., d)`` holding lattice points, return
    a complex array of shape ``Az.shape[:-1]`` with entries ``1/|Az|^nu`` and
    zero at points within ``eps`` of the origin.
    """
    r = np.linalg.norm(Az, axis=-1)
    out = np.zeros(r.shape, dtype=complex)
    mask = r > eps
    out[mask] = r[mask] ** (-float(nu))
    return out


def cNu(nu: float, d: int) -> float:
    """Prefactor of the distributional Fourier transform of 1/|x|^nu.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.
    """
    nu, d = float(nu), int(d)
    return float(
        (np.pi ** (nu - d / 2.0)) * gamma((d - nu) / 2.0) / gamma(nu / 2.0)
    )


def _is_nonpositive_integer(x: float, tol: float = 1e-14) -> bool:
    """Return True if *x* is within *tol* of 0, -1, -2, …"""
    if x > tol:
        return False
    return abs(x - round(x)) < tol


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphZeta:
    r"""Semi-analytical representation of a graph zeta function.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    The represented function is, with ``k`` in fractional Brillouin-zone
    coordinates and ``Z_ν(k) = epstein_zeta(ν, A, 0, A^{-T} k)`` the
    lattice Epstein zeta function,

    .. math::

        \zeta(\boldsymbol{k}) = \sum_{\boldsymbol{z} \in \text{window}}
            \mathtt{aMat}[\boldsymbol{z}]\, e^{+2\pi i\, \boldsymbol{z} \cdot \boldsymbol{k}}
            + \sum_j \mathtt{bVec}_j\, Z_{\mathtt{nuVec}_j}(\boldsymbol{k}) .

    Attributes
    ----------
    aMat : np.ndarray (complex), shape ``(n,)*d``
        The regular part in REAL SPACE — the Fourier coefficients of
        ``ζ_reg(k) = Σ_z aMat[z] e^{+2πi z·k}`` on the balanced label grid
        ``z ∈ {0, 1, ..., floor(n/2), -ceil(n/2) + 1, ..., -1}^d`` — not
        samples of ``ζ_reg`` in k-space.  :func:`graph_compress` writes
        it (a window-truncated ``|A z|^{-ν}`` absorbed from
        ``(bVec, nuVec)``), :func:`graph_sample` reads it back through an
        inverse FFT, and :func:`graph_multiply` corrects it with the
        forward FFT of a k-space residual.  For an edge kernel ``V`` the
        convention is ``aMat[z] = V(-A z)`` — the definition's phase is
        ``e^{-2πi (x_t - x_s)·k}`` — which for the even kernels the
        library handles equals ``V(A z)``; a compact interaction table is
        therefore scattered at its own labels
        (``gzl.construction._interaction_leaf``).
    bVec : np.ndarray (complex)
        Coefficient vector of the Epstein-zeta (analytic) part.
    nuVec : np.ndarray (float)
        Exponent vector of the Epstein-zeta part, aligned with ``bVec``.
    A : np.ndarray (float)
        Lattice matrix (d × d); columns are the primitive vectors.
    table_magnitude : float or None
        ``None`` on the power-law path — every object built without an
        interaction kernel, where nothing reads it and every operation
        runs byte for byte as before.  On the kernel path an UPPER
        bound on the ℓ¹ norm ``Σ_z |c(z)|`` of the Fourier coefficients
        that descend from compact interaction tables:
        :func:`gzl.construction._interaction_leaf` sets
        ``Σ_m |a(m)|`` of the scattered table, and :func:`graph_multiply`
        / :func:`graph_convolve` carry it forward through
        :func:`_compose_table_magnitude` (scalings by
        :func:`graph_attach` multiply it by ``|ζ_dec(0)|``).
        :func:`graph_zero_conditioned` reads it so that its cancellation
        ratio sees a table whose entries cancel inside the ``np.sum``
        over ``aMat`` — ``|Σ a|`` alone reports 1 for such a table.
    """

    aMat: np.ndarray
    bVec: np.ndarray
    nuVec: np.ndarray
    A: np.ndarray
    table_magnitude: float | None = None


def make_graph_obj(aMat, bVec, nuVec, A, table_magnitude=None) -> GraphZeta:
    """Construct a :class:`GraphZeta` from raw arrays.  ``A`` is the lattice
    matrix or the name of a lattice -- ``"chain"``, ``"square"``,
    ``"triangular"``, ``"cubic"``.  ``table_magnitude`` (default ``None``:
    the power-law path) is the kernel path's tracked table bound, see
    :class:`GraphZeta`.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.
    """
    return GraphZeta(
        aMat=np.asarray(aMat, dtype=complex),
        bVec=np.asarray(bVec, dtype=complex),
        nuVec=np.asarray(nuVec, dtype=float),
        A=np.asarray(_resolve_lattice(A), dtype=float),
        table_magnitude=(None if table_magnitude is None
                         else float(table_magnitude)),
    )


def make_epstein_graph(nu: float, A, n_samples: int) -> GraphZeta:
    """Create an elementary Epstein graph with a single exponent *nu*.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Parameters
    ----------
    nu : float
        Exponent of the Epstein zeta function.
    A : array-like (d × d) or str
        Lattice matrix, or the name of a lattice -- ``"chain"``,
        ``"square"``, ``"triangular"``, ``"cubic"``.
    n_samples : int
        Number of discretisation points per dimension.
    """
    A = np.asarray(_resolve_lattice(A), dtype=float)
    d = A.shape[0]
    aMat = np.zeros((n_samples,) * d, dtype=complex)
    return make_graph_obj(aMat, bVec=[1], nuVec=[nu], A=A)


# ---------------------------------------------------------------------------
# Sampling / evaluation
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8192)
def _epstein_grid_cached(nui: float, A_bytes: bytes, A_shape: tuple, n: int) -> np.ndarray:
    r"""Cached Epstein-zeta grid ``[ζ(ν; A, 0, A⁻ᵀ y)]`` over the BZ grid.

    The grid depends *only* on the exponent ``ν``, the lattice ``A`` and
    the discretisation ``n`` — not on the particular block — so it recurs
    ~100× across a corpus pass (same handful of bundle exponents, same
    lattice and grid).  Memoising it collapses that redundancy.  The
    returned array is treated read-only by callers.
    """
    # .copy(): np.frombuffer is read-only, but epsteinlib needs a writable
    # buffer for A.  Paid only on a cache miss (≈ once per distinct ν).
    A = np.frombuffer(A_bytes, dtype=np.float64).reshape(A_shape).copy()
    d = A.shape[0]
    Astar = np.linalg.inv(A.T)
    axes = [np.arange(n, dtype=float) / n for _ in range(d)]
    grid = np.meshgrid(*axes, indexing="ij")
    k_points = np.stack(grid, axis=-1).reshape(-1, d)
    zero = np.zeros(d, dtype=float)
    vals = np.empty(k_points.shape[0], dtype=float)
    for idx, y in enumerate(k_points):
        vals[idx] = np.real(epstein_zeta(nui, A, zero, Astar @ y))
    vals.flags.writeable = False        # protect the cached array
    return vals


def graph_sample(g: GraphZeta) -> np.ndarray:
    """Evaluate the graph zeta function on an equidistant grid over [0, 1)^d.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.
    """
    A, a, b, nu = g.A, g.aMat, g.bVec, g.nuVec
    d = A.shape[0]
    n = a.shape[0]

    A = np.ascontiguousarray(A, dtype=np.float64)
    A_bytes, A_shape = A.tobytes(), A.shape

    out = np.zeros((n,) * d, dtype=float)
    for bi, nui in zip(b, nu):
        vals = _epstein_grid_cached(float(nui), A_bytes, A_shape, int(n))
        out += (bi * vals.reshape((n,) * d)).real

    out += np.real(ifftn(a)) * (n ** d)
    return out


def graph_zero(g: GraphZeta) -> float:
    """Return the value of the graph zeta function at k = 0.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Samples a built :class:`GraphZeta` algebra object at the BZ origin.
    For end-to-end evaluation that takes a raw edge list and routes
    automatically (closed-form bridge / cycle, σ_max=4 algebra, or
    tensor), use :func:`gzl.evaluate_graph` instead.
    """
    s = graph_sample(g)
    d = g.A.shape[0]
    return float(s[(0,) * d])


@functools.lru_cache(maxsize=8192)
def _epstein_at_zero_cached(nui: float, A_bytes: bytes, A_shape: tuple) -> float:
    """``|Z_ν(0)|`` for one exponent, one epsteinlib call, memoised like
    the grids.  For ``ν > d`` the Fourier coefficients ``|A z|^{-ν}`` of
    the lattice Epstein zeta are positive, so this IS their ℓ¹ norm."""
    A = np.frombuffer(A_bytes, dtype=np.float64).reshape(A_shape).copy()
    zero = np.zeros(A.shape[0], dtype=float)
    return abs(float(np.real(epstein_zeta(nui, A, zero, zero))))


def _coefficient_l1(g: GraphZeta) -> float:
    r"""An upper bound on the ℓ¹ norm of the Fourier coefficients of
    ``ζ_g``: ``Σ_z |aMat[z]| + Σ_j |bVec_j| |Z_{ν_j}(0)|``.  Kernel path
    only (see :func:`_compose_table_magnitude`); costs one cached
    Epstein evaluation per analytic term."""
    A = np.ascontiguousarray(g.A, dtype=np.float64)
    A_bytes, A_shape = A.tobytes(), A.shape
    total = float(np.sum(np.abs(g.aMat)))
    for bi, nui in zip(g.bVec, g.nuVec):
        total += abs(complex(bi)) * _epstein_at_zero_cached(
            float(nui), A_bytes, A_shape)
    return total


def _compose_table_magnitude(
    g1: GraphZeta, g2: GraphZeta, *, partner_sup=None,
) -> float | None:
    r"""The :attr:`GraphZeta.table_magnitude` of a series
    (:func:`graph_multiply`) or parallel (:func:`graph_convolve`)
    composition of ``g1`` and ``g2``.

    ``None`` when neither operand carries one — the power-law path, which
    this helper leaves without a single extra operation.  Otherwise
    ``M1·L2 + M2·L1`` with ``M_i`` the operands' magnitudes (``0`` for a
    power-law operand) and ``L_i`` a norm of the partner: writing
    ``ζ_i = T_i + P_i`` (``T`` the coefficients descending from tables,
    ``P`` the rest), the table-derived part of the composition is
    ``T1·ζ2 + P1·T2``, and

    * the series collapse is a cyclic convolution of coefficient
      sequences, so Young's inequality gives ``L_i`` =
      :func:`_coefficient_l1` (``partner_sup=None``; aliasing onto the
      ``n``-window can only lower an ℓ¹ norm);
    * the parallel merge is a POINTWISE product of the real-space arrays
      :func:`graph_convolve` forms on the window, so Hölder gives
      ``L_i = max_z |c_i(z)|`` — the caller passes those two sup norms
      as ``partner_sup = (sup|c1|, sup|c2|)`` from the arrays it builds
      anyway.

    ``T1·T2`` is counted twice, so the bound is loose by a factor ~2 per
    composition plus, for a series collapse, the spread of the partner's
    coefficients (measured 2.9–20x on purely compact bundles, paths,
    triangle and C4 with signed radius-2 tables; realistic mixed blocks
    — a J1+J2-like signed table on a σ = 1.2 tail, C4, theta, diamond,
    C6 — report κ ≤ 2.3e3 against an honest ``Σ|aMat| / |value|`` of up
    to 9.2e2, nine orders below the front-end's 1e12 ceiling).  That is
    the price of keeping a cancelling table visible after a composition
    has hidden the cancellation inside an FFT: two tables with entries
    ``±1e6`` that sum to 0.2, in series, give ``(Σ a)² = 0.04`` with a
    7e-3 relative error while ``|Σ aMat|`` reports a ratio of 1
    (``tests/test_interaction_algebra.py``).
    """
    m1, m2 = g1.table_magnitude, g2.table_magnitude
    if m1 is None and m2 is None:
        return None
    out = 0.0
    if m1:
        out += float(m1) * (_coefficient_l1(g2) if partner_sup is None
                            else float(partner_sup[1]))
    if m2:
        out += float(m2) * (_coefficient_l1(g1) if partner_sup is None
                            else float(partner_sup[0]))
    return out


def graph_zero_conditioned(g: GraphZeta) -> tuple[float, float]:
    r"""Return ``(value, condition)`` for the graph zeta at ``k = 0``.

    ``value`` is exactly :func:`graph_zero`.  ``condition`` is the
    cancellation ratio of that evaluation,

    .. math::

        \kappa = \frac{\sum_i |b_i Z_{\nu_i}(0)| + |\sum_x a(x)|}
                      {\left|\sum_i b_i Z_{\nu_i}(0) + \sum_x a(x)\right|},

    i.e. the sum of the term magnitudes over the magnitude of their sum.
    Since every term is held in float64, the achievable relative accuracy
    of the result is bounded below by ``eps * kappa``; the empirical law
    ``rel_err ~ 1e-16 * kappa`` was measured to hold within one order
    across cycles (L = 3..17) and theta graphs at ``sigma`` from 0.01 to
    1.4 — it is a calibrated error bar, not a heuristic.

    The algebra loses conditioning when a deep chain crowds many
    exponents toward the convergence boundary ``nu -> d``: the chain
    itself stays healthy, but the final convolution shifts every exponent
    up by one edge weight, collapsing the spread of the Epstein weights
    that had been holding the alternating-sign coefficients in check.
    At ``sigma = 0.02`` with a 12-edge chain the ratio reaches 2e14 and
    the answer is destroyed; extended-precision prefactors do not help
    (the loss is carried by the float64 coefficient and Epstein arrays
    alike), and lowering ``sigma_max`` is far worse.  Detecting the
    condition and routing elsewhere is therefore the available remedy —
    see :func:`gzl.evaluate_graph`.

    No FFT is needed: at ``k = 0`` the Fourier part evaluates to
    ``sum(aMat)``.

    **The kernel path.**  The formula above folds the whole ``aMat`` into
    ONE term, ``|Σ_x a(x)|``, so a compact interaction table whose
    entries cancel inside that ``np.sum`` loses digits the ratio cannot
    see: a bridge carrying ``a(±1) = c, a(±2) = −c, a(±3) = 0.1`` plus
    ``K_3`` on the chain is 1.8e-8 off at ``c = 1e9`` (1.9e-5 at
    ``1e12``) while ``κ`` reports exactly 1.  An object built from
    interaction kernels therefore carries
    :attr:`GraphZeta.table_magnitude` ``M`` — an upper bound on the ℓ¹
    norm of the coefficients that descend from its tables, set to
    ``Σ_m |a(m)|`` at the leaf and propagated by
    :func:`_compose_table_magnitude` — and the Fourier term of the
    numerator becomes ``max(|Σ_x a(x)|, M)``.  The same bridge then
    reports ``κ ≈ 1.5e9`` and ``eps·κ`` is again an error bar.  Because
    ``M`` is a bound, ``κ`` on the kernel path is an UPPER bound on the
    TABLE-INDUCED part of the cancellation rather than the ratio itself
    (the power-law part's own coefficient cancellation stays folded into
    ``|Σa|`` exactly as on the float path, so against the honest ℓ¹ ratio
    ``κ`` can still sit up to ~20 % low on a two-bridge path at ν ≥ 2.5):
    never below the power-law formula, and loose by a factor ~2 per
    composition (the bound counts the table-table cross term of a product
    twice).  A
    power-law object (``table_magnitude is None``) computes exactly the
    formula above, byte for byte; a power-law *kernel* (an empty table,
    ``M = 0``) gives the same ``κ`` as the power-law path.
    """
    A, a, b, nu = g.A, g.aMat, g.bVec, g.nuVec
    d = A.shape[0]
    n = a.shape[0]

    A = np.ascontiguousarray(A, dtype=np.float64)
    A_bytes, A_shape = A.tobytes(), A.shape

    total = 0.0
    magnitude = 0.0
    for bi, nui in zip(b, nu):
        vals = _epstein_grid_cached(float(nui), A_bytes, A_shape, int(n))
        z0 = float(vals.reshape((n,) * d)[(0,) * d])
        term = float((bi * z0).real)
        total += term
        magnitude += abs(term)

    fourier = float(np.real(np.sum(a)))
    total += fourier
    if g.table_magnitude is None:
        magnitude += abs(fourier)
    else:
        # Kernel path: the tracked table bound stands in for the one term
        # the power-law formula folds aMat into, so a table cancelling
        # inside np.sum(a) is seen (see the docstring; the value is the
        # same either way).
        magnitude += max(abs(fourier), float(g.table_magnitude))

    if total == 0.0:
        return 0.0, float("inf")
    return total, magnitude / abs(total)


def _balanced_z_axis(n: int) -> np.ndarray:
    """Local copy of ``gzl.tensor_network._balanced_z_axis``.

    Returns ``[0, 1, …, n//2, -((n + 1) // 2) + 1, …, -1]`` — the
    signed lattice positions matching the layout that
    :func:`graph_compress` produces in ``aMat`` (positive labels first,
    negative-equivalent labels on the upper half of the index range).
    Used by :func:`graph_sample_at` to read out ζ_G at off-grid k
    without breaking lattice inversion symmetry.
    """
    pos = np.arange(0, n // 2 + 1, dtype=int)
    neg = np.arange(-((n + 1) // 2) + 1, 0, dtype=int)
    return np.concatenate([pos, neg])


def graph_sample_at(g: GraphZeta, k_frac) -> float:
    r"""Evaluate the graph zeta function at a single momentum.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Same value convention as :func:`graph_sample` (and as
    :func:`graph_zeta_general` with ``space='k'``): ``k_frac`` is in
    fractional Brillouin-zone coordinates :math:`[0, 1)^d`, so that
    ``graph_sample(g)[k_idx]`` equals
    ``graph_sample_at(g, k_idx / n)`` for any integer multi-index
    ``k_idx``.

    Cost is :math:`\mathcal{O}(n_\text{points}^d + N_\text{singular})`
    — one inner product over the regular Fourier coefficients plus one
    Epstein-zeta evaluation per singular term — versus
    :math:`\mathcal{O}(N_\text{singular} \cdot n^d)` for the full-grid
    :func:`graph_sample`.  Useful when the caller wants ζ_G at a few
    specific k's rather than the whole BZ grid.

    Parameters
    ----------
    g : GraphZeta
        Built algebra object.
    k_frac : array_like
        Length-``d`` fractional momentum, ``k_frac[i] in [0, 1)``.
        For ``d == 1`` a Python scalar is also accepted.

    ``k_frac`` need **not** lie on the BZ grid.  The regular part is
    summed over the *balanced* z-axis, i.e. as a real-space sum over the
    centred window ``[-n/2, n/2]^d``, so an off-grid k is evaluated at
    the ordinary truncation accuracy of that window rather than by
    trigonometric interpolation between grid nodes — the value converges
    in ``n_points`` and retains lattice inversion symmetry.  (Weighting
    the same array by *unsigned* indices ``0..n-1`` instead would attach
    a spurious ``exp(2πi k n)`` to every negative position: invisible at
    on-grid ``k = j / n``, badly aliased off it.)

    Returns
    -------
    float
        ζ_G at ``k_frac``, real-valued by lattice inversion symmetry.
    """
    A, a, b, nu = g.A, g.aMat, g.bVec, g.nuVec
    d = A.shape[0]
    n = a.shape[0]
    Astar = np.linalg.inv(A.T)
    k_arr = np.asarray(k_frac, dtype=float).reshape(-1)
    if k_arr.size != d:
        raise ValueError(
            f"graph_sample_at: expected k_frac of length {d}, got "
            f"shape {np.asarray(k_frac).shape}."
        )
    k_lattice = Astar @ k_arr

    out = 0.0
    # Singular contributions: each (b_i, ν_i) contributes
    # b_i · Z_{Λ, ν_i}(k_lattice), its cusp included at a tiny k (see
    # _epstein_zeta_k).
    vol = abs(float(np.linalg.det(A)))
    for bi, nui in zip(b, nu):
        ez = _epstein_zeta_k(float(nui), A, k_lattice, vol)
        out += (bi * ez).real

    # Regular Fourier contribution: Σ_z a[z] · exp(2πi k_frac · z),
    # using the balanced-z-axis layout so off-grid k retains lattice
    # inversion symmetry (real-valued result).  Matches graph_sample's
    # convention at on-grid k where the layouts coincide.
    if a.size:
        z_axis = _balanced_z_axis(n).astype(float)
        z_grid = np.meshgrid(*[z_axis for _ in range(d)], indexing="ij")
        k_dot_z = sum(z_grid[i] * float(k_arr[i]) for i in range(d))
        phase = np.exp(2j * np.pi * k_dot_z)
        out += np.real(np.sum(a * phase))

    return float(out)


# ---------------------------------------------------------------------------
# Internal: merge duplicate exponents
# ---------------------------------------------------------------------------

_ZERO_COEFF_TOL: float = 1e-16

# FP tolerance for "ν is the integer round(ν)" recognition.  Used (a) inside
# _mult_prefactor to force structural zeros at non-positive integer arguments
# of rgamma even when float64 arithmetic puts us 1–2 ulps off, and (b) in
# graph_multiply's cross-exponent assembly to snap cross_nu = nu1 + nu2 - d
# to an exact integer when within tolerance, preventing FP-perturbed copies
# of the same ν from accumulating in nuVec across long chains.
_FP_INTEGER_TOL: float = 1e-10

# --- degenerate-channel (sigma-resonance) regularisation -------------------
#
# A series cross term whose exponent nu_c = nu1 + nu2 - d lands on the
# Gamma((d - nu)/2) pole family nu_c = d + 2n (n >= 1) is a 0 x inf limit:
# its multiplication prefactor has a structural ZERO (rgamma(w) at w = -n),
# while the *next* multiplication's Gamma((d - nu_c)/2) has a POLE there.
# The product is finite — the O(1) exponent tower above d + 2n is
# delta-independent — so dropping the near-pole entry (merge-drop at the
# exact zero, or the compress_singularities band otherwise) deletes the
# whole tower and relocates it into aMat as a slowly decaying |x|^-(d+2n)
# tail; the final convolution then loses ~3 orders of accuracy through
# window truncation amplified by cancellation (the measured sigma
# resonance at m*sigma in 2N).
#
# Fix: degenerate channels are emitted at the CANONICAL exponent
# d + 2n + _POLE_CANONICAL_OFFSET with the residue-limit prefactor
# (_mult_prefactor_degenerate).  The tiny canonical entry regenerates the
# tower through the finite 1/offset pole at the next multiply, and the
# on-grid diff correction in graph_multiply absorbs the O(offset) smooth
# remainder.  graph_compress(compress_singularities=True) exempts exactly
# these canonical exponents, while genuine near-pole *inputs* with O(1)
# coefficients (a sigma = 2 edge, a Hadamard bundle at sigma = k - d/2,
# a parallel composition whose exponents sum to d + 2n) keep the band
# absorption.
#
# That absorption is accurate only when a convolution follows.  It puts
# the kernel into aMat truncated to the n-point window, so a value
# sampled from the product misses the tail sum_{|x| > n/2} |x|^-(d+2n),
# O(n^-2n).  A later graph_convolve weights that loss by the partner
# kernel and suppresses it to the algebra's truncation order.  Inside a
# 2-connected block every serial composition is closed by a parallel
# one, which is why the cycles, the butterfly and the blocks
# evaluate_graph routes here stay at background (measured at sigma = 0.5
# on the chain, k = 0 and 0.3, on the blocks T^2 and T^3 of the
# mathematics paper and on a triangle in series with a two-edge path,
# closed by an edge).  A product sampled directly does not: triangle
# times edge at nu = 1.5 on the chain measures 1.7e-4 at n = 250, n^-2,
# against 1.4e-8 at nu = 1.5 + pi/30.  Absorbing with the exact grid
# samples of Z_nu instead (periodised coefficients) repairs that product
# but breaks the convolution, whose real-space window then sees the
# images: the 13-cycle at sigma = 2 goes from 8e-8 to 9e-5 at n = 124.
# The candidate fix keeps such an entry analytic and takes the Gamma
# pole of its cross terms as a limit; it is neither implemented nor
# measured.
#
# Validated operating range: the finite route is flat at background error
# for detunings 3e-10 <= |delta| <= 1e-1 (no catastrophic cancellation),
# so any offset well inside that window works; 1e-6 keeps the induced
# redistribution O(offset * channel mass) far below the truncation floor.
_POLE_BIRTH_WINDOW: float = 1e-4
_POLE_CANONICAL_OFFSET: float = 1e-6
# Exact-match tolerance for recognising canonical exponents in
# graph_compress; canonical values are constructed by the same float
# expression everywhere, so this only needs to cover identity plus dust.
_POLE_CANONICAL_MATCH_TOL: float = 1e-12

# --- coefficient-gated absorption ring (the pole "flank") ------------------
#
# Outside the birth window the same Gamma((d - nu)/2) pole still amplifies:
# an exponent sitting a distance delta from d + 2k with an O(1) coefficient
# forces the ill-conditioned reg/singular split of that Epstein term through
# the multiplication prefactor (c_nu ~ R/delta), so the exactly-evaluated
# analytic part and the grid-TRUNCATED Fourier part each carry ~1/delta and
# must cancel.  The measured cost is ~C(n)/|delta| relative (worst known
# instance: a 6-cycle with one chord at sigma near 1/2, 15.5% at
# delta_sigma = 5.2e-5, C dying as n^-3.4).  The real-space kernel
# |x|^-nu has NO pole in nu, so absorbing such an entry removes the
# amplification entirely: measured 8.1e-2 -> 3.8e-5 at the window edge and
# still a net win out to |delta| ~ 0.3.
#
# The catch — and why this is gated, not a wider window: the SAME ring
# contains the degenerate-birth GENERATORS, whose coefficient is
# proportional to their own detuning and whose 1/delta pole at the next
# multiply is what regenerates the exponent tower.  Absorbing one of those
# is the sigma-resonance defect again: ungated absorption over this ring
# measures 23% error on a 13-cycle (worse than the original resonance).
#
# The two are separated by the entry's share of its own representation,
#
#     share_i = |b_i| / max_j |b_j|,
#
# which is scale-invariant, together with that share's relation to the
# detuning.  A generator's coefficient is proportional to its own
# detuning, so share ~ O(dist) and share/dist is an O(1) constant; a
# flank source carries a full-weight coefficient, so share ~ O(1) and
# share/dist grows like 1/dist.  Absorption therefore demands BOTH
#
#     share >= _POLE_ABSORB_SHARE          (the entry dominates), and
#     share >= _POLE_ABSORB_RATIO * dist   (it dominates by more than
#                                           its detuning explains),
#
# so a generator has to defeat two independent guards to be absorbed.
#
# Thresholds come from a labelled measurement over pure-generator blocks
# (cycles, whose only convolution is the last operation, with every edge
# exponent held >= 0.35 from the pole family, so every near-pole entry is
# a cross birth): L = 4..14, sigma grid 0.001, 24 415 samples at k = 1.
# Measured generator ceiling there: share <= 0.435 and share/dist <= 3.40.
# Real flank sources measure share 0.65 .. 1.00.  The shipped 0.5 / 12.0
# clear the generator ceiling by 1.15x and 3.5x while still absorbing
# every measured flank source out to dist ~ 0.05 (share 0.65) or ~ 0.08
# (share 1.0).
#
# k matters, and the asymmetry is structural: a generator at nu = d + 2k
# only matters if the tower it seeds survives compression, i.e. only if
# d + 2k < d + sigma_max.  At the production sigma_max = 4 that is k = 1
# alone; k = 2 generators sit at nu = d + 4 = d + sigma_max, so their
# tower is discarded regardless and absorbing one is measured harmless
# (1.03x on the worst case found, L = 14 at sigma = 1.3325).  The
# thresholds above are therefore set from the k = 1 population, which is
# the one that can actually be damaged.
#
# This also restores the invariant the theory assumes: the closure result
# for this algebra guarantees "every retained exponent lies outside
# d + 2N" only for sigma_max < 2, whereas production runs sigma_max = 4,
# which puts nu = d + 2 inside the retained range.  The ring is what
# repairs that, without giving up the accuracy sigma_max = 4 buys.
#
# The gate is self-limiting in the ring width: an entry stops qualifying
# once its share falls below ratio * dist, so widening the ring past the
# point where real flank sources drop out changes nothing.  Measured on
# the worst known flank block, absorption engages out to dist ~ 0.13 and
# then stops on its own; the 13-cycle generator stays bit-identical at
# every ring width tried, up to 0.30.  0.15 is the width at which the
# measured benefit saturates.
#
# RATIO is set by where absorption reliably PAYS, which is a separate
# question from where it is safe.  Absorbing costs the absorbed kernel's
# own truncation tail on the finite window; keeping the entry costs the
# ~1/dist amplification.  Close to the pole the amplification dominates
# by orders of magnitude; far from it the two are comparable and the
# outcome depends on the rest of the block, not on the entry.  Measured
# (16 flank blocks from the TFIM corpora, improvement = error without
# ring / error with ring, vs an independent tensor Aitken ladder to
# n = 8192):
#
#   dist    geo-mean   worst    blocks made worse by >1.5x
#   0.0005    23.7      1.00     0 / 16
#   0.002      9.3      1.00     0 / 16
#   0.005      5.0      0.87     0 / 16
#   0.010      2.7      1.00     0 / 16   (1 / 15 at n = 256)
#   0.020      1.8      0.21     1 / 6    <- lottery starts
#   0.050      0.96     0.19     3 / 6    <- net LOSS
#
# There is no locally computable way to tell the 0.02+ winners from the
# losers: the same absorbed entry (same nu, dist, share, n) shows a 28x
# loss on one block and a 5.5x win on another, so a cost model built from
# (nu, dist, share, n) provably cannot separate them.  The ring is
# therefore cut back to the region that is uniformly positive.  RATIO =
# 100 caps a full-weight entry at dist <= 0.01 and the common share-0.65
# flank at dist <= 0.0065, which keeps every large win (the worst known
# flank, 1.55e-1, sits at dist ~ 1e-4) and drops the lottery entirely.
_POLE_ABSORB_RING: float = 0.15
_POLE_ABSORB_RATIO: float = 100.0
_POLE_ABSORB_SHARE: float = 0.5


def _gamma_stable_real(x: float) -> float:
    return float(gammasgn(x)) * float(np.exp(gammaln(float(x))))


def _import_mpmath():
    """mpmath for ``precision="mpmath"``, which gzl does not install."""
    try:
        import mpmath
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'precision="mpmath" needs mpmath, which is not a dependency of '
            "gzl: pip install mpmath", name="mpmath") from exc
    return mpmath


def _mult_prefactor(
    nu1: float,
    nu2: float,
    d: int,
    vol: float,
    *,
    precision: str = "float64",
    fp_tol: float = _FP_INTEGER_TOL,
) -> float:
    """Multiplication prefactor for the Symanzik chain identity.

    Returns

        (π^(d/2) / vol) · Γ(a) Γ(b) Γ(c) / [Γ(u) Γ(v) Γ(w)]

    with ``a = (d−ν1)/2``, ``b = (d−ν2)/2``, ``c = (ν1+ν2−d)/2``,
    ``u = ν1/2``, ``v = ν2/2``, ``w = (2d−ν1−ν2)/2``.

    Three structural zeros (``rgamma(z) = 0`` at non-positive integer ``z``)
    can fire — at ``u``, ``v``, or ``w``.  In float64, FP perturbation of
    the input ν puts the argument 1–2 ulps off the integer; ``rgamma``
    then returns ~1e−16 instead of exactly 0, and that residue gets
    multiplied by chain-accumulated coefficients into spurious near-pole
    entries.  We snap to exact zero when within ``fp_tol`` of a non-positive
    integer.

    Parameters
    ----------
    precision : {"float64", "mpmath"}
        ``"float64"`` (default): standard scipy.special path.
        ``"mpmath"``: compute the prefactor in arbitrary precision (50 dps)
        using mpmath; the structural-zero snap still fires (now at the
        same fp_tol but evaluated against the mpmath ν).  Slower per call;
        intended for chains where prefactor accumulation matters more
        than per-multiply throughput.
    """
    a = 0.5 * (d - nu1)
    b = 0.5 * (d - nu2)
    c = 0.5 * (nu1 + nu2 - d)
    u = 0.5 * nu1
    v = 0.5 * nu2
    w = 0.5 * (2.0 * d - nu1 - nu2)

    # Structural-zero detection (applies to both float64 and mpmath paths):
    # rgamma is zero at non-positive integers; FP-near-integer arguments
    # that should give zero must be forced to zero rather than letting the
    # ulps-off residue propagate.
    if (
        _is_nonpositive_integer(u, fp_tol)
        or _is_nonpositive_integer(v, fp_tol)
        or _is_nonpositive_integer(w, fp_tol)
    ):
        return 0.0

    if precision == "mpmath":
        mpmath = _import_mpmath()
        with mpmath.workdps(50):
            mp = mpmath.mpf
            d_mp = mp(d)
            nu1_mp = mp(nu1)
            nu2_mp = mp(nu2)
            a_mp = (d_mp - nu1_mp) / 2
            b_mp = (d_mp - nu2_mp) / 2
            c_mp = (nu1_mp + nu2_mp - d_mp) / 2
            u_mp = nu1_mp / 2
            v_mp = nu2_mp / 2
            w_mp = (2 * d_mp - nu1_mp - nu2_mp) / 2
            result = (
                mpmath.pi ** (d_mp / 2)
                / mp(vol)
                * mpmath.gamma(a_mp)
                * mpmath.gamma(b_mp)
                * mpmath.gamma(c_mp)
                * mpmath.rgamma(u_mp)
                * mpmath.rgamma(v_mp)
                * mpmath.rgamma(w_mp)
            )
            return float(result)

    if precision != "float64":
        raise ValueError(
            f"_mult_prefactor: precision must be 'float64' or 'mpmath' "
            f"(got {precision!r})."
        )

    return float(
        (np.pi ** (d / 2.0))
        / vol
        * _gamma_stable_real(a)
        * _gamma_stable_real(b)
        * _gamma_stable_real(c)
        * rgamma(u)
        * rgamma(v)
        * rgamma(w)
    )


def _mult_prefactor_degenerate(
    nu1: float,
    n: int,
    d: int,
    vol: float,
    *,
    precision: str = "float64",
) -> float:
    """Residue-limit prefactor for a degenerate cross channel.

    For a channel with ``nu1 + nu2 = 2d + 2n`` (n >= 1) the ordinary
    :func:`_mult_prefactor` vanishes linearly in the detuning
    ``tau = nu1 + nu2 - (2d + 2n)`` (structural zero of ``rgamma(w)`` at
    ``w = -n``).  The canonical representative emitted at
    ``nu_c = d + 2n + _POLE_CANONICAL_OFFSET`` carries the limit slope
    times the offset::

        pf = pi^(d/2)/vol * G(a) G(b0) G(c0) / [G(u) G(v0)]
             * (-offset/2) * (-1)^n * n!

    with ``a = (d - nu1)/2``, ``b0 = (nu1 - d - 2n)/2`` (the b-argument at
    the pole-locked partner ``nu2 = 2d + 2n - nu1``), ``c0 = (d + 2n)/2``,
    ``u = nu1/2``, ``v0 = (2d + 2n - nu1)/2``, and the last factor the
    leading order of ``rgamma(-n - offset/2)``.

    The formula deliberately evaluates the Gamma factors at the
    pole-locked partner rather than at ``nu2 + offset``-shifted arguments:
    that keeps ``b0`` off exact non-positive integers even when ``nu1``
    itself is a canonical entry (e.g. the (d+2+off) x (d+2+off) channel,
    where a shifted evaluation would hit Gamma(-1) exactly).

    Returns 0.0 conservatively if ``a`` or ``b0`` lands on a Gamma pole
    (only possible if ``nu1`` sits exactly on the pole family, which the
    compression band removes before cross assembly).
    """
    off = _POLE_CANONICAL_OFFSET
    a = 0.5 * (d - nu1)
    b0 = 0.5 * (nu1 - d) - n
    c0 = 0.5 * d + n
    u = 0.5 * nu1
    v0 = 0.5 * (2.0 * d + 2.0 * n - nu1)

    if _is_nonpositive_integer(a, _FP_INTEGER_TOL) or _is_nonpositive_integer(
        b0, _FP_INTEGER_TOL
    ):
        return 0.0
    if _is_nonpositive_integer(u, _FP_INTEGER_TOL) or _is_nonpositive_integer(
        v0, _FP_INTEGER_TOL
    ):
        return 0.0

    if precision == "mpmath":
        mpmath = _import_mpmath()

        with mpmath.workdps(50):
            mp = mpmath.mpf
            d_mp = mp(d)
            nu1_mp = mp(nu1)
            n_mp = mp(n)
            result = (
                mpmath.pi ** (d_mp / 2)
                / mp(vol)
                * mpmath.gamma((d_mp - nu1_mp) / 2)
                * mpmath.gamma((nu1_mp - d_mp) / 2 - n_mp)
                * mpmath.gamma(d_mp / 2 + n_mp)
                * mpmath.rgamma(nu1_mp / 2)
                * mpmath.rgamma(d_mp + n_mp - nu1_mp / 2)
                * (-mp(off) / 2)
                * (-1) ** n
                * mpmath.factorial(n)
            )
            return float(result)

    if precision != "float64":
        raise ValueError(
            f"_mult_prefactor_degenerate: precision must be 'float64' or "
            f"'mpmath' (got {precision!r})."
        )

    return float(
        (np.pi ** (d / 2.0))
        / vol
        * _gamma_stable_real(a)
        * _gamma_stable_real(b0)
        * _gamma_stable_real(c0)
        * rgamma(u)
        * rgamma(v0)
        * (-0.5 * off)
        * ((-1.0) ** n)
        * float(factorial(n))
    )


def _cross_prefactor_and_exponent(
    nu1: float,
    nu2: float,
    d: int,
    vol: float,
    *,
    precision: str = "float64",
) -> tuple[float, float]:
    """Prefactor and exponent of one series cross term, pole-family aware.

    Non-degenerate channels reproduce the historical behaviour exactly
    (:func:`_mult_prefactor` plus the ``_FP_INTEGER_TOL`` integer snap on
    ``nu1 + nu2 - d``), bit for bit.  Channels whose exponent lands within
    ``_POLE_BIRTH_WINDOW`` of ``d + 2n`` are canonicalised: emitted at
    ``d + 2n + _POLE_CANONICAL_OFFSET`` with the residue-limit prefactor,
    so the finite 0 x inf tower limit survives (see the constants block).
    """
    s = nu1 + nu2
    n_near = int(round((s - 2.0 * d) / 2.0))
    if n_near >= 1 and abs(s - (2.0 * d + 2.0 * n_near)) < _POLE_BIRTH_WINDOW:
        nu_canon = (d + 2.0 * n_near) + _POLE_CANONICAL_OFFSET
        pf = _mult_prefactor_degenerate(nu1, n_near, d, vol, precision=precision)
        return pf, nu_canon

    pf = _mult_prefactor(nu1, nu2, d, vol, precision=precision)
    nu_c = s - d
    r = round(nu_c)
    if abs(nu_c - r) < _FP_INTEGER_TOL:
        nu_c = float(r)
    return pf, float(nu_c)


def _merge_duplicate_exponents(
    b_vec: np.ndarray, nu_vec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merge identical exponents by summing their coefficients.

    Uses exact float equality, with no tolerance, so only exponents that
    compare equal merge.  Entries whose merged coefficient has
    ``abs(b) < _ZERO_COEFF_TOL`` are dropped, so structurally vanishing
    terms (e.g. cross-products at ``nu1 + nu2 = 2*d + 2*n`` whose
    multiplication prefactor is zero) do not pollute ``nuVec`` and
    cannot trip ``_check_multiply_singularities`` on the next operation.
    """
    b_vec = np.asarray(b_vec, dtype=complex)
    nu_vec = np.asarray(nu_vec, dtype=float)

    if nu_vec.size == 0:
        return b_vec, nu_vec

    b_out: list[complex] = []
    nu_out: list[float] = []
    index: dict[float, int] = {}

    for bi, nui in zip(b_vec, nu_vec):
        key = float(nui)
        if key in index:
            b_out[index[key]] += bi
        else:
            index[key] = len(b_out)
            nu_out.append(key)
            b_out.append(complex(bi))

    keep = [i for i, bi in enumerate(b_out) if abs(bi) >= _ZERO_COEFF_TOL]
    if len(keep) == len(b_out):
        return np.asarray(b_out, dtype=complex), np.asarray(nu_out, dtype=float)

    return (
        np.asarray([b_out[i] for i in keep], dtype=complex),
        np.asarray([nu_out[i] for i in keep], dtype=float),
    )


# ---------------------------------------------------------------------------
# Compress
# ---------------------------------------------------------------------------

def graph_compress(
    g: GraphZeta,
    sigma_max: float,
    *,
    compress_singularities: bool = False,
    singularity_tol: float = 1e-4,
) -> GraphZeta:
    """Compress a graph by absorbing mild singularities into the Fourier part.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Exponents with ``nu > d + sigma_max`` are converted into their
    associated Fourier series (added to ``aMat``).  Stronger singularities
    (``nu <= d + sigma_max``) are kept analytically.

    Parameters
    ----------
    g : GraphZeta
        Input graph.
    sigma_max : float
        Singularity threshold:  ``nu > d + sigma_max`` is moved into
        ``aMat``; ``nu <= d + sigma_max`` is kept in ``(bVec, nuVec)``.
    compress_singularities : bool, default False
        When True, additionally absorb any exponent ``nu`` that lies
        within ``singularity_tol`` of the ``Gamma((d - nu)/2)`` pole
        family ``{d + 2k : k = 1, 2, 3, ...}`` into ``aMat`` —
        regardless of whether it sits below or above ``d + sigma_max``.
        This bypasses the pole that ``_check_multiply_singularities``
        flags for inputs of :func:`graph_multiply`, by ensuring no
        such ``nu`` reaches the multiplication prefactor.

        Canonical-offset entries at ``d + 2k + _POLE_CANONICAL_OFFSET``
        (born by the degenerate-channel regularisation in
        :func:`graph_multiply`) are exempt: they carry the finite
        0 x inf limit of the exponent tower above the pole and must
        stay analytic.  See the ``_POLE_CANONICAL_OFFSET`` constants
        block for the full mechanism.

        Absorption additionally extends into a *ring* out to
        ``_POLE_ABSORB_RING``, but only for entries that dominate their
        own representation — coefficient share at least
        ``_POLE_ABSORB_SHARE``, and at least ``_POLE_ABSORB_RATIO``
        times their distance from the pole.  Those are the
        O(1)-coefficient entries the pole amplifies by ~1/distance (the
        "flank").  Degenerate-birth generators, whose coefficient is
        proportional to that same distance, fail both tests and are
        kept.  See the ``_POLE_ABSORB_RING`` constants block.

        The ``k = 0`` pole at ``nu = d`` (the convergence boundary)
        is intentionally excluded: physical inputs satisfy ``nu > d``
        strictly, so the boundary is never hit; conversely the band
        around it is exactly where the σ_max algebra excels, so
        absorbing there would lose precision unnecessarily.

        Note: this addresses only the input pole family
        ``nu = d + 2n`` (n >= 1); the cross-sum pole
        ``nu1 + nu2 = d - 2n`` cannot occur for physical inputs
        (``nu1, nu2 > d`` implies ``nu1 + nu2 > 2*d``), and the
        cross-sum *zero* ``nu1 + nu2 = 2*d + 2n`` is handled
        separately by the zero-coefficient filter inside
        :func:`_merge_duplicate_exponents`.
    singularity_tol : float, default 1e-4
        Distance threshold for the pole-class detection.
    """
    A, a, b, nu = g.A, g.aMat, g.bVec, g.nuVec
    d = A.shape[0]
    n = a.shape[0]

    thr = float(d + sigma_max)

    drop_mask = nu > thr

    if compress_singularities and nu.size > 0:
        # Identify entries close to nu = d + 2k (k = 1, 2, 3, ...).
        # k = 0 (nu = d, convergence boundary) is intentionally excluded:
        # physical inputs satisfy nu > d strictly, and that band is
        # exactly where the sigma_max algebra is most useful.
        k_nearest = np.round((nu - d) / 2.0)
        dist = np.abs(nu - (d + 2.0 * k_nearest))
        near_family = k_nearest >= 1
        pole_mask = near_family & (dist < singularity_tol)

        # Coefficient-gated absorption ring: outside the birth window the
        # pole still amplifies an O(1)-coefficient entry by ~1/dist (the
        # "flank"), while a degenerate-birth generator — coefficient
        # proportional to its own detuning — must be KEPT so the next
        # multiplication's pole can regenerate its exponent tower.  The
        # scale-invariant ratio below separates them; see the
        # _POLE_ABSORB_RING constants block for the measured margins.
        absb = np.abs(b)
        scale = float(absb.max()) if absb.size else 0.0
        if scale > 0.0:
            share = absb / scale
            ring_mask = (
                near_family
                & (dist >= singularity_tol)
                & (dist < _POLE_ABSORB_RING)
                & (share >= _POLE_ABSORB_SHARE)
                & (share >= _POLE_ABSORB_RATIO * dist)
            )
        else:
            ring_mask = np.zeros(nu.shape, dtype=bool)

        # Canonical-offset entries born by the degenerate-channel
        # regularisation in graph_multiply carry the finite 0 x inf tower
        # limit and must stay analytic; everything else in the band
        # (a leaf at sigma = 2, a Hadamard bundle at sigma = k - d/2)
        # keeps the absorption, which is accurate for O(1)-coefficient
        # inputs only when a convolution follows the product.  See the
        # _POLE_CANONICAL_OFFSET constants block.
        canon_mask = (
            np.abs(nu - (d + 2.0 * k_nearest + _POLE_CANONICAL_OFFSET))
            < _POLE_CANONICAL_MATCH_TOL
        )
        drop_mask = drop_mask | ((pole_mask | ring_mask) & ~canon_mask)

    keep_mask = ~drop_mask

    keep_idx = np.where(keep_mask)[0]
    drop_idx = np.where(drop_mask)[0]

    add_to_a = np.zeros_like(a, dtype=complex)

    if drop_idx.size > 0:
        pos = np.arange(0, int(np.floor(n / 2)) + 1, dtype=int)
        neg = np.arange(-int(np.ceil(n / 2)) + 1, 0, dtype=int)
        z_axis = np.concatenate([pos, neg])

        z_grids = np.meshgrid(*([z_axis] * d), indexing="ij")
        z_array = np.stack(z_grids, axis=-1)
        Az = np.tensordot(z_array, A.T, axes=([d], [0]))

        for idx in drop_idx:
            add_to_a += b[idx] * _vint_array(Az, float(nu[idx]))

    a_new = a + add_to_a

    if keep_idx.size == 0:
        return make_graph_obj(
            a_new, np.array([], dtype=complex), np.array([], dtype=float), A,
            table_magnitude=g.table_magnitude,
        )

    b_new, nu_new = _merge_duplicate_exponents(b[keep_idx], nu[keep_idx])
    return make_graph_obj(a_new, b_new, nu_new, A,
                          table_magnitude=g.table_magnitude)


# ---------------------------------------------------------------------------
# Multiply
# ---------------------------------------------------------------------------

class PrefactorSingularityError(GraphZetaError):
    """Raised when ``graph_multiply`` encounters a singularity in the
    multiplication prefactor.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    The prefactor contains ``Gamma((d - nu) / 2)`` for each exponent nu
    kept in the graph.  This Gamma function has poles whenever
    ``(d - nu) / 2`` is a non-positive integer, i.e. when ``nu = d + 2n``
    for ``n = 0, 1, 2, …``

    For ``n >= 1`` this is never raised: :func:`graph_multiply` absorbs
    near-pole inputs with ``graph_compress(compress_singularities=True)``
    and canonicalises degenerate cross terms (see the
    ``_POLE_CANONICAL_OFFSET`` constants block).  The guard fires only
    for exponents at or below ``d``: ``nu = d``, the ``n = 0`` pole, or a
    pair with ``nu1 + nu2 = d - 2n``, where the third factor
    ``Gamma((nu1 + nu2 - d) / 2)`` has its poles.
    """


def _check_multiply_singularities(
    nu1: np.ndarray, nu2: np.ndarray, d: int, tol: float = 1e-10
) -> None:
    """Raise :class:`PrefactorSingularityError` if any (nu1_i, nu2_j) pair
    would produce a pole in the multiplication prefactor.

    The three Gamma arguments that can become non-positive integers are::

        a = (d - nu1_i) / 2      (pole when nu1_i = d + 2n)
        b = (d - nu2_j) / 2      (pole when nu2_j = d + 2n)
        c = (nu1_i + nu2_j - d) / 2   (pole when nu1_i + nu2_j = d - 2n)
    """
    singular: list[str] = []

    for nui in nu1:
        a = 0.5 * (d - float(nui))
        if _is_nonpositive_integer(a, tol):
            n = int(round(-a))
            singular.append(
                f"nu1 = {nui} hits Gamma((d - nu)/2) pole "
                f"(nu = d + 2*{n} = {d} + {2 * n})"
            )

    for nuj in nu2:
        b = 0.5 * (d - float(nuj))
        if _is_nonpositive_integer(b, tol):
            n = int(round(-b))
            singular.append(
                f"nu2 = {nuj} hits Gamma((d - nu)/2) pole "
                f"(nu = d + 2*{n} = {d} + {2 * n})"
            )

    for nui in nu1:
        for nuj in nu2:
            c = 0.5 * (float(nui) + float(nuj) - d)
            if _is_nonpositive_integer(c, tol):
                n = int(round(-c))
                singular.append(
                    f"nu1 + nu2 = {nui} + {nuj} = {nui + nuj} hits "
                    f"Gamma((nu1 + nu2 - d)/2) pole "
                    f"(nu1 + nu2 = d - 2*{n} = {d - 2 * n})"
                )

    if singular:
        details = "\n  ".join(singular)
        raise PrefactorSingularityError(
            f"graph_multiply: prefactor singularity detected (d = {d}).\n"
            f"  {details}\n"
            f"Shift exponents by a small irrational offset (e.g. pi/30) "
            f"to avoid nu = d + 2n."
        )


def graph_multiply(
    g1: GraphZeta,
    g2: GraphZeta,
    sigma_max: float = 4.0,
    *,
    precision: str = "float64",
) -> GraphZeta:
    """Pointwise multiplication of two graph zeta functions.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Parameters
    ----------
    g1, g2 : GraphZeta
        Operands.  Must share the same lattice matrix ``A`` and the same
        number of discretisation points.
    sigma_max : float
        Singularity threshold for :func:`graph_compress`.
    precision : {"float64", "mpmath"}, default ``"float64"``
        Working precision for the multiplication prefactor.  ``"mpmath"``
        evaluates the prefactor at 50 dps, providing extra resolution for
        long chains where the prefactor accumulates near a structural
        zero.  ν values are still stored as float64 in :class:`GraphZeta`,
        so the benefit is bounded by the input ν precision; the practical
        gain is in the prefactor's intermediate Γ products.  Needs mpmath,
        which is not a dependency of gzl (``pip install mpmath``).

    Returns
    -------
    GraphZeta

    Raises
    ------
    PrefactorSingularityError
        If an exponent at or below ``d`` hits a pole of the
        Gamma-function prefactor: ``nu = d``, or a pair with
        ``nu1 + nu2 = d - 2n``.  Exponents on ``nu = d + 2n`` with
        ``n >= 1`` do not raise; they are absorbed into the Fourier part
        (see Notes).
    ValueError
        If lattice matrices or grid shapes do not match.
    ModuleNotFoundError
        If ``precision="mpmath"`` and mpmath is not installed.

    Notes
    -----
    An operand exponent on or near ``nu = d + 2n`` (``n >= 1``) with an
    O(1) coefficient, such as an edge at ``sigma = 2`` or the parallel
    composition of two exponents that sum to ``d + 2n`` (two edges at
    ``sigma = 1/2`` on the chain), is absorbed into ``aMat`` with its
    kernel truncated to the window of the ``N``-point grid.  The
    truncated tail costs ``O(N^(-2n))`` in every value sampled from the
    product.  A later :func:`graph_convolve` suppresses that loss to the
    truncation order of the algebra, so blocks in which every serial
    composition is closed by a parallel one keep their accuracy; this
    covers :func:`gzl.evaluate_graph`, which factorises a graph into
    blocks.  A product that is sampled directly does not: triangle
    times edge at ``nu = 1.5`` on the chain has a relative error of
    1.7e-4 at ``N = 250``, against 1.4e-8 at ``nu = 1.5 + pi/30``.  For
    such chains, shift ``nu`` by an irrational offset of that size.
    Small offsets are no remedy: at ``nu = 1.5 + pi/300`` the open
    product recovers, but the closed block
    ``graph_convolve(graph_multiply(triangle, edge), edge)`` loses a
    factor 23 at ``N = 250`` against ``nu = 1.5`` through the flank of
    the pole (see the ``_POLE_ABSORB_RING`` constants block).
    Degenerate cross terms that land on the pole family are handled
    exactly (see the ``_POLE_CANONICAL_OFFSET`` constants block).
    """
    # --- validation ---
    if not np.allclose(g1.A, g2.A):
        raise ValueError("graph_multiply: lattice matrices A must match.")
    if g1.aMat.shape != g2.aMat.shape:
        raise ValueError("graph_multiply: sample grids must match.")

    # Pre-absorb the Gamma((d - nu)/2) pole family (nu = d + 2k) into aMat
    # so the multiplication prefactor on the analytic side cannot land on
    # those poles.  See graph_compress(compress_singularities=True).
    g1 = graph_compress(g1, sigma_max, compress_singularities=True)
    g2 = graph_compress(g2, sigma_max, compress_singularities=True)

    A = g1.A
    a1, b1, nu1 = g1.aMat, g1.bVec, g1.nuVec
    a2, b2, nu2 = g2.aMat, g2.bVec, g2.nuVec
    d = A.shape[0]

    # --- singularity guard ---
    _check_multiply_singularities(nu1, nu2, d)

    # Kernel path only (None on the power-law path, at no cost).
    table_mag = _compose_table_magnitude(g1, g2)

    vol = abs(np.linalg.det(A))
    n = a1.shape[0]

    s1 = graph_sample(g1)
    s2 = graph_sample(g2)

    k0 = (0,) * d
    g1k0 = s1[k0]
    g2k0 = s2[k0]

    cross_b_list: list[complex] = []
    cross_nu_list: list[float] = []
    for b1i, nu1i in zip(b1, nu1):
        for b2j, nu2j in zip(b2, nu2):
            pf, nu_c = _cross_prefactor_and_exponent(
                float(nu1i), float(nu2j), d, vol, precision=precision
            )
            cross_b_list.append(pf * b1i * b2j)
            cross_nu_list.append(nu_c)
    cross_b = np.asarray(cross_b_list, dtype=complex)
    cross_nu = np.asarray(cross_nu_list, dtype=float)

    b_new = np.concatenate([g2k0 * b1, g1k0 * b2, cross_b])
    nu_new = np.concatenate([nu1, nu2, cross_nu])

    a_new = a1 * g2k0 + a2 * g1k0
    g_start = make_graph_obj(a_new, b_new, nu_new, A)

    diff = s1 * s2 - graph_sample(g_start)
    a_new = a_new + fftn(diff) / (n**d)

    b_new, nu_new = _merge_duplicate_exponents(b_new, nu_new)
    return graph_compress(
        make_graph_obj(a_new, b_new, nu_new, A, table_magnitude=table_mag),
        sigma_max,
    )


# ---------------------------------------------------------------------------
# Power helpers
# ---------------------------------------------------------------------------

def _check_positive_int_exponent(n_exp, func_name: str) -> int:
    """Validate that ``n_exp`` is a positive Python integer.

    Accepts plain ``int`` and NumPy integer scalars; rejects floats (even if
    they happen to be whole-valued) to avoid silent surprises.
    """
    if isinstance(n_exp, bool) or not isinstance(n_exp, (int, np.integer)):
        raise TypeError(
            f"{func_name}: n_exp must be a positive int "
            f"(got {type(n_exp).__name__})."
        )
    n_int = int(n_exp)
    if n_int <= 0:
        raise ValueError(
            f"{func_name}: n_exp must be >= 1 (got {n_int})."
        )
    return n_int


def _binary_power(g, n, op, sigma_max, **op_kwargs):
    """Right-to-left exponentiation by squaring.

    Combines ``n`` copies of ``g`` under the binary operation ``op``
    (``graph_multiply`` or ``graph_convolve``) using
    ``floor(log2 n)`` squarings + ``popcount(n) - 1`` cross-combines,
    versus ``n - 1`` ops for the linear loop.

    Parameters
    ----------
    g : GraphZeta
    n : int
        Positive integer (already validated).
    op : callable
        Either :func:`graph_multiply` or :func:`graph_convolve`.
    sigma_max : float
    **op_kwargs
        Extra keyword arguments forwarded to every ``op`` call (e.g.
        ``precision="mpmath"`` for ``graph_multiply``).

    Notes
    -----
    Skips the final, unused ``op(base, base)`` once the bit-shift loop
    has consumed the last set bit of ``n``.
    """
    if n == 1:
        return g
    result = None
    base = g
    while n > 0:
        if n & 1:
            result = base if result is None else op(result, base, sigma_max, **op_kwargs)
        n >>= 1
        if n:
            base = op(base, base, sigma_max, **op_kwargs)
    return result


def _linear_power(g, n, op, sigma_max, **op_kwargs):
    """Reference linear loop (kept for the private ``_method='linear'`` path)."""
    out = g
    for _ in range(n - 1):
        out = op(out, g, sigma_max, **op_kwargs)
    return out


def graph_multiply_power(
    g: GraphZeta,
    n_exp: int,
    sigma_max: float = 4.0,
    *,
    precision: str = "float64",
    _method: str = "binary",
) -> GraphZeta:
    """Return the ``n_exp``-th pointwise power of *g* under :func:`graph_multiply`.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Combines ``n_exp`` copies of *g* using exponentiation by squaring
    (O(log n_exp) operations).  :func:`graph_multiply` is the *serial*
    composition of graphs, so for a single-edge Epstein graph this
    represents a serial chain of ``n_exp`` edges (a path graph with
    ``n_exp + 1`` nodes).

    Note
    ----
    ``n_exp`` counts edges, not nodes.  Building a chain with *N* nodes
    requires ``n_exp = N - 1``.

    Parameters
    ----------
    g : GraphZeta
        Input graph.
    n_exp : int
        Number of multiplicative factors.  Must be a positive integer
        (>= 1).  ``n_exp = 1`` returns *g* unchanged.
    sigma_max : float
        Singularity threshold forwarded to :func:`graph_multiply`.
    precision : {"float64", "mpmath"}, default ``"float64"``
        Forwarded to :func:`graph_multiply`.  ``"mpmath"`` needs mpmath,
        which is not a dependency of gzl (``pip install mpmath``).

    Raises
    ------
    TypeError
        If ``n_exp`` is not an integer.
    ValueError
        If ``n_exp < 1``.
    PrefactorSingularityError
        Propagated from :func:`graph_multiply`, which raises only for
        exponents at or below ``d``.

    Notes
    -----
    Exponents that the chain itself produces on the pole family
    ``ν = d + 2k`` are degenerate cross terms, which
    :func:`graph_multiply` handles exactly, so a chain of edges at
    ``sigma = 1/2`` needs no offset.  An input exponent on the pole
    family, an edge at ``sigma = 2``, is absorbed into the Fourier part
    with its window-truncated kernel instead (see the Notes of
    :func:`graph_multiply`): four edges at ``nu = 3`` on the chain have
    a relative error of 1.1e-4 at ``N = 250``, against 1.9e-7 at
    ``nu = 3 + pi/30``.  Shift such an input by an irrational offset of
    that size.

    Exponentiation by squaring is always used; pass
    ``_method='linear'`` for the linear loop.
    """
    n_int = _check_positive_int_exponent(n_exp, "graph_multiply_power")

    if _method == "binary":
        # No near-pole fallback to the linear schedule: graph_multiply
        # absorbs pole-family inputs (compress_singularities) and snaps
        # structural zeros in `_mult_prefactor`, and the linear loop is
        # the less accurate of the two (a longer floating-point chain).
        return _binary_power(g, n_int, graph_multiply, sigma_max,
                             precision=precision)
    elif _method == "linear":
        return _linear_power(g, n_int, graph_multiply, sigma_max,
                             precision=precision)
    raise ValueError(
        f"graph_multiply_power: _method must be 'binary' or 'linear' "
        f"(got {_method!r})."
    )


# ---------------------------------------------------------------------------
# Convolve
# ---------------------------------------------------------------------------

def graph_convolve(
    g1: GraphZeta,
    g2: GraphZeta,
    sigma_max: float = 4.0,
) -> GraphZeta:
    """Periodic convolution of two graph zeta functions.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Parameters
    ----------
    g1, g2 : GraphZeta
        Operands.  Must share the same lattice matrix ``A`` and the same
        number of discretisation points.
    sigma_max : float
        Singularity threshold for :func:`graph_compress`.

    Returns
    -------
    GraphZeta

    Raises
    ------
    ValueError
        If lattice matrices or grid shapes do not match.
    """
    if not np.allclose(g1.A, g2.A):
        raise ValueError("graph_convolve: lattice matrices A must match.")
    if g1.aMat.shape != g2.aMat.shape:
        raise ValueError("graph_convolve: sample grids must match.")

    g1 = graph_compress(g1, sigma_max)
    g2 = graph_compress(g2, sigma_max)

    # Kernel path only: the real-space window arrays c_i below are formed
    # only when an operand carries a table magnitude, and the power-law
    # arithmetic on a_new is untouched either way.
    kernel_path = (g1.table_magnitude is not None
                   or g2.table_magnitude is not None)

    A = g1.A
    a1, b1, nu1 = g1.aMat, g1.bVec, g1.nuVec
    a2, b2, nu2 = g2.aMat, g2.bVec, g2.nuVec
    d = A.shape[0]
    n = a1.shape[0]

    pos = np.arange(0, int(np.floor(n / 2)) + 1, dtype=int)
    neg = np.arange(-int(np.ceil(n / 2)) + 1, 0, dtype=int)
    z_axis = np.concatenate([pos, neg])

    z_grids = np.meshgrid(*([z_axis] * d), indexing="ij")
    z_array = np.stack(z_grids, axis=-1)
    Az = np.tensordot(z_array, A.T, axes=([d], [0]))

    a_new = a1 * a2
    c1 = a1.copy() if kernel_path else None
    c2 = a2.copy() if kernel_path else None

    for j in range(nu2.size):
        v = b2[j] * _vint_array(Az, float(nu2[j]))
        a_new = a_new + a1 * v
        if kernel_path:
            c2 = c2 + v

    for i in range(nu1.size):
        v = b1[i] * _vint_array(Az, float(nu1[i]))
        a_new = a_new + a2 * v
        if kernel_path:
            c1 = c1 + v

    table_mag = (
        _compose_table_magnitude(
            g1, g2,
            partner_sup=(float(np.max(np.abs(c1))), float(np.max(np.abs(c2)))),
        )
        if kernel_path else None
    )

    b_new = np.array(
        [b1[i] * b2[j] for i in range(nu1.size) for j in range(nu2.size)],
        dtype=complex,
    )
    nu_new = np.array(
        [nu1[i] + nu2[j] for i in range(nu1.size) for j in range(nu2.size)],
        dtype=float,
    )

    b_new, nu_new = _merge_duplicate_exponents(b_new, nu_new)
    return make_graph_obj(a_new, b_new, nu_new, A, table_magnitude=table_mag)


def graph_convolve_power(
    g: GraphZeta,
    n_exp: int,
    sigma_max: float = 4.0,
    *,
    _method: str = "binary",
) -> GraphZeta:
    """Return the ``n_exp``-fold convolution power of *g* under :func:`graph_convolve`.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Combines ``n_exp`` copies of *g* using exponentiation by squaring
    (O(log n_exp) operations).  :func:`graph_convolve` is the
    *parallel* composition of graphs, so for a single-edge Epstein graph
    this represents ``n_exp`` parallel edges between the same pair of
    nodes (a bundle).

    Parameters
    ----------
    g : GraphZeta
        Input graph.
    n_exp : int
        Number of convolutional factors.  Must be a positive integer
        (>= 1).  ``n_exp = 1`` returns *g* unchanged.
    sigma_max : float
        Singularity threshold forwarded to :func:`graph_convolve`.

    Raises
    ------
    TypeError
        If ``n_exp`` is not an integer.
    ValueError
        If ``n_exp < 1``.
    """
    n_int = _check_positive_int_exponent(n_exp, "graph_convolve_power")

    if _method == "binary":
        return _binary_power(g, n_int, graph_convolve, sigma_max)
    elif _method == "linear":
        return _linear_power(g, n_int, graph_convolve, sigma_max)
    raise ValueError(
        f"graph_convolve_power: _method must be 'binary' or 'linear' "
        f"(got {_method!r})."
    )


# ---------------------------------------------------------------------------
# 1-sum attachment
# ---------------------------------------------------------------------------

def graph_attach(g: GraphZeta, decoration: GraphZeta) -> GraphZeta:
    """1-sum attachment of *decoration* to *g* at a single shared vertex.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Realises the zeta-function identity

        zeta_out(k) = zeta_g(k) * zeta_decoration(0),

    i.e. a decoration glued to *g* at an arbitrary vertex contributes
    only its value at ``k = 0`` as a scalar; terminals and external
    momentum are inherited from *g*.  Since ``zeta_decoration(0)`` is
    a single number, repeated attachment of the same *decoration* to
    further graphs costs one scalar multiplication per attachment.

    Parameters
    ----------
    g : GraphZeta
        Host graph.  The result inherits its lattice ``A``, Fourier
        grid, terminal structure, and singular-exponent list.
    decoration : GraphZeta
        Graph to be attached at a single vertex.  Its terminals are
        forgotten; only its value at ``k = 0`` enters the result.
        Must share the same lattice matrix ``A`` as *g*, but may use
        a different discretisation grid (useful when finer accuracy
        for the attached scalar is desired).

    Returns
    -------
    GraphZeta

    Raises
    ------
    ValueError
        If the lattice matrices ``A`` of *g* and *decoration* differ.
    """
    if not np.allclose(g.A, decoration.A):
        raise ValueError(
            "graph_attach: lattice matrices A must match."
        )

    c = complex(graph_zero(decoration))
    return make_graph_obj(
        aMat=g.aMat * c,
        bVec=g.bVec * c,
        nuVec=g.nuVec,
        A=g.A,
        table_magnitude=(None if g.table_magnitude is None
                         else g.table_magnitude * abs(c)),
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def periodic_convolve_nd(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Periodic convolution of two sampled arrays via FFT.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.
    """
    f, g = np.asarray(f), np.asarray(g)
    if f.shape != g.shape:
        raise ValueError(
            f"Shapes must match, got {f.shape} vs {g.shape}."
        )
    return np.real(ifftn(fftn(f) * fftn(g))) / np.prod(f.shape)
