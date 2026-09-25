# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

"""
Computation of graph zeta functions: circle zeta in d=1,2,3
Andreas A. Buchheit, February 25 2025

Usage:
  from gzl import zeta_circle
  val = zeta_circle([2.1, 2.2, 2.3, 2.4], [[1, 0.1], [0, 1.1]])
"""

from __future__ import annotations

import cmath
import math
import numpy as np
import os
from collections import OrderedDict

from scipy.integrate import nquad
from scipy.fft import fftn, ifftn

# Install with: pip install epsteinlib
from epsteinlib import epstein_zeta

from gzl._errors import (
    GraphZetaError,
    UnsupportedLatticeSumError,
    UnsupportedRequestError,
)
from gzl._lattices import lattice_matrix
from gzl.interaction import is_interaction

#: The dimensions the cycle closed form is implemented in.  The router
#: sends a cycle in any other dimension to the σ-router.
_CLOSED_FORM_DIMS = (1, 2, 3)


# ----------------------------
# Quadrature data on [0, 1/2]
# ----------------------------

weight0Half = np.array([
    0.008779865082937965757958219034547945154926402319281819145374972549104080709452106431638027579239,
    0.02003952178994005245140831926571357739592444634864869130034976637239286861432198221209835085366,
    0.030379642671975796172353702268119156489167336422518668072768848135789935973131752700243425343932,
    0.039300791789548383642400484655960539151417009334330843742329260968692044092402411825459712607535,
    0.046384599369484453435429147531289259062230650734332914755008731267274587565881740884152288368595,
    0.051299615930323900991481016415304513927584765327354862929224322570841786206312565397411563671657,
    0.053815963365789447548969110829065008818749389513532200054944098135904696838498625845028014453647,
    0.0538159633657894475489691108290650088187493895135322000549440981359046968384986258450952714135325,
    0.051299615930323900991481016415304513927584765327354862929224322570841786206313200446397333244982,
    0.046384599369484453435429147531289259062230650734332914755008731267274587565883426615701780541431,
    0.039300791789548383642400484655960539151417009334330843742329260968692044092399154397257131136569,
    0.03037964267197579617235370226811915648916733642251866807276884813578993597313403998974996954461,
    0.020039521789940052451408319265713577395924446348648691300349766372392868614321481828491869342203,
    0.008779865082937965757958219034547945154926402319281819145374972549104080709451909165416116318964
], dtype=np.float64)

abscissa0Half = np.array([
    0.003429047825796915289600683323986799580977148190193529588982297055562611466877088065213447050318,
    0.017891279084106620665902215155531433880740197397540595320509388129467226613910055171924370212101,
    0.043199671232558751702551314337401259740074724631229704609822736439630042297492802476468698892584,
    0.078176773797078632462995049245166465615399696813207331095183377842953775954216699997736918927446,
    0.12118784091046147700867732036220283442277867935767349076212380772680376204198615873018260982168,
    0.170221907768027559891082043957881133291434699116539010850837453185666639871801827473659399871766,
    0.222986262823164084483438837445041313097012098631440610739794734672996617747067650875764981711886,
    0.277013737176835915516561162554958686902987901368559389260205265327003382252932920079756249882869,
    0.329778092231972440108917956042118866708565300883460989149162546814333360128198616070410171745438,
    0.378812159089538522991322679637797165577221320642326509237876192273196237958014077571175550971833,
    0.421823226202921367537004950754833534384600303186792668904816622157046224045783287393118092996889,
    0.45680032876744124829744868566259874025992527536877029539017726356036995770250818913474728114611,
    0.482108720915893379334097784844468566119259802602459404679490611870532773386090660517506890852789,
    0.496570952174203084710399316676013200419022851809806470411017702944437388533122976692485069762502
], dtype=np.float64)


# ----------------------------
# Product of Epstein zetas
# ----------------------------

def epstein_zeta_prod(nu_vec, A, A_star, y_vec, *, direct_1d: bool = False) -> float:
    """
    Product of Epstein zeta values for nu in nu_vec.

    ``epstein_zeta`` is evaluated once per *distinct* exponent and the
    result reused across equal exponents — a simple cycle has uniform
    ν, so this is a single ``epstein_zeta`` call instead of one per
    edge, and this routine is called once per quadrature node.  The
    product is the same left-to-right sequence of float64
    multiplications ``np.prod`` performs, so the value is unchanged
    bit-for-bit.

    ``direct_1d=True`` (d = 1 only; ``zeta_circle``'s retry) replaces a
    non-finite ``epstein_zeta`` value by :func:`_epstein_zeta_1d_direct`.
    """
    A = np.asarray(A, dtype=np.float64)
    A_star = np.asarray(A_star, dtype=np.float64)
    y_vec = np.asarray(y_vec, dtype=np.float64)
    zeros = np.zeros(len(A), dtype=np.float64)
    k = A_star @ y_vec

    cache: dict = {}
    prod = 1.0
    for nu in nu_vec:
        nu = float(nu)
        val = cache.get(nu)
        if val is None:
            val = float(epstein_zeta(nu, A, zeros, k).real)
            if direct_1d and not math.isfinite(val):
                val = _epstein_zeta_1d_direct(nu, A, y_vec[0])
            cache[nu] = val
        prod *= val
    return prod


#: Truncation tolerance of :func:`_epstein_zeta_1d_direct`, in units of
#: ``|a|^-nu``: ``L`` is the smallest integer whose tail bound
#: ``2 sum_{m > L} m^-nu <= 2 L^(1 - nu) / (nu - 1)`` is below it.  That is
#: under half an ulp of the leading coefficient 2 (``|S| <= 2 zeta(nu)``),
#: so the truncation is invisible in double precision.
_DIRECT_1D_TOL = 1e-18
#: Refuse rather than sum more terms than this.  Unreachable in practice:
#: the direct sum only runs where ``epstein_zeta`` overflowed, which needs
#: ``nu |log10 |y|| ~ 330`` (measured frontier), i.e. a large ``nu`` -- and
#: at ``nu = 25`` the rule above already stops at L = 6 (at ``nu = 4``,
#: 873 581; just below nu = 4 it refuses).
_DIRECT_1D_MAX_TERMS = 1_000_000


def _epstein_zeta_1d_direct(nu: float, A, y: float) -> float:
    r"""``zeta_E(nu, A, 0, A* y)`` at d = 1 by direct summation.

    On the chain ``A = (a)`` the Epstein zeta is a cosine series,

        zeta_E(nu, A, 0, A* y) = sum_{m != 0} exp(-2 pi i m y) |a m|^-nu
                               = |a|^-nu * 2 sum_{m >= 1} cos(2 pi m y) m^-nu

    (``_epstein_table_direct``'s convention at d = 1).  Measured against
    mpmath's generalised Clausen function (40 digits) at a in {1, 1.7, 0.6,
    2.3}, nu in {4, 5.5, 8, 11, 20, 30}, seven y each: within 2.7e-16 of
    the scale ``2 |a|^-nu``, where ``epstein_zeta`` is within 1.8e-15.

    WHY THIS EXISTS.  ``epstein_zeta`` returns ``nan+nanj`` at large
    ``nu`` and small ``|y|`` -- the overflow ``_epstein_table_direct``
    documents, frontier ``nu |log10 |y|| ~ 330`` at d = 1 -- and the
    adaptive d = 1 rule samples there: ``zeta_circle([121, 11, 11],
    eye(1))`` was NaN, and with it every corpus graph carrying such a
    bundle (a multiplicity-11 bundle at nu = 11 is ``121``).  The
    library's FINITE values are accurate right up to the frontier (unit
    chain, 400 log-spaced y in [1e-16, 1/2] against 40-digit sums: worst
    absolute deviation 3.1e-15 at nu = 121 and 6.7e-15 at nu = 300, on
    values of size up to 2), so this is a substitute for the non-finite nodes
    only, and only on a RETRY: ``zeta_circle`` first integrates exactly as
    before and reruns with ``direct_1d=True`` only when that returned a
    non-finite value.  The retry is not redundant with a per-node swap:
    QUADPACK can absorb a NaN node and still return a finite value
    (``[69, 1.5, 1.5]``: 1 NaN node of 693, roundoff flag, 1.1e-15 off),
    and every such value stays bit-identical this way.
    """
    nu = float(nu)
    if np.shape(A) != (1, 1):
        raise ValueError(f"_epstein_zeta_1d_direct: d = 1 only; got A={A!r}")
    if not nu > 1.0:
        raise ValueError(f"_epstein_zeta_1d_direct: needs nu > 1; got nu={nu}")
    log_L = np.log(2.0 / ((nu - 1.0) * _DIRECT_1D_TOL)) / (nu - 1.0)
    if log_L > np.log(_DIRECT_1D_MAX_TERMS):
        raise CycleQuadratureError(
            f"_epstein_zeta_1d_direct: nu={nu} needs more than "
            f"{_DIRECT_1D_MAX_TERMS:,} terms for a {_DIRECT_1D_TOL:g} tail."
        )
    L = max(1, int(np.ceil(np.exp(log_L))))
    while 2.0 * float(L) ** (1.0 - nu) / (nu - 1.0) > _DIRECT_1D_TOL:
        L += 1
    m = np.arange(1, L + 1, dtype=np.float64)
    # fsum, not a dot: the terms span many orders, and at moderate nu
    # (large L) a plain dot loses ~3e-14 of the scale to accumulation.
    s = 2.0 * math.fsum(np.cos(2.0 * np.pi * m * float(y)) * m ** (-nu))
    return float(np.abs(np.float64(np.asarray(A).reshape(-1)[0])) ** (-nu) * s)


# ----------------------------
# Integrals in 1D / 2D / 3D
# ----------------------------

def int_full_1d(f, epsabs: float = 1e-12) -> float:
    """``2 int_0^{1/2} f``, the d = 1 Brillouin-zone integral of an even
    integrand.  ``epsabs`` is an ABSOLUTE tolerance: callers whose integrand
    is not of order one pass it in units of the integrand's scale (see
    :func:`_zeta_circle_kernels`)."""
    return 2.0 * nquad(
        lambda x: f(x),
        [[0, 0.5]],
        opts={"epsabs": epsabs, "epsrel": 1e-14, "limit": 100, "points": [0]},
    )[0]


def int_triangle(f) -> float:
    """Epstein integral over a 2D triangle subregion via Duffy coordinates."""
    result = 0.0
    for i, wi in enumerate(weight0Half):
        result += nquad(
            lambda x: np.real(wi * 2.0 * x * f(x, 2.0 * abscissa0Half[i] * x)),
            [[0, 0.5]],
            opts={"epsabs": 1e-12, "epsrel": 1e-14, "limit": 100, "points": [0]},
        )[0]
    return float(result)


def int_corner_2d(f) -> float:
    """Integration over corner [0,1/2]^2."""
    return int_triangle(lambda y1, y2: f(y1, y2)) + int_triangle(lambda y1, y2: f(y2, y1))


def int_full_2d(f) -> float:
    """Full 2D integration over the Brillouin zone."""
    return 2.0 * sum(int_corner_2d(lambda y1, y2: f(p1 * y1, y2)) for p1 in (-1, 1))


def int_pyramid(f) -> float:
    """Epstein integral over a 3D pyramid subregion via Duffy coordinates."""
    result = 0.0
    for i, wi in enumerate(weight0Half):
        for j, wj in enumerate(weight0Half):
            result += nquad(
                lambda x: np.real(
                    wi * wj * (2.0 * x) ** 2
                    * f(x, 2.0 * abscissa0Half[i] * x, 2.0 * abscissa0Half[j] * x)
                ),
                [[0, 0.5]],
                opts={"epsabs": 1e-12, "epsrel": 1e-14, "limit": 100, "points": [0]},
            )[0]
    return float(result)


def int_corner_3d(f) -> float:
    """Integral over the corner [0,1/2]^3."""
    return (
        int_pyramid(lambda y1, y2, y3: f(y1, y2, y3))
        + int_pyramid(lambda y1, y2, y3: f(y2, y3, y1))
        + int_pyramid(lambda y1, y2, y3: f(y3, y1, y2))
    )


def int_full_3d(f) -> float:
    """Integral over full Brillouin zone."""
    return 2.0 * sum(
        int_corner_3d(lambda y1, y2, y3: f(p1 * y1, p2 * y2, y3))
        for p1 in (-1, 1) for p2 in (-1, 1)
    )



# ---------------------------------------------------------------------------
# Fixed-node (tanh-sinh) 2D rule, and the Epstein table it makes cacheable
# ---------------------------------------------------------------------------
#
# The adaptive rule above chooses its nodes per integrand.  That is why it
# is accurate, and also why it is slow in a corpus pass: `epstein_zeta` is
# 83-84% of `zeta_circle` at 15-18 us a call, and every cycle re-derives
# its own node set, so nothing is ever reused.  Measured at d = 2 the
# whole 0qp pass spends ~42 s here, INDEPENDENT of ``n_points``.
#
# A tanh-sinh (double-exponential) rule on the Duffy radial variable fixes
# the nodes instead.  Two consequences, and the second is the larger:
#
#   * fewer evaluations -- 4424 nodes against the adaptive rule's
#     17k-35k, and FLAT in cycle length where the adaptive count grows;
#   * the node set is IDENTICAL for every cycle and every lattice, so
#     ``zeta_E(nu, A, .)`` can be tabulated once per (nu, A) and reused by
#     every cycle that mentions that exponent.  The shipped corpora carry
#     six distinct exponents, so a pass needs ~6 x 4424 Epstein
#     evaluations rather than ~1.6M.
#
# ACCURACY IS NOT TRADED FOR THIS.  Measured against the adaptive rule and
# against a refined tanh-sinh (M = 80, h = 0.04) as an internal referee:
#
#     sigma in {0.5, 0.25, 0.1}      (nu = 2 + sigma, down to the nu > d edge)
#     uniform AND non-uniform nu     (bundle multiplicities up to m = 3)
#     8 lattices, cond(A) 1.0 - 5.0  (cubic, scaled, rectangular,
#                                     anisotropic, triangular, oblique,
#                                     sheared, generic)
#
# every case agreed to round-off (0.0 to 3.7e-16 relative), at 4.1x to
# 13x.  On the anisotropic cells it is the ADAPTIVE rule that drifts
# (1.7e-14 on diag(1,5)) while the fixed rule sits at 0.
#
# The parameters are NOT tunable knobs: M = 48, h = 0.08 is the pair that
# was validated, and the 14-node Duffy angular rule
# (``abscissa0Half``/``weight0Half``) must not be swapped for plain
# Gauss-Legendre -- that was measured to break by 40% at every order.
_TS_M = 48
_TS_H = 0.08
#: Combined-weight floor for the 2D grid.  See ``_ts_grid_2d`` — this is a
#: NaN guard, not a performance tweak: it keeps the rule away from the
#: machine-epsilon nodes where ``epstein_zeta`` overflows at large ``nu``.
_TS_WEIGHT_FLOOR = 1e-20
#: ON by default, at d = 2 AND d = 3.  Agreement with the adaptive rule
#: is round-off, not bit-identity, so this is a deliberate flip rather
#: than a silent one:
#: validated at sigma in {0.5, 0.25, 0.1} (down to the nu > d edge), with
#: uniform AND non-uniform nu, over 8 lattices from cubic to strongly
#: sheared -- every case 0.0 to 3.7e-16 relative.  The kill switch
#: remains for the LEGACY FLOAT PATH ONLY: set this False to restore the
#: adaptive arithmetic exactly for pure power-law cycles (plain exponents
#: or Interactions without a compact part).  A cycle carrying a compact
#: part REFUSES it (``NotImplementedError`` in ``_zeta_circle_kernels``):
#: the adaptive rule integrates on the shipped 14-node angular rule with
#: no refinement and no self-check, and that rule cannot resolve a dense
#: table's trigonometric polynomial -- measured (d = 2, square cell,
#: purely compact random radius-2 tables, exact finite-sum reference)
#: 1.0e-08 at E = 5, 9.5e-06 at E = 8 and 4.6e-04 at E = 12 where the
#: ladder below is at 1e-14.  The ladder is what makes the fixed rule
#: correct for those cycles; the switch has no replacement for it.
USE_TANH_SINH = True

_TS_GRID_CACHE: dict = {}
_TS_TABLE_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_TS_TABLE_BYTES = 0
TS_TABLE_MAX_BYTES = int(os.environ.get("GZ_TS_TABLE_BYTES", 256 * 1024 * 1024))
#: Cross-call memo of a compact part's transform ``sum_m a(m) cos(2 pi m.y)``
#: at a rule's nodes, keyed on the table's bytes, the rule and the node
#: subset (``_ts_compact_table``).  Same reason as the Epstein tables: one
#: table recurs on every cycle of a pass, and once those tables are cached
#: the transform is the rule's whole per-table cost (0.1 s on the refined
#: 3D rule).  Bounded like ``_TS_TABLE_CACHE``; a refined 3D transform is
#: 9.9 MB unfolded and 0.8 MB folded onto the cubic sheets.
_TS_TRANSFORM_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_TS_TRANSFORM_BYTES = 0
TS_TRANSFORM_MAX_BYTES = int(
    os.environ.get("GZ_TS_TRANSFORM_BYTES", 64 * 1024 * 1024)
)

#: Refinement of the fixed rule for cycles whose bundles carry a COMPACT
#: part, per dimension: ``(radial factor, angular node count)`` -- the
#: radial tanh-sinh runs at ``(factor * _TS_M, _TS_H / factor)`` and the
#: Duffy angular rule at that many Gauss-Legendre nodes on ``[0, 1/2]``
#: (the shipped 14-node rule IS Gauss-Legendre 14 on that interval, see
#: ``_angular_rule``).  A compact part's transform is a trigonometric
#: polynomial, and what the rule has to resolve is the table's MODE
#: CONTENT -- the number of non-zero labels times their magnitude -- not
#: its support radius: E tables with K labels of size J carry modes to
#: |m| = E R with amplitudes built from J^E, and a sparse shell table
#: (J1 + J2 on the triangular cell: 12 labels, 6 of them at Chebyshev
#: radius 2) is a far easier integrand than a dense table of the same
#: radius.  The shipped rule integrates a bare plane wave of order
#: |m| = 12 to only 1e-5; with the amplitudes the modes actually carry,
#: MEASURED (d = 2, nu = 4.5, against the torus mean(fft2(V)**E) at
#: n = 512, converged like n^-7; K = non-zero off-origin labels, the
#: dense rows random even tables in [-J, J] with a(0) = 0.3):
#:
#:     (E, R,  K, J)      shipped rule   radial x2 + 28 ang   x4 + 56 ang
#:     ( 3, 1,  4, 0.5)      4.4e-16          5.6e-16
#:     (12, 1,  4, 0.5)      1.7e-12          4.7e-15
#:     (12, 2, sparse, 2.0)  2.2e-10          5.7e-15      (original row;
#:             triangular J1 + J2 at 2.0, 12 labels, re-measured 2.8e-10 / 2.3e-15)
#:     (12, 2, 24, 2.0)      4.6e-04          1.1e-13          1.2e-14
#:     (12, 3, 48, 2.0)      9.3e-09          4.5e-15          1.3e-14   (all +2)
#:
#: i.e. on sparse tables the shipped rule is already at 1e-10 and ONE
#: refinement step is at round-off (16 168 nodes, 0.2 s of Epstein table
#: per distinct nu), while a dense radius-2 table has the shipped rule
#: SIX orders worse and reaches 1e-13 only at the first refinement and
#: 1e-14 at the second; refining either the radial or the angular rule
#: alone stalls.  The ladder is a SELF-CHECK: consecutive rules must
#: agree to ``_CYCLE_SELF_BAND`` of the integrand's magnitude
#: ``scale = sum_i |w_i| |prod_i|`` (an ABSOLUTE criterion on the
#: Brillouin-zone integral, not a relative one on the returned value --
#: see ``_zeta_circle_kernels``), and the finer of the agreeing pair is
#: returned.  At d = 2 the second refinement (64 352 nodes, 0.9 s per
#: nu) is the referee when the first pair disagrees (the two dense rows
#: above).  At d = 3 the base grid already has 155 184 nodes (12 sheets)
#: and the x2 rule 1 234 176; measured on a cubic cell at nu = 5.5,
#: E <= 6, R = 1 the two agree to 6e-13 (the torus reference's own
#: limit) at 7.5 s for both tables, and a x3 rule would be ~4.2M nodes
#: (a 4 GB cosine matrix per dense edge and 350k ``epstein_zeta`` calls
#: per distinct exponent on the cubic cell, 12x that on a cell with one
#: sheet class), so the d = 3 ladder stops at one refinement.
#:
#: When the WHOLE ladder disagrees -- at d = 3 that is a dense radius-2
#: table on a cycle of E >= 4 edges: the base rule 1e-09 (E = 4) to
#: 1e-06 (E = 6) off the exact value while the x2 rule is at 1e-13;
#: physical shell tables and every radius-1 table agree at the first
#: rung -- the last rung is refereed by EXACT ARITHMETIC instead of being
#: discarded.  Expanding the integrand prod_e (Z_e + C_e) by the number
#: of power-law factors, its two classes of highest trigonometric degree
#: are finite sums: the purely compact product prod_e C_e (closed E-walks
#: of table displacements) and the one-power-law class
#: sum_e Z_e prod_{f != e} C_f (= sum_x T_{E-1}(x) K_nu(x) with T_{E-1}
#: the (E-1)-fold table convolution), and the rung is accepted iff it
#: reproduces both to the band (``_exact_compact_classes``).  What that
#: certifies and what it does not: the remaining classes (>= 2 power-law
#: factors, degree <= (E - 2) R) rest on a MEASURED ordering, not on a
#: finite sum -- on the cubic cell (dense radius-2 tables, all 124
#: labels, nu = 5.5, b in {1e-3, 1, 5, 20}, one- and two-term tails,
#: E = 4, 6) the quadrature error is monotone in the number of power-law
#: factors at BOTH rules (E = 6, b = 1: base 1e-06 / 3e-08 / 1e-08 on the
#: all-compact / one-power-law / remaining classes; x2 rule 1e-13 / 1e-14
#: on the two exact classes), every refereed value sits 1e-13..1e-12
#: from a torus reference whose own limit that is, and at b >= 100 the
#: compact classes fall below the band and the ladder no longer
#: triggers.  Cost: the class products ride on the rung's own nodes (no
#: extra nodes, not measurable beside the cosine matrix) and the finite
#: sums run on a (2 E R + 1)^d grid (4 913 terms at E = 4 and 15 625 at
#: E = 6 for R = 2, d = 3) -- MEASURED on a cubic 4-cycle with a dense
#: radius-2 compact-only table: the whole disagreement path is 3.7 s
#: wall-clock (base + x2 rule, 1 389 360 nodes) of which the referee's
#: finite sums are 0.5 ms.  A cycle with more than one edge lacking a
#: compact part has its highest-degree class outside both finite sums;
#: there the referee declines, and only then is
#: :class:`CycleQuadratureError` raised, so the caller can fall back
#: (the front-end sends the block down the sigma-router, what
#: ``fast_cycles=True`` does).  At d = 2 the referee is likewise the last
#: resort after the second refinement; no table was found that needs it
#: there.
#:
#: The exact sums do NOT short-circuit the ladder.  An earlier revision
#: accepted the base rule outright when its two exact classes were
#: reproduced, on the reading that the classes with fewer compact factors
#: are smoother; class-resolved measurement refuted it.  On a signed
#: 12-label radius-2 table (d = 2, sheared cell, nu = 6, E = 12) the two
#: checked classes sit just under the band while the classes with 2..6
#: power-law factors sit above it, and their coherent sum left the
#: accepted value 8.7e-9 of scale off a torus reference (Richardson over
#: n = 512, 1024, self-consistent to 7e-16 of scale) that the first
#: refined rung reproduces to 6e-15; a signed J1 + J2 + J3 shell table on
#: the square cell (nu = 3.5, E = 6) was accepted 2.1e-9 off; and a large
#: tail (b = 20) fools the check by amplitude, the two checked classes
#: being tiny beside scale.  So the base rule is accepted only when the
#: first refined rung agrees with it, and the exact sums referee the
#: EXHAUSTED ladder alone.
#:
#: Validated range: sigma = nu - d >= 0.1.  Below it the fixed rules
#: disagree among themselves even for the PURE power law at E = 12
#: (square cell, nu = 2.05: base vs x2 8e-10 of scale, x2 vs x4 7e-9;
#: nu = 2.1 is inside the band at 4e-10, nu = 2.25 at 1e-12), a
#: pre-existing limit of the shipped rule -- which the pure path accepts
#: with no self-check at all -- that a compact part inherits: there the
#: ladder's verdict certifies the pure rule's own accuracy, not more.
_TS_REFINED = {2: ((2, 28), (4, 56)), 3: ((2, 28),)}
_CYCLE_SELF_BAND = 1e-9


class CycleQuadratureError(GraphZetaError):
    """No two consecutive fixed-node rules agree, to ``_CYCLE_SELF_BAND``
    of the integrand's magnitude ``sum_i |w_i| |prod_i|`` (an absolute
    criterion on the Brillouin-zone integral), on a cycle whose bundles
    carry a compact part, and the exact-sum referee could not certify the
    last rung either: the quadrature is not converged for this kernel.
    Also raised when the d = 1 adaptive rule returns a non-finite value
    (``_int_full_1d_finite``), which it must never return silently.
    The front-end catches it and evaluates the block on the sigma-router
    instead."""


def _int_full_1d_finite(integrand, retry, what, epsabs: float = 1e-12) -> float:
    """``int_full_1d(integrand)``, returned untouched when finite.

    Otherwise ONE retry on ``retry`` -- the same integrand with the direct
    Epstein sum at the non-finite nodes (``direct_1d=True``, see
    :func:`_epstein_zeta_1d_direct`) -- and :class:`CycleQuadratureError`
    if that is non-finite too: the d = 1 adaptive rule has no self-check,
    so a NaN would otherwise leave as the value.
    """
    val = int_full_1d(integrand, epsabs)
    if math.isfinite(val):
        return val
    val = int_full_1d(retry, epsabs)
    if not math.isfinite(val):
        raise CycleQuadratureError(
            f"zeta_circle: the d = 1 cycle integral is {val!r} for "
            f"{what!r} even with the direct Epstein sum at the non-finite "
            f"nodes: either the value lies outside the float64 range or "
            f"the quadrature failed.  The front-end evaluates the block on "
            f"the sigma-router instead."
        )
    return val


def _angular_rule(n_ang):
    """Nodes and weights of the Duffy angular rule on ``[0, 1/2]``.

    ``None`` returns the shipped 14-node arrays ``abscissa0Half`` /
    ``weight0Half`` (bit-identical, the legacy rule); an integer returns
    the ``n_ang``-node Gauss-Legendre rule mapped to ``[0, 1/2]``.  The
    shipped arrays ARE that rule at 14 nodes (their first abscissa is
    ``0.25 * (1 - 0.98628...)``), so the refinement keeps the rule's
    construction and only its resolution changes.
    """
    if n_ang is None:
        return (np.asarray(abscissa0Half, dtype=np.float64),
                np.asarray(weight0Half, dtype=np.float64))
    g, w = np.polynomial.legendre.leggauss(int(n_ang))
    return 0.25 * (g + 1.0), 0.25 * w


def _ts_table_clear() -> None:
    """Drop every cached Epstein table and compact transform."""
    global _TS_TABLE_BYTES, _TS_TRANSFORM_BYTES
    _TS_TABLE_CACHE.clear()
    _TS_TABLE_BYTES = 0
    _TS_TRANSFORM_CACHE.clear()
    _TS_TRANSFORM_BYTES = 0


def _tanh_sinh_1d(M: int, h: float, a: float = 0.0, b: float = 0.5):
    r"""Double-exponential nodes and weights on ``[a, b]``.

    ``x = tanh((pi/2) sinh(k h))`` clusters nodes exponentially at the
    endpoints, which is what lets a fixed rule match an adaptive one on
    an integrand whose derivative is singular at the Duffy corner.
    Nodes whose weight has underflowed contribute nothing and are
    dropped, which is why the count (77) is below the nominal 2M+1.
    """
    ks = np.arange(-M, M + 1)
    u = 0.5 * np.pi * np.sinh(ks * h)
    t = np.tanh(u)
    w = 0.5 * np.pi * h * np.cosh(ks * h) / np.cosh(u) ** 2
    keep = np.isfinite(w) & (w > 1e-18) & (np.abs(t) < 1.0)
    t, w = t[keep], w[keep]
    return 0.5 * (b - a) * (t + 1.0) + a, 0.5 * (b - a) * w


def _ts_grid_2d(M: int = _TS_M, h: float = _TS_H, n_ang=None):
    r"""BZ nodes ``y`` and combined weights for the 2D rule.

    Reproduces :func:`int_full_2d`'s decomposition exactly -- outer factor
    2, the ``p1 = -1, +1`` sum, the two Duffy triangles (plain and
    swapped), the 14-node angular rule, and the ``2x`` Jacobian -- with
    the radial integral evaluated on fixed nodes instead of adaptively.
    Independent of ``nu`` AND of the lattice, so it is built once.
    ``n_ang`` (default ``None`` = the shipped 14-node rule, same cache key
    as before) selects a refined angular rule; see ``_TS_REFINED``.
    """
    key = (int(M), float(h)) if n_ang is None else (int(M), float(h), int(n_ang))
    hit = _TS_GRID_CACHE.get(key)
    if hit is not None:
        return hit
    xs, ws = _tanh_sinh_1d(M, h)
    ang_a, ang_w = _angular_rule(n_ang)
    ys, wt = [], []
    for p1 in (-1, 1):
        for swap in (False, True):
            for wi, ai in zip(ang_w, ang_a):
                for x, wj in zip(xs, ws):
                    u_, v_ = (x, 2.0 * ai * x)
                    if swap:
                        u_, v_ = v_, u_
                    ys.append((p1 * u_, v_))
                    wt.append(2.0 * float(wi) * float(wj) * 2.0 * x)
    ys = np.asarray(ys, dtype=np.float64)
    wt = np.asarray(wt, dtype=np.float64)

    # Drop nodes whose COMBINED weight has underflowed.  ``_tanh_sinh_1d``
    # already filters its own ``w > 1e-18``, but the 2D weight multiplies
    # in the Duffy Jacobian ``2x`` with ``x`` as small as ~1e-16, so a node
    # can survive that filter and still arrive here weighing ~4e-33.
    #
    # This is not housekeeping — those nodes POISON the result.
    # ``epstein_zeta(nu, A, 0, y)`` returns ``nan+nanj`` for large ``nu`` at
    # machine-epsilon ``|y|``: an intermediate ``|y|^-nu`` overflows double
    # precision, and the frontier sits at ``nu * |log10|y|| ~ 350``
    # (epsteinlib 0.6.2 and 0.5.1; reported upstream).  The double
    # exponential is precisely the rule that puts nodes there — the
    # adaptive rule never samples that close to the Duffy corner, which is
    # why only the fixed rule tripped.  One NaN node makes the whole dot
    # product NaN, so an entire cycle came back NaN on 8.4e-31 of the
    # total weight.
    #
    # Concretely, before this filter: nu_max = 21.935030 finite,
    # 21.935031 NaN, taking with it every d = 2 corpus cycle whose bundle
    # multiplicity m satisfies m*nu >= 21.94 — order 11 at sigma = 0.5,
    # but order 5 at sigma = 8.  d = 1 keeps the adaptive rule and was
    # never affected; d = 3 now uses the fixed rule too and inherits this
    # filter through ``_ts_grid_3d``, whose smallest |y| is 2.1e-06 --
    # four orders further from the frontier than the 2D grid's 8.5e-10.
    #
    # The threshold is far below any node that can matter: 1e-20 against a
    # total weight of 1.0, and the sum of the dropped weights is 0.0 in
    # double precision.  It removes 392 of 4424 nodes, so the rule also
    # gets ~9% cheaper.
    keep = np.abs(wt) > _TS_WEIGHT_FLOOR
    out = (ys[keep], wt[keep])
    _TS_GRID_CACHE[key] = out
    return out


#: ``key -> (sheet operations, nodes per sheet)`` for the 3D grid, filled
#: by :func:`_ts_grid_3d`.  See :func:`_ts_sheet_classes`.
_TS_GRID_3D_OPS: dict = {}


def _table_invariant(T: np.ndarray, labels: np.ndarray,
                     values: np.ndarray) -> bool:
    """``a(T m) == a(m)`` on every label of a compact table, exactly, for a
    signed permutation ``T`` -- the condition for the table's transform
    ``sum_m a(m) cos(2 pi m.y)`` to satisfy ``c(T y) == c(y)``."""
    lookup = {tuple(int(x) for x in m): float(v) for m, v in zip(labels, values)}
    image = np.rint(labels @ T.T).astype(np.int64)
    return all(lookup.get(tuple(int(x) for x in m)) == float(v)
               for m, v in zip(image, values))


def _ts_sheet_classes(A: np.ndarray, ops: np.ndarray, tables=()) -> list:
    r"""Group the grid's sheets by whether this lattice can tell them apart.

    ``zeta_E(nu, A, 0, S y) = zeta_E(nu, A, 0, y)`` exactly when the signed
    permutation ``S`` preserves the Gram matrix ``G = A^T A``: substituting
    ``w = S^T z`` (a bijection of the lattice, since ``S`` is a signed
    permutation) turns the sum into ``sum_w e^{-2 pi i w.y} |A S w|^-nu``,
    which is the original iff ``S^T G S = G``.

    Returns a list of ``(representative, [members])``.  On a cubic cell
    every signed permutation qualifies, so all 12 sheets collapse to one
    and 11/12 of the Epstein evaluations disappear.  On ``diag(1,2,3)``
    only the sign flips qualify; on a sheared cell none do and the grouping
    is the identity -- correctness never depends on the symmetry holding.

    ``tables`` (pairs ``(labels, values)`` of compact parts) restricts the
    grouping further to the signed permutations that leave every table
    invariant (``_table_invariant``), which is what a fold of the whole
    integrand needs (``_ts_fold``); the Epstein table alone passes none.
    """
    G = np.asarray(A, dtype=np.float64).T @ np.asarray(A, dtype=np.float64)
    # RELATIVE to the cell, with no clamp.  Clamping the scale at 1.0
    # makes this an ABSOLUTE test, and a lattice whose Gram entries are
    # all below the tolerance then compares equal to everything: at
    # |A| ~ 1e-6 every sheet merged into one class and the table was
    # silently wrong.  A degenerate cell (G == 0) is not a lattice, and
    # `zeta_circle` rejects it upstream, but guard the division anyway.
    scale = float(np.max(np.abs(G)))
    if not np.isfinite(scale) or scale <= 0.0:
        return [(i, [i]) for i in range(len(ops))]
    reps: list = []
    for i, S in enumerate(ops):
        for r, members in reps:
            T = S @ np.linalg.inv(ops[r])
            if (np.max(np.abs(T.T @ G @ T - G)) <= 1e-12 * scale
                    and all(_table_invariant(T, lab, val) for lab, val in tables)):
                members.append(i)
                break
        else:
            reps.append((i, [i]))
    return reps


def _ts_grid_3d(M: int = _TS_M, h: float = _TS_H, n_ang=None):
    r"""BZ nodes ``y`` and combined weights for the 3D rule.

    Reproduces :func:`int_full_3d`'s decomposition exactly, term for
    term: the outer factor 2, the ``p1, p2 in {-1, +1}`` sign sum, the
    THREE cyclic reorderings that :func:`int_corner_3d` spells out, the
    14x14 angular rule, and :func:`int_pyramid`'s ``(2x)^2`` Duffy
    Jacobian -- with the radial integral on fixed tanh-sinh nodes
    instead of adaptive ones.  Independent of ``nu`` AND of the lattice,
    so it is built once.

    Node count is ``2 * 2 * 3 * 14 * 14 * len(xs)``, i.e. 12 angular
    sheets against the 2D rule's 4.  That is the price of the extra
    dimension and it is paid ONCE per process, not per cycle: the whole
    point is that the table below then serves every cycle in the pass.
    """
    key = (("3d", int(M), float(h)) if n_ang is None
           else ("3d", int(M), float(h), int(n_ang)))
    hit = _TS_GRID_CACHE.get(key)
    if hit is not None:
        return hit

    xs, ws = _tanh_sinh_1d(M, h)
    a, wa = _angular_rule(n_ang)

    # Duffy base point (y1, y2, y3) = (x, 2 a_i x, 2 a_j x) on the
    # (i, j, radial) grid, built by broadcast rather than a triple loop.
    x = xs[None, None, :]
    shape = (a.size, a.size, xs.size)
    base = tuple(np.ascontiguousarray(np.broadcast_to(v, shape)) for v in (
        x,                                     # y1 = x
        2.0 * a[:, None, None] * x,            # y2 = 2 a_i x
        2.0 * a[None, :, None] * x,            # y3 = 2 a_j x
    ))
    w = np.ascontiguousarray(np.broadcast_to(
        2.0                                    # int_full_3d's outer factor
        * wa[:, None, None] * wa[None, :, None]
        * ws[None, None, :]
        * (2.0 * x) ** 2,                      # int_pyramid's Jacobian
        shape))

    # int_corner_3d sums f(y1,y2,y3) + f(y2,y3,y1) + f(y3,y1,y2).
    PERMS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    ys, wt = [], []
    for p1 in (-1, 1):
        for p2 in (-1, 1):
            for perm in PERMS:
                q0, q1, q2 = (base[perm[0]], base[perm[1]], base[perm[2]])
                ys.append(np.stack((p1 * q0, p2 * q1, q2),
                                   axis=-1).reshape(-1, 3))
                wt.append(w.reshape(-1))
    ys = np.concatenate(ys, axis=0)
    wt = np.concatenate(wt, axis=0)
    # Every sheet is the FIRST sheet mapped by a signed permutation:
    # node_s[i] = S_s @ node_0[i].  Recording those matrices is what lets
    # the table below skip the sheets a lattice cannot tell apart.
    ops = []
    P = {perm: np.eye(3)[list(perm)] for perm in PERMS}
    D = lambda p1, p2: np.diag([float(p1), float(p2), 1.0])
    M0 = None
    for p1 in (-1, 1):
        for p2 in (-1, 1):
            for perm in PERMS:
                M = D(p1, p2) @ P[perm]
                if M0 is None:
                    M0 = M
                ops.append(M @ np.linalg.inv(M0))

    # Same combined-weight floor as the 2D rule, and for the same reason:
    # the double exponential puts nodes at machine-epsilon |y|, where
    # ``epstein_zeta`` returns nan+nanj for large nu, and one NaN node
    # makes the whole dot product NaN.  See ``_ts_grid_2d``.
    # The weight does not depend on (p1, p2, perm), so the floor removes
    # the SAME nodes from every sheet and the 12 equal blocks survive it.
    keep = np.abs(wt) > _TS_WEIGHT_FLOOR
    ys, wt = np.ascontiguousarray(ys[keep]), np.ascontiguousarray(wt[keep])
    assert len(ys) % len(ops) == 0, "sheet blocks must stay equal-sized"
    _TS_GRID_3D_OPS[key] = (np.stack(ops), len(ys) // len(ops))
    out = (ys, wt)
    _TS_GRID_CACHE[key] = out
    return out


def _ts_grid(d: int, M: int = _TS_M, h: float = _TS_H, n_ang=None):
    """Fixed-node BZ grid for dimension ``d`` (2 or 3)."""
    if d == 2:
        return _ts_grid_2d(M, h, n_ang)
    if d == 3:
        return _ts_grid_3d(M, h, n_ang)
    raise NotImplementedError(
        f"_ts_grid: fixed-node rule is implemented for d = 2 and d = 3 "
        f"(got d = {d}); d = 1 keeps its adaptive rule, which is a single "
        f"1-D integral and already cheap."
    )


#: Above this ``sigma = nu - d`` the Epstein table is summed as a short
#: Fourier series instead of calling ``epstein_zeta``.
#:
#: The window is bounded from BOTH sides and this sits inside it, measured
#: on the shipped 4032-node grid at d = 2 (times are for a whole table):
#:
#:     sigma      L    terms    direct    epsteinlib    max |diff|
#:        6.0   179   128881   12.39 s        0.057 s     3.0e-14
#:        8.0    65    17161    1.04 s        0.057 s     2.0e-14
#:       12.0    21     1849    0.118 s       0.061 s     7.1e-15
#:       14.0    15      961    0.060 s       0.061 s     4.4e-15   <- cost crossover
#:       16.0    11      529    0.031 s       0.063 s     2.7e-15   <- SIGMA_MAX_CIRCLE
#:       18.0     9      361    0.021 s       0.062 s     1.8e-15
#:       38.0     4       81    0.005 s       0.071 s     3.6e-15
#:
#: BELOW ~14 the series stops being small and the library is cheaper.
#: ABOVE ~36.6 the library is not merely slower but WRONG: it returns
#: ``nan+nanj`` at the small-|y| nodes this rule deliberately clusters
#: (frontier ``nu * |log10|y|| ~ 350``, and the grid's smallest |y| is
#: 8.5e-10).  So the switch is a correctness boundary with a cost
#: crossover conveniently below it, and 16 takes the margin: 2x cheaper
#: than the library at the switch, and far enough under 36.6 that a later
#: change to ``M``, ``h`` or ``_TS_WEIGHT_FLOOR`` — any of which moves the
#: grid's smallest |y|, and with it the frontier — cannot silently cross it.
SIGMA_MAX_CIRCLE = 16.0

#: Working-set ceiling for one ``_epstein_table_direct`` block, in bytes.
#: The (node x lattice-vector) phase matrix is what this bounds; see the
#: chunking note there.  256 MB keeps the transient far below the table
#: it produces (2.48 MB at d = 3) while staying large enough that the
#: block loop is a handful of iterations, not thousands.
_DIRECT_TABLE_MAX_BYTES = int(
    os.environ.get("GZ_DIRECT_TABLE_BYTES", 256 * 1024 * 1024)
)

#: Separate ceiling for the LATTICE-VECTOR grid, which is a one-off
#: allocation made before the chunk loop -- a different thing from the
#: per-chunk transient above, and it must stay independently settable:
#: shrinking the chunk budget to exercise chunking would otherwise trip
#: this guard instead, which is exactly what happened when they shared a
#: constant.
_DIRECT_VECTORS_MAX_BYTES = int(
    os.environ.get("GZ_DIRECT_VECTORS_BYTES", 1024 * 1024 * 1024)
)

#: Truncation tolerance for ``_epstein_table_direct``, per dimension.
#:
#: The radius comes from the tail bound ``(s_min L)^-sigma <= tol``, which
#: is rigorous but PESSIMISTIC by about 100x against the error actually
#: committed -- it ignores that the leading shell is O(1) and that the
#: shell count only grows as ``r^(d-1)``.  Measured on a 3000-node sample
#: of the d = 3 grid, max relative deviation from an L = 16 reference:
#:
#:     sigma   L=2     L=3     L=4     L=6     L=8    shipped L (tol 1e-18)
#:      16    3e-09   2e-11   5e-13   4e-15   9e-16       15  (17 sheared)
#:      22    3e-12   5e-15   7e-16   7e-16   6e-16        8
#:      42    3e-16   5e-16   3e-16   3e-16   5e-16        4
#:
#: cond(A) is not the driver: the sheared cell (cond 1.57) needs the same
#: L as the cubic one, because ``s_min`` already carries the anisotropy.
#:
#: At d = 3 that over-estimate costs ``(31/19)^3`` ~ 4x, on a table that
#: took 74.6 s at sigma = 16.  1e-14 lands at ~5e-16 -- the round-off of
#: the product-and-dot the table feeds, so nothing downstream can see the
#: difference.
#:
#: d <= 2 KEEPS 1e-18.  There the whole table is 0.03 s, so there is
#: nothing to buy, and holding it fixed keeps every shipped d <= 2 value
#: bit-identical.
_DIRECT_TOL_BY_D = {3: 1e-14}


def _epstein_table_direct(nu: float, A: np.ndarray, ys: np.ndarray,
                          tol: "float | None" = None) -> np.ndarray:
    r"""``zeta_E(nu, A, 0, A* y)`` at every node, by direct summation.

    For ``nu >> d`` the Epstein sum simply IS a short Fourier series,

        zeta_E(nu, A, 0, A* y)  =  sum_{z != 0} exp(-2 pi i z.y) |A z|^-nu

    (convention verified against ``epstein_zeta`` to absolute round-off),
    and ``|A z|^-nu`` decays so fast that a couple of shells is exact.
    At ``nu = 110`` two shells suffice; at ``nu = 20`` it is eight.

    WHY THIS EXISTS.  ``epstein_zeta`` returns ``nan+nanj`` for large
    ``nu`` at small ``|y|`` — an intermediate ``|y|^-nu`` overflows double
    precision, with the frontier at ``nu * |log10|y|| ~ 350`` (epsteinlib
    0.6.2 and 0.5.1; reported upstream).  The double-exponential rule
    clusters nodes at the Duffy corner, so it lands squarely in that
    region and one poisoned node makes the whole dot product NaN.

    Dropping the underflowed nodes (``_TS_WEIGHT_FLOOR``) moves the
    frontier from ``nu = 21.9`` to ``38.6`` but does not remove it, and
    the corpus needs far more than that: a bundle of multiplicity ``m``
    merges to ``m*nu``, so d = 2 order 11 reaches **110** at sigma = 8
    and 44 at sigma = 2.  Summing directly is exact at every one of them.

    The truncation radius is set from the tolerance and the lattice's
    SHORTEST vector, ``|A z| >= sigma_min(A) |z|``, so it stays correct on
    anisotropic and sheared cells rather than assuming ``|A z| ~ |z|``.
    """
    A = np.asarray(A, dtype=np.float64)
    d = A.shape[0]
    nu = float(nu)
    sigma = nu - d
    if tol is None:
        tol = _DIRECT_TOL_BY_D.get(d, 1e-18)
    if sigma <= 0.0:
        raise ValueError(
            f"_epstein_table_direct: needs nu > d for the tail to converge; "
            f"got nu={nu}, d={d}"
        )
    s_min = float(np.linalg.svd(A, compute_uv=False).min())
    if not np.isfinite(s_min) or s_min <= 0.0:
        raise ValueError(f"_epstein_table_direct: singular lattice matrix A={A!r}")
    # Shell-summed tail: sum_{|z| > L} |A z|^-nu <~ (s_min L)^(d - nu)
    # = (s_min L)^-sigma.  Using sigma rather than nu is what makes this a
    # bound rather than an optimistic estimate.
    L = int(np.ceil((tol ** (-1.0 / sigma)) / s_min)) + 1

    # BOUND THE LATTICE-VECTOR GRID TOO.  L grows as 1/s_min, so a
    # near-degenerate cell explodes it -- and `zs` and `coef` are built
    # BEFORE the chunk loop that `_DIRECT_TABLE_MAX_BYTES` governs, i.e.
    # exactly the failure the ceiling was added to prevent, one line
    # above the ceiling.  At d = 3, nu = 19 (the smallest nu reaching
    # this branch) a cell with s_min = 0.02 gives L = 376 and
    # |zs| = 4.3e8: 10.25 GB of `zs` plus 6.8 GB of `coef`.
    #
    # Refuse rather than allocate.  There is no correct smaller L -- the
    # radius is what the tolerance demands -- so silently clamping would
    # trade a crash for a wrong number, which is the trade this file
    # keeps getting wrong.
    n_vec = (2 * L + 1) ** d - 1
    need = n_vec * (8 * d + 24)
    if need > _DIRECT_VECTORS_MAX_BYTES:
        raise MemoryError(
            f"_epstein_table_direct: lattice-vector grid needs "
            f"{need / 1e9:.2f} GB (L = {L}, {n_vec:,} vectors) at nu={nu}, "
            f"d={d}, s_min(A)={s_min:.3g}, exceeding "
            f"{_DIRECT_VECTORS_MAX_BYTES / 1e9:.2f} GB.  L grows as "
            f"1/s_min, so this is a near-degenerate cell; raise "
            f"GZ_DIRECT_VECTORS_BYTES if the allocation is genuinely wanted."
        )

    rng = np.arange(-L, L + 1)
    zs = np.stack(np.meshgrid(*([rng] * d), indexing="ij"),
                  axis=-1).reshape(-1, d).astype(np.float64)
    zs = zs[np.any(zs != 0.0, axis=1)]
    lengths = np.linalg.norm(zs @ A.T, axis=1)
    coef = lengths ** (-nu)
    coef_c = coef.astype(np.complex128)

    # exp(-2 pi i z.y) for every (node, lattice vector) pair -- CHUNKED
    # over nodes.  The full matrix is len(ys) x len(zs) complex: harmless
    # at d = 2 (4,032 nodes) and 74 GB at d = 3, where the fixed grid has
    # 155,184.  Each node's dot product is independent of every other, so
    # blocking the node axis is EXACT in exact arithmetic -- but not
    # bitwise: BLAS picks a different inner blocking per shape, so the
    # values move by round-off (measured 1.4e-16 to 4.4e-16 at d = 2).
    ys = np.asarray(ys, dtype=np.float64)
    out = np.empty(len(ys), dtype=np.complex128)
    # 40 B/entry, not 16: the expression below holds THREE arrays of the
    # block's shape live at once -- the float64 matmul result (8 B), its
    # complex exponential (16 B), and numpy's temporary for the product
    # (16 B).  Pricing only the final complex matrix under-bounded the
    # transient by 2.5x (measured 641 MB against a declared 256 MB), which
    # matters because callers set this from a real memory budget and may
    # run several workers.
    block = max(1, int(_DIRECT_TABLE_MAX_BYTES // (40 * max(len(zs), 1))))
    for i in range(0, len(ys), block):
        sl = slice(i, i + block)
        out[sl] = np.exp(-2j * np.pi * (ys[sl] @ zs.T)) @ coef_c
    return out


def _ts_zeta_table(nu: float, A: np.ndarray, A_star: np.ndarray,
                   M: int = _TS_M, h: float = _TS_H, n_ang=None) -> np.ndarray:
    r"""``zeta_E(nu, A, 0, A* y)`` at every fixed node, memoised.

    THIS is where the pass-level saving comes from: the table depends on
    ``(nu, A)`` only, and the shipped corpora carry six distinct
    exponents, so every cycle that mentions one of them reuses the same
    array instead of re-evaluating the Epstein sum.  A table is 4424
    complex entries, ~71 kB.
    """
    global _TS_TABLE_BYTES
    key = (float(nu), int(M), float(h), A.shape, A.tobytes())
    if n_ang is not None:
        key = key + (int(n_ang),)
    hit = _TS_TABLE_CACHE.get(key)
    if hit is not None:
        _TS_TABLE_CACHE.move_to_end(key)
        return hit
    d_ = int(A.shape[0])
    ys, _ = _ts_grid(d_, M, h, n_ang)

    def _evaluate(nodes):
        if float(nu) - d_ >= SIGMA_MAX_CIRCLE:
            # epstein_zeta is NaN for large nu at the small-|y| nodes this
            # rule deliberately clusters; the series is short enough to sum
            # outright.
            return _epstein_table_direct(float(nu), A, nodes)
        zeros = np.zeros(len(A), dtype=np.float64)
        out = np.empty(len(nodes), dtype=np.complex128)
        for j in range(len(nodes)):
            out[j] = epstein_zeta(float(nu), A, zeros, A_star @ nodes[j])
        return out

    sheets = None
    if d_ == 3:
        gkey = (("3d", int(M), float(h)) if n_ang is None
                else ("3d", int(M), float(h), int(n_ang)))
        sheets = _TS_GRID_3D_OPS.get(gkey)
    if sheets is not None:
        # Evaluate one sheet per symmetry class and copy to its members.
        # This is exact, not an approximation: members are related by a
        # signed permutation that provably leaves zeta_E invariant on THIS
        # lattice.  Cubic collapses 12 sheets to 1.
        ops, block = sheets
        vals = np.empty(len(ys), dtype=np.complex128)
        for rep, members in _ts_sheet_classes(A, ops):
            got = _evaluate(ys[rep * block:(rep + 1) * block])
            for m in members:
                vals[m * block:(m + 1) * block] = got
    else:
        vals = _evaluate(ys)
    if not np.all(np.isfinite(vals)):
        raise FloatingPointError(
            f"_ts_zeta_table: non-finite Epstein values at nu={nu} on "
            f"{int((~np.isfinite(vals)).sum())} of {len(vals)} nodes.  This "
            f"should be unreachable: sigma >= {SIGMA_MAX_CIRCLE} sums directly "
            f"and below it epstein_zeta is in range."
        )
    vals.setflags(write=False)
    nb = vals.nbytes
    if nb <= TS_TABLE_MAX_BYTES:
        while _TS_TABLE_CACHE and _TS_TABLE_BYTES + nb > TS_TABLE_MAX_BYTES:
            _, ev = _TS_TABLE_CACHE.popitem(last=False)
            _TS_TABLE_BYTES -= ev.nbytes
        _TS_TABLE_CACHE[key] = vals
        _TS_TABLE_BYTES += nb
    return vals


def _refuse_low_cycle_exponents(nu_vec, d) -> None:
    """Refuse a cycle whose exponents the closed form does not evaluate.

    A NaN or ``-inf`` is a malformed argument (``ValueError``), and
    ``+inf`` is the nearest-neighbour limit, which the closed form does
    not evaluate (:class:`~gzl.UnsupportedRequestError`).  An exponent
    ``nu <= d`` is outside what gzl supports for any block but a bridge
    (:class:`~gzl.UnsupportedLatticeSumError`, itself a ``ValueError``):
    the cycle's sum may still converge, since that is governed by its
    cluster cut ``2 nu``, not by ``nu``.
    """
    if np.isnan(nu_vec).any():
        raise ValueError(f"nu_vec contains NaN: {nu_vec!r}")
    if np.isneginf(nu_vec).any():
        raise ValueError(f"nu_vec contains -inf: {nu_vec!r}")
    if np.isposinf(nu_vec).any():
        # Before the closed form, which did not return within minutes.
        raise UnsupportedRequestError(
            f"zeta_circle evaluates finite exponents; got {nu_vec!r}.  "
            f"nu = inf is the nearest-neighbour limit, which "
            f"evaluate_graph evaluates."
        )
    if not np.all(nu_vec > d):
        raise UnsupportedLatticeSumError(
            f"All nu_i must satisfy nu_i > d = {d}. Received nu_vec = "
            f"{nu_vec!r}.  A cycle with an exponent <= d is not supported "
            f"(see UnsupportedLatticeSumError)."
        )


def zeta_circle_tanh_sinh(nu_vec, A, M: int = _TS_M, h: float = _TS_H) -> float:
    r"""``zeta_circle`` on the fixed tanh-sinh grid.  d = 2 or d = 3.

    The integrand is a PRODUCT over edges of the same tabulated function,
    so once the tables are in hand a cycle costs one elementwise product
    and one dot -- no Epstein evaluations at all.  Exponents are
    multiplied by multiplicity via ``**count`` rather than repeated
    multiplication, which is both faster and exactly the same value for
    integer counts.
    """
    A = np.asarray(A, dtype=np.float64)
    d = int(A.shape[0])
    if d not in (2, 3):
        raise NotImplementedError(
            f"zeta_circle_tanh_sinh: d = 2 and d = 3 (got d = {d}); d = 1 "
            f"keeps its adaptive rule, which is one 1-D integral and already "
            f"cheap."
        )
    nu_vec = np.asarray(nu_vec, dtype=np.float64)
    _refuse_low_cycle_exponents(nu_vec, d)
    A_star = np.linalg.inv(A).T
    _, wt = _ts_grid(d, M, h)
    counts: dict = {}
    for nu in nu_vec:
        counts[float(nu)] = counts.get(float(nu), 0) + 1
    prod = np.ones(len(wt), dtype=np.complex128)
    for nu, c in counts.items():
        t = _ts_zeta_table(nu, A, A_star, M, h)
        prod = prod * (t if c == 1 else t ** c)
    return float(np.real(np.dot(wt, prod)))


# ----------------------------
# General kernels on a cycle
# ----------------------------
#
# A cycle carrying general interactions V_e(x) = a_e(x) + sum_j b_ej K_nu_ej
# is still the Brillouin-zone integral of the product of the per-edge
# lattice Fourier transforms,
#
#     zeta_cycle = int_BZ prod_e Vhat_e(y) dy,
#     Vhat_e(y)  = sum_j c_ej zeta_E(nu_ej; y) + sum_m v_em cos(2 pi m.y),
#
# where the second sum -- the transform of the compact part -- is an
# ENTIRE trigonometric polynomial (for a bundle, the pure power-law part
# expands over exponent multisets and the compact remainder lives on the
# union of the supports; see ``Interaction.fourier_terms``).  Nothing new
# is singular: the Duffy / tanh-sinh construction that resolves the
# |y|^(nu - d) Epstein singularities applies unchanged, and only the
# RESOLUTION of the fixed rule is at stake -- see ``_TS_REFINED``.

def _kernel_sup_bound_1d(kernels, A) -> float:
    r"""An upper bound on ``sup_y |prod_e Vhat_e(y)|`` at d = 1.

    ``|Vhat_e(y)| <= sum_j |c_j| Z(nu_j, 0) + sum_m |a_e(m)|`` with
    ``Z(nu, 0) = 2 zeta(nu) |a|^-nu`` the Epstein zeta at zero momentum on
    the chain ``A = (a)`` (every cosine at most one).  ``kernels`` is the
    ``fourier_terms`` list of ``_zeta_circle_kernels``.
    """
    from scipy.special import zeta as _hurwitz_zeta
    a = abs(float(np.asarray(A, dtype=np.float64).reshape(-1)[0]))
    bound = 1.0
    for terms, _labels, values in kernels:
        s_e = float(np.sum(np.abs(values))) if len(values) else 0.0
        for coef, nu in terms:
            nu = float(nu)
            s_e += abs(complex(coef)) * 2.0 * float(_hurwitz_zeta(nu, 1.0)) * a ** -nu
        bound *= s_e
    return bound


def _kernel_prod_at(kernels, A, A_star, y_vec, direct_1d=False) -> float:
    """``prod_e Vhat_e(y)`` at one fractional momentum, for the adaptive
    rules (d = 1, and d = 2, 3 with ``USE_TANH_SINH`` off).  Real by the
    evenness of every kernel.  ``direct_1d`` as in ``epstein_zeta_prod``."""
    y_vec = np.asarray(y_vec, dtype=np.float64).reshape(-1)
    zeros = np.zeros(len(A), dtype=np.float64)
    k_lat = A_star @ y_vec
    z_cache: dict = {}
    total = 1.0 + 0.0j
    for terms, labels, values in kernels:
        v = 0.0 + 0.0j
        for coef, nu in terms:
            nu = float(nu)
            if nu not in z_cache:
                z = epstein_zeta(nu, A, zeros, k_lat)
                if direct_1d and not cmath.isfinite(z):
                    z = complex(_epstein_zeta_1d_direct(nu, A, y_vec[0]))
                z_cache[nu] = z
            v += coef * z_cache[nu]
        if len(values):
            v += float(np.cos(2.0 * np.pi * (labels.astype(np.float64) @ y_vec)) @ values)
        total *= v
    return float(np.real(total))


def _compact_transform_at(ys, labels, values):
    """``sum_m a(m) cos(2 pi m . y)`` at every node ``y`` of ``ys``, the
    compact part's trigonometric polynomial, evaluated in node blocks so
    the ``(nodes x labels)`` cosine matrix never exceeds
    ``_DIRECT_TABLE_MAX_BYTES`` (three float64 arrays of the block's
    shape are live at once: the matmul result, its cosine and numpy's
    temporary for the product -- 24 B per entry; unchunked, the refined
    3D rule on a 124-label table held 2.5 GB).  Exact to round-off, not
    bit-stable across block sizes: at the shipped ceiling every block is
    large enough to take the same BLAS path as the one-shot matrix and
    the two agree bit for bit, but a block of a few rows rounds
    differently (measured up to 2e-14 absolute), so nothing may assume
    bit-identity if the ceiling or the table size changes."""
    lab = labels.T.astype(np.float64)
    out = np.empty(len(ys), dtype=np.float64)
    block = max(1, int(_DIRECT_TABLE_MAX_BYTES // (24 * max(len(values), 1))))
    for i in range(0, len(ys), block):
        sl = slice(i, i + block)
        out[sl] = np.cos(2.0 * np.pi * (ys[sl] @ lab)) @ values
    return out


def _ts_fold(A, M, h, n_ang, tables):
    r"""Fold the 3D rule's sheets that the lattice AND every compact table
    leave invariant.

    Returns ``(idx, W, reps)`` -- the nodes of one representative sheet
    per class, their weights summed over the class's members, and the
    representatives (part of the transform memo's key) -- or ``None``
    when nothing folds (``d != 3``, or a cell / table without a sheet
    symmetry).  Exact: on a member sheet every factor of the integrand
    takes the representative's values node for node -- the Epstein
    table by ``_ts_sheet_classes``'s Gram condition, the transform by
    the table's invariance under the same signed permutation -- so the
    fold is a reassociation of the same sum.  On the cubic cell with
    shell tables all 12 sheets collapse to one: the per-cycle product on
    the refined rule drops from 1.2M nodes to 103k.
    """
    d = int(A.shape[0])
    if d != 3:
        return None
    ys, wt = _ts_grid(d, M, h, n_ang)
    gkey = (("3d", int(M), float(h)) if n_ang is None
            else ("3d", int(M), float(h), int(n_ang)))
    ops, block = _TS_GRID_3D_OPS[gkey]
    classes = _ts_sheet_classes(A, ops, tables=tables)
    if len(classes) == len(ops):
        return None
    idx = np.concatenate([np.arange(r * block, (r + 1) * block)
                          for r, _ in classes])
    W = np.concatenate([
        np.add.reduce(np.stack([wt[m * block:(m + 1) * block]
                                for m in members]), axis=0)
        for _, members in classes])
    return idx, W, tuple(int(r) for r, _ in classes)


def _ts_compact_table(ys, labels, values, rule_key) -> np.ndarray:
    """``_compact_transform_at`` memoised across calls
    (``_TS_TRANSFORM_CACHE``); ``rule_key`` names the rule and the node
    subset the transform was taken on."""
    global _TS_TRANSFORM_BYTES
    key = (labels.tobytes(), values.tobytes(), labels.shape) + tuple(rule_key)
    hit = _TS_TRANSFORM_CACHE.get(key)
    if hit is not None:
        _TS_TRANSFORM_CACHE.move_to_end(key)
        return hit
    out = _compact_transform_at(ys, labels, values)
    out.setflags(write=False)
    nb = out.nbytes
    if nb <= TS_TRANSFORM_MAX_BYTES:
        while (_TS_TRANSFORM_CACHE
               and _TS_TRANSFORM_BYTES + nb > TS_TRANSFORM_MAX_BYTES):
            _, ev = _TS_TRANSFORM_CACHE.popitem(last=False)
            _TS_TRANSFORM_BYTES -= ev.nbytes
        _TS_TRANSFORM_CACHE[key] = out
        _TS_TRANSFORM_BYTES += nb
    return out


def _fixed_rule_classes(kernels, A, M, h, n_ang, fold=False):
    r"""The fixed rule on the full product AND on the two classes of its
    expansion that the exact-sum referee checks (see ``_TS_REFINED``).

    Writing each edge's transform as ``Z_e + C_e`` -- the power-law terms
    plus the compact part's trigonometric polynomial -- this returns
    ``(value, scale, p0, p1)``: the rule on ``prod_e (Z_e + C_e)``, the
    magnitude ``scale = sum_i |w_i| |prod_i|`` the self-check is measured
    against, the rule on the purely compact product ``prod_e C_e`` and
    the rule on the one-power-law class ``sum_e Z_e prod_{f != e} C_f``.
    ``value`` is the arithmetic the rule always had: the per-edge factor
    ``Z_e + C_e`` is formed and accumulated in the same order, so the
    pure-power-law kernel path is unchanged bit for bit.  ``fold=True``
    (the compact path only) sums over the representative sheets of
    ``_ts_fold`` instead of every node when the lattice and the tables
    allow it -- the same sum reassociated, so it must never be asked of
    the pure path.  The transform of each table is memoised across calls
    (``_ts_compact_table``): a uniform cycle carries one table on every
    edge, a pass carries the same few tables on every cycle, and once the
    Epstein tables are cached the transform is the rule's whole per-table
    cost.
    """
    A = np.asarray(A, dtype=np.float64)
    d = int(A.shape[0])
    A_star = np.linalg.inv(A).T
    ys, wt = _ts_grid(d, M, h, n_ang)
    sub = None
    if fold:
        sub = _ts_fold(A, M, h, n_ang, [(labels, values)
                                        for _, labels, values in kernels
                                        if len(values)])
    reps = None
    if sub is not None:
        idx, wt, reps = sub
        ys = ys[idx]
    n = len(wt)
    prod = np.ones(n, dtype=np.complex128)
    p0 = np.ones(n, dtype=np.complex128)
    p1 = np.zeros(n, dtype=np.complex128)
    rule_key = (d, int(M), float(h), n_ang, reps)
    for terms, labels, values in kernels:
        z = np.zeros(n, dtype=np.complex128)
        for coef, nu in terms:
            table = _ts_zeta_table(float(nu), A, A_star, M, h, n_ang)
            z += coef * (table if sub is None else table[idx])
        c = np.zeros(n, dtype=np.complex128)
        if len(values):
            c += _ts_compact_table(ys, labels, values, rule_key)
        p1 = p1 * c + p0 * z
        p0 = p0 * c
        prod = prod * (z + c)
    val = float(np.real(np.dot(wt, prod)))
    scale = float(np.dot(np.abs(wt), np.abs(prod)))
    return (val, scale,
            float(np.real(np.dot(wt, p0))), float(np.real(np.dot(wt, p1))))


def _zeta_circle_kernels_fixed(kernels, A, M, h, n_ang, return_scale=False):
    """The fixed-rule cycle value for per-edge transform data ``kernels``
    (a list of ``fourier_terms`` triples); with ``return_scale`` also the
    magnitude ``sum_i |w_i| |prod_i|`` the self-check is measured against."""
    val, scale, _p0, _p1 = _fixed_rule_classes(kernels, A, M, h, n_ang)
    return (val, scale) if return_scale else val


def _exact_compact_classes(kernels, A):
    r"""Exact Brillouin-zone integrals of the two highest-degree classes of
    ``prod_e (Z_e + C_e)``, as finite sums: ``(ex0, ex1)``.

    ``ex0 = int prod_e C_e`` counts the closed E-walks of table
    displacements, ``sum_{m_1 + ... + m_E = 0} prod_e v_{e, m_e}``;
    ``ex1 = int sum_e Z_e prod_{f != e} C_f = sum_e sum_x T_e(x) K_e(x)``
    with ``T_e`` the (E-1)-fold convolution of the other edges' tables and
    ``K_e(x) = sum_j c_ej |A x|^{-nu_ej}`` (``K_e(0) = 0``) the regularised
    power law whose transform ``Z_e`` is -- finite because ``T_e`` is.
    Both run by FFT on a ``(2 E R + 1)^d`` grid (``R`` the largest
    Chebyshev radius of a table), on which no E-walk of steps ``<= R``
    can wind and ``T_e`` is held without wrap.  A class with a factor that
    has no compact part is identically zero and is returned as ``0.0``.
    """
    A = np.asarray(A, dtype=np.float64)
    d = int(A.shape[0])
    E = len(kernels)
    R = max((int(np.max(np.abs(labels))) if len(values) else 0)
            for _, labels, values in kernels)
    n = 2 * E * R + 1
    hats = []
    for _, labels, values in kernels:
        if not len(values):
            hats.append(None)
            continue
        g = np.zeros((n,) * d, dtype=np.float64)
        g[tuple(labels[:, i] % n for i in range(d))] = values
        hats.append(fftn(g))
    ex0 = 0.0
    if all(hf is not None for hf in hats):
        acc = np.ones((n,) * d, dtype=np.complex128)
        for hf in hats:
            acc = acc * hf
        ex0 = float(np.mean(acc).real)
    # Balanced labels of the grid: index i <-> label i mod n (n is odd).
    ax = np.fft.fftfreq(n, 1.0 / n)
    X = np.stack(np.meshgrid(*([ax] * d), indexing="ij"), axis=-1).reshape(-1, d)
    dist = np.linalg.norm(X @ A.T, axis=1)
    nz = dist > 0.0
    ex1 = 0.0
    for e, (terms, _, _) in enumerate(kernels):
        others = [hats[f] for f in range(E) if f != e]
        if not terms or any(hf is None for hf in others):
            continue
        acc = np.ones((n,) * d, dtype=np.complex128)
        for hf in others:
            acc = acc * hf
        T = np.real(ifftn(acc)).reshape(-1)
        K = np.zeros(len(X), dtype=np.float64)
        for coef, nu in terms:
            K[nz] += float(coef) * dist[nz] ** (-float(nu))
        ex1 += float(np.dot(T, K))
    return ex0, ex1


def _zeta_circle_kernels(kernels, A) -> float:
    r"""``zeta_circle`` for a cycle whose edges carry Interaction-likes.

    Pure power-law bundles (several terms, any weights) use the shipped
    rule exactly as a uniform cycle does.  A compact part on any edge runs
    the base AND the refined rules (``_TS_REFINED``) and returns the
    refined value once two consecutive rules agree to ``_CYCLE_SELF_BAND``
    of the integrand's magnitude ``scale = sum_i |w_i| |prod_i|``; the
    base rule is never accepted on its own (the comment beside
    ``_TS_REFINED`` records the measurement that refuted the exact-class
    shortcut).  That band is an ABSOLUTE-error criterion on the
    Brillouin-zone integral,
    not a relative one on the returned value: an odd cycle whose transform
    product changes sign can have ``|value| << scale``, and the accepted
    absolute error is then ``_CYCLE_SELF_BAND * scale`` -- the right
    notion for a series coefficient that sums such terms, but a weaker
    relative statement about this one number.  When the whole ladder
    disagrees, the last rung is refereed by the exact finite sums of
    ``_exact_compact_classes`` (see ``_TS_REFINED`` for what that
    certifies and what it does not); if that refuses too,
    :class:`CycleQuadratureError` is raised, carrying the rung values,
    ``scale`` and the referee's verdict.

    Cost, once the Epstein tables are cached (per process; the shipped
    corpora carry a handful of distinct exponents): at d = 2 the first
    refined rung is 16 168 nodes, 0.3 s of Epstein table per distinct
    exponent and then about 2 ms per cycle.  At d = 3 it is 1 234 176
    nodes, 5.5 s per distinct exponent on the cubic cell (12 sheets
    collapse to one; a cell without sheet symmetry pays 12x that), and
    the per-cycle product is folded onto the sheets the lattice and every
    table leave invariant (``_ts_fold``, 12x on the cubic cell with shell
    tables).  The compact transform is memoised across calls per table
    and rule.
    """
    A = np.asarray(A, dtype=np.float64)
    d = int(A.shape[0])
    for k in kernels:
        if np.isfinite(k.tail_exponent) and not (k.tail_exponent > d):
            raise UnsupportedLatticeSumError(
                f"All edges must have tail exponent > d = {d}; got "
                f"{k.tail_exponent} for {k!r}.  A cycle with an exponent "
                f"<= d is not supported (see UnsupportedLatticeSumError)."
            )
    data = [k.fourier_terms(A) for k in kernels]
    has_compact = any(k.has_compact for k in kernels)
    A_star = np.linalg.inv(A).T
    if d == 1 or not USE_TANH_SINH or d > 3:
        if has_compact and d in (2, 3):
            raise NotImplementedError(
                "USE_TANH_SINH=False is the legacy float path only; a cycle "
                "with a compact part needs the fixed-rule ladder (the "
                "adaptive rule integrates on the shipped 14-node angular "
                "rule with no refinement and no self-check, measured "
                "1e-08..5e-04 off on dense radius-2 tables at d = 2)."
            )
        integrand = lambda *ys: _kernel_prod_at(data, A, A_star, np.array(ys))  # noqa: E731
        if d == 1:
            retry = lambda y: _kernel_prod_at(data, A, A_star, np.array([y]),  # noqa: E731
                                              direct_1d=True)
            # int_full_1d's absolute tolerance is sized for an integrand of
            # order one; a kernel's integrand is bounded by the product of
            # its transforms' sup norms, which can be any size (small
            # couplings, a stretched chain).  Scale the tolerance down with
            # it -- never up, so every kernel with scale >= 1 keeps the
            # historical tolerance, bit for bit.
            scale = _kernel_sup_bound_1d(data, A)
            epsabs = 1e-12 * min(1.0, scale) if scale > 0.0 else 1e-12
            return _int_full_1d_finite(integrand, retry, kernels, epsabs)
        if d == 2:
            return int_full_2d(integrand)
        if d == 3:
            return int_full_3d(integrand)
        raise NotImplementedError(
            f"zeta_circle only implemented for d=1,2,3 (got d={d})."
        )
    if not has_compact:
        return _zeta_circle_kernels_fixed(data, A, _TS_M, _TS_H, None)
    cur, scale, p0, p1 = _fixed_rule_classes(data, A, _TS_M, _TS_H, None,
                                             fold=True)
    values = [cur]
    for factor, n_ang in _TS_REFINED[d]:
        prev, prev_scale = cur, scale
        cur, scale, p0, p1 = _fixed_rule_classes(
            data, A, factor * _TS_M, _TS_H / factor, n_ang, fold=True,
        )
        values.append(cur)
        tol = _CYCLE_SELF_BAND * max(abs(prev), abs(cur), scale, prev_scale)
        if abs(cur - prev) <= tol:
            return cur
    # The ladder is exhausted.  Referee the last rung by exact arithmetic
    # on the two classes of the integrand with the highest trigonometric
    # degree (``_TS_REFINED`` says why that certifies the rest).  A cycle
    # with more than one edge lacking a compact part has its
    # highest-degree class outside both finite sums, so the referee
    # declines rather than accept on a vacuous check.
    n_pure = sum(1 for _, _, v in data if not len(v))
    tol = _CYCLE_SELF_BAND * max(abs(cur), scale)
    if n_pure <= 1:
        ex0, ex1 = _exact_compact_classes(data, A)
        res0, res1 = abs(p0 - ex0), abs(p1 - ex1)
        if res0 <= tol and res1 <= tol:
            return cur
        verdict = (
            f"refused the last rung: its purely compact class and its "
            f"one-power-law class miss their exact finite sums by {res0:.2e} "
            f"and {res1:.2e} against a tolerance of {tol:.2e}"
        )
    else:
        verdict = (
            f"declined: {n_pure} edges carry no compact part, so the "
            f"highest-degree class of the integrand is not a finite sum"
        )
    raise CycleQuadratureError(
        f"zeta_circle: no two consecutive fixed rules agree to "
        f"{_CYCLE_SELF_BAND:g} of the integrand's magnitude "
        f"scale = sum_i |w_i| |prod_i| = {scale:.6e} (an absolute-error "
        f"criterion on the Brillouin-zone integral, not a relative one on "
        f"the value) on a cycle of {len(kernels)} edges with a compact part "
        f"of Chebyshev radius {max(k.support_radius for k in kernels)} "
        f"(rung values {values!r}), and the exact-sum referee {verdict}.  "
        f"The quadrature is not converged for this kernel; evaluate the "
        f"block on the sigma-router instead."
    )


# ----------------------------
# Circle zeta
# ----------------------------

#: The direct d = 1 cycle sum is accepted once its tail bound is below this
#: fraction of the partial sum (~8.7e-19: under the partial sum's own
#: rounding).
_CYCLE_DIRECT_TOL = 2.0 ** -60
#: Largest truncation the direct d = 1 cycle sum may use.  Its cost is about
#: ``E^2 (2M)^2`` multiply-adds; cycles that would need more (slow tails,
#: small exponents) keep the quadrature, where they are accurate.
_CYCLE_DIRECT_MAX_M = 512


def _cycle_1d_partial(nus: np.ndarray, M: int) -> float:
    r"""``C_M = sum_{d_1 + ... + d_E = 0, 0 < |d_i| <= M} prod_i |d_i|^-nu_i``.

    A chain of convolutions of positive sequences, closed by a dot product
    with the last one (every sequence is even, so the E-fold convolution at
    zero is ``sum_d g(d) f_E(d)``): no term can cancel another.
    """
    d = np.abs(np.arange(-M, M + 1, dtype=np.float64))
    d[M] = np.inf                                   # d = 0 is excluded
    seqs = [d ** -float(nu) for nu in nus]
    g = seqs[0]
    for s in seqs[1:-1]:
        g = np.convolve(g, s)
    c = (len(g) - 1) // 2
    return math.fsum(g[c - M:c + M + 1] * seqs[-1])


def _cycle_1d_direct(nu_vec) -> float | None:
    r"""The d = 1 unit-chain cycle as a certified positive lattice sum.

    On the unit chain the cycle is the sum over compositions of zero,

        zeta_circle(nu, (1)) = sum_{d_1 + ... + d_E = 0, d_i != 0}
                               prod_i |d_i|^-nu_i,

    and the partial sum ``C_M`` over ``|d_i| <= M`` (:func:`_cycle_1d_partial`)
    has only positive terms.  A dropped tuple has some ``|d_i| > M``, and
    since the other ``E - 1`` entries sum to ``-d_i``, some ``|d_j| > M / (E -
    1)`` as well.  Summing over the pair ``(i, j)`` with the constraint
    dropped (every term is positive, so that only adds terms),

        0 <= C - C_M <= B(M) = sum_{i != j} t_i(M) t_j(M / (E - 1))
                               prod_{k != i, j} 2 zeta(nu_k),

    with ``t_i(K) = sum_{|d| > K} |d|^-nu_i = 2 zeta(nu_i, K + 1)`` (Hurwitz)
    and ``M / (E - 1)`` rounded down.  ``M`` doubles from 4 until ``B(M) <=
    _CYCLE_DIRECT_TOL * C_M``; since ``C <= C_4 + B(4)``, a cycle whose bound
    cannot get below ``_CYCLE_DIRECT_TOL (C_4 + B(4))`` by
    ``_CYCLE_DIRECT_MAX_M`` is rejected before any large convolution runs,
    and ``None`` sends it to the quadrature.

    WHY.  The momentum-space integral ``int_0^1 prod_i Z(nu_i, y) dy`` that
    the quadrature evaluates cancels: at large exponents ``Z ~ 2 cos 2 pi y``
    and the product integrates to (nearly) zero, so its relative error is
    about machine epsilon times ``int |f| / |int f|``.  Against 30-digit
    references for all 189 cycle shapes of the shipped corpora at nu in
    {1.1, ..., 11}, the quadrature was 1.5e-3 off for the (4, 4, 4)-bundle
    triangle at nu = 11 (exponents 44, value 3e-13 against an integrand of
    order 8), 9e-7 for (3, 4, 4), 1e-9 for (4, 4, 4) at nu = 6 -- and at
    most 6e-14 at nu <= 1.5, where it is kept.  Those errors reached the
    chain series coefficients at 8.8e-8 relative (1qp k = 0, nu = 11,
    c_10).  The direct sum is exactly the regime where the quadrature fails:
    large exponents, fast tails, small ``M``.
    """
    from scipy.special import zeta as hurwitz_zeta

    nus = np.asarray(nu_vec, dtype=np.float64).reshape(-1)
    E = len(nus)
    if E < 2 or not np.all(nus > 1.0) or not np.all(np.isfinite(nus)):
        return None
    l1 = np.array([2.0 * float(hurwitz_zeta(nu, 1.0)) for nu in nus])

    def tail_bound(M):
        far = [2.0 * float(hurwitz_zeta(nu, M + 1.0)) for nu in nus]
        K = M // (E - 1)
        near = [2.0 * float(hurwitz_zeta(nu, K + 1.0)) for nu in nus]
        terms = []
        for i in range(E):
            for j in range(E):
                if i != j:
                    rest = float(np.prod(np.delete(l1, [i, j]))) if E > 2 else 1.0
                    terms.append(far[i] * near[j] * rest)
        return math.fsum(terms)

    M = 4
    c4 = _cycle_1d_partial(nus, M)
    ceiling = c4 + tail_bound(M)             # C <= C_4 + B(4)
    while M <= _CYCLE_DIRECT_MAX_M and tail_bound(M) > _CYCLE_DIRECT_TOL * ceiling:
        M *= 2
    while M <= _CYCLE_DIRECT_MAX_M:
        c = c4 if M == 4 else _cycle_1d_partial(nus, M)
        if c > 0.0 and tail_bound(M) <= _CYCLE_DIRECT_TOL * c:
            return c
        M *= 2
    return None


def _zeta_circle_1d_rescaled(nu_vec, a: float) -> float:
    r"""The d = 1 power-law cycle on the chain ``A = (a)``, ``a != 1``.

    Every term of the lattice sum carries ``|a|^-nu_e`` per edge,

        zeta_circle(nu, (a)) = prod_e |a|^-nu_e * zeta_circle(nu, (1)),

    so the cycle is evaluated on the unit chain and rescaled exactly.  The
    quadrature must not see the stretched cell: its absolute tolerance
    (``int_full_1d``) is sized for an integrand of order one, and on
    ``(a)`` the integrand is ``|a|^-sum(nu)`` times smaller -- measured
    4.7e-4 relative off a 40-digit reference for ``[69, 1.5, 1.5]`` on
    ``(1.7)``, 7.4e-7 for ``[100, 2.5, 2.5]``, before this rescaling.  The
    factor is a product of per-edge powers (each correctly rounded), not
    ``|a|^-sum(nu)`` of a rounded sum, so it adds at most ``E`` roundings.
    A value outside the float64 range raises :class:`CycleQuadratureError`
    (the front-end then falls back to the sigma-router).
    """
    unit = zeta_circle(nu_vec, np.array([[1.0]]))
    val = unit
    try:
        for nu in np.asarray(nu_vec, dtype=np.float64).reshape(-1):
            val *= a ** -float(nu)
    except OverflowError:          # float ** raises where float * gives inf
        val = math.inf
    if not math.isfinite(val):
        raise CycleQuadratureError(
            f"zeta_circle: the d = 1 cycle {list(map(float, nu_vec))!r} on "
            f"the chain a = {a!r} is {unit!r} * a^-sum(nu), outside the "
            f"float64 range."
        )
    return val


def zeta_circle(nu_vec, A) -> float:
    """
    Circle zeta function for nu_vec and lattice matrix A in dimension d=1,2,3.

    Parameters
    ----------
    nu_vec : array-like
        Vector of nu values; must satisfy nu_i > d.  An entry may also be
        an :class:`~gzl.Interaction` (or a product of them, a
        parallel bundle): the cycle is then the Brillouin-zone integral of
        the product of the general per-edge transforms, see
        :func:`_zeta_circle_kernels`.  A plain power law given as an
        Interaction is the float it always was.
    A : array-like (d x d) or str
        The lattice matrix, whose columns are the primitive vectors; code
        uses A_star = inv(A)^T.  A lattice name -- ``"chain"``,
        ``"square"``, ``"triangular"``, ``"cubic"`` -- means that matrix.

    Raises
    ------
    ValueError
        ``A`` is not a finite, non-singular square matrix, or an exponent
        is NaN or ``-inf``.
    UnsupportedLatticeSumError
        An exponent is ``<= d``.
    UnsupportedRequestError
        ``d`` is not 1, 2 or 3, or an exponent is ``+inf`` (the
        nearest-neighbour limit, which :func:`gzl.evaluate_graph`
        evaluates).
    """
    # Before the Interaction branch below, which forwards A on.
    A = lattice_matrix(A, "zeta_circle")
    if A.shape[0] not in _CLOSED_FORM_DIMS:
        raise UnsupportedRequestError(
            f"zeta_circle is implemented for d = 1, 2, 3; got d = "
            f"{A.shape[0]}.  evaluate_graph evaluates a cycle in any "
            f"dimension."
        )
    if any(is_interaction(x) for x in nu_vec):
        from gzl.interaction import coerce_nu
        nu_tail, kernels = coerce_nu(list(nu_vec), len(nu_vec))
        if kernels is not None:
            return _zeta_circle_kernels(kernels, A)
        nu_vec = nu_tail  # every entry demoted to its plain exponent

    A = np.array(A, dtype=np.float64)
    dim = len(A)

    nu_vec = np.array(nu_vec, dtype=np.float64)
    _refuse_low_cycle_exponents(nu_vec, dim)

    A_star = np.linalg.inv(A).T

    def integrand(*ys):
        y_vec = np.array(ys, dtype=np.float64)
        return epstein_zeta_prod(nu_vec, A, A_star, y_vec)

    if dim == 1:
        a = abs(float(A[0, 0]))
        if a != 1.0:
            return _zeta_circle_1d_rescaled(nu_vec, a)
        direct = _cycle_1d_direct(nu_vec)
        if direct is not None:
            return direct
        retry = lambda y: epstein_zeta_prod(  # noqa: E731
            nu_vec, A, A_star, np.array([y], dtype=np.float64), direct_1d=True)
        return _int_full_1d_finite(integrand, retry, nu_vec)
    if dim == 2:
        if USE_TANH_SINH:
            return zeta_circle_tanh_sinh(nu_vec, A)
        return int_full_2d(integrand)
    if dim == 3:
        if USE_TANH_SINH:
            return zeta_circle_tanh_sinh(nu_vec, A)
        return int_full_3d(integrand)

    raise NotImplementedError(f"zeta_circle only implemented for d=1,2,3 (got d={dim}).")


# ----------------------------
# Exports
# ----------------------------

__all__ = [
    "zeta_circle",
    "epstein_zeta_prod",
    "int_full_1d",
    "int_full_2d",
    "int_full_3d",
    "int_triangle",
    "int_pyramid",
]
