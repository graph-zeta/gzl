# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""The exceptions shared by the front-end and the modules beneath it.

``GraphZetaError`` used to be defined in :mod:`gzl.frontend`; it
lives here so that :mod:`gzl.interaction` (imported before the
front-end) can derive its own errors from it without a circular import.
``UnsupportedLatticeSumError`` and ``UnsupportedRequestError`` live here
for the same reason: the closed forms in :mod:`gzl.circle` and the
constructors in :mod:`gzl.construction` raise them too.  The front-end
re-exports them under the same names, so ``except gzl.GraphZetaError``
and ``frontend.GraphZetaError`` are unchanged.
"""


class GraphZetaError(ValueError):
    """Base class of the errors gzl raises when a graph or a request lies
    outside what it evaluates.

    Part of the stable API (see "API stability" in DOCUMENTATION.md), as
    are its subclasses raised by the stable functions.  It is a
    ``ValueError``, so an ``except ValueError`` catches it too.  A
    malformed argument, such as an array of the wrong shape, raises a
    plain ``ValueError`` or ``TypeError`` instead.
    """


class UnsupportedLatticeSumError(GraphZetaError):
    """Raised for an exponent ``nu <= d`` that gzl does not evaluate.

    The lattice sum converges absolutely when ``nu_e > d`` on every
    edge, and beyond that the graph zeta function is defined by
    meromorphic continuation in the exponents.  gzl evaluates that
    continuation for a BRIDGE block, whose value is an Epstein zeta
    function, and raises this error at its pole ``nu = d`` for a momentum
    in the reciprocal lattice (``k = 0`` included), and within ``2**-30``
    of the pole, where epsteinlib returns NaN for a finite value.  Every other block with a bundle exponent
    ``nu <= d``, after parallel edges are merged, raises this error at
    any momentum.  A bundle's exponent is the SUM over its parallel
    edges, so two parallel edges of ``nu = 0.6`` on the chain form one
    edge of ``1.2`` and are evaluated.

    This is a SUPPORT limit, not a divergence proof, and the distinction
    matters because the obvious reading is wrong.  ``Re nu_e > d`` on
    every edge is *sufficient* for absolute convergence, not necessary:
    what governs a graph zeta is the cluster cut
    ``I_S = sum_{e in cut(S)} nu_e`` over escaping free-vertex sets, which
    :func:`gzl._elimination.min_free_cut_nu` already computes.  K_4
    at ``d = 1`` has cut 2.10 at ``nu = 0.7`` and 2.40 at ``nu = 0.8`` --
    both clear ``d``, so by that criterion those sums CONVERGE, and this
    guard refuses them anyway.  So does the triangle at ``nu = 0.9`` on
    the chain (cut 1.8).

    What is true is that SOME sums with ``nu <= d`` diverge and no engine
    here can tell which.  ``direct_sum_extrapolated`` refuses
    ``nu_min <= d`` outright; the torus engines have no guard of their
    own and would sum a FINITE torus.  For K_4 at d = 1, nu = 0.8 (cut
    2.40 > d) the torus does converge, but slowly: 5.86, 10.89, 13.62 at
    n_points = 6, 10, 14 and 17.98, 19.73, 20.55 at 32, 64, 128, against
    about 21.16 from a Richardson-extrapolated brute-force box sum.  No
    accuracy gate is calibrated for that regime, and calibrating one is
    the prerequisite for widening this rule.

    Gating on the cluster cut rather than ``min nu`` would admit a real
    class of high-connectivity blocks.  That is unmeasured and
    deliberately not attempted here.
    """


class UnsupportedRequestError(GraphZetaError, NotImplementedError):
    """Raised for a well-formed request that gzl does not evaluate.

    The arguments are valid, but the combination asks for something no
    evaluator implements: more than one free terminal, a 1qp graph whose
    path from ``source`` to ``terminal`` runs through a block that mixes
    ``nu = inf`` with finite exponents, or the cycle closed form outside
    ``d = 1, 2, 3`` or at ``nu = inf``.  It is a ``GraphZetaError``, so
    ``except ValueError`` catches it, and also a ``NotImplementedError``,
    which the first and the third of these raised before, so an
    ``except NotImplementedError`` keeps working.

    A malformed argument, such as a NaN momentum or a singular lattice
    matrix, raises a plain ``ValueError`` or ``TypeError`` instead.
    """
