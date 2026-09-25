# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""What the stable functions do with arguments they cannot use.

A malformed argument raises a plain ``ValueError`` or ``TypeError`` at the
boundary, which is not a ``GraphZetaError`` (see "API stability" in
DOCUMENTATION.md).  A well-formed request that gzl does not evaluate
raises a ``GraphZetaError`` subclass.  Before, several of these returned
a number: NaN for a NaN momentum, 0.0 for ``n_points = 2.5``, the value
of ``[[0, 1]]`` for the edge list ``[[0, 1, 2]]`` or ``[[0.5, 1]]``, and
the ``k = 0`` value of a bridge at ``k = 1e-35``.  Others failed deep
inside an engine, with an ``IndexError``, a ``LinAlgError``, a reshape
error or an ``AssertionError``, or did not return at all.
"""

from __future__ import annotations

import math
import signal

import numpy as np
import pytest
from scipy.special import gamma, zeta

import gzl
from gzl import (
    GraphZetaError,
    NPointsRequiredError,
    TopologyEvaluatorUnavailableError,
    UnsupportedRequestError,
    compute_series_coefficients,
    evaluate_corpus,
    evaluate_graph,
    zeta_circle,
)
from gzl.core import _CUSP_SCALED_Y2, _epstein_zeta_k
from epsteinlib import epstein_zeta

BRIDGE = [[0, 1]]
TRIANGLE = [[0, 1], [1, 2], [2, 0]]
DIAMOND = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]]
K4 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
SINGULAR = [[1.0, 1.0], [1.0, 1.0]]


def _malformed(exc_info):
    """The error is a plain ValueError / TypeError, not a GraphZetaError."""
    assert not isinstance(exc_info.value, GraphZetaError), exc_info.value


# ---------------------------------------------------------------------------
# (a) momentum
# ---------------------------------------------------------------------------

class TestNonFiniteMomentum:

    @pytest.mark.parametrize("k", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("edges,kw", [
        (BRIDGE, {}),                        # was a GraphZetaError about epsteinlib
        (TRIANGLE, {"n_points": 8}),         # was nan
    ])
    def test_single_k(self, k, edges, kw):
        with pytest.raises(ValueError, match="finite") as ei:
            evaluate_graph(edges, 3.0, "chain", terminal=1, momentum=k, **kw)
        _malformed(ei)

    def test_one_bad_entry_in_a_batch(self):
        with pytest.raises(ValueError, match="finite") as ei:
            evaluate_graph(TRIANGLE, 3.0, "square", terminal=1, n_points=8,
                           momentum=[[0.1, 0.2], [0.3, np.nan]])
        _malformed(ei)

    def test_a_vacuum_graph_refuses_it_too(self):
        # A vacuum graph does not read the momentum, and returned a number.
        with pytest.raises(ValueError, match="finite") as ei:
            evaluate_graph(TRIANGLE, 3.0, "chain", momentum=np.nan)
        _malformed(ei)

    def test_the_corpus_functions(self):
        for f in (compute_series_coefficients, evaluate_corpus):
            with pytest.raises(ValueError, match="finite") as ei:
                f("tfim1qp", 3.0, "chain", 8, order_max=2,
                  momentum=[0.1, np.inf])
            _malformed(ei)


# ---------------------------------------------------------------------------
# (b) the lattice matrix
# ---------------------------------------------------------------------------

class TestLatticeMatrix:

    @pytest.mark.parametrize("A,match", [
        (SINGULAR, "singular"),     # nan for a bridge, LinAlgError for a triangle
        ([[0.0]], "singular"),
        (1.0, "square"),            # IndexError
        ([1.0], "square"),
        ([[1.0, 0.0]], "square"),   # epsteinlib's message
        ([[np.nan]], "finite"),
        ([[1.0, np.inf], [0.0, 1.0]], "finite"),
    ])
    @pytest.mark.parametrize("edges", [BRIDGE, TRIANGLE])
    def test_evaluate_graph(self, A, match, edges):
        with pytest.raises(ValueError, match=match) as ei:
            evaluate_graph(edges, 3.0, A, n_points=8)
        _malformed(ei)

    def test_zeta_circle(self):
        with pytest.raises(ValueError, match="singular") as ei:
            zeta_circle([3.0] * 3, SINGULAR)
        _malformed(ei)

    def test_the_corpus_functions(self):
        for f in (compute_series_coefficients, evaluate_corpus):
            with pytest.raises(ValueError, match="singular") as ei:
                f("tfim0qp", 3.0, SINGULAR, 8, order_max=2)
            _malformed(ei)

    def test_a_valid_matrix_is_used_as_given(self):
        A = np.array([[2.0]])
        assert evaluate_graph(BRIDGE, 3.0, A) == evaluate_graph(
            BRIDGE, 3.0, [[2]])
        assert evaluate_graph(BRIDGE, 3.0, A) == pytest.approx(
            2 * zeta(3.0) / 8.0, rel=1e-14)


# ---------------------------------------------------------------------------
# (c) edges and vertex labels
# ---------------------------------------------------------------------------

class TestEdgesAndLabels:

    @pytest.mark.parametrize("edges,match", [
        ([[0, 1, 2]], "shape"),          # evaluated [[0, 1]]
        ([0, 1, 2], "shape"),            # odd flat form: failed in a reshape
        ([[[0, 1]]], "shape"),
        ([[0, 1], [1]], r"\(E, 2\)"),
        ([[0.5, 1]], "whole"),           # truncated to [[0, 1]]
        ([[0, np.nan]], "whole"),
    ])
    def test_malformed_edge_lists(self, edges, match):
        with pytest.raises(ValueError, match=match) as ei:
            evaluate_graph(edges, 3.0, "chain")
        _malformed(ei)

    @pytest.mark.parametrize("edges", [[[True, False]], [["a", "b"]]])
    def test_labels_that_are_not_numbers(self, edges):
        with pytest.raises(TypeError) as ei:
            evaluate_graph(edges, 3.0, "chain")
        _malformed(ei)

    def test_whole_float_labels_are_the_integers(self):
        assert (evaluate_graph([[0.0, 1.0], [1.0, 2.0]], 3.0, "chain")
                == evaluate_graph([[0, 1], [1, 2]], 3.0, "chain"))

    def test_the_flat_form_is_the_edge_list(self):
        assert (evaluate_graph([0, 1, 1, 2, 2, 0], 3.0, "chain")
                == evaluate_graph(TRIANGLE, 3.0, "chain"))

    def test_the_empty_graph_is_still_one(self):
        assert evaluate_graph([], 3.0, "chain") == 1.0

    @pytest.mark.parametrize("kw", [
        {"source": 0.5},
        {"terminal": 1.5, "momentum": 0.1},
        {"terminal": [1.5], "momentum": 0.1},
    ])
    def test_fractional_vertex_references(self, kw):
        with pytest.raises(ValueError, match="whole number") as ei:
            evaluate_graph(TRIANGLE, 3.0, "chain", n_points=8, **kw)
        _malformed(ei)

    def test_a_boolean_vertex_reference(self):
        with pytest.raises(TypeError) as ei:
            evaluate_graph(TRIANGLE, 3.0, "chain", source=True)
        _malformed(ei)

    def test_whole_float_references_are_the_integers(self):
        kw = dict(momentum=0.1, n_points=8)
        assert (evaluate_graph(TRIANGLE, 3.0, "chain", source=0.0,
                               terminal=np.float64(1.0), **kw)
                == evaluate_graph(TRIANGLE, 3.0, "chain", source=0,
                                  terminal=1, **kw))


# ---------------------------------------------------------------------------
# (d) n_points
# ---------------------------------------------------------------------------

class TestGridSize:

    @pytest.mark.parametrize("n,exc", [
        (2.5, ValueError),               # evaluated at n_points = 2
        (-1, ValueError),
        (True, TypeError),
        ("8", TypeError),
    ])
    def test_malformed(self, n, exc):
        with pytest.raises(exc) as ei:
            evaluate_graph(DIAMOND, 3.0, "chain", n_points=n)
        _malformed(ei)

    def test_the_corpus_functions(self):
        for f in (compute_series_coefficients, evaluate_corpus):
            with pytest.raises(ValueError, match="whole number") as ei:
                f("tfim0qp", 3.0, "chain", 8.5, order_max=2)
            _malformed(ei)

    @pytest.mark.parametrize("n", [1, 2])
    def test_a_torus_that_cannot_hold_the_block(self, n):
        # The diamond returned exactly 0.0 at n_points = 1 and 2: its
        # triangles need three distinct sites.
        with pytest.raises(NPointsRequiredError, match="n_points >= 3"):
            evaluate_graph(DIAMOND, 3.0, "chain", n_points=n)

    def test_k4_needs_four_sites(self):
        # 1.5e-16, a zero in round-off, at n_points = 3 on the chain.
        with pytest.raises(NPointsRequiredError, match="n_points >= 4"):
            evaluate_graph(K4, 3.0, "chain", n_points=3)
        assert evaluate_graph(K4, 3.0, "chain", n_points=4) > 0.0

    def test_the_1qp_grid(self):
        # A triangle on the spine returned [0.] at n_points = 1.
        with pytest.raises(NPointsRequiredError):
            evaluate_graph(TRIANGLE, 3.0, "chain", terminal=1, n_points=1)

    def test_an_odd_cycle_on_the_spine_at_two_sites(self):
        C5 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
        with pytest.raises(NPointsRequiredError, match="n_points >= 3"):
            evaluate_graph(C5, 3.0, "chain", terminal=2, momentum=0.1,
                           n_points=2)
        C4 = [[0, 1], [1, 2], [2, 3], [3, 0]]
        assert evaluate_graph(C4, 3.0, "chain", terminal=2, momentum=0.1,
                              n_points=2) != 0.0

    def test_a_grid_that_holds_the_block_is_evaluated(self):
        # Coarse, but a truncation and not a zero: the diamond fits on 3
        # sites of the chain and on 2 x 2 of the square.
        assert evaluate_graph(DIAMOND, 3.0, "chain", n_points=3) > 0.0
        assert evaluate_graph(DIAMOND, 4.0, "square", n_points=2) > 0.0
        # A bipartite block of 54 vertices fits on 2 sites.
        M54 = ([(i, (i + 1) % 54) for i in range(54)]
               + [(i, i + 27) for i in range(27)])
        from gzl.frontend import _torus_holds
        import networkx as nx
        assert _torus_holds(nx.Graph(M54), 2)

    def test_closed_forms_need_no_grid(self):
        grid = evaluate_graph(BRIDGE, 3.0, "chain", terminal=1, n_points=1)
        assert grid.shape == (1,)
        assert grid[0] == pytest.approx(2 * zeta(3.0), rel=1e-15)
        assert evaluate_graph(TRIANGLE, 3.0, "chain", n_points=1) == (
            evaluate_graph(TRIANGLE, 3.0, "chain"))

    def test_an_integral_float_is_the_integer(self):
        assert (evaluate_graph(DIAMOND, 3.0, "chain", n_points=8.0)
                == evaluate_graph(DIAMOND, 3.0, "chain", n_points=8))


# ---------------------------------------------------------------------------
# (e) n_points = 0 where a grid is needed
# ---------------------------------------------------------------------------

class TestNoGrid:

    @pytest.mark.parametrize("edges,t", [(TRIANGLE, 1), (DIAMOND, 2)])
    @pytest.mark.parametrize("k", [0.1, [0.1, 0.2]])
    def test_a_finite_momentum(self, edges, t, k):
        # "cannot reshape array of size 1 into shape (0,)" or an
        # AssertionError from tensor_network before.
        with pytest.raises(NPointsRequiredError):
            evaluate_graph(edges, 3.0, "chain", terminal=t, momentum=k)

    def test_a_closed_form_spine_needs_none(self):
        v = evaluate_graph([[0, 1], [1, 2]], 3.0, "chain", terminal=2,
                           momentum=0.1)
        assert math.isfinite(v)

    def test_the_1qp_corpus_on_its_grid(self):
        # A plain ValueError, where evaluate_graph raises this.
        with pytest.raises(NPointsRequiredError):
            compute_series_coefficients("tfim1qp", 3.0, "chain", 0)



# ---------------------------------------------------------------------------
# (f) unsupported request shapes
# ---------------------------------------------------------------------------

class TestUnsupportedRequest:

    def test_the_class(self):
        assert issubclass(UnsupportedRequestError, GraphZetaError)
        assert issubclass(UnsupportedRequestError, NotImplementedError)
        assert gzl.UnsupportedRequestError is UnsupportedRequestError

    def test_two_free_terminals(self):
        with pytest.raises(UnsupportedRequestError):
            evaluate_graph(TRIANGLE, 3.0, "chain", terminal=[1, 2],
                           momentum=0.1, n_points=8)

    def test_nu_inf_beside_finite_exponents_at_finite_k(self):
        with pytest.raises(UnsupportedRequestError, match="mixes") as ei:
            evaluate_graph(TRIANGLE, [np.inf, 3.0, 3.0], "chain", terminal=1,
                           momentum=0.1, n_points=8)
        # The name of this one says a dependency is missing.
        assert not isinstance(ei.value, TopologyEvaluatorUnavailableError)


# ---------------------------------------------------------------------------
# (g) cycles beyond d = 3
# ---------------------------------------------------------------------------

class TestCyclesBeyondThreeDimensions:

    A4 = np.eye(4)

    def test_evaluate_graph_takes_the_sigma_router(self):
        v, info = evaluate_graph(TRIANGLE, 5.0, self.A4, n_points=4,
                                 return_diagnostics=True)
        assert v == evaluate_graph(TRIANGLE, 5.0, self.A4, n_points=4,
                                   fast_cycles=True)
        assert info["n_simple_cycles"] == 0
        assert info["n_block_cycle_fallback"] == 1

    def test_the_block_cache_agrees(self):
        cache: dict = {}
        first = evaluate_graph(TRIANGLE, 5.0, self.A4, n_points=4,
                               block_cache=cache)
        again = evaluate_graph([[5, 6], [6, 7], [7, 5]], 5.0, self.A4,
                               n_points=4, block_cache=cache)
        assert first == again

    def test_zeta_circle_refuses(self):
        with pytest.raises(UnsupportedRequestError, match="d = 1, 2, 3"):
            zeta_circle([5.0] * 3, self.A4)


# ---------------------------------------------------------------------------
# (h) the cycle closed form at nu = inf
# ---------------------------------------------------------------------------

class TestCycleAtInfinity:

    def _no_hang(self, f):
        def alarm(*_):
            raise TimeoutError("did not return within 10 s")
        old = signal.signal(signal.SIGALRM, alarm)
        signal.alarm(10)
        try:
            return f()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    def test_plus_infinity_is_unsupported(self):
        with pytest.raises(UnsupportedRequestError, match="finite"):
            self._no_hang(lambda: zeta_circle([np.inf] * 3, "chain"))

    def test_minus_infinity_is_malformed(self):
        with pytest.raises(ValueError, match="-inf") as ei:
            zeta_circle([-np.inf, 3.0, 3.0], "chain")
        _malformed(ei)


# ---------------------------------------------------------------------------
# (i) a bridge at a tiny momentum
# ---------------------------------------------------------------------------

def _chain_bridge_near_zero(nu, k):
    """``sum_{x != 0} exp(-2 pi i k x) |x|^-nu`` for ``0 < k << 1``.

    Twice the real part of ``Li_nu(exp(2 pi i k))``, from Lindelöf's
    expansion ``Li_s(e^mu) = Gamma(1 - s) (-mu)^(s - 1) + sum_n
    zeta(s - n) mu^n / n!``.  The ``n = 1`` term is imaginary and the next
    real one is of order ``k**2``, so at ``k <= 1e-30`` the first two real
    terms are the value to round-off.  Independent of epsteinlib and of
    the form :func:`gzl.core._epstein_zeta_k` uses.
    """
    return (2 * gamma(1 - nu) * (2 * np.pi * k) ** (nu - 1)
            * math.cos(math.pi * (1 - nu) / 2) + 2 * zeta(nu))


class TestTinyMomentum:

    @pytest.mark.parametrize("nu", [0.5, 0.9, 1.01, 1.5])
    @pytest.mark.parametrize("k", [1e-33, 1e-35, 1e-60, -1e-35])
    def test_the_chain_bridge(self, nu, k):
        # epsteinlib returns the k = 0 value below |k| ~ 1e-32: -18.86
        # for the exact 49431.86 at nu = 0.9, k = 1e-35, and 83 % off at
        # nu = 1.01.
        want = _chain_bridge_near_zero(nu, abs(k))
        got = evaluate_graph(BRIDGE, nu, "chain", terminal=1, momentum=k)
        assert got == pytest.approx(want, rel=1e-13)

    @pytest.mark.parametrize("t,ratio", [
        (1e-31, 10.0),      # below gzl's threshold, above epsteinlib's snap
        (1e-35, 1000.0),    # below the snap
    ])
    def test_it_continues_epsteinlib_below_the_threshold(self, t, ratio):
        # For nu < d the cusp dominates, so |y|^(nu - d) sets the ratio of
        # a value epsteinlib evaluates (at 1e-29) to one the decomposition
        # evaluates.  The constant term is 1e-15 of either.
        A = np.array([[1.0, 0.5], [0.0, math.sqrt(3) / 2]])
        u = np.array([0.6, 0.8])
        nu = 1.5
        hi = evaluate_graph(BRIDGE, nu, A, terminal=1, momentum=1e-29 * u)
        lo = evaluate_graph(BRIDGE, nu, A, terminal=1, momentum=t * u)
        assert lo / hi == pytest.approx(ratio, rel=1e-12)

    @pytest.mark.parametrize("nu", [0.9, 2.5, 3.5])
    def test_epsteinlib_bit_for_bit_elsewhere(self, nu):
        A = np.eye(2)
        z = np.zeros(2)
        for y in (z, np.array([1e-20, 0.0]), np.array([0.3, 0.1])):
            assert _epstein_zeta_k(nu, A, y) == epstein_zeta(nu, A, z, y)
        # nu >= d + 1: the cusp is below 1e-30 of the scale.
        y = np.array([1e-40, 0.0])
        assert (1e-40) ** 2 < _CUSP_SCALED_Y2
        assert _epstein_zeta_k(3.0, A, y) == epstein_zeta(3.0, A, z, y)

    def test_the_interaction_bridge(self):
        P = gzl.Interaction.power_law(0.9)
        assert P.lattice_sum("chain", 1e-35) == pytest.approx(
            _chain_bridge_near_zero(0.9, 1e-35), rel=1e-13)
