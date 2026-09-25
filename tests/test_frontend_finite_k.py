# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Finite-momentum tests for the topology-first front-end
:func:`gzl.evaluate_graph`.

Covers: terminal-argument parsing (single int, length-0 / length-1
array, length-≥-2 array → ``NotImplementedError``); momentum-argument
parsing and dispatch (``None`` ⇒ scalar at k = 0; scalar / ``(d,)``
ndarray ⇒ single-k float; 1-D / ``(N, d)`` ndarray ⇒ batch);
per-block correctness (bridge against ``epstein_zeta``; algebra and
tensor against the existing full-grid path); lattice inversion
symmetry; the default grid against single-k calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl import evaluate_graph


A1 = np.array([[1.0]])
A2 = np.array([[1.0, 0.0], [0.0, 1.0]])


# ---------------------------------------------------------------------------
# Terminal argument parsing
# ---------------------------------------------------------------------------

class TestTerminalParsing:
    """The new ``terminal`` parameter accepts ``None``, a single int,
    or a length-≤-1 array.  Length-≥-2 arrays raise."""

    def test_terminal_zero_length_array_is_vacuum(self):
        edges = np.array([[0, 1]], dtype=int)
        v_none = evaluate_graph(edges, 3.0, A1, terminal=None)
        v_len0 = evaluate_graph(edges, 3.0, A1, terminal=[])
        assert v_none == v_len0

    def test_terminal_int_and_length_one_array_match(self):
        # Use an explicit single-k momentum so the call returns float
        # rather than the default-grid array (the equality check is
        # then well-defined on scalars).
        edges = np.array([[0, 1]], dtype=int)
        v_int  = evaluate_graph(edges, 3.0, A1, terminal=1, momentum=0.0)
        v_arr  = evaluate_graph(edges, 3.0, A1, terminal=[1], momentum=0.0)
        v_arr2 = evaluate_graph(
            edges, 3.0, A1, terminal=np.array([1]), momentum=0.0,
        )
        assert v_int == v_arr == v_arr2

    def test_terminal_length_two_raises(self):
        edges = np.array([[0, 1]], dtype=int)
        with pytest.raises(NotImplementedError):
            evaluate_graph(edges, 3.0, A1, terminal=[0, 1])


# ---------------------------------------------------------------------------
# Momentum argument shape rules
# ---------------------------------------------------------------------------

class TestMomentumShapes:
    """`momentum` accepts None, scalar (d=1), (d,), (N,d) — returns
    float, float, or (N,) array accordingly."""

    def test_none_for_1qp_returns_grid(self):
        # 1qp + momentum=None ⇒ default full BZ grid.
        edges = np.array([[0, 1]], dtype=int)
        out = evaluate_graph(edges, 3.0, A1, terminal=1, n_points=32)
        assert isinstance(out, np.ndarray)
        assert out.shape == (32,)
        assert out.dtype == float

    def test_none_for_1qp_without_n_points_raises(self):
        edges = np.array([[0, 1]], dtype=int)
        from gzl import NPointsRequiredError
        with pytest.raises(NPointsRequiredError):
            evaluate_graph(edges, 3.0, A1, terminal=1)  # n_points=0

    def test_scalar_d1_returns_float(self):
        edges = np.array([[0, 1]], dtype=int)
        v = evaluate_graph(edges, 3.0, A1, terminal=1, momentum=0.25)
        assert isinstance(v, float)

    def test_d_length_1d_returns_float(self):
        edges = np.array([[0, 1]], dtype=int)
        v = evaluate_graph(
            edges, 3.0, A1, terminal=1, momentum=np.array([0.25]),
        )
        assert isinstance(v, float)

    def test_batch_returns_array(self):
        edges = np.array([[0, 1]], dtype=int)
        ks = np.array([0.0, 0.25, 0.5])
        out = evaluate_graph(
            edges, 3.0, A1, terminal=1, momentum=ks,
        )
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)

    def test_2d_batch_returns_array(self):
        edges = np.array([[0, 1]], dtype=int)
        ks = np.array([[0.0, 0.0], [0.25, 0.0], [0.5, 0.5]])
        out = evaluate_graph(
            edges, 3.0, A2, terminal=1, momentum=ks,
        )
        assert out.shape == (3,)

    def test_unsupported_momentum_shape_raises(self):
        edges = np.array([[0, 1]], dtype=int)
        with pytest.raises(ValueError):
            evaluate_graph(edges, 3.0, A1, terminal=1,
                           momentum=np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# Vacuum + momentum: shape follows momentum, value broadcasts ζ_G(0)
# ---------------------------------------------------------------------------

class TestVacuumBroadcast:
    """When `terminal == source` (or terminal is None) the result is
    k-independent; the output shape still follows `momentum` so user
    code can iterate over a corpus uniformly."""

    def test_vacuum_with_single_momentum_returns_float(self):
        edges = np.array([[0, 1]], dtype=int)
        v_none = evaluate_graph(edges, 3.0, A1)
        v_k    = evaluate_graph(edges, 3.0, A1, momentum=0.25)
        assert v_none == v_k

    def test_vacuum_with_batch_returns_constant_array(self):
        edges = np.array([[0, 1]], dtype=int)
        v_none = evaluate_graph(edges, 3.0, A1)
        out = evaluate_graph(
            edges, 3.0, A1, momentum=np.array([0.0, 0.25, 0.5]),
        )
        assert out.shape == (3,)
        assert np.allclose(out, v_none)

    def test_terminal_eq_source_is_vacuum_like(self):
        edges = np.array([[0, 1]], dtype=int)
        v_vac = evaluate_graph(edges, 3.0, A1)
        v_eq  = evaluate_graph(edges, 3.0, A1, source=0, terminal=0)
        assert v_vac == v_eq


# ---------------------------------------------------------------------------
# Per-block correctness at finite k
# ---------------------------------------------------------------------------

class TestBridgeAtFiniteK:
    """For a single-bridge graph, ζ_G(k) is exactly
    ``epstein_zeta(ν, A, 0, k_lattice).real``.  The block-cut router
    should fire the bridge closed-form path."""

    def test_bridge_matches_epstein_zeta(self):
        from epsteinlib import epstein_zeta
        edges = np.array([[0, 1]], dtype=int)
        nu = 3.0
        for k_frac in [0.0, 0.123, 0.25, 0.31415, 0.5]:
            v = evaluate_graph(edges, nu, A1, terminal=1, momentum=k_frac)
            ref = float(
                epstein_zeta(nu, A1, np.zeros(1), np.array([k_frac])).real
            )
            assert v == pytest.approx(ref, rel=1e-12, abs=1e-12), (
                f"k={k_frac}: v={v} vs ref={ref}"
            )


class TestSpineFactorisation:
    """Off-spine decorations contribute their k = 0 scalar; only
    on-spine blocks see the external momentum.  Cross-check by
    multiplying the on-spine bridge result against the off-spine
    cycle / bridge scalars."""

    def test_bridge_with_pendant_cycle_factorises(self):
        from epsteinlib import epstein_zeta
        from gzl import zeta_circle

        # Spine bridge 0-1; off-spine triangle pendant on vertex 1.
        edges = np.array([
            [0, 1],         # spine bridge
            [1, 2], [2, 3], [3, 1],   # triangle pendant at 1
        ], dtype=int)
        nu = 3.0
        k_frac = 0.123

        # Full graph at finite k:
        v = evaluate_graph(edges, nu, A1, terminal=1, momentum=k_frac)

        # Hand-computed factorisation:
        #   ζ_bridge(k) * ζ_triangle(0)
        bridge_k = float(
            epstein_zeta(nu, A1, np.zeros(1), np.array([k_frac])).real
        )
        triangle_0 = float(zeta_circle(np.full(3, nu), A1))
        ref = bridge_k * triangle_0

        assert v == pytest.approx(ref, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# Lattice inversion symmetry: ζ_G(k) == ζ_G(-k)
# ---------------------------------------------------------------------------

class TestInversionSymmetry:
    """Bravais lattices are closed under x → −x, so ζ_G(k) = ζ_G(−k)
    and the result is real.  The single-k path should recover both."""

    def test_bridge(self):
        edges = np.array([[0, 1]], dtype=int)
        for k in [0.123, 0.31415, -0.4]:
            v_pos = evaluate_graph(edges, 3.0, A1, terminal=1, momentum=k)
            v_neg = evaluate_graph(edges, 3.0, A1, terminal=1, momentum=-k)
            assert v_pos == pytest.approx(v_neg, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# The default grid against single-k calls
# ---------------------------------------------------------------------------

class TestTheGridAgreesWithSingleK:
    """The default BZ grid of a 1qp call agrees with single-k calls at
    the grid's own momenta."""

    def test_grid_recovers_single_k(self):
        # Grid entry at k_idx should match the single-k call at
        # k_frac = k_idx / n.
        edges = np.array([[0, 1]], dtype=int)
        n = 32
        grid = evaluate_graph(
            edges, 3.0, A1, terminal=1, n_points=n,
        )
        for k_idx in [0, 4, 8, 16]:
            k_frac = k_idx / n
            v = evaluate_graph(
                edges, 3.0, A1, terminal=1,
                momentum=np.array([k_frac]),
            )
            assert grid[k_idx] == pytest.approx(v, rel=1e-12, abs=1e-12), (
                f"k_idx={k_idx}: grid={grid[k_idx]} vs single-k={v}"
            )


class TestOnSpineDiagnostics:
    """On-spine blocks must tally the route they actually took.

    ``_block_at_finite_k`` used to return no diagnostics at all, and the
    caller guessed the route from the block's shape: nearest-neighbour
    blocks were billed as tensor, two-vertex blocks as bridges, and
    *everything else* as tensor.  So every finite-k direct-sum
    evaluation was counted under ``n_block_tensor``,
    ``n_block_direct_sum`` was structurally always zero on the spine,
    and ``max_tensor_block_tw`` was never updated there.

    That made any before/after comparison built on
    ``return_diagnostics=True`` wrong for the dense blocks on the
    spine, so it is pinned here.
    """

    A2 = np.eye(2)
    K4 = np.array(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=int,
    )

    def test_dense_spine_block_is_billed_to_the_torus(self):
        # Dense spine blocks take the torus; they used to take the
        # box.  At finite k this is the arm that matters most: the
        # k = 0 Richardson ladder is unavailable here (the tail carries
        # a cos(2πk·x) factor and stops being a power law), and the
        # shipped box ladder reaches only L = 8 — measured to leave the
        # order-11 1qp coefficient ~4% wrong in L2.
        _v, info = evaluate_graph(
            self.K4, 2.75, self.A2, source=0, terminal=1,
            n_points=4, return_diagnostics=True,
        )
        assert info["n_block_direct_sum"] == 0, info
        assert info["n_block_dense_torus"] == 1, info
        assert info["max_tensor_block_tw"] >= 3, info

    def test_dense_spine_block_still_reaches_the_box_on_request(self):
        _v, info = evaluate_graph(
            self.K4, 2.75, self.A2, source=0, terminal=1,
            n_points=4, dense_engine="direct_sum", return_diagnostics=True,
        )
        assert info["n_block_direct_sum"] == 1, info
        assert info["n_block_dense_torus"] == 0, info

    def test_bridge_on_spine_is_billed_as_a_bridge(self):
        edges = np.array([[0, 1]], dtype=int)
        _v, info = evaluate_graph(
            edges, 3.0, self.A2, source=0, terminal=1,
            n_points=4, return_diagnostics=True,
        )
        assert info["n_bridges"] == 1, info
        assert info["n_block_direct_sum"] == 0, info

    def test_counts_are_not_double_billed(self):
        """One block, one tally -- the caller must not add its own."""
        _v, info = evaluate_graph(
            self.K4, 2.75, self.A2, source=0, terminal=1,
            n_points=4, return_diagnostics=True,
        )
        routed = (
            info["n_block_direct_sum"] + info["n_block_tensor"]
            + info["n_block_algebra"] + info["n_bridges"]
            + info["n_simple_cycles"]
        )
        assert routed == 1, info
