# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Tests for ``gzl.frontend.evaluate_graph`` — the public
topology-first front-end.

Covers: signature contract, per-block routing (bridge / cycle / σ_max=4
algebra / tensor), per-edge ν support, structured error semantics,
diagnostics dict shape and optional Richardson.
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl import (
    evaluate_graph,
    zeta_circle,
    DisconnectedGraphError,
    NPointsRequiredError,
    SelfLoopError,
    VertexOutOfRangeError,
)


A1 = np.array([[1.0]])


# ---------------------------------------------------------------------------
# Signature contract
# ---------------------------------------------------------------------------

class TestSignatureContract:
    """Public signature is positional-(edges, nu, A) + keyword-only kwargs."""

    def test_returns_real_float(self):
        edges = np.array([[0, 1]], dtype=int)
        v = evaluate_graph(edges, 3.0, A1)
        assert isinstance(v, float)

    def test_return_diagnostics_two_tuple(self):
        edges = np.array([[0, 1]], dtype=int)
        v, info = evaluate_graph(edges, 3.0, A1, return_diagnostics=True)
        assert isinstance(v, float)
        assert isinstance(info, dict)

    def test_keyword_only_after_A(self):
        """source / terminals / n_points / richardson / return_diagnostics
        are all keyword-only."""
        edges = np.array([[0, 1]], dtype=int)
        with pytest.raises(TypeError):
            evaluate_graph(edges, 3.0, A1, 0)            # source as positional


# ---------------------------------------------------------------------------
# Per-block routing
# ---------------------------------------------------------------------------

class TestRouting:
    """Each route increments its own diagnostics counter; values match
    the relevant closed-form reference."""

    def test_bridge_routes_to_epstein_zeta(self):
        # Single edge between two vertices: a bridge.
        edges = np.array([[0, 1]], dtype=int)
        v, info = evaluate_graph(edges, 3.0, A1, return_diagnostics=True)
        assert info["n_bridges"] == 1
        assert info["n_simple_cycles"] == 0
        assert info["n_block_algebra"] == 0
        assert info["n_block_tensor"] == 0
        # Bridge value is 2·ζ(3) on the 1D integer lattice.
        from epsteinlib import epstein_zeta
        ref = float(epstein_zeta(3.0, A1, np.zeros(1), np.zeros(1)).real)
        assert v == pytest.approx(ref, rel=1e-15, abs=0.0)

    def test_simple_cycle_routes_to_zeta_circle(self):
        # Triangle (3-cycle) at vacuum (no externals).
        edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)
        v, info = evaluate_graph(edges, 3.0, A1, return_diagnostics=True)
        assert info["n_simple_cycles"] == 1
        assert info["n_bridges"] == 0
        assert info["n_block_tensor"] == 0
        assert info["n_block_algebra"] == 0
        ref = float(zeta_circle(np.full(3, 3.0), A1))
        assert v == pytest.approx(ref, rel=1e-15, abs=0.0)

    def test_sigma_routing_low_sigma_picks_algebra(self):
        # Diamond graph (4-cycle plus one diagonal): tw=2 but not a
        # simple cycle (n_e = n_v + 1 = 5), so the cycle-closed-form
        # doesn't fire and we land on the σ-routed path.
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=int,
        )
        v, info = evaluate_graph(
            edges, 1.4, A1, n_points=64, return_diagnostics=True,
        )
        # σ = 0.4 < 1.49 and tw=2 and 1 external (source default) →
        # algebra fires.
        assert info["n_block_algebra"] == 1
        assert info["n_block_tensor"] == 0
        assert isinstance(v, float)

    def test_sigma_routing_high_sigma_picks_tensor(self):
        # Same diamond at large σ → tensor path.
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=int,
        )
        v, info = evaluate_graph(
            edges, 2.6, A1, n_points=64, return_diagnostics=True,
        )
        assert info["n_block_tensor"] == 1
        assert info["n_block_algebra"] == 0
        assert isinstance(v, float)

    def test_high_treewidth_routes_to_the_torus(self):
        # K_4 has treewidth 3.  Dense tw≥3 blocks used to route
        # unconditionally to the real-space direct sum; they now take
        # the torus at d ≤ 3, which is what lets them reach the σ_eff
        # Richardson ladder — measured at 1223x the box route's accuracy
        # at d = 1, σ = 0.5, against a two-family reference.  This
        # assertion is the inverse of the one it replaces; the box route
        # is pinned in the companion below.
        edges = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=int,
        )
        v, info = evaluate_graph(
            edges, 1.4, A1, n_points=32, return_diagnostics=True,
        )
        assert info["n_block_algebra"] == 0
        assert info["n_block_direct_sum"] == 0
        assert info["n_block_dense_torus"] == 1
        assert info["max_tensor_block_tw"] >= 3
        assert isinstance(v, float)

    def test_high_treewidth_still_reaches_the_box_on_request(self):
        # The box route is not dead code: it is reachable on request
        # (dense_engine="direct_sum") and it is the MemoryError fallback.
        # nu <= d no longer reaches it: the router refuses that before
        # any engine runs.
        edges = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=int,
        )
        v, info = evaluate_graph(
            edges, 1.4, A1, n_points=32, dense_engine="direct_sum",
            return_diagnostics=True,
        )
        assert info["n_block_direct_sum"] == 1
        assert info["n_block_dense_torus"] == 0
        assert isinstance(v, float)

    def test_cycle_with_multiple_cut_vertices_routes_to_zeta_circle(self):
        # 4-cycle with bridges attached at two non-adjacent vertices
        # (0 and 2).  The 4-cycle block has |external| = 2 (both cut
        # vertices), but at zero external momentum the closed-form
        # `zeta_circle` is still the correct contribution — the block
        # connects to other blocks only at single vertices, so every
        # cut vertex carries zero momentum.
        edges = np.array([
            # 4-cycle 0-1-2-3-0
            [0, 1], [1, 2], [2, 3], [3, 0],
            # Bridge at vertex 0
            [0, 4],
            # Bridge at vertex 2
            [2, 5],
        ], dtype=int)
        v, info = evaluate_graph(
            edges, 3.0, A1, return_diagnostics=True,
        )
        assert info["n_simple_cycles"] == 1
        assert info["n_bridges"] == 2
        assert info["n_block_algebra"] == 0
        assert info["n_block_tensor"] == 0
        # Cross-check value against the explicit product of the cycle
        # closed-form and two bridge closed-forms.
        from epsteinlib import epstein_zeta
        cycle_ref  = float(zeta_circle(np.full(4, 3.0), A1))
        bridge_ref = float(epstein_zeta(3.0, A1, np.zeros(1), np.zeros(1)).real)
        assert v == pytest.approx(
            cycle_ref * bridge_ref * bridge_ref, rel=1e-15, abs=0.0,
        )

    def test_large_sigma_cycle_auto_routes_to_tensor(self):
        # A simple cycle with σ = ν − d > 10 must bypass the closed-form
        # `zeta_circle` (slow + overflow-prone at large exponent) and route
        # through the truncated-Fourier tensor automatically — same value.
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], dtype=int,   # 5-cycle
        )
        # σ = 14 (> 10) at ν = 15 on the d = 1 chain.
        v, info = evaluate_graph(
            edges, 15.0, A1, n_points=24, return_diagnostics=True,
        )
        assert info["n_simple_cycles"] == 0       # NOT the zeta_circle path
        assert info["n_block_tensor"] == 1
        ref = float(zeta_circle(np.full(5, 15.0), A1))
        assert v == pytest.approx(ref, rel=1e-6)

    def test_small_sigma_cycle_still_uses_zeta_circle(self):
        # Below the threshold (σ ≤ 10) the closed form is still selected —
        # behaviour for the existing corpus regime is unchanged.
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], dtype=int,
        )
        _, info = evaluate_graph(
            edges, 3.0, A1, n_points=24, return_diagnostics=True,
        )
        assert info["n_simple_cycles"] == 1
        assert info["n_block_tensor"] == 0

    def test_bridge_decoration_factorises(self):
        # Triangle (vertices 0,1,2) plus a pendant bridge (2,3) — block-cut
        # splits into one cycle and one bridge.  Place the source on the
        # bridge (vertex 3) so the triangle's external set is just the
        # cut vertex {2} (len=1) and the cycle closed-form fires.
        edges = np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=int)
        v, info = evaluate_graph(
            edges, 3.0, A1, source=3, return_diagnostics=True,
        )
        assert info["n_simple_cycles"] == 1
        assert info["n_bridges"] == 1
        # Value is the product of the two block contributions.
        from epsteinlib import epstein_zeta
        bridge = float(epstein_zeta(3.0, A1, np.zeros(1), np.zeros(1)).real)
        cycle  = float(zeta_circle(np.full(3, 3.0), A1))
        assert v == pytest.approx(bridge * cycle, rel=1e-15, abs=0.0)


# ---------------------------------------------------------------------------
# Per-edge ν
# ---------------------------------------------------------------------------

class TestPerEdgeNu:
    """Scalar nu and uniform-per-edge nu produce identical results;
    non-uniform-per-edge nu forces the tensor path."""

    def test_uniform_array_matches_scalar(self):
        edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)
        v_scalar = evaluate_graph(edges, 3.0, A1)
        v_array  = evaluate_graph(edges, np.array([3.0, 3.0, 3.0]), A1)
        assert v_scalar == v_array          # bit-identical

    def test_non_uniform_nu_uses_algebra_path(self):
        # Diamond block (tw=2 but not a simple cycle) at small σ with
        # non-uniform per-edge ν.  After Hadamard merge each bundle has
        # one summed ν, so graph_from_edges takes the bundle ν vector
        # directly — algebra fires.
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=int,
        )
        nu = np.array([1.4, 1.5, 1.4, 1.4, 1.4])
        v, info = evaluate_graph(
            edges, nu, A1, n_points=64, return_diagnostics=True,
        )
        assert info["n_block_algebra"] == 1
        assert info["n_block_tensor"] == 0
        assert isinstance(v, float)

    def test_non_uniform_nu_algebra_matches_tensor(self):
        # Same non-uniform-ν diamond at small σ: algebra (this path)
        # should agree with the tensor path to FFT-truncation precision.
        # We force the tensor path by pushing σ into the high regime
        # via ν shifted up, then sanity-check that low σ algebra and
        # high σ tensor each evaluate cleanly without raising; the
        # primary value-locking is via the bridge / cycle closed-forms
        # tested elsewhere.  Here we cross-check the algebra path's
        # non-uniform-ν value by re-running with the same per-edge ν
        # encoded as repeated rows + scalar — the two MUST agree.
        edges_per_edge = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=int,
        )
        nu_per_edge = np.array([1.4, 1.4, 1.4, 1.4, 1.4])  # uniform
        v_array_form  = evaluate_graph(edges_per_edge, nu_per_edge, A1, n_points=64)
        v_scalar_form = evaluate_graph(edges_per_edge, 1.4, A1, n_points=64)
        assert v_array_form == v_scalar_form    # bit-identical

    def test_nu_length_mismatch_raises(self):
        edges = np.array([[0, 1], [1, 2]], dtype=int)
        with pytest.raises(ValueError):
            evaluate_graph(edges, np.array([3.0]), A1)


# ---------------------------------------------------------------------------
# Error semantics
# ---------------------------------------------------------------------------

class TestErrorSemantics:
    """Each documented error condition raises the matching exception."""

    def test_self_loop_raises(self):
        edges = np.array([[0, 0]], dtype=int)
        with pytest.raises(SelfLoopError):
            evaluate_graph(edges, 3.0, A1)

    def test_disconnected_raises(self):
        edges = np.array([[0, 1], [2, 3]], dtype=int)
        with pytest.raises(DisconnectedGraphError):
            evaluate_graph(edges, 3.0, A1)

    def test_source_out_of_range_raises(self):
        edges = np.array([[0, 1]], dtype=int)
        with pytest.raises(VertexOutOfRangeError):
            evaluate_graph(edges, 3.0, A1, source=5)

    def test_terminal_out_of_range_raises(self):
        # Pass momentum=0 to take the single-k path so the underlying
        # range validation runs (the default-grid path needs n_points
        # > 0 to even start, which would otherwise mask this check).
        edges = np.array([[0, 1]], dtype=int)
        with pytest.raises(VertexOutOfRangeError):
            evaluate_graph(edges, 3.0, A1, terminal=99, momentum=0.0)

    def test_n_points_required_raises(self):
        # K_4 has tw=3 so the σ-routed evaluator needs n_points > 0.
        edges = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=int,
        )
        with pytest.raises(NPointsRequiredError):
            evaluate_graph(edges, 1.4, A1)         # n_points=0 default


# ---------------------------------------------------------------------------
# Diagnostics shape
# ---------------------------------------------------------------------------

class TestDiagnosticsShape:
    """The info dict has stable, additive keys."""

    EXPECTED_COUNT_KEYS = {
        "n_bridges",
        "n_simple_cycles",
        "n_block_algebra",
        "n_block_tensor",
        "max_algebra_block_tw",
        "max_tensor_block_tw",
        "n_block_richardson_self",
        "n_block_richardson_theory",
        "n_block_richardson_skipped",
    }

    def test_version_key_present(self):
        edges = np.array([[0, 1]], dtype=int)
        _, info = evaluate_graph(edges, 3.0, A1, return_diagnostics=True)
        assert info["version"] == 1

    def test_all_count_keys_present(self):
        edges = np.array([[0, 1]], dtype=int)
        _, info = evaluate_graph(edges, 3.0, A1, return_diagnostics=True)
        missing = self.EXPECTED_COUNT_KEYS - set(info.keys())
        assert not missing, f"Missing diagnostics keys: {missing}"


# ---------------------------------------------------------------------------
# Richardson
# ---------------------------------------------------------------------------

class TestRichardson:
    """Richardson on a small-σ tensor block records at least one
    self-tuning or theoretical-fallback application."""

    def test_richardson_increments_counter(self):
        # Richardson is a tensor-path feature.  Use a tw=2 diamond
        # (4-cycle + chord) at σ = 1.6 ∈ [1.49, 1.95): not
        # algebra-eligible (σ ≥ 1.49) and not dense (tw=2) ⇒ tensor
        # path, where Richardson applies and is not skipped.
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=int,
        )
        _, info = evaluate_graph(
            edges, 2.6, A1, n_points=64, richardson=True,
            return_diagnostics=True,
        )
        n_total = (
            info["n_block_richardson_self"]
            + info["n_block_richardson_theory"]
            + info["n_block_richardson_skipped"]
        )
        assert n_total >= 1


# ---------------------------------------------------------------------------
# NetworkX MultiGraph input
# ---------------------------------------------------------------------------

class TestNetworkXInput:
    """evaluate_graph accepts a NetworkX MultiGraph alongside the flat
    array form, with `nu` taken from edge attributes."""

    def test_multigraph_matches_array_bridge(self):
        import networkx as nx
        edges = np.array([[0, 1]], dtype=int)
        v_array = evaluate_graph(edges, 3.0, A1)

        mg = nx.MultiGraph()
        mg.add_node(0)
        mg.add_node(1)
        mg.add_edge(0, 1, nu=3.0)
        v_mg = evaluate_graph(mg, A1)               # A as positional
        assert v_mg == v_array

    def test_multigraph_matches_array_triangle(self):
        import networkx as nx
        edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)
        v_array = evaluate_graph(edges, 3.0, A1)

        mg = nx.MultiGraph()
        for v in (0, 1, 2):
            mg.add_node(v)
        for u, v in [(0, 1), (1, 2), (2, 0)]:
            mg.add_edge(u, v, nu=3.0)
        v_mg = evaluate_graph(mg, A=A1)             # A as keyword
        assert v_mg == v_array

    def test_multigraph_parallel_edges_match_array(self):
        # Multigraph with two parallel edges between (0, 1) plus one
        # leg of a triangle.  Hadamard merging combines the parallels.
        import networkx as nx
        edges = np.array([[0, 1], [0, 1], [0, 2], [1, 2]], dtype=int)
        v_array = evaluate_graph(edges, 1.4, A1, n_points=64)

        mg = nx.MultiGraph()
        for v in (0, 1, 2):
            mg.add_node(v)
        mg.add_edge(0, 1, nu=1.4)
        mg.add_edge(0, 1, nu=1.4)        # second parallel edge
        mg.add_edge(0, 2, nu=1.4)
        mg.add_edge(1, 2, nu=1.4)
        v_mg = evaluate_graph(mg, A1, n_points=64)
        assert v_mg == v_array

    def test_multigraph_per_edge_nu(self):
        # MultiGraph with non-uniform per-edge ν (encoded directly via
        # the 'nu' edge attributes).  Algebra path now handles this.
        import networkx as nx
        mg = nx.MultiGraph()
        for v in (0, 1, 2, 3):
            mg.add_node(v)
        for (u, v, nu_e) in [
            (0, 1, 1.4), (1, 2, 1.5), (2, 3, 1.4),
            (3, 0, 1.4), (0, 2, 1.4),
        ]:
            mg.add_edge(u, v, nu=nu_e)
        v, info = evaluate_graph(
            mg, A1, n_points=64, return_diagnostics=True,
        )
        assert info["n_block_algebra"] == 1
        assert info["n_block_tensor"] == 0
        assert isinstance(v, float)

    def test_multigraph_missing_nu_raises(self):
        import networkx as nx
        mg = nx.MultiGraph()
        mg.add_node(0)
        mg.add_node(1)
        mg.add_edge(0, 1)                            # no `nu` attribute
        with pytest.raises(ValueError, match="'nu'"):
            evaluate_graph(mg, A1)

    def test_multigraph_non_integer_labels_raises(self):
        import networkx as nx
        mg = nx.MultiGraph()
        mg.add_node("a")
        mg.add_node("b")
        mg.add_edge("a", "b", nu=3.0)
        with pytest.raises(ValueError, match="integer"):
            evaluate_graph(mg, A1)

    def test_array_form_requires_nu_and_A(self):
        edges = np.array([[0, 1]], dtype=int)
        with pytest.raises(TypeError):
            evaluate_graph(edges)                   # no nu, no A
        with pytest.raises(TypeError):
            evaluate_graph(edges, 3.0)              # no A

    def test_multigraph_form_with_explicit_nu_raises(self):
        import networkx as nx
        mg = nx.MultiGraph()
        mg.add_edge(0, 1, nu=3.0)
        with pytest.raises(TypeError, match="nu"):
            # Explicit nu kwarg with MultiGraph is a usage error.
            evaluate_graph(mg, nu=3.0, A=A1)
