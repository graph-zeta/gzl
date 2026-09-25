# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
r"""The vertex set is the edge support, on every public entry point.

Before the unification the four engines split three ways on one and the
same sparsely-labelled input: hybrid and slab used the edge support,
the tensor summed every gap label freely over the torus (a factor
``n**d`` each — measured ``36**6`` on the documented K4 example), the
box multiplied in ``(2L+1)**d`` per gap, and the router refused a
connected graph with a ``DisconnectedGraphError`` naming phantom
isolated nodes.  These tests pin the resolved convention:

* a sparsely-labelled graph and its order-preservingly relabelled
  contiguous twin are BIT-IDENTICAL through every engine — the
  normalisation feeds the engine the same arrays, so this is equality,
  not closeness;
* a source / terminal / root label that appears in no edge is refused
  with a message saying it is not a vertex;
* a NetworkX input CAN express an isolated vertex, and is refused —
  its lattice-sum factor diverges on the infinite lattice — including
  the top-labelled isolated node that used to vanish silently.
"""
from __future__ import annotations

import numpy as np
import pytest

from gzl import (
    DisconnectedGraphError,
    VertexOutOfRangeError,
    direct_sum_extrapolated,
    direct_sum_zero_momentum,
    evaluate_graph,
    graph_from_edges_uniform,
    graph_zero,
    graph_zeta_general,
    graph_zeta_general_at_zero,
    hybrid_zeta,
    slab_zeta,
)

from tests._env_gate import assert_pinned

# The documented example: a K4 on labels {0, 5, 7, 9} and its
# order-preserving contiguous twin (0,5,7,9 -> 0,1,2,3).
K4_SPARSE = [(0, 5), (5, 9), (0, 9), (0, 7), (5, 7), (9, 7)]
K4_CONTIG = [(0, 1), (1, 3), (0, 3), (0, 2), (1, 2), (3, 2)]
TRI_SPARSE = [(0, 5), (5, 9), (0, 9)]
TRI_CONTIG = [(0, 1), (1, 2), (0, 2)]


def _nu(edges, d):
    return np.full(len(edges), d + 0.5)


class TestSparseEqualsContiguousBitwise:
    def test_tensor_on_the_documented_k4_example(self):
        d, n = 2, 6
        A = np.eye(d)
        sparse = complex(graph_zeta_general_at_zero(
            np.array(K4_SPARSE), _nu(K4_SPARSE, d), A, n))
        contig = complex(graph_zeta_general_at_zero(
            np.array(K4_CONTIG), _nu(K4_CONTIG, d), A, n))
        assert sparse == contig     # bit-equal, not close: same arrays inside

    def test_tensor_terminal_labels_are_remapped(self):
        d, n = 1, 6
        A = np.eye(d)
        sparse = graph_zeta_general(
            np.array(TRI_SPARSE), _nu(TRI_SPARSE, d), A, n,
            source=0, terminals=(9,))
        contig = graph_zeta_general(
            np.array(TRI_CONTIG), _nu(TRI_CONTIG, d), A, n,
            source=0, terminals=(2,))
        np.testing.assert_array_equal(np.asarray(sparse), np.asarray(contig))

    def test_box_per_l_engine(self):
        d = 2
        A = np.eye(d)
        sparse = complex(direct_sum_zero_momentum(
            np.array(K4_SPARSE), _nu(K4_SPARSE, d), A, L=2))
        contig = complex(direct_sum_zero_momentum(
            np.array(K4_CONTIG), _nu(K4_CONTIG, d), A, L=2))
        assert sparse == contig

    def test_box_extrapolated(self):
        d = 1
        A = np.eye(d)
        kw = dict(L_list=(2, 3, 4), n_correction_terms=2)
        sparse = complex(direct_sum_extrapolated(
            np.array(TRI_SPARSE), _nu(TRI_SPARSE, d), A, **kw))
        contig = complex(direct_sum_extrapolated(
            np.array(TRI_CONTIG), _nu(TRI_CONTIG, d), A, **kw))
        assert sparse == contig

    def test_router(self):
        d = 1
        A = np.eye(d)
        sparse = float(evaluate_graph(
            np.array(TRI_SPARSE), d + 0.5, A, n_points=8))
        contig = float(evaluate_graph(
            np.array(TRI_CONTIG), d + 0.5, A, n_points=8))
        assert sparse == contig

    def test_sigma_max_algebra(self):
        d = 1
        A = np.eye(d)
        sparse = complex(graph_zero(graph_from_edges_uniform(
            np.array(TRI_SPARSE), d + 0.5, A, 8)))
        contig = complex(graph_zero(graph_from_edges_uniform(
            np.array(TRI_CONTIG), d + 0.5, A, 8)))
        assert sparse == contig

    def test_hybrid_and_slab_already_had_the_convention(self):
        # Regression: the two engines that were right all along stay so.
        d, n = 2, 6
        A = np.eye(d)
        assert complex(hybrid_zeta(np.array(K4_SPARSE),
                                   _nu(K4_SPARSE, d), A, n)) \
            == complex(hybrid_zeta(np.array(K4_CONTIG),
                                   _nu(K4_CONTIG, d), A, n))
        assert float(slab_zeta(K4_SPARSE, _nu(K4_SPARSE, d), A, n)[0]) \
            == float(slab_zeta(K4_CONTIG, _nu(K4_CONTIG, d), A, n)[0])

    def test_the_engines_agree_with_each_other_on_sparse_input(self):
        # The point of the unification: one graph, one value.  Engine-
        # to-engine agreement is to round-off (different associations),
        # not bitwise.
        d, n = 2, 6
        A = np.eye(d)
        t = complex(graph_zeta_general_at_zero(
            np.array(K4_SPARSE), _nu(K4_SPARSE, d), A, n)).real
        h = complex(hybrid_zeta(np.array(K4_SPARSE),
                                _nu(K4_SPARSE, d), A, n)).real
        s = float(slab_zeta(K4_SPARSE, _nu(K4_SPARSE, d), A, n)[0])
        np.testing.assert_allclose([t, s], h, rtol=1e-12)


class TestALabelInNoEdgeIsNotAVertex:
    def test_tensor_source(self):
        with pytest.raises(ValueError, match="not a vertex"):
            graph_zeta_general(
                np.array(K4_SPARSE), _nu(K4_SPARSE, 2), np.eye(2), 6,
                source=1)

    def test_tensor_terminal(self):
        with pytest.raises(ValueError, match="not a vertex"):
            graph_zeta_general(
                np.array(TRI_SPARSE), _nu(TRI_SPARSE, 1), np.eye(1), 6,
                source=0, terminals=(2,))

    def test_box_root(self):
        with pytest.raises(ValueError, match="not a vertex"):
            direct_sum_zero_momentum(
                np.array(K4_SPARSE), _nu(K4_SPARSE, 2), np.eye(2), L=2,
                root=1)

    def test_router_terminal(self):
        with pytest.raises(VertexOutOfRangeError, match="not a vertex"):
            evaluate_graph(np.array(TRI_SPARSE), 1.5, np.eye(1),
                           n_points=8, source=0, terminal=2,
                           momentum=np.zeros(1))

    def test_sigma_terminal(self):
        with pytest.raises(ValueError, match="not a vertex"):
            graph_from_edges_uniform(
                np.array(TRI_SPARSE), 1.5, np.eye(1), 8, s=0, t=1)

    def test_negative_labels_are_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            graph_zeta_general_at_zero(
                np.array([(-1, 0), (0, 2), (-1, 2)]),
                _nu(TRI_SPARSE, 1), np.eye(1), 6)


class TestANetworkxIsolatedNodeIsRefused:
    def _mg(self, extra_node):
        nx = pytest.importorskip("networkx")
        mg = nx.MultiGraph()
        for u, v in TRI_CONTIG:
            mg.add_edge(u, v, nu=1.5)
        mg.add_node(extra_node)
        return mg

    def test_interior_label(self):
        # Node 1 exists in edges; an EXTRA node inside the label range
        # cannot exist for a contiguous triangle, so use a 4-node graph:
        nx = pytest.importorskip("networkx")
        mg = nx.MultiGraph()
        for u, v in [(0, 1), (1, 3), (0, 3)]:
            mg.add_edge(u, v, nu=1.5)
        mg.add_node(2)
        with pytest.raises(DisconnectedGraphError, match="isolated"):
            evaluate_graph(mg, np.eye(1), n_points=8)

    def test_top_label_no_longer_vanishes_silently(self):
        # Before the unification this node simply disappeared from the
        # result (edge-list conversion cannot carry it) — the worst of
        # the three historical behaviours, because it was silent.
        mg = self._mg(extra_node=3)
        with pytest.raises(DisconnectedGraphError, match="isolated"):
            evaluate_graph(mg, np.eye(1), n_points=8)


class TestLabelNormalisationEdgeCases:
    """Regressions for edge cases of the label normalisation."""

    def test_node_insertion_order_does_not_move_router_values(self):
        # Inserting frontend graph nodes in bundle-key order moved values:
        # NetworkX traversal follows insertion order, so a contiguous
        # triangle whose vertices are first mentioned out of order
        # shifted 1 ULP (permuted nu_vec into zeta_circle).  Sorted
        # insertion IS the historical range(V) order; the value below is
        # the pre-unification router output, bit-exact.
        v = float(evaluate_graph(np.array([(2, 1), (0, 2), (0, 1)]),
                                 np.array([1.5, 1.7, 1.9]),
                                 np.eye(1), n_points=8))
        assert_pinned(v, "0x1.7cef19885b154p+1",
                      "triangle router value, sorted insertion")

    def test_a_sparse_networkx_multigraph_matches_the_edge_list_path(self):
        nx = pytest.importorskip("networkx")
        mg = nx.MultiGraph()
        for u, v in TRI_SPARSE:
            mg.add_edge(u, v, nu=1.5)
        via_mg = float(evaluate_graph(mg, np.eye(1), n_points=8))
        via_edges = float(evaluate_graph(np.array(TRI_SPARSE), 1.5,
                                         np.eye(1), n_points=8))
        assert via_mg == via_edges

    def test_an_out_of_range_isolated_nx_node_is_refused_as_isolated(self):
        # The old [0, n_nodes) contiguity gate fired first and masked the
        # isolated-node refusal with a misleading relabelling hint.
        nx = pytest.importorskip("networkx")
        mg = nx.MultiGraph()
        for u, v in TRI_CONTIG:
            mg.add_edge(u, v, nu=1.5)
        mg.add_node(7)
        with pytest.raises(DisconnectedGraphError, match="isolated"):
            evaluate_graph(mg, np.eye(1), n_points=8)

    def test_hybrid_refuses_a_terminal_that_is_no_vertex(self):
        # Was a bare KeyError from the dense core.
        with pytest.raises(ValueError, match="not a vertex"):
            hybrid_zeta(np.array(TRI_SPARSE), _nu(TRI_SPARSE, 1),
                        np.eye(1), 8, source=0, terminal=7,
                        momentum=np.array([0.25]))

    def test_hybrid_refuses_a_source_that_is_no_vertex(self):
        # Was silently replaced by the planner's pin.
        with pytest.raises(ValueError, match="not a vertex"):
            hybrid_zeta(np.array(TRI_SPARSE), _nu(TRI_SPARSE, 1),
                        np.eye(1), 8, source=1)

    def test_the_default_source_reaches_a_sparse_graph_without_label_0(self):
        # evaluate_graph's old default source=0 refused a vacuum graph
        # on labels {5, 7, 9} the caller never pinned; None resolves to
        # the smallest present label, and the vacuum value is
        # pin-independent.
        tri_no0 = [(5, 7), (7, 9), (5, 9)]
        v = float(evaluate_graph(np.array(tri_no0), 1.5, np.eye(1),
                                 n_points=8))
        ref = float(evaluate_graph(np.array(TRI_CONTIG), 1.5, np.eye(1),
                                   n_points=8))
        assert v == ref

    def test_an_empty_edge_list_keeps_its_documented_valueerror(self):
        # Running the relabel before the empty check raised IndexError
        # from the helper's empty-support indexing.
        with pytest.raises(ValueError, match="empty"):
            direct_sum_zero_momentum(np.empty((0, 2), dtype=int),
                                     np.empty(0), np.eye(1), L=2)
        with pytest.raises(ValueError, match="empty"):
            direct_sum_extrapolated(np.empty((0, 2), dtype=int),
                                    np.empty(0), np.eye(1),
                                    L_list=(2, 3, 4),
                                    n_correction_terms=2)

    def test_negative_labels_are_refused_by_the_router_too(self):
        with pytest.raises(VertexOutOfRangeError, match="non-negative"):
            evaluate_graph(np.array([(-1, 0), (0, 2), (-1, 2)]), 1.5,
                           np.eye(1), n_points=8)

    def test_graph_from_tw2_refuses_an_isolated_nx_node(self):
        nx = pytest.importorskip("networkx")
        from gzl import graph_from_tw2
        mg = nx.MultiGraph()
        for u, v in TRI_CONTIG:
            mg.add_edge(u, v, nu=1.5)
        mg.add_node(3)
        with pytest.raises(ValueError, match="isolated"):
            graph_from_tw2(mg, 0, 1, np.eye(1), 8)

    def test_direct_sum_self_loop_names_the_callers_label(self):
        with pytest.raises(ValueError, match="vertex 5"):
            direct_sum_zero_momentum(
                np.array([(0, 5), (5, 5), (0, 9)]), np.full(3, 1.5),
                np.eye(1), L=2)
