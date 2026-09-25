# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The cluster-cut truncation exponent.

The torus/box truncation tail is governed by the slowest *escaping
cluster*, not only by the slowest single vertex: a connected set S of
free vertices can leave together, stretching only the edges that cut S.
``_min_free_cut_nu`` minimises that cut weight; the old single-vertex
rule is its |S| = 1 special case and can therefore only over-estimate.
"""
import numpy as np
import pytest

from gzl.direct_sum import _min_free_cut_nu, _min_free_incident_nu

K4 = {(0, 1): 1.0, (0, 2): 1.0, (0, 3): 1.0,
      (1, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0}


def _scaled(edge_map, nu):
    return {k: v * nu for k, v in edge_map.items()}


class TestClusterCutExponent:

    def test_never_exceeds_single_vertex_rule(self):
        """σ_cut ≤ σ_vertex: singletons are among the candidates."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            nv = int(rng.integers(4, 8))
            em = {}
            for u in range(nv):
                for v in range(u + 1, nv):
                    if rng.random() < 0.55:
                        em[(u, v)] = float(rng.integers(1, 4)) * 1.75
            verts = {x for e in em for x in e}
            if len(verts) < nv:                     # keep it connected-ish
                continue
            assert _min_free_cut_nu(em, 0) <= _min_free_incident_nu(em, 0) + 1e-12

    def test_k4_agrees_with_single_vertex(self):
        """K4: the full free set cuts the same 3 edges as any vertex."""
        em = _scaled(K4, 1.75)
        assert _min_free_cut_nu(em, 0) == pytest.approx(
            _min_free_incident_nu(em, 0))

    def test_detects_a_cluster_the_vertex_rule_misses(self):
        """A heavy-degree vertex pair joined by a heavy internal edge.

        Every single vertex is expensive to move, but the PAIR escapes
        across only the two light edges binding it to the pin.
        """
        em = {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 9.0}
        assert _min_free_incident_nu(em, 0) == pytest.approx(10.0)
        assert _min_free_cut_nu(em, 0) == pytest.approx(2.0)

    def test_externals_cannot_escape(self):
        """Vertices named external are pinned and never join a cluster.

        A triangle 0-1-2 with a pendant 3 on vertex 2: the pendant is the
        cheapest escape, cutting one edge.  Named external it cannot
        leave, and the cheapest cluster is {1}, cutting two.
        """
        em = _scaled({(0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0, (2, 3): 1.0},
                     1.75)
        assert _min_free_cut_nu(em, 0) == pytest.approx(1.75)
        assert _min_free_cut_nu(em, 0, externals={3}) == pytest.approx(3.5)
