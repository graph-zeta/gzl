# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The distinct treewidth >= 3 blocks of the shipped TFIM corpora.

Every graph of both corpora is Hadamard-merged (parallel edges become
one bundle whose weight ``w`` is the summed multiplicity) and split
into its biconnected blocks.  Blocks of treewidth >= 3 are deduplicated
by WL hash with an isomorphism collision guard on ``w``, mirroring the
frontend's block cache.  The census tests of the planner and of the
tau*d acceptance sweep run over this set.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
from networkx.algorithms.approximation import treewidth_min_fill_in

from gzl import data_path

#: The shipped corpora live in the package data directory.
DATA_DIR = data_path()
CORPORA = ("tfim_softcore_corpus_0qp.npz", "tfim_softcore_corpus_1qp.npz")


def _iter_corpus_graphs(path):
    f = np.load(path, allow_pickle=True)
    edges_off = np.asarray(f["edges_off"]).astype(int)
    edges_flat = np.asarray(f["edges_flat"]).astype(int)
    mults = np.asarray(f["multiplicities"]).astype(int)
    for i in range(len(edges_off) - 1):
        s, e = edges_off[i], edges_off[i + 1]
        yield edges_flat[s:e], mults[s:e]


def _merged_graph(edges, mults):
    """Simple graph with per-bundle total multiplicity as edge attr w."""
    g = nx.Graph()
    for (u, v), m in zip(edges.tolist(), mults.tolist()):
        u, v = int(u), int(v)
        if u == v:
            continue
        if g.has_edge(u, v):
            g[u][v]["w"] += int(m)
        else:
            g.add_edge(u, v, w=int(m))
    return g


def _block_key_graph(block_nodes, g, keep_mults):
    sub = g.subgraph(block_nodes).copy()
    if not keep_mults:
        for _, _, dta in sub.edges(data=True):
            dta["w"] = 1
    return sub


def enumerate_blocks(corpus_dir, keep_mults):
    """Distinct tw >= 3 blocks over both corpora.

    Returns ``(blocks, n_instances)``: a list of ``{"g", "count", "tw"}``
    records deduplicated by WL hash with an isomorphism collision guard
    (edge attr ``w``), most frequent first, and the number of block
    occurrences they stand for.
    """
    buckets: dict = {}
    n_instances = 0
    for name in CORPORA:
        path = Path(corpus_dir) / name
        if not path.is_file():
            # A missing corpus used to be a warning and an empty census --
            # which the callers without a liveness assert then "passed".
            raise FileNotFoundError(f"corpus not found: {path}")
        for edges, mults in _iter_corpus_graphs(path):
            g = _merged_graph(edges, mults)
            if g.number_of_nodes() == 0:
                continue
            for comp in nx.biconnected_components(g):
                if len(comp) < 4:
                    continue
                sub = _block_key_graph(comp, g, keep_mults)
                tw, _ = treewidth_min_fill_in(sub)
                if tw < 3:
                    continue
                n_instances += 1
                h = nx.weisfeiler_lehman_graph_hash(
                    sub, edge_attr="w", iterations=4,
                )
                found = False
                for entry in buckets.setdefault(h, []):
                    if nx.is_isomorphic(
                        entry["g"], sub,
                        edge_match=lambda a, b: a["w"] == b["w"],
                    ):
                        entry["count"] += 1
                        found = True
                        break
                if not found:
                    buckets[h].append({"g": sub, "count": 1, "tw": tw})
    blocks = [e for bl in buckets.values() for e in bl]
    blocks.sort(key=lambda e: (-e["count"], e["g"].number_of_nodes()))
    return blocks, n_instances
