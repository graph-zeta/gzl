# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""A graph ``evaluate_graph`` refuses stops a corpus pass with its own
exception.

The corpus front-ends used to catch five of the router's refusals and
hand the graph to a whole-graph ``k = 0`` fallback cascade (the tensor
at ``nu >= d + 2``, the sigma_max algebra below, the tensor or the box
for treewidth > 2).  No refusal it caught had a number to give:

* a disconnected graph came back as ONE component's value from the
  algebra (2*zeta(2.5) = 2.683 for two bridges at 2.5 on the chain, a
  complex-typed value, not even the product 7.198 of the components)
  and as a torus volume from the tensor (39.69 / 80.93 / 162.39 at
  n_points = 8 / 16 / 32, nu = 3.5);
* a self-loop came back as a plain ``ValueError`` from the constructor
  or a ``NotImplementedError`` from the tensor, not ``SelfLoopError``;
* ``n_points = 0`` with a block that needs a grid ended in an
  ``AssertionError`` inside the tensor ("axis derivation broke");
* a 1qp graph at finite momentum was re-raised as
  ``TopologyEvaluatorUnavailableError`` with the cause dropped.

The shipped corpora never reached the cascade (the last test here pins
that), so removing it moves no shipped value.
"""

from __future__ import annotations

import re
import warnings

import h5py
import numpy as np
import pytest

import gzl
from gzl import (
    DisconnectedGraphError,
    NPointsRequiredError,
    SelfLoopError,
    VertexOutOfRangeError,
    compute_series_coefficients,
    evaluate_corpus,
)
from gzl.series import _load_corpus

TWO_BRIDGES = [[0, 1], [2, 3]]


def _corpus(tmp_path, edges, mults=None, hopping=None, name="g"):
    """A one-graph HDF5 corpus at order 2."""
    path = tmp_path / f"{name}.h5"
    with h5py.File(path, "w") as fh:
        g = fh.create_group("O2").create_group("graph_0")
        g.create_dataset("edges", data=np.asarray(edges, dtype=int))
        g.create_dataset("multiplicities", data=np.asarray(
            mults if mults is not None else np.ones(len(edges)), dtype=int))
        g.create_dataset("prefactor", data=np.float64(1.0))
        if hopping is not None:
            g.create_dataset("hopping", data=np.asarray(hopping, dtype=int))
    return path


class TestRefusalsPropagate:

    # 0.9 is below d (the bundle guard of the old cascade refused it),
    # 2.5 took the algebra, 3.5 the tensor.
    @pytest.mark.parametrize("nu", [0.9, 2.5, 3.5])
    @pytest.mark.parametrize("mult", [1, 2])
    @pytest.mark.parametrize("front_end", [evaluate_corpus,
                                           compute_series_coefficients])
    def test_a_disconnected_graph(self, tmp_path, nu, mult, front_end):
        path = _corpus(tmp_path, TWO_BRIDGES, [mult, mult])
        with pytest.raises(DisconnectedGraphError):
            front_end(path, nu, "chain", 16)

    @pytest.mark.parametrize("momentum", [None, 0.0, 0.25])
    def test_a_disconnected_1qp_graph_keeps_its_exception(self, tmp_path,
                                                          momentum):
        # It was re-raised as TopologyEvaluatorUnavailableError, a
        # missing-dependency error, with the real cause dropped.
        path = _corpus(tmp_path, TWO_BRIDGES, hopping=[0, 1])
        with pytest.raises(DisconnectedGraphError):
            compute_series_coefficients(path, 2.5, "chain", 8,
                                        momentum=momentum)

    @pytest.mark.parametrize("edges", [[[0, 1], [1, 1]],
                                       [[0, 1], [1, 2], [0, 2], [2, 2]]],
                             ids=["bridge+loop", "triangle+loop"])
    @pytest.mark.parametrize("nu", [2.5, 3.5])
    def test_a_self_loop(self, tmp_path, edges, nu):
        path = _corpus(tmp_path, edges)
        with pytest.raises(SelfLoopError):
            evaluate_corpus(path, nu, "chain", 8)

    def test_n_points_zero_with_a_block_that_needs_a_grid(self):
        # Order 6 of the shipped 0qp corpus has the first such block.
        # This ended in an AssertionError inside the tensor.
        with pytest.raises(NPointsRequiredError) as exc:
            compute_series_coefficients("tfim0qp", 3.0, "chain", 0,
                                        order_max=6)
        notes = getattr(exc.value, "__notes__", [])
        assert any(re.fullmatch(r"corpus graph \S+ at order 6, nu = 3", n)
                   for n in notes), notes

    def test_a_source_that_is_no_vertex(self, tmp_path):
        # A vacuum graph is pinned at label 0, and this one has none.
        path = _corpus(tmp_path, [[1, 2]])
        with pytest.raises(VertexOutOfRangeError):
            evaluate_corpus(path, 3.5, "chain", 8)

    def test_a_terminal_that_is_no_vertex(self, tmp_path):
        path = _corpus(tmp_path, [[0, 1]], hopping=[0, 5])
        with pytest.raises(VertexOutOfRangeError) as exc:
            evaluate_corpus(path, 2.5, "chain", 8, momentum=0.0)
        assert "corpus graph graph_0 at order 2, nu = 2.5, s = 0, t = 5" \
            in exc.value.__notes__

    def test_the_note_carries_an_interaction_label(self, tmp_path):
        V = gzl.Interaction(b=[1.0, -0.3], nu=[3.0, 5.0], label="two-term")
        path = _corpus(tmp_path, TWO_BRIDGES)
        with pytest.raises(DisconnectedGraphError) as exc:
            evaluate_corpus(path, V, "chain", 8)
        assert "corpus graph graph_0 at order 2, two-term" \
            in exc.value.__notes__

    def test_the_cli_names_the_graph(self, tmp_path, capsys):
        from gzl import cli
        argv = ["series", "--corpus", "tfim0qp", "--A", "chain",
                "--n-points", "0", "--order-max", "6", "--nu", "3.0",
                "--output", str(tmp_path / "o.csv")]
        assert cli.main(argv) == 1
        err = capsys.readouterr().err
        assert "gzl: error: n_points > 0 is required" in err
        assert re.search(r"gzl: \(corpus graph \S+ at order 6, nu = 3\)",
                         err)
        assert "Traceback" not in err


class TestDeprecatedSettings:
    """The settings of the removed cascade are accepted, ignored and
    reported."""

    SETTINGS = {
        "nu_tensor_threshold": 0.5,
        "tw_threshold": 3,
        "sigma_max": 0.0,
        "high_tw_fallback": "not a route",     # was validated, lazily
        "direct_sum_L_list": (4, 5),
        "direct_sum_K": 1,
    }

    @pytest.mark.parametrize("name", sorted(SETTINGS))
    def test_compute_series_coefficients(self, name):
        ref = compute_series_coefficients("tfim0qp", 3.0, "chain", 8,
                                          order_max=4)
        with pytest.warns(DeprecationWarning,
                          match=f"{name} has no effect") as rec:
            got = compute_series_coefficients(
                "tfim0qp", 3.0, "chain", 8, order_max=4,
                **{name: self.SETTINGS[name]})
        # It points at the caller, not into gzl.
        ours = [w for w in rec if "no effect" in str(w.message)]
        assert len(ours) == 1 and ours[0].filename == __file__
        assert got.keys() == ref.keys()
        for o in ref:
            assert float(got[o]).hex() == float(ref[o]).hex()

    @pytest.mark.parametrize("name", sorted(set(SETTINGS) - {"tw_threshold"}))
    def test_evaluate_corpus(self, name):
        ref = evaluate_corpus("tfim0qp", 3.0, "chain", 8, order_max=4)
        with pytest.warns(DeprecationWarning, match=f"{name} has no effect"):
            got = evaluate_corpus("tfim0qp", 3.0, "chain", 8, order_max=4,
                                  **{name: self.SETTINGS[name]})
        assert [(r["graph_id"], float(r["value"]).hex()) for r in got] \
            == [(r["graph_id"], float(r["value"]).hex()) for r in ref]

    def test_one_warning_names_them_all(self):
        with pytest.warns(DeprecationWarning) as rec:
            compute_series_coefficients("tfim0qp", 3.0, "chain", 8,
                                        order_max=2, **self.SETTINGS)
        ours = [w for w in rec if "no effect" in str(w.message)]
        assert len(ours) == 1
        assert ("nu_tensor_threshold, tw_threshold, sigma_max, "
                "high_tw_fallback, direct_sum_L_list, direct_sum_K have "
                "no effect") in str(ours[0].message)

    def test_no_setting_no_warning(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            compute_series_coefficients("tfim0qp", 3.0, "chain", 8,
                                        order_max=2)
            evaluate_corpus("tfim0qp", 3.0, "chain", 8, order_max=2)
        assert not [w for w in rec if "no effect" in str(w.message)]


@pytest.mark.parametrize("name", ["tfim0qp", "tfim1qp"])
def test_the_shipped_corpora_hold_no_graph_the_router_refuses(name):
    """Every shipped graph is loop-free and connected, and its pins are
    vertices: the removed cascade never evaluated one of them."""
    corpus = _load_corpus(name)
    n = 0
    for blk in corpus.values():
        hopping = blk.get("hopping")
        for i, edges in enumerate(blk["edges_list"]):
            edges = np.asarray(edges)
            assert np.all(edges[:, 0] != edges[:, 1])
            parent = {int(v): int(v) for v in np.unique(edges)}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for u, v in edges:
                parent[find(int(u))] = find(int(v))
            assert len({find(v) for v in parent}) == 1
            s, t = ((int(hopping[i][0]), int(hopping[i][1]))
                    if hopping is not None else (0, 0))
            assert s in parent and t in parent
            n += 1
    assert n == {"tfim0qp": 8403, "tfim1qp": 22677}[name]
