# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""Interaction sweeps through the corpus front-ends and the CLI: the
demoted power-law sweep is bit-identical to the float sweep, an
Interaction sweep equals the manual per-graph sum, the record and CSV
schemas gain the label column whenever the user passed an Interaction
(a labelled plain power law keeps its identity while it is routed as its
float), a generator / iterator sweep is materialised exactly once, the
1qp grid pass is pinned at k = 0 / on a batch / against the bridge closed
form, and the config forms are validated."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from gzl import Interaction, data_path, evaluate_graph
from gzl.series import (
    _coerce_sweep,
    _load_corpus,
    compute_series_coefficients,
    evaluate_corpus,
    main,
)

# The corpora ship with the package: a missing one is an error, not a skip.
CORPUS_0QP = data_path("tfim_softcore_corpus_0qp.npz")
CORPUS_1QP = data_path("tfim_softcore_corpus_1qp.npz")

A1 = np.eye(1)
V_NN_TAIL = Interaction.from_table({(1,): 0.4, (-1,): 0.4, (0,): 0.1}, b=[1.0], nu=[3.0],
                                   label="nn+tail")
V_TWO = Interaction(b=[1.0, -0.3], nu=[3.0, 5.0], label="two-term")


class TestCoerceSweep:
    def test_legacy_forms(self):
        arr, sweep, _scalar = _coerce_sweep(3.0)
        assert np.array_equal(arr, [3.0]) and sweep == [3.0]
        arr, sweep, _scalar = _coerce_sweep(np.array([3.0, 4.0]))
        assert np.array_equal(arr, [3.0, 4.0]) and sweep == [3.0, 4.0]
        with pytest.raises(ValueError, match="1-D"):
            _coerce_sweep(np.ones((2, 2)))

    def test_interaction_forms(self):
        """A sweep point is the object the user passed -- a plain power
        law included, so its label survives into the records; the tail
        array is what the thresholds read."""
        arr, sweep, _scalar = _coerce_sweep(V_NN_TAIL)
        assert sweep == [V_NN_TAIL] and np.array_equal(arr, [3.0])
        P4 = Interaction.power_law(4.0)
        arr, sweep, _scalar = _coerce_sweep([V_NN_TAIL, 3.5, P4])
        assert sweep[0] is V_NN_TAIL and sweep[1] == 3.5 and sweep[2] is P4
        assert np.array_equal(arr, [3.0, 3.5, 4.0])
        P25 = Interaction.power_law(2.5)
        arr, sweep, _scalar = _coerce_sweep(P25)
        assert sweep == [P25] and np.array_equal(arr, [2.5])

    def test_generator_iterator_map_and_object_ndarray_are_materialised_once(self):
        """The pre-fix ``any(is_interaction(x) for x in nu)`` probe consumed
        an iterator up to its first Interaction and the pass ran on the
        remainder (a 3-point generator sweep came back with 2 points, an
        ``iter([3.0, V1, V2])`` with 1) -- silently, with coefficient
        arrays of the wrong length."""
        P3 = Interaction.power_law(3.0)

        def shells(J):
            return Interaction.from_shells(A1, {1.0: J}, b=[1.0], nu=[3.0])

        for form in ((x for x in [V_NN_TAIL, V_TWO, 3.0]),
                     iter([3.0, V_NN_TAIL, V_TWO]),
                     map(shells, [0.1, 0.2, 0.3])):
            arr, sweep, _scalar = _coerce_sweep(form)
            assert len(sweep) == 3 and arr.shape == (3,)
        arr, sweep, _scalar = _coerce_sweep(x for x in [V_NN_TAIL, V_TWO, 3.0])
        assert sweep[0] is V_NN_TAIL and sweep[1] is V_TWO and sweep[2] == 3.0
        assert np.array_equal(arr, [3.0, 3.0, 3.0])
        arr, sweep, _scalar = _coerce_sweep(iter([3.0, V_NN_TAIL, V_TWO]))
        assert sweep[0] == 3.0 and sweep[1] is V_NN_TAIL and sweep[2] is V_TWO
        # an object-dtype ndarray is a sequence, not a float array
        arr, sweep, _scalar = _coerce_sweep(np.array([V_NN_TAIL, P3], dtype=object))
        assert sweep[0] is V_NN_TAIL and sweep[1] is P3
        assert np.array_equal(arr, [3.0, 3.0])
        # a float-only generator is the legacy sweep
        arr, sweep, _scalar = _coerce_sweep(x for x in [3.0, 4.0])
        assert np.array_equal(arr, [3.0, 4.0]) and sweep == [3.0, 4.0]


class TestCorpus:
    def test_demoted_power_law_sweep_is_bit_identical(self):
        c1 = compute_series_coefficients(CORPUS_0QP, np.array([3.0, 4.0]), A1, 8, order_max=5)
        c2 = compute_series_coefficients(CORPUS_0QP, [Interaction.power_law(3.0),
                                                      Interaction.power_law(4.0)], A1, 8, order_max=5)
        assert c1.keys() == c2.keys()
        for o in c1:
            assert np.array_equal(c1[o], c2[o])

    def test_generator_sweep_runs_the_full_pass(self):
        sweep = [V_NN_TAIL, 3.0, V_TWO]
        ref = compute_series_coefficients(CORPUS_0QP, sweep, A1, 8, order_max=3)
        gen = compute_series_coefficients(CORPUS_0QP, (x for x in sweep), A1, 8, order_max=3)
        assert ref.keys() == gen.keys()
        for o in ref:
            assert gen[o].shape == (3,)
            assert np.array_equal(ref[o], gen[o])
        recs = evaluate_corpus(CORPUS_0QP, iter(sweep), A1, 8, order_max=3)
        assert sorted({r["nu_index"] for r in recs}) == [0, 1, 2]

    def test_interaction_sweep_equals_the_manual_sum(self):
        sweep = [V_NN_TAIL, 3.0, V_TWO]
        coef = compute_series_coefficients(CORPUS_0QP, sweep, A1, 8, order_max=4)
        corpus = _load_corpus(CORPUS_0QP)
        for o in coef:
            assert coef[o].shape == (3,)
            blk = corpus[o]
            for j, nu in enumerate(sweep):
                cache = {}
                tot = 0.0
                for e, m, pf in zip(blk["edges_list"], blk["multiplicities_list"], blk["prefactor"]):
                    tot += float(pf) * evaluate_graph(np.repeat(e, m, axis=0), nu, A1, n_points=8,
                                                      block_cache=cache, accuracy="floor")
                assert coef[o][j].real == pytest.approx(tot, rel=1e-12, abs=1e-14)
        per = evaluate_corpus(CORPUS_0QP, sweep, A1, 8, order_max=4)
        assert set(per[0]) == {"order", "graph_id", "nu", "nu_index", "route",
                               "prefactor", "value", "contribution", "s", "t",
                               "interaction", "nu_label"}
        assert per[0]["interaction"] is V_NN_TAIL and per[0]["nu_label"] == "nn+tail"
        assert per[0]["nu"] == 3.0

    def test_float_sweep_records_keep_the_legacy_schema(self):
        recs = evaluate_corpus(CORPUS_0QP, 3.0, A1, 8, order_max=3)
        assert set(recs[0]) == {"order", "graph_id", "nu", "nu_index", "route",
                                "prefactor", "value", "contribution", "s", "t"}
        assert "interaction" not in recs[0]

    def test_evaluate_corpus_records_carry_the_interaction(self):
        recs = evaluate_corpus(CORPUS_0QP, V_NN_TAIL, A1, 8, order_max=3)
        assert recs[0]["interaction"] is V_NN_TAIL and recs[0]["nu_label"] == "nn+tail"
        assert all(r["route"].startswith("topology") for r in recs)

    def test_labelled_plain_interactions_keep_their_identity_in_the_records(self):
        """A shells table that cancels its own tail exactly (J = 1 at
        distance 1 against |x|^-3) and a labelled ``power_law`` are both
        plain power laws: routed as the float -- bit-identical to the
        float sweep -- but the records must still say WHICH object and
        label produced them, and the keys must be present even though
        every point is plain."""
        VJ = Interaction.from_shells(A1, {1.0: 1.0}, b=[1.0], nu=[3.0], total=True, label="J1")
        P4 = Interaction.power_law(4.0, label="my4")
        assert VJ.as_plain_float() == 3.0 and P4.as_plain_float() == 4.0
        recs = evaluate_corpus(CORPUS_0QP, [VJ, P4], A1, 8, order_max=3)
        ref = evaluate_corpus(CORPUS_0QP, np.array([3.0, 4.0]), A1, 8, order_max=3)
        assert len(recs) == len(ref) > 0
        for r, f in zip(recs, ref):
            assert (r["order"], r["graph_id"], r["nu_index"]) == (f["order"], f["graph_id"], f["nu_index"])
            assert r["value"] == f["value"] and r["route"] == f["route"] and r["nu"] == f["nu"]
            obj = VJ if r["nu_index"] == 0 else P4
            assert r["interaction"] is obj and r["nu_label"] == obj.label
        per = evaluate_corpus(CORPUS_0QP, [VJ, P4], A1, 8, order_max=3)
        assert {"interaction", "nu_label"} <= set(per[0])
        assert {p["nu_label"] for p in per} == {"J1", "my4"}

    def test_purely_compact_sweep_runs_in_nn_mode(self):
        NN = Interaction.nearest_neighbour(A1)
        c_inf = compute_series_coefficients(CORPUS_0QP, np.inf, A1, 0, order_max=4)
        c_nn = compute_series_coefficients(CORPUS_0QP, NN, A1, 0, order_max=4)
        for o in c_inf:
            assert np.array_equal(c_inf[o], c_nn[o])


class TestDocstrings:
    def test_front_ends_state_the_sweep_form_and_the_first_point_diagnostics(self):
        for fn in (compute_series_coefficients, evaluate_corpus):
            doc = fn.__doc__
            assert "Interaction" in doc and "nu_label" in doc
            assert "first sweep point" in doc


class TestSweepEdgeCases:
    def test_object_ndarray_of_any_shape_and_an_empty_sweep(self):
        from gzl.series import _coerce_sweep
        V1 = Interaction.from_table({(1,): 0.4, (-1,): 0.4}, b=[1.0], nu=[3.0])
        col = np.array([V1, 3.0, V1], dtype=object).reshape(3, 1)
        nu_arr, sweep, _scalar = _coerce_sweep(col)
        assert nu_arr.shape == (3,) and sweep[0] is V1 and sweep[1] == 3.0
        for empty in ([], np.array([]), iter([]), (x for x in [])):
            with pytest.raises(ValueError, match="empty"):
                _coerce_sweep(empty)
        gen = (x for x in [V1, 3.0])
        _coerce_sweep(gen)                       # consumed here ...
        with pytest.raises(ValueError, match="empty"):
            _coerce_sweep(gen)                   # ... so a second pass is refused, not (0,)


class TestCorpus1qp:
    def test_1qp_grid_sweep_shape_and_bridge_row(self):
        coef = compute_series_coefficients(CORPUS_1QP, V_NN_TAIL, A1, 8, order_max=2)
        assert sorted(coef) == [1, 2]
        for o in coef:
            assert coef[o].shape == (8,)
        # the grid's k = 0 entry is the momentum=np.zeros(d) pass, and a
        # batch of grid points equals the grid values
        at0 = compute_series_coefficients(CORPUS_1QP, V_NN_TAIL, A1, 8, order_max=2,
                                          momentum=np.zeros(1))
        batch = compute_series_coefficients(CORPUS_1QP, V_NN_TAIL, A1, 8, order_max=2,
                                            momentum=np.array([[0.0], [0.25], [0.5]]))
        for o in coef:
            # V_NN_TAIL is ONE Interaction, so there is no sweep axis:
            # an explicit k is a bare scalar and a batch is (N,).
            assert np.shape(at0[o]) == () and batch[o].shape == (3,)
            assert coef[o][0] == pytest.approx(float(at0[o]), rel=1e-12, abs=0.0)
            np.testing.assert_allclose(coef[o][[0, 2, 4]], batch[o], rtol=1e-12, atol=0.0)
        # order 1 is the single hopping bridge: its row is prefactor times
        # the exact lattice sum Σ_x V(x) e^{-2πi k x} at every grid k
        blk = _load_corpus(CORPUS_1QP)[1]
        assert len(blk["edges_list"]) == 1 and blk["hopping"][0].tolist() == [0, 1]
        k = (np.arange(8) / 8.0)[:, None]
        row = float(blk["prefactor"][0]) * V_NN_TAIL.lattice_sum(A1, k)
        assert row.shape == (8,) and np.array_equal(coef[1].real, row)
        assert np.all(coef[1].imag == 0.0)


class TestCli:
    def _toml(self, tmp_path, body):
        cfg = tmp_path / "cfg.toml"
        out = tmp_path / "out.csv"
        cfg.write_text(f'[corpus]\npath = "{CORPUS_0QP}"\n[lattice]\nA = [[1.0]]\n'
                       f'[evaluation]\nn_points = 8\norder_max = 3\n{body}\n'
                       f'[output]\ncsv = "{out}"\n')
        return cfg, out

    def test_interaction_sweep_csv_schema(self, tmp_path):
        cfg, out = self._toml(tmp_path, '''
[[interactions]]
label = "nn+tail"
b = [1.0]
nu = [3.0]
compact = [[1, 0.4], [-1, 0.4], [0, 0.1]]
[[interactions]]
label = "two-term"
b = [1.0, -0.3]
nu = [3.0, 5.0]
''')
        assert main(["--config", str(cfg)]) == 0
        rows = list(csv.reader(open(out)))
        assert rows[0] == ["order", "nu", "interaction", "coefficient"]
        coef = compute_series_coefficients(CORPUS_0QP, [V_NN_TAIL, V_TWO], A1, 8, order_max=3)
        got = {(int(r[0]), r[2]): r[3] for r in rows[1:]}
        for o in coef:
            assert got[(o, "nn+tail")] == f"{coef[o][0].real:.12e}"
            assert got[(o, "two-term")] == f"{coef[o][1].real:.12e}"
            assert float(rows[1][1]) == 3.0

    def test_single_interaction_table_and_shells(self, tmp_path):
        cfg, out = self._toml(tmp_path, '''
[interaction]
label = "shell"
b = [0.5]
nu = [3.0]
shells = [[1.0, 0.7]]
''')
        assert main(["--config", str(cfg)]) == 0
        rows = list(csv.reader(open(out)))
        V = Interaction.from_shells(A1, {1.0: 0.7}, b=[0.5], nu=[3.0], label="shell")
        coef = compute_series_coefficients(CORPUS_0QP, V, A1, 8, order_max=3)
        assert rows[1][2] == "shell" and rows[1][3] == f"{np.real(coef[2]):.12e}"

    def test_labelled_plain_interactions_share_one_schema_across_both_csvs(self, tmp_path):
        """Two labelled plain power laws: the coefficient CSV and the
        ``--per-graph`` CSV must agree that the user passed Interactions
        (both carry the ``interaction`` column with the user's labels),
        and the coefficients are the legacy float sweep's, rendered."""
        cfg, out = self._toml(tmp_path, '''
[[interactions]]
label = "L3"
b = [1.0]
nu = [3.0]
[[interactions]]
label = "L4"
b = [1.0]
nu = [4.0]
''')
        assert main(["--config", str(cfg)]) == 0
        rows = list(csv.reader(open(out)))
        assert rows[0] == ["order", "nu", "interaction", "coefficient"]
        assert [r[2] for r in rows[1:3]] == ["L3", "L4"]
        coef = compute_series_coefficients(CORPUS_0QP, np.array([3.0, 4.0]), A1, 8, order_max=3)
        for r in rows[1:]:
            j = {"L3": 0, "L4": 1}[r[2]]
            assert r[3] == f"{coef[int(r[0])][j].real:.12e}"
        assert main(["--config", str(cfg), "--per-graph"]) == 0
        rows_pg = list(csv.reader(open(out)))
        assert rows_pg[0] == ["order", "graph_id", "nu", "interaction", "value", "prefactor",
                              "contribution", "s", "t"]
        assert {r[3] for r in rows_pg[1:]} == {"L3", "L4"}
        assert {r[2] for r in rows_pg[1:]} == {"3.0000000000", "4.0000000000"}

    def test_interaction_and_nu_are_mutually_exclusive(self, tmp_path):
        cfg, _ = self._toml(tmp_path, '''
nu_start = 3.0
nu_end = 3.0
nu_step = 1.0
[interaction]
b = [1.0]
nu = [3.0]
''')
        assert main(["--config", str(cfg)]) == 2

    def test_invalid_table_is_a_clean_error(self, tmp_path):
        cfg, _ = self._toml(tmp_path, '''
[interaction]
b = [1.0]
nu = [3.0]
compact = [[1, 0.4]]
''')
        assert main(["--config", str(cfg)]) == 2

    def test_npz_without_labels_is_a_clean_error(self, tmp_path, capsys):
        bad = tmp_path / "bad.npz"
        np.savez(bad, foo=np.zeros(3))
        cfg, _ = self._toml(tmp_path, f'[interaction]\nb = [1.0]\nnu = [3.0]\ncompact_npz = "{bad}"\n')
        assert main(["--config", str(cfg)]) == 2
        assert "invalid interaction table" in capsys.readouterr().err

    def test_interactions_as_a_table_is_refused_with_the_array_message(self, tmp_path, capsys):
        cfg, _ = self._toml(tmp_path, '[interactions]\nlabel = "x"\nb = [1.0]\nnu = [3.0]\n')
        assert main(["--config", str(cfg)]) == 2
        assert "[[interactions]] must be an array of tables" in capsys.readouterr().err

    @pytest.mark.parametrize("form", ["compact", "compact_npz"])
    def test_total_is_refused_outside_the_shells_form(self, tmp_path, capsys, form):
        if form == "compact":
            line = "compact = [[1, 0.4], [-1, 0.4]]"
        else:
            p = tmp_path / "t.npz"
            np.savez(p, labels=np.array([[1], [-1]]), values=np.array([0.4, 0.4]))
            line = f'compact_npz = "{p}"'
        cfg, _ = self._toml(tmp_path, f'[interaction]\nb = [1.0]\nnu = [3.0]\n{line}\ntotal = true\n')
        assert main(["--config", str(cfg)]) == 2
        assert "total" in capsys.readouterr().err

    def test_empty_interactions_list_is_refused_explicitly(self, tmp_path, capsys):
        cfg = tmp_path / "cfg.toml"
        cfg.write_text(f'interactions = []\n[corpus]\npath = "{CORPUS_0QP}"\n[lattice]\nA = [[1.0]]\n'
                       f'[evaluation]\nn_points = 8\norder_max = 3\n'
                       f'[output]\ncsv = "{tmp_path / "out.csv"}"\n')
        assert main(["--config", str(cfg)]) == 2
        err = capsys.readouterr().err
        assert "[[interactions]]" in err and "empty" in err

    def test_duplicate_labels_are_refused(self, tmp_path, capsys):
        cfg, _ = self._toml(tmp_path, '''
[[interactions]]
label = "same"
b = [1.0]
nu = [3.0]
compact = [[1, 0.4], [-1, 0.4]]
[[interactions]]
label = "same"
b = [1.0]
nu = [3.0]
compact = [[1, -0.4], [-1, -0.4]]
''')
        assert main(["--config", str(cfg)]) == 2
        assert "duplicate" in capsys.readouterr().err

    def test_per_graph_with_interaction(self, tmp_path):
        cfg, out = self._toml(tmp_path, '''
[interaction]
label = "nn+tail"
b = [1.0]
nu = [3.0]
compact = [[1, 0.4], [-1, 0.4], [0, 0.1]]
''')
        assert main(["--config", str(cfg), "--per-graph"]) == 0
        rows = list(csv.reader(open(out)))
        assert rows[0][:4] == ["order", "graph_id", "nu", "interaction"]
        assert rows[1][3] == "nn+tail" and rows[1][2] == "3.0000000000"
