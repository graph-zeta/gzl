# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""The ``gzl.series`` CLI paths that :mod:`tests.test_series_cli` leaves open.

That file pins the legacy ``order, nu, coefficient`` emit and the exit
codes.  What it does not reach is everything selected by ``--momentum``
(the extended ``k_index, k_frac_*`` schema and the C-order flattening
that indexes it), the ``--nu-start/--nu-end/--nu-step`` flags as
opposed to their config spelling, flag-over-config precedence, the
``--per-graph`` dump on a plain power-law pass, and what a malformed
``--A`` does.

These are written as *characterisation* tests: they record what the CLI
does, so that a refactor of it provably changes nothing.
``TestLatticeRefusals`` pins the refusal contract for a malformed or
missing ``--A``.

The oracle is the one :mod:`tests.test_series_cli` established: a CSV
cell must be the rendering, through the writer's own format strings, of
a direct :func:`compute_series_coefficients` call.

Kept cheap on purpose -- the 0qp corpus at ``order_max = 3`` and the 1qp
corpus at ``order_max = 2``, both at ``n_points = 8`` in d = 1, so the
file runs in the default suite.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from gzl.series import compute_series_coefficients, evaluate_corpus, main


# The writer's own format strings, quoted from ``gzl.series.main`` and
# ``_run_per_graph``.  Reused rather than re-derived, so a change in
# rendering fails here and not only a change in value.
_NU_FMT = "{:.10f}"
_KFRAC_FMT = "{:.10f}"
_COEF_FMT = "{:.12e}"

N_POINTS = 8
NU = 3.0

#: Vacuum corpus: two orders (2, 3) at ``order_max = 3``, k-independent.
BASE_0QP = ["--corpus", "tfim0qp", "--A", "chain", "--n-points", str(N_POINTS),
            "--order-max", "3", "--nu", str(NU)]
#: One-quasiparticle corpus: orders (1, 2), and genuinely k-dependent,
#: which is what lets the k-indexing convention be tested at all.
BASE_1QP = ["--corpus", "tfim1qp", "--A", "chain", "--n-points", str(N_POINTS),
            "--order-max", "2", "--nu", str(NU)]


def _run(tmp_path: Path, args: list[str], name: str = "out.csv") -> Path:
    out = tmp_path / name
    assert main(args + ["--output", str(out)]) == 0
    return out


def _rows(path: Path) -> list[list[str]]:
    with open(path, newline="") as f:
        return list(csv.reader(f))


class TestMomentumForms:
    """``--momentum`` selects the CSV schema, and the k columns must
    index the coefficients in the order the writer flattens them."""

    def test_scalar_momentum_keeps_the_legacy_schema(self, tmp_path):
        rows = _rows(_run(tmp_path, BASE_0QP + ["--momentum", "0.5"]))
        assert rows[0] == ["order", "nu", "coefficient"]
        assert len(rows) == 1 + 2          # orders 2, 3 at one nu

    def test_vector_momentum_keeps_the_legacy_schema(self, tmp_path):
        """A ``(d,)`` vector is one k, not a batch of d."""
        rows = _rows(_run(tmp_path, BASE_0QP + ["--momentum", "[0.25]"]))
        assert rows[0] == ["order", "nu", "coefficient"]
        assert len(rows) == 1 + 2

    def test_batch_momentum_emits_the_extended_schema(self, tmp_path):
        rows = _rows(_run(tmp_path, BASE_1QP
                          + ["--momentum", "[[0.0],[0.25],[0.5]]"]))
        assert rows[0] == ["order", "nu", "k_index", "k_frac_0", "coefficient"]
        assert len(rows) == 1 + 2 * 3      # orders 1, 2 x three k

    def test_full_grid_emits_one_row_per_grid_point(self, tmp_path):
        """A 1qp corpus with no ``--momentum`` is the full BZ grid."""
        rows = _rows(_run(tmp_path, BASE_1QP))
        assert rows[0] == ["order", "nu", "k_index", "k_frac_0", "coefficient"]
        assert len(rows) == 1 + 2 * N_POINTS

    @pytest.mark.parametrize("spelling", ["none", "default"])
    def test_none_and_default_are_the_omitted_flag(self, tmp_path, spelling):
        omitted = _run(tmp_path, BASE_0QP, "omitted.csv")
        spelled = _run(tmp_path, BASE_0QP + ["--momentum", spelling],
                       f"{spelling}.csv")
        assert spelled.read_text() == omitted.read_text()

    def test_k_frac_column_is_the_requested_momentum(self, tmp_path):
        rows = _rows(_run(tmp_path, BASE_1QP
                          + ["--momentum", "[[0.0],[0.25],[0.5]]"]))[1:]
        for row in rows:
            k_index, k_frac = int(row[2]), row[3]
            assert k_frac == _KFRAC_FMT.format([0.0, 0.25, 0.5][k_index])

    def test_grid_batch_and_single_agree_at_a_shared_k(self, tmp_path):
        """The three momentum spellings are three views of one pass.

        ``k = 1/4`` is grid point 2 of 8, batch entry 1 of 3, and the
        whole of a single-k run.  All three must carry the same digits,
        which is what pins the C-order flattening in
        ``_build_k_frac_table`` against the accumulator's trailing axes.
        """
        grid = _rows(_run(tmp_path, BASE_1QP, "grid.csv"))[1:]
        batch = _rows(_run(tmp_path, BASE_1QP
                           + ["--momentum", "[[0.0],[0.25],[0.5]]"],
                           "batch.csv"))[1:]
        single = _rows(_run(tmp_path, BASE_1QP + ["--momentum", "0.25"],
                            "single.csv"))[1:]

        by_order_grid = {r[0]: r[4] for r in grid if int(r[2]) == 2}
        by_order_batch = {r[0]: r[4] for r in batch if int(r[2]) == 1}
        by_order_single = {r[0]: r[2] for r in single}

        assert by_order_grid == by_order_batch == by_order_single
        assert len(by_order_grid) == 2

    def test_extended_rows_match_the_library_call(self, tmp_path):
        """Every cell of the extended schema, against the library."""
        momentum = np.array([[0.0], [0.25], [0.5]])
        rows = _rows(_run(tmp_path, BASE_1QP
                          + ["--momentum", "[[0.0],[0.25],[0.5]]"]))[1:]
        ref = compute_series_coefficients(
            "tfim1qp", np.array([NU]), "chain", N_POINTS,
            momentum=momentum, order_max=2,
        )
        expected = []
        for order in sorted(ref):
            flat = np.asarray(ref[order][0]).reshape(-1)
            for idx in range(momentum.shape[0]):
                expected.append([
                    str(order), _NU_FMT.format(NU), str(idx),
                    _KFRAC_FMT.format(momentum[idx, 0]),
                    _COEF_FMT.format(float(flat[idx])),
                ])
        assert rows == expected


class TestNuFlagsAndPrecedence:
    """``--nu-start/--nu-end/--nu-step`` as flags, and flag over config."""

    def _config(self, tmp_path: Path, body: str) -> Path:
        cfg = tmp_path / "c.toml"
        cfg.write_text('[lattice]\nA = "chain"\n\n[evaluation]\n'
                       f"n_points = {N_POINTS}\norder_max = 3\n{body}")
        return cfg

    def test_nu_grid_flags_match_the_config_spelling(self, tmp_path):
        cfg = self._config(tmp_path,
                           "nu_start = 3.0\nnu_end = 4.0\nnu_step = 0.5\n")
        from_cfg = _run(tmp_path, ["--corpus", "tfim0qp",
                                   "--config", str(cfg)], "cfg.csv")
        from_flags = _run(tmp_path, ["--corpus", "tfim0qp", "--A", "chain",
                                     "--n-points", str(N_POINTS),
                                     "--order-max", "3",
                                     "--nu-start", "3.0", "--nu-end", "4.0",
                                     "--nu-step", "0.5"], "flags.csv")
        assert from_flags.read_text() == from_cfg.read_text()
        assert len(_rows(from_flags)) == 1 + 2 * 3   # two orders x three nu

    def test_flag_overrides_the_config_value(self, tmp_path):
        """``_resolve`` is flag > config > default."""
        cfg = self._config(tmp_path, "nu = 3.0\n")
        rows = _rows(_run(tmp_path, ["--corpus", "tfim0qp",
                                     "--config", str(cfg), "--nu", "4.0"]))
        assert {r[1] for r in rows[1:]} == {_NU_FMT.format(4.0)}

    def test_single_nu_flag_overrides_the_config_grid(self, tmp_path):
        cfg = self._config(tmp_path,
                           "nu_start = 3.0\nnu_end = 4.0\nnu_step = 0.5\n")
        rows = _rows(_run(tmp_path, ["--corpus", "tfim0qp",
                                     "--config", str(cfg), "--nu", "3.5"]))
        assert {r[1] for r in rows[1:]} == {_NU_FMT.format(3.5)}
        assert len(rows) == 1 + 2


class TestPerGraph:
    """``--per-graph`` dumps ``evaluate_corpus`` records instead of the
    summed coefficients."""

    def test_header_and_rows_match_evaluate_corpus(self, tmp_path):
        rows = _rows(_run(tmp_path, BASE_0QP + ["--per-graph"]))
        assert rows[0] == ["order", "graph_id", "nu", "value", "prefactor",
                           "contribution", "s", "t"]
        records = evaluate_corpus("tfim0qp", np.array([NU]), "chain",
                                  N_POINTS, order_max=3)
        assert len(rows) == 1 + len(records)
        expected = [
            [str(r["order"]), r["graph_id"], _NU_FMT.format(r["nu"]),
             _COEF_FMT.format(complex(r["value"]).real),
             _COEF_FMT.format(r["prefactor"]),
             _COEF_FMT.format(complex(r["contribution"]).real),
             str(r["s"]), str(r["t"])]
            for r in records
        ]
        assert rows[1:] == expected

    def test_contributions_sum_to_the_coefficients(self, tmp_path):
        """The two CSVs the CLI can write are one pass seen two ways."""
        summed = _rows(_run(tmp_path, BASE_0QP, "summed.csv"))[1:]
        per_graph = _rows(_run(tmp_path, BASE_0QP + ["--per-graph"],
                               "pg.csv"))[1:]
        totals: dict[str, float] = {}
        for row in per_graph:
            totals[row[0]] = totals.get(row[0], 0.0) + float(row[5])
        for row in summed:
            assert totals[row[0]] == pytest.approx(float(row[2]), rel=1e-12)

    def test_refuses_an_array_valued_pass(self, tmp_path, capsys):
        """A 1qp full-BZ grid has no scalar per graph; exit 2, not a
        silent dump of the first grid cell."""
        assert main(BASE_1QP + ["--per-graph",
                                "--output", str(tmp_path / "o.csv")]) == 2
        assert "--per-graph" in capsys.readouterr().err
        assert not (tmp_path / "o.csv").exists()


class TestLatticeRefusals:
    """A bad ``--A`` returns 2, like every other refusal.

    It used to raise, which contradicted the contract the CLI states
    for itself -- see ``tests/test_series_cli.py::TestCliRefusals``,
    "``main`` returns 2 (rather than raising) for a missing required
    input, so a shell driver can branch on it".  A malformed TOML
    config did the same and was unified in the same change.
    """

    @pytest.mark.parametrize("bad_A", ["[[1.0", "nonsense", "[[1, 2]]"])
    def test_malformed_A_returns_two(self, tmp_path, capsys, bad_A):
        assert main(["--corpus", "tfim0qp", "--A", bad_A,
                     "--n-points", str(N_POINTS), "--order-max", "1",
                     "--nu", str(NU),
                     "--output", str(tmp_path / "o.csv")]) == 2
        assert "--A must" in capsys.readouterr().err

    def test_missing_A_returns_two(self, tmp_path, capsys):
        assert main(["--corpus", "tfim0qp", "--n-points", str(N_POINTS),
                     "--order-max", "1", "--nu", str(NU),
                     "--output", str(tmp_path / "o.csv")]) == 2
        assert "--A" in capsys.readouterr().err
