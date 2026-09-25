# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""First tests of the ``gzl.series`` command-line driver.

``python -m gzl.series --config <toml>`` is the documented entry
point for a series pass, and it had no test coverage at all: nothing
pinned its exit codes, its CSV schema, or — the failure mode that has
actually bitten this repo twice (``core_grading``, ``richardson``) —
whether the wrapper computes the same numbers as the library it wraps.

The oracle here is exactly that: the CSV must be the byte-for-byte
rendering, through the writer's own format strings, of a direct
:func:`compute_series_coefficients` call made with the arguments
``main`` resolves from the config.  Every default the CLI leaves
untouched (``dense_engine``, ``sp_n_points``, ``core_grading``) already
matches the
library default, so the direct call passes only ``order_max``; if a
future edit desynchronises one of them, the coefficient columns move
and this test says so.

Kept cheap on purpose — the 0qp corpus at ``order_max = 3``,
``n_points = 8``, d = 1 and two ν points is ~15 ms of pass work, so the
file runs in the default suite.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from gzl import data_path
from gzl.series import compute_series_coefficients, main


CORPUS_NPZ = data_path("tfim_softcore_corpus_0qp.npz")

# The legacy (no k-axis) writer's format strings, quoted from
# ``gzl.series.main``.  Reused rather than re-derived so the test
# fails if the rendering changes, not merely if the values do.
_NU_FMT = "{:.10f}"
_COEF_FMT = "{:.12e}"

ORDER_MAX = 3
N_POINTS = 8
NUS = np.array([3.0, 4.0])          # == nu_grid(3.0, 4.0, 1.0)
A_1D = np.eye(1)


def _write_config(tmp_path: Path, *, nu_section: str,
                  order_max: int = ORDER_MAX) -> tuple[Path, Path, Path]:
    """Write a TOML config; return (config, csv, diagnostics_csv)."""
    out_csv = tmp_path / "series.csv"
    diag_csv = tmp_path / "diagnostics.csv"
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[corpus]\n"
        f'path = "{CORPUS_NPZ.as_posix()}"\n'
        "\n"
        "[lattice]\n"
        "A = [[1.0]]\n"
        "\n"
        "[evaluation]\n"
        f"n_points = {N_POINTS}\n"
        f"order_max = {order_max}\n"
        f"{nu_section}"
        "\n"
        "[output]\n"
        f'csv = "{out_csv.as_posix()}"\n'
        f'diagnostics_csv = "{diag_csv.as_posix()}"\n'
    )
    return cfg, out_csv, diag_csv


def _read_csv(path: Path) -> list[list[str]]:
    with open(path, newline="") as f:
        return list(csv.reader(f))


class TestCliLegacy:
    """The legacy ``order, nu, coefficient`` schema — the shape a
    vacuum (0qp) corpus with no ``momentum`` produces."""

    NU_SECTION = (
        "nu_start = 3.0\n"
        "nu_end = 4.0\n"
        "nu_step = 1.0\n"
    )

    @pytest.mark.parametrize("order_max", [ORDER_MAX, 8])
    def test_every_row_matches_the_library_call(self, tmp_path, order_max):
        r"""The whole point of the file: the CSV is the library's own
        answer, rendered by the writer's format strings.

        ``main`` resolves ``nu_start/nu_end/nu_step`` through
        ``nu_grid(3.0, 4.0, 1.0)``, which is exactly ``[3.0, 4.0]``,
        and passes every other knob at its library default — so the
        direct call below is argument-for-argument what the CLI runs.
        Row order is ``for order in sorted(...)`` then ``for j, nu``.
        The expected rows spell out the header, the ``order_max``
        truncation and the ten-decimal ν column, which is the join key
        downstream (the σ-sweep CSVs are merged on it).

        Run twice: at ``order_max = 3`` the corpus contributes one
        graph per order, and at ``order_max = 8`` it contributes 77
        across seven orders, which is what puts more than one router
        route under the equality (0.1 s of pass work — still a default
        suite test)."""
        cfg, out_csv, _ = _write_config(
            tmp_path, nu_section=self.NU_SECTION, order_max=order_max)
        assert main(["--config", str(cfg)]) == 0

        coefficients = compute_series_coefficients(
            CORPUS_NPZ, NUS, A_1D, N_POINTS, order_max=order_max,
        )
        expected = [["order", "nu", "coefficient"]]
        for order in sorted(coefficients):
            arr = coefficients[order]
            for j, nu_val in enumerate(NUS):
                expected.append([
                    str(order),
                    _NU_FMT.format(nu_val),
                    _COEF_FMT.format(complex(arr[j]).real),
                ])

        rows = _read_csv(out_csv)
        assert len(rows) == 1 + len(coefficients) * NUS.size > 1
        assert rows == expected

    def test_diagnostics_csv_layout(self, tmp_path):
        """``order, n_graphs, wall_s`` block, a blank separator, then a
        ``route, n_graphs`` block.  Wall times are not asserted."""
        cfg, out_csv, diag_csv = _write_config(
            tmp_path, nu_section=self.NU_SECTION)
        assert main(["--config", str(cfg)]) == 0
        rows = _read_csv(diag_csv)
        assert rows[0] == ["order", "n_graphs", "wall_s"]
        blank = rows.index([])
        assert rows[blank + 1] == ["route", "n_graphs"]
        orders = [int(r[0]) for r in rows[1:blank]]
        assert orders == sorted(orders)
        assert max(orders) <= ORDER_MAX


class TestCliSingleNu:
    """``[evaluation] nu`` — the single-ν form, including the
    documented ``"inf"`` spelling of the nearest-neighbour kernel."""

    def test_finite_single_nu(self, tmp_path):
        cfg, out_csv, _ = _write_config(tmp_path, nu_section='nu = 3.5\n')
        assert main(["--config", str(cfg)]) == 0
        rows = _read_csv(out_csv)
        assert rows[0] == ["order", "nu", "coefficient"]
        assert {r[1] for r in rows[1:]} == {"3.5000000000"}

    def test_nu_inf_matches_the_library_call(self, tmp_path):
        r"""``nu = "inf"`` reaches ``_parse_nu_cli`` as a string and
        becomes ``float('inf')``.  The legacy writer renders it with
        ``f"{nu:.10f}"``, and CPython's float formatting emits ``inf``
        for a non-finite value under any presentation type — so the
        column is the literal ``inf``, not an exception and not a
        padded number.  (The per-graph writer has an explicit
        ``_fmt_nu`` for this; the legacy writer gets it for free, and
        that is what this test pins.)"""
        cfg, out_csv, _ = _write_config(tmp_path, nu_section='nu = "inf"\n')
        assert main(["--config", str(cfg)]) == 0
        nus = np.array([np.inf])
        coefficients = compute_series_coefficients(
            CORPUS_NPZ, nus, A_1D, N_POINTS, order_max=ORDER_MAX,
        )
        expected = [["order", "nu", "coefficient"]]
        for order in sorted(coefficients):
            expected.append([
                str(order),
                _NU_FMT.format(np.inf),
                _COEF_FMT.format(complex(coefficients[order][0]).real),
            ])
        assert len(expected) > 1, "no coefficient rows to compare"
        assert expected[1][1] == "inf"
        assert _read_csv(out_csv) == expected


class TestCliRefusals:
    """Exit codes.  ``main`` returns 2 (rather than raising) for a
    missing required input, so a shell driver can branch on it.

    The banner is on stderr, so that ``--output -`` can put the CSV on
    stdout; only the data stream is stdout."""

    @pytest.mark.parametrize("corpus_section", ["[corpus]\n", ""])
    def test_missing_corpus_defaults_to_tfim1qp(self, tmp_path, capsys,
                                                corpus_section):
        """No ``--corpus`` and no ``[corpus] path`` (with or without the
        section): the run uses ``tfim1qp``, says so, and writes exactly the
        CSV an explicit ``path = "tfim1qp"`` writes."""
        body = ("\n[lattice]\nA = [[1.0]]\n"
                "\n[evaluation]\nn_points = 8\nnu = 3.0\norder_max = 3\n")
        default_cfg = tmp_path / "default.toml"
        default_cfg.write_text(
            corpus_section + body
            + f'\n[output]\ncsv = "{(tmp_path / "default.csv").as_posix()}"\n')
        named_cfg = tmp_path / "named.toml"
        named_cfg.write_text(
            '[corpus]\npath = "tfim1qp"\n' + body
            + f'\n[output]\ncsv = "{(tmp_path / "named.csv").as_posix()}"\n')

        assert main(["--config", str(default_cfg)]) == 0
        assert "(tfim1qp, default)" in capsys.readouterr().err
        assert main(["--config", str(named_cfg)]) == 0
        out = capsys.readouterr().err
        assert "(tfim1qp)" in out and "(tfim1qp, default)" not in out
        assert ((tmp_path / "default.csv").read_text()
                == (tmp_path / "named.csv").read_text())

    def test_flags_only_run_defaults_to_tfim1qp(self, tmp_path, capsys):
        args = ["--A", "[[1.0]]", "--n-points", "8", "--order-max", "3",
                "--nu", "3.0"]
        assert main(args + ["--output", str(tmp_path / "d.csv")]) == 0
        assert "(tfim1qp, default)" in capsys.readouterr().err
        assert main(args + ["--corpus", "tfim1qp",
                            "--output", str(tmp_path / "n.csv")]) == 0
        assert (tmp_path / "d.csv").read_text() == (tmp_path / "n.csv").read_text()

    def test_missing_output_returns_two(self, tmp_path, capsys):
        cfg = tmp_path / "c.toml"
        cfg.write_text(
            f'[corpus]\npath = "{CORPUS_NPZ.as_posix()}"\n'
            "\n[lattice]\nA = [[1.0]]\n"
            "\n[evaluation]\nn_points = 8\nnu = 3.0\n"
        )
        assert main(["--config", str(cfg)]) == 2
        assert "output" in capsys.readouterr().err

    def test_missing_nu_spec_returns_two(self, tmp_path, capsys):
        cfg = tmp_path / "c.toml"
        cfg.write_text(
            f'[corpus]\npath = "{CORPUS_NPZ.as_posix()}"\n'
            "\n[lattice]\nA = [[1.0]]\n"
            "\n[evaluation]\nn_points = 8\n"
            f'\n[output]\ncsv = "{(tmp_path / "o.csv").as_posix()}"\n'
        )
        assert main(["--config", str(cfg)]) == 2
        assert "nu" in capsys.readouterr().err

    def test_missing_required_section_returns_two(self, tmp_path, capsys):
        """A malformed config is the command used wrongly, like every
        other refusal here -- it used to raise instead."""
        cfg = tmp_path / "c.toml"
        cfg.write_text('[corpus]\npath = "x.npz"\n')
        assert main(["--config", str(cfg)]) == 2
        assert "missing required sections" in capsys.readouterr().err
