# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Tests for ``benchmarks/mc_comparison.py``.

The benchmark reproduces the published Monte Carlo comparison
(arXiv:2609.18918, Sec. 8.3); a full run takes minutes on several cores.
These tests pin its configuration and table rules against the published
table through the benchmark's own code (``plan``, ``build_rows``,
``summarize``, ``compare_published``), unit-test the pieces a wrong number
could hide in (the wavevector read-off, the error aggregation, the summary
statistics, the grid shift), and run the ``--quick`` preset on all 15
series.  ``benchmarks`` is not a package, so the script is loaded from its
path.
"""

from __future__ import annotations

import collections
import csv
import re
import importlib.util
import io
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "mc_comparison.py"

# Tabulated exponents per series, in table order (the published sigma ranges).
PUBLISHED_COUNTS = [4, 13, 4, 9, 8, 10, 6, 11, 5, 7, 8, 8, 5, 8, 8]


@pytest.fixture(scope="module")
def bench():
    name = "_mc_comparison_benchmark"
    spec = importlib.util.spec_from_file_location(name, BENCH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses resolve annotations through it
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(name, None)


def _row(s, sigma, order, pull, *, tn=1.0, mc_err=1.0, grid_shift=1e-9,
         flip=False):
    return dict(config=s.config, source=s.source, lattice=s.lattice,
                sigma=sigma, alpha=sigma + 2, order=order, tn=tn, mc=1.0,
                mc_err=mc_err, pull=pull, rel=0.0, grid_shift=grid_shift,
                n_points=48, sign_flip=flip)


def _series(bench, config):
    return next(s for s in bench.SERIES if s.config == config)


def test_configurations_match_the_published_table(bench):
    """15 series, 123 exponents, the caption's grids, one published row each."""
    assert len(bench.SERIES) == 15
    # The dimensions are derived from the library's lattice table, not
    # spelled out here any more: pin what that derivation must give.
    assert bench.LATTICE_DIM == {"chain": 1, "square": 2,
                                 "triangular": 2, "cubic": 3}
    assert set(bench.LATTICE_TITLE) == set(bench.LATTICE_DIM)
    counts = []
    for s in bench.SERIES:
        d = bench.LATTICE_DIM[s.lattice]
        assert len(s.momentum or (0.0,) * d) == d
        counts.append(sum(1 for label, _ in s.points
                          if float(label) - s.sig_shift >= bench.SIG_MIN))
    assert counts == PUBLISHED_COUNTS
    assert sum(len(s.points) for s in bench.SERIES) == 123
    assert bench.PUBLISHED_GRIDS == {1: (2048, 1024), 2: (48, 32), 3: (10, 8)}
    assert bench.CELL_GRIDS == {1: 512, 2: 48, 3: 10}
    assert set(bench.PUBLISHED_TABLE) == {(s.config, s.source)
                                          for s in bench.SERIES}
    for s in bench.SERIES:
        text = bench.PUBLISHED_TABLE[(s.config, s.source)]
        assert len(text.split("|")[0].split()) == len(bench.ORDERS)


def _dummy_rows(bench, points, pull=0.5):
    """One cell-level row per reference value, as evaluate_point writes it."""
    rows = []
    for p in points:
        s = bench.SERIES[p.index]
        for o, (v, e) in sorted(p.mc.items()):
            rows.append(dict(config=s.config, source=s.source,
                             lattice=s.lattice, sigma=p.sigma, alpha=p.nu,
                             order=o, tn=v, mc=v, mc_err=e, pull=pull, rel=0.0,
                             grid_shift=None, n_points=p.n_hi, sign_flip=False))
    return rows


def test_table_rules_on_the_shipped_references(bench):
    """Through plan, build_rows and summarize on the shipped reference files
    (nothing evaluated): the published 998 compared and 753 kept
    coefficients, the 17 dropped ones, and in every row exactly the orders
    the published table prints."""
    idx = list(range(len(bench.SERIES)))
    points, _ = bench.plan(idx, bench.PUBLISHED_GRIDS)
    assert len(points) == 123
    assert {(p.n_hi, p.n_lo) for p in points} == set(
        bench.PUBLISHED_GRIDS.values())
    rows = _dummy_rows(bench, points)
    table = bench.build_rows(idx, rows)
    sm = bench.summarize(table, rows)
    assert sm["compared"] == 998
    assert sm["kept"] == 753
    dropped = sum(1 for t in table for r in t.compared_rows
                  if r["order"] == t.series.drop_order)
    assert dropped == 17                 # "the 17 coefficients" of the paper
    assert [len(t.sigmas) for t in table] == PUBLISHED_COUNTS
    for t in table:
        printed, _ = bench.published_row(t.series)
        assert sorted(t.cells) == sorted(printed), t.series.name
        assert {o for o in t.excluded if o > bench.N_ANALYTIC} == (
            {t.series.drop_order} if t.series.drop_order else set())
        if t.series.drop_order is not None:
            assert t.series.drop_order == max(t.cells)
    # Every tabulated position printed as published, at a dummy pull.
    check = bench.compare_published(table)
    assert check["cells"] == 15 * len(bench.ORDERS) == 195
    assert check["maxima"] == 15


def test_every_reference_file_ships(bench):
    for s in bench.SERIES:
        for _, fname in s.points:
            assert bench.data_path(f"MC_patched/TFIM/{s.lattice}/{fname}").is_file()


def test_grids_and_snap(bench):
    assert bench._parse_grids("1:2048,1024 2:48 3:10,8") == {
        1: (2048, 1024), 2: (48, None), 3: (10, 8)}
    assert bench.parse_args([]).grids is None          # -> published grids
    assert bench._snap("1.66667", 1.0, 1) == (2 / 3, 5 / 3)
    assert bench._snap("0.333333", 0.0, 1) == (1 / 3, 4 / 3)
    assert bench._snap("2.25", 2.0, 2) is None          # already exact
    assert bench._snap("2.875", 1.0, 1) is None
    g = dict(bench.PUBLISHED_GRIDS)
    assert bench.published_setup(g, {1, 2, 3}) == "caption"
    g[1] = (512, 256)
    assert bench.published_setup(g, {1, 2, 3}) == "cells"
    assert bench.published_setup(g, {2, 3}) == "caption"
    g[2] = (12, 8)
    assert bench.published_setup(g, {1, 2}) is None


def test_pick_reads_the_reported_node(bench):
    """Grid mode reads k at node round(k n) mod n, per axis."""
    n = 48
    zone = np.arange(n * n, dtype=float).reshape(1, n, n)
    # triangular K = (1/3, -1/3): nodes 16 and -16 mod 48 = 32
    assert bench._pick(zone, (1 / 3, -1 / 3), n, True) == 16 * n + 32
    assert bench._pick(zone, (0.5, 0.5), n, True) == 24 * n + 24
    chain = np.arange(10, dtype=float).reshape(1, 10)
    assert bench._pick(chain, (0.5,), 10, True) == 5
    # single k: the array holds that one value
    assert bench._pick(np.array([[7.5 + 0j]]), (0.5,), 10, False) == 7.5


def test_read_coeffs_adds_the_cluster_sizes_errors_in_quadrature(bench):
    """The per-cluster-size split is summed; the errors add in quadrature."""
    path = bench.data_path("MC_patched/TFIM/square/gs_energy_series_coeffs_"
                           "2d_square_tfim_sigma1_order13.csv")
    split = collections.defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            split[int(r["order"])].append((float(r["prefactor"]),
                                           float(r["error"])))
    got = bench.read_coeffs(path)
    assert sorted(got) == sorted(split)
    for o, parts in split.items():
        v = math.fsum(x for x, _ in parts)
        e = math.sqrt(math.fsum(y * y for _, y in parts))
        scale = math.fsum(abs(x) for x, _ in parts)
        assert got[o][0] == pytest.approx(v, rel=0, abs=1e-14 * scale)
        assert got[o][1] == pytest.approx(e, rel=1e-13)
    o = max(split)                       # 12 cluster sizes at order 13
    assert len(split[o]) == 12
    assert got[o][1] < 0.9 * sum(y for _, y in split[o])     # not linear


def test_summary_statistics_match_the_papers_definitions(bench):
    """<= in 'within', and the paper's median kept[n // 2] (upper median)."""
    s = bench.SERIES[0]
    kept = [1.5, 0.2, 1.0, 2.0]
    t = bench.Row(s, [0.5], {}, set(), None, kept=kept,
                  compared=[*kept, 3.0], compared_rows=[])
    rows = [{"sign_flip": True}, {"sign_flip": False}, {"sign_flip": False}]
    sm = bench.summarize([t], rows)
    assert sm["evaluated"] == 3
    assert (sm["compared"], sm["kept"]) == (5, 4)
    assert sm["max_kept"] == 2.0
    assert sm["median_kept"] == 1.5      # sorted [0.2, 1.0, 1.5, 2.0][2]
    assert sm["within1_kept"] == 50.0    # 0.2 and exactly 1.0
    assert sm["within2_kept"] == 100.0   # exactly 2.0 counts
    assert sm["within1_all"] == 40.0
    assert sm["within2_all"] == 80.0
    assert sm["flips"] == 1
    assert sm["shift"] is None
    t.kept = [0.3, 0.1, 0.2]             # odd count: the plain median
    assert bench.summarize([t], rows)["median_kept"] == 0.2


def test_table_rules_on_synthetic_cells(bench):
    """sigma floor, majority coverage, analytic and dropped-top exclusions."""
    i = next(k for k, s in enumerate(bench.SERIES) if s.drop_order == 10)
    s = bench.SERIES[i]
    rows = []
    for sigma in (0.3, 0.5, 1.0, 2.0):
        for o in range(2, 9):
            rows.append(_row(s, sigma, o, 0.1 * o + sigma))
    rows.append(_row(s, 1.0, 9, 5.0))       # 1 of 3 exponents: not tabulated
    rows.append(_row(s, 1.0, 10, 30.0))     # 2 of 3: tabulated, top -> dropped
    rows.append(_row(s, 2.0, 10, 20.0))
    rows.append(_row(s, 0.3, 10, 99.0))     # below SIG_MIN: never counts
    (t,) = bench.build_rows([i], rows)
    assert t.sigmas == [0.5, 1.0, 2.0]
    assert sorted(t.cells) == [2, 3, 4, 5, 6, 7, 8, 10]
    assert t.excluded == {2, 3, 10}
    assert t.cells[10] == 30.0
    assert t.row_max == pytest.approx(0.8 + 2.0)
    assert len(t.compared) == 7 * 3 + 2
    assert len(t.kept) == 5 * 3

    # Exactly half the exponents is not a majority.
    half = [_row(s, sg, o, 0.5) for sg in (0.5, 1.0) for o in (2, 4)]
    half.append(_row(s, 1.0, 5, 0.5))
    (h,) = bench.build_rows([i], half)
    assert sorted(h.cells) == [2, 4]

    st = bench.Style(color=False)
    text = bench.render_table([t], bench.PUBLISHED_GRIDS, st, control=True)
    line = next(x for x in text.splitlines() if x.startswith("  0qp"))
    cells = line.split("│")[1].split() + line.split("│")[0].split()[-3:]
    assert "30.0†" in cells and "(2.3)" in cells and "2.8" in cells
    assert "5.0" not in line                 # order 9 fails the majority rule
    assert "0.5–2 (3)" in line


def test_grid_shift_is_reported_in_sigma_mc(bench):
    """Delta-grid = |c(n) - c(n_control)| / sigma_MC; an off-grid row is
    marked and left out of the summary's largest move."""
    s = _series(bench, "cubic 1qp k=(pi,pi,pi)")
    i = bench.SERIES.index(s)
    rows = [_row(s, 1.0, 4, 0.2, tn=100.0, mc_err=2.0, grid_shift=1e-2),
            _row(s, 2.0, 4, 0.1, tn=100.0, mc_err=20.0, grid_shift=1e-2)]
    (t,) = bench.build_rows([i], rows)
    assert t.shift == pytest.approx(0.5)     # 1e-2 * 100 / 2
    assert t.shift_at["sigma"] == 1.0
    sm = bench.summarize([t], rows)
    assert sm["shift"] == pytest.approx(0.5) and sm["shift_row"] is t
    sm = bench.summarize([t], rows, skip_shift={s})
    assert sm["shift"] is None and sm["shift_skipped"]

    st = bench.Style(color=False)
    line = next(x for x in bench.render_table(
        [t], bench.PUBLISHED_GRIDS, st, control=True,
        marked_shift={s}).splitlines() if x.startswith("  1qp"))
    assert line.endswith("0.50*")
    notes = bench.render_notes(st, control=False, marked_shift=set())
    assert "not measured" in notes


def test_published_check_names_the_differing_cell(bench):
    """A run that prints the published row has no differences; a cell that
    rounds differently is reported, with the d = 1 grid hint only at the
    caption's d = 1 grid."""
    s = _series(bench, "chain 0qp")
    i = bench.SERIES.index(s)
    printed, rmax = bench.published_row(s)
    rows = [_row(s, 1.0, o, float(v)) for o, v in printed.items()]
    (t,) = bench.build_rows([i], rows)
    check = bench.compare_published([t])
    assert (check["cells"], check["maxima"], check["diffs"]) == (13, 1, [])
    assert f"{t.row_max:.1f}" == rmax

    rows = [r if r["order"] != 11 else {**r, "pull": 1.2506} for r in rows]
    (t,) = bench.build_rows([i], rows)
    check = bench.compare_published([t])
    assert [(o, own, pub) for _, o, own, pub in check["diffs"]] == [
        (11, "1.3", "1.2")]
    st = bench.Style(color=False)
    text = bench.render_differences(check, bench.PUBLISHED_GRIDS, st)
    assert "chain 0qp (Langheld 2022), order 11" in text
    assert "1.3 here, 1.2 published" in text and "1.2506" in text
    assert "1:512,256" in text
    grids = {**bench.PUBLISHED_GRIDS, 1: (512, 256)}
    assert "1:512,256" not in bench.render_differences(check, grids, st)


def test_quick_run_on_every_series(bench, capsys, tmp_path):
    """--quick through main() on all 15 series: every reader, both order-1
    sign flips (triangular Gamma and K), the control pass on the control
    grid, and the output a non-published run should print."""
    out = tmp_path / "cells.csv"
    rc = bench.main(["--quick", "--jobs", "1", "--no-color", "--csv", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "NOT the published comparison" in text
    assert "cubic, Λ = ℤ³" in text and "triangular, Λ = A_trig ℤ²" in text
    assert "Summary" in text and "orientation only" in text
    assert "printed cells as published" not in text   # not a published setup
    # The TITLE names the method, the version line the library.
    assert "graph-zeta against published" in text or "graph-zeta vs" in text
    assert re.search(r"gzl \S+ \(source checkout\)", text), text[:400]

    with open(out, newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == bench.CSV_COLUMNS
        rows = list(reader)
    assert {(r["config"], r["source"]) for r in rows} == set(
        bench.PUBLISHED_TABLE)
    assert all(int(r["order"]) <= bench.QUICK_ORDER_MAX for r in rows)
    pulls = [float(r["pull"]) for r in rows]
    assert all(math.isfinite(p) for p in pulls)
    assert max(pulls) < 4.0
    flipped = sorted(r["config"] for r in rows if r["sign_flip"] == "True")
    assert flipped == ["triangular 1qp Gamma", "triangular 1qp K"]
    assert all(r["order"] == "1" for r in rows if r["sign_flip"] == "True")
    for r in rows:                          # reported on n_hi, control on n_lo
        d = bench.LATTICE_DIM[r["lattice"]]
        assert int(r["n_points"]) == bench.QUICK_GRIDS[d][0]
    assert sum(1 for r in rows if float(r["grid_shift"]) > 0) > len(rows) // 2


def test_quick_run_without_a_control_grid(bench, capsys):
    rc = bench.main(["--quick", "--only", "cubic 1qp k=0", "--grids", "3:6",
                     "--jobs", "1", "--no-color"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "not measured: no control grid" in text
    grids = next(x for x in text.splitlines() if x.startswith("  grids"))
    assert "n=6" in grids and "control" not in grids


def test_default_jobs_stay_within_physical_memory(bench, monkeypatch):
    monkeypatch.setattr(bench.os, "cpu_count", lambda: 10)
    monkeypatch.setattr(bench, "_physical_memory", lambda: 16e9)
    assert bench.default_jobs(123, quick=False) == (5, True)    # 16 / 3.2
    assert bench.default_jobs(123, quick=True) == (8, False)
    monkeypatch.setattr(bench, "_physical_memory", lambda: 64e9)
    assert bench.default_jobs(123, quick=False) == (8, False)
    assert bench.default_jobs(3, quick=False) == (3, False)
    monkeypatch.setattr(bench, "_physical_memory", lambda: None)
    assert bench.default_jobs(123, quick=False) == (8, False)


class _Stream(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


def test_progress_status_line_and_heartbeat(bench):
    """A terminal gets a status line rewritten in place and cleared before
    each printed line; a log gets a heartbeat only after a silence."""
    st = bench.Style(color=False)
    tty = _Stream(True)
    pr = bench.Progress(123, time.perf_counter(), st, stream=tty)
    pr.done = 37
    pr.status()
    assert tty.getvalue().startswith("\r  … 37/123 exponents done,")
    pr.line("[ 1/15] done")
    text = tty.getvalue()
    assert text.endswith("\r[ 1/15] done\n")
    assert "\r" + " " * pr_width(text) + "\r" in text

    log = _Stream(False)
    pr = bench.Progress(10, time.perf_counter(), st, stream=log)
    pr.status()
    assert log.getvalue() == ""              # no silence yet
    pr.HEARTBEAT = 0.0
    pr.status()
    assert log.getvalue().startswith("  … 0/10 exponents done,")
    assert "\r" not in log.getvalue()


def pr_width(text):
    """Width of the first status line written to a fake terminal."""
    return len(text.split("\r")[1])


def test_unknown_series_is_refused(bench, capsys):
    assert bench.main(["--only", "no such lattice", "--no-color"]) == 2
    assert "Available" in capsys.readouterr().out

