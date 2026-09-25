# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""graph-zeta against Monte Carlo series of the long-range TFIM.

Reproduces the Monte Carlo comparison of

    A. A. Buchheit and A. Rupp, "Graph lattice sums and graph zeta functions
    for long-range interacting quantum lattice models", arXiv:2609.18918
    (2026), Sec. 8.3, Table 2 and Fig. 7,

from the shipped corpora and Monte Carlo files alone, and prints the table
in a terminal.  A plain run is the paper's computation at the grids its
caption states; ``--quick`` is a seconds-long smoke run that is NOT the
published comparison.  A run at a published setup checks its printed cells
against the published table and lists every cell that differs.

Run with:  python benchmarks/mc_comparison.py [--jobs N] [--quick] [--csv out.csv]

(from a source checkout, put the checkout first:
``PYTHONPATH=$(git rev-parse --show-toplevel) python benchmarks/mc_comparison.py``).

Reproducing the published cells
-------------------------------
A plain run prints 194 of the 195 published cells.  The one that differs is
chain 0qp (Langheld 2022), order 11, which prints 1.3 instead of 1.2: its
largest pull, at sigma = 1/2, is 1.2506 at n = 2048 and 1.2434 at n = 512.
Every summary figure is the same at both grids, and ``--grids '1:512,256'``
reproduces the published cells digit for digit.

Model and units
---------------
The long-range transverse-field Ising model on a Bravais lattice
Lambda = A Z^d is

    H = -h sum_x sigma^z_x - (J/2) sum_{x != y} sigma^x_x sigma^x_y |x-y|^-nu,

with nu = d + sigma.  Measured in units of the bare one-quasiparticle gap 2h
and shifted by -h|Lambda|, this is the pCUT normal form with expansion
parameter lambda = J/(2h); the ferromagnet is lambda > 0.  Two observables
are compared:

  0qp   ground-state energy density, series starting at order 2;
  1qp   excitation gap at one fixed momentum, series starting at order 1.

Momenta are in reciprocal-basis (fractional) units, so k = 1/2 is the zone
boundary and k = 0 the zone centre; the triangular K point is (1/3, -1/3).

The sources label their files differently: the .dat tables (Fey 2020 thesis,
Fey et al. PRL 2019) by alpha = nu, the .csv releases (Langheld et al. 2022,
Adelhardt) by sigma.  Each series therefore carries the shift from its file
label to sigma.  The labels are used exactly as written, so a rounded label
such as alpha = 1.66667 is evaluated at nu = 1.66667, not 5/3 -- that is what
the published table did; ``--snap-sigma`` evaluates at the exact rational
instead.

What a table cell is
--------------------
One comparison is one (series, sigma, order r) coefficient, scored by its
pull |c_r - c_r^MC| / sigma_MC.  A table cell is the maximum pull over the
series' exponents with sigma >= 1/2; at sigma = 0.1 the Monte Carlo sums
converge too slowly to be trusted, and exponents 0.1 < sigma < 1/2 are
evaluated and written to the CSV but not tabulated.  An order is tabulated
only where the reference supplies it for a strict majority of the series'
exponents (analytic orders whenever any value exists), which drops the
sparsely covered top orders of some tables.

Two kinds of cell are shown but excluded from the row maximum and from the
"kept" statistics (grey in the published table):

* orders r <= 3, in parentheses.  Every corpus graph there is a bridge or a
  cycle without momentum, both closed forms (Epstein zeta, zeta_circle), so
  the graph-zeta value is exact to machine precision and the pull measures
  only the Monte Carlo error;
* the top order of the 0qp tables of the Fey 2020 thesis (F.4 square, F.7
  triangular), marked with a dagger, which is underconverged: for the square
  lattice the two available Monte Carlo runs of that series agree to
  0.3 sigma_MC at order 9 but differ by 25 sigma_MC at order 10, and the
  quoted uncertainties degrade by a factor of 75 from order 9 to order 10.

Some sources store the order-1 gap coefficient with the opposite overall
sign.  Order 1 is analytically -Z_nu(k), so the branch is unambiguous: the
reference is flipped when its negative is closer, and every flip is counted
in the progress line and the CSV (``sign_flip``).

Convergence column
------------------
Each exponent is evaluated on the reported grid n_hi and on a coarser
control grid n_lo.  The table's Delta-grid column is the row's largest
|c(n_hi) - c(n_lo)| / sigma_MC: how far a coefficient moved from the control
grid, in the units of the pull.  The CSV's ``grid_shift`` column keeps the
relative form |c(n_hi) - c(n_lo)| / |c(n_hi)|.  A move below
0.1 sigma_MC, the printed resolution of a cell, cannot shift a printed pull
by more than one unit in its last digit.  A larger move is highlighted in
the table, and the summary names the cell.  When the series converges fast
in n, the move from n_lo to n_hi overstates the error left at n_hi.  The
largest move in the published setup is cubic k = (1/2,1/2,1/2) at
sigma = 2, order 10: 0.28 sigma_MC from n = 8 to 10, but 0.04 sigma_MC from
n = 10 to 12, where the pull is 0.27 (0.31 at n = 10).  The treewidth >= 3
blocks that enter the chain 1qp series at orders 10 and 11 converge only
like 1/n on the torus, which is why d = 1 needs a fine grid.

Sources
-------
Short forms used below; ``gzl/data/PROVENANCE.csv`` maps every file to
its source and the distribution's ``NOTICE`` gives the full references.

* Fey 2020 thesis, Table F.x -- S. Fey, doctoral thesis, FAU Erlangen-Nuernberg
  (2020), Appendix F.
* Fey et al. PRL 2019, Table y -- S. Fey, S. C. Kapfer and K. P. Schmidt,
  Phys. Rev. Lett. 122, 017203 (2019), Supplemental Material, Tables I-IV.
* Langheld et al. 2022 (Zenodo) -- SciPost Phys. 13, 088 (2022); data record
  doi:10.5281/zenodo.6645107.
* Adelhardt, private communication -- unpublished coefficients by
  P. Adelhardt, distributed with permission.
"""

from __future__ import annotations

import argparse
import csv
import collections
import importlib.metadata
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import numpy as np

import gzl
from gzl import compute_series_coefficients, data_path
from gzl._lattices import LATTICES as NAMED_LATTICES

# --------------------------------------------------------------------------
# Comparison rules (as in the published table)
# --------------------------------------------------------------------------

SIG_FLOOR = 0.1     # sigma at or below this is not evaluated at all
SIG_MIN = 0.5       # exponents below this are evaluated but not tabulated
N_ANALYTIC = 3      # orders 1..3 are closed-form
ORDERS = range(1, 14)

# The published grids, (n_hi, n_lo) per dimension, as the caption of Table 2
# states them: n_hi is reported, n_lo is the control grid that exposes the
# residual discretisation error.
PUBLISHED_GRIDS = {1: (2048, 1024), 2: (48, 32), 3: (10, 8)}

# The reported grids that reproduce every published cell digit for digit
# (module docstring).
CELL_GRIDS = {1: 512, 2: 48, 3: 10}

# A move from the control grid of at least this many sigma_MC is highlighted:
# the printed resolution of a cell.
SHIFT_FLAG = 0.1

# The published summary, arXiv:2609.18918 Sec. 8.3 (Table 2 caption and
# text), with the format each figure is printed in there.
PUBLISHED = {
    "compared": (998, "d"),         # coefficients in the table, all data
    "kept": (753, "d"),             # after the parenthesised exclusions
    "max_kept": (3.1, ".1f"),
    "median_kept": (0.12, ".2f"),
    "within1_kept": (90, ".0f"),    # per cent
    "within1_all": (89, ".0f"),
    "within2_all": (97, ".0f"),
}
PUBLISHED_RUNTIME = "within 5 min on 8 cores (Apple M1 Max)"

# Table 2 as printed, keyed like the CSV (config, source): the cells of
# orders 1..13 ("-" = no reference value) and, after "|", the row maximum.
# Transcribed by program from the paper's LaTeX source (the rules below
# rebuild all 195 cells and 15 maxima from a run at CELL_GRIDS).
# A run at a published setup compares its printed cells with these.
PUBLISHED_TABLE = {
    ("chain 0qp", "SciPost2022 (Zenodo)"):
        "-   -   3.5 1.4 0.4 1.5 1.8 1.1 1.5 1.3 1.2 1.4 1.5 | 1.8",
    ("chain 1qp k=0", "Diss2020 F.2"):
        "-   0.0 0.5 0.3 0.4 0.3 0.2 0.2 0.2 -   -   -   -   | 0.4",
    ("chain 1qp k=0", "SciPost2022 (Zenodo)"):
        "-   -   2.5 1.8 0.6 1.2 2.0 1.5 1.3 0.9 1.8 -   -   | 2.0",
    ("chain 1qp k=pi", "Diss2020 F.3"):
        "-   -   0.3 0.4 0.7 0.3 0.3 0.2 0.2 -   -   -   -   | 0.7",
    ("chain 1qp k=pi", "Adelhardt2024"):
        "-   2.4 1.9 2.3 1.6 2.1 1.3 2.0 1.8 1.4 -   -   -   | 2.3",
    ("square 0qp", "Diss2020 F.4"):
        "-   0.2 0.2 0.3 0.3 0.3 0.2 0.4 0.2 30.9 -  -   -   | 0.4",
    ("square 0qp (Adelhardt)", "Adelhardt (2D, unpublished)"):
        "-   3.5 1.4 2.0 1.6 2.7 2.1 2.1 1.9 0.7 0.9 2.1 1.0 | 2.7",
    ("square 1qp k=0", "PRL2019 Tab. I / F.5"):
        "-   0.3 0.4 0.2 0.7 0.2 0.2 0.3 0.2 -   -   -   -   | 0.7",
    ("square 1qp k=0 (Adelhardt)", "Adelhardt (2D, unpublished)"):
        "-   1.5 0.9 0.9 1.1 2.6 1.2 1.8 2.1 3.1 1.8 -   -   | 3.1",
    ("square 1qp k=(pi,pi)", "PRL2019 Tab. III / F.6"):
        "-   0.1 0.2 0.3 0.4 0.2 0.2 0.2 0.1 -   -   -   -   | 0.4",
    ("triangular 0qp", "Diss2020 F.7"):
        "-   0.2 0.2 0.2 0.2 0.2 0.2 0.3 0.1 22.8 -  -   -   | 0.3",
    ("triangular 1qp Gamma", "PRL2019 Tab. II / F.8"):
        "0.2 0.2 0.9 1.2 0.4 0.2 0.5 0.2 0.2 -   -   -   -   | 1.2",
    ("triangular 1qp K", "PRL2019 Tab. IV / F.9"):
        "0.3 0.1 0.2 0.1 0.1 0.2 0.3 0.1 0.2 -   -   -   -   | 0.3",
    ("cubic 1qp k=0", "Diss2020 F.10"):
        "0.4 0.4 0.6 0.5 0.5 0.2 0.2 0.5 0.2 -   -   -   -   | 0.5",
    ("cubic 1qp k=(pi,pi,pi)", "Diss2020 F.11"):
        "0.2 0.2 0.2 0.2 0.2 0.1 0.1 0.2 0.2 0.3 -   -   -   | 0.3",
}


def published_row(s) -> tuple[dict, str] | None:
    """({order: printed cell}, printed row maximum) of one series, or None."""
    text = PUBLISHED_TABLE.get((s.config, s.source))
    if text is None:
        return None
    cells, rmax = text.split("|")
    toks = cells.split()
    return ({o: c for o, c in enumerate(toks, 1) if c != "-"}, rmax.strip())


# --quick: a smoke preset, NOT the published comparison.  One exponent per
# series (the largest, which converges fastest), orders <= 8, small grids.
QUICK_GRIDS = {1: (64, 32), 2: (12, 8), 3: (6, 4)}
QUICK_ORDER_MAX = 8

# --------------------------------------------------------------------------
# Lattices and configurations
# --------------------------------------------------------------------------

# The lattices are the library's, by name: a series carries the name
# through to compute_series_coefficients, which resolves it.  Only the
# dimension is needed here, and it is the row count of the library's own
# matrix -- this file keeps no second copy of the cells to drift from it.
LATTICE_DIM = {name: len(rows) for name, rows in NAMED_LATTICES.items()}
LATTICE_TITLE = {
    "chain": "chain, Λ = ℤ",
    "square": "square, Λ = ℤ²",
    "triangular": "triangular, Λ = A_trig ℤ²",
    "cubic": "cubic, Λ = ℤ³",
}


@dataclass(frozen=True)
class Series:
    """One published Monte Carlo series family: one row of the table."""

    lattice: str            # a lattice name the library resolves
    label: str              # row label in the table
    ref: str                # bibliographic short form of the source
    tag: str                # compact source tag for the table column
    corpus: str             # "0qp" or "1qp"
    momentum: tuple | None  # reciprocal-basis units; None for the ground state
    reader: str             # "dat", "csv" or "coeffs"
    sig_shift: float        # sigma = float(file label) - sig_shift
    points: tuple           # ((file label, file name), ...), one per exponent
    config: str             # CSV "config" column, the row key of Table 2
    source: str             # CSV "source" column, the source tag of the row
    drop_order: int | None = None   # underconverged top order, excluded

    @property
    def name(self) -> str:
        return f"{self.lattice} {self.label}"


def _files(pattern: str, labels) -> tuple:
    """(label, file name) pairs from a file-name pattern with one ``{}``."""
    return tuple((lab, pattern.format(lab)) for lab in labels)


_DAT_1D = "TFIM_lr_1d__MC__list_of_prefactors__{k}__alpha_{{}}.dat"
_DAT_SQ = "TFIM_lr_2d_square__MC__list_of_prefactors__{k}__alpha_{{}}.dat"
_DAT_TR = "TFIM_lr_2d_triangular__MC__list_of_prefactors__{k}__alpha_{{}}.dat"
_DAT_CU = "TFIM_lr_3d_cube__MC__list_of_prefactors__{k}__alpha_{{}}.dat"

_LANGHELD = "Langheld et al. 2022 (Zenodo)"
_ADELHARDT = "Adelhardt, private communication"

SERIES = (
    # ---- chain, d = 1 --------------------------------------------------
    Series("chain", "0qp", ref=_LANGHELD, tag="Langheld 2022",
           corpus="0qp", momentum=None, reader="csv", sig_shift=0.0,
           points=_files("gs_energy_series_1d_chain_tfim_sigma{}_order13.csv",
                         ["%.1f" % s for s in (0.5, 1.0, 2.0, 3.0)]),
           config="chain 0qp", source="SciPost2022 (Zenodo)"),
    Series("chain", "1qp k=0", ref="Fey 2020 thesis, Table F.2",
           tag="Fey 2020 F.2",
           corpus="1qp", momentum=(0.0,), reader="dat", sig_shift=1.0,
           points=_files(_DAT_1D.format(k="k0"),
                         ["1.25", "1.66667", "2", "2.25", "2.5", "2.75",
                          "2.875", "3", "3.125", "3.25", "3.5", "4", "6",
                          "10"]),
           config="chain 1qp k=0", source="Diss2020 F.2"),
    Series("chain", "1qp k=0", ref=_LANGHELD, tag="Langheld 2022",
           corpus="1qp", momentum=(0.0,), reader="csv", sig_shift=0.0,
           points=_files("1qp_gap_series_1d_chain_tfim_k0_sigma{}_order11.csv",
                         ["%.1f" % s for s in (0.5, 1.0, 2.0, 3.0)]),
           config="chain 1qp k=0", source="SciPost2022 (Zenodo)"),
    Series("chain", "1qp k=½", ref="Fey 2020 thesis, Table F.3",
           tag="Fey 2020 F.3",
           corpus="1qp", momentum=(0.5,), reader="dat", sig_shift=1.0,
           points=_files(_DAT_1D.format(k="kPi"),
                         ["1.5", "1.66667", "2", "2.25", "2.5", "3", "4", "5",
                          "7"]),
           config="chain 1qp k=pi", source="Diss2020 F.3"),
    Series("chain", "1qp k=½", ref=_ADELHARDT, tag="Adelhardt p.c.",
           corpus="1qp", momentum=(0.5,), reader="csv", sig_shift=0.0,
           points=_files("1qp_gap_series_1d_chain_tfim_kPi_sigma{}_order10.csv",
                         ["%g" % s for s in (0.25, 0.333333, 0.5, 0.75, 1.0,
                                             1.25, 1.5, 2.0, 3.0, 4.0)]),
           config="chain 1qp k=pi", source="Adelhardt2024"),
    # ---- square, d = 2 -------------------------------------------------
    Series("square", "0qp", ref="Fey 2020 thesis, Table F.4",
           tag="Fey 2020 F.4",
           corpus="0qp", momentum=None, reader="dat", sig_shift=2.0,
           points=_files(_DAT_SQ.format(k="0qp"),
                         ["2.5", "3", "3.33333", "3.5", "4", "4.5", "5", "6",
                          "8", "10"]),
           config="square 0qp", source="Diss2020 F.4", drop_order=10),
    Series("square", "0qp", ref=_ADELHARDT, tag="Adelhardt p.c.",
           corpus="0qp", momentum=None, reader="coeffs", sig_shift=0.0,
           points=_files(
               "gs_energy_series_coeffs_2d_square_tfim_sigma{}_order13.csv",
               ["%g" % s for s in (0.5, 1.0, 2.0, 3.0, 4.0, 8.0)]),
           config="square 0qp (Adelhardt)",
           source="Adelhardt (2D, unpublished)"),
    Series("square", "1qp k=0", ref="Fey et al. PRL 2019, Table I",
           tag="Fey 2019 I",
           corpus="1qp", momentum=(0.0, 0.0), reader="dat", sig_shift=2.0,
           points=_files(_DAT_SQ.format(k="k0"),
                         ["2.25", "2.33333", "2.75", "3", "3.25", "3.33333",
                          "3.5", "4", "4.5", "5", "6", "8", "10"]),
           config="square 1qp k=0", source="PRL2019 Tab. I / F.5"),
    Series("square", "1qp k=0", ref=_ADELHARDT, tag="Adelhardt p.c.",
           corpus="1qp", momentum=(0.0, 0.0), reader="coeffs", sig_shift=0.0,
           points=_files(
               "1qp_gap_series_coeffs_2d_square_tfim_kx0_ky0_sigma{}_order11.csv",
               ["%g" % s for s in (0.5, 2.0, 3.0, 4.0, 8.0)]),
           config="square 1qp k=0 (Adelhardt)",
           source="Adelhardt (2D, unpublished)"),
    Series("square", "1qp k=(½,½)", ref="Fey et al. PRL 2019, Table III",
           tag="Fey 2019 III",
           corpus="1qp", momentum=(0.5, 0.5), reader="dat", sig_shift=2.0,
           points=_files(_DAT_SQ.format(k="kPi"),
                         ["3", "3.5", "4", "5", "6", "8", "10"]),
           config="square 1qp k=(pi,pi)", source="PRL2019 Tab. III / F.6"),
    # ---- triangular, d = 2 ---------------------------------------------
    Series("triangular", "0qp", ref="Fey 2020 thesis, Table F.7",
           tag="Fey 2020 F.7",
           corpus="0qp", momentum=None, reader="dat", sig_shift=2.0,
           points=_files(_DAT_TR.format(k="0qp"),
                         ["2.25", "2.5", "3", "3.5", "4", "5", "6", "8", "10"]),
           config="triangular 0qp", source="Diss2020 F.7", drop_order=10),
    Series("triangular", "1qp Γ", ref="Fey et al. PRL 2019, Table II",
           tag="Fey 2019 II",
           corpus="1qp", momentum=(0.0, 0.0), reader="dat", sig_shift=2.0,
           points=_files(_DAT_TR.format(k="k0"),
                         ["2.25", "2.33333", "2.5", "3", "3.33333", "3.5", "4",
                          "5", "6", "8"]),
           config="triangular 1qp Gamma", source="PRL2019 Tab. II / F.8"),
    # The file names round the K point to kx = -ky = 2.09 = 2pi/3; the exact
    # wavevector is (1/3, -1/3) in reciprocal-basis units.
    Series("triangular", "1qp K", ref="Fey et al. PRL 2019, Table IV",
           tag="Fey 2019 IV",
           corpus="1qp", momentum=(1 / 3, -1 / 3), reader="dat", sig_shift=2.0,
           points=_files(_DAT_TR.format(k="kx2.09_ky-2.09"),
                         ["3.5", "4", "5", "6", "8"]),
           config="triangular 1qp K", source="PRL2019 Tab. IV / F.9"),
    # ---- cubic, d = 3 --------------------------------------------------
    Series("cubic", "1qp k=0", ref="Fey 2020 thesis, Table F.10",
           tag="Fey 2020 F.10",
           corpus="1qp", momentum=(0.0, 0.0, 0.0), reader="dat", sig_shift=3.0,
           points=_files(_DAT_CU.format(k="k0"),
                         ["3.25", "3.5", "4", "5", "6", "7", "8", "9", "10"]),
           config="cubic 1qp k=0", source="Diss2020 F.10"),
    # These file names carry the MC control parameters rho and gamma, so the
    # alpha label and the file suffix are paired explicitly.
    Series("cubic", "1qp k=(½,½,½)", ref="Fey 2020 thesis, Table F.11",
           tag="Fey 2020 F.11",
           corpus="1qp", momentum=(0.5, 0.5, 0.5), reader="dat", sig_shift=3.0,
           points=tuple(
               (a, _DAT_CU.format(k="kPi").format(suffix)) for a, suffix in (
                   ("3.5", "3.5_rho1.75_gamma1.75"), ("4", "4_rho2_gamma2"),
                   ("5", "5_rho2.5_gamma2.5"), ("6", "6_rho3_gamma3"),
                   ("7", "7_rho3.5_gamma3.5"), ("8", "8_rho4_gamma4"),
                   ("9", "9_rho4.5_gamma4.5"), ("10", "10_rho4_gamma4"))),
           config="cubic 1qp k=(pi,pi,pi)", source="Diss2020 F.11"),
)

# Columns of the cell-level CSV.
CSV_COLUMNS = ("config", "source", "lattice", "sigma", "alpha", "order", "tn",
               "mc", "mc_err", "pull", "rel", "grid_shift", "n_points",
               "sign_flip")

# --------------------------------------------------------------------------
# Readers.  Each returns {order: (value, standard_error)} and drops the rows
# whose value and error are both zero, the sources' "not measured" marker.
# --------------------------------------------------------------------------


def read_dat(path) -> dict:
    """Whitespace table ``order value error`` (Fey thesis and PRL files)."""
    out = {}
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            if len(tok) >= 3:
                o, v, e = int(tok[0]), float(tok[1]), float(tok[2])
                if e > 0 and not (v == 0.0 and e == 0.0):
                    out[o] = (v, e)
    return out


def read_csv(path) -> dict:
    """One row per order: ``order,prefactor,error,relative error``."""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            o, v, e = int(r["order"]), float(r["prefactor"]), float(r["error"])
            if e > 0 and not (v == 0.0 and e == 0.0):
                out[o] = (v, e)
    return out


def read_coeffs(path) -> dict:
    """Per-cluster-size split, one row per (order, Nsites).  The coefficient
    is the sum over Nsites; independent errors add in quadrature."""
    agg = collections.defaultdict(lambda: [0.0, 0.0])
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            o = int(r["order"])
            agg[o][0] += float(r["prefactor"])
            agg[o][1] += float(r["error"]) ** 2
    return {o: (v, e ** 0.5) for o, (v, e) in agg.items()
            if not (v == 0.0 and e == 0.0)}


READERS = {"dat": read_dat, "csv": read_csv, "coeffs": read_coeffs}

# --------------------------------------------------------------------------
# Evaluation.  The unit of work is one (series, exponent): one corpus pass on
# the reported grid and one on the control grid.  Units are independent, so
# they run in a process pool with no shared state.
# --------------------------------------------------------------------------


class Point(NamedTuple):
    """One exponent of one series, ready to evaluate (picklable)."""

    index: int              # position of the series in SERIES
    sigma: float
    nu: float
    mc: dict                # {order: (value, error)}
    corpus: str
    lattice: str            # the name, resolved by the library
    momentum: tuple | None
    n_hi: int
    n_lo: int | None        # None: no control pass
    grid_mode: bool


def _snap(label: str, shift: float, d: int) -> tuple[float, float] | None:
    """(sigma, nu) at the exact rational a rounded label stands for, or None.

    ``1.66667`` is 5/3 printed to six digits: snap when a denominator <= 12
    lies within 1e-4 and the label is not already that number.
    """
    x = float(label)
    q = Fraction(x).limit_denominator(12)
    if float(q) == x or abs(float(q) - x) >= 1e-4:
        return None
    s = q - Fraction(shift)
    return float(s), float(s + d)


def _pick(arr, mom, n: int, grid: bool) -> float:
    """The coefficient at the reported wavevector.

    Single-k: the array holds that one value.  Grid mode: it holds the whole
    zone, shape (1,) + (n,) * d, and the wavevector is read at the node
    round(k_i n) mod n -- the nearest node when k is not on the grid.
    """
    a = np.asarray(arr)
    if not grid:
        return float(a.ravel()[0].real)
    idx = tuple(int(round(float(k) * n)) % n for k in np.atleast_1d(mom))
    return float(a[(0,) + idx].real)


def _peak_rss() -> int | None:
    """This process's peak resident memory in bytes (None where unknown)."""
    try:
        import resource
    except ImportError:                     # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024    # Linux: KiB


def evaluate_point(p: Point):
    """Evaluate one exponent.

    Returns (series index, rows, flips, seconds, peak bytes), where the peak
    is the resident high-water mark of the process that ran the unit.
    """
    t0 = time.perf_counter()
    s = SERIES[p.index]
    mom = None if p.momentum is None else np.array(p.momentum, dtype=float)
    # Grid mode asks for the whole Brillouin zone (momentum=None) and reads
    # the reported wavevector out of it instead of asking for it alone.
    grid = p.grid_mode and mom is not None
    kw = dict(momentum=None if grid else mom, order_max=max(p.mc))
    nu = np.array([p.nu])
    c_hi = compute_series_coefficients(p.corpus, nu, p.lattice, p.n_hi, **kw)
    c_lo = (compute_series_coefficients(p.corpus, nu, p.lattice, p.n_lo, **kw)
            if p.n_lo else None)

    rows, flips = [], 0
    for o in sorted(p.mc):
        v, e = p.mc[o]
        tn = _pick(c_hi[o], mom, p.n_hi, grid)
        # Some sources store the order-1 gap with the opposite overall sign.
        # Order 1 is analytically -Z_nu(k), so the branch is unambiguous.
        flip = (o == 1 and abs(tn + v) < abs(tn - v))
        if flip:
            v = -v
            flips += 1
        shift = None
        if c_lo is not None:
            tl = _pick(c_lo[o], mom, p.n_lo, grid)
            shift = abs(tn - tl) / max(abs(tn), 1e-300)
        rows.append(dict(
            config=s.config, source=s.source, lattice=s.lattice,
            sigma=p.sigma, alpha=p.nu, order=o,
            tn=tn, mc=v, mc_err=e,
            pull=abs(tn - v) / e,
            rel=abs(tn - v) / max(abs(v), 1e-300),
            grid_shift=shift, n_points=p.n_hi, sign_flip=flip))
    return p.index, rows, flips, time.perf_counter() - t0, _peak_rss()


def plan(series_idx, grids, *, control=True, grid_mode=False, snap=False,
         quick=False):
    """Read the references and build the work units.

    Returns (points, snapped): the Point list and the (series, label, sigma
    as written, sigma used) of every snapped label.
    """
    points, snapped = [], []
    for i in series_idx:
        s = SERIES[i]
        d = LATTICE_DIM[s.lattice]
        n_hi, n_lo = grids[d]
        corpus = str(data_path(f"tfim_softcore_corpus_{s.corpus}.npz"))
        todo = []
        for label, fname in s.points:
            mc = READERS[s.reader](data_path(f"MC_patched/TFIM/{s.lattice}/{fname}"))
            if not mc:
                continue
            sigma = float(label) - s.sig_shift
            if sigma <= SIG_FLOOR:
                continue
            nu = sigma + d
            if snap and (exact := _snap(label, s.sig_shift, d)) is not None:
                snapped.append((s, label, sigma, exact[0]))
                sigma, nu = exact
            todo.append((sigma, nu, mc))
        if quick and todo:
            sigma, nu, mc = max(todo, key=lambda t: t[0])
            mc = {o: ve for o, ve in mc.items() if o <= QUICK_ORDER_MAX}
            todo = [(sigma, nu, mc)]
        for sigma, nu, mc in todo:
            points.append(Point(i, sigma, nu, mc, corpus, s.lattice,
                                s.momentum, n_hi,
                                n_lo if control else None, grid_mode))
    return points, snapped


def _cost(p: Point) -> float:
    """Rough relative cost, to start the heaviest units first."""
    d = LATTICE_DIM[SERIES[p.index].lattice]
    return p.n_hi ** d * 2.0 ** max(p.mc)


def off_grid(points) -> list[tuple[int, int]]:
    """Grid mode: (series index, n) whose wavevector is not an n-grid node."""
    bad = set()
    for p in points:
        if not p.grid_mode or p.momentum is None:
            continue
        for n in (p.n_hi, p.n_lo):
            if n and any(abs(k * n - round(k * n)) > 1e-9 for k in p.momentum):
                bad.add((p.index, n))
    return sorted(bad)

# --------------------------------------------------------------------------
# The table (the published rules)
# --------------------------------------------------------------------------


def shift_sigma(r: dict) -> float | None:
    """|c(n) - c(n_control)| / sigma_MC of one cell-level row, or None."""
    if r["grid_shift"] is None:
        return None
    return r["grid_shift"] * abs(r["tn"]) / r["mc_err"]


@dataclass
class Row:
    series: Series
    sigmas: list            # tabulated exponents, sorted
    cells: dict             # order -> max pull over sigma
    excluded: set           # analytic orders and the dropped top order
    row_max: float | None   # over the non-excluded orders
    kept: list              # pulls that enter the "kept" statistics
    compared: list          # every pull in a tabulated cell
    compared_rows: list     # the CSV rows behind `compared`
    shift: float | None = None      # max shift_sigma over compared_rows
    shift_at: dict | None = None    # the row where it occurs

    def top_row(self, order: int) -> dict:
        """The cell-level row that sets the cell of one order."""
        return max((r for r in self.compared_rows if r["order"] == order),
                   key=lambda r: r["pull"])


def build_rows(series_idx, rows) -> list[Row]:
    """Apply the table rules to the cell-level rows of the given series."""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["config"], r["source"])].append(r)
    out = []
    for i in series_idx:
        s = SERIES[i]
        per = collections.defaultdict(list)
        sigs = set()
        for r in by.get((s.config, s.source), ()):
            if r["sigma"] < SIG_MIN - 1e-9:
                continue
            per[r["order"]].append(r)
            sigs.add(r["sigma"])
        nsig = len(sigs)
        # An order is tabulated where the reference covers a strict majority
        # of the exponents; analytic orders whenever any value exists.
        per = {o: v for o, v in per.items()
               if o <= N_ANALYTIC or 2 * len(v) > nsig}
        # The published rule drops the top tabulated order of the two Fey 0qp
        # tables.  It is order 10 in both (pinned by the smoke test); naming
        # it keeps --quick's order cap from dropping a different order.
        top = s.drop_order if s.drop_order in per else None
        cells, excluded, kept, compared, compared_rows = {}, set(), [], [], []
        for o in sorted(per):
            pulls = [r["pull"] for r in per[o]]
            cells[o] = max(pulls)
            compared.extend(pulls)
            compared_rows.extend(per[o])
            if o <= N_ANALYTIC or o == top:
                excluded.add(o)
            else:
                kept.extend(pulls)
        row_max = max((cells[o] for o in cells if o not in excluded),
                      default=None)
        shifts = [(x, r) for r in compared_rows
                  if (x := shift_sigma(r)) is not None]
        shift, shift_at = (max(shifts, key=lambda t: t[0]) if shifts
                           else (None, None))
        out.append(Row(s, sorted(sigs), cells, excluded, row_max, kept,
                       compared, compared_rows, shift, shift_at))
    return out


def summarize(table_rows, rows, *, skip_shift=frozenset()) -> dict:
    """The summary figures of the published text, and the run's extremes.

    ``skip_shift``: series whose grid shift compares two different
    wavevectors (grid mode off the grid) and is left out of the largest one.
    """
    kept = sorted(p for t in table_rows for p in t.kept)
    comp = [p for t in table_rows for p in t.compared]

    def within(xs, k):
        return 100.0 * sum(1 for x in xs if x <= k) / len(xs) if xs else None

    shifts = [(t.shift, t) for t in table_rows
              if t.shift is not None and t.series not in skip_shift]
    shift, at = max(shifts, key=lambda x: x[0]) if shifts else (None, None)
    return {
        "evaluated": len(rows),
        "compared": len(comp),
        "kept": len(kept),
        "max_kept": kept[-1] if kept else None,
        # The published median is kept[n // 2], the upper median.  It is
        # the median for an odd count (the published 753); for an even count
        # (an --only subset) it is the upper of the two middle values.
        "median_kept": kept[len(kept) // 2] if kept else None,
        "within1_kept": within(kept, 1.0),
        "within2_kept": within(kept, 2.0),
        "within1_all": within(comp, 1.0),
        "within2_all": within(comp, 2.0),
        "flips": sum(1 for r in rows if r["sign_flip"]),
        "shift": shift,
        "shift_row": at,
        "shift_skipped": any(t.shift is not None and t.series in skip_shift
                             for t in table_rows),
    }


def compare_published(table_rows) -> dict:
    """This run's printed cells and row maxima against the published table.

    Every order position of a row is a cell, "-" (no reference value)
    included, so the full table has 15 x 13 = 195.  A cell is compared as
    printed (one decimal).  Returns the counts and the differing cells as
    (row, order or None for the maximum, this run, published).
    """
    n_cells = n_max = 0
    diffs = []
    for t in table_rows:
        pub = published_row(t.series)
        if pub is None:
            continue
        cells, rmax = pub
        for o in ORDERS:
            n_cells += 1
            own = f"{t.cells[o]:.1f}" if o in t.cells else "-"
            if own != cells.get(o, "-"):
                diffs.append((t, o, own, cells.get(o, "-")))
        n_max += 1
        own = "-" if t.row_max is None else f"{t.row_max:.1f}"
        if own != rmax:
            diffs.append((t, None, own, rmax))
    return {"cells": n_cells, "maxima": n_max, "diffs": diffs,
            "cell_diffs": sum(1 for d in diffs if d[1] is not None),
            "max_diffs": sum(1 for d in diffs if d[1] is None)}

# --------------------------------------------------------------------------
# Terminal output
# --------------------------------------------------------------------------

_ASCII = str.maketrans({
    "½": "1/2", "Γ": "Gamma", "Λ": "L", "ℤ": "Z",
    "²": "^2", "³": "^3", "σ": "sigma", "–": "-",
    "—": "-", "│": "|", "─": "-", "ᴬ": "A",
    "·": "*", "Δ": "d", "≤": "<=", "≥": ">=",
    "→": "->", "≈": "~", "−": "-", "†": "!", "…": "...",
})
_ANSI = re.compile(r"\033\[[0-9;]*m")

# Prose (legend, notes) is wrapped to this width or to the table, whichever
# is narrower, so it reads on a 100-column terminal.
NOTE_WIDTH = 100


def visible_width(text: str) -> int:
    """Widest line of ``text`` on screen, ANSI styling excluded."""
    return max((len(_ANSI.sub("", x)) for x in text.splitlines()), default=0)


class Style:
    """ANSI styling and a UTF-8-or-ASCII switch for everything printed."""

    def __init__(self, color: bool, stream=None):
        self.color = color
        stream = stream or sys.stdout
        try:
            "σ½Γ│ℤᴬ–…".encode(
                getattr(stream, "encoding", None) or "ascii")
            self.unicode = True
        except (UnicodeEncodeError, LookupError):
            self.unicode = False

    def t(self, text: str) -> str:
        return text if self.unicode else text.translate(_ASCII)

    def _sgr(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def bold(self, s):
        return self._sgr(s, "1")

    def dim(self, s):
        return self._sgr(s, "2")

    def head(self, s):
        return self._sgr(s, "1;36")

    def warn(self, s):
        return self._sgr(s, "1;31")

    def caution(self, s):
        return self._sgr(s, "33")

    def pull(self, s, value):
        """Colour a pull by size: > 3 red, > 2 yellow."""
        if value is None or not self.color:
            return s
        if value > 3:
            return self._sgr(s, "31")
        if value > 2:
            return self._sgr(s, "33")
        return s


def fmt_sigma(x: float) -> str:
    """The recurring Monte Carlo exponents as fractions, the rest as %g."""
    for num, den in ((1, 2), (2, 3), (1, 3), (4, 3), (5, 3), (7, 3), (8, 3)):
        if abs(abs(x) - num / den) < 2e-5:
            return f"{'-' if x < 0 else ''}{num}/{den}"
    return "%g" % x


def fmt_range(x: float) -> str:
    """An exponent in the sigma-range column (1/2 as 0.5, like the paper)."""
    return "0.5" if x == 0.5 else fmt_sigma(x)


def fmt_shift(x: float | None) -> str:
    """A grid shift in sigma_MC: two decimals from 0.01 up, else 1e-04."""
    if x is None:
        return "–"
    if x == 0.0:
        return "0"
    return f"{x:.2f}" if x >= 0.01 else f"{x:.0e}"


def fmt_grids(grids, control: bool) -> str:
    parts = []
    for d in sorted(grids):
        hi, lo = grids[d]
        parts.append(f"d={d}: n={hi}" + (f" (control {lo})" if control and lo
                                          else ""))
    return " · ".join(parts)


def _home(path: Path) -> str:
    """``path`` with the home directory written as ~."""
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def _library() -> tuple[str, str]:
    """(version and git revision, package directory) of the gzl that
    was imported -- not of whatever distribution happens to be installed."""
    pkg = Path(gzl.__file__).resolve().parent
    root = pkg.parent
    # The version of the code that runs is the one the imported package
    # states.  This used to be regexed out of pyproject.toml, which stopped
    # working the moment pyproject started reading it from here instead
    # (`dynamic = ["version"]`) -- and `gzl.__version__` was always the
    # better source for what this function promises, since it comes from the
    # package that was imported rather than from a file beside it.
    version = getattr(gzl, "__version__", None)
    if version is None:                                  # pragma: no cover
        try:
            version = importlib.metadata.version("gzl")
        except importlib.metadata.PackageNotFoundError:
            version = None
    # A source checkout (PYTHONPATH, editable install) has the project's
    # pyproject.toml beside the package.  The installed dist-info can be a
    # different, stale checkout.
    origin = "installed"
    try:
        if re.search(r'(?m)^name\s*=\s*"gzl"',
                     (root / "pyproject.toml").read_text()):
            origin = "source checkout"
    except OSError:
        pass
    text = ("gzl (version unknown)" if version is None
            else f"gzl {version} ({origin})")
    if (root / ".git").exists():
        try:
            rev = subprocess.run(["git", "-C", str(root), "rev-parse", "--short",
                                  "HEAD"], capture_output=True, text=True,
                                 timeout=5)
            dirty = subprocess.run(["git", "-C", str(root), "status",
                                    "--porcelain", "--untracked-files=no",
                                    "--", "gzl"],
                                   capture_output=True, text=True, timeout=5)
            if rev.returncode == 0:
                text += f" · git {rev.stdout.strip()}"
                if dirty.returncode == 0 and dirty.stdout.strip():
                    text += " (modified)"
        except (OSError, subprocess.SubprocessError):
            pass
    return text, _home(pkg)


def _host() -> str:
    text = f"{platform.system()} {platform.machine()} · {os.cpu_count()} CPUs"
    try:
        text += " · load %.1f / %.1f / %.1f" % os.getloadavg()
    except (OSError, AttributeError):
        pass
    return text


def render_table(table_rows, grids, st: Style, *, control: bool,
                 marked_shift=frozenset()) -> str:
    """The comparison table, laid out like the published one.

    Every order column is as wide as its widest entry plus one space, so a
    full run stays within 120 columns.
    """
    labels = [st.t(t.series.label) for t in table_rows]
    ranges = []
    for t in table_rows:
        if not t.sigmas:
            ranges.append("no data")
            continue
        lo, hi = fmt_range(t.sigmas[0]), fmt_range(t.sigmas[-1])
        span = lo if lo == hi else f"{lo}–{hi}"
        ranges.append(st.t(f"{span} ({len(t.sigmas)})"))
    tags = [st.t(t.series.tag) for t in table_rows]
    sig_head = st.t("σ range")
    shift_head = st.t("Δgrid")
    wl = max([len("Series") - 2] + [len(x) for x in labels])
    wr = max([len(sig_head)] + [len(x) for x in ranges])
    wt = max([len("Source")] + [len(x) for x in tags])
    bar = st.t("│")
    dash = st.t("–")

    def order_head(o):
        return st.t(f"{o}ᴬ") if o <= N_ANALYTIC else str(o)

    # (text, kind) per cell, before padding: padding goes inside the styling.
    texts = []
    for t in table_rows:
        row = {}
        for o in ORDERS:
            if o not in t.cells:
                row[o] = (dash, "dim")
            elif o in t.excluded:
                # analytic orders in parentheses, a dropped top order with †
                row[o] = ((f"({t.cells[o]:.1f})" if o <= N_ANALYTIC
                           else st.t(f"{t.cells[o]:.1f}†")), "dim")
            else:
                row[o] = (f"{t.cells[o]:.1f}", "pull")
        texts.append(row)
    wo = {o: 1 + max([len(order_head(o))] + [len(r[o][0]) for r in texts])
          for o in ORDERS}
    maxes = [dash if t.row_max is None else f"{t.row_max:.1f}"
             for t in table_rows]
    wm = 1 + max([len("max")] + [len(x) for x in maxes])
    shifts = [dash if (not control or t.shift is None) else fmt_shift(t.shift)
              for t in table_rows]
    ws = 1 + max([len(shift_head)] + [len(x) for x in shifts])

    head = (f"{'Series':<{wl + 4}}{sig_head:<{wr + 2}}{'Source':<{wt}}"
            + "".join(f"{order_head(o):>{wo[o]}}" for o in ORDERS
                      if o <= N_ANALYTIC)
            + f" {bar}"
            + "".join(f"{order_head(o):>{wo[o]}}" for o in ORDERS
                      if o > N_ANALYTIC)
            + f" {bar}{'max':>{wm}}{shift_head:>{ws}}")
    rule = st.t("─") * len(head)
    lines = [st.bold(head), rule]

    last = None
    for t, lab, rng, tag, row, rmax_txt, sh_txt in zip(
            table_rows, labels, ranges, tags, texts, maxes, shifts,
            strict=True):
        s = t.series
        if s.lattice != last:
            d = LATTICE_DIM[s.lattice]
            if last is not None:
                lines.append("")
            lines.append(st.head(st.t(f"{LATTICE_TITLE[s.lattice]}   "
                                      f"(d = {d}, n = {grids[d][0]})")))
            last = s.lattice
        cells = []
        for o in ORDERS:
            txt, kind = row[o]
            txt = f"{txt:>{wo[o]}}"
            cells.append(st.dim(txt) if kind == "dim"
                         else st.pull(txt, t.cells[o]))
            if o == N_ANALYTIC:
                cells.append(f" {bar}")
        rmax = (st.pull(st.bold(f"{rmax_txt:>{wm}}"), t.row_max)
                if t.row_max is not None else st.dim(f"{rmax_txt:>{wm}}"))
        shift = f"{sh_txt:>{ws}}"
        if control and t.shift is not None and t.shift >= SHIFT_FLAG:
            shift = st.caution(shift)
        if s in marked_shift:
            shift += "*"
        lines.append(f"  {lab:<{wl}}  {rng:<{wr}}  {tag:<{wt}}"
                     + "".join(cells) + f" {bar}" + rmax + shift)
    lines.append(rule)
    return "\n".join(lines)


def _wrap(items, st: Style, width: int) -> list[str]:
    """(key, text) pairs as a hanging-indent list wrapped to ``width``."""
    keys = [st.t(k) for k, _ in items]
    kw = max(len(k) for k in keys)
    out = []
    for key, (_, text) in zip(keys, items, strict=True):
        out.extend(textwrap.wrap(
            st.t(text), width=width, initial_indent=f"  {key:<{kw}}  ",
            subsequent_indent=" " * (kw + 4), break_on_hyphens=False,
            break_long_words=False))
    return out


def render_notes(st: Style, *, control: bool, marked_shift,
                 partial: bool = False, width: int = NOTE_WIDTH) -> str:
    """The legend under the table.  ``partial``: some dimensions were run
    without a control grid, so their rows show no Delta-grid."""
    items = [
        ("cell", "the largest pull |c_r − c_r^MC| / σ_MC over the series'"
                 " exponents σ ≥ 1/2 at order r."),
        ("–", "no reference value."),
        ("( )", "orders rᴬ ≤ 3, analytic: graph-zeta is exact there, so the"
                " pull measures the Monte Carlo error only.  Shown, but"
                " excluded from max and from the kept statistics."),
        ("†", "the top order of the Fey 2020 0qp tables F.4 and F.7,"
              " underconverged in the reference; excluded likewise."),
    ]
    if control:
        items.append((
            "Δgrid", "the row's largest |c(n) − c(n_control)| / σ_MC: how far a"
                     " coefficient moved from the control grid, in the units of"
                     f" the pull.  From {SHIFT_FLAG} on (highlighted) the move"
                     " reaches the printed resolution of a cell; when the"
                     " series converges fast in n it overstates the error"
                     " left at n."
                     + ("  – = no control grid for that dimension."
                        if partial else "")))
    else:
        items.append(("Δgrid", "not measured: no control grid."))
    if marked_shift:
        items.append((
            "*", "the wavevector is not a node of the grid and its nearest node"
                 " is read, so this Δgrid compares two different wavevectors;"
                 " the summary leaves it out."))
    items.append((
        "Sources", "Fey 2020 F.x = Fey 2020 thesis, Table F.x · Fey 2019 y ="
                   " Fey et al. PRL 2019, Table y (suppl.) · Langheld 2022 ="
                   " Langheld et al. 2022 (Zenodo) · Adelhardt p.c. ="
                   " Adelhardt, private communication."))
    return "\n".join(st.dim(x) for x in _wrap(items, st, width))


def _cell_name(t) -> str:
    return f"{t.series.name} ({t.series.tag})"


def render_summary(sm: dict, st: Style, *, published: bool, wall: float,
                   jobs: int, busy: float, peak: tuple | None,
                   check: dict | None) -> str:
    """The totals at the published precision, beside the published ones.

    Each figure of this run is rounded the way the paper prints it, so the
    two columns compare digit for digit; the unrounded value is in brackets
    after them.  A figure that differs from the published one is
    highlighted -- in a published setup only.  Longer context goes on an
    indented line of its own.
    """
    rows = []   # (label, this run, published, unrounded, highlight, note)

    def fig(label, key, unit="", exact=None):
        x = sm[key]
        pub = PUBLISHED.get(key)
        f = pub[1] if pub else ".0f"
        val = "–" if x is None else format(x, f) + unit
        ref = format(pub[0], pub[1]) + unit if pub else ""
        rows.append((label, val, ref,
                     f"[{format(x, exact)}]" if exact and x is not None else "",
                     published and pub is not None and val != ref, ""))

    fig("coefficients evaluated (all σ, all orders)", "evaluated")
    fig("coefficients compared (σ ≥ 1/2, tabulated)", "compared")
    fig("coefficients kept (not ( ) or †)", "kept")
    fig("max pull, kept", "max_kept", " σ_MC", ".4f")
    fig("median pull, kept", "median_kept", " σ_MC", ".4f")
    fig("within 1 σ_MC, kept", "within1_kept", " %", ".2f")
    fig("within 2 σ_MC, kept", "within2_kept", " %", ".2f")
    fig("within 1 σ_MC, all compared", "within1_all", " %", ".2f")
    fig("within 2 σ_MC, all compared", "within2_all", " %", ".2f")
    if check is not None:
        n_ok = check["cells"] - check["cell_diffs"]
        m_ok = check["maxima"] - check["max_diffs"]
        rows.append(("printed cells as published",
                     f"{n_ok} / {check['cells']}", "", "",
                     check["cell_diffs"] > 0, ""))
        rows.append(("printed row maxima as published",
                     f"{m_ok} / {check['maxima']}", "", "",
                     check["max_diffs"] > 0, ""))
    fig("order-1 sign flips of the reference", "flips")
    if sm["shift"] is None:
        rows.append(("largest move from the control grid", "–", "", "", False,
                     "not measured: no control grid"))
    else:
        t = sm["shift_row"]
        r = t.shift_at
        note = (f"at {_cell_name(t)}, σ = {fmt_sigma(r['sigma'])},"
                f" order {r['order']}, where the pull is {r['pull']:.2f}")
        if sm["shift_skipped"]:
            note += "; the off-grid rows (*) are left out"
        rows.append(("largest move from the control grid",
                     f"{fmt_shift(sm['shift'])} σ_MC", "", "",
                     sm["shift"] >= SHIFT_FLAG, note))
    if peak is not None:
        rows.append(("peak memory of one worker" if peak[1]
                     else "peak memory of this process",
                     f"{peak[0] / 1e9:.1f} GB", "", "", False,
                     f"reached while evaluating {peak[1]}" if peak[1] else ""))
    rows.append((f"wall time, {jobs} worker{'s' * (jobs != 1)}",
                 f"{wall:.1f} s", PUBLISHED_RUNTIME, "", False, ""))
    rows.append(("compute time summed over units", f"{busy:.1f} s", "", "",
                 False, ""))

    lw = max(len(st.t(r[0])) for r in rows) + 2
    vw = max(len(st.t(r[1])) for r in rows)
    pw = max([len("published")] + [len(st.t(r[2])) for r in rows
                                    if r[2] != PUBLISHED_RUNTIME])
    out = [st.bold(st.t(f"{'Summary':<{lw}}{'this run':>{vw}}   published"))]
    for label, val, ref, exact, flag, note in rows:
        v = f"{st.t(val):>{vw}}"
        line = f"  {st.t(label):<{lw - 2}}" + (st.caution(v) if flag else v)
        if ref or exact:
            line += "   " + st.dim(f"{st.t(ref):<{pw}}")
        if exact:
            line += "   " + st.dim(exact)
        out.append(line.rstrip())
        if note:
            out.extend(st.dim(x) for x in _wrap([("", note)], st, NOTE_WIDTH))
    if not published:
        out.append("")
        out.extend(st.dim(x) for x in _wrap(
            [("", "Not the published setup (--quick, --only, --grid-mode,"
                  " --snap-sigma, or grids other than the published ones), so"
                  " the published column is for orientation only.")],
            st, NOTE_WIDTH))
    return "\n".join(out)


def render_differences(check: dict, grids, st: Style) -> str:
    """The printed cells that differ from the published table and, when a
    d = 1 cell differs at a d = 1 grid other than CELL_GRIDS[1], the --grids
    value that reproduces the published d = 1 cells."""
    diffs = check["diffs"]
    if not diffs:
        return ""
    names = []
    for t, o, own, pub in diffs:
        where = f"order {o}" if o is not None else "row max"
        names.append((f"{_cell_name(t)}, {where}", t, o, own, pub))
    w = max(len(st.t(n[0])) for n in names)
    out = [st.bold("Printed cells that differ from the published table")]
    for name, t, o, own, pub in names:
        line = f"  {st.t(name):<{w}}   {own} here, {pub} published"
        if o is not None and o in t.cells:
            r = t.top_row(o)
            line += st.t(f"   (largest pull {r['pull']:.4f},"
                         f" σ = {fmt_sigma(r['sigma'])})")
        out.append(line)
    if (1 in grids and grids[1][0] != CELL_GRIDS[1]
            and any(LATTICE_DIM[t.series.lattice] == 1 for t, *_ in diffs)):
        out.extend(st.dim(x) for x in _wrap(
            [("", f"--grids '1:{CELL_GRIDS[1]},{CELL_GRIDS[1] // 2}'"
                  " reproduces every published d = 1 cell digit for"
                  " digit.")], st, NOTE_WIDTH))
    return "\n".join(out)


class Progress:
    """Feedback between the per-series lines.

    On a terminal: a status line, rewritten in place, with the number of
    exponents done and the elapsed time.  Otherwise (a log file, a pipe): a
    heartbeat line after every ``HEARTBEAT`` seconds without output.
    """

    HEARTBEAT = 60.0

    def __init__(self, total: int, t0: float, st: Style, stream=None):
        self.stream = stream or sys.stdout
        try:
            self.tty = self.stream.isatty()
        except (AttributeError, ValueError):
            self.tty = False
        self.total, self.t0, self.st = total, t0, st
        self.done = 0
        self.shown = 0                  # width of the status line on screen
        self.last = time.perf_counter()

    def _text(self) -> str:
        return self.st.t(f"  … {self.done}/{self.total} exponents done,"
                         f" {time.perf_counter() - self.t0:.0f} s")

    def status(self) -> None:
        if self.tty:
            text = self._text()
            self.stream.write("\r" + text.ljust(self.shown))
            self.stream.flush()
            self.shown = len(text)
        elif time.perf_counter() - self.last >= self.HEARTBEAT:
            self.line(self.st.dim(self._text()))

    def clear(self) -> None:
        if self.tty and self.shown:
            self.stream.write("\r" + " " * self.shown + "\r")
            self.stream.flush()
            self.shown = 0

    def line(self, text: str) -> None:
        self.clear()
        print(text, file=self.stream, flush=True)
        self.last = time.perf_counter()

# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

DESCRIPTION = """\
graph-zeta against Monte Carlo series of the long-range transverse-field
Ising model (arXiv:2609.18918, Sec. 8.3, Table 2).  The shipped pCUT corpora are
evaluated for 15 published series families -- chain, square, triangular and cubic
lattices; ground-state energy (0qp) and one-quasiparticle gap (1qp) at the zone
centre and boundary -- 123 exponents sigma in all, and every series coefficient is
compared with the Monte Carlo value for the same lattice, momentum, exponent and
order, in units of the Monte Carlo standard error.

The table shows, per series and order r, the largest pull |c_r - c_r^MC| / sigma_MC
over the exponents with sigma >= 1/2.  Orders r <= 3 are exact closed forms and
measure only the Monte Carlo error; they (in parentheses) and the underconverged
top order of two reference tables (marked with a dagger) are shown but excluded
from the row maximum.  A coarser control grid gives how far each coefficient
moved (Delta-grid, in sigma_MC), the summary sets the totals beside the published
ones, and the printed cells are checked against the published table.

With default options this is the published computation at the grids the paper's
caption states (n = 2048, 48, 10).  There one printed cell differs from the
published table (chain 0qp, order 11: 1.3 here, 1.2 published), and
--grids '1:512,256' reproduces every published cell.  --quick is a smoke run,
not the comparison.

Memory: at the published grids a worker process holds up to about 3 GB (measured
2.5-3.1 GB), so 6 workers need about 18 GB.  The default --jobs keeps within the
machine's physical memory; the summary reports the peak of the run."""


def _parse_grids(text: str) -> dict:
    grids = {}
    try:
        for tok in text.split():
            d, hl = tok.split(":")
            vals = [int(x) for x in hl.split(",")]
            if len(vals) not in (1, 2) or min(vals) < 2:
                raise ValueError
            grids[int(d)] = (vals[0], vals[1] if len(vals) == 2 else None)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected e.g. '1:2048,1024 2:48,32 3:10,8', got {text!r}") from None
    return grids


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="mc_comparison.py", description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=0, metavar="N",
                    help="worker processes (spawn context); default: the"
                         " number of work units, capped at min(CPUs, 8) and"
                         " at physical memory / 3.2 GB; 1 runs in this"
                         " process.  At the published grids each worker"
                         " holds up to about 3 GB")
    ap.add_argument("--grids", type=_parse_grids, default=None,
                    metavar="'D:N,NLO ...'",
                    help="reported and control grid per dimension; default"
                         " '1:2048,1024 2:48,32 3:10,8', the grids of the"
                         " paper's caption.  '1:512,256' reproduces every"
                         " published d = 1 cell digit for digit.  Without a"
                         " control grid (D:N) no grid shift is measured")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the control-grid pass (halves the work; no"
                         " grid shift is reported)")
    ap.add_argument("--grid-mode", action="store_true",
                    help="evaluate each 1qp series on the whole Brillouin"
                         " zone and read the wavevector off the grid, instead"
                         " of asking for that one wavevector")
    ap.add_argument("--only", action="append", default=[], metavar="SUBSTRING",
                    help="run only the series whose name and source (as"
                         " printed) contain SUBSTRING (case-sensitive;"
                         " repeatable), e.g. --only 'cubic 1qp k=0' or"
                         " --only 'chain 1qp k=0 Langheld'")
    ap.add_argument("--quick", action="store_true",
                    help="smoke preset, NOT the published comparison: one"
                         " exponent per series, orders <= %d, grids %s"
                         % (QUICK_ORDER_MAX, " ".join(
                             f"{d}:{h},{lo}" for d, (h, lo) in QUICK_GRIDS.items())))
    ap.add_argument("--snap-sigma", action="store_true",
                    help="evaluate rounded exponent labels (1.66667, 3.33333,"
                         " ...) at the exact rational they stand for; off by"
                         " default, as in the published table")
    ap.add_argument("--csv", metavar="PATH",
                    help="write the cell-level CSV: one row per evaluated"
                         " coefficient (all sigma, all orders)")
    ap.add_argument("--no-color", action="store_true",
                    help="plain output (also when NO_COLOR is set, or stdout"
                         " is not a terminal and FORCE_COLOR is unset)")
    return ap.parse_args(argv)


def select(only) -> list[int]:
    if not only:
        return list(range(len(SERIES)))
    # Case-sensitive: "1qp K" (the triangular K point) is not "1qp k=0".
    # "chain 1qp k=0 Langheld" picks one of the two chain k = 0 series.
    return [i for i, s in enumerate(SERIES)
            if any(k in h for k in only
                   for h in (f"{s.name} {s.ref}", f"{s.name} {s.tag}", s.config))]


def published_setup(grids, dims) -> str | None:
    """"caption" or "cells" when every reported grid is a published one."""
    if all(grids[d][0] == PUBLISHED_GRIDS[d][0] for d in dims):
        return "caption"
    if all(grids[d][0] == CELL_GRIDS[d] for d in dims):
        return "cells"
    return None


# Memory of one worker process at the published grids.  The heaviest single
# exponent (square 0qp, Adelhardt, sigma = 2) peaks at 2.2 GB in a fresh
# process, and the library's bounded caches (edge kernels, bridges, Epstein
# grids) stay resident across the exponents a worker serves: measured over a
# full run on 5 workers, every worker ended at 2.5-3.1 GB.  A fresh process
# per exponent (maxtasksperchild=1) caps a worker at 2.2 GB but made the
# same run 12% slower (340 s against 305 s), so workers are reused.  The
# default --jobs stays within physical memory.
WORKER_BYTES = 3.2e9


def _physical_memory() -> float | None:
    try:
        return float(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


def default_jobs(n_units: int, *, quick: bool) -> tuple[int, bool]:
    """(workers, whether memory rather than CPUs set the number)."""
    jobs = min(n_units, os.cpu_count() or 1, 8)
    mem = _physical_memory()
    if quick or not mem:
        return max(1, jobs), False
    by_mem = max(1, int(mem // WORKER_BYTES))
    return max(1, min(jobs, by_mem)), by_mem < jobs


def main(argv=None) -> int:
    args = parse_args(argv)
    color = (not args.no_color and not os.environ.get("NO_COLOR")
             and (sys.stdout.isatty() or bool(os.environ.get("FORCE_COLOR"))))
    st = Style(color)

    idx = select(args.only)
    if not idx:
        print("No series matches --only %s.  Available:" % " / ".join(args.only))
        for s in SERIES:
            print(f"  {st.t(s.name):<28} {s.ref}")
        return 2
    grids = dict(PUBLISHED_GRIDS)
    if args.quick:
        grids.update(QUICK_GRIDS)
    if args.grids:
        grids.update(args.grids)
    dims = {LATTICE_DIM[SERIES[i].lattice] for i in idx}
    if missing := sorted(dims - set(grids)):
        print(f"--grids gives no grid for d = {missing}")
        return 2

    try:
        points, snapped = plan(idx, grids, control=not args.no_control,
                               grid_mode=args.grid_mode, snap=args.snap_sigma,
                               quick=args.quick)
    except FileNotFoundError as exc:
        print(f"A reference file does not ship with this gzl:\n  {exc}\n"
              "Install a gzl that ships its data (gzl/data), see"
              " gzl/data/PROVENANCE.csv.")
        return 1
    points.sort(key=_cost, reverse=True)
    n_units = len(points)
    jobs, mem_capped = default_jobs(n_units, quick=args.quick)
    if args.jobs:
        jobs, mem_capped = args.jobs, False
    jobs = max(1, min(jobs, n_units))

    # Whether a grid shift is measured: --no-control, or a --grids entry
    # without a control grid, leaves it out.
    control = any(p.n_lo is not None for p in points)
    setup = None if args.quick else published_setup(grids, dims)
    # The published summary figures apply to the full default comparison;
    # the control pass does not change a compared number.
    published = (setup is not None and not args.only and not args.snap_sigma
                 and not args.grid_mode)
    passes = sum(2 if p.n_lo is not None else 1 for p in points)
    bad = off_grid(points)
    marked = {SERIES[i] for i, n in bad}

    # ---- header -------------------------------------------------------
    rule = st.t("─") * 78
    lib, pkg = _library()
    grid_tag = {"caption": "   [published grids]",
                "cells": "   [grids reproducing the published cells]"}.get(
                    published_setup(grids, dims), "")
    print(st.bold(st.t("graph-zeta vs Monte Carlo · LRTFIM series"
                       " coefficients")))
    print(rule)
    info = [
        ("compares", "c_r(σ, k) against the Monte Carlo series of the"
                     " long-range TFIM"),
        ("", "pull = |c_r − c_r^MC| / σ_MC; table cell = max over"
             " σ ≥ 1/2"),
        ("published", "Buchheit & Rupp, arXiv:2609.18918, Sec. 8.3, Table 2"),
        ("library", lib),
        ("package", pkg),
        ("grids", fmt_grids({d: grids[d] for d in sorted(dims)}, control)
         + grid_tag),
        ("momentum", "whole Brillouin zone, wavevector read off the grid"
         if args.grid_mode else "the reported wavevector only (single k)"),
        ("work", f"{len(idx)} series · {n_units} exponent{'s' * (n_units != 1)}"
                 f" · {passes} corpus pass{'es' * (passes != 1)}"
                 f" · {jobs} worker{'s' * (jobs != 1)}"
                 + (" (spawn)" if jobs > 1 else " (in process)")
                 + (f", fewer than CPUs: about {WORKER_BYTES / 1e9:.0f} GB"
                    " each" if mem_capped else "")),
        ("host", _host()),
    ]
    if args.quick:
        info.append(("preset", f"--quick: one exponent per series, orders"
                               f" ≤ {QUICK_ORDER_MAX}. NOT the published"
                               " comparison."))
    if snapped:
        info.append(("snapped", f"{len(snapped)} rounded exponent labels"
                                " evaluated at exact rationals (--snap-sigma)"))
    for key, val in info:
        print(f"  {key:<10} {st.t(val)}")
    for i, n in bad:
        s = SERIES[i]
        k = ", ".join(fmt_sigma(x) for x in s.momentum)
        print(st.warn(st.t(f"  WARNING    {s.name}: k = ({k}) is not a node of"
                           f" the n = {n} grid; the nearest node is read.")))
    print(rule)

    # ---- evaluation ---------------------------------------------------
    t0 = time.perf_counter()
    progress = Progress(n_units, t0, st)
    remaining = collections.Counter(p.index for p in points)
    done_rows = collections.defaultdict(list)
    flips = collections.Counter()
    busy = 0.0
    finished = 0
    peak = None                             # (bytes, the unit that set it)
    width = max(len(st.t(SERIES[i].name)) for i in idx)
    wtag = max(len(st.t(SERIES[i].tag)) for i in idx)

    def collect(result, point):
        nonlocal busy, finished, peak
        i, rows, fl, secs, rss = result
        done_rows[i].extend(rows)
        flips[i] += fl
        busy += secs
        progress.done += 1
        if rss is not None and (peak is None or rss > peak[0]):
            # A worker's high-water mark, and the exponent during which it
            # was reached (the caches of earlier exponents included).
            peak = (rss, st.t(f"{SERIES[i].name} ({SERIES[i].tag}),"
                              f" σ = {fmt_sigma(point.sigma)}")
                    if jobs > 1 else None)
        remaining[i] -= 1
        if remaining[i] == 0:
            finished += 1
            s = SERIES[i]
            progress.line(
                f"  [{finished:>2}/{len(idx)}] {time.perf_counter() - t0:7.1f} s"
                f"  {st.t(s.name):<{width}}  {st.t(s.tag):<{wtag}}"
                f"  {len(done_rows[i]):>4} cells, {flips[i]} order-1 sign"
                f" flip{'s' * (flips[i] != 1)}")

    progress.status()
    if jobs == 1:
        for p in points:
            collect(evaluate_point(p), p)
            progress.status()
    else:
        # spawn, not fork: numpy's threaded backend deadlocks a forked child
        # on macOS.  imap_unordered so each unit reports as it lands, and a
        # one-second timeout so the status line keeps ticking in between.
        with mp.get_context("spawn").Pool(jobs) as pool:
            # imap_unordered returns results, not their inputs: tag each one.
            results = pool.imap_unordered(_evaluate_tagged,
                                          list(enumerate(points)), chunksize=1)
            while True:
                try:
                    k, result = results.next(timeout=1.0)
                except mp.TimeoutError:
                    progress.status()
                    continue
                except StopIteration:
                    break
                collect(result, points[k])
                progress.status()
    progress.clear()
    wall = time.perf_counter() - t0

    # Deterministic order: series, sigma, order.  The pool returns the
    # units as they finish.
    rows = [r for i in idx
            for r in sorted(done_rows[i], key=lambda r: (r["sigma"], r["order"]))]

    # ---- table and summary ---------------------------------------------
    table_rows = build_rows(idx, rows)
    check = compare_published(table_rows) if setup is not None else None
    table = render_table(table_rows, grids, st, control=control,
                         marked_shift=marked)
    table_w = visible_width(table)
    print()
    try:
        cols = shutil.get_terminal_size((0, 0)).columns if sys.stdout.isatty() else 0
    except (OSError, ValueError):
        cols = 0
    if 0 < cols < table_w:
        print(st.dim(f"(the table is {table_w} columns wide and this terminal"
                     f" {cols}: widen it, or pipe the output through less -S)"))
    print(table)
    print(render_notes(st, control=control, marked_shift=marked,
                       partial=any(p.n_lo is None for p in points),
                       width=min(NOTE_WIDTH, max(78, table_w))))
    print()
    print(render_summary(summarize(table_rows, rows, skip_shift=marked), st,
                         published=published, wall=wall, jobs=jobs, busy=busy,
                         peak=peak, check=check))
    if check is not None and check["diffs"]:
        print()
        print(render_differences(check, grids, st))
    if snapped:
        print()
        print(st.bold("Snapped exponent labels (--snap-sigma)"))
        for s, label, before, after in snapped:
            print(st.t(f"  {s.name:<28} label {label:<9} σ {before!r} →"
                       f" {after!r}"))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} evaluated coefficients to {args.csv}")
    return 0


def _evaluate_tagged(item):
    """``evaluate_point`` for the pool, returned with its position."""
    k, p = item
    return k, evaluate_point(p)


if __name__ == "__main__":
    sys.exit(main())
