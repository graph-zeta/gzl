# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""2-D square-lattice TFIM gap-series benchmark against Monte Carlo.

Twin of ``test_cubic_benchmark.py`` for the 2-D square lattice
(``A = eye(2)``, k = 0): the ferromagnetic gap Δ_{sq,f}(σ, r) vs the
curated Monte-Carlo reference shipped in
``gzl/data/MC_patched/TFIM/square/`` (``σ = α − 2``): the order-11
k = 0 series of P. Adelhardt (private communication; see
``gzl/data/PROVENANCE.csv``), not Fey (2020) Table F.5, which stops
at order 9.

This case is feasible because dense blocks run through the planned
torus elimination, which pays the planner's exponent rather than the
dense bound.  The full 2-D 1qp corpus through order 11 contains the
treewidth-4 K₅-minor blocks, whose dense bound at d = 2 is ``n¹²``.

* **Fast tier (default)** — σ = 2, r ∈ {2, 3, 4} at ``n_points = 8``,
  ~2 % relative.  Seconds.
* **`@pytest.mark.slow`** — σ = 2, r ∈ {2 … 8} at ``n_points = 16``,
  asserting every order is within 3σ of the MC error bars (the
  full-O11 run at n=16 lands 10/10 within 2σ; r ≤ 8 keeps the slow
  test ≲ 1 min by skipping the 15 541-graph O11 step).
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from gzl import data_path
from gzl.series import compute_series_coefficients


NPZ_1QP = data_path("tfim_softcore_corpus_1qp.npz")
A2 = np.eye(2, dtype=float)


def load_mc_square(sigma: float) -> dict[int, tuple[float, float]]:
    """``{order: (value, error)}`` of the shipped k = 0 square-lattice gap
    series at ``sigma``.  The analytic order-1 placeholder (value and
    error both 0) is dropped."""
    path = data_path(
        "MC_patched/TFIM/square/"
        f"1qp_gap_series_2d_square_tfim_kx0_ky0_sigma{sigma:g}_order11.csv"
    )
    out: dict[int, tuple[float, float]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            v, e = float(row["prefactor"]), float(row["error"])
            if v == 0.0 and e == 0.0:
                continue
            out[int(row["order"])] = (v, e)
    return out


def _run(sigma, order_max, n_points):
    """1qp gap-series coefficients at k = 0 for ν = 2 + σ on d = 2."""
    coeff = compute_series_coefficients(
        NPZ_1QP, np.array([2.0 + sigma]), A2, n_points,
        momentum=np.zeros(2), order_max=order_max,
    )
    return {o: float(np.asarray(coeff[o])[0].real) for o in coeff}


class TestSquareBenchmarkFast:
    """σ = 2, r ∈ {2, 3, 4} at n = 8 — fast regression pin.

    σ = 2 is the smooth regime; even n = 8 lands well inside the MC
    error bars (the full n = 16 run is ~1e-5).  Tolerance 2 % leaves a
    wide margin while still catching a real evaluator regression.
    """

    SIGMA = 2.0
    ORDERS = (2, 3, 4)
    N_POINTS = 8
    REL_TOL = 0.02

    def test_table_F5_agreement(self):
        tn = _run(self.SIGMA, max(self.ORDERS), self.N_POINTS)
        mc = load_mc_square(self.SIGMA)
        failures: list[str] = []
        for r in self.ORDERS:
            mc_v, _ = mc[r]
            rel = abs(tn[r] - mc_v) / abs(mc_v)
            if rel > self.REL_TOL:
                failures.append(
                    f"(σ=2, r={r}): TN={tn[r]:+.6e}, MC={mc_v:+.6e}, "
                    f"rel={rel:.3e}"
                )
        assert not failures, (
            "2-D square MC reference mismatch (>2%):\n  "
            + "\n  ".join(failures)
        )


@pytest.mark.slow
class TestSquareBenchmarkSlow:
    """σ = 2, r ∈ {2 … 8} at n = 16 — every order within 3σ of MC."""

    SIGMA = 2.0
    ORDER_MAX = 8
    N_POINTS = 16

    def test_table_F5_within_mc_error(self):
        tn = _run(self.SIGMA, self.ORDER_MAX, self.N_POINTS)
        mc = load_mc_square(self.SIGMA)
        failures: list[str] = []
        for r in range(2, self.ORDER_MAX + 1):
            mc_v, mc_e = mc[r]
            sd = abs(tn[r] - mc_v) / mc_e if mc_e > 0 else float("inf")
            if sd > 3.0:
                failures.append(
                    f"(σ=2, r={r}): TN={tn[r]:+.8e}, MC={mc_v:+.8e}"
                    f"±{mc_e:.2e}, Δ/σ_MC={sd:.2f}"
                )
        assert not failures, (
            "2-D square gap series outside 3σ of MC:\n  "
            + "\n  ".join(failures)
        )
