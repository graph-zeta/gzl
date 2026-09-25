# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Regression tests for the topology-first front-end against the
TFIM-softcore Monte-Carlo reference data.

These tests aggregate the per-graph contributions of `evaluate_graph`
over the consolidated 0qp and 1qp graph corpora (via
`compute_series_coefficients`, which delegates to `evaluate_graph`),
and assert agreement with the published MC chain reference data shipped
in ``gzl/data/MC_patched/TFIM/chain/``.

* 0qp ground-state-energy series, orders 1–10, σ ∈ {0.5, 1.0, 2.0,
  3.0}.
* 1qp gap-coefficient series at external momentum k = 0, orders 1–8,
  same σ set.

Tolerance: ``|TN − MC| ≤ max(10·MC_err, 1e-5·|MC|, 1e-12)`` —
generous enough to absorb statistical MC noise (MC error bars are
1-σ Gaussian estimates; 10σ is roughly the headroom needed for
occasional under-estimated errors at large σ where MC stats are
small) plus the FFT-truncation residual at ``n_points = 128``,
tight enough to catch a real regression.

Each parametrised σ runs the corpus through order 10 (0qp) or 8
(1qp) in under a second, so both classes run in the default suite.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from gzl import data_path
from gzl.series import compute_series_coefficients


CORPUS_NPZ_0QP = data_path("tfim_softcore_corpus_0qp.npz")
CORPUS_NPZ_1QP = data_path("tfim_softcore_corpus_1qp.npz")

A1D = np.array([[1.0]])
N_POINTS = 128
SIGMAS = (0.5, 1.0, 2.0, 3.0)


def _mc_file(lattice: str, name: str) -> Path:
    """A Monte Carlo reference shipped in ``gzl/data/MC_patched/TFIM``."""
    return data_path(f"MC_patched/TFIM/{lattice}/{name}")


# ---------------------------------------------------------------------------
# MC loaders
# ---------------------------------------------------------------------------

def _read_mc_csv(path: Path) -> dict[int, tuple[float, float]]:
    """Read an MC CSV with ``order, prefactor, error, ...`` columns
    into ``{order: (value, statistical_error)}``."""
    out: dict[int, tuple[float, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["order"])] = (
                float(row["prefactor"]),
                float(row["error"]),
            )
    return out


def _mc_path_0qp(sigma: float) -> Path:
    return _mc_file("chain", f"gs_energy_series_1d_chain_tfim_sigma{sigma:.1f}_order13.csv")


def _mc_path_1qp_k0(sigma: float) -> Path:
    return _mc_file("chain", f"1qp_gap_series_1d_chain_tfim_k0_sigma{sigma:.1f}_order11.csv")


def _tolerance(mc_val: float, mc_err: float) -> float:
    """Acceptance band: 10 × MC stat-error, with a relative floor at
    ``1e-5·|mc_val|`` (handles occasional MC under-estimated errors
    at large σ where stats are tight; tight enough to spot a real
    regression in either direction) and an absolute floor at
    ``1e-12`` (FP noise for orders where MC error is exact zero)."""
    return max(10.0 * mc_err, 1e-5 * abs(mc_val), 1e-12)


# ---------------------------------------------------------------------------
# 0qp ground-state-energy series — orders 1..10
# ---------------------------------------------------------------------------

class TestZeroQpRegressionVsMC:
    """Aggregate `evaluate_graph` over the 0qp corpus per σ-value and
    pin every order's coefficient against the MC reference within
    the statistical band (or within the FP-noise floor for orders
    where MC error is exactly zero)."""

    @pytest.mark.parametrize("sigma", SIGMAS)
    def test_orders_1_through_10(self, sigma: float) -> None:
        nu = float(sigma) + 1.0       # d = 1, so ν = d + σ
        mc_path = _mc_path_0qp(sigma)
        mc = _read_mc_csv(mc_path)

        result = compute_series_coefficients(
            CORPUS_NPZ_0QP,
            nu=nu,
            A=A1D,
            n_points=N_POINTS,
            order_max=10,
        )
        # Every order present in the corpus AND in the MC reference
        # must agree within the statistical band.
        compared = 0
        for order, val_arr in sorted(result.items()):
            if order not in mc or order < 2:
                # Order 0 / 1 are trivial / zero by construction; skip.
                continue
            mc_val, mc_err = mc[order]
            # Skip placeholder MC rows (the MC files carry 0 ± 0 at
            # some low orders for small σ where no value was
            # computed).
            if mc_val == 0.0 and mc_err == 0.0:
                continue
            tn_val = float(np.asarray(val_arr).reshape(-1)[0].real)
            tol = _tolerance(mc_val, mc_err)
            assert abs(tn_val - mc_val) < tol, (
                f"0qp σ={sigma} order={order}: "
                f"TN={tn_val:.10e}, MC={mc_val:.10e}±{mc_err:.2e}, "
                f"|diff|={abs(tn_val - mc_val):.3e} > tol={tol:.3e}"
            )
            compared += 1
        assert compared >= 5, (
            f"σ={sigma}: only {compared} orders compared (expected ≥ 5)."
        )


# ---------------------------------------------------------------------------
# 1qp gap-series coefficient at k = 0 — orders 1..8
# ---------------------------------------------------------------------------

class TestOneQpZeroMomentumRegressionVsMC:
    """Aggregate `evaluate_graph` over the 1qp corpus at zero external
    momentum (k = 0) per σ-value and pin every order's coefficient
    against the patched MC gap-series reference."""

    @pytest.mark.parametrize("sigma", SIGMAS)
    def test_orders_1_through_8(self, sigma: float) -> None:
        nu = float(sigma) + 1.0       # d = 1
        mc_path = _mc_path_1qp_k0(sigma)
        mc = _read_mc_csv(mc_path)

        result = compute_series_coefficients(
            CORPUS_NPZ_1QP,
            nu=nu,
            A=A1D,
            n_points=N_POINTS,
            order_max=8,
            # On a 1qp corpus the default is now the full BZ grid;
            # pin to scalar at k = 0 for this k=0-vs-MC test.
            momentum=np.zeros(1),
        )
        compared = 0
        for order, val_arr in sorted(result.items()):
            if order not in mc or order < 2:
                continue
            mc_val, mc_err = mc[order]
            # The MC files carry 0 ± 0 at orders ≤ 2 for small σ as a
            # placeholder (no MC value computed in the long-range
            # regime).  Skip placeholder rows.
            if mc_val == 0.0 and mc_err == 0.0:
                continue
            tn_val = float(np.asarray(val_arr).reshape(-1)[0].real)
            tol = _tolerance(mc_val, mc_err)
            assert abs(tn_val - mc_val) < tol, (
                f"1qp k=0 σ={sigma} order={order}: "
                f"TN={tn_val:.10e}, MC={mc_val:.10e}±{mc_err:.2e}, "
                f"|diff|={abs(tn_val - mc_val):.3e} > tol={tol:.3e}"
            )
            compared += 1
        assert compared >= 5, (
            f"σ={sigma}: only {compared} orders compared (expected ≥ 5)."
        )
