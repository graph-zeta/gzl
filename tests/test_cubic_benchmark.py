# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""3-D simple-cubic-lattice TFIM gap-series benchmark against Fey 2020.

Validates :func:`gzl.series.compute_series_coefficients` at
``A = eye(3)`` on the 1qp TFIM-softcore corpus at k = 0 — i.e. the
ferromagnetic gap Δ_{cub,f}(α, r) — against the Monte-Carlo reference
values published in Sebastian Fey's 2020 dissertation, Appendix F,
Table F.10 (p.162), read from the shipped files in
``gzl/data/MC_patched/TFIM/cubic/``.

A single fast tier — α ∈ {5, 6}, r ∈ {1, 2, 3} at ``n_points = 8``,
agreement to ~1% relative, ~1 min on a 2024-era laptop.  (An earlier
``@pytest.mark.slow`` n=16 sweep over α ∈ {3.5 … 10} was removed: at
d = 3 it ran 8–15 h, far outside any CI/dev budget.  The comparison at
the published grids is ``benchmarks/mc_comparison.py``.)
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl import data_path
from gzl.series import compute_series_coefficients


NPZ_1QP = data_path("tfim_softcore_corpus_1qp.npz")
A3 = np.eye(3, dtype=float)


def _fey_f10(alpha: float) -> dict[int, tuple[float, float]]:
    """``{order: (value, error)}`` of Fey (2020) Table F.10 at ``alpha``,
    from the shipped file (cell-verified against the print, see
    ``gzl/data/PROVENANCE.csv``)."""
    rows = np.loadtxt(data_path(
        "MC_patched/TFIM/cubic/"
        f"TFIM_lr_3d_cube__MC__list_of_prefactors__k0__alpha_{alpha:g}.dat"
    ))
    return {int(o): (v, e) for o, v, e in rows}


def _run(alpha_list, order_max, n_points, *, fast_cycles=False,
         richardson=False) -> dict[int, np.ndarray]:
    """Compute the 1qp gap-series coefficients at k = 0 for the given α grid."""
    return compute_series_coefficients(
        NPZ_1QP, np.asarray(alpha_list, dtype=float), A3, n_points,
        momentum=np.zeros(3),
        order_max=order_max,
        fast_cycles=fast_cycles,
    )


# ---------------------------------------------------------------------------
# Default tier: fast subset, ~1 min wall-clock
# ---------------------------------------------------------------------------

class TestCubicBenchmarkFast:
    """Default fast subset: α ∈ {5, 6} × r ∈ {1, 2, 3} at n = 8.

    At n_points = 8 the FFT discretisation residual is ~1/8^(2-σ) so at
    σ = 2 (α = 5) we're at ~1/64 ≈ 1.5%; observed residuals are
    well below 1% on this α ≥ 5 / r ≤ 3 corner.  Tolerance is 1% to
    leave ~10× safety margin while still catching evaluator
    regressions that would push the error to ~10%.
    """

    ALPHAS = (5.0, 6.0)
    ORDERS = (1, 2, 3)
    N_POINTS = 8
    REL_TOL = 0.01

    def test_table_F10_agreement(self):
        coefficients = _run(self.ALPHAS, max(self.ORDERS), self.N_POINTS)
        failures: list[str] = []
        for j, alpha in enumerate(self.ALPHAS):
            mc_at_alpha = _fey_f10(alpha)
            for r in self.ORDERS:
                tn = float(np.asarray(coefficients[r])[j].real)
                mc_v, mc_e = mc_at_alpha[r]
                rel = abs(tn - mc_v) / abs(mc_v)
                if rel > self.REL_TOL:
                    failures.append(
                        f"(α={alpha}, r={r}): TN={tn:+.6e}, "
                        f"MC={mc_v:+.6e}, rel={rel:.4e}"
                    )
        assert not failures, (
            "Fey Table F.10 mismatch (>1%):\n  "
            + "\n  ".join(failures)
        )
