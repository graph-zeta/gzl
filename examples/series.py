# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The one-quasiparticle gap series of the long-range transverse-field
Ising model on the cubic lattice, compared with Monte Carlo.

The corpus "tfim1qp" ships with gzl, and so do the Monte Carlo values of
S. Fey, PhD thesis, FAU Erlangen-Nürnberg (2020), Table F.10, for the
same series at nu = 4 and k = 0.

Run with:  python examples/series.py
"""

import numpy as np

from gzl import compute_series_coefficients, data_path, evaluate_corpus

nu, n_points = 4.0, 8
c = compute_series_coefficients("tfim1qp", nu, "cubic", n_points,
                                order_max=9, momentum=[0, 0, 0])

mc = np.loadtxt(data_path("MC_patched/TFIM/cubic/"
                          "TFIM_lr_3d_cube__MC__list_of_prefactors__k0__alpha_4.dat"))
print("order   gzl            Monte Carlo")
for r, value, error in mc:
    r = int(r)
    print(f"{r:5d}   {float(c[r]): .6e}   {value: .6e} +- {error:.1e}")
    assert abs(c[r] - value) < 2 * error

# The same pass graph by graph.  Each record holds the graph zeta value
# of one graph and its contribution to the coefficient of its order.
records = evaluate_corpus("tfim1qp", nu, "cubic", n_points,
                          order_max=9, momentum=[0, 0, 0])
for r in c:
    total = sum(rec["contribution"] for rec in records if rec["order"] == r)
    assert abs(total / c[r] - 1) < 1e-12
print(f"{len(records)} graphs")
