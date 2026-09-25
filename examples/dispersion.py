# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The one-quasiparticle dispersion of the long-range transverse-field
Ising chain.

One pass over the corpus gives the series coefficients c_r(k) on the
whole Brillouin-zone grid.  To order 8 the dispersion is

    omega(k, lambda) = 1 + sum_r c_r(k) lambda^r.

Run with:  python examples/dispersion.py
"""

import numpy as np

from gzl import compute_series_coefficients

nu, n_points = 3.0, 64
c = compute_series_coefficients("tfim1qp", nu, "chain", n_points, order_max=8)
k = np.arange(n_points) / n_points          # in units of 2 pi

for lam in (0.1, -0.1):
    omega = 1 + sum(c[r] * lam**r for r in c)
    k_min = k[np.argmin(omega)]
    print(f"lambda = {lam:+.1f}: smallest gap {omega.min():.4f} at k = {k_min}")
    # A ferromagnetic coupling, lambda > 0, has the smallest gap at k = 0,
    # an antiferromagnetic one at k = 1/2, that is k = pi.
    assert k_min == (0.0 if lam > 0 else 0.5)
