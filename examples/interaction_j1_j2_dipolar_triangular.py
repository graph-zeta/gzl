# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Nearest and next-nearest neighbour couplings with a dipolar tail on the
triangular lattice.

Two sites at distance r are coupled by J1 for r = 1, by J2 for r = sqrt 3,
and by Jd / r^3 otherwise.  An Interaction describes such a coupling, and
gzl takes it wherever it takes a power law.

Run with:  python examples/interaction_j1_j2_dipolar_triangular.py
"""

import numpy as np

from gzl import Interaction, compute_series_coefficients, evaluate_graph

J1, J2, Jd = 0.5, 0.2, 0.1
K = Interaction.from_shells("triangular", {1.0: J1, 3**0.5: J2}, b=[Jd], nu=[3.0])

# A single edge from s = 0 to t = 1 is the Fourier transform of the
# coupling, which the Interaction also computes directly.
n = 24
grid = evaluate_graph([(0, 1)], K, "triangular", terminal=1, n_points=n)
k = np.stack(np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing="ij"), axis=-1)
fourier = K.lattice_sum("triangular", k.reshape(-1, 2)).reshape(n, n)
assert np.abs(grid - fourier).max() < 1e-12 * np.abs(fourier).max()

# The one-quasiparticle gap series of the transverse-field Ising model
# with this coupling, at the centre Gamma and the corner K of the
# Brillouin zone.  At first order the gap is lowered by the Fourier
# transform of the coupling.
for name, momentum in (("Gamma", [0, 0]), ("K", [1 / 3, -1 / 3])):
    c = compute_series_coefficients("tfim1qp", K, "triangular", 16,
                                    order_max=6, momentum=momentum)
    print(f"{name:5s} " + "  ".join(f"{float(c[r]):+.4f}" for r in sorted(c)))
    assert abs(c[1] / -K.lattice_sum("triangular", momentum) - 1) < 1e-12
