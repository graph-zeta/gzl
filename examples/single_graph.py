# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""One graph zeta function, as a number and over the Brillouin zone.

The graph is a triangle 0-1-2 with a pendant edge 2-3 on the integer
chain, with the power law |x|^-3 on every edge.

Run with:  python examples/single_graph.py
"""

from scipy.special import zeta

from gzl import evaluate_graph, zeta_circle

edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
nu = 3.0

# Without terminals the value is a number.  It is the product of the
# triangle and the edge, which are known in closed form: the triangle is
# a circle zeta function, and one edge on the chain is 2 zeta(nu).
value = evaluate_graph(edges, nu, "chain")
exact = zeta_circle([nu, nu, nu], "chain") * 2 * zeta(nu)
print(f"vacuum value         {value:.12f}")
print(f"closed form          {exact:.12f}")
assert abs(value / exact - 1) < 1e-12

# With the terminals s = 0 and t = 3 the value depends on the momentum k,
# in units of the reciprocal lattice vector.  It is evaluated at one k,
# or on the whole Brillouin zone when the momentum is left out.
n = 128
at_k = evaluate_graph(edges, nu, "chain", source=0, terminal=3,
                      momentum=0.25, n_points=n)
grid = evaluate_graph(edges, nu, "chain", source=0, terminal=3, n_points=n)
print(f"k = 1/4              {at_k:.12f}")
print(f"grid point k = 1/4   {grid[n // 4]:.12f}")
assert abs(grid[n // 4] / at_k - 1) < 1e-12

# At k = 0 the momentum drops out, and the grid gives the vacuum value
# up to its truncation.
print(f"grid point k = 0     {grid[0]:.12f}")
assert abs(grid[0] / value - 1) < 1e-7
