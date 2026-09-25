# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

"""Shift-free precision sweep of the cycle-graph zeta against ``zeta_circle``.

Companion to ``accuracy_circle.py``.  That script avoids the
``Gamma((d - nu)/2)`` poles of the multiplication prefactor by adding
a small irrational offset (``+ pi/300``) to the per-edge exponent so
that the integer-shift values ``nu = d + 2k`` (k = 1, 2, 3, ...) are
never exactly hit.  The new ``compress_singularities=True`` option of
:func:`graph_compress` (auto-engaged at the entry of
:func:`graph_multiply`) absorbs those exponents into the Fourier part
``aMat`` before the prefactor sees them, so the offset is no longer
needed: any ``nu > d`` works directly.

This script sweeps ``nu`` from ``d + 0.1`` to ``d + 3.0`` in steps of
``0.1`` (passing through ``nu = d + 2 = 3.0`` exactly in d=1, the
former pole), for two cycle lengths (N = 6 and N = 13) and a ladder
of grid sizes (n_points in {64, 128, 256, 512}), reporting the
relative error vs. the adaptive-integration ``zeta_circle`` reference.

Run with:  python benchmarks/accuracy_circle_shift_free.py

For a finer sweep (steps of 0.01), pass ``--fine``.  This is much
slower (~30 nu * 4 grids * 2 chains * full multiply chain).
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from gzl import (
    graph_convolve,
    graph_multiply_power,
    graph_zero,
    make_epstein_graph,
    zeta_circle,
)


def _circle_value(nu: float, n_nodes: int, n_points: int, A: np.ndarray,
                  sigma_max: float = 4.0) -> float:
    """Evaluate the cycle-graph zeta at zero momentum for given (nu, N, n_points)."""
    g0 = make_epstein_graph(nu, A, n_points)
    chain = graph_multiply_power(g0, n_nodes - 1, sigma_max)
    closed = graph_convolve(chain, g0, sigma_max)
    return float(graph_zero(closed))


def _ref_value(nu: float, n_nodes: int, A: np.ndarray) -> float:
    return float(zeta_circle(np.full(n_nodes, nu), A))


def _sweep(nu_grid, n_nodes_list, n_points_list, A: np.ndarray,
           sigma_max: float = 4.0):
    print()
    print("=" * 110)
    print(f"  Shift-free precision sweep:  d = {A.shape[0]},  "
          f"sigma_max = {sigma_max}")
    print(f"  nu grid: {nu_grid[0]:.2f} ... {nu_grid[-1]:.2f}  "
          f"(step {nu_grid[1]-nu_grid[0]:.2f}),  "
          f"chains N in {n_nodes_list},  n_points in {n_points_list}")
    print("=" * 110)

    header = f"{'nu':>6}  {'reference':>16}  "
    for N in n_nodes_list:
        for n_pts in n_points_list:
            header += f"  N={N:<2d} n={n_pts:<3d}"
    print(header)
    print("-" * len(header))

    t0 = time.perf_counter()
    for nu in nu_grid:
        # zeta_circle reference is the same for all n_points; compute once per
        # (nu, N) pair below.
        line = f"{nu:>6.2f}"
        first_ref = None
        for N in n_nodes_list:
            ref = _ref_value(nu, N, A)
            if first_ref is None:
                line += f"  {ref:>16.8e}"
                first_ref = ref
            for n_pts in n_points_list:
                try:
                    val = _circle_value(nu, N, n_pts, A, sigma_max)
                    rel = abs(val - ref) / abs(ref)
                    line += f"  {rel:>9.2e}"
                except Exception as e:
                    line += f"  {'err':>9}"
        print(line)
    print("=" * 110)
    print(f"sweep wall: {time.perf_counter() - t0:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fine",
        action="store_true",
        help="Use a finer nu-grid (step 0.01 instead of 0.1).",
    )
    args = parser.parse_args()

    A = np.array([[1.0]])
    d = A.shape[0]

    # nu grid: d + 0.1 ... d + 3.0
    step = 0.01 if args.fine else 0.1
    nu_grid = np.round(np.arange(d + 0.1, d + 3.0 + 0.5 * step, step), 6)

    n_nodes_list = [6, 13]
    n_points_list = [64, 128, 256, 512]

    _sweep(nu_grid, n_nodes_list, n_points_list, A)

    # Spotlight: integer-shift exponents (nu = d + 2 = 3.0 in d=1) used to
    # require an irrational offset.  Now they evaluate directly.  Show the
    # rel error at exactly nu = d + 2 separately.
    print()
    print("Integer-shift spotlight (no offset):")
    print(f"  {'nu':>6}  {'N':>3}  {'n_points':>9}  "
          f"{'reference':>16}  {'value':>16}  {'rel err':>10}")
    for nu in [d + 2.0]:
        for N in [6, 13]:
            for n_pts in [64, 128, 256, 512]:
                ref = _ref_value(nu, N, A)
                val = _circle_value(nu, N, n_pts, A)
                rel = abs(val - ref) / abs(ref)
                print(f"  {nu:>6.2f}  {N:>3d}  {n_pts:>9d}  "
                      f"{ref:>16.8e}  {val:>16.8e}  {rel:>10.2e}")


if __name__ == "__main__":
    main()
