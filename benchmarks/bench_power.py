# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

"""Wall-clock benchmark for graph_multiply_power / graph_convolve_power.

Compares the new ``_method='binary'`` exponentiation-by-squaring path
against the legacy ``_method='linear'`` loop for a range of exponents.

Run with:  python benchmarks/bench_power.py
"""

from __future__ import annotations

import time

import numpy as np

from gzl import (
    graph_convolve_power,
    graph_multiply_power,
    make_epstein_graph,
)


def _bench(fn, *args, repeat=3, **kwargs):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    A = np.array([[1.0]])
    n_points = 500
    nu_mul = 2.5 + np.pi / 300   # > d so multiplications stay singular
    nu_conv = 1.5 + np.pi / 300  # mild singularity for convolution chain

    g_mul = make_epstein_graph(nu_mul, A, n_points)
    g_conv = make_epstein_graph(nu_conv, A, n_points)

    exponents = [2, 4, 8, 16, 32, 64]

    print()
    print("=" * 70)
    print(f"  Benchmark: graph_*_power, n_points = {n_points}")
    print("=" * 70)
    print(f"{'op':<20} {'n_exp':>6} {'linear (s)':>14} "
          f"{'binary (s)':>14} {'speedup':>9}")
    print("-" * 70)

    for n in exponents:
        t_lin = _bench(graph_multiply_power, g_mul, n, _method="linear")
        t_bin = _bench(graph_multiply_power, g_mul, n, _method="binary")
        print(f"{'graph_multiply_power':<20} {n:>6} "
              f"{t_lin:>14.4f} {t_bin:>14.4f} {t_lin / t_bin:>8.2f}x")

    print()
    for n in exponents:
        t_lin = _bench(graph_convolve_power, g_conv, n, _method="linear")
        t_bin = _bench(graph_convolve_power, g_conv, n, _method="binary")
        print(f"{'graph_convolve_power':<20} {n:>6} "
              f"{t_lin:>14.4f} {t_bin:>14.4f} {t_lin / t_bin:>8.2f}x")

    print("=" * 70)


if __name__ == "__main__":
    main()
