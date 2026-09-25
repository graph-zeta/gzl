# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

"""Accuracy of circle-graph zeta vs ``zeta_circle`` reference.

A circle (cycle) graph with ``N`` nodes is built as

    g_circle = graph_convolve(graph_multiply_power(g0, N - 1), g0),

i.e. an ``(N - 1)``-edge serial chain closed by one parallel edge.
This script computes ``graph_zero(g_circle)`` for a sweep of node
counts, once per power-method (``_method='linear'`` and
``_method='binary'``), and compares both to the adaptive-integration
reference ``zeta_circle``.

Hypothesis: the binary path performs ``O(log N)`` multiplications
instead of ``O(N)``, so accumulated FFT noise should be smaller and
the relative error against ``zeta_circle`` should be lower.

Run with:  python benchmarks/accuracy_circle.py
"""

from __future__ import annotations

import numpy as np

from gzl import (
    graph_convolve,
    graph_multiply_power,
    graph_zero,
    make_epstein_graph,
    zeta_circle,
)


def _circle_zeta(g0, n_nodes: int, method: str) -> complex:
    chain = graph_multiply_power(g0, n_nodes - 1, _method=method)
    closed = graph_convolve(chain, g0)
    return graph_zero(closed)


def _run_sweep(label, nu, n_nodes_list, n_points, A, sigma_max=4.0):
    g0 = make_epstein_graph(nu, A, n_points)
    print()
    print("=" * 82)
    print(f"  {label}:  nu = {nu:.6f}, "
          f"n_points = {n_points}, sigma_max = {sigma_max}")
    print("=" * 82)
    print(f"{'N':>4} {'reference':>20} "
          f"{'rel err linear':>18} {'rel err binary':>18} "
          f"{'binary/linear':>14}")
    print("-" * 82)

    for n_nodes in n_nodes_list:
        ref = zeta_circle(np.full(n_nodes, nu), A)

        # Inline circle assembly with explicit sigma_max forwarded.
        chain_l = graph_multiply_power(g0, n_nodes - 1, sigma_max,
                                       _method="linear")
        chain_b = graph_multiply_power(g0, n_nodes - 1, sigma_max,
                                       _method="binary")
        from gzl import graph_convolve as _conv
        z_lin = graph_zero(_conv(chain_l, g0, sigma_max))
        z_bin = graph_zero(_conv(chain_b, g0, sigma_max))

        err_lin = abs(z_lin - ref) / abs(ref)
        err_bin = abs(z_bin - ref) / abs(ref)
        ratio = err_bin / err_lin if err_lin > 0 else float("nan")

        print(f"{n_nodes:>4} {ref.real:>20.12e} "
              f"{err_lin:>18.3e} {err_bin:>18.3e} "
              f"{ratio:>14.3f}")
    print("=" * 82)


def main() -> None:
    A = np.array([[1.0]])
    n_points = 500

    # (a) Stress regime: nu just above d=1, where errors are dominated by
    #     accumulated FFT noise from many graph_multiply calls.  Binary
    #     should win here because it does O(log N) instead of O(N) ops.
    _run_sweep(
        "Stress regime (nu just above d)",
        nu=1.0 + np.pi / 300,
        n_nodes_list=[4, 6, 8, 10, 12, 14, 16, 18, 20, 24],
        n_points=n_points, A=A,
    )

    # (b) Mild regime: nu = 1.5 + offset, well separated from d.  Both
    #     paths typically produce nearly identical accuracy here; the
    #     win is wall-clock, not error.
    _run_sweep(
        "Mild regime (nu = 1.5 + offset)",
        nu=1.5 + np.pi / 300,
        n_nodes_list=[4, 6, 8, 10, 12, 16, 24],
        n_points=n_points, A=A,
    )

    # (c) Improved-Fourier regime: sigma_max = 0 absorbs all
    #     singularities, leaving pure-Fourier multiplication.  Linear
    #     and binary paths are bit-equivalent under this setting.
    _run_sweep(
        "Improved Fourier (sigma_max = 0)",
        nu=1.5 + np.pi / 300,
        n_nodes_list=[4, 8, 16, 32],
        n_points=n_points, A=A, sigma_max=0.0,
    )

    print()
    print("Notes:")
    print("  ratio < 1  →  binary path is more accurate at this N.")
    print("  In the mild and improved-Fourier regimes the two paths")
    print("  agree to FP noise; in the stress regime, accumulated FFT")
    print("  rounding makes the binary path significantly better at")
    print("  moderate N (the long absolute scales beyond ~1e15 are at")
    print("  the boundary of double-precision and become unreliable).")


if __name__ == "__main__":
    main()
