# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""The finite-k bridge grid evaluates one node per inversion orbit.

``zeta_nu(k) = sum_x |x|^-nu exp(-2 pi i k.x)`` is even in ``k`` on any
Bravais lattice, and the unsigned node grid ``j in [0, n)^d`` is closed
under ``j -> -j mod n`` at every ``n``, so the on-spine bridge branch of
``_block_at_finite_k`` calls epsteinlib once per orbit and copies the
partner.  epsteinlib is scalar-only, and these grids were 65% of the
d = 2 order-9 1qp pass at ``n_points = 512`` (1.31e6 calls, ~15 us each).

What this pins, measured over every census cell at both parities:

* the evaluated node is computed EXACTLY as the per-node loop did
  (same ``j / n`` floats, same call) -- bit-identical;
* a copied partner is the same closed form at the mirrored argument;
  ``zeta(k) == zeta(-k)`` exactly, and epsteinlib returns the two to
  round-off: ``|zeta(k) - zeta(-k)| <= 1.1e-14`` absolute, at most
  4.1 ulp of ``max|zeta|`` on the grid -- reproduced here with
  epsteinlib alone, no gzl code in between.  A RELATIVE
  deviation is large only where zeta itself nearly vanishes (5e-12 at a
  node with ``|zeta| = 1.6e-3`` is that same 8e-15), which is why the
  bound below is in units of the grid's scale;
* the single-k branch is untouched.
"""
from __future__ import annotations

import numpy as np
import pytest
from epsteinlib import epstein_zeta

from gzl import evaluate_graph

BRIDGE = np.array([[0, 1]])
CELLS = {
    "unit/d1": [[1.0]],
    "scaled/d1": [[1.7]],
    "square/d2": [[1.0, 0.0], [0.0, 1.0]],
    "triangular/d2": [[1.0, 0.5], [0.0, 0.8660254037844386]],
    "sheared/d2": [[1.0, 0.37], [0.0, 1.13]],
    "cubic/d3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "tetragonal/d3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.3]],
    "sheared/d3": [[1.0, 0.3, 0.0], [0.0, 1.1, 0.2], [0.0, 0.0, 0.9]],
}
EPS = np.finfo(float).eps
# Measured 4.1 ulp of the grid's scale (tetragonal d = 3, n = 6, nu = 4 and
# triangular d = 2, n = 7, nu = 4 are the worst); the margin is 2x.
SCALE_ULP = 8.0


def _direct_grid(nu, A, n):
    """The per-node loop the branch used to run, verbatim."""
    d = A.shape[0]
    Astar = np.linalg.inv(A.T)
    zero = np.zeros(d)
    axes = [np.arange(n, dtype=float) / n for _ in range(d)]
    kk = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d)
    out = np.empty(kk.shape[0])
    for i in range(kk.shape[0]):
        out[i] = float(epstein_zeta(nu, A, zero, Astar @ kk[i]).real)
    return out.reshape((n,) * d)


def _orbit_masks(n, d):
    j = np.stack(np.meshgrid(*[np.arange(n)] * d, indexing="ij"),
                 axis=-1).reshape(-1, d)
    mirror = np.ravel_multi_index(tuple(((-j) % n).T), (n,) * d)
    flat = np.arange(j.shape[0])
    return flat <= mirror, mirror


def _parities(d):
    return (7, 8) if d <= 2 else (5, 6)


@pytest.mark.parametrize("name", sorted(CELLS))
@pytest.mark.parametrize("nu_off", [1.5, 0.5])
def test_representatives_are_bit_identical_and_partners_within_round_off(
        name, nu_off):
    A = np.array(CELLS[name], dtype=float)
    d = A.shape[0]
    nu = float(d) + nu_off
    for n in _parities(d):
        new = np.asarray(evaluate_graph(BRIDGE, nu, A, source=0, terminal=1,
                                        n_points=n)).ravel()
        old = _direct_grid(nu, A, n).ravel()
        reps, mirror = _orbit_masks(n, d)
        assert np.array_equal(new[reps], old[reps])
        assert np.array_equal(new[mirror[reps]], new[reps])   # copies
        scale = np.abs(old).max()
        assert np.abs(new - old).max() <= SCALE_ULP * EPS * scale


def test_the_partner_deviation_is_epsteinlibs_own_round_off():
    # zeta(k) and zeta(-k) from epsteinlib alone, no gzl code:
    # the copied-node deviation above is bounded by THIS.  On this cell
    # the two are NOT bit-identical (so the bound is doing work), and
    # they agree to round-off of the grid's scale.
    A = np.array(CELLS["sheared/d3"], dtype=float)
    n, nu, d = 5, 4.0, 3
    Astar = np.linalg.inv(A.T)
    zero = np.zeros(d)
    vals = {j: float(epstein_zeta(nu, A, zero, Astar @ (np.array(j) / n)).real)
            for j in np.ndindex(*(n,) * d)}
    scale = max(abs(v) for v in vals.values())
    worst = max(abs(v - vals[tuple((-np.array(j)) % n)])
                for j, v in vals.items())
    assert 0.0 < worst <= SCALE_ULP * EPS * scale


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_tiny_grids_cover_every_node_once(n):
    A = np.eye(2)
    new = np.asarray(evaluate_graph(BRIDGE, 4.0, A, source=0, terminal=1,
                                    n_points=n))
    old = _direct_grid(4.0, A, n)
    assert new.shape == old.shape == (n, n)
    assert np.all(np.isfinite(new))
    reps, _ = _orbit_masks(n, 2)
    assert np.array_equal(new.ravel()[reps], old.ravel()[reps])
    assert np.abs(new - old).max() <= SCALE_ULP * EPS * np.abs(old).max()


def test_single_k_is_untouched():
    A = np.array(CELLS["sheared/d2"], dtype=float)
    k = np.array([0.3, 0.7])
    got = evaluate_graph(BRIDGE, 4.0, A, source=0, terminal=1, n_points=8,
                         momentum=k)
    want = float(epstein_zeta(4.0, A, np.zeros(2), np.linalg.inv(A.T) @ k).real)
    assert float(got) == want


def test_bridge_accounting_unchanged():
    _, info = evaluate_graph(BRIDGE, 4.0, np.eye(2), source=0, terminal=1,
                             n_points=8, return_diagnostics=True)
    assert info["n_bridges"] == 1
