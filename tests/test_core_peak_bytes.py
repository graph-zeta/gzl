# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""The dense-table term of ``_core_peak_bytes`` is charged per two-axis
factor a bucket can hold, not per axis.

A bag-1 bucket holds only one-axis factors and ``dense_step`` sums it in
its scalar branch, so it builds no ``n^d x n^d`` table.  Charging it one
priced a 2-node core -- a 131 kB contraction -- at ``8 N^2``, which is
exactly 2 GiB at d = 2, n = 128, and refused it against
``_DENSE_CORE_MAX_BYTES``.  A treewidth-2 block has no box fallback on a
budget refusal, so every d = 2 pass at ``n_points >= 128`` that met one
died after a second.  No test pinned the term before this file.
"""
from __future__ import annotations

from math import comb

import numpy as np
import pytest

from gzl import _elimination
from gzl.frontend import _DENSE_CORE_MAX_BYTES
from gzl.hybrid import (
    _CORE_RETENTION,
    _DENSE_TABLE_FACTORS,
    _core_peak_bytes,
    hybrid_zeta,
)


def _dense(bag):
    """A step that must run dense: no peel partner."""
    return _elimination.Step(vertex=bag[0], bag=tuple(bag), partners=())


def _old_model(bag, n, d):
    """The shipped price before the correction: ``bag`` tables."""
    N = n ** d
    return (8.0 * float(n) ** (d * (bag - 1)) * _CORE_RETENTION
            + _DENSE_TABLE_FACTORS * bag * 8.0 * float(N) ** 2)


@pytest.mark.parametrize("n, d", [(128, 2), (16, 3), (4096, 1)])
def test_bag1_dense_step_carries_no_table(n, d):
    # One-axis bucket: the scalar branch, result = one double.
    assert _core_peak_bytes([_dense((3,))], n, d) == 8.0 * _CORE_RETENTION


@pytest.mark.parametrize("n, d", [(128, 2), (16, 3)])
def test_bag2_dense_step_carries_one_table(n, d):
    N = n ** d
    want = 8.0 * float(n) ** d * _CORE_RETENTION + 8.0 * float(N) ** 2
    assert _core_peak_bytes([_dense((2, 3))], n, d) == want


@pytest.mark.parametrize("bag", [3, 4, 5])
def test_bag3_and_up_is_the_calibrated_model_unchanged(bag):
    # bag <= C(bag, 2) from bag = 3, so the validation table in the
    # docstring keeps every one of its numbers.
    n, d = 16, 3
    st = _dense(tuple(range(1, bag + 1)))
    assert _core_peak_bytes([st], n, d) == _old_model(bag, n, d)


@pytest.mark.parametrize("bag", [1, 2, 3, 4, 5, 6])
def test_correction_only_ever_lowers_the_estimate(bag):
    # It can turn a refusal into an admission, never the reverse.
    n, d = 20, 2
    st = _dense(tuple(range(1, bag + 1)))
    new = _core_peak_bytes([st], n, d)
    assert new <= _old_model(bag, n, d)
    assert (new < _old_model(bag, n, d)) == (bag < 3)
    assert min(bag, comb(bag, 2)) == (0 if bag == 1 else 1 if bag == 2 else bag)


def test_the_refused_two_node_core_is_admitted_at_d2_n128():
    # The SP-reduced theta block: one residual edge (0, 3), pin 0.
    p = _elimination.plan_for_order(
        [(0, 3)], [3], 0, keep=frozenset(), non_kernel_edges=(),
    )
    assert [len(s.bag) for s in p.steps] == [1]
    assert not p.steps[0].partners
    est = _core_peak_bytes(p.steps, 128, 2)
    assert est < _DENSE_CORE_MAX_BYTES
    assert est < 1e3            # 40 B, not 2 GiB


def test_theta_block_vacuum_runs_under_the_shipped_cap_at_d2_n128():
    # The exact block that killed the d = 2 pass at n_points = 128:
    # 0 -- 1 direct at nu = 8 plus two 2-edge paths 0-2-1 and 0-3-1.
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]])
    nu = np.array([8.0, 4.0, 4.0, 4.0, 4.0])
    A = np.eye(2)
    v128 = hybrid_zeta(edges, nu, A, 128, max_core_bytes=_DENSE_CORE_MAX_BYTES)
    v96 = hybrid_zeta(edges, nu, A, 96, max_core_bytes=_DENSE_CORE_MAX_BYTES)
    assert np.isfinite(v128)
    # The truncation error here decays as n^-(2 nu - d) = n^-6.
    assert abs(v128 - v96) <= 1e-8 * abs(v96)
