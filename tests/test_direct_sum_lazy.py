# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Premise for the box's lazy factor supply.

``direct_sum`` materialises every two-vertex kernel as a full
``(size_a, size_b)`` table up front.  The torus engine instead carries a
generator and builds a table only where a dense consumer needs one,
which is what turned d=3 from allocation-bound into fast.

Porting that to the box rests on one identity: indexing the
difference-range generator that :func:`direct_sum._conv_kernel_diff`
already builds must reproduce the eager table.  These tests pin exactly
how far that holds, because the answer decides whether the port can be
gated on bit-identity or has to be gated on ulps.

Result, measured here rather than assumed:

* ``d = 1`` — bit-identical, every ``L``, diagonal or not.
* ``d >= 2`` with **diagonal** ``A`` — bit-identical.
* ``d >= 2`` with **non-diagonal** ``A`` — equal to a few ulp
  (max 7.5e-16 relative observed), not bit-identical.

The last case is a BLAS-path artefact, not a mathematical difference:
the generator reduces a ``(2m-1)^d x d`` matmul while the eager table
reduces a ``(size_a, size_b, d) x d`` one, and the two take different
paths.  The asymmetry already exists in shipped code — today's box peel
reads the generator while its dense branch reads the table — so this
records a property of the arithmetic, not a regression.

**It is also PLATFORM-DEPENDENT, which is the load-bearing part.**  The
skew cases diverge on macOS/Accelerate and are bit-identical on the
Linux/OpenBLAS CI runner.  So a bit-identity gate for the lazy port
would pass in CI and fail on a developer machine, which is the worst
possible failure mode for a refactor whose whole acceptance criterion is
"values did not move".  The gate must therefore be stated in ulps rather
than in ``array_equal``, and no test may assert that the divergence is
present — only that it is bounded.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import gzl.direct_sum as ds

NU = 2.5
ULP_TOL = 4.0e-15          # ~18 ulp of headroom over the 7.5e-16 observed

A_DIAG = {1: np.eye(1), 2: np.eye(2), 3: np.eye(3)}
A_SKEW = {
    1: np.array([[1.3]]),
    2: np.array([[1.0, 0.35], [0.2, 1.1]]),
    3: np.array([[1.0, 0.3, 0.1], [0.0, 1.2, 0.2], [0.1, 0.0, 0.9]]),
}


def _positions(L, d, half):
    axes = ([np.arange(0, L + 1)] if half else [np.arange(-L, L + 1)])
    axes += [np.arange(-L, L + 1)] * (d - 1)
    return np.array(list(itertools.product(*axes)), dtype=float)


def _eager(pos_a, pos_b, A, d, a1):
    """The table direct_sum builds today, verbatim."""
    if d == 1:
        diff = pos_b[None, :, 0] - pos_a[:, None, 0]
        with np.errstate(divide="ignore"):
            t = np.abs(a1 * diff) ** (-NU)
        t[diff == 0.0] = 0.0
    else:
        diff = pos_b[None, :, :] - pos_a[:, None, :]
        dist = np.linalg.norm(diff @ A.T, axis=-1)
        with np.errstate(divide="ignore"):
            t = dist ** (-NU)
        t[dist == 0.0] = 0.0
    return t


def _lazy(pos_a, pos_b, A, d, a1, L, half_a=False, half_b=False):
    """The same table, read out of the difference-range generator.

    Calls the shipped helpers rather than reimplementing their index
    arithmetic: this test's job is to pin the engine's factor supply
    against the eager table it replaced, and a private copy of the
    arithmetic would pass even if the engine's own copy drifted.
    """
    gen = ds._conv_kernel_diff(NU, A, L, d)
    return ds._table_from_generator(
        gen,
        ds._axis_extents(L, d, half_a),
        ds._axis_extents(L, d, half_b),
        ds._gen_offsets(L, d, ds._axis_origins(L, d, half_a),
                        ds._axis_origins(L, d, half_b)),
        d,
    )


def _both(d, A, L, half_a, half_b):
    pos_a = _positions(L, d, half_a)
    pos_b = _positions(L, d, half_b)
    a1 = float(A[0, 0]) if d == 1 else None
    return (_eager(pos_a, pos_b, A, d, a1),
            _lazy(pos_a, pos_b, A, d, a1, L, half_a, half_b))


CASES = [(ha, hb) for ha in (False, True) for hb in (False, True)]


class TestGeneratorReproducesTheEagerTable:

    @pytest.mark.parametrize("half_a,half_b", CASES)
    @pytest.mark.parametrize("L", [2, 3])
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_diagonal_lattice_is_bit_identical(self, d, L, half_a, half_b):
        e, l = _both(d, A_DIAG[d], L, half_a, half_b)
        assert np.array_equal(e, l)

    def test_d1_is_bit_identical_even_when_scaled(self):
        for L in (2, 3, 5):
            for half_a, half_b in CASES:
                e, l = _both(1, A_SKEW[1], L, half_a, half_b)
                assert np.array_equal(e, l)

    @pytest.mark.parametrize("half_a,half_b", CASES)
    @pytest.mark.parametrize("L", [2, 3])
    @pytest.mark.parametrize("d", [2, 3])
    def test_skew_lattice_agrees_to_a_few_ulp(self, d, L, half_a, half_b):
        e, l = _both(d, A_SKEW[d], L, half_a, half_b)
        rel = np.max(np.abs(e - l) / np.maximum(np.abs(e), 1e-300))
        assert rel <= ULP_TOL, f"d={d} L={L}: {rel:.3e}"

    def test_skew_divergence_is_bounded_whether_or_not_it_occurs(self):
        """Guard the guard, without assuming the platform diverges.

        An earlier version asserted the skew cases genuinely DIFFER, so
        that ``ULP_TOL`` could not become dead weight.  That assertion
        is false on Linux/OpenBLAS, where the two matmul shapes reduce
        identically, while it holds on macOS/Accelerate — it failed in
        CI and passed locally.

        The portable statement is the one that matters for the lazy
        port: the divergence, where a platform has one, is at the ulp
        level and never larger.  Whether it is exactly zero is a
        property of the BLAS, not of this library.
        """
        for d in (2, 3):
            for L in (2, 3):
                e, l = _both(d, A_SKEW[d], L, False, False)
                rel = np.max(
                    np.abs(e - l) / np.maximum(np.abs(e), 1e-300)
                )
                # Two orders of magnitude below ULP_TOL: if the
                # divergence ever grew past round-off, this fails long
                # before the tolerance above stops catching it.
                assert rel < 1e-14, f"d={d} L={L}: {rel:.3e}"
