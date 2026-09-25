# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""The connectivity-graded core is offered to every block that reaches
hybrid, treewidth 2 included.

At sigma >= 1.49 a tw-2 block skips the sigma_max algebra and hybrid is
its only engine, but core grading was gated on ``tw_block >= 3`` at both
arms.  Measured on the TFIM 1qp corpus at d = 2, nu = 4, order <= 9,
full BZ grid: the nine costliest blocks at n_points = 64 were all tw 2
and carried 16.6 s of an 18.2 s pass (every tw-3 block combined: 0.61 s)
while ``_core_n_for_block`` already sized their cores at 18.  The pass
scaled as n^3.9 in the delivery grid; graded, it scales as ~n^1.2
(2.38 -> 1.04 s at n = 32, 18.22 -> 1.52 s at n = 64) and the
coefficients move by <= 4.8e-09 relative, two decades under the pass
floor.

The grading is value-neutral to round-off at sigma = 2 because the SP
part stays on the delivery grid and only the irreducible core drops:
``core_n_points`` at finite k.  The k = 0 arm is deliberately NOT
extended: there hybrid re-pins the vacuum call at a free pin with
nothing kept, the tw-2 residual is an O(n^d) core with nothing to save,
and the rates at the external pin do not describe that contraction --
grading it moved the pass by 2.2e-06 for 1.00x speed.  Nor is every
finite-k tw-2 block graded: only those whose plan exponent with both
pins kept is >= 2, since an exponent-1 core costs O(n^d) whatever its
grid, and moving its value buys nothing.

Nor is an explicit zero momentum graded.  hybrid contracts
``momentum = 0`` as the vacuum problem -- terminal dropped, re-pinned at
the planner pin -- so the core sized with both endpoints pinned is sized
for a contraction that does not run, and the vacuum residual on it
converges at the block rate.  On the chain at n = 512 that moved the
order-11 coefficient at k = 0 by 1.4e-02 (sigma = 2) against the value
before tw-2 grading, while the grid readout of the same zone centre
moved by 4.1e-06; see ``TestExplicitZeroMomentum``.
"""
from __future__ import annotations

import numpy as np
import pytest

from gzl import data_path, evaluate_graph
from gzl.series import compute_series_coefficients

A1 = np.eye(1)
A2 = np.eye(2)
NU = 4.0                    # sigma = 2: past the 1.49 algebra threshold

CORPUS_1QP = data_path("tfim_softcore_corpus_1qp.npz")

# The corpus block the pass grades at n = 64 (64 -> 18 with its doubled
# edges, 64 -> 38 at uniform nu): K4 minus an edge with the terminals on
# the degree-2 pair, so the free degree-3 pair is adjacent -- one peel
# at bag 3, then a dense step at bag 2 whose partner is blocked by the
# merged factor.  Plan exponent 2 with both pins kept.
DIAMOND = np.array([[0, 1], [0, 3], [1, 2], [1, 3], [2, 3]])
# A tw-2 block the rule WOULD grade (sigma_blk 6 < sigma_core 10, core
# 64 -> 38) but whose every free vertex peels at bag 2: plan exponent 1,
# an O(n^d) core with nothing to save.
V6E7 = np.array([[0, 1], [0, 2], [0, 5], [1, 4], [1, 5], [2, 3], [3, 4]])
# The theta block that killed the pass at n = 128: 0 -- 1 direct at
# nu = 8 (a doubled edge) plus two 2-edge paths 0-2-1 and 0-3-1.
THETA = np.array([[0, 1], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3]])
# Two blocks of the 1qp TFIM corpus that carried the explicit-zero loss
# on the chain, listed with their multiplicities.  BUNDLED (order 11,
# hopping 0 - 1, nu = 3) is a 5-fold bundle between the endpoints, a
# doubled edge 0-2 and single edges 1-3, 1-4, 2-3, 2-4.  With both
# endpoints kept hybrid reduces it to the single edge 0-1, so its graded
# core is exact on the grid; re-pinned for the vacuum problem it keeps a
# free vertex that escapes at rate 5 on a core sized for rate 20 (20 of
# 512 points).  TWO_TERMINAL_CORE (hopping 2 - 3, nu = 2) keeps a genuine
# core with both endpoints: sized for rate 11 at 22 points, its vacuum
# residual converges at rate 3.
BUNDLED = np.repeat(np.array([[0, 1], [0, 2], [1, 3], [1, 4], [2, 3], [2, 4]]),
                    [5, 2, 1, 1, 1, 1], axis=0)
TWO_TERMINAL_CORE = np.repeat(
    np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [3, 5], [4, 5]]),
    [3, 1, 2, 2, 1, 1, 1], axis=0)
# K4 with the edge 2-3 subdivided: a dense (tw 3) block.
K4S = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 4], [4, 3]])


def _diag(edges, n, **kw):
    kw.setdefault("n_points", n)
    val, info = evaluate_graph(edges, NU, A2, return_diagnostics=True, **kw)
    return np.asarray(val), info


class TestFiniteK:

    def test_a_tw2_grid_block_is_graded_and_value_neutral(self):
        graded, info = _diag(DIAMOND, 64, source=0, terminal=2)
        raw, info_raw = _diag(DIAMOND, 64, source=0, terminal=2,
                              core_grading=False)
        assert info["n_block_core_graded_tw2"] >= 1
        assert info_raw["n_block_core_graded_tw2"] == 0
        assert info["n_block_core_graded"] == 0          # not a dense block
        assert graded.shape == (64, 64)
        # Measured <= 5.3e-9 on this block at three k against the raw
        # torus, with the torus itself converged to 4e-16 (raw64 vs
        # raw96) and the box 2e-12 .. 5.6e-9 away.
        assert np.max(np.abs(graded - raw)) <= 1e-7 * np.max(np.abs(raw))

    def test_single_k_matches_the_grid_node(self):
        grid, _ = _diag(DIAMOND, 32, source=0, terminal=2)
        one, info = _diag(DIAMOND, 32, source=0, terminal=2,
                          momentum=np.array([0.25, 0.5]))
        assert info["n_block_core_graded_tw2"] >= 1
        assert float(one) == pytest.approx(float(grid[8, 16]), rel=1e-12)

    def test_the_grid_runs_at_n128_where_the_pass_used_to_die(self):
        v128, info = _diag(DIAMOND, 128, source=0, terminal=2)
        v96, _ = _diag(DIAMOND, 96, source=0, terminal=2)
        assert info["n_block_core_graded_tw2"] >= 1
        # Same trigonometric polynomial evaluated on both grids: compare
        # on the common nodes k = j / 32.
        assert np.allclose(v128[::4, ::4], v96[::3, ::3], rtol=1e-7, atol=0)

    @pytest.mark.parametrize("edges, t", [
        (V6E7, 1),
        (np.array([[0, 1], [0, 2], [1, 2]]), 1),     # one free vertex
    ])
    def test_an_exponent_1_core_is_left_alone(self, edges, t):
        # The rule WOULD grade these (sigma_core > sigma_blk), and they
        # were 673 of the 823 tw-2 occurrences at n = 64 -- but an
        # O(n^d) core has nothing to save, so the exponent gate keeps
        # their values bit-identical instead of moving them for no time.
        from gzl.frontend import _block_plan_exponent
        assert _block_plan_exponent(edges, 0, (0, t)) == 1
        graded, info = _diag(edges, 64, source=0, terminal=t)
        raw, _ = _diag(edges, 64, source=0, terminal=t, core_grading=False)
        assert info["n_block_core_graded_tw2"] == 0
        assert np.array_equal(graded, raw)


class TestVacuum:
    """The k = 0 arm is untouched: tw-2 vacuum blocks are never graded.

    hybrid re-pins a vacuum call at the planner's free pin and keeps
    nothing, so a tw-2 block SP-collapses to an O(n^d) residual.  The
    rates the arm sizes with are taken at the EXTERNAL pin (the theta
    below: core 14, block 6 -- a gap) and do not describe that
    contraction, which converges at the block rate.  Wiring the rule in
    anyway moved the order-9 d = 2 pass by 2.2e-06 at n = 64, above the
    pass floor, for 1.00x speed.  So: dense blocks only there, as before.
    """

    @pytest.mark.parametrize("kw", [{}, {"richardson": False}])
    def test_tw2_vacuum_blocks_are_never_graded(self, kw):
        from gzl.frontend import _block_cut_rates
        sb, sc = _block_cut_rates(THETA, np.full(len(THETA), NU), 2, 0,
                                  externals=(0,))
        assert sb < sc                     # the gap the arm must NOT spend
        graded, info = _diag(THETA, 64, **kw)
        raw, _ = _diag(THETA, 64, core_grading=False, **kw)
        assert info["n_block_core_graded_tw2"] == 0
        assert info["n_block_core_graded"] == 0
        assert float(graded) == float(raw)

    def test_the_theta_block_runs_at_n128(self):
        # The core byte model (no table term at bag 1, see
        # hybrid._core_peak_bytes) admits its 2-node core; grading tw-2
        # cores at finite k is what keeps a pass with finite-k neighbours
        # affordable.  Both are exercised at n = 128.
        v128, _ = _diag(THETA, 128)
        v96, _ = _diag(THETA, 96)
        assert float(v128) == pytest.approx(float(v96), rel=1e-8)


def _scalar(v):
    # A length-1 momentum in d = 1 is a batch of one, and a series
    # coefficient is stored complex; unwrap either form to a real float.
    return float(np.real(np.asarray(v)).ravel()[0])


class TestExplicitZeroMomentum:
    """An on-spine tw-2 block at an explicit zero momentum is not graded.

    It is a finite-k request (it reaches ``_block_at_finite_k``), but
    hybrid contracts ``momentum = 0`` as the vacuum problem, which the
    ``TestVacuum`` argument above already keeps off the graded core.
    Measured per block against its ungraded value when it was graded:
    BUNDLED 2.6e-06 at n = 512, TWO_TERMINAL_CORE 1.6e-03 at n = 512,
    DIAMOND 8.9e-09 at n = 64.  Ungraded, all three are exact to the
    n-point truncation and cost no more.
    """

    CASES = [
        pytest.param(BUNDLED, 3.0, A1, 512, 0, 1, id="bundled-chain"),
        pytest.param(TWO_TERMINAL_CORE, 2.0, A1, 512, 2, 3, id="core-chain"),
        pytest.param(DIAMOND, NU, A2, 64, 0, 2, id="diamond-square"),
    ]

    @pytest.mark.parametrize("edges, nu, A, n, s, t", CASES)
    def test_an_explicit_zero_is_the_ungraded_value(self, edges, nu, A, n, s, t):
        kw = dict(source=s, terminal=t, n_points=n,
                  momentum=np.zeros(A.shape[0]), accuracy="floor")
        val, info = evaluate_graph(edges, nu, A, return_diagnostics=True, **kw)
        raw = evaluate_graph(edges, nu, A, core_grading=False, **kw)
        assert info["n_block_core_graded_tw2"] == 0
        assert _scalar(val) == _scalar(raw)

    @pytest.mark.parametrize("edges, nu, A, n, s, t", CASES)
    def test_it_equals_the_ungraded_grid_node(self, edges, nu, A, n, s, t):
        # On the full n-point torus the vacuum rewrite and the kept
        # terminal are the same lattice sum, reassociated (measured
        # <= 1.5e-14).  This is what explicit zero ships now.
        kw = dict(source=s, terminal=t, n_points=n, accuracy="floor")
        k0 = evaluate_graph(edges, nu, A, momentum=np.zeros(A.shape[0]), **kw)
        grid = evaluate_graph(edges, nu, A, core_grading=False, **kw)
        assert _scalar(k0) == pytest.approx(np.asarray(grid).flat[0], rel=1e-13)

    @pytest.mark.parametrize("edges, nu, A, n, s, t", CASES)
    def test_the_grid_and_a_nonzero_k_are_still_graded(self, edges, nu, A, n, s, t):
        d = A.shape[0]
        kw = dict(source=s, terminal=t, n_points=n, accuracy="floor",
                  return_diagnostics=True)
        _, info = evaluate_graph(edges, nu, A, **kw)
        assert info["n_block_core_graded_tw2"] >= 1
        k = 0.5 if d == 1 else np.full(d, 0.5)
        _, info = evaluate_graph(edges, nu, A, momentum=k, **kw)
        assert info["n_block_core_graded_tw2"] >= 1

    def test_the_grid_node_agrees_where_its_core_is_exact(self):
        # BUNDLED keeps only the edge 0-1 with both endpoints pinned, so
        # the graded grid node 0 is exact (measured 3.3e-16) and explicit
        # zero now lands on it; it used to be 2.6e-06 away.
        kw = dict(source=0, terminal=1, n_points=512, accuracy="floor")
        grid = evaluate_graph(BUNDLED, 3.0, A1, **kw)
        k0 = evaluate_graph(BUNDLED, 3.0, A1, momentum=0.0, **kw)
        assert _scalar(k0) == pytest.approx(grid[0], rel=1e-13)

    def test_a_batch_treats_its_zero_entry_like_a_scalar_zero(self):
        kw = dict(source=2, terminal=3, n_points=512, accuracy="floor")
        batch = evaluate_graph(TWO_TERMINAL_CORE, 2.0, A1,
                               momentum=np.array([0.0, 0.5]), **kw)
        for i, k in enumerate((0.0, 0.5)):
            one = evaluate_graph(TWO_TERMINAL_CORE, 2.0, A1, momentum=k, **kw)
            assert batch[i] == _scalar(one)

    def test_the_dense_arm_still_grades_an_explicit_zero(self):
        # The change is scoped to the tw <= 2 extension.  The dense-on-
        # torus branch keeps its graded core at k = 0, value unchanged.
        _, info = evaluate_graph(K4S, 2.5, A2, n_points=32, terminal=1,
                                 momentum=np.zeros(2), return_diagnostics=True)
        assert info["n_block_core_graded"] >= 1
        assert info["n_block_core_graded_tw2"] == 0

    def test_the_corpus_pass_at_explicit_zero(self):
        # Chain, sigma = 2, n = 256, orders <= 9.  What is left between
        # graded and ungraded is the dense arm's own grading: measured
        # 1.4e-11, 4.2e-10, 4.9e-09 at orders 7, 8, 9.  With the tw-2
        # blocks graded it was 1.0e-07, 2.6e-06, 1.9e-04.
        kw = dict(momentum=np.zeros(1), order_max=9)
        graded = compute_series_coefficients(CORPUS_1QP, np.array([3.0]), A1, 256, **kw)
        raw = compute_series_coefficients(CORPUS_1QP, np.array([3.0]), A1, 256,
                                          core_grading=False, **kw)
        for o in sorted(raw):
            g, r = _scalar(graded[o]), _scalar(raw[o])
            assert abs(g - r) <= 5e-8 * abs(r), f"order {o}: {g!r} vs {r!r}"


class TestScope:

    def test_bridges_and_dense_counters_are_untouched(self):
        # A bridge is closed-form and never reaches hybrid.
        _, info = _diag(np.array([[0, 1]]), 64)
        assert info["n_block_core_graded_tw2"] == 0
        assert info["n_block_core_graded"] == 0

    def test_off_the_torus_engine_a_tw2_block_still_grades(self):
        # Hybrid is a tw-2 block's only engine, so the dense-engine
        # choice must not switch its grading off.
        _, info = _diag(DIAMOND, 64, source=0, terminal=2,
                        dense_engine="direct_sum")
        assert info["n_block_core_graded_tw2"] >= 1
