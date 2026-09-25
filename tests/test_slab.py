# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""The slab: a dense core contracted with a second pin.

Three properties carry this engine and each has a class below.

* **It is the same lattice sum, on any lattice.**  The slab reassociates
  the torus contraction; it does not re-truncate it.  So its value must
  equal the shipped tensor engine's, and it must do so on cells that are
  not cubic — which is where the engine's predecessor, a standalone
  analysis script, was never tested.
* **The orbit reduction is exact, and its group is measured.**  The
  reduction is the whole cost case, and the obvious analytic derivation
  of its group is WRONG at even ``n``.  The reduced sum is therefore
  checked against the unreduced one on every cell, not argued for.
* **The router prices it before allocating it.**  ``choose_slab`` and
  the frontend's ``_slab_grid`` both rest on ``slab_schedule``
  reproducing the exponent the executor actually achieves.

THE BUG THIS MODULE EXISTS TO KEEP DEAD.  ``PT G P = G`` — "sigma
preserves the metric" — admits ``-I`` on every lattice, because ``K`` is
even.  At even ``n`` on a cell with a cross term that is false: the
balanced axis at ``n = 6`` is ``[0, 1, 2, 3, -2, -1]``, label ``3`` has
no partner ``-3`` in the window, and on the triangular cell
``|A (3, 1)|**2 = 13`` while ``|A (3, -1)|**2 = 7``.  Trusting the
algebra made the prism come out 4.23e-03 wrong with 6 of 15 orbits
non-constant, while the unreduced sum was exact to 1.7e-16 — a smooth,
plausible, entirely wrong number.  ``lattice_window_group`` therefore
tests candidates against ``_label_distances``, the array the kernels are
built from.
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl import graph_zeta_general_at_zero, slab_zeta
from gzl import frontend
from gzl.slab import (
    SlabTooLargeError,
    choose_slab,
    lattice_window_group,
    orbit_reps,
    slab_peak_bytes,
    slab_schedule,
)

from tests._env_gate import assert_pinned

K4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PRISM = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
         (0, 3), (1, 4), (2, 5)]
K33 = [(i, j) for i in (0, 1, 2) for j in (3, 4, 5)]
K5 = [(i, j) for i in range(5) for j in range(i + 1, 5)]
V6E11 = [(0, 1), (0, 4), (1, 2), (1, 3), (1, 5), (2, 3),
         (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
CORES = {"K4": K4, "prism": PRISM, "K33": K33, "K5": K5, "V6E11": V6E11}

TRIANGULAR = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
SHEARED3 = np.array([[1.0, 0.3, 0.0], [0.0, 1.1, 0.2], [0.0, 0.0, 0.9]])
CELLS = {
    1: [("unit", np.eye(1)), ("scaled", np.array([[1.7]]))],
    2: [("square", np.eye(2)), ("triangular", TRIANGULAR),
        ("rect", np.diag([1.0, 1.4])),
        ("sheared", np.array([[1.0, 0.37], [0.0, 1.13]]))],
    3: [("cubic", np.eye(3)), ("tetragonal", np.diag([1.0, 1.0, 1.3])),
        ("orthorhombic", np.diag([1.0, 1.2, 1.4])), ("sheared", SHEARED3)],
}
# BOTH parities at every dimension.  The even entry is what arms the
# reflection bug; an odd-only sweep passes with a wrong group.
NS = {1: (7, 8), 2: (5, 6), 3: (4, 5)}


class TestItIsTheSameLatticeSum:
    r"""Against the shipped tensor engine, on cells that are not cubic."""

    @pytest.mark.parametrize("core", sorted(CORES))
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_agrees_with_the_tensor_engine_on_every_cell(self, core, d):
        E = CORES[core]
        for cell_name, A in CELLS[d]:
            for n in NS[d]:
                nu = np.full(len(E), d + 0.5)
                ref = float(np.real(graph_zeta_general_at_zero(
                    np.array(E), nu, A, n)))
                got = slab_zeta(E, nu, A, n)[0]
                assert got == pytest.approx(ref, rel=1e-12), (
                    f"{core} on {cell_name} at d={d}, n={n}"
                )

    def test_the_pin_and_the_slab_vertex_are_both_free(self):
        r"""Every (source, slab) pair is the same value.

        The value is pin-independent by translation invariance and
        slab-independent because the slab is only a reassociation.  A
        bug in either would show up as a spread here and nowhere else in
        this file, since every other test takes the default choice.
        """
        nu = np.full(len(PRISM), 2.5)
        vals = [slab_zeta(PRISM, nu, TRIANGULAR, 6, source=s, slab=t)[0]
                for s in range(6) for t in range(6) if s != t]
        assert len(vals) == 30
        assert max(vals) == pytest.approx(min(vals), rel=1e-12)

    @pytest.mark.parametrize("name,E", [
        ("bridge", [(0, 1)]),
        ("path", [(0, 1), (1, 2)]),
        ("star", [(0, 1), (0, 2), (0, 3)]),
        ("triangle", [(0, 1), (1, 2), (0, 2)]),
    ])
    def test_shapes_with_nothing_left_to_contract(self, name, E):
        r"""The router never sends these -- it routes tw >= 3 only -- but
        the engine is public, and the bridge is the ONLY case that
        reaches the "no factors at all" branch, where both endpoints are
        pinned and every cell is a bare scalar.  Nothing else in this
        file executes that line.
        """
        nu = np.full(len(E), 2.5)
        ref = float(np.real(graph_zeta_general_at_zero(
            np.array(E), nu, np.eye(1), 8)))
        got, _, inner = slab_zeta(E, nu, np.eye(1), 8)
        assert got == pytest.approx(ref, rel=1e-12)
        assert inner == (0 if len(E) == 1 else 1)

    def test_parallel_edges_are_carried_as_a_hadamard_product(self):
        r"""The router Hadamard-merges before dispatching; the PUBLIC
        API does not, so ``slab_zeta`` has to handle a repeated ``(u, v)``
        itself.  ``K_a * K_b = K_(a+b)``, so the doubled edge must equal
        the single edge at the summed exponent."""
        par = K4 + [(0, 1)]
        nu_par = np.array([2.5] * 6 + [2.5])
        merged_nu = np.array([5.0, 2.5, 2.5, 2.5, 2.5, 2.5])
        for d, n in ((1, 8), (2, 6)):
            A = np.eye(d)
            ref = float(np.real(graph_zeta_general_at_zero(
                np.array(par), nu_par, A, n)))
            assert slab_zeta(par, nu_par, A, n)[0] == pytest.approx(
                ref, rel=1e-12)
            assert slab_zeta(K4, merged_nu, A, n)[0] == pytest.approx(
                ref, rel=1e-12)

    def test_the_value_does_not_depend_on_edge_order_or_labelling(self):
        nu, A = np.full(6, 2.5), np.eye(2)
        base = slab_zeta(K4, nu, A, 6)[0]
        assert slab_zeta(K4[::-1], nu[::-1], A, 6)[0] == base
        perm = {0: 2, 1: 0, 2: 3, 3: 1}
        assert slab_zeta([(perm[u], perm[v]) for u, v in K4],
                         nu, A, 6)[0] == base

    def test_labels_are_the_ones_present_not_zero_through_max(self):
        r"""Pins the convention, shared by every engine since the
        label-convention unification: the vertex set is the labels
        appearing in ``edges``, and a gap label is not a vertex.
        ``graph_zeta_general_at_zero`` historically took ``0..max`` and
        counted every gap as an isolated vertex worth ``n**d`` (this
        test used to pin that disagreement as a ``(n**2)**6`` ratio);
        labels are now normalised at every public entry
        (gzl/_labels.py), and the fuller cross-engine matrix
        lives in tests/test_label_convention.py.
        """
        from gzl import hybrid_zeta
        nc = [(0, 5), (5, 9), (0, 9), (0, 7), (5, 7), (9, 7)]  # K4
        nu, A, n = np.full(6, 2.5), np.eye(2), 6
        compact = slab_zeta(K4, nu, A, n)[0]
        assert slab_zeta(nc, nu, A, n)[0] == pytest.approx(compact, rel=1e-12)
        assert float(np.real(hybrid_zeta(
            np.array(nc), nu, A, n))) == pytest.approx(compact, rel=1e-12)
        tensor = float(np.real(graph_zeta_general_at_zero(
            np.array(nc), nu, A, n)))
        assert tensor == pytest.approx(compact, rel=1e-12)

    def test_a_bundle_of_unequal_exponents_is_carried_per_edge(self):
        nu = np.array([2.5, 3.0, 4.5, 2.5, 6.0, 3.5])
        ref = float(np.real(graph_zeta_general_at_zero(
            np.array(K4), nu, np.eye(2), 6)))
        assert slab_zeta(K4, nu, np.eye(2), 6)[0] == pytest.approx(
            ref, rel=1e-12)


class TestTheOrbitReductionIsExact:
    r"""Reduced against unreduced — the control the group cannot fake."""

    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_reduced_equals_unreduced_on_every_cell(self, d):
        for core in ("K5", "prism", "V6E11"):
            E = CORES[core]
            for cell_name, A in CELLS[d]:
                for n in NS[d]:
                    nu = np.full(len(E), d + 0.5)
                    red = slab_zeta(E, nu, A, n, use_symmetry=True)[0]
                    unr = slab_zeta(E, nu, A, n, use_symmetry=False)[0]
                    assert red == pytest.approx(unr, rel=1e-12), (
                        f"{core} on {cell_name} at d={d}, n={n}"
                    )

    def test_the_even_n_triangular_case_that_caught_the_algebra(self):
        r"""The regression itself, pinned as a value.

        The prism on a triangular cell at n = 6 is the smallest case
        where a metric-only group is wrong.  Left as its own test rather
        than folded into the sweep because the sweep would still pass
        with the reduction disabled, and this one would not: it asserts
        the group actually SHRANK at even n.
        """
        nu = np.full(len(PRISM), 2.5)
        ref = float(np.real(graph_zeta_general_at_zero(
            np.array(PRISM), nu, TRIANGULAR, 6)))
        assert slab_zeta(PRISM, nu, TRIANGULAR, 6)[0] == pytest.approx(
            ref, rel=1e-12)
        # 4 at odd n, 2 at even n: the two reflections the metric
        # argument keeps and the window does not.
        assert len(lattice_window_group(TRIANGULAR, 5)) == 4
        assert len(lattice_window_group(TRIANGULAR, 6)) == 2

    def test_use_symmetry_false_actually_disables_the_reduction(self):
        r"""Hole this closes: with ``if use_symmetry:`` mutated to
        ``if True:`` -- a single token, making the unreduced arm
        unreachable -- the whole 54-test module passed, because the
        headline control below compares reduced against unreduced by
        VALUE and both arms were then the same code.  ``x == x``.

        Cell COUNTS cannot degenerate that way: unreduced is ``n**d`` by
        definition, and any reduction at all is fewer.
        """
        for d, n in ((1, 8), (2, 6), (3, 5)):
            A = np.eye(d)
            _, cells_off, _ = slab_zeta(K5, np.full(len(K5), d + 0.5), A, n,
                                        use_symmetry=False)
            _, cells_on, _ = slab_zeta(K5, np.full(len(K5), d + 0.5), A, n,
                                       use_symmetry=True)
            assert cells_off == n ** d, f"d={d}: unreduced must be n**d"
            assert cells_on < cells_off, f"d={d}: reduction did nothing"

    def test_the_reduction_reduces_by_the_cell_s_own_factor(self):
        r"""Hole this closes: replacing ``orbit_reps``'s body with
        ``arange(n**d), ones(n**d)`` -- no reduction whatsoever, the
        entire cost case of the engine gone -- passed all 54 tests.
        Every existing check was on the VALUE, which a no-op reduction
        preserves exactly.

        These are absolute counts, not derived from the function under
        test: at d = 3, n = 12 the group folds 1728 outer cells onto 84
        (cubic), 196 (tetragonal), 343 (orthorhombic) or 1728 (a cell
        with cross terms at even n, where only the identity survives).
        """
        expected = {
            84: np.eye(3),
            196: np.diag([1.0, 1.0, 1.3]),
            343: np.diag([1.0, 1.2, 1.4]),
            1728: np.array([[1.0, 0.3, 0.2], [0.0, 1.0, 0.3], [0.0, 0.0, 1.0]]),
        }
        for n_cells, A in expected.items():
            reps, mult = orbit_reps(12, 3, A)
            assert len(reps) == n_cells, f"{A.diagonal()} -> {len(reps)}"
            assert int(round(float(mult.sum()))) == 12 ** 3

    def test_multiplicities_partition_the_torus(self):
        for d in (1, 2, 3):
            for _, A in CELLS[d]:
                for n in NS[d]:
                    reps, mult = orbit_reps(n, d, A)
                    assert len(reps) == len(set(reps.tolist()))
                    assert int(round(float(mult.sum()))) == n ** d
                    assert np.all(mult >= 1.0)


class TestTheGroupIsDerivedNotAssumed:
    r"""What ``lattice_window_group`` finds, per cell."""

    def test_the_sizes_are_the_cell_s_own(self):
        assert len(lattice_window_group(np.eye(3), 12)) == 48
        assert len(lattice_window_group(np.diag([1.0, 1.0, 1.3]), 12)) == 16
        assert len(lattice_window_group(np.diag([1.0, 1.2, 1.4]), 12)) == 8
        # Even a fully generic cell keeps a reduction at ODD n, and the
        # cubic-only predicate this replaced threw that away.
        assert len(lattice_window_group(SHEARED3, 5)) == 2

    def test_a_near_symmetry_is_not_a_symmetry(self):
        r"""Accepting one returns a smooth wrong number, so the
        tolerance is tight enough to call a 1e-6 deviation what it is:
        this cell is tetragonal, not cubic."""
        near = np.diag([1.0, 1.0, 1.0 + 1e-6])
        assert len(lattice_window_group(near, 12)) == 16

    def test_the_identity_is_always_in_it(self):
        for d in (1, 2, 3):
            for _, A in CELLS[d]:
                g = lattice_window_group(A, 6)
                assert (tuple(range(d)), (1,) * d) in g
                assert len(g) >= 1


class TestThePriceIsTheMeasurement:
    r"""The symbolic schedule against what the executor achieves.

    This is what lets the router price a slab before allocating one.
    The analysis tool this code began as measured the exponent by
    RUNNING one cell; a router cannot.
    """

    @staticmethod
    def _executed_exponent(E, nu, A, n, source, slab):
        r"""The exponent the EXECUTOR actually achieves, by recording it.

        ``slab_zeta`` returns ``slab_schedule``'s number verbatim, so
        comparing the two is ``x == x`` -- an earlier version of the test
        below did exactly that and could not fail.  The only way to check
        the price is to instrument the two methods the shared executor
        dispatches to and read the bag sizes off the real contraction,
        which is what ``tests/test_tau_d_acceptance.py`` does for the
        same reason.
        """
        from gzl import _contract
        from gzl.tensor_network import (
            TorusTruncation, _edge_kernel_torus, _pin_from_generator_at,
        )
        from gzl.slab import _edge_value_both_pinned

        steps = []

        class _Recording(TorusTruncation):
            def peel_step(self, bucket, token, v, out_axes, dtype):
                steps.append((len({a for f in bucket
                                   for a in self.scope_of(f)}), True))
                return super().peel_step(bucket, token, v, out_axes, dtype)

            def dense_step(self, bucket, v, out_axes, dtype):
                steps.append((len({a for f in bucket
                                   for a in self.scope_of(f)}), False))
                return super().dense_step(bucket, v, out_axes, dtype)

        d = int(np.asarray(A).shape[0])
        order, _, _ = slab_schedule(E, source, slab)
        kern = {float(x): _edge_kernel_torus(float(x), A, n)
                for x in set(np.asarray(nu, float).tolist())}
        trunc = _Recording(n, d, np.asarray(A, float))
        # The first cell that actually CONTRACTS, not cell 0: when the
        # slab vertex is adjacent to the source, z = 0 puts them on one
        # site and K(0) = 0 kills the cell before any step runs.
        for z in range(n ** d):
            factors, scalar = [], 1.0
            for (a, b), nn in zip(E, np.asarray(nu, float).tolist()):
                K = kern[float(nn)]
                ia = 0 if a == source else (z if a == slab else None)
                ib = 0 if b == source else (z if b == slab else None)
                if ia is not None and ib is not None:
                    val = _edge_value_both_pinned(K, ia, ib)
                    if val == 0.0:
                        scalar = 0.0
                        break
                    scalar *= val
                elif ia is not None:
                    factors.append(([b], _pin_from_generator_at(
                        K, ia, pin_row=True), None))
                elif ib is not None:
                    factors.append(([a], _pin_from_generator_at(
                        K, ib, pin_row=False), None))
                else:
                    factors.append(([a, b], None, K))
            if scalar == 0.0 or not factors:
                continue
            _contract.eliminate(list(factors), list(order), trunc)
            break
        assert steps, "the probe recorded no contraction step"
        return max(bag - (1 if peeled else 0) for bag, peeled in steps)

    @pytest.mark.parametrize("core", sorted(CORES))
    def test_every_candidate_slab_is_priced_correctly(self, core):
        r"""The symbolic price against what the executor DOES.

        This is the claim the router rests on -- it prices a slab before
        allocating one, which the analysis tool this code began as could
        not do because it measured the exponent by running a cell.
        """
        E = CORES[core]
        nodes = sorted({v for e in E for v in e})
        A, nu = np.eye(3), np.full(len(E), 3.5)
        for cand in nodes[1:]:
            _, _, symbolic = slab_schedule(E, 0, cand)
            measured = self._executed_exponent(E, nu, A, 4, 0, cand)
            assert symbolic == measured, (
                f"{core}, slab={cand}: priced {symbolic}, executed {measured}"
            )

    def test_the_slab_drops_the_exponent_by_one_on_the_dense_cores(self):
        r"""The mechanism, as a number.  K5 is plan exponent 3 and its
        slab inner exponent is 2 — that factor of ``n**d`` in peak
        memory is the entire reason this engine exists."""
        from gzl import _elimination
        for core in ("K5", "V6E11"):
            E = CORES[core]
            nodes = sorted({v for e in E for v in e})
            full = _elimination.plan(
                [tuple(e) for e in E], nodes).exponent
            _, _, inner = slab_schedule(E, 0, choose_slab(E, 3, source=0))
            assert inner == full - 1, core

    def test_choose_slab_prefers_the_exponent_not_the_degree(self):
        r"""V6E11 has four degree-4 free vertices and they are not
        equivalent: one leaves a K4 behind (exponent unchanged) and
        another does not.  A degree heuristic picks wrong."""
        chosen = choose_slab(V6E11, 3, source=0)
        _, _, best = slab_schedule(V6E11, 0, chosen)
        for cand in range(1, 6):
            _, _, ex = slab_schedule(V6E11, 0, cand)
            assert ex >= best

    def test_choose_slab_is_deterministic(self):
        # The block cache would otherwise see two values for one block.
        for core, E in CORES.items():
            first = choose_slab(E, 3, source=0)
            assert all(choose_slab(E, 3, source=0) == first
                       for _ in range(3)), core

    def test_a_block_with_no_inner_vertex_is_still_priced(self):
        r"""A block whose only vertices are the source and the slab has
        nothing to eliminate, so the schedule is empty and the inner
        model prices it at exactly 0.0 -- whereupon ``max_bytes`` could
        never refuse it, however large, while ``slab_zeta`` went on to
        build an ``8 * n**d`` kernel (953.7 MiB at d = 3, n = 500).

        Not reachable from the router (`_slab_grid` needs plan exponent
        3, and this block is exponent 0), but ``slab_zeta`` is public and
        its docstring promises the refusal.
        """
        assert slab_schedule([(0, 1)], 0, 1)[2] == 0
        # The floor is the OUTER loop's setup, priced from the arrays
        # that exist: 4d + 3 planes of 8 bytes per site.  Measured
        # against ru_maxrss at d = 3: 112.5 B/site at n = 60, 88.3 at
        # n = 150, against the 120 priced -- over by 7-35%, which is the
        # direction that refuses rather than allocates.
        for d, n in ((3, 12), (3, 200), (2, 4000)):
            need = slab_peak_bytes([(0, 1)], 0, 1, n, d)
            assert need >= 8.0 * (4 * d + 3) * n ** d, f"d={d} n={n}: {need}"
        with pytest.raises(SlabTooLargeError):
            slab_zeta([(0, 1)], [3.5], np.eye(3), 200, max_bytes=1024.0)
        # ...and a zero budget must refuse, not be vacuously satisfied.
        with pytest.raises(SlabTooLargeError):
            slab_zeta([(0, 1)], [3.5], np.eye(3), 64, max_bytes=0.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
    def test_a_budget_that_is_not_a_budget_is_refused(self, bad):
        r"""``need > nan`` is False, so a NaN budget silently disabled
        the cap entirely -- the caller asked for a limit and got none.
        ``inf`` and a negative value are rejected for the same reason:
        ``None`` is the documented way to say "no limit"."""
        with pytest.raises(ValueError, match="max_bytes"):
            slab_zeta(K4, np.full(6, 2.5), np.eye(1), 8, max_bytes=bad)

    def test_the_kernel_floor_does_not_move_a_real_threshold(self):
        r"""The floor is `8 * n**d`; a real dense block's inner term is
        orders of magnitude above it, so adding the floor must leave
        those prices untouched."""
        from gzl.hybrid import _core_peak_bytes
        _, steps, _ = slab_schedule(K5, 0, choose_slab(K5, 3, source=0))
        inner = float(_core_peak_bytes(steps, 12, 3))
        assert slab_peak_bytes(K5, 0, choose_slab(K5, 3, source=0), 12, 3) == inner
        assert inner > 100.0 * 8.0 * 12 ** 3

    def test_the_byte_model_is_the_right_ORDER_not_just_self_consistent(self):
        r"""Hole this closes: the refusal test below derives its budget
        from the function it tests (``max_bytes=need/2``), so it is
        invariant under ANY rescaling -- multiplying
        ``slab_peak_bytes`` by 1e-6 left all 54 tests green.

        Anchored against arithmetic instead.  K5's slab at d = 3 has
        inner exponent 2, so its dominant step holds an
        ``n**(d*2)``-element float64 array: 8 x 12**6 = 23.9 MB.  The
        model adds chunk/table terms on top, so it must land ABOVE that
        and within a small factor of it -- not 24 bytes, and not 24 TB.
        """
        need = slab_peak_bytes(K5, 0, choose_slab(K5, 3, source=0), 12, 3)
        floor = 8.0 * float(12 ** (3 * 2))          # 23.9 MB, the result array
        assert floor < need < 32.0 * floor, (
            f"modelled {need / 1024**2:.1f} MiB against a "
            f"{floor / 1024**2:.1f} MiB arithmetic floor"
        )

    def test_the_byte_model_refuses_before_allocating(self):
        need = slab_peak_bytes(K5, 0, choose_slab(K5, 3, source=0), 12, 3)
        assert need > 0.0
        with pytest.raises(SlabTooLargeError):
            slab_zeta(K5, np.full(len(K5), 3.5), np.eye(3), 12,
                      max_bytes=need / 2.0)
        # A MemoryError subclass, so the frontend's existing box
        # fallbacks catch a priced refusal exactly as they catch
        # hybrid's.
        assert issubclass(SlabTooLargeError, MemoryError)


class TestTheEngineRefusesWhatItCannotDo:

    def test_self_loops(self):
        with pytest.raises(ValueError):
            slab_zeta([(0, 0), (0, 1)], [2.5, 2.5], np.eye(1), 6)

    def test_the_slab_vertex_may_not_be_the_source(self):
        with pytest.raises(ValueError):
            slab_zeta(K4, np.full(6, 2.5), np.eye(2), 6, source=1, slab=1)

    def test_a_vertex_outside_the_block(self):
        with pytest.raises(ValueError):
            slab_zeta(K4, np.full(6, 2.5), np.eye(2), 6, slab=9)

    def test_a_mismatched_nu_vector(self):
        with pytest.raises(ValueError):
            slab_zeta(K4, np.full(3, 2.5), np.eye(2), 6)


class TestTheRoutingPredicate:
    r"""``frontend._slab_grid`` — every refusal lands on the box."""

    A3 = np.eye(3)
    K5A = np.array(K5, dtype=int)
    NU = np.full(len(K5), 3.5)
    V = list(range(5))

    def _grid(self, **kw):
        kw = {"edges": self.K5A, "nu_vec": self.NU, "d": 3, "A": self.A3,
              **kw}
        got = frontend._slab_grid(
            kw["edges"], kw["nu_vec"], kw["d"], kw["A"],
            nn_mode=kw.get("nn_mode", False))
        return None if got is None else got[0]

    def test_the_measured_class_is_taken(self):
        assert self._grid() == 12
        assert frontend._SLAB_N_BY_CLASS == {(3, 3): 12}

    def test_the_pin_is_the_planner_s_and_nothing_is_kept(self):
        r"""The k = 0 dense block is a SCALAR however many cut vertices
        it carries, and the box arm beside this one already lets
        ``_pick_root`` choose.  Classifying at the router's root instead
        is the difference between exponent 3 and exponent 2 on the
        census shapes -- i.e. between this arm and the box.
        """
        from gzl import _elimination
        # V6E11-with-an-edge-removed is exponent 2 at the free pin and 3
        # at vertex 0, so a root-pinned classifier would wrongly take it.
        E = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4),
             (2, 3), (2, 4), (3, 4)]
        nodes = list(range(5))
        assert _elimination.plan(E, nodes, pin=0).exponent == 3
        assert _elimination.plan(E, nodes).exponent == 2
        assert frontend._slab_grid(
            np.array(E, dtype=int), np.full(len(E), 3.5), 3, self.A3) is None
        # ...and on a shape that IS exponent 3 freely, the pin handed
        # back is the planner's own.
        n, pin = frontend._slab_grid(self.K5A, self.NU, 3, self.A3)
        assert (n, pin) == (12, _elimination.plan(
            [tuple(e) for e in self.K5A.tolist()], list(range(5))).pin)

    def test_the_exponent_2_class_is_deliberately_absent(self):
        r"""A first calibration was cost-blind, and the measurement
        refuted it.

        At free-pin exponent 2 the box runs in 0.2-1.7 s against the
        slab's 13-27 s, and the accuracy verdict does NOT pay for that:
        strict resolved bounds over the 8 shapes are 0.46x, 0.72x,
        0.91x, 1.25x, 1.30x, 1.54x, 1.70x, 4.94x -- worse on three.  At
        exponent 3 the slab wins on both axes on all five (1.61x -
        30.38x accuracy, 3.2x - 7.4x speed), which is what the entry
        buys.
        """
        assert (3, 2) not in frontend._SLAB_N_BY_CLASS

    def test_the_group_is_queried_at_the_shipped_grid(self):
        r"""Hole this closes: hardcoding the group query to a fixed odd
        probe size (``_slab_window_group(A, 5)`` instead of
        ``int(n_slab)``) passed every test -- while `_slab_grid`'s own
        docstring calls out that the group is n-dependent and shrinks at
        even n on a cell with a cross term.

        No cell crosses `_SLAB_MIN_GROUP` on parity alone, so this
        asserts the CALL rather than an outcome.
        """
        seen = []
        real = frontend._slab_window_group

        def spy(A, n, **kw):
            seen.append(int(n))
            return real(A, n, **kw)

        frontend._slab_window_group = spy
        try:
            assert self._grid() == 12
        finally:
            frontend._slab_window_group = real
        assert seen == [12], f"group queried at {seen}, not the shipped grid"
        # ...and the group really is n-dependent, which is why it matters.
        cross = np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.3]])
        assert len(real(cross, 11)) == 4
        assert len(real(cross, 12)) == 2

    def test_an_unmeasured_dimension_declines(self):
        for d in (1, 2):
            assert self._grid(d=d, A=np.eye(d)) is None

    def test_a_divergent_sum_keeps_the_box(self):
        # The box is the only engine that refuses instead of returning a
        # confident wrong number.
        assert self._grid(nu_vec=np.full(len(K5), 2.5)) is None
        assert self._grid(nu_vec=np.full(len(K5), 3.0)) is None

    def test_nearest_neighbour_mode_declines(self):
        assert self._grid(nn_mode=True) is None
        assert self._grid(nu_vec=np.full(len(K5), np.inf)) is None

    def test_a_cell_too_poor_in_symmetry_declines_on_cost(self):
        r"""COST, not correctness: the engine is exact on all of these
        (``TestItIsTheSameLatticeSum``).  The measured 3.2-7.4x win
        rests on a 48-fold cubic reduction, and a cell offering 8 or
        fewer would cost 6x the measured time for the same value."""
        assert self._grid(A=np.diag([1.0, 1.0, 1.3])) == 12   # 16 -> ok
        assert self._grid(A=np.diag([1.0, 1.2, 1.4])) is None  # 8
        assert self._grid(A=SHEARED3) is None                  # 1 at n=12
        assert frontend._SLAB_MIN_GROUP == {3: 16}


@pytest.mark.slow
class TestTheArmRoutesEndToEnd:
    r"""K5 through the router: the gate declines it and the slab takes it.

    V6E11, which the split lets the gate admit to the torus, is pinned in
    ``test_dense_torus_accuracy_gate.TestTheShippedEntryEndToEnd``.
    """

    A3 = np.eye(3)

    def test_k5_takes_the_slab_and_the_box_does_not_run(self):
        from gzl import evaluate_graph
        edges = np.array(K5, dtype=int)
        v, info = evaluate_graph(edges, np.full(len(K5), 3.5), self.A3,
                                 n_points=16, richardson=True,
                                 return_diagnostics=True)
        assert info["n_block_dense_torus_declined"] == 1
        assert info["n_block_slab"] == 1
        assert info["n_block_direct_sum"] == 0
        # Against a slab reference at n = 16 (6.35e-08 for the box,
        # 1.29e-08 here; strict resolved bound 4.76x).
        assert float(v) == pytest.approx(5.919613159355887, rel=1e-7)


class TestTheOuterPinIsPriced:
    r"""The finite-k second pin's ENUMERATED vertex is an association
    choice, priced per sector (``_fk_outer_best``), not fixed at the
    terminal.  ``M(gz; x) = M(z; g^-1 x)`` under the window group, so
    the outer sum can enumerate any vertex while the terminal stays
    the open (FFT) axis -- the same torus sum exactly reassociated.
    On the two order-11 census heavies (t-outer inner exponent 3) the
    best alternative prices exponent 2 and 1: 355 s -> 1.5 s and
    429 s -> 1.0 s through the router.  Ties keep the t-outer route
    bit-for-bit.
    """

    # The 355-s order-11 census sector: t-outer prices inner exponent 3,
    # outer=1 prices 2.
    HEAVY = [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (1, 5), (2, 3),
             (2, 5), (3, 4), (3, 5), (4, 5)]

    def test_pricing_finds_the_lever_and_respects_ties(self):
        from gzl.slab import _fk_outer_best, slab_schedule
        _, _, e_t = slab_schedule(self.HEAVY, 0, 4)
        assert e_t == 3
        e_b, u = _fk_outer_best(self.HEAVY, 0, 4)
        assert e_b == 2 and u is not None
        # V6E11 with the hop on two deg-4 vertices: no strictly better
        # outer exists, and the incumbent must be kept (None).
        V6E11 = [(0, 1), (0, 4), (1, 2), (1, 3), (1, 5), (2, 3),
                 (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
        _, tie_u = _fk_outer_best(V6E11, 1, 2)
        assert tie_u is None

    def test_reassociation_is_exact_and_the_fold_is_controlled(
            self, monkeypatch):
        r"""Same value as the t-outer route to round-off, the orbit
        fold against its own unreduced control, and -- the part the
        first shipped version lacked -- a COUNT-based check that the
        two control arms are different code: the reduced leg must
        contract one cell per orbit, the unreduced leg one per site,
        so a silent ``use_symmetry`` no-op (the exact historical
        disease ``test_use_symmetry_false_actually_disables_the_
        reduction`` narrates for the vacuum arm) fails here on the
        call counts even though the values would agree.  Run on the
        cubic cell AND on the triangular d = 2 cell at EVEN n -- the
        prism-lesson parity, where the window group shrinks.
        """
        import gzl.slab as slab_mod
        from gzl.slab import (_slab_zeta_finite_k,
                                     _slab_zeta_finite_k_outer,
                                     _fk_outer_best, orbit_reps)
        from gzl.hybrid import _dense_core as _real_core
        calls = [0]

        def counting_core(*a, **k):
            calls[0] += 1
            return _real_core(*a, **k)

        TRI = np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2.0]])
        for A, n in ((np.eye(3), 4), (TRI, 6)):
            d = A.shape[0]
            nus = np.full(len(self.HEAVY), float(d) + 0.5)
            _, u = _fk_outer_best(self.HEAVY, 0, 4)
            g0, _ = _slab_zeta_finite_k(
                self.HEAVY, nus, A, n, source=0, terminal=4,
                momentum=None)
            with monkeypatch.context() as m:
                m.setattr(slab_mod, "_dense_core", counting_core,
                          raising=False)
                import gzl.hybrid as hyb
                m.setattr(hyb, "_dense_core", counting_core)
                calls[0] = 0
                g1, _ = _slab_zeta_finite_k_outer(
                    self.HEAVY, nus, A, n, source=0, terminal=4,
                    outer=u, momentum=None)
                n_reduced = calls[0]
                calls[0] = 0
                g2, _ = _slab_zeta_finite_k_outer(
                    self.HEAVY, nus, A, n, source=0, terminal=4,
                    outer=u, momentum=None, use_symmetry=False)
                n_unreduced = calls[0]
            reps, _ = orbit_reps(n, d, A)
            assert n_reduced == len(reps)
            assert n_unreduced == n ** d
            assert n_reduced < n_unreduced
            for a, b in ((g0, g1), (g1, g2)):
                rel = np.max(np.abs(np.asarray(a) - np.asarray(b))
                             / np.maximum(np.abs(np.asarray(a)), 1e-300))
                assert rel < 1e-12

    def test_the_shipped_batch_path_has_the_right_phases(self):
        r"""The router always calls the arm with a 2-D momentum batch,
        so the batch-cos branch is the SHIPPED phase path -- and a
        phase bug vanishes at k = 0, the one point the router test
        pins.  Assert the batch values against the independently
        phased FFT grid at every on-grid k (511 of them nonzero).
        """
        from gzl.slab import (_slab_zeta_finite_k_outer,
                                     _fk_outer_best)
        nus = np.full(len(self.HEAVY), 3.5)
        A3 = np.eye(3)
        n = 4
        _, u = _fk_outer_best(self.HEAVY, 0, 4)
        grid, _ = _slab_zeta_finite_k_outer(
            self.HEAVY, nus, A3, n, source=0, terminal=4, outer=u,
            momentum=None)
        axes = [np.arange(n, dtype=float) / n] * 3
        ks = np.stack(np.meshgrid(*axes, indexing="ij"),
                      axis=-1).reshape(-1, 3)
        batch, _ = _slab_zeta_finite_k_outer(
            self.HEAVY, nus, A3, n, source=0, terminal=4, outer=u,
            momentum=ks)
        rel = np.max(np.abs(np.asarray(batch)
                            - np.asarray(grid).reshape(-1))
                     / np.maximum(np.abs(np.asarray(grid).reshape(-1)),
                                  1e-300))
        assert rel < 1e-12

    def test_a_budget_that_is_not_a_budget_is_refused(self):
        from gzl.slab import _slab_zeta_finite_k_outer
        from gzl.slab import SlabTooLargeError
        nus = np.full(len(self.HEAVY), 3.5)
        with pytest.raises(SlabTooLargeError):
            _slab_zeta_finite_k_outer(
                self.HEAVY, nus, np.eye(3), 4, source=0, terminal=4,
                outer=1, momentum=None, max_bytes=float("nan"))

    def test_the_router_takes_it_and_the_value_is_pinned(self):
        r"""The heavy sector through the router at the pass grid: the
        outer-pin arm fires (1.5 s, was 355 s) and the k = 0 cell is
        bit-pinned.
        """
        from gzl import evaluate_graph
        g, info = evaluate_graph(
            np.array(self.HEAVY), 3.5, np.eye(3), source=0, terminal=4,
            n_points=8, return_diagnostics=True)
        assert info["n_block_fk_outer_pin"] == 1
        assert info["n_block_slab_fk_floor"] == 1
        assert info["n_block_direct_sum"] == 0
        assert_pinned(np.asarray(g)[0, 0, 0], "0x1.99a3e38001dcbp+6",
                      "outer-pin arm, k = 0 cell")
        # A k != 0 cell too: the k = 0 pin alone is blind to any phase
        # bug (cos(0) = 1 regardless).
        assert_pinned(np.asarray(g)[1, 2, 3], "-0x1.adc4d0117db10p+1",
                      "outer-pin arm, k != 0 cell")
