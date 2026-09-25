# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

"""Tests for ``gzl.series.compute_series_coefficients`` and
``nu_grid``.

Uses the consolidated corpora so we don't have to fabricate synthetic test
data — they are small enough, and they ship with the package
(``gzl.data_path``), so a missing corpus is an error, not a skip.  The H5
layout does not ship; the NPZ-vs-H5 comparisons rewrite the shipped corpus
in it, in a temporary directory.
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl import data_path
from gzl.series import (
    _parse_momentum,
    compute_series_coefficients,
    evaluate_corpus,
    nu_grid,
)
# The CLI half of the pair below; it lives beside the `gzl series`
# driver, while `_parse_momentum` stays with the library it routes for.
from gzl._cli_series import _parse_momentum_cli


CORPUS_NPZ = data_path("tfim_softcore_corpus_0qp.npz")


@pytest.fixture(scope="module")
def corpus_h5(tmp_path_factory):
    """The shipped 0qp corpus through order 4, rewritten in the per-graph
    subgroup H5 layout that ``tools/consolidate_corpus.py`` writes."""
    import h5py

    from gzl.series import _load_corpus

    path = tmp_path_factory.mktemp("corpus") / "tfim_softcore_corpus_0qp.h5"
    with h5py.File(path, "w") as fh:
        for order, block in _load_corpus(CORPUS_NPZ).items():
            if order > 4:
                continue
            grp = fh.create_group(f"O{order}")
            for i, gid in enumerate(block["graph_id"]):
                g = grp.create_group(str(gid))
                g.create_dataset("edges", data=block["edges_list"][i])
                g.create_dataset("multiplicities",
                                 data=block["multiplicities_list"][i])
                g.create_dataset("prefactor",
                                 data=np.float64(block["prefactor"][i]))
    return path


# ---------------------------------------------------------------------------
# nu_grid helper
# ---------------------------------------------------------------------------

class TestNuGrid:

    def test_uniform(self):
        g = nu_grid(1.0, 3.0, 0.5)
        np.testing.assert_allclose(g, [1.0, 1.5, 2.0, 2.5, 3.0])

    def test_endpoint_inclusive_when_clean(self):
        g = nu_grid(0.1, 1.0, 0.1)
        assert g[-1] == pytest.approx(1.0, abs=1e-12)

    def test_single_point(self):
        # Trivial range: start == end gives a single point even with non-zero step.
        g = nu_grid(2.5, 2.5, 0.1)
        assert g.shape == (1,)
        assert g[0] == pytest.approx(2.5)

    def test_step_must_be_positive(self):
        with pytest.raises(ValueError, match="step"):
            nu_grid(1.0, 2.0, 0.0)
        with pytest.raises(ValueError, match="step"):
            nu_grid(1.0, 2.0, -0.1)

    def test_end_must_be_at_least_start(self):
        with pytest.raises(ValueError, match="end"):
            nu_grid(2.0, 1.0, 0.1)


# ---------------------------------------------------------------------------
# Corpus-driven tests
# ---------------------------------------------------------------------------

class TestCorpusNPZ:
    """Lightweight checks against the consolidated NPZ corpus.  Use small
    n_points and only a few orders so the suite stays fast."""

    A = np.array([[1.0]])

    def test_scalar_nu_has_no_sweep_axis(self):
        """A single coupling gets no sweep axis.

        This asserted ``arr.shape == (1,)`` until the axis was keyed on
        the argument.  A caller who passes one ν did not ask about a
        sweep, so a leading length-1 axis reported the implementation
        rather than the question, and every such caller paid for it
        with a ``[0]``.
        """
        result = compute_series_coefficients(
            CORPUS_NPZ,
            nu=3.0,
            A=self.A,
            n_points=16,
            order_max=4,
        )
        assert set(result.keys()) <= {2, 3, 4}
        for order, value in result.items():
            assert np.shape(value) == (), f"order {order}: {np.shape(value)}"

    def test_a_one_point_sweep_keeps_its_axis(self):
        """``[3.0]`` is a sweep of one, not a scalar.

        Keyed on the argument, never on the resulting length: a rule
        that collapsed ``n_nu == 1`` would silently turn a one-point
        sweep into a scalar, and with it the CLI's whole-Brillouin-zone
        CSV into its three-column form.
        """
        kwargs = dict(A=self.A, n_points=16, order_max=4)
        scalar = compute_series_coefficients(CORPUS_NPZ, nu=3.0, **kwargs)
        for spelling in ([3.0], np.array([3.0]), nu_grid(3.0, 3.0, 1.0)):
            swept = compute_series_coefficients(CORPUS_NPZ, nu=spelling,
                                                **kwargs)
            for order in scalar:
                assert np.shape(swept[order]) == (1,), spelling
                # Same numbers, different shape -- that is the whole
                # difference between the two spellings.
                assert float(swept[order][0]) == float(scalar[order])

    @pytest.mark.parametrize("spelling", [3.0, np.float64(3.0),
                                          np.array(3.0)])
    def test_every_scalar_spelling_is_scalar(self, spelling):
        result = compute_series_coefficients(
            CORPUS_NPZ, nu=spelling, A=self.A, n_points=16, order_max=3)
        for order, value in result.items():
            assert np.shape(value) == (), (spelling, order)

    def test_scalar_nu_on_a_grid_keeps_only_the_k_axes(self):
        """The axis that goes is the sweep, not the momentum grid."""
        result = compute_series_coefficients(
            data_path("tfim_softcore_corpus_1qp.npz"), nu=3.0, A=self.A,
            n_points=8, order_max=2)
        for order, value in result.items():
            assert np.shape(value) == (8,), (order, np.shape(value))

    def test_array_nu_shapes(self):
        nus = nu_grid(3.0, 5.0, 1.0)   # [3.0, 4.0, 5.0]
        result = compute_series_coefficients(
            CORPUS_NPZ,
            nu=nus,
            A=self.A,
            n_points=16,
            order_max=4,
        )
        for order, arr in result.items():
            assert arr.shape == (3,)

    def test_order_max_truncates(self):
        result = compute_series_coefficients(
            CORPUS_NPZ,
            nu=3.0,
            A=self.A,
            n_points=16,
            order_max=3,
        )
        assert all(o <= 3 for o in result)

    def test_diagnostics_returned(self):
        result, diag = compute_series_coefficients(
            CORPUS_NPZ,
            nu=3.0,
            A=self.A,
            n_points=16,
            order_max=4,
            return_diagnostics=True,
        )
        assert "n_graphs_per_order" in diag
        assert "wall_per_order" in diag
        # Order 4 has 3 graphs in the 0qp corpus
        assert diag["n_graphs_per_order"][4] == 3


class TestNPZvsH5Equivalence:
    """The same data under two different layouts must give bit-identical
    results."""

    A = np.array([[1.0]])

    @pytest.mark.parametrize("order_max", [3, 4])
    def test_consistent_per_graph(self, order_max, corpus_h5):
        result_npz = compute_series_coefficients(
            CORPUS_NPZ, nu=3.0, A=self.A, n_points=16, order_max=order_max,
        )
        result_h5 = compute_series_coefficients(
            corpus_h5, nu=3.0, A=self.A, n_points=16, order_max=order_max,
        )
        for order in result_npz:
            np.testing.assert_allclose(
                result_npz[order], result_h5[order],
                rtol=1e-12, atol=1e-15,
                err_msg=f"NPZ vs H5 differ at order {order}",
            )


# ---------------------------------------------------------------------------
# momentum kwarg (BZ-resolved evaluation)
# ---------------------------------------------------------------------------

CORPUS_1QP_NPZ = data_path("tfim_softcore_corpus_1qp.npz")


class TestMomentumKwarg0qp:
    """``momentum`` accepts None / single / batch / 'grid' on the 0qp
    (vacuum) corpus and broadcasts the k-independent scalar across the
    requested trailing shape."""

    A = np.array([[1.0]])
    nu = 3.0
    n_points = 8       # small grid; tests stay sub-second

    def _scalar_at_k0(self, order_max):
        """Reference: ``momentum=None`` (legacy k=0 path)."""
        return compute_series_coefficients(
            CORPUS_NPZ, nu=self.nu, A=self.A, n_points=self.n_points,
            order_max=order_max,
        )

    def test_default_is_scalar_on_0qp(self):
        # 0qp corpus has no `hopping` field → momentum=None defaults
        # to a scalar at k = 0.
        ref = self._scalar_at_k0(order_max=4)
        for order, arr in ref.items():
            assert np.shape(arr) == (), (
                f"0qp default at scalar nu should be a bare scalar; "
                f"got shape {np.shape(arr)}"
            )

    def test_single_zero_matches_none(self):
        ref = self._scalar_at_k0(order_max=4)
        single = compute_series_coefficients(
            CORPUS_NPZ, nu=self.nu, A=self.A, n_points=self.n_points,
            order_max=4, momentum=np.zeros(1),
        )
        for order in ref:
            np.testing.assert_allclose(
                ref[order], single[order], rtol=1e-13, atol=1e-15,
            )

    def test_batch_shape_and_broadcast(self):
        # 0qp is k-independent: every batch entry equals the scalar.
        ref = self._scalar_at_k0(order_max=4)
        ks = np.array([[0.0], [0.25], [0.5]])
        batch = compute_series_coefficients(
            CORPUS_NPZ, nu=self.nu, A=self.A, n_points=self.n_points,
            order_max=4, momentum=ks,
        )
        for order, arr in batch.items():
            assert arr.shape == (3,)
            np.testing.assert_allclose(
                arr, np.full(3, float(ref[order])),
                rtol=1e-13, atol=1e-15,
            )

    def test_invalid_momentum_string(self):
        with pytest.raises(ValueError, match="momentum must be None"):
            compute_series_coefficients(
                CORPUS_NPZ, nu=self.nu, A=self.A, n_points=self.n_points,
                order_max=2, momentum="grid",
            )


class TestMomentumKwarg1qp:
    """1qp corpus exercises real finite-k evaluation.

    ``momentum=None`` on a 1qp corpus now defaults to the full BZ grid
    (matching :func:`evaluate_graph` for ``terminal != source``).  The
    grid's k_index=0 slice must equal the explicit ``momentum=np.zeros(d)``
    scalar; the k_index=n_points//2 slice must equal ``momentum=[0.5]``;
    and a batch of three single-k calls must match the batch path
    entry-by-entry."""

    A = np.array([[1.0]])
    nu = 3.0
    n_points = 8
    # Order ≤ 3: 7 graphs total in the 1qp corpus — keeps the test
    # sub-second while still exercising bridges, simple cycles, and
    # at least one σ-routed block.
    order_max = 3

    def test_default_is_grid_on_1qp(self):
        grid = compute_series_coefficients(
            CORPUS_1QP_NPZ, nu=self.nu, A=self.A, n_points=self.n_points,
            order_max=self.order_max,
        )
        for order, arr in grid.items():
            assert arr.shape == (self.n_points,), (
                f"1qp default at scalar nu should be the bare BZ grid; "
                f"got shape {arr.shape} at order {order}"
            )

    def test_grid_k0_slice_matches_explicit_zero(self):
        grid = compute_series_coefficients(
            CORPUS_1QP_NPZ, nu=self.nu, A=self.A, n_points=self.n_points,
            order_max=self.order_max,
        )
        explicit_zero = compute_series_coefficients(
            CORPUS_1QP_NPZ, nu=self.nu, A=self.A, n_points=self.n_points,
            order_max=self.order_max, momentum=np.zeros(1),
        )
        for order in grid:
            np.testing.assert_allclose(
                grid[order][0], float(explicit_zero[order]),
                rtol=1e-12, atol=1e-13,
                err_msg=f"grid k_index=0 ≠ scalar k=0 at order {order}",
            )

    def test_grid_kpi_slice_matches_single_kpi(self):
        # n_points=8 ⇒ k_index 4 ↔ k_frac=0.5 (k_phys=π).
        grid = compute_series_coefficients(
            CORPUS_1QP_NPZ, nu=self.nu, A=self.A, n_points=self.n_points,
            order_max=self.order_max,
        )
        single = compute_series_coefficients(
            CORPUS_1QP_NPZ, nu=self.nu, A=self.A, n_points=self.n_points,
            order_max=self.order_max, momentum=np.array([0.5]),
        )
        kpi_idx = self.n_points // 2
        for order in grid:
            np.testing.assert_allclose(
                grid[order][kpi_idx], float(single[order]),
                rtol=1e-10, atol=1e-12,
                err_msg=f"grid k_idx={kpi_idx} ≠ single k=0.5 at order {order}",
            )

    def test_batch_matches_single_calls(self):
        ks = [np.array([0.0]), np.array([0.25]), np.array([0.5])]
        singles = [
            compute_series_coefficients(
                CORPUS_1QP_NPZ, nu=self.nu, A=self.A, n_points=self.n_points,
                order_max=self.order_max, momentum=k,
            ) for k in ks
        ]
        batch = compute_series_coefficients(
            CORPUS_1QP_NPZ, nu=self.nu, A=self.A, n_points=self.n_points,
            order_max=self.order_max, momentum=np.stack(ks),
        )
        for order in batch:
            assert batch[order].shape == (3,)
            for j, single in enumerate(singles):
                np.testing.assert_allclose(
                    batch[order][j], float(single[order]),
                    rtol=1e-12, atol=1e-13,
                    err_msg=f"batch[{j}] ≠ single at order {order}",
                )

    def test_default_requires_n_points_on_1qp(self):
        # 1qp default is grid → needs n_points > 0.
        with pytest.raises(ValueError, match="1qp corpus.*n_points"):
            compute_series_coefficients(
                CORPUS_1QP_NPZ, nu=self.nu, A=self.A, n_points=0,
                order_max=self.order_max,
            )


# ---------------------------------------------------------------------------
# CLI momentum parsing (--momentum flag)
# ---------------------------------------------------------------------------

class TestParseMomentumCli:
    """``--momentum <scalar>`` must reach ``_parse_momentum`` as a
    Python scalar: the CLI used to wrap with ``np.asarray``, producing
    a 0-d array that tripped the public-API shape check.

    The two halves now live in different modules -- the parser in
    ``gzl._cli_series``, the router in ``gzl.series`` -- so this is
    also what pins the seam between them.
    """

    def test_scalar_flows_through_parse_momentum_in_d1(self):
        # End-to-end of the bug: parser → router → "single" mode.
        m = _parse_momentum_cli("0.5")
        mode, mom_for_eval, trailing = _parse_momentum(
            m, d=1, n_points=8, has_hopping=True,
        )
        assert mode == "single"
        np.testing.assert_array_equal(mom_for_eval, np.array([0.5]))
        assert trailing == ()

    def test_scalar_rejected_in_d2(self):
        # Adjacent guardrail: the fix touches the int/float branch, so
        # pin that the d=1-only error still fires (and isn't replaced
        # by the old shape-mismatch one).
        m = _parse_momentum_cli("0.5")
        with pytest.raises(ValueError, match="scalar momentum is.*d = 1"):
            _parse_momentum(m, d=2, n_points=8, has_hopping=True)


# ---------------------------------------------------------------------------
# evaluate_corpus — per-graph values; the layer compute_series_coefficients
# sums.
# ---------------------------------------------------------------------------

def _group_sum(records, n_nu):
    """Σ contribution grouped by (order, nu_index) → {order: ndarray(n_nu)}."""
    out: dict[int, np.ndarray] = {}
    for r in records:
        arr = out.setdefault(r["order"], np.zeros(n_nu, dtype=complex))
        arr[r["nu_index"]] += r["contribution"]
    return out


class TestEvaluateCorpus:
    """evaluate_corpus returns the per-graph values; summing them must
    reproduce compute_series_coefficients exactly."""

    A = np.array([[1.0]])
    nus = np.array([2.5, 3.0])
    n_points = 64
    order_max = 5

    def test_record_schema(self):
        recs = evaluate_corpus(CORPUS_NPZ, self.nus, self.A, self.n_points,
                               order_max=self.order_max)
        assert recs, "expected non-empty records"
        assert set(recs[0]) == {
            "order", "graph_id", "nu", "nu_index", "route",
            "prefactor", "value", "contribution", "s", "t",
        }
        # one record per (graph, ν).
        assert len(recs) % self.nus.size == 0

    def test_sum_invariant_matches_coefficients(self):
        """Summing `evaluate_corpus`'s contributions per (order, nu) gives
        `compute_series_coefficients` back, bit for bit (same iteration,
        same block cache).

        This is what makes the two functions one pass seen two ways, and
        it is the reason `compute_series_coefficients` does not also
        return the records: there is one way to ask for them.  Grouping
        on `nu_index` rather than on the value of `nu`, since an
        Interaction sweep can repeat a tail exponent.
        """
        recs = evaluate_corpus(CORPUS_NPZ, self.nus, self.A, self.n_points,
                               order_max=self.order_max)
        coeffs = compute_series_coefficients(
            CORPUS_NPZ, self.nus, self.A, self.n_points,
            order_max=self.order_max,
        )
        summed = _group_sum(recs, self.nus.size)
        assert set(summed) == set(coeffs)
        for order in coeffs:
            np.testing.assert_array_equal(summed[order], coeffs[order])


class TestEvaluateCorpusNN:
    """nu = np.inf turns evaluate_corpus into an exact embedding-factor dump."""

    def test_inf_gives_integer_embedding_factors(self):
        recs = evaluate_corpus(CORPUS_NPZ, np.inf, np.eye(2), 0, order_max=6)
        assert recs
        for r in recs:
            v = complex(r["value"])
            assert np.isfinite(v.real) and abs(v.imag) < 1e-9
            assert r["route"].startswith("topology")
            assert v.real == pytest.approx(round(v.real), abs=1e-6)

    def test_inf_bridge_is_coordination_number(self):
        # The order-2 graph is the single bond (bridge) → square coordination 4.
        recs = evaluate_corpus(CORPUS_NPZ, np.inf, np.eye(2), 0, order_max=2)
        o2 = [r for r in recs if r["order"] == 2]
        assert o2 and o2[0]["value"] == pytest.approx(4.0, abs=1e-9)


class TestEvaluateCorpusNPZvsH5:
    A = np.array([[1.0]])

    def test_per_graph_values_match(self, corpus_h5):
        kw = dict(nu=3.0, A=self.A, n_points=16, order_max=4)
        rn = evaluate_corpus(CORPUS_NPZ, **kw)
        rh = evaluate_corpus(corpus_h5, **kw)
        assert len(rn) == len(rh)
        for a, b in zip(rn, rh):
            assert a["order"] == b["order"] and a["graph_id"] == b["graph_id"]
            np.testing.assert_allclose(a["value"], b["value"],
                                       rtol=1e-10, atol=1e-12)
