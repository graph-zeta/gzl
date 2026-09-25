# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The values a lattice sum returns are real, and the collapse is checked.

A graph zeta is real when the lattice has one site per cell and the
kernels are real and even: the inversion x -> -x maps every
configuration to its own conjugate.  The engines still compute in
complex arithmetic, so somewhere the imaginary part has to go.  These
tests pin WHERE that happens, that it is checked rather than assumed,
and that the values themselves did not move.
"""

from __future__ import annotations

import numpy as np
import pytest

import gzl.series as series_mod
from gzl import (
    compute_series_coefficients,
    direct_sum_extrapolated,
    direct_sum_zero_momentum,
    evaluate_corpus,
    evaluate_graph,
    graph_zeta_general_at_zero,
    hybrid_zeta,
    slab_zeta,
    zeta_circle,
)
from gzl._real import _REAL_RTOL, _as_real

TRIANGLE = np.array([[0, 1], [1, 2], [0, 2]])
NU3 = np.full(3, 3.5)
A1 = np.array([[1.0]])


class TestTheRule:
    """:func:`gzl._real._as_real` itself."""

    def test_a_real_scalar_passes_through_as_a_float(self):
        out = _as_real(2.5, where="t")
        assert isinstance(out, float) and out == 2.5

    def test_a_real_array_passes_through_bit_for_bit(self):
        a = np.array([1.5, -0.25, 1e-300])
        out = _as_real(a, where="t")
        assert out.dtype == np.float64
        assert out.tobytes() == a.tobytes()

    def test_a_zero_imaginary_part_is_dropped_bit_for_bit(self):
        a = np.array([1.5 + 0j, -0.25 + 0j])
        out = _as_real(a, where="t")
        assert out.dtype == np.float64
        assert out.tobytes() == np.array([1.5, -0.25]).tobytes()

    def test_round_off_below_the_bound_is_dropped(self):
        value = 1.0 + 1e-12j
        assert _as_real(value, where="t") == 1.0

    def test_an_imaginary_part_above_the_bound_raises_and_says_why(self):
        with pytest.raises(ValueError, match="not real"):
            _as_real(1.0 + 1e-3j, where="a_function_name")
        with pytest.raises(ValueError, match="a_function_name"):
            _as_real(1.0 + 1e-3j, where="a_function_name")
        with pytest.raises(ValueError, match="even"):
            _as_real(1.0 + 1e-3j, where="t")

    def test_the_bound_is_relative_to_the_result_scale(self):
        # The same absolute imaginary part: kept beside a large real
        # part, refused beside a small one.
        assert _as_real(1e6 + 1e-3j, where="t") == 1e6
        with pytest.raises(ValueError):
            _as_real(1e-6 + 1e-3j, where="t")

    def test_the_scale_of_an_array_is_the_whole_array(self):
        # One large entry licenses round-off in the small ones; that is
        # the same tolerance the engines' own cancellation obeys.
        a = np.array([1e6 + 0j, 1e-9 + 1e-3j])
        assert _as_real(a, where="t")[1] == 1e-9
        with pytest.raises(ValueError):
            _as_real(np.array([1e-9 + 1e-3j]), where="t")

    def test_the_bound_is_the_documented_one(self):
        assert _REAL_RTOL == 1e-8

    def test_a_nan_imaginary_part_raises_rather_than_slipping_through(self):
        # Every comparison against NaN is False, so a ``>`` test would
        # have dropped this one silently.
        with pytest.raises(ValueError):
            _as_real(complex(1.0, float("nan")), where="t")
        with pytest.raises(ValueError):
            _as_real(np.array([complex(np.nan, 0.0), complex(1.0, 1e3)]),
                     where="t")

    def test_an_infinite_real_part_does_not_license_everything(self):
        # inf in the scale would make the bound inf and switch the
        # guard off for every other entry.
        with pytest.raises(ValueError):
            _as_real(np.array([complex(np.inf, 0.0), complex(1.0, 1e3)]),
                     where="t")

    def test_a_float64_array_is_not_copied(self):
        a = np.array([1.5, 2.5])
        assert _as_real(a, where="t") is a

    def test_the_result_is_float64_whatever_the_complex_width(self):
        out = _as_real(np.array([1 + 0j], dtype=np.complex64), where="t")
        assert out.dtype == np.float64


class TestTheFourReturns:
    """The entry points that used to return ``complex``."""

    def test_graph_zeta_general_at_zero_is_a_float(self):
        v = graph_zeta_general_at_zero(TRIANGLE, NU3, A1, 8)
        assert type(v) is float

    def test_direct_sum_zero_momentum_is_a_float(self):
        v = direct_sum_zero_momentum(TRIANGLE, NU3, A1, 6)
        assert type(v) is float

    def test_direct_sum_extrapolated_is_a_float(self):
        v = direct_sum_extrapolated(TRIANGLE, NU3, A1, L_list=(4, 5, 6, 7))
        assert type(v) is float

    @pytest.mark.parametrize("corpus,kwargs", [
        ("tfim0qp", {}),
        ("tfim1qp", {"momentum": 0.0}),
        ("tfim1qp", {}),                      # the full BZ grid
    ])
    def test_the_series_coefficients_are_float64(self, corpus, kwargs):
        c = compute_series_coefficients(corpus, nu=3.5, A="chain",
                                        n_points=8, order_max=3, **kwargs)
        for arr in c.values():
            assert arr.dtype == np.float64

    def test_the_rest_of_the_surface_was_already_real(self):
        assert type(evaluate_graph(TRIANGLE, 3.5, "chain")) is float
        assert type(zeta_circle(NU3, A1)) is float
        assert type(hybrid_zeta(TRIANGLE, NU3, A1, 8)) is float
        assert type(slab_zeta(TRIANGLE, NU3, A1, 8)[0]) is float
        rec = evaluate_corpus("tfim0qp", nu=3.5, A="chain",
                              n_points=8, order_max=2)[0]
        assert type(rec["value"]) is float
        assert type(rec["contribution"]) is float


class TestValuesDidNotMove:
    """Dropping a zero imaginary part cannot change a number."""

    def test_the_box_agrees_with_the_torus_as_before(self):
        # The two reference families still meet where they always did:
        # measured 4.5e-9 of scale apart here, which is their own
        # truncation difference, not anything this change touches.
        box = direct_sum_extrapolated(TRIANGLE, NU3, A1, L_list=(6, 7, 8, 9))
        torus = graph_zeta_general_at_zero(TRIANGLE, NU3, A1, 64)
        assert abs(box - torus) <= 1e-7 * abs(box)

    def test_a_pass_still_sums_its_own_per_graph_records(self):
        kw = dict(nu=3.5, A="chain", n_points=8, order_max=3)
        coeffs = compute_series_coefficients("tfim0qp", **kw)
        records = evaluate_corpus("tfim0qp", **kw)
        for order, value in coeffs.items():
            # Scalar nu, k = 0 on a vacuum corpus: `value` IS the
            # coefficient, with no sweep axis to index past.
            total = sum(r["contribution"] for r in records
                        if r["order"] == order and r["nu_index"] == 0)
            assert float(value) == pytest.approx(total, rel=1e-13, abs=1e-13)


class TestTheGuardFires:
    """A complex value reaching the accumulator is refused, not truncated."""

    @pytest.mark.parametrize("momentum,shape", [
        (None, "scalar"),                               # trailing_shape ()
        (np.array([[0.0], [0.25], [0.5]]), "batch"),    # trailing_shape (3,)
    ])
    def test_a_complex_contribution_raises_out_of_the_pass(
            self, monkeypatch, momentum, shape):
        # The failure mode this guard exists for: half the answer gone,
        # no error, a plausible number left behind.  The batch path
        # builds its per-graph array with np.full, which DROPS an
        # imaginary part on a warning rather than raising, so it has to
        # go through the guard first.
        monkeypatch.setattr(series_mod, "evaluate_graph",
                            lambda *a, **k: 1.0 + 0.5j)
        with pytest.raises(ValueError, match="compute_series_coefficients: "
                                             "the value is not real"):
            compute_series_coefficients("tfim0qp", nu=3.5, A="chain",
                                        n_points=8, order_max=2,
                                        momentum=momentum)
