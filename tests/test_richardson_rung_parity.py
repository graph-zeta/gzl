# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Richardson ladder rungs must share the parity of the top rung.

The balanced-z label axis is symmetric for odd ``n`` and asymmetric for
even ``n`` (the label ``n/2`` has no partner ``-n/2``), which splits the
truncation coefficient into two families.  A ladder mixing both fits a
single power law through two different constants and degrades badly:
``_richardson_rungs(64, 1)`` used to return ``[64, 48, 36, 27]``, and 64
is the production d=1 ``n_points``.
"""
import numpy as np
import pytest
from epsteinlib import epstein_zeta

from gzl import graph_zeta_general_at_zero
from gzl.frontend import _richardson_extrapolate, _richardson_rungs


class TestRungParity:

    @pytest.mark.parametrize("d", [1, 2])
    def test_all_rungs_share_parity(self, d):
        for n_top in range(8, 257):
            rungs = _richardson_rungs(n_top, d)
            assert len({m % 2 for m in rungs}) <= 1, (n_top, d, rungs)
            assert all(m % 2 == n_top % 2 for m in rungs), (n_top, d, rungs)

    def test_rungs_stay_descending_and_distinct(self):
        for d in (1, 2):
            for n_top in range(8, 257):
                r = _richardson_rungs(n_top, d)
                assert r == sorted(set(r), reverse=True), (n_top, d, r)

    def test_production_d1_rung_set(self):
        """n_points=64 is the production d=1 setting."""
        assert _richardson_rungs(64, 1) == [64, 48, 36, 28]


class TestParityGainAgainstExactTruth:
    """A bridge has an exact closed form, so this has no reference noise."""

    @pytest.mark.parametrize("sigma", [0.75, 0.5, 0.25])
    def test_bridge_ladder_beats_mixed_parity(self, sigma):
        A, zero = np.eye(1), np.zeros(1)
        nu = 1.0 + sigma
        E, nv = np.array([(0, 1)]), np.array([nu])
        exact = float(np.real(epstein_zeta(nu, A, zero, zero)))

        def lad(rungs):
            vals = [float(np.real(graph_zeta_general_at_zero(E, nv, A, n)))
                    for n in rungs]
            return _richardson_extrapolate(rungs, vals, sigma)

        err = lambda v: abs(v - exact) / abs(exact)
        mixed = err(lad([64, 48, 36, 27]))     # the old, pre-fix set
        clean = err(lad(_richardson_rungs(64, 1)))
        assert clean < mixed / 100.0, (clean, mixed)
        assert clean < 1e-6


class TestTheRungConditionalStabilityTolerance:
    r"""``_RICHARDSON_STABILITY`` is rung-conditional, and the condition
    is load-bearing.

    A flat 0.1 is wrong as a *test*, not merely as a calibration: on a
    sequence lying EXACTLY in the fitted basis the drop-one refit gap
    grows with the subleading amplitude while the fit stays exact, so
    0.1 rejects any exactly-fitted two-term tail with ``b/a >~ 8``.
    Measured over 596 ladders the rung-conditional form accepts 69 more
    good ladders and **zero** harmful ones; a flat 0.5 admits one at
    18.86x worse than the raw torus, on a ladder reaching down to rung 4.
    """

    def test_the_constants_are_the_measured_ones(self):
        from gzl import frontend
        assert frontend._RICHARDSON_STABILITY == 0.1
        assert frontend._RICHARDSON_STABILITY_DEEP == 0.5
        assert frontend._RICHARDSON_STABILITY_RUNG == 6

    def test_a_deep_ladder_gets_the_loose_tolerance(self):
        r"""Constructed so the drop-one gap lands between 0.1 and 0.5:
        accepted now, ``unstable`` under the old flat bound.
        """
        from gzl import frontend
        import numpy as np

        p = 11.0
        rungs = [12, 10, 8]

        def seq(b_over_a):
            return [1.0 - n ** (-p) - b_over_a * n ** (-(p + 1))
                    for n in rungs]

        # b/a = 22 puts the stability ratio at ~0.149 (see the constant's
        # own table), i.e. inside (0.1, 0.5).
        vals = seq(22.0)
        got, why = frontend._richardson_ladder(
            lambda n: vals[rungs.index(n)], 12, 3, p, vals[0])
        assert why == "ok"
        assert got == pytest.approx(1.0, abs=1e-12)

        old = frontend._RICHARDSON_STABILITY_DEEP
        try:
            frontend._RICHARDSON_STABILITY_DEEP = frontend._RICHARDSON_STABILITY
            _, why_flat = frontend._richardson_ladder(
                lambda n: vals[rungs.index(n)], 12, 3, p, vals[0])
        finally:
            frontend._RICHARDSON_STABILITY_DEEP = old
        assert why_flat == "unstable"

    def test_a_shallow_ladder_keeps_the_tight_tolerance(self, monkeypatch):
        r"""The rung condition is what stops a flat 0.5 admitting an
        18.86x regression: a ladder reaching below rung 6 is not
        asymptotic, and its drop-one gap is measuring that.

        The construction of the deep ladder above on the rungs [8, 6, 4]
        of ``n_top = 8`` at d = 3: the drop-one gap is 0.255 of the
        correction, inside (0.1, 0.5).  Refused as it stands, and
        accepted once the rung condition is moved below the lowest rung.
        """
        from gzl import frontend

        p = 11.0
        rungs = frontend._richardson_rungs(8, 3)
        assert rungs == [8, 6, 4]
        assert min(rungs) < frontend._RICHARDSON_STABILITY_RUNG
        vals = [1.0 - n ** (-p) - 22.0 * n ** (-(p + 1)) for n in rungs]

        _, why = frontend._richardson_ladder(
            lambda n: vals[rungs.index(n)], 8, 3, p, vals[0])
        assert why == "unstable"

        monkeypatch.setattr(frontend, "_RICHARDSON_STABILITY_RUNG", min(rungs))
        got, why_deep = frontend._richardson_ladder(
            lambda n: vals[rungs.index(n)], 8, 3, p, vals[0])
        assert why_deep == "ok"
        assert got == pytest.approx(1.0, abs=1e-12)
