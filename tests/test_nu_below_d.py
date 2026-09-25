# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""Exponents at or below the dimension: one rule at the front door.

A bridge keeps its value, the meromorphic continuation of its Epstein
zeta function, except at its pole.  Every other block with a bundle
exponent ``nu <= d`` raises :class:`gzl.UnsupportedLatticeSumError`, at
every momentum, under every engine setting and in the legacy
nearest-neighbour mode.

Before this rule the same request had four outcomes depending on the
route: the bridge continuation, a plain ``ValueError`` from the cycle
closed form or from the algebra, ``UnsupportedLatticeSumError`` from the
dense guard, and silent numbers on the routes around them (a
treewidth-2 block with three attachments on the torus, any block in
nearest-neighbour mode, the bridge pole as NaN, which the shipped
``tfim0qp`` corpus reached at ``nu = 0.5``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import gzl
from gzl import Interaction, UnsupportedLatticeSumError, evaluate_graph

#: 2 zeta(0.9), the continuation of the chain's bridge, from mpmath.
BRIDGE_CHAIN_09 = -18.8602280388045

BRIDGE = [[0, 1]]
PATH = [[0, 1], [1, 2]]
TRIANGLE = [[0, 1], [1, 2], [2, 0]]
DIAMOND = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]]
K4 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
# A treewidth-2 block with three attachments, which the router sends to
# the torus rather than to the algebra.  It returned 292.5 / 867.5 /
# 886.5 at n_points = 8 / 16 / 32 before the rule.
DIAMOND_3PENDANTS = DIAMOND + [[0, 4], [1, 5], [2, 6]]
TRIANGLE_3PENDANTS = TRIANGLE + [[0, 3], [1, 4], [2, 5]]


def _ev(edges, nu, A="chain", **kw):
    kw.setdefault("n_points", 8)
    return evaluate_graph(np.array(edges), nu, A, **kw)


# ---------------------------------------------------------------------------
# A bridge keeps its value
# ---------------------------------------------------------------------------

class TestBridgeKeepsItsValue:

    def test_the_chain_bridge_is_the_zeta_continuation(self):
        assert _ev(BRIDGE, 0.9) == pytest.approx(BRIDGE_CHAIN_09, rel=1e-13)

    def test_a_path_is_the_square_of_its_bridge(self):
        assert _ev(PATH, 0.9) == pytest.approx(BRIDGE_CHAIN_09 ** 2,
                                               rel=1e-13)

    def test_at_the_dimension_away_from_the_pole(self):
        # sum_{x != 0} exp(-2 pi i k x) / |x| = -2 ln|2 sin(pi k)|,
        # which is -ln 2 at k = 1/4.
        v = _ev(BRIDGE, 1.0, terminal=1, momentum=0.25)
        assert v == pytest.approx(-math.log(2.0), rel=1e-12)

    def test_in_legacy_nearest_neighbour_mode(self):
        # One nu = inf edge puts the call in nearest-neighbour mode, which
        # sent the bridge to the torus: 8.21 / -37.51 / -37.72 at
        # n_points = 8 / 16 / 32.  The nearest-neighbour factor of the
        # chain is 2.
        v = _ev(PATH, [np.inf, 0.9])
        assert v == pytest.approx(2.0 * BRIDGE_CHAIN_09, rel=1e-13)

    def test_a_bundle_is_the_sum_of_its_edges(self):
        doubled = [e for e in TRIANGLE for _ in range(2)]
        assert _ev(doubled, 0.6) == _ev(TRIANGLE, 1.2)

    @pytest.mark.parametrize("nu", [
        0.5, 2.5,
        # Interaction.lattice_sum carries its own copy of the reduction
        # (gzl/interaction.py); without it (1, 0) gave 5.0e24 here.
        Interaction(b=[1.0, 1.0], nu=[0.5, 2.5]),
    ], ids=["0.5", "2.5", "interaction"])
    def test_an_integer_momentum_on_the_triangular_lattice(self, nu):
        # inv(A.T) @ (1, 0) misses the reciprocal lattice by about 1e-17
        # on the triangular cell, and the closed form returned the cusp
        # there: relative errors of 2.5e24 at nu = 0.5, 6.8e-9 at 2.5.
        at = lambda k: _ev(BRIDGE, nu, A="triangular", terminal=1,  # noqa: E731
                           momentum=np.array(k, dtype=float))
        assert at([1.0, 0.0]) == at([0.0, 0.0])
        assert at([-0.75, 0.25]) == at([0.25, 0.25])


# ---------------------------------------------------------------------------
# ... except at its pole
# ---------------------------------------------------------------------------

class TestTheBridgePoleRaises:

    @pytest.mark.parametrize("A, d", [("chain", 1), ("square", 2),
                                      ("triangular", 2), ("cubic", 3)])
    def test_vacuum(self, A, d):
        with pytest.raises(UnsupportedLatticeSumError, match="pole"):
            _ev(BRIDGE, float(d), A=A)

    @pytest.mark.parametrize("momentum", [None, 0.0, 1.0,
                                          np.array([0.0, 0.25])])
    def test_every_request_that_contains_the_pole(self, momentum):
        # None is the full grid, whose cell k = 0 is the pole; a batch is
        # refused as a whole for the same reason.
        with pytest.raises(UnsupportedLatticeSumError):
            _ev(BRIDGE, 1.0, terminal=1, momentum=momentum)

    def test_an_off_spine_bridge_sees_k_zero(self):
        # Source 1 and terminal 2: the bridge 0-1 hangs off the spine and
        # contributes its value at k = 0, whatever k is asked.
        with pytest.raises(UnsupportedLatticeSumError):
            _ev(PATH, [1.0, 2.5], source=1, terminal=2, momentum=0.25)

    @pytest.mark.parametrize("nu", [1.0 + 5e-10, 1.0 - 5e-10])
    def test_inside_epsteinlibs_window_around_the_pole(self, nu):
        # epsteinlib returns NaN within 2**-30 of nu = d.
        with pytest.raises(UnsupportedLatticeSumError):
            _ev(BRIDGE, nu)

    def test_in_legacy_nearest_neighbour_mode(self):
        # On the torus this returned finite numbers, 7.83 / 6.88 / 8.91.
        with pytest.raises(UnsupportedLatticeSumError):
            _ev(PATH, [np.inf, 1.0])

    @pytest.mark.parametrize("request_kw", [
        {}, {"terminal": 1}, {"terminal": 1, "momentum": 0.0},
        {"terminal": 1, "momentum": 1.0},
        {"terminal": 1, "momentum": np.array([0.0, 0.25])},
    ], ids=["vacuum", "grid", "k=0", "k=1", "batch"])
    def test_an_interaction_term_below_the_tail(self, request_kw):
        # Tail exponent 0.5, but the |x|^-1 term sits at the pole.
        with pytest.raises(UnsupportedLatticeSumError, match="pole"):
            _ev(BRIDGE, Interaction(b=[1.0, 1.0], nu=[0.5, 1.0]),
                **request_kw)

    def test_away_from_the_pole_the_interaction_keeps_its_value(self):
        v = _ev(BRIDGE, Interaction(b=[1.0, 1.0], nu=[0.5, 1.0]),
                terminal=1, momentum=0.25)
        assert np.isfinite(v)

    @pytest.mark.parametrize("nu", [344.0, 400.0])
    def test_far_from_the_pole_a_non_finite_value_is_not_unsupported(self,
                                                                      nu):
        # epsteinlib returns NaN on the unit chain from nu = 344 on, where
        # 2 zeta(nu) is about 2.  That is its float64 range, not the pole.
        with pytest.raises(gzl.GraphZetaError) as exc:
            _ev(BRIDGE, nu)
        assert not isinstance(exc.value, UnsupportedLatticeSumError)

    def test_on_the_triangular_lattice_at_an_integer_momentum(self):
        # Returned a finite 273.34 before the momentum was reduced.
        with pytest.raises(UnsupportedLatticeSumError):
            _ev(BRIDGE, 2.0, A="triangular", terminal=1,
                momentum=np.array([1.0, 0.0]))

    def test_a_pole_is_never_cached(self):
        cache = {}
        for _ in range(2):
            with pytest.raises(UnsupportedLatticeSumError):
                _ev(BRIDGE, 1.0, block_cache=cache)
        assert not any(isinstance(v, float) and math.isnan(v)
                       for v in _cache_values(cache))


def _cache_values(cache):
    for entry in cache.values():
        for item in (entry if isinstance(entry, list) else [entry]):
            yield item[-1] if isinstance(item, tuple) else item


# ---------------------------------------------------------------------------
# Every other block raises
# ---------------------------------------------------------------------------

BLOCKS = {
    "triangle": TRIANGLE,
    "diamond": DIAMOND,
    "K4": K4,
    "diamond with three attachments": DIAMOND_3PENDANTS,
}
REQUESTS = {
    "vacuum": {},
    "grid": {"terminal": 1},
    "single k": {"terminal": 1, "momentum": 0.25},
    "batch": {"terminal": 1, "momentum": np.array([0.0, 0.25])},
}


class TestEveryOtherBlockRaises:

    @pytest.mark.parametrize("request_kw", REQUESTS.values(),
                             ids=REQUESTS.keys())
    @pytest.mark.parametrize("edges", BLOCKS.values(), ids=BLOCKS.keys())
    @pytest.mark.parametrize("nu", [0.9, 1.0])
    def test_at_every_momentum(self, edges, nu, request_kw):
        with pytest.raises(UnsupportedLatticeSumError, match="bridges only"):
            _ev(edges, nu, **request_kw)

    @pytest.mark.parametrize("kw", [
        {"dense_engine": "torus"}, {"dense_engine": "direct_sum"},
        {"engine": "tensor"}, {"richardson": False}, {"fast_cycles": True},
        {"accuracy": "floor"}, {"n_points": 0},
    ], ids=lambda kw: "-".join(f"{k}={v}" for k, v in kw.items()))
    @pytest.mark.parametrize("edges", [
        K4, TRIANGLE_3PENDANTS, DIAMOND_3PENDANTS,
    ], ids=["K4", "triangle with three attachments",
            "diamond with three attachments"])
    def test_under_every_setting(self, edges, kw):
        # ``match``: refused by the rule before routing, not by whichever
        # engine guard the route happens to reach.
        with pytest.raises(UnsupportedLatticeSumError, match="bridges only"):
            _ev(edges, 0.9, **kw)

    def test_in_legacy_nearest_neighbour_mode(self):
        # One nu = inf pendant skipped the dense guard: 12.28 / 21.37.
        with pytest.raises(UnsupportedLatticeSumError, match="bridges only"):
            _ev(K4 + [[3, 4]], [0.9] * 6 + [np.inf])

    def test_one_low_bundle_is_enough(self):
        with pytest.raises(UnsupportedLatticeSumError, match="bridges only"):
            _ev(TRIANGLE, [0.9, 2.5, 2.5])

    def test_an_interaction_tail(self):
        with pytest.raises(UnsupportedLatticeSumError, match="bridges only"):
            _ev(TRIANGLE, Interaction(b=[2.0], nu=[0.9]))

    def test_a_compact_kernel_has_no_tail_and_is_evaluated(self):
        # The homomorphism count of a triangle on the triangular lattice.
        v = _ev(TRIANGLE, Interaction.nearest_neighbour("triangular"),
                A="triangular")
        assert v == pytest.approx(12.0, rel=1e-12)

    def test_the_cycle_closed_form(self):
        with pytest.raises(UnsupportedLatticeSumError):
            gzl.zeta_circle(np.full(3, 0.9), "chain")

    def test_the_algebra_constructor(self):
        with pytest.raises(UnsupportedLatticeSumError):
            gzl.graph_from_edges(np.array(TRIANGLE), np.full(3, 0.9),
                                 "chain", 16)

    def test_the_box(self):
        with pytest.raises(UnsupportedLatticeSumError):
            gzl.direct_sum_extrapolated(np.array(K4), np.full(6, 0.9),
                                        "chain")


# ---------------------------------------------------------------------------
# A NaN is a malformed argument, not an unsupported exponent
# ---------------------------------------------------------------------------

class TestNaN:

    @pytest.mark.parametrize("edges", [BRIDGE, TRIANGLE, K4],
                             ids=["bridge", "triangle", "K4"])
    def test_evaluate_graph(self, edges):
        with pytest.raises(ValueError, match=r"number or \+inf") as exc:
            _ev(edges, np.nan)
        assert not isinstance(exc.value, UnsupportedLatticeSumError)

    def test_zeta_circle(self):
        with pytest.raises(ValueError, match="NaN") as exc:
            gzl.zeta_circle(np.array([np.nan, 2.5, 2.5]), "chain")
        assert not isinstance(exc.value, UnsupportedLatticeSumError)

    @pytest.mark.parametrize("position", [0, 1])
    def test_the_box_in_any_edge_order(self, position):
        # min() over Python floats returns NaN only when NaN comes first.
        nu = [2.5] * 6
        nu[position] = np.nan
        with pytest.raises(ValueError, match="NaN") as exc:
            gzl.direct_sum_extrapolated(np.array(K4), nu, "chain",
                                        L_list=(2, 3, 4))
        assert not isinstance(exc.value, UnsupportedLatticeSumError)

    @pytest.mark.parametrize("nu", [-np.inf, [np.inf, -np.inf]],
                             ids=["-inf", "inf and -inf in parallel"])
    def test_minus_infinity(self, nu):
        # A -inf bridge returned the nu = +inf count, 2.0 on the chain,
        # and an inf | -inf bundle a NaN.
        edges = BRIDGE if np.isscalar(nu) else BRIDGE * 2
        with pytest.raises(ValueError, match=r"\+inf") as exc:
            _ev(edges, nu)
        assert not isinstance(exc.value, UnsupportedLatticeSumError)


# ---------------------------------------------------------------------------
# The corpus front-ends and the CLI
# ---------------------------------------------------------------------------

class TestCorpus:

    def test_the_shipped_corpus_at_a_pole_raises(self):
        # The order-2 graph is one edge of multiplicity 2, a bundle of
        # 2 nu = 1.0 = d.  This returned {2: nan}.
        with pytest.raises(UnsupportedLatticeSumError):
            gzl.compute_series_coefficients("tfim0qp", 0.5, "chain", 8,
                                            order_max=2)

    def test_the_shipped_corpus_keeps_a_bridge_below_d(self):
        # One bundle of 2 x 0.45 = 0.9 with prefactor -1/4.
        res = gzl.compute_series_coefficients("tfim0qp", 0.45, "chain", 8,
                                              order_max=2)
        assert res[2] == pytest.approx(-0.25 * BRIDGE_CHAIN_09, rel=1e-13)

    @pytest.mark.parametrize("per_graph", [False, True],
                             ids=["coefficients", "per-graph"])
    def test_the_cli_reports_it_in_one_line(self, tmp_path, capsys,
                                            per_graph):
        from gzl import cli
        argv = ["series", "--corpus", "tfim0qp", "--A", "chain",
                "--n-points", "8", "--order-max", "2", "--nu", "0.5",
                "--output", str(tmp_path / "o.csv")]
        assert cli.main(argv + (["--per-graph"] if per_graph else [])) == 1
        err = capsys.readouterr().err
        assert "gzl: error:" in err and "pole" in err
        assert "Traceback" not in err

    def test_per_graph_reports_an_unwritable_output_in_one_line(self,
                                                                 capsys):
        from gzl import cli
        argv = ["series", "--corpus", "tfim0qp", "--A", "chain",
                "--n-points", "8", "--order-max", "2", "--nu", "3.0",
                "--per-graph", "--output", "/nonexistent-root/o.csv"]
        assert cli.main(argv) == 1
        err = capsys.readouterr().err
        assert "gzl: error:" in err and "Traceback" not in err

    def test_per_graph_forwards_the_routing_settings(self, tmp_path,
                                                     monkeypatch, capsys):
        # --per-graph used to pass on only nu_tensor_threshold and
        # sigma_max, and dropped the rest silently.  high_tw_fallback
        # tuned the removed k = 0 fallback cascade: it is accepted,
        # reported, and not passed on.
        from gzl import _cli_series, cli
        seen = {}

        def fake(*args, **kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr(_cli_series, "evaluate_corpus", fake)
        config = tmp_path / "c.toml"
        config.write_text(
            '[corpus]\npath = "tfim0qp"\n'
            '[lattice]\nA = "chain"\n'
            '[evaluation]\nn_points = 8\norder_max = 2\n'
            'nu_start = 3.0\nnu_end = 3.0\nnu_step = 1.0\n'
            '[routing]\nhigh_tw_fallback = "direct_sum"\n')
        argv = ["series", "--config", str(config),
                "--per-graph", "--dense-engine", "torus",
                "--sp-n-points", "12", "--no-core-grading",
                "--output", str(tmp_path / "o.csv")]
        assert cli.main(argv) == 0
        assert seen["dense_engine"] == "torus"
        assert seen["sp_n_points"] == 12
        assert seen["core_grading"] is False
        assert "high_tw_fallback" not in seen
        err = capsys.readouterr().err
        assert "gzl: warning: high_tw_fallback has no effect" in err
