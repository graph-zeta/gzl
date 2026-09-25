# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""A cache key must name the EVALUATOR, not just the block.

``_momentum_cache_key`` collapses ``k = 0`` to ``None`` because the exact
``zeta_block`` is momentum-independent there.  But an on-spine block is
dispatched through ``_block_at_finite_k`` whatever the momentum, and that
routine uses a closed form only for BRIDGES (``epstein_zeta``, exact at
any k).  For a CYCLE it deliberately falls through to the sigma-routed
algebra/tensor evaluator, because ``zeta_circle`` is k = 0 only.

Storing that discretisation-carrying value under the closed-form
``("cycle", ...)`` key poisoned the entry for every later vacuum or
off-spine reuse of the same cycle, so a shared-cache corpus pass depended
on the ORDER the graphs were visited in -- 2.5e-4 at n_points = 16 and
4.7e-3 at n_points = 8, reachable straight from the documented
``momentum=np.zeros(d)`` call on a 1qp corpus.

``zeta_circle`` is a closed form, so these assertions carry no reference
uncertainty.
"""
import numpy as np
import pytest

from gzl import evaluate_graph
from gzl.circle import zeta_circle

A1 = np.eye(1)
NU = 2.75
TRIANGLE = np.array([[0, 1], [1, 2], [2, 0]])
SQUARE = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])


def _vacuum(edges, cache, n):
    return evaluate_graph(edges, NU, A1, source=0, terminal=0,
                          n_points=n, block_cache=cache)


def _onqp_at_k0(edges, cache, n, terminal):
    return evaluate_graph(edges, NU, A1, source=0, terminal=terminal,
                          momentum=[0.0], n_points=n, block_cache=cache)


class TestSpineCycleDoesNotPoisonTheClosedFormKey:

    @pytest.mark.parametrize("n_points", [8, 16, 32])
    def test_vacuum_cycle_keeps_its_closed_form_value(self, n_points):
        """A prior on-spine k=0 visit must not downgrade the closed form."""
        exact = float(np.real(zeta_circle(np.array([NU] * 3), A1)))
        cache = {}
        _onqp_at_k0(TRIANGLE, cache, n_points, terminal=2)   # poisons, pre-fix
        got = _vacuum(TRIANGLE, cache, n_points)
        assert got == pytest.approx(exact, rel=1e-12)

    @pytest.mark.parametrize("edges,length", [(TRIANGLE, 3), (SQUARE, 4)])
    @pytest.mark.parametrize("n_points", [8, 16])
    def test_shared_cache_is_order_independent(self, edges, length, n_points):
        """Same corpus, same parameters, two visit orders, one answer."""
        c = {}
        a_1qp = _onqp_at_k0(edges, c, n_points, terminal=length - 1)
        a_vac = _vacuum(edges, c, n_points)

        c = {}
        b_vac = _vacuum(edges, c, n_points)
        b_1qp = _onqp_at_k0(edges, c, n_points, terminal=length - 1)

        assert a_vac == pytest.approx(b_vac, rel=1e-14), "vacuum is order-dependent"
        assert a_1qp == pytest.approx(b_1qp, rel=1e-14), "1qp is order-dependent"

    def test_cached_equals_uncached_for_both_roles(self):
        """The shared-cache value must equal a fresh evaluation."""
        n = 16
        c = {}
        _onqp_at_k0(TRIANGLE, c, n, terminal=2)
        cached = _vacuum(TRIANGLE, c, n)
        fresh = _vacuum(TRIANGLE, None, n)
        assert cached == pytest.approx(fresh, rel=1e-14)

    def test_bridge_closed_form_key_is_still_shared(self):
        """Bridges ARE closed-form at any k, so their key must still hit.

        Guards against over-correcting: the fix must not stop bridges
        from sharing the epstein_zeta entry across spine/off-spine roles.
        """
        n = 16
        tri_bridge = np.array([[0, 1], [1, 2], [2, 0], [2, 3]])
        c = {}
        a = evaluate_graph(tri_bridge, NU, A1, source=0, terminal=3,
                           momentum=[0.0], n_points=n, block_cache=c)
        b = evaluate_graph(tri_bridge, NU, A1, source=0, terminal=3,
                           momentum=[0.0], n_points=n, block_cache=None)
        assert a == pytest.approx(b, rel=1e-14)
        assert any(k[0] == "bridge" for k in c), "bridge key vanished"


class TestGridMomentumCacheKeying:
    """Grid momentum: a block reused at a different n or k must not
    hit a stale entry."""

    FAMILY = [
        (np.array([[0, 1], [1, 2], [2, 0]]), 0, 2),
        (np.array([[0, 1], [1, 2], [2, 0], [2, 3]]), 0, 3),
        (np.array([[0, 1], [1, 2], [2, 3], [3, 0]]), 0, 2),
        (np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]]), 0, 2),
        (np.array([[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 2]]), 0, 4),
    ]

    def test_one_cache_across_momenta_and_n_never_serves_stale(self):
        shared = {}
        for n in (8, 16):
            for mom in (None, [0.0], [0.25], [0.5]):
                for edges, s, t in self.FAMILY:
                    got = evaluate_graph(edges, NU, A1, source=s, terminal=t,
                                         momentum=mom, n_points=n,
                                         block_cache=shared)
                    want = evaluate_graph(edges, NU, A1, source=s, terminal=t,
                                          momentum=mom, n_points=n,
                                          block_cache=None)
                    got = np.asarray(got, dtype=float)
                    want = np.asarray(want, dtype=float)
                    rel = np.max(np.abs(got - want)
                                 / np.maximum(np.abs(want), 1e-300))
                    assert rel < 1e-13, (
                        f"stale cache hit at n={n} momentum={mom}: rel {rel:.3e}"
                    )
