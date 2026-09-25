# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Validation of the hybrid SP-reduction + dense-core engine
(:mod:`gzl.hybrid`).

The hybrid collapses every series/parallel part of a block by FFT and
contracts only the irreducible core densely.  It must reproduce
:func:`gzl.tensor_network.graph_zeta_general` (same truncated
torus kernel) to floating-point round-off in every mode (vacuum, single
k, full BZ grid) and every dimension — that bit-equivalence is the core
check.  We also pin down the structural wins: a cycle reduces to a single
FFT, and a self-loop / decoration contributes its scalar.
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl import evaluate_graph
from gzl.tensor_network import (
    graph_zeta_general,
    graph_zeta_general_at_zero,
)
from gzl.hybrid import hybrid_zeta


TRIANGLE = [(0, 1), (1, 2), (0, 2)]
SQUARE   = [(0, 1), (1, 2), (2, 3), (0, 3)]
CYCLE8   = [(i, (i + 1) % 8) for i in range(8)]
THETA    = [(0, 1), (0, 1), (0, 1)]
K4       = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PRISM    = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
            (0, 3), (1, 4), (2, 5)]
K33      = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5),
           (2, 3), (2, 4), (2, 5)]
# subdivided K4: every edge split by a midpoint (tw 3 core + degree-2 chains)
_SUBK4 = []
_m = 4
for (u, v) in K4:
    _SUBK4 += [(u, _m), (_m, v)]; _m += 1

A1 = np.array([[1.0]])
A2 = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
_ALL = {"triangle": TRIANGLE, "square": SQUARE, "cycle8": CYCLE8,
        "theta": THETA, "K4": K4, "prism": PRISM, "K33": K33,
        "subdiv_K4": _SUBK4}


@pytest.mark.parametrize("name", list(_ALL))
@pytest.mark.parametrize("A,n", [(A1, 12), (A2, 7)])
def test_full_grid_matches_tensor(name, A, n):
    edges = np.array(_ALL[name], dtype=int)
    nu_vec = np.full(len(edges), 2.5)
    d = A.shape[0]
    V = int(edges.max()) + 1
    s, t = 0, min(2, V - 1)
    ref = np.asarray(graph_zeta_general(
        edges, nu_vec, A, n, source=s, terminals=(t,), space='k',
    )).real.reshape((n,) * d)
    got = hybrid_zeta(edges, nu_vec, A, n, source=s, terminal=t)
    assert got.shape == (n,) * d
    rel = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
    assert rel <= 1e-9, f"{name} d={d}: grid rel={rel:.2e}"


@pytest.mark.parametrize("name", ["square", "K4", "prism", "K33", "subdiv_K4"])
def test_single_k_matches_tensor(name):
    edges = np.array(_ALL[name], dtype=int)
    nu_vec = np.full(len(edges), 2.5)
    n = 14
    s, t = 0, 2
    kf = np.array([3.0]) / n
    ref = float(np.asarray(graph_zeta_general(
        edges, nu_vec, A1, n, source=s, terminals=(t,), momentum=kf,
    )).real)
    got = hybrid_zeta(edges, nu_vec, A1, n, source=s, terminal=t, momentum=kf)
    assert abs(got - ref) <= 1e-9 * abs(ref), f"{name}: {got} vs {ref}"


@pytest.mark.parametrize("name", ["triangle", "K4", "prism", "K33"])
@pytest.mark.parametrize("A,n", [(A1, 12), (A2, 7)])
def test_vacuum_matches_tensor(name, A, n):
    edges = np.array(_ALL[name], dtype=int)
    nu_vec = np.full(len(edges), 2.5)
    ref = float(graph_zeta_general_at_zero(edges, nu_vec, A, n).real)
    got = hybrid_zeta(edges, nu_vec, A, n)               # vacuum (terminal=None)
    assert abs(got - ref) <= 1e-9 * abs(ref)


def test_mixed_exponents_match():
    edges = np.array(K4, dtype=int)
    nu_vec = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    n = 12
    ref = np.asarray(graph_zeta_general(
        edges, nu_vec, A1, n, source=0, terminals=(2,), space='k')).real.reshape(n)
    got = hybrid_zeta(edges, nu_vec, A1, n, source=0, terminal=2)
    assert np.max(np.abs(got - ref)) <= 1e-9 * np.max(np.abs(ref))


def test_grid_k0_equals_vacuum():
    edges = np.array(PRISM, dtype=int)
    nu_vec = np.full(9, 2.5)
    n = 12
    grid = hybrid_zeta(edges, nu_vec, A1, n, source=0, terminal=3)
    vac = hybrid_zeta(edges, nu_vec, A1, n)
    assert abs(grid[0] - vac) <= 1e-9 * abs(vac)


def test_cycle_reduces_to_one_fft():
    """A long cycle fully SP-reduces: the result must still match the
    tensor, confirming the series/parallel collapse is exact."""
    edges = np.array([(i, (i + 1) % 16) for i in range(16)], dtype=int)
    nu_vec = np.full(16, 2.5)
    n = 24
    ref = np.asarray(graph_zeta_general(
        edges, nu_vec, A1, n, source=0, terminals=(8,), space='k')).real.reshape(n)
    got = hybrid_zeta(edges, nu_vec, A1, n, source=0, terminal=8)
    assert np.max(np.abs(got - ref)) <= 1e-9 * np.max(np.abs(ref))


# ---------------------------------------------------------------------------
# Frontend wiring: engine="hybrid" (default) vs engine="tensor"
# ---------------------------------------------------------------------------

def test_evaluate_graph_default_engine_is_hybrid_and_matches_tensor():
    """evaluate_graph defaults to the hybrid engine and agrees with the
    tensor engine across vacuum / single-k / full-grid, d=1 and d=2."""
    for A, n in [(A1, 16), (A2, 8)]:
        d = A.shape[0]
        # 1qp full grid
        h = np.asarray(evaluate_graph(np.array(K4), 2.5, A,
                                      source=0, terminal=2, n_points=n))
        t = np.asarray(evaluate_graph(np.array(K4), 2.5, A, source=0,
                                      terminal=2, n_points=n, engine="tensor"))
        assert h.shape == (n,) * d
        assert np.max(np.abs(h - t)) <= 1e-9 * np.max(np.abs(t))
    # vacuum + single-k (1D)
    hv = evaluate_graph(np.array(K4), 2.5, A1, n_points=16)
    tv = evaluate_graph(np.array(K4), 2.5, A1, n_points=16, engine="tensor")
    assert abs(hv - tv) <= 1e-9 * abs(tv)
    hk = evaluate_graph(np.array(K4), 2.5, A1, source=0, terminal=2,
                        momentum=np.array([0.25]), n_points=16)
    tk = evaluate_graph(np.array(K4), 2.5, A1, source=0, terminal=2,
                        momentum=np.array([0.25]), n_points=16, engine="tensor")
    assert abs(hk - tk) <= 1e-9 * abs(tk)


def test_invalid_engine_raises():
    with pytest.raises(ValueError):
        evaluate_graph(np.array(K4), 2.5, A1, n_points=8, engine="bogus")


# ---------------------------------------------------------------------------
# Off-grid momenta (regression: signed vs unsigned terminal index)
# ---------------------------------------------------------------------------
#
# ``M(x_t)`` is indexed on the *balanced* z-axis, so the single-k phase
# sum must use signed labels.  Weighting it with the unsigned index
# ``0..n-1`` instead attaches a spurious ``exp(-2πi k n)`` to every
# negative position.  That factor is exactly 1 at on-grid ``k = j / n``,
# so on-grid tests cannot see the defect — every check above uses an
# on-grid ``kf``.  Off-grid it aliases: the value tracks
# ``frac(k · n_points)`` rather than k, does not converge in
# ``n_points``, and was observed 41 % wrong on the C4 fixture below.

_OFF_GRID_K = np.array([0.1])           # off-grid for every n_points used here


@pytest.mark.parametrize("name", ["square", "K4", "prism", "K33", "subdiv_K4"])
@pytest.mark.parametrize("A,n", [(A1, 14), (A2, 7)])
def test_single_k_off_grid_matches_tensor(name, A, n):
    """Bit-equivalence with the tensor engine must hold *off* the BZ grid too.

    The tensor engine already weights by the balanced-z labels, so it is
    the reference the hybrid has to reproduce at arbitrary k.
    """
    edges = np.array(_ALL[name], dtype=int)
    nu_vec = np.full(len(edges), 2.5)
    d = A.shape[0]
    kf = np.array([0.1, 0.23][:d])
    assert not np.allclose(kf * n, np.round(kf * n)), "fixture k must be off-grid"
    ref = float(np.asarray(graph_zeta_general(
        edges, nu_vec, A, n, source=0, terminals=(2,), momentum=kf,
    )).real)
    got = hybrid_zeta(edges, nu_vec, A, n, source=0, terminal=2, momentum=kf)
    assert abs(got - ref) <= 1e-9 * abs(ref), f"{name} d={d}: {got} vs {ref}"


@pytest.mark.parametrize("nu,tol", [(25.0, 1e-9), (3.0, 1e-7)])
def test_off_grid_momentum_is_n_points_independent(nu, tol):
    """ζ_G at fixed k must not depend on the discretisation it was
    computed with — neither on- nor off-grid.

    C4 on the spine is the compact fixture: it is a simple cycle, so the
    router sends it past the k = 0 closed form into a grid-based engine.
    ``k = 0.1`` is off-grid for all three ``n_points`` and
    ``frac(k · n_points)`` differs at each (0.4, 0.8, 0.6), which is what
    the aliased value tracked.
    """
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    n_list = (64, 128, 256)

    for k, label in [(0.1, "off-grid"), (0.125, "on-grid")]:
        vals = [
            float(evaluate_graph(edges, nu, A1, source=0, terminal=2,
                                 momentum=np.array([k]), n_points=n))
            for n in n_list
        ]
        spread = max(vals) - min(vals)
        assert spread <= tol * abs(vals[0]), (
            f"ν={nu} {label} k={k}: value varies with n_points "
            f"{dict(zip(n_list, vals))} (spread {spread:.3e})"
        )


@pytest.mark.parametrize("k", [0.0, 0.037, 0.1, 0.31415, 0.5, 0.77])
def test_off_grid_matches_nearest_neighbour_closed_form(k):
    """Engine-independent oracle at large ν.

    For C4 on the d=1 chain with terminals 0 and 2, the ν → ∞ (nearest
    neighbour) limit is a bare walk count: x_2 = 0 in four of the six
    surviving configurations and x_2 = ±2 in one each, so
    ζ = 4 + 2 cos(4π k) up to O(2^{-ν}) corrections.  This pins the
    absolute value at arbitrary off-grid k, independently of any other
    engine in the library.
    """
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    got = float(hybrid_zeta(edges, np.full(4, 25.0), A1, 64,
                            source=0, terminal=2, momentum=np.array([k])))
    assert got == pytest.approx(4.0 + 2.0 * np.cos(4.0 * np.pi * k), abs=1e-9)


def test_nu_inf_falls_back_to_tensor():
    """ν = ∞ (NN indicator) must still evaluate (hybrid declines it, the
    router uses its tensor / NN path) and match engine='tensor'."""
    edges = np.array(SQUARE)
    h = np.asarray(evaluate_graph(edges, np.inf, A1, source=0, terminal=2,
                                  n_points=12))
    t = np.asarray(evaluate_graph(edges, np.inf, A1, source=0, terminal=2,
                                  n_points=12, engine="tensor"))
    assert np.max(np.abs(h - t)) <= 1e-9 * (np.max(np.abs(t)) + 1e-300)


# 7-vertex / 12-edge irreducible core (min degree 3 — no SP reduction), the
# order-13 production shape whose dense contraction ran for hours before the
# elimination-order guard: NumPy's default einsum path caps intermediates at
# the largest input (n^2) and silently degenerates to the naive one-shot
# n^(V-1) contraction.  With the explicit memory budget the greedy path is a
# bucket elimination — this test hangs for minutes if that regresses.
CORE7V12E = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
             (1, 4), (2, 5), (3, 6), (4, 5), (4, 6), (5, 6)]


def test_dense_core_uses_elimination_order():
    from gzl.hybrid import HybridCoreTooLargeError

    edges = np.array(CORE7V12E, dtype=int)
    nu_vec = np.full(len(edges), 2.5)
    # correctness against the tensor engine (vacuum + full grid)
    n = 16
    hv = hybrid_zeta(edges, nu_vec, A1, n)
    tv = graph_zeta_general_at_zero(edges, nu_vec, A1, n)
    assert abs(hv - tv) <= 1e-9 * abs(tv)
    hg = np.asarray(hybrid_zeta(edges, nu_vec, A1, n, terminal=6))
    tg = np.asarray(graph_zeta_general(edges, nu_vec, A1, n, terminals=(6,),
                                       space="k"))
    assert np.max(np.abs(hg - tg)) <= 1e-9 * np.max(np.abs(tg))
    # production-scale smoke: sub-second with a pairwise path, ~n^6
    # element-ops (minutes) if the path degenerates
    hybrid_zeta(edges, nu_vec, A1, 64, terminal=6)
    # An explicit memory cap below the schedule's real peak must raise.
    # (The einsum_path FLOP guard this line used to exercise has been
    # removed: it priced a greedy path the executor no longer takes, and
    # refused cores the executor runs faster and leaner than the tensor
    # engine the frontend falls back to.  The cap that replaces it is
    # priced on the executor's own post-peel peak and is OFF by default.)
    with pytest.raises(HybridCoreTooLargeError):
        hybrid_zeta(edges, nu_vec, A1, 96, terminal=6, max_core_bytes=1e3)
    # ... and with no cap the same call simply runs.
    hybrid_zeta(edges, nu_vec, A1, 96, terminal=6)


REFUSED_BY_OLD_GUARD = [
    (0, 1), (0, 3), (0, 4), (0, 5), (1, 3), (1, 5), (1, 6),
    (2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (4, 5),
]


def test_core_the_old_flop_guard_wrongly_refused():
    """Regression for the removal of the einsum_path FLOP guard.

    The pre-flight ``np.einsum_path`` FLOP guard priced a greedy path
    the executor no longer takes and refused this treewidth-4 census
    core at ~5.07e14 estimated FLOPs against its 1e12 budget.  Measured at
    d=2 n=24: hybrid contracts it in 9.45 s / 6.03 GB, while the
    tensor engine the frontend falls back to on refusal takes
    12.48 s / 7.46 GB for the same value — so the refusal handed the
    work to a strictly worse engine.  Four census cores of IDENTICAL
    planner cost scored 6.6e11 and passed; the spread was 767x.

    Checked here at n=16 (same schedule class, sub-second): it must
    run, and agree with the tensor engine.
    """
    edges = np.array(REFUSED_BY_OLD_GUARD, dtype=int)
    nu_vec = np.full(len(edges), 4.5)
    A2 = np.eye(2)
    hv = hybrid_zeta(edges, nu_vec, A2, 16)
    tv = graph_zeta_general_at_zero(edges, nu_vec, A2, 16)
    assert abs(hv - tv) <= 1e-9 * abs(tv)


A3 = np.eye(3)


class TestDenseCoreMemoryLean:
    """The dense core materialises circulant edge matrices lazily along
    the vetted elimination path and frees them eagerly; source-incident
    edges never build an n^{2d} matrix at all; the budget guards run on
    zero-stride shape dummies BEFORE any n^{2d} allocation."""

    def test_d3_bit_equivalence_vacuum_grid_offgrid(self):
        # d=3 coverage: K4 + prism vs the tensor engine in all three
        # momentum modes, including nu <= d
        k_off = np.array([0.13, 0.27, 0.41])
        for edges in (K4, PRISM):
            n_t = max(max(e) for e in edges)
            for nu in (3.5, 2.0):
                nu_vec = np.full(len(edges), nu)
                for n in (4, 6):
                    hv = hybrid_zeta(edges, nu_vec, A3, n)
                    tv = graph_zeta_general_at_zero(edges, nu_vec, A3, n)
                    assert abs(hv - tv) <= 1e-12 * abs(tv)
                    hg = np.asarray(hybrid_zeta(edges, nu_vec, A3, n,
                                                terminal=n_t))
                    tg = np.asarray(graph_zeta_general(
                        edges, nu_vec, A3, n, terminals=(n_t,), space="k",
                    )).real.reshape((n,) * 3)
                    assert np.max(np.abs(hg - tg)) <= 1e-12 * np.max(np.abs(tg))
                    hk = hybrid_zeta(edges, nu_vec, A3, n, terminal=1,
                                     momentum=k_off)
                    tk = float(np.real(graph_zeta_general(
                        edges, nu_vec, A3, n, terminals=(1,), momentum=k_off,
                    )))
                    assert abs(hk - tk) <= 1e-12 * (abs(tk) + 1e-300)

    def test_cap_fires_before_kernel_allocation(self):
        """A core rejected by the memory cap must raise from the
        SYMBOLIC schedule, not after materialising the n^{2d} kernel
        matrices (which cost ~19 GiB at d=3 n=24 and were once built
        before the guard ran).  Verified by peak RSS of a fresh
        subprocess staying far below one kernel matrix (24^6 doubles =
        1.5 GiB).

        The cap is explicit here because it is OFF by default — this
        same K4 was refused by the old einsum_path FLOP guard, and its
        measured peak (1.4 GB post-peel) is one the executor handles."""
        import subprocess, sys, textwrap
        code = textwrap.dedent("""
            import resource, sys
            import numpy as np
            from gzl.hybrid import hybrid_zeta, HybridCoreTooLargeError
            K4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            try:
                hybrid_zeta(K4, np.full(6, 3.5), np.eye(3), 24, terminal=3,
                            max_core_bytes=1e6)
                sys.exit(2)                      # must not succeed
            except HybridCoreTooLargeError:
                pass

            def peak():
                # Linux: exec folds the parent's high-water mark into
                # ru_maxrss, so read VmHWM, this process image alone (kB).
                if sys.platform.startswith("linux"):
                    with open("/proc/self/status") as fh:
                        for line in fh:
                            if line.startswith("VmHWM:"):
                                return int(line.split()[1]) * 1024
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            sys.exit(0 if peak() < 800e6 else 3)    # macOS: bytes
        """)
        r = subprocess.run([sys.executable, "-c", code], timeout=120)
        assert r.returncode == 0, {2: "guard did not raise",
                                   3: "guard raised only after large "
                                      "allocations"}.get(r.returncode,
                                                         r.returncode)

    def test_letters_overflow_raises_core_too_large(self):
        # 54 core vertices at d=1 exceed the 52 einsum letters; this used
        # to raise a bare ValueError that escaped the frontend fallback
        from gzl.hybrid import HybridCoreTooLargeError
        edges = ([(i, (i + 1) % 54) for i in range(54)]
                 + [(i, i + 27) for i in range(27)])
        with pytest.raises(HybridCoreTooLargeError):
            hybrid_zeta(edges, np.full(len(edges), 0.9), A1, 4)

    def test_frontend_falls_back_on_letters_overflow(self):
        """54-vertex circulant: the hybrid letters overflow must fall back
        to the tensor engine instead of leaking.

        Uses nu > d.  The original spelling used nu = 0.9 <= d = 1 and
        relied on direct_sum declining the DIVERGENT sum to get the block
        as far as _block_general -- but a divergent lattice sum is now
        refused outright, and agreement between two engines on a
        truncation artefact was never the property worth pinning here.
        The letters path is what this test is for, and nu > d exercises
        it just as well.
        """
        edges = np.array([(i, (i + 1) % 54) for i in range(54)]
                         + [(i, i + 27) for i in range(27)], dtype=int)
        h = evaluate_graph(edges, 1.5, A1, n_points=4, block_cache={})
        t = evaluate_graph(edges, 1.5, A1, n_points=4, engine="tensor",
                           block_cache={})
        assert abs(h - t) <= 1e-9 * abs(t)

    def test_a_budget_refusal_goes_to_the_box_not_the_uncapped_tensor(
            self, monkeypatch):
        """A byte-budget refusal must NOT be handed to graph_zeta_general.

        The tensor has no byte cap and an unchunked peel, and it runs the
        contraction on the RAW block rather than hybrid's SP minor -- so
        it is never smaller than what hybrid just refused.  At d = 3 that
        turned a refused 2 GB core into a 512 GiB ``np.ones`` on K5, which
        first appears at order 10 of the shipped 1qp corpus.  numpy does
        not raise for that allocation (a 1 TB ``np.empty`` succeeds), so
        the MemoryError nets never armed and the OS killed the process.

        Structural refusals are the opposite case and are covered by
        ``test_frontend_falls_back_on_letters_overflow``: there the tensor
        genuinely handles more vertices, so falling through is right.
        """
        import gzl.frontend as fe

        seen = []
        for name in ("graph_zeta_general", "graph_zeta_general_at_zero",
                     "direct_sum_extrapolated"):
            fn = getattr(fe, name)

            def spy(*a, _n=name, _f=fn, **k):
                seen.append(_n)
                return _f(*a, **k)

            monkeypatch.setattr(fe, name, spy)

        # A cap far below what K5 needs, so hybrid refuses on BUDGET.
        monkeypatch.setattr(fe, "_DENSE_CORE_MAX_BYTES", 1.0e5)
        k5 = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3],
                       [1, 4], [2, 3], [2, 4], [3, 4]], dtype=int)
        val = evaluate_graph(k5, 4.5, A3, n_points=6, block_cache={})

        assert "direct_sum_extrapolated" in seen, (
            f"budget refusal must reach the bounded box engine; saw {seen}")
        assert not [s for s in seen if s.startswith("graph_zeta_general")], (
            f"budget refusal must NOT reach the uncapped tensor; saw {seen}")
        assert np.isfinite(val)

    def test_frontend_falls_back_on_memory_error(self, monkeypatch):
        """A genuine MemoryError from the hybrid engine (e.g. kernel
        allocation at large n) must fall back to the tensor engine.  The
        subdivided theta (three parallel 2-paths — no Hadamard merge, tw 2,
        sigma >= 1.49) genuinely routes through _block_general; the call
        counter proves the monkeypatched engine was actually reached."""
        import gzl.frontend as fe

        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise MemoryError("simulated hybrid OOM")

        edges = np.array([(0, 2), (2, 1), (0, 3), (3, 1), (0, 4), (4, 1)])
        ref = evaluate_graph(edges, 4.0, A1, source=0, terminal=1,
                             n_points=12, engine="tensor", block_cache={})
        monkeypatch.setattr(fe, "hybrid_zeta", boom)
        val = evaluate_graph(edges, 4.0, A1, source=0, terminal=1,
                             n_points=12, block_cache={})
        assert calls, "hybrid engine was never invoked — vacuous test"
        assert np.max(np.abs(np.asarray(val) - np.asarray(ref))) <= 1e-12 * (
            np.max(np.abs(np.asarray(ref))) + 1e-300
        )


class TestShearedLatticeOrientation:
    """Edge orientation must be tracked through the SP reduction.

    ``_sp_reduce`` stores kernels in a ``networkx.MultiGraph``, which
    normalises endpoint order — so the ordered pair a kernel was built
    for does not survive insertion, and reading it back with the
    opposite convention is wrong by ``z -> -z``.

    That is invisible on an even kernel, and the kernel is even on every
    orthogonal lattice and at every odd ``n``.  It is *not* even on a
    sheared cell at even ``n``: ``_balanced_z_axis`` holds ``n/2``
    without ``-n/2``, so the shell carrying an ``n/2`` component has no
    partner.  Since every ``n_points`` the repo uses is even (16, 24,
    32, 64, 128, 256), a square-lattice or odd-``n`` test asserts
    nothing here — these must parametrise over even ``n`` AND a skew
    ``A``.

    The reference is orientation-invariant by construction: the m-th
    matrix power of the edge difference table contracts the m-cycle with
    no FFT and no orientation convention at all.
    """

    A_HEX = np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2.0]])
    A_SKEW = np.array([[1.0, 0.35], [0.2, 1.1]])
    NU = 4.0
    RTOL = 1.0e-12

    def _oracle(self, A, n, m):
        from numpy.linalg import matrix_power
        from gzl.tensor_network import (
            _edge_difference_table, _edge_kernel_torus,
        )
        K = _edge_kernel_torus(self.NU, A, n)
        return float(matrix_power(_edge_difference_table(K, n), m)[0, 0])

    @pytest.mark.parametrize("lattice", ["hex", "skew", "square"])
    @pytest.mark.parametrize("m", [4, 6])
    @pytest.mark.parametrize("n", [8, 12, 16, 11])
    def test_cycle_matches_orientation_invariant_oracle(self, lattice, m, n):
        A = {"hex": self.A_HEX, "skew": self.A_SKEW,
             "square": np.eye(2)}[lattice]
        E = np.array([[i, (i + 1) % m] for i in range(m)], dtype=int)
        nu = np.full(m, self.NU)
        ref = self._oracle(A, n, m)
        got = float(np.real(hybrid_zeta(E, nu, A, n)))
        rel = abs(got - ref) / abs(ref)
        assert rel < self.RTOL, (
            f"{lattice} m={m} n={n}: hybrid {got!r} vs oracle {ref!r}, "
            f"rel {rel:.3e}"
        )

    def test_the_kernel_really_is_odd_where_it_matters(self):
        """Root cause: without this the test above proves nothing.

        If the kernel were even on a sheared cell at even n, the
        orientation bug would be unobservable and a passing test would
        carry no information.
        """
        from gzl.tensor_network import (
            _edge_kernel_torus, _reverse_generator,
        )
        for n in (8, 12, 16):
            K = _edge_kernel_torus(self.NU, self.A_HEX, n)
            assert not np.allclose(K, _reverse_generator(K)), (
                f"hex n={n}: kernel is even, so this suite cannot see "
                f"an orientation defect"
            )
        K11 = _edge_kernel_torus(self.NU, self.A_HEX, 11)
        assert np.allclose(K11, _reverse_generator(K11))

    def test_mixed_orientation_bundle(self):
        """A parallel bundle given in both orientations must merge as one."""
        from gzl.tensor_network import graph_zeta_general_at_zero
        E = np.array([[0, 1], [1, 0], [0, 1]], dtype=int)
        nu = np.full(3, self.NU)
        for n in (8, 12):
            h = float(np.real(hybrid_zeta(E, nu, self.A_HEX, n)))
            t = float(np.real(
                graph_zeta_general_at_zero(E, nu, self.A_HEX, n)))
            assert abs(h - t) / abs(t) < self.RTOL, (
                f"n={n}: hybrid {h!r} vs tensor {t!r}"
            )

    def test_agrees_with_the_tensor_engine_on_a_sheared_lattice(self):
        """End-to-end: the route evaluate_graph actually takes."""
        from gzl import evaluate_graph
        diamond = np.array(
            [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]], dtype=int,
        )
        for n in (8, 16):
            h = evaluate_graph(diamond, self.NU, self.A_HEX,
                               n_points=n, engine="hybrid")
            t = evaluate_graph(diamond, self.NU, self.A_HEX,
                               n_points=n, engine="tensor")
            assert abs(h - t) / abs(t) < 1e-13, f"n={n}: {h!r} vs {t!r}"
