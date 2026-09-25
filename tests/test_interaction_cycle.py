# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""The generalised cycle closed form: ``zeta_circle`` with Interaction-likes
per edge.  Oracles: bit-identity of the demoted power law, the exact
per-edge linearity of the Brillouin-zone integral, the torus truncation
``mean(fft(V)**E)`` converged past round-off at steep exponents, and the
self-check ladder's failure path."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from gzl import GraphZetaError, Interaction, zeta_circle
from gzl import circle as C
from gzl.interaction import _KernelProduct
from gzl.tensor_network import _balanced_z_axis

A_CHAIN = np.eye(1)
A_SQUARE = np.eye(2)
A_HEX = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
A_SHEARED = np.array([[1.0, 0.3], [0.0, 1.1]])
A_CUBIC = np.eye(3)


def torus_grid(V, A, n):
    d = A.shape[0]
    z = _balanced_z_axis(n)
    labels = np.stack(np.meshgrid(*([z] * d), indexing="ij"), axis=-1)
    return V.sample(labels, A, power="torus")


def torus_cycle(kernels, A, n):
    """The torus truncation of a cycle with per-edge kernels: the E-fold
    cyclic convolution at the origin, ``mean(prod_e fft(V_e))``.  Exact
    for compact kernels once n exceeds the reach; converges like
    n^-(2 nu - d) for a power-law tail (the cluster cut of a cycle)."""
    prod = None
    for V in kernels:
        f = np.fft.fftn(torus_grid(V, A, n))
        prod = f if prod is None else prod * f
    return float(np.mean(prod).real)


def cross(d, val, origin=0.0):
    tab = {}
    for i in range(d):
        for s in (-1, 1):
            m = [0] * d
            m[i] = s
            tab[tuple(m)] = val
    if origin:
        tab[(0,) * d] = origin
    return tab


def dense_even_table(d, R, J, a0, seed):
    """Every label of the Chebyshev ball of radius ``R`` populated with a
    random even table in ``[-J, J]`` (``a0`` at the origin): the densest
    mode content a radius-``R`` table can carry, which is what decides the
    fixed rule's resolution (see ``circle._TS_REFINED``)."""
    rng = np.random.default_rng(seed)
    tab = {}
    for m in itertools.product(range(-R, R + 1), repeat=d):
        if m == (0,) * d:
            continue
        neg = tuple(-x for x in m)
        tab[m] = tab[neg] if neg in tab else float(rng.uniform(-J, J))
    tab[(0,) * d] = a0
    return tab


def base_rule(kernels, A):
    """The shipped fixed rule alone (no ladder) on per-edge kernels."""
    data = [k.fourier_terms(A) for k in kernels]
    return C._zeta_circle_kernels_fixed(data, A, C._TS_M, C._TS_H, None, return_scale=True)


def refined_rule(kernels, A, rung=0):
    """One rung of ``_TS_REFINED`` alone on per-edge kernels."""
    data = [k.fourier_terms(A) for k in kernels]
    factor, n_ang = C._TS_REFINED[A.shape[0]][rung]
    return C._zeta_circle_kernels_fixed(data, A, factor * C._TS_M, C._TS_H / factor, n_ang,
                                        return_scale=True)


class TestLegacyIdentity:
    @pytest.mark.parametrize("A,nu", [(A_CHAIN, 1.5), (A_SQUARE, 2.5), (A_HEX, 2.5), (A_CUBIC, 3.5)])
    def test_plain_power_law_interaction_bit_equals_the_float_call(self, A, nu):
        E = 3
        want = zeta_circle([nu] * E, A)
        got = zeta_circle([Interaction.power_law(nu)] * E, A)
        assert got == want
        got = zeta_circle([nu, Interaction.power_law(nu), nu], A)
        assert got == want

    def test_shipped_angular_rule_is_gauss_legendre_14(self):
        a, w = C._angular_rule(14)
        assert np.allclose(a, C.abscissa0Half, rtol=0, atol=1e-15)
        assert np.allclose(w, C.weight0Half, rtol=0, atol=1e-15)
        a0, w0 = C._angular_rule(None)
        assert np.array_equal(a0, C.abscissa0Half) and np.array_equal(w0, C.weight0Half)

    def test_legacy_grid_and_table_keys_are_unchanged(self):
        C._ts_grid(2, C._TS_M, C._TS_H)
        assert (C._TS_M, C._TS_H) in C._TS_GRID_CACHE
        A = A_SQUARE
        C._ts_zeta_table(2.5, A, np.linalg.inv(A).T, C._TS_M, C._TS_H)
        assert (2.5, C._TS_M, C._TS_H, A.shape, A.tobytes()) in C._TS_TABLE_CACHE
        C._ts_zeta_table(2.5, A, np.linalg.inv(A).T, 2 * C._TS_M, C._TS_H / 2, 28)
        assert (2.5, 2 * C._TS_M, C._TS_H / 2, A.shape, A.tobytes(), 28) in C._TS_TABLE_CACHE

    def test_pure_power_law_kernel_path_agrees_with_the_shipped_rule(self):
        # the kernel path multiplies the table per edge instead of t ** count
        for A in (A_SQUARE, A_HEX):
            E, nu = 5, 2.5
            data = [Interaction.power_law(nu).fourier_terms(A)] * E
            got = C._zeta_circle_kernels_fixed(data, A, C._TS_M, C._TS_H, None)
            assert got == pytest.approx(C.zeta_circle_tanh_sinh([nu] * E, A), rel=1e-14, abs=0.0)


class TestPurePowerLaws:
    def test_scalar_weight_factors_out(self):
        V = Interaction.power_law(2.5, b=2.0)
        assert zeta_circle([V] * 3, A_SQUARE) == pytest.approx(8.0 * zeta_circle([2.5] * 3, A_SQUARE), rel=1e-14)

    @pytest.mark.parametrize("A", [A_CHAIN, A_SQUARE, A_SHEARED])
    def test_two_term_kernel_is_multilinear_in_the_edges(self, A):
        d = A.shape[0]
        b = (0.7, -0.3)
        nu = (d + 1.0, d + 2.5)
        V = Interaction(b=list(b), nu=list(nu))
        E = 3
        want = 0.0
        for choice in itertools.product(range(2), repeat=E):
            coef = np.prod([b[j] for j in choice])
            want += coef * zeta_circle([nu[j] for j in choice], A)
        got = zeta_circle([V] * E, A)
        assert got == pytest.approx(want, rel=1e-13)


class TestCompactParts:
    def test_hexagonal_j1_j2_tail_against_the_torus(self):
        # the base rule is accepted outright for this 12-label table (the
        # exact-class referee passes); its residual against the torus is
        # 3.6e-12 at E = 6 and 3.7e-11 at E = 12 (the refined rung would
        # be 1e-13), inside the 1e-9 contract by more than an order
        V = Interaction.from_shells(A_HEX, {1.0: 0.5, np.sqrt(3): 0.2}, b=[0.3], nu=[4.5])
        for E in (3, 6, 12):
            ref = torus_cycle([V] * E, A_HEX, 512)
            assert zeta_circle([V] * E, A_HEX) == pytest.approx(ref, rel=1e-10)

    def test_anisotropic_table_with_origin(self):
        T = Interaction.from_table({(1, 0): 0.6, (-1, 0): 0.6, (0, 1): 0.2, (0, -1): 0.2, (0, 0): 0.3},
                                   b=[1.0], nu=[4.5])
        for E in (3, 7):
            ref = torus_cycle([T] * E, A_SQUARE, 512)
            assert zeta_circle([T] * E, A_SQUARE) == pytest.approx(ref, rel=1e-13)

    def test_purely_compact_cycle_is_exact(self):
        T = Interaction.from_table(cross(2, 0.5, origin=0.25))
        for E in (3, 5):
            exact = torus_cycle([T] * E, A_SQUARE, E + 3)          # no wrap: exact finite sum
            assert torus_cycle([T] * E, A_SQUARE, 2 * E + 3) == pytest.approx(exact, rel=1e-14)
            assert zeta_circle([T] * E, A_SQUARE) == pytest.approx(exact, rel=1e-13)

    def test_bundle_on_one_edge(self):
        V = Interaction.from_shells(A_HEX, {1.0: 0.5}, b=[0.3], nu=[4.5])
        kernels = [V ** 2, V, V, V]
        assert isinstance(kernels[0], _KernelProduct)
        ref = torus_cycle(kernels, A_HEX, 512)
        assert zeta_circle(kernels, A_HEX) == pytest.approx(ref, rel=1e-13)

    def test_mixed_floats_and_interactions(self):
        T = Interaction.from_table(cross(2, 0.4), b=[1.0], nu=[4.5])
        kernels = [T, 4.5, T, 5.0]
        ref = torus_cycle([T, Interaction.power_law(4.5), T, Interaction.power_law(5.0)], A_SQUARE, 512)
        assert zeta_circle(kernels, A_SQUARE) == pytest.approx(ref, rel=1e-13)

    def test_d1_adaptive_rule(self):
        V = Interaction.from_table({(1,): 0.5, (-1,): 0.5, (0,): 0.2}, b=[1.0], nu=[3.5])
        for E in (3, 8):
            ref = torus_cycle([V] * E, A_CHAIN, 4096)
            assert zeta_circle([V] * E, A_CHAIN) == pytest.approx(ref, rel=1e-13)

    def test_escalation_to_the_second_refinement(self):
        # modes to |m| = 36: the base rule is ~1e-8 off, the first
        # refinement at round-off, so the pair disagrees and the referee
        # decides (measured 1.3e-14 through the ladder)
        big = {m: 2.0 for m in itertools.product(range(-3, 4), repeat=2) if m != (0, 0)}
        Vb = Interaction.from_table(big, b=[1.0], nu=[4.5])
        ref = torus_cycle([Vb] * 12, A_SQUARE, 512)
        data = [Vb.fourier_terms(A_SQUARE)] * 12
        base = C._zeta_circle_kernels_fixed(data, A_SQUARE, C._TS_M, C._TS_H, None)
        assert abs(base / ref - 1.0) > 1e-10                       # the ladder is needed
        assert zeta_circle([Vb] * 12, A_SQUARE) == pytest.approx(ref, rel=1e-12)

    @pytest.mark.slow
    def test_d3_cubic_against_the_torus(self):
        T = Interaction.from_table(cross(3, 0.4), b=[1.0], nu=[5.5])
        ref = torus_cycle([T] * 3, A_CUBIC, 64)                    # itself ~1e-12 converged
        assert zeta_circle([T] * 3, A_CUBIC) == pytest.approx(ref, rel=1e-11)


class TestD3Referee:
    """The d = 3 ladder has ONE refinement and so no rung to referee a
    base/refined disagreement.  Dense radius-2 tables (all 124 off-origin
    labels populated) on cubic cycles of E >= 4 edges disagree -- the base
    rule is 1e-09 (E = 4) to 1e-06 (E = 6) off the exact finite sum while
    the refined rule sits at 1e-13 -- and before the exact-sum referee
    the correct refined value was discarded and the router fell back to
    the pass-grid torus (~1e-7).  The referee accepts the last rung when
    it reproduces the exact finite sums of the integrand's two
    highest-degree classes (``circle._exact_compact_classes``).
    Reference: the torus truncation that no E-walk can wrap
    (``n = 2E + 3 > E R``)."""

    @pytest.mark.parametrize("E", [4, 6])
    def test_dense_r2_compact_only_cubic_cycle_is_refereed_and_exact(self, E):
        T = Interaction.from_table(dense_even_table(3, 2, 0.5, 0.3, seed=21))
        assert len(T.compact) == 125
        exact = torus_cycle([T] * E, A_CUBIC, 2 * E + 3)
        base, scale = base_rule([T] * E, A_CUBIC)
        assert abs(base - exact) > C._CYCLE_SELF_BAND * scale     # the ladder disagrees
        got = zeta_circle([T] * E, A_CUBIC)                      # was: CycleQuadratureError
        assert got == pytest.approx(exact, rel=1e-12)

    def test_26_label_r1_cubic_six_cycle_is_resolved_at_the_first_rung(self):
        """The boundary from the other side: a full-cube radius-1 table
        (26 labels) on a 6-cycle is resolved by the base rule -- base and
        first refinement agree to the band, the refined value ships (the
        base is never accepted on its own), and it is the finite sum to
        1e-12.  A random even table has no cubic symmetry, so nothing
        folds and the shipped value is the unfolded rung bit for bit."""
        T = Interaction.from_table(dense_even_table(3, 1, 0.5, 0.3, seed=7))
        assert len(T.compact) == 27
        exact = torus_cycle([T] * 6, A_CUBIC, 6 + 3)
        base, base_scale = base_rule([T] * 6, A_CUBIC)
        rung1, scale = refined_rule([T] * 6, A_CUBIC)
        assert abs(base - rung1) <= C._CYCLE_SELF_BAND * max(scale, base_scale)
        got = zeta_circle([T] * 6, A_CUBIC)
        assert got == rung1
        assert got == pytest.approx(exact, rel=1e-12)

    @pytest.mark.slow
    def test_dense_r2_mixed_cubic_six_cycle_is_refereed(self):
        """The router-level headline: b = 1, nu = 5.5 on a dense radius-2
        table, cubic 6-cycle -- evaluate_graph used to deliver the
        pass-grid torus at 8.3e-07 while the discarded refined rung was at
        8.0e-14.  Reference: torus n = 64 (n^-8 at nu = 5.5; its 48/64
        Richardson pair sits 2.7e-13 from the rung)."""
        V = Interaction.from_table(dense_even_table(3, 2, 0.5, 0.3, seed=21), b=[1.0], nu=[5.5])
        ref = torus_cycle([V] * 6, A_CUBIC, 64)
        base, scale = base_rule([V] * 6, A_CUBIC)
        assert abs(base - ref) > C._CYCLE_SELF_BAND * scale
        assert zeta_circle([V] * 6, A_CUBIC) == pytest.approx(ref, rel=1e-11)


class TestExactClassSums:
    """The referee's finite sums, against independent computations."""

    def test_purely_compact_class_is_the_closed_walk_sum(self):
        T = Interaction.from_table(dense_even_table(2, 2, 2.0, 0.3, seed=13))
        data = [T.fourier_terms(A_SQUARE)] * 5
        ex0, ex1 = C._exact_compact_classes(data, A_SQUARE)
        assert ex0 == pytest.approx(torus_cycle([T] * 5, A_SQUARE, 2 * 5 + 3), rel=1e-13)
        assert ex1 == 0.0                                    # no power-law factor anywhere

    @pytest.mark.parametrize("A", [A_SQUARE, A_HEX])
    def test_one_power_law_class_is_the_table_convolution_against_the_kernel(self, A):
        # one pure edge and three purely compact edges: the whole integral
        # IS the one-power-law class, and the fixed rule is at round-off
        # for a radius-1 table, so the two computations must meet.
        T = Interaction.from_table(dense_even_table(2, 1, 0.5, 0.3, seed=5))
        P = Interaction.power_law(4.5, b=0.7)
        kernels = [P, T, T, T]
        data = [k.fourier_terms(A) for k in kernels]
        ex0, ex1 = C._exact_compact_classes(data, A)
        assert ex0 == 0.0
        assert ex1 == pytest.approx(zeta_circle(kernels, A), rel=1e-12)
        # a bundle's power-law expansion enters the same way
        kernels = [P * P, T, T, T]
        ex0, ex1 = C._exact_compact_classes([k.fourier_terms(A) for k in kernels], A)
        assert ex1 == pytest.approx(zeta_circle(kernels, A), rel=1e-12)

    def test_two_pure_edges_leave_the_one_power_law_class_empty(self):
        T = Interaction.from_table(cross(2, 0.4, origin=0.1))
        kernels = (T, Interaction.power_law(4.5), Interaction.power_law(5.0), T)
        data = [k.fourier_terms(A_SQUARE) for k in kernels]
        assert C._exact_compact_classes(data, A_SQUARE) == (0.0, 0.0)


class TestKillSwitch:
    def test_compact_part_refuses_the_legacy_adaptive_path(self, monkeypatch):
        """``USE_TANH_SINH = False`` restores the adaptive rule for pure
        power laws only: with a compact part the adaptive rule shares the
        14-node angular rule with the base fixed rule and has no ladder,
        so it was silently 1e-8..5e-4 off on dense tables at d = 2."""
        monkeypatch.setattr(C, "USE_TANH_SINH", False)
        T = Interaction.from_table(cross(2, 0.4), b=[1.0], nu=[4.5])
        with pytest.raises(NotImplementedError, match="legacy float path"):
            zeta_circle([T] * 3, A_SQUARE)
        T3 = Interaction.from_table(cross(3, 0.4), b=[1.0], nu=[5.5])
        with pytest.raises(NotImplementedError, match="legacy float path"):
            zeta_circle([T3] * 3, A_CUBIC)
        # the escape hatch stays for pure power laws on the kernel path ...
        P = Interaction.power_law(2.5, b=2.0)
        assert zeta_circle([P] * 3, A_SQUARE) == pytest.approx(
            8.0 * zeta_circle([2.5] * 3, A_SQUARE), rel=1e-10)
        # ... and d = 1 keeps its adaptive rule with a compact part
        V1 = Interaction.from_table({(1,): 0.5, (-1,): 0.5, (0,): 0.2}, b=[1.0], nu=[3.5])
        assert zeta_circle([V1] * 3, A_CHAIN) == pytest.approx(
            torus_cycle([V1] * 3, A_CHAIN, 4096), rel=1e-13)


class TestFailurePath:
    def test_unconverged_ladder_raises_a_graph_zeta_error(self, monkeypatch):
        T = Interaction.from_table(cross(2, 0.4), b=[1.0], nu=[4.5])
        monkeypatch.setattr(C, "_CYCLE_SELF_BAND", 1e-18)
        with pytest.raises(C.CycleQuadratureError, match="absolute-error criterion") as info:
            zeta_circle([T] * 3, A_SQUARE)
        assert issubclass(C.CycleQuadratureError, GraphZetaError)
        # the referee was consulted and refused (all edges carry a table)
        assert "refused the last rung" in str(info.value)

    def test_referee_declines_with_two_edges_lacking_a_compact_part(self, monkeypatch):
        T = Interaction.from_table(cross(2, 0.4), b=[1.0], nu=[4.5])
        monkeypatch.setattr(C, "_CYCLE_SELF_BAND", 1e-18)
        with pytest.raises(C.CycleQuadratureError, match="declined: 2 edges carry no compact part"):
            zeta_circle([T, 4.5, T, 5.0], A_SQUARE)

    def test_tail_at_or_below_d_is_refused_but_compact_edges_pass(self):
        bad = Interaction(b=[1.0, 0.5], nu=[2.0, 4.0])
        with pytest.raises(ValueError, match="tail exponent"):
            zeta_circle([bad, 4.5, 4.5], A_SQUARE)
        T = Interaction.from_table(cross(2, 0.4))
        assert np.isfinite(zeta_circle([T, 4.5, 4.5], A_SQUARE))


class TestRefereeClasses:
    """Both classes of the exact-sum referee decide: a residual planted in
    either one refuses the rung and is named in the error."""

    @pytest.mark.parametrize("which", [0, 1])
    def test_each_class_of_the_referee_can_refuse(self, monkeypatch, which):
        # a one-refinement ladder on the dense 24-label square table at
        # E = 8 is exhausted (base 4e-7 off its refined rung), so the
        # referee decides; a residual planted in EITHER class refuses
        T = Interaction.from_table(dense_even_table(2, 2, 2.0, 0.3, seed=5), b=[1.0], nu=[4.5])
        monkeypatch.setattr(C, "_TS_REFINED", {2: ((2, 28),), 3: C._TS_REFINED[3]})
        orig = C._exact_compact_classes

        def planted(*a, **k):
            ex = list(orig(*a, **k))
            ex[which] += 1.0
            return tuple(ex)
        monkeypatch.setattr(C, "_exact_compact_classes", planted)
        with pytest.raises(C.CycleQuadratureError, match="miss their exact finite sums"):
            zeta_circle([T] * 8, A_SQUARE)
        # and without the planted residual the same exhausted ladder is accepted by the referee
        monkeypatch.setattr(C, "_exact_compact_classes", orig)
        val = zeta_circle([T] * 8, A_SQUARE)
        assert val == pytest.approx(torus_cycle([T] * 8, A_SQUARE, 256), rel=1e-9)

    def test_the_chunked_transform_equals_the_matrix(self, monkeypatch):
        rng = np.random.default_rng(3)
        ys = rng.uniform(-0.5, 0.5, size=(1001, 2))
        T = Interaction.from_table(dense_even_table(2, 2, 0.5, 0.3, seed=5))
        _, labels, values = T.fourier_terms(A_SQUARE)
        full = np.cos(2.0 * np.pi * (ys @ labels.T.astype(float))) @ values
        monkeypatch.setattr(C, "_DIRECT_TABLE_MAX_BYTES", 24 * len(values) * 100)   # ~11 blocks
        chunked = C._compact_transform_at(ys, labels, values)
        assert np.allclose(chunked, full, rtol=1e-15, atol=1e-15 * np.abs(full).max())

    def test_the_per_table_memo_keys_on_values_too(self):
        # two tables with the SAME labels and different values on one
        # cycle: the memo must not serve one for the other
        V1 = Interaction.from_table({(1, 0): 0.3, (-1, 0): 0.3, (0, 1): 0.3, (0, -1): 0.3},
                                    b=[1.0], nu=[4.5])
        V2 = Interaction.from_table({(1, 0): 0.6, (-1, 0): 0.6, (0, 1): 0.6, (0, -1): 0.6},
                                    b=[1.0], nu=[4.5])
        mixed = zeta_circle([V1, V2, V1], A_SQUARE)
        assert mixed != zeta_circle([V1] * 3, A_SQUARE)
        assert mixed == pytest.approx(torus_cycle([V1, V2, V1], A_SQUARE, 256), rel=1e-9)


class TestLadderFirstAcceptance:
    """The base rule is accepted only when the first refined rung agrees
    with it; the exact-class referee never short-circuits the ladder.
    Measured (the comment beside ``circle._TS_REFINED``): the shortcut
    accepted signed radius-2 tables 2e-9..9e-9 of scale off the torus."""

    @staticmethod
    def _count_rules(monkeypatch):
        calls = []
        orig = C._fixed_rule_classes

        def counting(*a, **k):
            calls.append(a[2])          # the radial M of the rule that ran
            return orig(*a, **k)

        monkeypatch.setattr(C, "_fixed_rule_classes", counting)
        return calls

    @staticmethod
    def _torus_reference(V, A, E):
        """Richardson over the torus at n = 512, 1024 (basis 2 nu - d), with
        the (256, 512) pair as its own self-consistency check."""
        d = A.shape[0]
        p = 2.0 * V.nu[0] - d
        T = {n: torus_cycle([V] * E, A, n) for n in (256, 512, 1024)}
        rich = lambda a, b: (2.0 ** p * b - a) / (2.0 ** p - 1.0)   # noqa: E731
        ra, rb = rich(T[256], T[512]), rich(T[512], T[1024])
        return rb, abs(ra - rb)

    @pytest.mark.parametrize("A, shells", [
        (A_HEX, {1.0: 0.5}),
        (A_HEX, {1.0: 0.5, np.sqrt(3): 0.2}),
        (np.eye(3), {1.0: 0.5}),
    ])
    def test_small_support_tables_run_the_base_and_one_refinement(self, monkeypatch, A, shells):
        d = A.shape[0]
        V = Interaction.from_shells(A, shells, b=[0.1], nu=[float(d + 1)], total=True)
        E = 6 if d == 2 else 4
        calls = self._count_rules(monkeypatch)
        val = zeta_circle([V] * E, A)
        f, n_ang = C._TS_REFINED[d][0]
        assert calls == [C._TS_M, f * C._TS_M]           # base, first refinement, agreed
        data = [V.fourier_terms(A)] * E
        refined, scale, _, _ = C._fixed_rule_classes(
            data, A, f * C._TS_M, C._TS_H / f, n_ang, fold=True)
        assert val == refined                             # the finer of the agreeing pair is shipped

    @pytest.mark.parametrize("name, A, V, E", [
        ("sheared 12-label radius-2 table, nu = 6, 12 edges", A_SHEARED,
         Interaction.from_table({**cross(2, 0.5, origin=0.3),
                                 (2, 0): 0.35, (-2, 0): 0.35, (0, 2): 0.35, (0, -2): 0.35,
                                 (1, 1): -0.25, (-1, -1): -0.25, (1, -1): -0.25, (-1, 1): -0.25},
                                b=[1.0], nu=[6.0]), 12),
        ("square J1 - J2 + J3 shells, nu = 3.5, 6 edges", A_SQUARE,
         Interaction.from_shells(A_SQUARE, {1.0: 0.5, np.sqrt(2): -0.25, 2.0: 0.35},
                                 b=[1.0], nu=[3.5]), 6),
    ])
    def test_a_base_rule_the_exact_classes_pass_is_still_refined(self, name, A, V, E):
        # the two cases that refuted the exact-class shortcut: both exact
        # classes are reproduced by the base rule, which is nevertheless
        # 2e-9 / 9e-9 of scale off the torus; the ladder lands at 1e-14
        data = [V.fourier_terms(A)] * E
        base, scale, p0, p1 = C._fixed_rule_classes(data, A, C._TS_M, C._TS_H, None)
        ex0, ex1 = C._exact_compact_classes(data, A)
        tol = C._CYCLE_SELF_BAND * max(abs(base), scale)
        assert abs(p0 - ex0) <= tol and abs(p1 - ex1) <= tol      # the shortcut would accept
        ref, self_err = self._torus_reference(V, A, E)
        assert self_err <= 1e-11 * scale                            # the reference is converged
        assert abs(base - ref) > C._CYCLE_SELF_BAND * scale         # and the base rule is off
        val = zeta_circle([V] * E, A)
        assert abs(val - ref) <= 1e-12 * scale                      # measured 9e-15

    def test_a_dense_table_the_base_rule_misses_still_climbs(self, monkeypatch):
        rng = np.random.default_rng(0)
        dense = {}
        for i in range(-2, 3):
            for j in range(-2, 3):
                if (i, j) != (0, 0) and (i, j) > (-i, -j):
                    v = float(rng.uniform(-2, 2))
                    dense[(i, j)] = v
                    dense[(-i, -j)] = v
        dense[(0, 0)] = 0.3
        V = Interaction.from_table(dense, b=[1.0], nu=[4.5])
        calls = self._count_rules(monkeypatch)
        val = zeta_circle([V] * 8, np.eye(2))
        assert len(calls) >= 2 and calls[0] == C._TS_M    # base refused, ladder climbed
        data = [V.fourier_terms(np.eye(2))] * 8
        f, n_ang = C._TS_REFINED[2][0]
        refined, scale, _, _ = C._fixed_rule_classes(
            data, np.eye(2), f * C._TS_M, C._TS_H / f, n_ang)
        base, _, _, _ = C._fixed_rule_classes(data, np.eye(2), C._TS_M, C._TS_H, None)
        assert abs(base - refined) > 1e-9 * scale               # the base really is off (measured 4e-7)
        assert val == refined or abs(val - refined) <= 1e-9 * scale


class TestSheetFold:
    """At d = 3 the compact path sums over one representative sheet per
    class of sheets the lattice AND every table leave invariant -- a
    reassociation of the same sum, 12x fewer nodes on the cubic cell."""

    @staticmethod
    def _tables(kernels, A):
        return [(lab, val) for _, lab, val in (K.fourier_terms(A) for K in kernels) if len(val)]

    def test_cubic_shell_table_folds_twelve_sheets_to_one(self):
        V = Interaction.from_shells(A_CUBIC, {1.0: 0.5, np.sqrt(2): 0.2}, b=[0.1], nu=[4.0], total=True)
        data = [V.fourier_terms(A_CUBIC)] * 4
        idx, W, reps = C._ts_fold(A_CUBIC, C._TS_M, C._TS_H, None, self._tables([V], A_CUBIC))
        ys, wt = C._ts_grid(3, C._TS_M, C._TS_H, None)
        assert reps == (0,) and len(idx) == len(wt) // 12
        assert np.allclose(W, 12.0 * wt[idx], rtol=1e-15, atol=0.0)   # equal weights on every sheet
        full = C._fixed_rule_classes(data, A_CUBIC, C._TS_M, C._TS_H, None)
        fold = C._fixed_rule_classes(data, A_CUBIC, C._TS_M, C._TS_H, None, fold=True)
        for a, b in zip(full, fold):                 # value, scale, both classes
            assert b == pytest.approx(a, rel=1e-12, abs=0.0)   # measured 2e-14

    def test_a_table_without_the_cubic_symmetry_folds_less(self):
        # x shell 0.5, y and z shells 0.3: the sheets related by a
        # permutation that moves x are told apart; sign flips still fold
        tab = {(1, 0, 0): 0.5, (-1, 0, 0): 0.5, (0, 1, 0): 0.3, (0, -1, 0): 0.3,
               (0, 0, 1): 0.3, (0, 0, -1): 0.3}
        W = Interaction.from_table(tab, b=[0.1], nu=[4.0])
        data = [W.fourier_terms(A_CUBIC)] * 4
        idx, _, reps = C._ts_fold(A_CUBIC, C._TS_M, C._TS_H, None, self._tables([W], A_CUBIC))
        assert 1 < len(reps) < 12                   # measured (0, 1, 2)
        full = C._fixed_rule_classes(data, A_CUBIC, C._TS_M, C._TS_H, None)
        fold = C._fixed_rule_classes(data, A_CUBIC, C._TS_M, C._TS_H, None, fold=True)
        assert fold[0] == pytest.approx(full[0], rel=1e-12, abs=0.0)
        # and the symmetric cross with the same labels folds to one class
        Wsym = Interaction.from_table({m: 0.5 for m in tab}, b=[0.1], nu=[4.0])
        assert C._ts_fold(A_CUBIC, C._TS_M, C._TS_H, None, self._tables([Wsym], A_CUBIC))[2] == (0,)

    def test_a_cell_without_sheet_symmetry_does_not_fold(self):
        A = np.array([[1.0, 0.2, 0.0], [0.0, 1.1, 0.1], [0.0, 0.0, 0.9]])
        V = Interaction.from_shells(A, {1.0: 0.5}, b=[0.1], nu=[4.0], total=True)
        data = [V.fourier_terms(A)] * 3
        assert C._ts_fold(A, C._TS_M, C._TS_H, None, self._tables([V], A)) is None
        full = C._fixed_rule_classes(data, A, C._TS_M, C._TS_H, None)
        fold = C._fixed_rule_classes(data, A, C._TS_M, C._TS_H, None, fold=True)
        assert fold == full                          # bit for bit: nothing folded

    def test_a_2d_rule_never_folds(self):
        V = Interaction.from_shells(A_HEX, {1.0: 0.5}, b=[0.1], nu=[3.0], total=True)
        assert C._ts_fold(A_HEX, C._TS_M, C._TS_H, None, self._tables([V], A_HEX)) is None

    def test_the_transform_memo_keys_on_the_rule_and_the_node_subset(self):
        C._ts_table_clear()
        V = Interaction.from_shells(A_CUBIC, {1.0: 0.5}, b=[0.1], nu=[4.0], total=True)
        data = [V.fourier_terms(A_CUBIC)] * 3
        C._fixed_rule_classes(data, A_CUBIC, C._TS_M, C._TS_H, None)
        C._fixed_rule_classes(data, A_CUBIC, C._TS_M, C._TS_H, None, fold=True)
        lengths = sorted(v.shape[0] for v in C._TS_TRANSFORM_CACHE.values())
        ys, _ = C._ts_grid(3, C._TS_M, C._TS_H, None)
        assert lengths == [len(ys) // 12, len(ys)]   # two entries, unfolded and folded
        assert C._TS_TRANSFORM_BYTES == sum(v.nbytes for v in C._TS_TRANSFORM_CACHE.values())
        C._ts_table_clear()
        assert not C._TS_TRANSFORM_CACHE and C._TS_TRANSFORM_BYTES == 0
