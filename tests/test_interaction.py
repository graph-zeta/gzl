# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""Unit oracles for :mod:`gzl.interaction`: constructors and invariants,
bit-identity of the sampled kernels against the engines' own builders,
exactness of the lazy products, the bridge closed form against Epstein
zeta and brute force, fingerprints, and the boundary normaliser
``coerce_nu``."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from epsteinlib import epstein_zeta

from gzl import Interaction, InteractionSupportError, GraphZetaError
from gzl import frontend
from gzl.direct_sum import _conv_kernel_diff
from gzl.interaction import (
    KEY_DECIMALS,
    POWER_BOX,
    POWER_TORUS,
    _call_vector,
    _KernelProduct,
    _lattice_fingerprint,
    coerce_nu,
)
from gzl.tensor_network import _balanced_z_axis, _edge_kernel_torus

A_CHAIN = np.array([[1.0]])
A_SQUARE = np.eye(2)
A_SQUARE_HALF = 0.5 * np.eye(2)
A_SHEARED = np.array([[1.0, 0.3], [0.0, 1.1]])
A_TRIANGULAR = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
A_CUBIC = np.eye(3)
A_BCC = 0.5 * np.array([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]])
A_FCC = 0.5 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
# Shortest lattice vector A(-1, 1) = (-0.1, 0.436), |.| = 0.447: NOT a
# primitive column (both columns have length ~1), so the NN shell is not the
# column guess.
A_SHEARED_SHORT = np.array([[1.0, 0.9], [0.0, 0.436]])
# Second column 1e-10 longer than the first: ONE shell to the engines'
# 1e-9 shell rule (four labels), two shells to an exact reading.
A_NEAR_DEGENERATE = np.diag([1.0, 1.0 + 1e-10])
NN_LATTICES = [A_CHAIN, A_SQUARE, A_SQUARE_HALF, A_TRIANGULAR, A_CUBIC, A_BCC, A_FCC,
               A_SHEARED_SHORT]


def torus_labels(n: int, d: int) -> np.ndarray:
    """The balanced-window integer labels of an ``n^d`` torus, ``(n,)*d + (d,)``."""
    z = _balanced_z_axis(n)
    return np.stack(np.meshgrid(*([z] * d), indexing="ij"), axis=-1)


def box_labels(L: int, d: int) -> np.ndarray:
    """The ``[-2L, 2L]^d`` difference grid the box engine samples on."""
    ax = np.arange(-2 * L, 2 * L + 1)
    return np.stack(np.meshgrid(*([ax] * d), indexing="ij"), axis=-1)


def cross_table(d: int, val: float, origin: float = 0.0) -> dict:
    """``val`` on the 2d axis-neighbours, ``origin`` at 0."""
    tab = {}
    for i in range(d):
        for s in (-1, 1):
            m = [0] * d
            m[i] = s
            tab[tuple(m)] = val
    if origin != 0.0:
        tab[(0,) * d] = origin
    return tab


Z0 = lambda nu, A: float(np.real(epstein_zeta(nu, A, np.zeros(A.shape[0]), np.zeros(A.shape[0]))))  # noqa: E731


# ---------------------------------------------------------------------------
# Constructors and invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_plain_power_law_flags(self):
        V = Interaction(b=[1.0], nu=[3.0])
        assert V.is_plain_power_law and V.is_pure_power_law and not V.has_compact
        assert V.tail_exponent == 3.0 and V.support_radius == 0 and V.dim is None
        assert V.as_plain_float() == 3.0
        assert Interaction.power_law(2.5).as_plain_float() == 2.5
        assert Interaction.power_law(2.5, b=2.0).as_plain_float() is None

    def test_terms_are_sorted_merged_and_pruned(self):
        V = Interaction(b=[0.5, 1.0, 0.25, -0.25], nu=[5.0, 3.0, 5.0, 4.0])
        assert V.nu == (3.0, 4.0, 5.0)
        assert V.b == (1.0, -0.25, 0.75)
        V = Interaction(b=[1.0, -1.0, 2.0], nu=[3.0, 3.0, 4.0])
        assert V.nu == (4.0,) and V.b == (2.0,)

    def test_odd_table_raises(self):
        with pytest.raises(ValueError, match="even"):
            Interaction.from_table({(1, 0): 0.3, (-1, 0): 0.2})
        with pytest.raises(ValueError, match="even"):
            Interaction.from_table({(1,): 0.3})

    def test_zero_table_entries_are_dropped(self):
        V = Interaction.from_table({(1,): 0.0, (-1,): 0.0}, b=[1.0], nu=[3.0])
        assert V.compact == () and V.is_plain_power_law

    def test_origin_is_a_legitimate_table_entry(self):
        V = Interaction.from_table({(0,): 0.7, (1,): 0.3, (-1,): 0.3})
        assert dict(V.compact)[(0,)] == 0.7
        assert V.tail_exponent == math.inf and V.support_radius == 1

    def test_rejections(self):
        with pytest.raises(ValueError, match="at least one"):
            Interaction()
        with pytest.raises(ValueError, match="nearest_neighbour"):
            Interaction(b=[1.0], nu=[np.inf])
        with pytest.raises(ValueError, match="positive"):
            Interaction(b=[1.0], nu=[0.0])
        with pytest.raises(ValueError, match="same length"):
            Interaction(b=[1.0, 2.0], nu=[3.0])
        with pytest.raises(ValueError, match="not integer"):
            Interaction.from_table({(0.5,): 1.0, (-0.5,): 1.0})
        with pytest.raises(ValueError, match="dimensions"):
            Interaction.from_table({(1,): 1.0, (-1,): 1.0, (1, 0): 1.0, (-1, 0): 1.0})
        with pytest.raises(TypeError):
            Interaction(b=[1.0], nu=[3.0], label=3)

    def test_error_hierarchy(self):
        assert issubclass(InteractionSupportError, GraphZetaError)
        assert frontend.GraphZetaError is GraphZetaError

    def test_b_defaults_to_ones_when_only_nu_is_given(self):
        V = Interaction.from_table(cross_table(2, 0.3), nu=[3.0, 5.0])
        assert V.b == (1.0, 1.0) and V.nu == (3.0, 5.0)
        assert Interaction(nu=[2.5]) == Interaction.power_law(2.5)
        with pytest.raises(ValueError, match="same length"):
            Interaction(b=[1.0], nu=[])          # b without nu still raises

    def test_constructor_accepts_a_lattice_matrix(self):
        tab = cross_table(2, 0.3)
        V = Interaction(compact=tab, lattice=A_SQUARE)
        assert V.lattice == Interaction.from_table(tab, lattice=A_SQUARE).lattice
        assert V.lattice == _lattice_fingerprint(A_SQUARE)
        assert Interaction(compact=tab, lattice=V.lattice) == V
        # A lattice NAME is that matrix (tests/test_lattice_names.py);
        # anything else a str could be is refused with the names listed.
        assert Interaction(compact=tab, lattice="square") == V
        with pytest.raises(ValueError, match="'triangular'"):
            Interaction(compact=tab, lattice="hexagonal")
        with pytest.raises(TypeError, match="lattice"):
            Interaction(compact=tab, lattice=object())

    def test_fingerprint_normalises_negative_zero(self):
        A1 = np.array([[1.0, 1e-17], [0.0, 1.0]])
        A2 = np.array([[1.0, -1e-17], [0.0, 1.0]])
        assert _lattice_fingerprint(A1) == _lattice_fingerprint(A2) == _lattice_fingerprint(np.eye(2))
        V = Interaction.from_function(lambda x: np.exp(-np.linalg.norm(x, axis=-1)), A1, 1.5)
        V.sample(torus_labels(5, 2), A2)         # the same cell to 12 decimals: accepted


class TestFromFunction:
    @pytest.mark.parametrize("A", [A_SQUARE, A_SHEARED, A_TRIANGULAR, A_CHAIN, A_CUBIC])
    def test_label_set_is_the_ball_and_values_are_radial(self, A):
        radius = 2.3
        V = Interaction.from_function(lambda x: np.exp(-np.linalg.norm(x, axis=-1)),
                                      A, radius, b=[1.0], nu=[4.0])
        d = A.shape[0]
        want = {}
        for m in itertools.product(range(-5, 6), repeat=d):
            r = float(np.linalg.norm(A @ np.asarray(m, dtype=float)))
            if r <= radius:
                want[m] = math.exp(-r)
        got = dict(V.compact)
        assert set(got) == set(want)
        for m, v in want.items():
            assert got[m] == pytest.approx(v, rel=1e-15, abs=0.0)
        assert got[(0,) * d] == 1.0          # V(0) is sampled, not zeroed
        assert V.lattice is not None

    def test_single_point_callable_is_accepted(self):
        V = Interaction.from_function(lambda x: 0.5 if float(np.linalg.norm(x)) < 1.5 else 0.25,
                                      A_SQUARE, 2.0)
        got = dict(V.compact)
        assert got[(1, 0)] == 0.5 and got[(2, 0)] == 0.25 and got[(0, 0)] == 0.5
        assert got[(1, 1)] == 0.5      # |A(1,1)| = sqrt(2) < 1.5

    def test_anisotropic_even_function(self):
        # V(x) = x_1^2 - 0.5 x_2^2 depends on the direction, is even, and
        # is sampled on the physical points A m of the sheared cell.
        V = Interaction.from_function(lambda x: x[..., 0] ** 2 - 0.5 * x[..., 1] ** 2,
                                      A_SHEARED, 1.6)
        got = dict(V.compact)
        x10 = A_SHEARED @ np.array([1.0, 0.0])
        x01 = A_SHEARED @ np.array([0.0, 1.0])
        assert got[(1, 0)] == pytest.approx(x10[0] ** 2 - 0.5 * x10[1] ** 2, rel=1e-15)
        assert got[(0, 1)] == pytest.approx(x01[0] ** 2 - 0.5 * x01[1] ** 2, rel=1e-15)
        assert got[(1, 0)] != got[(0, 1)]
        assert got[(-1, 0)] == got[(1, 0)]
        assert (0, 0) not in got            # V(0) = 0 is dropped as an exact zero

    def test_odd_function_raises_and_ulp_odd_is_symmetrised(self):
        with pytest.raises(ValueError, match="even"):
            Interaction.from_function(lambda x: x[..., 0], A_SQUARE, 1.5)
        eps = 1e-14
        V = Interaction.from_function(lambda x: 1.0 + eps * x[..., 0], A_SQUARE, 1.5)
        got = dict(V.compact)
        assert got[(1, 0)] == got[(-1, 0)] == 1.0

    def test_non_finite_at_a_lattice_point_raises(self):
        with pytest.raises(ValueError, match="x = 0"):
            Interaction.from_function(lambda x: 1.0 / np.linalg.norm(x, axis=-1), A_SQUARE, 1.5)

    def test_complex_valued_function_is_refused(self):
        with pytest.raises(ValueError, match="real"):
            Interaction.from_function(
                lambda x: np.exp(-np.linalg.norm(x, axis=-1)) * (1.0 + 0.5j), A_SQUARE, 1.5)
        with pytest.raises(ValueError, match="real"):     # per-point callable, complex value
            Interaction.from_function(
                lambda x: complex(math.exp(-float(np.linalg.norm(x))), 0.5), A_SQUARE, 1.5)

    def test_per_point_callable_raising_any_exception_on_the_batch_falls_back(self):
        # A position table looked up per point: on the (N, d) batch the key
        # is a length-2K tuple, so the batch call raises KeyError.
        table = {(1, 0): 0.5, (-1, 0): 0.5, (0, 1): 0.5, (0, -1): 0.5, (0, 0): 0.0}
        V = Interaction.from_function(
            lambda x: table[tuple(np.round(x).astype(int).ravel().tolist())], A_SQUARE, 1.0)
        assert dict(V.compact) == {m: v for m, v in table.items() if v != 0.0}

    def test_vanishing_table_raises_a_dedicated_message(self):
        with pytest.raises(ValueError, match="vanishes"):
            Interaction.from_function(lambda x: np.zeros(x.shape[:-1]), A_SQUARE, 1.5)
        with pytest.raises(ValueError, match="vanishes"):       # not silently a plain power law
            Interaction.from_function(lambda x: np.zeros(x.shape[:-1]), A_SQUARE, 1.5,
                                      b=[1.0], nu=[3.0])

    def test_evenness_tolerance_is_relative_to_the_table_scale(self):
        amp = 1e-3
        with pytest.raises(ValueError, match="even"):          # odd at 1e-10 relative
            Interaction.from_function(lambda x: amp * (1.0 + 1e-10 * x[..., 0]), A_SQUARE, 1.5)
        V = Interaction.from_function(lambda x: amp * (1.0 + 1e-13 * x[..., 0]), A_SQUARE, 1.5)
        got = dict(V.compact)                                    # odd at 1e-13 relative: symmetrised
        assert got[(1, 0)] == got[(-1, 0)] == pytest.approx(amp, rel=1e-12)
        with pytest.raises(ValueError, match="even"):          # a genuinely odd tiny function
            Interaction.from_function(lambda x: 1e-13 * x[..., 0], A_SQUARE, 1.5)

    def test_from_total_stores_the_deviation(self):
        b, nu = [0.7], [3.5]
        V = Interaction.from_total(lambda x: np.exp(-np.linalg.norm(x, axis=-1)),
                                   A_SQUARE, 2.0, b, nu)
        got = dict(V.compact)
        assert got[(0, 0)] == 1.0
        r = 1.0
        assert got[(1, 0)] == pytest.approx(math.exp(-r) - 0.7 * r ** -3.5, rel=1e-15)
        # inside the radius the SAMPLED kernel is V(r) itself
        S = V.sample(torus_labels(7, 2), A_SQUARE, power=POWER_BOX)
        z = list(_balanced_z_axis(7))
        assert S[z.index(1), z.index(0)] == pytest.approx(math.exp(-1.0), rel=1e-14)


class TestFromShells:
    def test_hexagonal_shells(self):
        J1, J2, Jd = 0.5, 0.2, 0.1
        V = Interaction.from_shells(A_TRIANGULAR, {1.0: J1, np.sqrt(3): J2},
                                    b=[Jd], nu=[3.0], label="J1J2dip")
        assert len(V.compact) == 12 and V.label == "J1J2dip"
        # the tail is subtracted at the SHELL distance: one float per shell
        assert len({v for _, v in V.compact}) == 2
        labels = np.asarray([m for m, _ in V.compact])
        dist = np.linalg.norm(labels @ A_TRIANGULAR.T, axis=1)
        assert np.sum(np.isclose(dist, 1.0)) == 6 and np.sum(np.isclose(dist, np.sqrt(3))) == 6
        # total=True: the sampled kernel equals J on the shell
        S = V.sample(torus_labels(9, 2), A_TRIANGULAR, power=POWER_TORUS)
        z = list(_balanced_z_axis(9))
        assert S[z.index(1), z.index(0)] == pytest.approx(J1, rel=1e-14)
        assert S[z.index(1), z.index(1)] == pytest.approx(J2, rel=1e-14)    # |A(1,1)| = sqrt(3)
        # total=False adds on top of the tail
        W = Interaction.from_shells(A_TRIANGULAR, {1.0: J1}, b=[Jd], nu=[3.0], total=False)
        S = W.sample(torus_labels(9, 2), A_TRIANGULAR, power=POWER_TORUS)
        assert S[z.index(1), z.index(0)] == pytest.approx(J1 + Jd, rel=1e-14)

    def test_missing_shell_raises_and_names_the_shells(self):
        with pytest.raises(ValueError, match="no lattice shell"):
            Interaction.from_shells(A_SQUARE, {1.2: 0.5})

    def test_missing_shell_names_the_nearest_shells_below_and_above(self):
        with pytest.raises(ValueError, match=r"1 \(below\) and 1\.414213562 \(above\)"):
            Interaction.from_shells(A_SQUARE, {1.2: 0.5, 1.0: 0.3})
        with pytest.raises(ValueError, match=r"none \(below\) and 1 \(above\)"):
            Interaction.from_shells(A_SQUARE, {0.5: 0.5})
        with pytest.raises(ValueError, match=r"2 \(below\) and 3 \(above\)"):
            Interaction.from_shells(A_CHAIN, {2.5: 0.5})


class TestEvennessScale:
    def test_a_dominating_on_site_value_does_not_license_an_odd_coupling(self):
        # the evenness scale is the largest OFF-ORIGIN |V|: with V(0) = 1e6
        # an odd part of 1e-7 on the couplings (5e-7 relative to them) is
        # refused, not symmetrised away
        def V(x):
            r = np.linalg.norm(x, axis=-1)
            return np.where(r == 0, 1e6, np.exp(-r) + 1e-7 * x[..., 0])
        with pytest.raises(ValueError, match="must be even"):
            Interaction.from_function(V, np.eye(1), 2.0)
        # the same function without the odd part is accepted with V(0) kept
        T = Interaction.from_function(lambda x: np.where(np.linalg.norm(x, axis=-1) == 0, 1e6,
                                                          np.exp(-np.linalg.norm(x, axis=-1))),
                                      np.eye(1), 2.0)
        assert dict(T.compact)[(0,)] == 1e6

    def test_missing_shell_message_states_the_match_rule_with_full_precision(self):
        A_hex = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
        with pytest.raises(ValueError, match=r"1\.732050808 \(above\).*1e-09 relative"):
            Interaction.from_shells(A_hex, {1.7320508: 0.5})


    def test_a_dotted_toml_key_in_shells_is_named(self):
        with pytest.raises(ValueError, match="DOTTED key"):
            Interaction.from_config({"b": [1.0], "nu": [3.0], "shells": {"1": {"0": 0.5}}}, np.eye(2))

    def test_a_disjoint_compact_product_is_refused_as_the_zero_kernel(self):
        T1 = Interaction.from_table({(1,): 0.5, (-1,): 0.5})
        T3 = Interaction.from_table({(2,): 3.0, (-2,): 3.0})
        with pytest.raises(ValueError, match="zero kernel"):
            T1 * T3
        T13 = Interaction.from_table({(1,): 0.5, (-1,): 0.5, (2,): 1.0, (-2,): 1.0})
        assert (T1 * T13).support_radius == 1              # a non-empty intersection is fine


class TestNearestNeighbour:
    @pytest.mark.parametrize("A", NN_LATTICES)
    @pytest.mark.parametrize("n", [5, 6])
    def test_bit_equals_the_nu_inf_torus_kernel(self, A, n):
        d = A.shape[0]
        K = _edge_kernel_torus(np.inf, A, n)
        S = Interaction.nearest_neighbour(A).sample(torus_labels(n, d), A)
        assert np.array_equal(K, S)
        assert Interaction.nearest_neighbour(A).tail_exponent == math.inf

    @pytest.mark.parametrize("A", [A_NEAR_DEGENERATE, A_SHEARED_SHORT])
    @pytest.mark.parametrize("n", [5, 6])
    def test_shell_slack_matches_both_engines(self, A, n):
        d = A.shape[0]
        NN = Interaction.nearest_neighbour(A)
        assert np.array_equal(_edge_kernel_torus(np.inf, A, n), NN.sample(torus_labels(n, d), A))
        assert np.array_equal(_conv_kernel_diff(np.inf, A, 2, d),
                              NN.sample(box_labels(2, d), A, power=POWER_BOX))
        assert len(NN.compact) == (4 if A is A_NEAR_DEGENERATE else 2)


# ---------------------------------------------------------------------------
# Sampling: bit identity with the engines' builders and the window guard
# ---------------------------------------------------------------------------

class TestSample:
    @pytest.mark.parametrize("A", [A_CHAIN, A_SQUARE, A_SHEARED, A_TRIANGULAR])
    @pytest.mark.parametrize("n", [7, 8])
    def test_power_law_bit_equals_torus_builder(self, A, n):
        d = A.shape[0]
        for nu in (1.5, 2.5, 4.0):
            K = _edge_kernel_torus(nu, A, n)
            S = Interaction.power_law(nu).sample(torus_labels(n, d), A, power=POWER_TORUS)
            assert np.array_equal(K, S)

    @pytest.mark.parametrize("A", [A_CHAIN, A_SQUARE, A_SHEARED])
    def test_power_law_bit_equals_box_builder(self, A):
        d = A.shape[0]
        L = 3
        for nu in (1.5, 2.5):
            K = _conv_kernel_diff(nu, A, L, d)
            S = Interaction.power_law(nu).sample(box_labels(L, d), A, power=POWER_BOX)
            assert np.array_equal(K, S)

    def test_two_forms_differ_so_the_form_must_come_from_the_engine(self):
        labels = torus_labels(64, 1)
        t = Interaction.power_law(2.5).sample(labels, A_CHAIN, power=POWER_TORUS)
        b = Interaction.power_law(2.5).sample(labels, A_CHAIN, power=POWER_BOX)
        assert np.allclose(t, b) and not np.array_equal(t, b)

    def test_table_is_scattered_and_origin_kept(self):
        V = Interaction.from_table(cross_table(2, 0.3, origin=0.7), b=[1.0], nu=[3.0])
        S = V.sample(torus_labels(7, 2), A_SQUARE)
        z = list(_balanced_z_axis(7))
        assert S[z.index(0), z.index(0)] == 0.7
        assert S[z.index(1), z.index(0)] == 1.0 + 0.3
        assert S[z.index(2), z.index(0)] == 1.0 / (2.0 ** 3.0)

    def test_window_guard(self):
        V = Interaction.from_table({(2,): 0.5, (-2,): 0.5}, b=[1.0], nu=[3.0])
        with pytest.raises(InteractionSupportError, match="support"):
            V.sample(torus_labels(4, 1), A_CHAIN)          # window holds +2 but not -2
        V.sample(torus_labels(5, 1), A_CHAIN)                # (n - 1)//2 = 2 fits
        with pytest.raises(InteractionSupportError):
            V.sample(np.array([[0], [1], [-1]]), A_CHAIN)

    def test_lattice_and_dimension_checks(self):
        V = Interaction.from_function(lambda x: np.exp(-np.linalg.norm(x, axis=-1)),
                                      A_SQUARE, 1.5)
        with pytest.raises(ValueError, match="different lattice"):
            V.sample(torus_labels(7, 2), A_SHEARED)
        T = Interaction.from_table(cross_table(2, 0.3))
        with pytest.raises(ValueError, match="dimensional"):
            T.sample(torus_labels(7, 1), A_CHAIN)


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

class TestProducts:
    def test_product_samples_as_the_pointwise_product(self):
        labels = torus_labels(9, 2)
        I1 = Interaction(b=[1.0, -0.2], nu=[3.0, 5.0])
        I2 = Interaction.from_table(cross_table(2, 0.3, origin=0.7), b=[1.0], nu=[3.0])
        I3 = Interaction.from_table(cross_table(2, -0.4))
        factors = [I1, I2, I3, I1, I2, I3]
        for k in range(2, 7):
            P = _KernelProduct(factors[:k])
            want = factors[0].sample(labels, A_SQUARE)
            for f in factors[1:k]:
                want = want * f.sample(labels, A_SQUARE)
            assert np.array_equal(P.sample(labels, A_SQUARE), want)
        P = I2 ** 3
        want = I2.sample(labels, A_SQUARE) ** 1
        want = want * I2.sample(labels, A_SQUARE) * I2.sample(labels, A_SQUARE)
        assert np.array_equal(P.sample(labels, A_SQUARE), want)

    def test_product_bookkeeping(self):
        I1 = Interaction(b=[1.0, -0.2], nu=[3.0, 5.0])
        T = Interaction.from_table({(0,): 0.5, (2,): 0.1, (-2,): 0.1})
        P = I1 * T
        assert isinstance(P, _KernelProduct)
        assert P.tail_exponent == math.inf and P.support_radius == 2 and P.has_compact
        assert not P.is_pure_power_law
        Q = I1 * I1
        assert Q.tail_exponent == 6.0 and Q.is_pure_power_law and Q.key()[0] == "H"
        assert (I1 ** 1) is I1
        assert (I1 * I1 * I1).factors == (I1, I1, I1)
        assert (I1 * (I1 * T)).factors == (I1, I1, T)
        assert Interaction.power_law(3.0) ** 2 == _KernelProduct((Interaction.power_law(3.0),) * 2)
        assert (Interaction.power_law(3.0) ** 2).as_plain_float() == 6.0
        assert P.as_plain_float() is None
        with pytest.raises(ValueError):
            I1 ** 0
        with pytest.raises(ValueError):
            I1 ** 1.5

    def test_scaling_and_sums_are_exact(self):
        I1 = Interaction(b=[1.0, -0.2], nu=[3.0, 5.0])
        T = Interaction.from_table(cross_table(2, 0.3, origin=0.7))
        assert (0.5 * I1).b == (0.5, -0.1)
        assert (np.float64(0.5) * I1).b == (0.5, -0.1)
        assert (I1 * 2).b == (2.0, -0.4)
        S = I1 + T
        assert S.nu == (3.0, 5.0) and dict(S.compact)[(0, 0)] == 0.7
        assert (S - T).compact == () and (S - T).nu == (3.0, 5.0)
        assert (-T).compact[0][1] == -0.3 or (-T).compact[0][1] == -0.7
        with pytest.raises(ValueError):
            0.0 * I1

    def test_all_compact_product_support_is_the_intersection(self):
        T1 = Interaction.from_table(cross_table(2, 0.3))                                   # R = 1
        T2 = Interaction.from_table({(2, 0): 0.5, (-2, 0): 0.5, (1, 0): 0.2, (-1, 0): 0.2})  # R = 2
        M = Interaction.from_table(cross_table(2, 0.3), b=[1.0], nu=[3.0])
        P = T1 * T2
        assert P.support_radius == 1
        assert (Interaction.power_law(3.0) * T2).support_radius == 2      # a tail factor: max
        assert (T1 * M * T2).support_radius == 2
        big = torus_labels(9, 2)
        assert np.array_equal(P.sample(big, A_SQUARE),
                              T1.sample(big, A_SQUARE) * T2.sample(big, A_SQUARE))
        small = torus_labels(3, 2)              # holds +-1 only: T2 alone is refused ...
        with pytest.raises(InteractionSupportError):
            T2.sample(small, A_SQUARE)
        S = P.sample(small, A_SQUARE)           # ... the product needs only the intersection
        z = list(_balanced_z_axis(3))
        assert S[z.index(1), z.index(0)] == 0.3 * 0.2 and S[z.index(0), z.index(1)] == 0.0
        assert P.lattice_sum(A_SQUARE) == pytest.approx(2 * 0.3 * 0.2, rel=1e-15)


# ---------------------------------------------------------------------------
# The bridge closed form
# ---------------------------------------------------------------------------

class TestLatticeSum:
    @pytest.mark.parametrize("A", [A_CHAIN, A_SQUARE, A_SHEARED, A_CUBIC])
    def test_power_law_is_epstein_zeta(self, A):
        for nu in (A.shape[0] + 0.5, A.shape[0] + 2.0):
            got = Interaction.power_law(nu).lattice_sum(A)
            assert got == pytest.approx(Z0(nu, A), rel=1e-15, abs=0.0)

    def test_two_terms_are_linear(self):
        V = Interaction(b=[1.0, -0.2], nu=[3.0, 5.0])
        want = Z0(3.0, A_SQUARE) - 0.2 * Z0(5.0, A_SQUARE)
        assert V.lattice_sum(A_SQUARE) == pytest.approx(want, rel=1e-15, abs=0.0)

    def test_compact_only_is_the_table_sum(self):
        T = Interaction.from_table(cross_table(2, 0.3, origin=0.7))
        assert T.lattice_sum(A_SQUARE) == 4 * 0.3 + 0.7

    def test_finite_k_compact_is_the_structure_factor(self):
        T = Interaction.from_table(cross_table(2, 0.3, origin=0.7))
        k = np.array([0.1, 0.25])
        want = 0.7 + 0.3 * (2 * math.cos(2 * math.pi * 0.1) + 2 * math.cos(2 * math.pi * 0.25))
        assert T.lattice_sum(A_SQUARE, k) == pytest.approx(want, rel=1e-15)
        grid = np.array([[0.0, 0.0], k, [0.5, 0.5]])
        out = T.lattice_sum(A_SQUARE, grid)
        # The scalar and the batch call reach the same sum through a
        # (1, M) and a (3, M) matrix-vector product; BLAS may order the two
        # differently (they agreed bitwise on macOS/Accelerate and differed
        # on Linux CI), so the identity is exact-to-round-off, not bitwise.
        assert out.shape == (3,)
        assert out[0] == pytest.approx(T.lattice_sum(A_SQUARE), rel=1e-15, abs=0.0)
        assert out[1] == pytest.approx(want, rel=1e-15)

    def test_finite_k_power_law_is_epstein(self):
        V = Interaction(b=[1.0, 0.5], nu=[3.0, 4.5])
        k = np.array([0.3, 0.1])
        Astar = np.linalg.inv(A_SQUARE.T)
        want = sum(bj * float(np.real(epstein_zeta(nj, A_SQUARE, np.zeros(2), Astar @ k)))
                   for bj, nj in V.terms)
        assert V.lattice_sum(A_SQUARE, k) == pytest.approx(want, rel=1e-15, abs=0.0)

    def test_pure_power_law_product_is_the_summed_exponent(self):
        P = Interaction.power_law(3.0) ** 2
        assert P.lattice_sum(A_SQUARE) == pytest.approx(Z0(6.0, A_SQUARE), rel=1e-15, abs=0.0)
        Q = Interaction(b=[1.0, -0.2], nu=[3.0, 5.0]) ** 2
        want = Z0(6.0, A_SQUARE) - 0.4 * Z0(8.0, A_SQUARE) + 0.04 * Z0(10.0, A_SQUARE)
        assert Q.lattice_sum(A_SQUARE) == pytest.approx(want, rel=1e-14, abs=0.0)

    def test_compact_product_is_the_finite_sum(self):
        T = Interaction.from_table(cross_table(2, 0.3, origin=0.7))
        want = 4 * 0.3 ** 3 + 0.7 ** 3
        assert (T ** 3).lattice_sum(A_SQUARE) == pytest.approx(want, rel=1e-15)

    def test_mixed_product_against_brute_force(self):
        # steep tails so the finite window is converged to 1e-14
        M = Interaction(b=[1.0, -0.2], nu=[8.0, 10.0])
        T = Interaction.from_table(cross_table(2, 0.3, origin=0.7), b=[1.0], nu=[8.0])
        P = (M * T) ** 2
        ax = np.arange(-25, 26)
        labels = np.stack(np.meshgrid(ax, ax, indexing="ij"), axis=-1)
        brute = float(np.sum(P.sample(labels, A_SQUARE, power=POWER_BOX)))
        assert P.lattice_sum(A_SQUARE) == pytest.approx(brute, rel=1e-13)
        k = np.array([0.2, 0.35])
        phase = np.cos(2 * np.pi * (labels @ k))
        brute_k = float(np.sum(P.sample(labels, A_SQUARE, power=POWER_BOX) * phase))
        assert P.lattice_sum(A_SQUARE, k) == pytest.approx(brute_k, rel=1e-12)

    def test_d1_one_dimensional_k_array_is_a_batch(self):
        T = Interaction.from_table({(1,): 0.4, (-1,): 0.4, (0,): 0.2}, b=[1.0], nu=[3.0])
        kk = np.array([0.0, 0.25, 0.5])
        out = T.lattice_sum(A_CHAIN, kk)
        assert out.shape == (3,)
        for i, k in enumerate(kk):
            assert out[i] == pytest.approx(T.lattice_sum(A_CHAIN, np.array([k])), rel=1e-13)
        assert out[1] != out[0] and out[2] != out[0]
        assert isinstance(T.lattice_sum(A_CHAIN, np.array([0.25])), float)
        assert isinstance(T.lattice_sum(A_CHAIN, 0.25), float)
        assert T.lattice_sum(A_CHAIN, kk.reshape(-1, 1)).shape == (3,)

    def test_one_dimensional_k_of_the_wrong_length_raises(self):
        T = Interaction.from_table(cross_table(2, 0.3))
        with pytest.raises(ValueError, match=r"\(N, 2\)"):
            T.lattice_sum(A_SQUARE, np.array([0.1, 0.2, 0.3]))
        with pytest.raises(ValueError, match=r"\(N, 2\)"):
            T.lattice_sum(A_SQUARE, np.zeros((3, 3)))
        with pytest.raises(ValueError, match=r"\(N, 2\)"):
            T.lattice_sum(A_SQUARE, 0.25)

    def test_multiset_expansion_merges_commensurate_exponents(self):
        V = Interaction(b=[1.0, 1.0, 1.0], nu=[2.0, 3.0, 4.0])
        P = V ** 13
        assert len(P._tail_expansion()) == 27          # 26..52, not C(15, 2) = 105
        W = Interaction(b=[1.0, 1.0, 1.0], nu=[2.3, 3.1, 5.7])
        assert len((W ** 13)._tail_expansion()) == 105


# ---------------------------------------------------------------------------
# Fingerprints and labels
# ---------------------------------------------------------------------------

class TestKey:
    def test_key_decimals_match_the_frontend(self):
        assert KEY_DECIMALS == frontend._NU_KEY_DECIMALS

    def test_pinned_key(self):
        V = Interaction(b=[1.0, -0.2], nu=[3.0, 5.0])
        assert V.key() == ("I", (3.0, 5.0), (1.0, -0.2), (), b"")
        T = Interaction.from_table({(1,): 0.3, (-1,): 0.3}, b=[1.0], nu=[3.0])
        assert T.key() == ("I", (3.0,), (1.0,), (((-1,), 0.3), ((1,), 0.3)), b"")
        F = Interaction.from_function(lambda x: 0.3 * (np.linalg.norm(x, axis=-1) < 1.5),
                                      A_CHAIN, 1.5)
        assert F.key()[4] == F.lattice and F.key()[:4] == ("I", (), (), F.compact)
        G = Interaction.from_table(dict(F.compact))
        assert G.key() != F.key() and G.compact == F.compact      # same table, other provenance

    def test_equal_tails_different_tables_differ(self):
        T1 = Interaction.from_table(cross_table(2, 0.3), b=[1.0], nu=[3.0])
        T2 = Interaction.from_table(cross_table(2, 0.31), b=[1.0], nu=[3.0])
        assert T1.tail_exponent == T2.tail_exponent
        assert T1.key() != T2.key() and hash(T1.key()) != hash(T2.key())
        assert T1.key() != 3.0 and T1.key() != round(3.0, 12)
        keys = sorted([T2.key(), T1.key(), (T1 * T2).key(), Interaction.power_law(3.0).key()])
        assert len(keys) == 4

    def test_key_payloads_are_python_scalars(self):
        T = Interaction.from_table({(np.int64(1),): np.float64(0.3), (np.int64(-1),): np.float64(0.3)},
                                   b=np.array([1.0]), nu=np.array([3.0]))
        k = T.key()
        assert type(k[1][0]) is float and type(k[2][0]) is float
        assert type(k[3][0][0][0]) is int and type(k[3][0][1]) is float
        assert str(k) == str(Interaction.from_table({(1,): 0.3, (-1,): 0.3}, b=[1.0], nu=[3.0]).key())

    def test_exponents_are_quantised_like_the_frontend_block_cache(self):
        a = Interaction.power_law(3.0, b=0.5)
        assert a.key() == Interaction.power_law(3.0 + 1e-13, b=0.5).key()
        assert a.key() != Interaction.power_law(3.0 + 1e-11, b=0.5).key()

    def test_labels(self):
        assert Interaction.power_law(2.5).default_label == "nu=2.5"
        T = Interaction.from_table(cross_table(2, 0.3))
        # "K#" since the kernel is K: this label reaches the CSV's
        # `interaction` column, so the letter a user sees matches the
        # letter the documentation uses.
        assert T.default_label.startswith("K#") and len(T.default_label) == 12
        assert Interaction.from_table(cross_table(2, 0.3), label="nn").default_label == "nn"
        assert (Interaction.power_law(2.5) * T).default_label == "nu=2.5*" + T.default_label


# ---------------------------------------------------------------------------
# The boundary normaliser
# ---------------------------------------------------------------------------

class TestCoerceNu:
    def test_legacy_forms_are_verbatim(self):
        arr, kern = coerce_nu(2.5, 3)
        assert kern is None and np.array_equal(arr, np.full(3, 2.5)) and arr.dtype == float
        arr, kern = coerce_nu([2.5, 3.0, 3.5], 3)
        assert kern is None and np.array_equal(arr, np.asarray([2.5, 3.0, 3.5]))
        arr, kern = coerce_nu(np.array([[2.5, 3.0]]), 2)
        assert kern is None and arr.shape == (2,)
        arr, kern = coerce_nu(np.inf, 2)
        assert kern is None and np.all(np.isinf(arr))
        with pytest.raises(ValueError, match="nu has length"):
            coerce_nu([2.5, 3.0], 3)
        with pytest.raises(ValueError, match="nu has length"):
            coerce_nu(np.array([2.5, 3.0]), 3)

    def test_plain_interactions_are_demoted(self):
        arr, kern = coerce_nu(Interaction.power_law(2.5), 3)
        assert kern is None and np.array_equal(arr, np.full(3, 2.5))
        arr, kern = coerce_nu([Interaction.power_law(2.5), 3.0], 2)
        assert kern is None and np.array_equal(arr, np.asarray([2.5, 3.0]))
        arr, kern = coerce_nu([Interaction.power_law(3.0) ** 2, 3.0], 2)
        assert kern is None and np.array_equal(arr, np.asarray([6.0, 3.0]))

    def test_genuine_interactions_give_a_uniform_kernel_list(self):
        T = Interaction.from_table(cross_table(2, 0.3), b=[1.0], nu=[3.0])
        arr, kern = coerce_nu(T, 2)
        assert kern == [T, T] and np.array_equal(arr, np.asarray([3.0, 3.0]))
        arr, kern = coerce_nu([2.5, T], 2)
        assert kern[0] == Interaction.power_law(2.5) and kern[1] is T
        assert np.array_equal(arr, np.asarray([2.5, 3.0]))
        C = Interaction.from_table(cross_table(2, 0.3))
        arr, kern = coerce_nu([C, 2.5], 2)
        assert np.isinf(arr[0]) and arr[1] == 2.5
        with pytest.raises(ValueError, match="nu has length"):
            coerce_nu([T, T, T], 2)
        with pytest.raises(ValueError, match="nearest_neighbour"):
            coerce_nu([T, np.inf], 2)

    def test_object_ndarray_takes_the_sequence_branch(self):
        T = Interaction.from_table(cross_table(2, 0.3), b=[1.0], nu=[3.0])
        arr, kern = coerce_nu(np.array([T, 2.5], dtype=object), 2)
        assert kern[0] is T and kern[1] == Interaction.power_law(2.5)
        assert np.array_equal(arr, [3.0, 2.5])
        arr, kern = coerce_nu(np.array([T, T, T]), 3)
        assert kern == [T, T, T]
        arr, kern = coerce_nu(np.array([2.5, 3.0], dtype=object), 2)
        assert kern is None and np.array_equal(arr, [2.5, 3.0]) and arr.dtype == float


# ---------------------------------------------------------------------------
# Config form
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_shells_compact_and_npz(self, tmp_path):
        V = Interaction.from_config({"label": "x", "b": [0.1], "nu": [3.0],
                                     "shells": [[1.0, 0.5], [math.sqrt(3), 0.2]]}, A_TRIANGULAR)
        assert V == Interaction.from_shells(A_TRIANGULAR, {1.0: 0.5, math.sqrt(3): 0.2},
                                            b=[0.1], nu=[3.0], label="x")
        rows = [[1, 0, 0.3], [-1, 0, 0.3], [0, 1, 0.3], [0, -1, 0.3]]
        W = Interaction.from_config({"b": [1.0], "nu": [3.0], "compact": rows}, A_SQUARE)
        assert dict(W.compact) == cross_table(2, 0.3)
        path = tmp_path / "t.npz"
        np.savez(path, labels=np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), values=np.full(4, 0.3))
        U = Interaction.from_config({"b": [1.0], "nu": [3.0], "compact_npz": str(path)}, A_SQUARE)
        assert U.compact == W.compact
        assert Interaction.from_config({"b": [1.0], "nu": [3.0]}, A_SQUARE).is_plain_power_law

    def test_config_rejections(self):
        with pytest.raises(ValueError, match="unknown"):
            Interaction.from_config({"nu": [3.0], "b": [1.0], "foo": 1}, A_SQUARE)
        with pytest.raises(ValueError, match="only one"):
            Interaction.from_config({"shells": [[1.0, 0.5]], "compact": [[1, 0, 0.3]]}, A_SQUARE)
        with pytest.raises(ValueError, match="entries"):
            Interaction.from_config({"compact": [[1, 0.3]]}, A_SQUARE)

    def test_compact_rows_go_through_the_integrality_guard(self):
        with pytest.raises(ValueError, match="not integer"):
            Interaction.from_config({"compact": [[1.5, 0, 0.3], [-1.5, 0, 0.3]]}, A_SQUARE)
        with pytest.raises(ValueError, match="not integer"):
            Interaction.from_config({"compact": [[0.9999999999, 0.3], [-0.9999999999, 0.3]]}, A_CHAIN)
        with pytest.raises(ValueError, match="twice"):
            Interaction.from_config({"compact": [[1, 0.3], [1, 0.4], [-1, 0.3]]}, A_CHAIN)
        V = Interaction.from_config({"compact": [[1.0, 0.3], [-1.0, 0.3]]}, A_CHAIN)
        assert dict(V.compact) == {(1,): 0.3, (-1,): 0.3}      # integral floats are fine

    def test_compact_npz_is_validated(self, tmp_path):
        p = tmp_path / "float_labels.npz"
        np.savez(p, labels=np.array([[0.9999999999, 0.0], [-0.9999999999, 0.0]]), values=[0.3, 0.3])
        with pytest.raises(ValueError, match="not integer"):
            Interaction.from_config({"compact_npz": str(p)}, A_SQUARE)
        p = tmp_path / "missing.npz"
        np.savez(p, labels=np.array([[1, 0], [-1, 0]]))
        with pytest.raises(ValueError, match="values"):
            Interaction.from_config({"compact_npz": str(p)}, A_SQUARE)
        p = tmp_path / "wrong_d.npz"
        np.savez(p, labels=np.array([[1, 0, 0], [-1, 0, 0]]), values=[0.3, 0.3])
        with pytest.raises(ValueError, match=r"\(N, 2\)"):
            Interaction.from_config({"compact_npz": str(p)}, A_SQUARE)
        p = tmp_path / "reflow.npz"        # (2, 3) labels that re-flow to (3, 2)
        np.savez(p, labels=np.array([[1, 0, 0], [0, -1, 0]]), values=[0.4, 0.1, 0.4])
        with pytest.raises(ValueError, match=r"\(N, 2\)"):
            Interaction.from_config({"compact_npz": str(p)}, A_SQUARE)
        p = tmp_path / "chain.npz"         # 1-D labels: allowed in d = 1 only
        np.savez(p, labels=np.array([1, -1]), values=[0.3, 0.3])
        V = Interaction.from_config({"compact_npz": str(p)}, A_CHAIN)
        assert dict(V.compact) == {(1,): 0.3, (-1,): 0.3}
        with pytest.raises(ValueError, match=r"\(N, 2\)"):
            Interaction.from_config({"compact_npz": str(p)}, A_SQUARE)

    def test_shells_accepts_a_mapping_and_refuses_bad_rows(self):
        want = Interaction.from_shells(A_TRIANGULAR, {1.0: 0.5, math.sqrt(3): 0.2}, b=[0.1], nu=[3.0])
        got = Interaction.from_config(
            {"b": [0.1], "nu": [3.0], "shells": {1.0: 0.5, math.sqrt(3): 0.2}}, A_TRIANGULAR)
        assert got == want
        got = Interaction.from_config(                     # TOML inline-table keys are strings
            {"b": [0.1], "nu": [3.0], "shells": {"1.0": 0.5, str(math.sqrt(3)): 0.2}}, A_TRIANGULAR)
        assert got == want
        with pytest.raises(ValueError, match="pair"):
            Interaction.from_config({"shells": [[1.0, 0.5, 0.1]]}, A_SQUARE)
        with pytest.raises(ValueError, match="pair"):
            Interaction.from_config({"shells": [[1.0]]}, A_SQUARE)

    def test_total_must_be_a_bool_and_needs_shells(self):
        with pytest.raises(TypeError, match="bool"):
            Interaction.from_config({"shells": [[1.0, 0.5]], "b": [0.1], "nu": [3.0],
                                     "total": "false"}, A_SQUARE)
        with pytest.raises(ValueError, match="shells"):
            Interaction.from_config({"compact": [[1, 0, 0.3], [-1, 0, 0.3]], "total": True}, A_SQUARE)
        W = Interaction.from_config({"shells": {1.0: 0.5}, "b": [0.1], "nu": [3.0], "total": False},
                                    A_SQUARE)
        assert W == Interaction.from_shells(A_SQUARE, {1.0: 0.5}, b=[0.1], nu=[3.0], total=False)


class TestCompactArray:
    def test_table_alone_is_scattered_with_the_same_guard(self):
        T = Interaction.from_table(cross_table(2, 0.3, origin=0.7), b=[1.0], nu=[3.0])
        labels = torus_labels(7, 2)
        a = T.compact_array(labels)
        z = list(_balanced_z_axis(7))
        assert a[z.index(0), z.index(0)] == 0.7 and a[z.index(1), z.index(0)] == 0.3
        assert a[z.index(2), z.index(0)] == 0.0
        assert np.array_equal(T.sample(labels, A_SQUARE) - a,
                              Interaction.power_law(3.0).sample(labels, A_SQUARE))
        assert np.array_equal(Interaction.power_law(3.0).compact_array(labels), np.zeros((7, 7)))
        with pytest.raises(InteractionSupportError):
            T.compact_array(torus_labels(2, 2))


class TestCallableAndArithmeticEdges:
    """Boundaries of the callable sampler and the kernel arithmetic."""

    def test_a_per_point_callable_is_not_trusted_on_a_shape_coincidence(self):
        # A per-point ``K`` handed the whole (N, d) batch indexes ``x[0]``,
        # which on a batch is the first ROW, so it returns d values.  When
        # K == d that shape matches the expected (K,) and a WRONG table was
        # stored with no exception and no warning.
        V = lambda x: np.asarray(x)[0] ** 2          # noqa: E731 -- per-point
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        assert np.asarray(V(pts)).shape == pts.shape[:-1]      # the coincidence
        assert not np.allclose(np.asarray(V(pts)), [1.0, 16.0, 49.0])
        assert np.allclose(_call_vector(V, pts), [1.0, 16.0, 49.0])

    def test_a_genuinely_vectorised_callable_still_takes_the_batch_path(self):
        calls = {"n": 0}

        def W(x):
            calls["n"] += 1
            return np.sum(np.asarray(x, dtype=float) ** 2, axis=-1)

        pts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        assert np.allclose(_call_vector(W, pts), [5.0, 25.0, 61.0, 113.0])
        assert calls["n"] <= 3          # one batch plus the two end probes

    def test_sum_over_kernels_works_and_a_bare_scalar_does_not(self):
        # ``sum([...])`` starts from the int 0 and is the natural way to
        # build a multi-term kernel from a list of shell contributions
        V1, V2 = Interaction.power_law(3.0), Interaction.power_law(5.0)
        total = sum([V1, V2])
        assert isinstance(total, Interaction) and set(total.nu) == {3.0, 5.0}
        assert total == V1 + V2
        with pytest.raises(TypeError):
            V1 + 1.0
        with pytest.raises(TypeError):
            2.0 + V1
