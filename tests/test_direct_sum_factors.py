# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The box's lazy factor supply.

``direct_sum`` used to materialise every two-vertex kernel as a full
``(size_a, size_b)`` table during the factor build, so the whole initial
factor set was live before a single vertex was eliminated.  It now
stores the difference-range generator plus the index arithmetic that
reads a table out of it, and builds each table at the one step that
consumes it.

What is pinned here is the part a value test cannot see: that the
generator really is *shared* rather than rebuilt per factor, that the
tables are not built behind the shim's back, and that the index
arithmetic is right for every combination of half-box and full-box
endpoints.  ``tests/test_direct_sum_lazy.py`` pins the numerical premise
(generator indexing reproduces the eager table);
``tests/test_executor_goldens.py`` pins that no value moved.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import gzl.direct_sum as ds

K4 = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
PRISM = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3],
                  [0, 3], [1, 4], [2, 5]])
# Two distinct vertex pairs of equal collapsed nu sharing an eliminated
# vertex: the duplicate-shaped-factor case, where a cache keyed on nu
# hands the same generator object to two different factors.
DIAMOND = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])


class TestIndexArithmetic:
    """Reading the generator must reproduce the positions it stands for."""

    @staticmethod
    def _positions(L, d, half):
        axes = ([np.arange(0, L + 1)] if half else [np.arange(-L, L + 1)])
        axes += [np.arange(-L, L + 1)] * (d - 1)
        return np.array(list(itertools.product(*axes)), dtype=float)

    @pytest.mark.parametrize("half_a,half_b",
                             [(a, b) for a in (False, True)
                              for b in (False, True)])
    @pytest.mark.parametrize("L", [2, 3])
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_two_axis_factor_matches_its_positions(self, d, L, half_a, half_b):
        A = np.eye(d)
        nu = 2.5
        gen = ds._conv_kernel_diff(nu, A, L, d)
        got = ds._table_from_generator(
            gen,
            ds._axis_extents(L, d, half_a),
            ds._axis_extents(L, d, half_b),
            ds._gen_offsets(L, d, ds._axis_origins(L, d, half_a),
                            ds._axis_origins(L, d, half_b)),
            d,
        )
        pa = self._positions(L, d, half_a)
        pb = self._positions(L, d, half_b)
        diff = pb[None, :, :] - pa[:, None, :]
        dist = np.linalg.norm(diff @ A.T, axis=-1)
        with np.errstate(divide="ignore"):
            want = dist ** (-nu)
        want[dist == 0.0] = 0.0
        assert np.array_equal(got, want)

    @pytest.mark.parametrize("half", [False, True])
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_one_axis_factor_is_the_root_incident_kernel(self, d, half):
        """The root sits at the origin, so this is K(x_v), not a table row."""
        A = np.eye(d)
        L, nu = 3, 2.5
        gen = ds._conv_kernel_diff(nu, A, L, d)
        org = ds._axis_origins(L, d, half)
        got = ds._row_from_generator(
            gen, ds._axis_extents(L, d, half),
            tuple(int(org[c] + 2 * L) for c in range(d)), d,
        )
        pos = self._positions(L, d, half)
        dist = np.linalg.norm(pos @ A.T, axis=-1)
        with np.errstate(divide="ignore"):
            want = dist ** (-nu)
        want[dist == 0.0] = 0.0
        assert np.array_equal(got, want)

    def test_offsets_differ_between_half_and_full_endpoints(self):
        """Guard against the offset silently collapsing to a constant.

        If ``_gen_offsets`` ignored the origins, every test above would
        still pass on the full/full case — which is the common one — and
        fail only where the Z2 marker is involved.
        """
        L, d = 4, 2
        full = ds._axis_origins(L, d, False)
        half = ds._axis_origins(L, d, True)
        assert ds._gen_offsets(L, d, full, full) != \
            ds._gen_offsets(L, d, half, full)
        assert ds._gen_offsets(L, d, half, full) != \
            ds._gen_offsets(L, d, full, half)


class TestTheGeneratorIsShared:
    """One generator per distinct nu, serving factors and the peel alike."""

    @staticmethod
    def _count_generator_builds(edges, nu, A, L, **kw):
        real = ds._conv_kernel_diff
        calls = []

        def counting(nu_e, A_, L_, d_):
            calls.append(float(nu_e))
            return real(nu_e, A_, L_, d_)

        ds._conv_kernel_diff = counting
        try:
            ds.direct_sum_zero_momentum(
                edges, np.full(len(edges), nu), A, L, root=0, **kw)
        finally:
            ds._conv_kernel_diff = real
        return calls

    def test_uniform_nu_builds_exactly_one_generator(self):
        calls = self._count_generator_builds(K4, 2.5, np.eye(1), 6)
        assert calls == [2.5], (
            f"a uniform-nu graph built {len(calls)} generators; the cache "
            f"is not shared across factors"
        )

    def test_a_larger_graph_still_builds_one(self):
        calls = self._count_generator_builds(PRISM, 2.5, np.eye(1), 5)
        assert calls == [2.5]

    def test_distinct_nu_builds_one_each(self):
        """A merged multi-edge raises one bundle's exponent."""
        real = ds._conv_kernel_diff
        calls = []

        def counting(nu_e, A_, L_, d_):
            calls.append(float(nu_e))
            return real(nu_e, A_, L_, d_)

        nu = np.array([2.5, 2.5, 2.5, 5.0, 2.5, 2.5])
        ds._conv_kernel_diff = counting
        try:
            ds.direct_sum_zero_momentum(K4, nu, np.eye(1), 6, root=0)
        finally:
            ds._conv_kernel_diff = real
        assert sorted(calls) == [2.5, 5.0], calls

    def test_the_peel_reuses_the_factor_build_generator(self):
        """With the gate forced open, the peel must not build its own.

        Before the cache was hoisted to call level, ``_eliminate_all``
        owned a private one — so the peel rebuilt a generator the factor
        build had already made, at every call.
        """
        old_margin, old_bag = ds._FFT_MARGIN, ds._FFT_MIN_BAG
        ds._FFT_MARGIN, ds._FFT_MIN_BAG = 0.0, 0
        try:
            calls = self._count_generator_builds(PRISM, 2.5, np.eye(1), 5)
        finally:
            ds._FFT_MARGIN, ds._FFT_MIN_BAG = old_margin, old_bag
        assert calls == [2.5], (
            f"the peel built its own generator ({len(calls)} total)"
        )

    def test_the_padded_kernel_transform_is_built_once(self):
        """It depends only on (nu, A, L, d), never on phi.

        This matters more than it looks: a chunked peel calls the step
        once per chunk, so an un-hoisted transform would be recomputed
        per chunk rather than per step.
        """
        real = np.fft.rfftn
        shapes = []

        def counting(a, s=None, axes=None, norm=None):
            shapes.append(tuple(a.shape))
            return real(a, s=s, axes=axes, norm=norm)

        old_margin, old_bag = ds._FFT_MARGIN, ds._FFT_MIN_BAG
        ds._FFT_MARGIN, ds._FFT_MIN_BAG = 0.0, 0
        np.fft.rfftn = counting
        try:
            ds.direct_sum_zero_momentum(
                PRISM, np.full(len(PRISM), 2.5), np.eye(1), 5, root=0)
        finally:
            np.fft.rfftn = real
            ds._FFT_MARGIN, ds._FFT_MIN_BAG = old_margin, old_bag
        kernel_shape = (4 * 5 + 1,)          # (2m - 1,) at L = 5, d = 1
        n_kernel = sum(1 for sh in shapes if sh == kernel_shape)
        n_peels = len(shapes) - n_kernel
        assert n_peels >= 2, "not enough peels to make the point"
        assert n_kernel == 1, (
            f"{n_kernel} kernel transforms for {n_peels} peels; the "
            f"transform is not cached"
        )


class TestLazinessIsReal:
    """The tables must not be built during the factor build."""

    def test_no_table_exists_before_elimination(self):
        """Every initial kernel factor is a generator, not an array."""
        seen = {}
        real = ds._eliminate_all

        def capture(factors, *a, **kw):
            seen["factors"] = list(factors)
            return real(factors, *a, **kw)

        ds._eliminate_all = capture
        try:
            ds.direct_sum_zero_momentum(
                K4, np.full(6, 2.5), np.eye(1), 6, root=0)
        finally:
            ds._eliminate_all = real
        assert seen["factors"], "no factors captured"
        for f in seen["factors"]:
            assert f.get("tensor") is None, (
                f"factor {f['scope']} was materialised during the build"
            )
            assert f.get("gen") is not None

    def test_duplicate_shaped_factors_stay_distinct(self):
        """Two factors sharing one generator object must not alias.

        The cache hands the same array to every factor of equal nu, so a
        shim that memoised onto the generator — or a peel that removed a
        factor by identity — would collapse two genuinely different
        tables into one.
        """
        seen = {}
        real = ds._eliminate_all

        def capture(factors, *a, **kw):
            seen["factors"] = list(factors)
            return real(factors, *a, **kw)

        ds._eliminate_all = capture
        try:
            ds.direct_sum_zero_momentum(
                DIAMOND, np.full(len(DIAMOND), 2.5), np.eye(1), 6, root=0)
        finally:
            ds._eliminate_all = real
        kernels = [f for f in seen["factors"] if "conv_nu" in f]
        assert len(kernels) >= 2
        # Same generator object, different factor dicts, different scopes.
        assert len({id(f["gen"]) for f in kernels}) == 1
        assert len({id(f) for f in kernels}) == len(kernels)
        assert len({f["scope"] for f in kernels}) == len(kernels)


class TestTheD1ConventionSplitIsGone:
    """Both box engines must build the same generator for the same lattice.

    ``direct_sum_zero_momentum`` used to pass ``a1 = A[0, 0]`` (an
    ``abs(a1*x)`` fast path) while ``_direct_sum_open_terminal`` passed
    ``None`` (the general-d norm), so at d = 1 the two engines sampled
    the same kernel two different ways.  They agreed to round-off over
    the values a lattice sum uses, but nothing enforced that, and the
    shared factor layer cannot carry two conventions.
    """

    def test_conv_kernel_diff_takes_no_convention_argument(self):
        """The four geometric parameters plus the kernel OBJECT: an
        ``interaction`` replaces the power law wholesale under ONE
        sampling rule for both engines (``Interaction.sample`` on the
        same difference grid), which is the opposite of a second d = 1
        convention; it defaults to ``None`` so every legacy positional
        call is the unchanged power-law build."""
        import inspect
        sig = inspect.signature(ds._conv_kernel_diff)
        params = list(sig.parameters)
        assert params == ["nu_e", "A", "L", "d", "interaction"], params
        assert sig.parameters["interaction"].default is None

    @pytest.mark.parametrize("a", [1.0, 1.3, 0.7, 3.0])
    def test_the_two_d1_conventions_agree_on_lattice_values(self, a):
        """Why removing the split was safe, stated as a check.

        ``norm`` of a length-1 vector is ``sqrt(x*x)``, which is *not*
        ``abs(x)`` in general — it breaks where ``x*x`` overflows or
        underflows.  Over the range a box sum actually uses it is
        elementwise identical, which is the claim the removal rests on.
        """
        L, nu = 20, 2.5
        p = np.arange(-2 * L, 2 * L + 1, dtype=float)
        with np.errstate(divide="ignore"):
            fast = np.abs(a * p) ** (-nu)
        fast[p == 0.0] = 0.0
        general = ds._conv_kernel_diff(nu, np.array([[a]]), L, 1)
        assert np.array_equal(fast, general)

    def test_both_engines_build_the_same_d1_generator(self):
        A, L, nu = np.array([[1.3]]), 5, 2.5
        built = []
        real = ds._conv_kernel_diff

        def capture(nu_e, A_, L_, d_):
            g = real(nu_e, A_, L_, d_)
            built.append(g)
            return g

        ds._conv_kernel_diff = capture
        try:
            ds.direct_sum_zero_momentum(
                K4, np.full(6, nu), A, L, root=0)
            n_vacuum = len(built)
            ds._direct_sum_open_terminal(
                ds._collapse_multi_edges(K4, np.full(6, nu)),
                A, L, 0, 3, 1)
        finally:
            ds._conv_kernel_diff = real
        assert n_vacuum >= 1 and len(built) > n_vacuum
        assert np.array_equal(built[0], built[n_vacuum])


class TestComplexBucketDegradesToDense:
    """A complex factor must reroute the step off the FFT, not crash it.

    The engine itself only ever builds real factors (finite k enters as
    a real cos weight), so this is reachable only through direct
    ``_eliminate_all`` callers — but the shared bucket loop needs the
    same uniform rule the torus already has (tn's floating guard), and
    before this guard existed the call died with ``TypeError: ufunc
    'rfft_n_odd' not supported``.
    """

    @staticmethod
    def _run(complex_weight: bool, use_conv: bool):
        L, d = 4, 1
        n_full, n_half = 2 * L + 1, L + 1
        A = np.eye(1)
        gen = ds._conv_kernel_diff(2.5, A, L, d)
        ext = ds._axis_extents(L, d, False)
        off = ds._gen_offsets(L, d, ds._axis_origins(L, d, False),
                              ds._axis_origins(L, d, False))
        rng = np.random.default_rng(7)
        w1 = rng.standard_normal(n_full)
        w3 = np.abs(rng.standard_normal(n_full)) + 0.1
        if complex_weight:
            # BOTH vertex weights go complex, so every peel-eligible
            # bucket in the schedule below contains a complex factor.
            # A single complex weight would leave the other kernel's
            # bucket all-real, and a peel there is legitimate — which
            # is exactly what an earlier version of this test got
            # wrong.
            w1 = w1 + 1j * rng.standard_normal(n_full)
            w3 = w3 + 1j * rng.standard_normal(n_full)
        factors = [
            {"scope": (1, 2), "gen": gen, "ext": (ext, ext), "off": off,
             "d": d, "conv_nu": 2.5},
            {"scope": (2, 3), "gen": gen, "ext": (ext, ext), "off": off,
             "d": d, "conv_nu": 2.5},
            {"scope": (1,), "tensor": w1},
            {"scope": (3,), "tensor": w3},
        ]
        # Order [1, 3, 2]: the step for vertex 1 sees {kernel(1,2),
        # weight(1)} — a peelable bucket carrying the weight — and
        # likewise vertex 3; vertex 2 goes last with dense merged
        # factors only.
        old = (ds._USE_CONV, ds._FFT_MARGIN, ds._FFT_MIN_BAG)
        ds._USE_CONV, ds._FFT_MARGIN, ds._FFT_MIN_BAG = use_conv, 0.0, 0
        try:
            out = ds._eliminate_all(
                [dict(f) for f in factors], [1, 3, 2], None,
                n_full, n_half, np.ones(n_half), A, L, d)
        finally:
            ds._USE_CONV, ds._FFT_MARGIN, ds._FFT_MIN_BAG = old
        result = np.array(1.0 + 0.0j)
        for f in out:
            assert f["scope"] == ()
            result = result * ds._factor_array(f)
        return complex(result)

    def test_real_bucket_still_peels_and_matches_dense(self):
        on = self._run(complex_weight=False, use_conv=True)
        off = self._run(complex_weight=False, use_conv=False)
        assert abs(on - off) <= 1e-12 * abs(off)

    def test_complex_bucket_does_not_crash_and_matches_dense(self):
        on = self._run(complex_weight=True, use_conv=True)
        off = self._run(complex_weight=True, use_conv=False)
        assert on.imag != 0.0, "the complex weight was lost entirely"
        assert abs(on - off) <= 1e-12 * abs(off)

    def test_the_guard_reroutes_rather_than_disables(self):
        """With a complex weight the peel must fire zero times — and the
        same schedule with a real weight must still peel, so the guard
        cannot be a blanket opt-out."""
        counts = []
        real_step = ds._fft_conv_step

        def counting(*a, **kw):
            counts.append(1)
            return real_step(*a, **kw)

        ds._fft_conv_step = counting
        try:
            self._run(complex_weight=True, use_conv=True)
            n_complex = len(counts)
            counts.clear()
            self._run(complex_weight=False, use_conv=True)
            n_real = len(counts)
        finally:
            ds._fft_conv_step = real_step
        assert n_complex == 0, "the FFT ran on a complex bucket"
        assert n_real >= 2, "the guard turned the peel off wholesale"
