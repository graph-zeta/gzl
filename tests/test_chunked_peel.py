# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The box peel under a memory budget: chunked, and bit-identical.

The peel's working set — phi, its padded spectrum, the spectral
product, the padded inverse — scales with the product of the SPECTATOR
axis sizes, while the transform itself only ever runs along the
eliminated vertex's ``d`` position axes.  Splitting one spectator axis
into chunks therefore bounds peak memory without touching the
arithmetic: per-fibre FFTs are independent, and slice-then-multiply is
elementwise-identical to multiply-then-slice.  That claim is gated here
as ``np.array_equal``, chunk size by chunk size, not argued.

Two things are deliberately NOT here:

* No terminal streaming on ``direct_sum_zero_momentum`` — it eliminates
  every free vertex, so the schedule would degenerate to running the
  suffix once, saving nothing.
* No wall-clock assertion.  Chunking engages only above the budget,
  where the alternative is swap or OOM; below it the single-shot path
  runs byte for byte as shipped.
"""

from __future__ import annotations

import numpy as np
import pytest

import gzl.direct_sum as ds

K4 = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
K5 = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3],
               [1, 4], [2, 3], [2, 4], [3, 4]])
PRISM = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3],
                  [0, 3], [1, 4], [2, 5]])


def _run(edges, nu, A, L, *, force_chunk=None, use_symmetry=True):
    """One vacuum evaluation with the gate forced open and the chunk
    length pinned (None = single-shot)."""
    old = (ds._FFT_MARGIN, ds._FFT_MIN_BAG, ds._FORCE_CHUNK)
    ds._FFT_MARGIN, ds._FFT_MIN_BAG, ds._FORCE_CHUNK = 0.0, 0, force_chunk
    try:
        return complex(ds.direct_sum_zero_momentum(
            edges, np.full(len(edges), nu), A, L, root=0,
            use_symmetry=use_symmetry))
    finally:
        ds._FFT_MARGIN, ds._FFT_MIN_BAG, ds._FORCE_CHUNK = old


CASES = [
    ("K4 d=1 L=6", K4, 2.5, np.eye(1), 6),
    ("K4 d=1 L=8", K4, 2.5, np.eye(1), 8),
    ("PRISM d=1 L=5", PRISM, 2.5, np.eye(1), 5),
    ("K4 d=2 L=4", K4, 3.0, np.eye(2), 4),
    ("K5 d=2 L=3", K5, 3.0, np.eye(2), 3),
    ("K4 d=3 L=2", K4, 4.5, np.eye(3), 2),
]


class TestChunkedEqualsSingleShot:
    """The whole gate: same bits, every chunk size, every case."""

    @pytest.mark.parametrize("chunk", [1, 3, 7])
    @pytest.mark.parametrize("name,edges,nu,A,L", CASES)
    def test_bit_identical(self, name, edges, nu, A, L, chunk):
        base = _run(edges, nu, A, L, force_chunk=None)
        got = _run(edges, nu, A, L, force_chunk=chunk)
        assert got == base, (
            f"{name} chunk={chunk}: {got} != {base} "
            f"(diff {abs(got - base):.3e})"
        )

    @pytest.mark.parametrize("chunk", [1, 3])
    def test_bit_identical_with_the_marker_active(self, chunk):
        """The Z2 marker halves one axis; the chunk axis is chosen from
        the spectators, which may include the marker's half axis.

        The marker no longer needs forcing: it is the last
        eliminated vertex unconditionally (the ``_plan_marker``
        pre-pass whose margin-0 interaction used to disable it under
        this file's forced-open gate was deleted).  A spy still
        asserts the chunked peel really ran with the marker set —
        without it this test could silently compare two marker-off
        runs again if the marker rule ever changes.
        """
        seen = {"marker": [], "chunk_axes": []}
        real_chunked = ds.BoxTruncation._chunked_peel

        def spy(self, others, phi_scope, axes, w, chunk_axis,
                chunk_len, kdiff, kdiff_hat, m, d,
                w_ext, u_ext, gen_start):
            seen["marker"].append(self.marker)
            seen["chunk_axes"].append(
                (chunk_axis, self.axis_size(chunk_axis)))
            return real_chunked(self, others, phi_scope, axes, w,
                                chunk_axis, chunk_len, kdiff,
                                kdiff_hat, m, d,
                                w_ext, u_ext, gen_start)

        base = _run(K4, 2.5, np.eye(2), 3, force_chunk=None,
                    use_symmetry=True)
        ds.BoxTruncation._chunked_peel = spy
        try:
            got = _run(K4, 2.5, np.eye(2), 3, force_chunk=chunk,
                       use_symmetry=True)
        finally:
            ds.BoxTruncation._chunked_peel = real_chunked
        assert got == base
        assert seen["marker"] and all(mk is not None
                                      for mk in seen["marker"]), (
            "the marker never engaged; this test compared two "
            "marker-off runs"
        )

    @pytest.mark.parametrize("chunk", [1, 4])
    def test_bit_identical_on_the_open_terminal(self, chunk):
        em = ds._collapse_multi_edges(K4, np.full(6, 2.5))
        old = (ds._FFT_MARGIN, ds._FFT_MIN_BAG, ds._FORCE_CHUNK)

        def run(fc):
            ds._FFT_MARGIN, ds._FFT_MIN_BAG, ds._FORCE_CHUNK = 0.0, 0, fc
            M, _ = ds._direct_sum_open_terminal(em, np.eye(1), 5, 0, 3, 1)
            return M

        entered = {"n": 0}
        real_chunked = ds.BoxTruncation._chunked_peel

        def spy(self, *a, **kw):
            entered["n"] += 1
            return real_chunked(self, *a, **kw)

        try:
            base = run(None)
            ds.BoxTruncation._chunked_peel = spy
            try:
                got = run(chunk)
            finally:
                ds.BoxTruncation._chunked_peel = real_chunked
        finally:
            ds._FFT_MARGIN, ds._FFT_MIN_BAG, ds._FORCE_CHUNK = old
        assert entered["n"] >= 1, "the forced chunk never engaged"
        assert np.array_equal(got, base)

    def test_chunking_actually_happened(self):
        """The forced-chunk runs must call the FFT step once per chunk —
        otherwise every equality above compares the single-shot path
        with itself."""
        calls = {"n": 0}
        real = ds._fft_conv_step

        def counting(*a, **kw):
            calls["n"] += 1
            return real(*a, **kw)

        ds._fft_conv_step = counting
        try:
            _run(K4, 2.5, np.eye(1), 6, force_chunk=None)
            n_single = calls["n"]
            calls["n"] = 0
            _run(K4, 2.5, np.eye(1), 6, force_chunk=1)
            n_chunked = calls["n"]
        finally:
            ds._fft_conv_step = real
        assert n_chunked > n_single, (
            f"forcing chunk=1 did not multiply FFT-step calls "
            f"({n_single} -> {n_chunked}); the chunk hook is dead"
        )

    def test_kernel_transform_still_built_once_per_step(self):
        """The hat is hoisted per STEP, never per chunk — the existing
        factor-supply test pins the single-shot path; this pins the
        chunked one, where the regression would actually bite."""
        real = np.fft.rfftn
        shapes = []

        def counting(a, s=None, axes=None, norm=None):
            shapes.append(tuple(a.shape))
            return real(a, s=s, axes=axes, norm=norm)

        L = 6
        kernel_shape = (4 * L + 1,)
        np.fft.rfftn = counting
        try:
            _run(K4, 2.5, np.eye(1), L, force_chunk=1)
        finally:
            np.fft.rfftn = real
        n_kernel = sum(1 for sh in shapes if sh == kernel_shape)
        n_other = len(shapes) - n_kernel
        assert n_other > n_kernel, "not enough chunked peels to test"
        assert n_kernel == 1, (
            f"{n_kernel} kernel transforms for {n_other} chunk "
            f"transforms; the hat is being rebuilt per chunk"
        )


class TestBudgetEngagement:
    """The automatic path: chunking must engage above the budget and
    stay out of the way below it."""

    def test_default_budget_never_chunks_the_suite_sizes(self):
        """Chunking must stay out of the way below the budget — proven
        by the chunked path never being entered, not by a value match
        (which holds either way, since chunking is exact)."""
        entered = {"n": 0}
        real_chunked = ds.BoxTruncation._chunked_peel

        def spy(self, *a, **kw):
            entered["n"] += 1
            return real_chunked(self, *a, **kw)

        calls = {"n": 0}
        real_step = ds._fft_conv_step

        def counting(*a, **kw):
            calls["n"] += 1
            return real_step(*a, **kw)

        ds.BoxTruncation._chunked_peel = spy
        ds._fft_conv_step = counting
        try:
            _run(K4, 3.0, np.eye(2), 4, force_chunk=None)
        finally:
            ds._fft_conv_step = real_step
            ds.BoxTruncation._chunked_peel = real_chunked
        assert calls["n"] >= 1, "no peel ran at all"
        assert entered["n"] == 0, (
            f"the default budget chunked a suite-sized problem "
            f"({entered['n']} chunked peels); either the budget or the "
            f"model moved"
        )

    def test_tiny_budget_forces_chunks_and_the_value_survives(self):
        base = _run(K5, 3.0, np.eye(2), 3, force_chunk=None)
        calls = {"n": 0}
        real = ds._fft_conv_step

        def counting(*a, **kw):
            calls["n"] += 1
            return real(*a, **kw)

        old_budget = ds._MEM_BUDGET_BYTES
        ds._MEM_BUDGET_BYTES = 1        # everything over budget
        ds._fft_conv_step = counting
        try:
            got = _run(K5, 3.0, np.eye(2), 3, force_chunk=None)
            n_chunked = calls["n"]
        finally:
            ds._MEM_BUDGET_BYTES = old_budget
            ds._fft_conv_step = real
        assert got == base
        # chunk_len floors at 1, so the FFT step runs once per unit of
        # the chunk axis on every peelable step.
        assert n_chunked > 3

    def test_model_is_positive_and_monotone(self):
        """Sanity on the decision model itself."""
        m, d = 13, 2
        a = ds._peel_workset_bytes([13 ** 2, 13 ** 2, 13 ** 2], 0, 2, m, d)
        b = ds._peel_workset_bytes([13 ** 2, 13 ** 2, 13 ** 2, 13 ** 2],
                                   0, 3, m, d)
        assert 0 < a < b


class TestModelUpperBoundsMeasured:
    """The memory model must bound reality, not merely correlate.

    A model that under-predicts is worse than none: the chunk decision
    trusts it, and on the CI runner an OOM surfaces as "The operation
    was canceled", not as a test failure.  The whole-call bound is
    ``2 x max(peel_model, dense_model)``: chunking bounds the peel, at
    which point the DENSE steps become the ceiling (a peel-only model
    under-predicts exactly where chunking succeeds), and the previous
    step's result stays referenced while the next step's working set
    builds, which stacks up to one extra working set on the per-step
    max.

    The measurement runs in a CLEAN SUBPROCESS, for two reasons the
    in-process version got wrong.  First, ``ru_maxrss`` is a
    process-lifetime HIGH WATER mark: an ``after - before`` delta in a
    shared pytest process reads 0 whenever any earlier test allocated
    more, making the bound pass vacuously.  Second, its UNITS are
    platform-defined — bytes on macOS, kilobytes on Linux — so an
    unscaled in-process delta made the CI measurement ~1000x too small
    and the gate trivially green exactly where it matters.  On Linux a
    subprocess is not clean either: exec folds the parent's high-water
    mark into the child's ``ru_maxrss``, so the probe reads ``VmHWM``
    there instead.
    """

    _PROBE = """
import resource, sys
import numpy as np
sys.path.insert(0, {root!r})
import gzl.direct_sum as ds
K5 = np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]])
ds._FFT_MARGIN, ds._FFT_MIN_BAG = 0.0, 0
def peak():
    # On Linux, exec folds the parent's high-water mark into the child's
    # ru_maxrss, so a probe started from a large pytest process (an xdist
    # worker late in its files) reads the parent's peak.  VmHWM is the
    # peak of this process image alone, in kB.
    if sys.platform.startswith("linux"):
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024   # ru_maxrss units

baseline = peak()
ds.direct_sum_zero_momentum(K5, np.full(10, {nu}), np.eye({d}), {L}, root=0)
print(peak() - baseline)
"""

    @classmethod
    def _model_and_measure(cls, nu, A, L, d):
        import subprocess
        import sys
        from pathlib import Path

        m = 2 * L + 1
        n_full = m ** d
        # K5, root pinned: 4 free vertices; the peel drops one axis, so
        # phi carries 3; dense steps see 3-axis bags after the first
        # peel merges its factor.
        per_unit = ds._peel_workset_bytes([n_full] * 3, 0, 2, m, d)
        chunk_len = max(1, min(int(ds._MEM_BUDGET_BYTES // per_unit),
                               n_full))
        psi = 8 * n_full ** 3
        tables = 4 * 8 * n_full ** 2 + 2 * 8 * (2 * m - 1) ** d
        peel_model = per_unit * chunk_len + psi + tables
        # One dense step: two full-bag arrays live at the peak plus the
        # summed result, bounded by a third.
        dense_model = 3 * 8 * n_full ** 3 + tables + psi
        model = 2 * max(peel_model, dense_model)

        root = str(Path(ds.__file__).resolve().parents[1])
        probe = cls._PROBE.format(root=root, nu=nu, d=d, L=L)
        out = subprocess.run([sys.executable, "-c", probe],
                             capture_output=True, text=True, timeout=600)
        assert out.returncode == 0, out.stderr[-2000:]
        measured = int(out.stdout.strip())
        # Liveness: a subprocess that did no real work (or a probe that
        # silently failed to import the worktree's module) would report
        # a near-zero increment and pass any bound.
        assert measured > psi // 4, (
            f"measured increment {measured} B is implausibly small "
            f"against a {psi} B psi accumulator; the probe is not "
            f"measuring the run it claims to"
        )
        return model, measured

    def test_k5_d2_L6(self):
        model, measured = self._model_and_measure(3.0, np.eye(2), 6, 2)
        assert model >= measured, (
            f"model {model / 2**30:.2f} GB < measured "
            f"{measured / 2**30:.2f} GB — the chunk decision is being "
            f"made on an under-prediction"
        )

    @pytest.mark.slow
    def test_k5_d3_L3(self):
        model, measured = self._model_and_measure(4.5, np.eye(3), 3, 3)
        assert model >= measured, (
            f"model {model / 2**30:.2f} GB < measured "
            f"{measured / 2**30:.2f} GB"
        )
