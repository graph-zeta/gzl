# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""The torus FFT peel, run under a memory budget.

``tensor_network._peel_convolution`` used to build the whole bag product
in one shot: ``Phi``, its spectrum and the inverse, all at ``N**k``.  The
box already chunked its peel and the torus did not, which is why the box
could be *leaner* than the torus while holding a *larger* array.

This file is the torus twin of ``tests/test_chunked_peel.py`` and keeps
its shape, including the two controls that matter more than the equality
itself: that chunking actually ran (an inert hook passes every equality
test), and that the shipped sizes do NOT chunk (engagement must be
earned, or the suite silently measures a path production never takes).

The value claim is EQUALITY, not closeness: each fibre of the chunk axis
is multiplied and transformed exactly as it would be inside the full-bag
product, so slice-then-convolve is elementwise identical to
convolve-then-slice.  Every assertion here is ``np.array_equal``.
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl import tensor_network as tn
from gzl.hybrid import hybrid_zeta


K4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
K5 = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3),
      (1, 4), (2, 3), (2, 4), (3, 4)]
PRISM = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
         (0, 3), (1, 4), (2, 5)]
# K5 with one edge subdivided: the V6E11 shape that falls to the box at
# d = 3 in a real order-11 cubic pass.
V6E11 = [(0, 1), (0, 4), (1, 2), (1, 3), (1, 5), (2, 3),
         (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]

# (label, edges, d, n, nu).  Kept small on purpose: the point is to
# cover d = 1, 2, 3 and both terminal modes, not to be a benchmark.
CASES = [
    ("k4_d1", K4, 1, 16, 2.5),
    ("k4_d2", K4, 2, 8, 3.5),
    ("k4_d3", K4, 3, 6, 4.5),
    ("k5_d2", K5, 2, 6, 3.5),
    ("k5_d3", K5, 3, 6, 3.5),
    ("prism_d1", PRISM, 1, 12, 2.5),
    ("v6e11_d3", V6E11, 3, 6, 3.5),
]


def _run(edges, d, n, nu, mode):
    """Evaluate one block through hybrid, in one of three momentum modes.

    ``vacuum`` and ``grid`` are separate cases deliberately: at
    ``momentum = 0`` a 1qp call folds into the vacuum branch and returns
    the vacuum value identically, so a "terminal" test written at k = 0
    never exercises the open-terminal path at all.  ``grid`` (no
    momentum, terminal set) and ``offk`` (an off-grid k) are what
    actually reach it.
    """
    E = np.array(edges, dtype=int)
    nu_vec = np.full(len(edges), float(nu))
    A = np.eye(d)
    if mode == "vacuum":
        return np.asarray(hybrid_zeta(E, nu_vec, A, n))
    if mode == "grid":
        return np.asarray(hybrid_zeta(E, nu_vec, A, n, source=0, terminal=1))
    k = np.full(d, 0.3137)
    return np.asarray(
        hybrid_zeta(E, nu_vec, A, n, source=0, terminal=1, momentum=k)
    )


@pytest.fixture(autouse=True)
def _restore_hooks():
    """Both globals are read at call time; put them back regardless."""
    force, budget = tn._FORCE_CHUNK, tn._PEEL_BUDGET_BYTES
    yield
    tn._FORCE_CHUNK, tn._PEEL_BUDGET_BYTES = force, budget


class _PeelSpy:
    """Count chunked vs single-shot peels, and kernel transforms."""

    def __init__(self, monkeypatch):
        self.chunked = 0
        self.single = 0
        self.rfftn = 0
        orig_chunked = tn._chunked_peel_convolution
        orig_peel = tn._peel_convolution
        orig_rfftn = np.fft.rfftn

        def spy_chunked(*a, **kw):
            self.chunked += 1
            return orig_chunked(*a, **kw)

        def spy_peel(*a, **kw):
            self.single += 1
            return orig_peel(*a, **kw)

        def spy_rfftn(*a, **kw):
            self.rfftn += 1
            return orig_rfftn(*a, **kw)

        monkeypatch.setattr(tn, "_chunked_peel_convolution", spy_chunked)
        monkeypatch.setattr(tn, "_peel_convolution", spy_peel)
        monkeypatch.setattr(np.fft, "rfftn", spy_rfftn)


class TestChunkedEqualsSingleShot:
    r"""Bit-identity, driven by the ``_FORCE_CHUNK`` hook.

    Forcing the length rather than shrinking the budget is deliberate:
    it exercises chunk boundaries that do not divide the axis (3 and 7
    against axis lengths 16, 36, 64, 216, ...) without depending on
    whatever the byte model happens to decide.
    """

    @pytest.mark.parametrize("label,edges,d,n,nu", CASES,
                             ids=[c[0] for c in CASES])
    @pytest.mark.parametrize("mode", ["vacuum", "grid", "offk"])
    @pytest.mark.parametrize("chunk", [1, 3, 7])
    def test_forced_chunk_is_bit_identical(self, label, edges, d, n, nu,
                                           mode, chunk):
        tn._FORCE_CHUNK = None
        base = _run(edges, d, n, nu, mode)
        tn._FORCE_CHUNK = chunk
        got = _run(edges, d, n, nu, mode)
        assert np.array_equal(base, got), (
            f"{label} {mode} chunk={chunk}: max |diff| = "
            f"{np.max(np.abs(base - got)):.3e}"
        )

    def test_a_tiny_budget_is_bit_identical_too(self):
        """The budget route, not just the test hook, must preserve values."""
        base = _run(K5, 3, 6, 3.5, "vacuum")
        tn._PEEL_BUDGET_BYTES = 1
        assert np.array_equal(base, _run(K5, 3, 6, 3.5, "vacuum"))


class TestAntiVacuity:
    r"""An inert hook passes every equality test above.

    These are the assertions that fail if the chunked branch is never
    entered, or is entered but degenerates to one chunk.
    """

    def test_chunking_actually_happened(self, monkeypatch):
        spy = _PeelSpy(monkeypatch)
        tn._FORCE_CHUNK = 3
        _run(K5, 3, 6, 3.5, "vacuum")
        assert spy.chunked > 0, "the chunked branch never ran"

    def test_the_budget_route_also_engages(self, monkeypatch):
        spy = _PeelSpy(monkeypatch)
        tn._PEEL_BUDGET_BYTES = 1
        _run(K5, 3, 6, 3.5, "vacuum")
        assert spy.chunked > 0

    def test_kernel_transform_is_built_once_per_step(self, monkeypatch):
        r"""``g_spec`` is hoisted out of the chunk loop.

        It depends only on the kernel and the step geometry, so building
        it per chunk would redo an identical transform for every chunk.
        With ``c`` chunks a peel step must call ``rfftn`` ``c + 1``
        times (one per chunk's phi, plus one for the kernel), never
        ``2c``.
        """
        n, d = 6, 3
        axis_len = n ** d
        chunk = 7
        n_chunks = -(-axis_len // chunk)

        spy = _PeelSpy(monkeypatch)
        tn._FORCE_CHUNK = chunk
        _run(K5, d, n, 3.5, "vacuum")

        # One peel step here; allow the rest of the call its own
        # transforms by asserting the loop did not double up.
        assert spy.rfftn < 2 * n_chunks * max(spy.chunked, 1), (
            f"{spy.rfftn} rfftn calls for {spy.chunked} chunked peel(s) "
            f"of {n_chunks} chunks — the kernel transform looks unhoisted"
        )


class TestBudgetEngagement:
    r"""Engagement must be earned, in both directions."""

    @pytest.mark.parametrize("label,edges,d,n,nu", CASES,
                             ids=[c[0] for c in CASES])
    def test_default_budget_never_chunks_the_suite_sizes(
            self, label, edges, d, n, nu, monkeypatch):
        spy = _PeelSpy(monkeypatch)
        _run(edges, d, n, nu, "vacuum")
        assert spy.chunked == 0, (
            f"{label} chunked at the default budget — the suite would be "
            f"measuring a path production does not take at this size"
        )

    def test_d2_n14_stays_single_shot(self, monkeypatch):
        r"""The regression case: a model that over-predicts costs memory.

        K5 at d = 2, n = 14 has a real full-bag working set of ~0.19 GB,
        comfortably inside the 256 MB budget.  An earlier, cruder model
        over-predicted by ~20 %, engaged here, and made the call *worse*
        — 0.19 GB to 0.30 GB and +11 % wall.  Chunking is not free, and
        this pins the boundary.
        """
        spy = _PeelSpy(monkeypatch)
        _run(K5, 2, 14, 2.5, "grid")
        assert spy.chunked == 0

    def test_a_d3_k5_core_at_n8_does_chunk(self, monkeypatch):
        r"""The block this whole change exists for.

        K5 at d = 3, n = 8 has post-peel exponent 3, so the peel's
        working set is ~3 GiB at the default budget.  Measured: peak RSS
        3.843 GiB single-shot against 1.639 GiB chunked, bit-identical
        and slightly faster.  If this stops chunking, the byte ceiling
        refuses the block again and it falls back to the real-space box.
        """
        spy = _PeelSpy(monkeypatch)
        _run(K5, 3, 8, 3.5, "vacuum")
        assert spy.chunked > 0


class TestTheModel:
    r"""Properties of ``_peel_workset_bytes`` / ``_peel_chunk_length``.

    Deliberately NOT an ``ru_maxrss`` upper-bound gate: this model
    prices one chunk's working set and is blind to the accumulator, the
    factor tables and every other subsystem, so asserting it bounds
    process RSS would be asserting something false.  See its docstring
    for the measured process-level calibration.
    """

    def test_positive_and_monotone(self):
        a = tn._peel_workset_bytes(3, 512, 8)
        assert a > 0
        assert tn._peel_workset_bytes(4, 512, 8) > a      # more axes
        assert tn._peel_workset_bytes(3, 4096, 16) > a    # bigger axis

    def test_it_over_prices_the_true_working_set(self):
        r"""The live set peaks at ``phi + spectrum``, not at all three.

        The model charges two real arrays plus two half-spectra where
        the peel holds one real plus one half-spectrum at its worst
        moment, i.e. one full real array of deliberate margin.
        """
        n, N, k = 8, 512, 3
        elems = N ** (k - 1)
        half = (n // 2 + 1) / n
        true_peak = 8 * elems + 16 * elems * half
        assert tn._peel_workset_bytes(k, N, n) >= true_peak

    def test_a_fitting_bag_is_left_alone(self):
        assert tn._peel_chunk_length(2, 64, 8, 64) is None

    def test_an_oversized_bag_is_chunked_within_the_budget(self):
        n, N, k = 8, 512, 3
        c = tn._peel_chunk_length(k, N, n, N)
        assert c is not None and 1 <= c < N
        assert (tn._peel_workset_bytes(k, N, n) * c
                <= tn._PEEL_BUDGET_BYTES)

    def test_the_force_hook_overrides_the_budget(self):
        tn._FORCE_CHUNK = 5
        assert tn._peel_chunk_length(2, 64, 8, 64) == 5

    def test_the_force_hook_is_clamped_to_the_axis(self):
        tn._FORCE_CHUNK = 10 ** 9
        assert tn._peel_chunk_length(3, 512, 8, 512) == 512

    def test_a_sliver_of_overage_does_not_chunk(self):
        r"""The degenerate split, pinned.

        Measured at d = 3, n = 6: a 256.3 MB working set against a
        256 MB budget yields a chunk length of 215 out of 216 — two
        iterations, an extra transform setup for one fibre, and a 0.5 %
        lower peak.  ``_peel_chunk_length`` must decline that and take
        the overage instead.
        """
        n, N, k = 6, 216, 3
        per_unit = tn._peel_workset_bytes(k, N, n)
        assert per_unit * N > tn._PEEL_BUDGET_BYTES     # over budget...
        assert tn._peel_chunk_length(k, N, n, N) is None  # ...and declined

    def test_the_halving_gate_has_a_live_boundary(self):
        r"""Liveness for the gate above: it must not decline everything.

        Just past 2x benefit the split is taken, so the rule is a
        threshold rather than a disguised "never chunk".
        """
        n, N, k = 8, 512, 3
        per_unit = tn._peel_workset_bytes(k, N, n)
        tn._PEEL_BUDGET_BYTES = per_unit * (N // 3)     # 3x benefit
        c = tn._peel_chunk_length(k, N, n, N)
        assert c is not None and 2 * c <= N
