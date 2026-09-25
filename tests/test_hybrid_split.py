# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Split-resolution SP collapse in :func:`gzl.hybrid.hybrid_zeta`.

``sp_n_points`` collapses the series/parallel part of a block on a fine
torus and contracts the irreducible core on the coarse one.  The point is
the truncation RATE, not the cost: error falls as ``n^{-sigma_eff}`` with
``sigma_eff = (minimum cluster cut) - d``, the cheapest escape is a
degree-2 vertex (cut ``2 nu``), and SP reduction is exactly what removes
degree-2 vertices.  A 3-connected core has edge connectivity >= 3, so its
own cut is ``>= 3 nu`` — the split buys a full ``nu`` in the exponent.

Three things are gated here, in increasing order of what they can catch:

1. **The flag is structurally off by default**, and ``sp_n_points ==
   n_points`` is bit-for-bit the uniform value.  This is the only
   bitwise gate the hybrid engine has: its frozen goldens run at
   ``rtol = 4e-15`` (``executor_goldens.json``) and <= 4 per-entry ulp
   (``hybrid_s7_refs.json``), so a live-vs-live ``array_equal`` is
   strictly stronger than anything the fixtures can say, and it is what
   keeps a regression from hiding inside that headroom.
2. **Both families reach the same limit** — the split must be a better
   approximation to the same object, not a different object.
3. **The rate really improves**, by ``nu`` in the exponent.

ESTIMATOR DISCIPLINE.  Two cheap slope estimators are biased in opposite
directions, so neither is quoted alone: successive differences read too
STEEP on a non-geometric ladder, and a three-parameter fit reads too
SHALLOW against subleading terms.  Where a converged reference is
affordable (d = 1) these tests use TRUE errors against it.  Where it is
not (d = 2 needs ``n^{tau d} = n^4``, i.e. 34 GB at n = 256) they use
successive differences on a GEOMETRIC ladder — unbiased there — and read
the GAIN between the two families rather than either absolute slope.

Physical ``nu = d + sigma`` with ``sigma > 0``, so every case here has
``nu > d``.  ``nu < d`` is a different regime (the series rule picks up a
star-triangle term) and is out of scope.
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl.hybrid import HybridCoreTooLargeError, hybrid_zeta
from gzl.tensor_network import (
    TorusTruncation,
    _edge_kernel_torus,
    graph_zeta_general,
)


NU = 2.5

K4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
#: K4 with edge (2, 3) subdivided by vertex 4 — a tw = 3 core wearing one
#: degree-2 chain, the smallest graph on which the split has anything to do.
K4SUB = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (4, 3)]

A1 = np.eye(1)
A2 = np.eye(2)
#: Sheared cell: ``_balanced_z_axis`` holds n/2 without -n/2, so at even n
#: the kernel is genuinely not even here and orientation is load-bearing.
AHEX = np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2.0]])


def _uniform(edges, A, n, **kw):
    return hybrid_zeta(edges, [NU] * len(edges), A, n, **kw)


def _split(edges, A, n, n_f, **kw):
    return hybrid_zeta(edges, [NU] * len(edges), A, n,
                       sp_n_points=n_f, **kw)


def _slope(ns, es):
    return float(np.polyfit(np.log(np.asarray(ns, dtype=float)),
                            np.log(np.asarray(es, dtype=float)), 1)[0])


# ---------------------------------------------------------------------------
# 1. The flag is off by default, and switching it on at equal n is exact
# ---------------------------------------------------------------------------

class TestDefaultOffAndIdentity:
    r"""``sp_n_points`` must not perturb the default path at all."""

    @pytest.mark.parametrize("d, A", [(1, A1), (2, A2), (2, AHEX)])
    @pytest.mark.parametrize("n", [8, 12, 16])
    @pytest.mark.parametrize("mode", ["vacuum", "grid", "single"])
    def test_equal_grids_are_bit_identical(self, d, A, n, mode):
        r"""``sp_n_points == n_points`` reproduces the uniform value BITWISE.

        This is not vacuous: the restriction gather still runs (the code
        branches on ``sp_n_points is not None``, never on ``n_sp == n``),
        so what is asserted is that the coarse->fine index map
        ``_balanced_z_axis(n) % n`` is exactly ``arange(n)`` and the
        gather is the identity.  Were it merely *numerically* neutral —
        say a resample that happened to agree to round-off — this would
        fail while every frozen fixture, at rtol 4e-15, still passed.
        """
        kw = {"vacuum": {},
              "grid": dict(source=0, terminal=1),
              "single": dict(source=0, terminal=1,
                             momentum=np.full(d, 0.137))}[mode]
        got = np.asarray(_split(K4SUB, A, n, n, **kw))
        want = np.asarray(_uniform(K4SUB, A, n, **kw))
        assert np.array_equal(got, want), (
            f"split at sp_n_points == n_points moved off the uniform bits "
            f"(d={d}, n={n}, {mode}): max|diff| = "
            f"{np.max(np.abs(got - want)):.3e}"
        )

    @pytest.mark.parametrize("A", [A1, A2])
    @pytest.mark.parametrize("n_c, n_f", [(8, 32), (12, 48), (16, 64)])
    def test_no_sp_structure_makes_the_split_a_noop(self, A, n_c, n_f):
        r"""On a 3-connected block the split has nothing to collapse.

        Plain K4 has no degree-2 vertex, so every residual kernel is an
        original ``|A z|^{-nu}``, and restricting a pure power law is
        exact — the coarse array is the fine one gathered, entry for
        entry.  Bitwise equality here isolates the restriction operator
        from the collapse: it can only pass if the gather reproduces a
        directly-built coarse kernel exactly.

        This is also the measured no-op class — 13.5% of shipped tw>=3
        block occurrences are already 3-connected.
        """
        assert _split(K4, A, n_c, n_f) == _uniform(K4, A, n_c)


# ---------------------------------------------------------------------------
# 2. Both families converge to the same value
# ---------------------------------------------------------------------------

class TestSameLimit:
    r"""The split must be a better estimate of the SAME object."""

    def test_split_and_uniform_agree_better_and_better(self):
        r"""``|uniform(n) - split(n)|`` must fall to zero with n.

        Note what this gap actually measures.  The split's own error at
        these n is ~1e-11, three orders below the uniform family's, so
        the difference is dominated by the UNIFORM error and tracks it —
        that is precisely the statement "both approach the same limit,
        and the split gets there first".  A gap that stalled would mean
        the two families were converging to different objects, which is
        the failure this exists to catch.

        Gating the ratio across the ladder rather than an absolute
        floor: an absolute threshold would silently encode this machine's
        round-off, whereas the ratio is the rate claim itself.
        """
        ns = (16, 32, 64, 128)
        gaps = [abs(_uniform(K4SUB, A1, n) - _split(K4SUB, A1, n, 1024))
                for n in ns]
        assert all(b < a for a, b in zip(gaps, gaps[1:])), (
            f"|uniform - split| must fall monotonically; got {gaps}"
        )
        # uniform converges at 2 nu - d = 4, so 8x in n is >= 1e3 in error
        assert gaps[0] / gaps[-1] > 1.0e3, (
            f"gap fell only {gaps[0] / gaps[-1]:.3g}x over n = 16 -> 128; "
            f"the two families may not share a limit.  Got {gaps}"
        )


# ---------------------------------------------------------------------------
# 3. The rate improves by nu in the exponent
# ---------------------------------------------------------------------------

class TestRateGain:

    def test_d1_true_errors_against_a_converged_reference(self):
        r"""d = 1, sigma = 1.5.  TRUE errors against n = 4096.

        The reference's own error is ~1e-13 (it agrees with n = 2048 to
        2.8e-13 relative), four orders below the smallest row read here,
        so no row sits under its reference.  The uniform slope landing on
        its prediction is what validates the estimator; only then does
        the split slope mean anything.
        """
        nu_vec = [NU] * len(K4SUB)
        ref = float(np.real(graph_zeta_general(K4SUB, nu_vec, A1, 4096)))

        ns = (24, 32, 48, 64, 96)
        eu = [abs(_uniform(K4SUB, A1, n) - ref) / abs(ref) for n in ns]
        es = [abs(_split(K4SUB, A1, n, 1024) - ref) / abs(ref) for n in ns]

        s_u, s_s = _slope(ns, eu), _slope(ns, es)
        # predicted: uniform 2 nu - d = 4.0, split 3 nu - d = 6.5
        assert -4.4 < s_u < -3.6, f"uniform slope {s_u} off prediction -4.0"
        assert s_s < -5.5, f"split slope {s_s} does not reach the core rate"
        assert s_s - s_u < -1.5, (
            f"rate gain {s_s - s_u:+.3f} short of the predicted -nu = -2.5"
        )
        # and it must pay in absolute terms, not only in slope
        assert eu[1] / es[1] > 10.0
        assert eu[-1] / es[-1] > 100.0

    @pytest.mark.parametrize("A, lad", [(A2, (8, 16, 32, 64)),
                                        (AHEX, (8, 16, 32, 64)),
                                        (AHEX, (9, 17, 33, 65))])
    def test_reference_free_gain_on_a_geometric_ladder(self, A, lad):
        r"""d = 2, sigma = 0.5.  No converged reference is affordable.

        A tw = 3 block costs ``n^{tau d} = n^4`` at d = 2 — 34 GB at
        n = 256 — so this reads successive differences on a GEOMETRIC
        ladder, where that estimator is unbiased, and gates the GAIN
        between the two families rather than either absolute slope.

        The two AHEX ladders are the even/odd pair on a sheared cell,
        where the kernel is genuinely not even (the balanced axis holds
        n/2 without -n/2) and restriction does not commute with
        reversal.  They check that the RATE survives that asymmetry in
        both parities.

        They are NOT the orientation gate, despite an earlier version of
        this docstring saying so: removing orientation tracking leaves
        both ladders bit-identical, because nothing here builds a
        parallel bundle on a descending endpoint pair.  The actual
        orientation gate is
        ``tests/test_sp_shared.py::TestParallelMergeOrientation``,
        which compares the merged bundle kernel entry-by-entry.
        """
        nu_vec = [NU] * len(K4SUB)
        U = [hybrid_zeta(K4SUB, nu_vec, A, n) for n in lad]
        S = [hybrid_zeta(K4SUB, nu_vec, A, n, sp_n_points=128) for n in lad]
        du = [abs(U[i] - U[i + 1]) for i in range(len(lad) - 1)]
        ds = [abs(S[i] - S[i + 1]) for i in range(len(lad) - 1)]

        gain = _slope(lad[:-1], ds) - _slope(lad[:-1], du)
        assert gain < -1.5, (
            f"rate gain {gain:+.3f} on ladder {lad} short of -nu = -2.5"
        )
        # same limit, on this lattice too
        assert abs(U[-1] - S[-1]) < 1e-3 * abs(U[-1])


class TestFiniteK:
    r"""The gain must survive at finite k, including the zone boundary."""

    def test_gain_is_uniform_across_the_brillouin_zone(self):
        r"""Rungs are all multiples of 8 so ``k = j/8`` is ON-GRID at
        every rung.  A non-commensurate ladder compares the value at
        different k from rung to rung and the comparison means nothing.
        """
        lad = (16, 24, 32, 48, 64)
        assert all(n % 8 == 0 for n in lad)
        nu_vec = [NU] * len(K4SUB)
        U = {n: hybrid_zeta(K4SUB, nu_vec, A1, n, source=0, terminal=1)
             for n in lad}
        S = {n: hybrid_zeta(K4SUB, nu_vec, A1, n, source=0, terminal=1,
                            sp_n_points=1024) for n in lad}

        for j in range(5):                       # k = 0 .. 1/2 (zone bdry)
            du, ds = [], []
            for a, b in zip(lad, lad[1:]):
                du.append(abs(U[a][j * a // 8] - U[b][j * b // 8]))
                ds.append(abs(S[a][j * a // 8] - S[b][j * b // 8]))
            gain = _slope(lad[:-1], ds) - _slope(lad[:-1], du)
            assert gain < -0.8, (
                f"k = {j}/8: gain {gain:+.3f} — the split must converge "
                f"faster at every k, the zone boundary included"
            )


# ---------------------------------------------------------------------------
# 4. Refusals, and the memory cap
# ---------------------------------------------------------------------------

class TestRefusals:

    def test_coarser_sp_grid_is_refused(self):
        with pytest.raises(ValueError, match="below n_points"):
            _split(K4SUB, A1, 32, 16)

    def test_infinite_nu_is_refused(self):
        r"""The nu = inf kernel is the NN indicator and the contraction is
        exact integer arithmetic; an FFT collapse returns 2.0 as
        1.9999999999999998.  There is also nothing to buy — an indicator
        has no power-law tail to truncate.
        """
        # Now refused by hybrid_zeta outright, before sp_n_points is
        # even looked at -- the refusal below subsumes this one.
        nu_vec = [NU] * len(K4SUB)
        nu_vec[0] = np.inf
        with pytest.raises(ValueError, match="nu = inf"):
            hybrid_zeta(K4SUB, nu_vec, A1, 16, sp_n_points=64)

    def test_memory_cap_prices_the_core_grid_not_the_sp_grid(self):
        r"""``max_core_bytes`` bounds the CORE contraction, which runs at
        ``n_points``.  A cap that priced ``sp_n_points`` instead would
        refuse work the engine does comfortably — the fine grid only ever
        carries ``n_f^d`` FFTs, never the core's ``n^{tau d}``.
        """
        nu_vec = [NU] * len(K4SUB)
        # The K4SUB core is K4: tw 3, post-peel exponent 2, so the cap
        # sees n**2 * 8 * _CORE_RETENTION.  This cap sits between
        # n = 16 (1.0e4 B) and n = 64 (1.6e5 B).
        cap = 5.0e4

        # Coarse grid under the cap: runs, no matter how fine the SP grid.
        for n_f in (16, 512, 4096):
            hybrid_zeta(K4SUB, nu_vec, A1, 16, sp_n_points=n_f,
                        max_core_bytes=cap)

        # Coarse grid over the cap: refused — and refused identically
        # whether or not the split is on, because the cap never sees n_f.
        for kw in ({}, {"sp_n_points": 4096}):
            with pytest.raises(HybridCoreTooLargeError):
                hybrid_zeta(K4SUB, nu_vec, A1, 64,
                            max_core_bytes=cap, **kw)


# ---------------------------------------------------------------------------
# 5. The restriction operator itself
# ---------------------------------------------------------------------------

class TestRestrictTo:

    @pytest.mark.parametrize("d, A", [(1, A1), (2, A2), (2, AHEX)])
    @pytest.mark.parametrize("n_f, n_c", [(32, 32), (32, 16), (33, 11),
                                          (48, 16), (24, 9), (16, 15)])
    def test_restriction_reproduces_a_directly_built_coarse_kernel(
            self, d, A, n_f, n_c):
        r"""The gather must be exact, not approximate — including across
        a parity change and on a sheared cell.
        """
        fine, coarse = TorusTruncation(n_f, d, A), TorusTruncation(n_c, d, A)
        got = fine.restrict_to(_edge_kernel_torus(NU, A, n_f), coarse)
        assert np.array_equal(got, _edge_kernel_torus(NU, A, n_c))

    def test_upward_restriction_is_refused(self):
        with pytest.raises(ValueError, match="restrict upward"):
            TorusTruncation(8, 1, A1).restrict_to(
                _edge_kernel_torus(NU, A1, 8), TorusTruncation(16, 1, A1))

    def test_dimension_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            TorusTruncation(16, 2, A2).restrict_to(
                _edge_kernel_torus(NU, A2, 16), TorusTruncation(8, 1, A1))

    def test_lattice_mismatch_is_refused(self):
        r"""The labels are shared integers and the positions are ``A z``,
        so restricting across different ``A`` would silently reinterpret
        every entry rather than fail.
        """
        with pytest.raises(ValueError, match="share the lattice"):
            TorusTruncation(16, 2, A2).restrict_to(
                _edge_kernel_torus(NU, A2, 16),
                TorusTruncation(8, 2, AHEX))


# ---------------------------------------------------------------------------
# 5. The downward half of the same axis: a COARSE core
# ---------------------------------------------------------------------------

class TestEmbedInto:
    r"""``embed_into`` is ``restrict_to`` run the other way.

    Both are pure index moves built from the same vector
    ``_balanced_z_axis(n_coarse) % n_fine`` — one gathers with it, one
    scatters with it — so the pair has an exact algebraic relation that
    is worth gating directly: restricting a fine array and embedding the
    result back must reproduce the original *on the coarse labels* and
    zero elsewhere.
    """

    @pytest.mark.parametrize("d, A", [(1, A1), (2, A2), (2, AHEX)])
    @pytest.mark.parametrize("n", [8, 11, 16])
    def test_equal_size_embedding_is_the_identity_bitwise(self, d, A, n):
        r"""At equal size the index vector is ``arange(n)``.  This is the
        upward twin of the property the split-resolution gate rests on,
        and it is what makes ``core_n_points == n_points`` exact.
        """
        T = TorusTruncation(n, d, A)
        gen = _edge_kernel_torus(NU, A, n)
        assert np.array_equal(T.embed_into(gen, T), gen)

    @pytest.mark.parametrize("d, A", [(1, A1), (2, A2), (2, AHEX)])
    @pytest.mark.parametrize("n_f, n_c", [(32, 16), (33, 11), (48, 16),
                                          (24, 9), (16, 15)])
    def test_embed_after_restrict_keeps_the_coarse_labels_and_zeroes_the_rest(
            self, d, A, n_f, n_c):
        fine, coarse = TorusTruncation(n_f, d, A), TorusTruncation(n_c, d, A)
        gen = _edge_kernel_torus(NU, A, n_f)
        back = coarse.embed_into(fine.restrict_to(gen, coarse), fine)
        # the coarse labels survive exactly...
        assert np.array_equal(fine.restrict_to(back, coarse),
                              fine.restrict_to(gen, coarse))
        # ...and nothing else was invented
        assert np.count_nonzero(back) <= n_c ** d

    def test_downward_embedding_is_refused(self):
        r"""Refused rather than silently doing ``restrict_to``'s job: a
        caller who passed the two truncations the wrong way round would
        otherwise get plausible finite numbers back.
        """
        with pytest.raises(ValueError, match="embed downward"):
            TorusTruncation(16, 1, A1).embed_into(
                _edge_kernel_torus(NU, A1, 16), TorusTruncation(8, 1, A1))

    def test_wrong_shape_is_refused(self):
        with pytest.raises(ValueError, match="expected"):
            TorusTruncation(8, 1, A1).embed_into(
                _edge_kernel_torus(NU, A1, 16), TorusTruncation(32, 1, A1))

    def test_lattice_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="share the lattice"):
            TorusTruncation(8, 2, A2).embed_into(
                _edge_kernel_torus(NU, A2, 8), TorusTruncation(16, 2, AHEX))


class TestCoreGridDefaultOffAndIdentity:
    r"""``core_n_points`` must not perturb the default path at all.

    Same argument as :class:`TestDefaultOffAndIdentity`: a live-vs-live
    ``array_equal`` is strictly stronger than the frozen fixtures, whose
    tolerances (``rtol = 4e-15``, <= 4 ulp) leave headroom a regression
    could hide in.
    """

    @pytest.mark.parametrize("d, A", [(1, A1), (2, A2), (2, AHEX)])
    @pytest.mark.parametrize("n", [8, 12, 16])
    @pytest.mark.parametrize("sp", [None, 64])
    def test_none_is_bit_identical(self, d, A, n, sp):
        kw = {} if sp is None else {"sp_n_points": max(sp, n)}
        for src, term in ((0, None), (0, 1)):
            a = hybrid_zeta(K4SUB, [NU] * len(K4SUB), A, n,
                            source=src, terminal=term, **kw)
            b = hybrid_zeta(K4SUB, [NU] * len(K4SUB), A, n,
                            source=src, terminal=term,
                            core_n_points=None, **kw)
            assert np.array_equal(np.asarray(a), np.asarray(b))

    @pytest.mark.parametrize("d, A", [(1, A1), (2, A2), (2, AHEX)])
    @pytest.mark.parametrize("n", [8, 11, 12, 16])
    @pytest.mark.parametrize("sp", [None, 64])
    def test_equal_to_n_points_is_bit_identical(self, d, A, n, sp):
        r"""``core_n_points == n_points`` still takes the coarse-core
        branch — one extra gather and one embed that are both exact
        identities — so it is a usable self-check rather than a
        different code path wearing the same number.
        """
        kw = {} if sp is None else {"sp_n_points": max(sp, n)}
        for src, term in ((0, None), (0, 1)):
            a = hybrid_zeta(K4SUB, [NU] * len(K4SUB), A, n,
                            source=src, terminal=term, **kw)
            b = hybrid_zeta(K4SUB, [NU] * len(K4SUB), A, n,
                            source=src, terminal=term,
                            core_n_points=n, **kw)
            assert np.array_equal(np.asarray(a), np.asarray(b))


class TestCoreGridReadout:
    r"""The three readout modes must agree on what a coarse core means.

    A core contracted on ``n_core`` labels yields a terminal-position
    amplitude whose BZ transform is a trigonometric polynomial with
    exactly those coefficients.  The single-k branch reads that
    polynomial off the coarse array directly; the grid branch embeds and
    transforms.  They are two evaluations of the SAME polynomial, so
    they must agree to round-off — and that is what licenses calling the
    embed exact for the approximant rather than an interpolation.
    """

    @pytest.mark.parametrize("d, A, n, n_core", [
        (1, A1, 32, 8), (1, A1, 40, 10), (2, A2, 24, 6), (2, AHEX, 24, 8),
    ])
    def test_grid_branch_matches_single_k_from_the_same_coarse_core(
            self, d, A, n, n_core):
        grid = np.asarray(hybrid_zeta(
            K4SUB, [NU] * len(K4SUB), A, n, source=0, terminal=1,
            core_n_points=n_core)).reshape((n,) * d)
        axes = [np.arange(n, dtype=float) / n] * d
        kk = np.stack(np.meshgrid(*axes, indexing="ij"),
                      axis=-1).reshape(-1, d)
        direct = np.array([
            hybrid_zeta(K4SUB, [NU] * len(K4SUB), A, n, source=0, terminal=1,
                        momentum=k, core_n_points=n_core) for k in kk
        ]).reshape((n,) * d)
        assert np.allclose(grid, direct, rtol=0.0,
                           atol=1e-13 * np.max(np.abs(direct)))

    @pytest.mark.parametrize("d, A", [(1, A1), (2, A2)])
    def test_vacuum_ignores_n_points_once_both_other_grids_are_pinned(
            self, d, A):
        r"""With ``sp_n_points`` and ``core_n_points`` both given,
        ``n_points`` enters a vacuum dense block NOWHERE: the SP part is
        collapsed on the fine grid, the core is contracted on the coarse
        one, and the result is a scalar that needs no embedding.  So
        raising ``n_points`` must not move it by even one ulp.

        This is the property the whole scheme rests on — it is what
        decouples dense-core cost from the grid the rest of the pass
        runs on.  Note it needs BOTH knobs: with ``sp_n_points`` left
        off the SP part is collapsed at ``n_points`` itself, so the
        value does (correctly) keep improving with it.
        """
        vals = [hybrid_zeta(K4SUB, [NU] * len(K4SUB), A, n,
                            sp_n_points=256, core_n_points=8)
                for n in (8, 12, 16, 24)]
        assert all(v == vals[0] for v in vals)

    @pytest.mark.parametrize("d, A", [(1, A1), (2, A2)])
    def test_n_points_still_drives_the_sp_grid_when_sp_is_left_off(
            self, d, A):
        r"""The complement of the test above, and the reason it needs its
        hypothesis: with only ``core_n_points`` set, ``n_points`` IS the
        SP grid, so the value must still respond to it.  Without this the
        pinning test above would pass just as well on a value that had
        stopped depending on anything at all.
        """
        vals = [hybrid_zeta(K4SUB, [NU] * len(K4SUB), A, n, core_n_points=8)
                for n in (16, 32, 64)]
        assert len(set(vals)) == len(vals)


class TestCoreGridRateAndRefusals:

    @pytest.mark.parametrize("d, A, nu, n, n_core, ref_n, ref_sp", [
        # (d, lattice, nu, delivery grid, coarse core, reference)
        (1, A1, 2.5, 64, 32, 128, 8192),
        (2, A2, 2.5, 24, 12,  40, 2048),
    ])
    def test_a_coarse_split_core_beats_a_uniform_core_at_the_full_grid(
            self, d, A, nu, n, n_core, ref_n, ref_sp):
        r"""The claim the parameter exists for, and the honest form of it.

        The alternative to a coarse split core is the uniform core at the
        full delivery grid, so that is what it is measured against.  The
        assertion is an ORDERING plus a cost bound, not a ratio: the
        coarse core must be at least as accurate while contracting
        strictly less than a third of the entries.

        MEASURED margins at these points: d=1 2.9e-07 vs 1.0e-07 at 1/4
        the entries; d=2 3.4e-04 vs 2.1e-04 at 1/16.  The sizing is NOT
        ``n^(sigma_blk/sigma_core)`` with unit prefactor — that
        under-sizes by a measured factor of 2.1 to 3.0 — which is why
        the shipped rule carries a calibrated kappa and why this test
        pins explicit n_core values rather than recomputing the rule.
        """
        nus = [nu] * len(K4SUB)
        ref = hybrid_zeta(K4SUB, nus, A, ref_n, sp_n_points=ref_sp)
        err_coarse = abs(hybrid_zeta(K4SUB, nus, A, n, sp_n_points=ref_sp,
                                     core_n_points=n_core) - ref) / abs(ref)
        err_uniform = abs(hybrid_zeta(K4SUB, nus, A, n) - ref) / abs(ref)
        assert err_coarse <= err_uniform
        assert (n_core / n) ** (2 * d) < 1.0 / 3.0

    def test_core_finer_than_n_points_is_refused(self):
        with pytest.raises(ValueError, match="exceeds n_points"):
            hybrid_zeta(K4SUB, [NU] * len(K4SUB), A1, 8, core_n_points=16)

    def test_infinite_nu_is_refused(self):
        r"""The NN indicator needs ``n`` above the block's longest cycle
        for the constraint to close, so a coarser core changes which
        walks exist rather than approximating the answer.
        """
        with pytest.raises(ValueError, match="nu = inf"):
            hybrid_zeta(K4SUB, [np.inf] * len(K4SUB), A1, 16,
                        core_n_points=8)
