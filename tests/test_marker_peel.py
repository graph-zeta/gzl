# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Gates for the offset-aware marker peel.

The Z2 marker's half axis used to block the FFT peel (``_conv_partner``
refused marker-touching steps) and a symbolic pre-pass
(``_plan_marker``) traded the guaranteed factor-2 of the half axis
against the blocked FFT.  The peel is now *offset-aware*: a half axis
is a window with a different origin and extent, the kernel sub-range /
transform length / output slice are derived from the two endpoint
windows (see ``_fft_conv_step``), and the marker is the last eliminated
vertex unconditionally.  The pre-pass and its scope builder are
deleted.

The fatal-flaw class this file guards: a marker elimination that peels
but *drops the Z2 weights* undercounts every ``x_1 > 0`` slice by 2x —
no exception, no NaN, no shape signal, and every downstream Richardson
fit converges smoothly to the wrong number.  Hence gate (a) asserts the
fold at the executor, (d) that the FFT step really fires on marker
steps (impossible before the peel was offset-aware), and (c) the
end-to-end ladders against values frozen before that change
(``tests/fixtures/marker_peel_refs.json``) with the fit residual
re-checked.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

import gzl.direct_sum as ds


PATH = [(0, 1), (1, 2)]

_REFS = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "marker_peel_refs.json")
    .read_text()
)["records"]


def _z2_weights(m, d):
    axes = [np.arange((m + 1) // 2)] + [np.arange(m)] * (d - 1)
    pos = np.array(np.meshgrid(*axes, indexing="ij")).reshape(d, -1).T
    return np.where(pos[:, 0] == 0, 1.0, 2.0)


def _marker_peel_counter(monkeypatch):
    """Record which peels touched the marker, and count FFT steps."""
    seen = {"fft": 0, "marker_w": 0, "marker_u": 0, "plain": 0}
    orig_step = ds._fft_conv_step
    orig_peel = ds.BoxTruncation.peel_step

    def counting_step(*a, **kw):
        seen["fft"] += 1
        return orig_step(*a, **kw)

    def counting_peel(self, bucket, token, w, out_axes, dtype):
        u, _ = token
        if self._half(w):
            seen["marker_w"] += 1
        elif self._half(u):
            seen["marker_u"] += 1
        else:
            seen["plain"] += 1
        return orig_peel(self, bucket, token, w, out_axes, dtype)

    monkeypatch.setattr(ds, "_fft_conv_step", counting_step)
    monkeypatch.setattr(ds.BoxTruncation, "peel_step", counting_peel)
    return seen


class TestPeelGeometry:
    """The plan's window formulas, pinned per axis."""

    @pytest.mark.parametrize("d", [1, 2])
    def test_windows(self, d):
        L = 4
        m = 2 * L + 1
        trunc = ds.BoxTruncation(L, d, np.eye(d), marker=1)

        # full/full: the historic whole-range step
        w_ext, u_ext, start = trunc._peel_geometry(2, 3)
        assert w_ext == (m,) * d and u_ext == (m,) * d
        assert start == (0,) * d

        # partner is the marker: kdiff[L : 4L+1], outputs [2L : 3L+1]
        w_ext, u_ext, start = trunc._peel_geometry(2, 1)
        assert w_ext[0] == m and u_ext[0] == L + 1
        assert start[0] == L
        assert w_ext[0] + u_ext[0] - 1 == 3 * L + 1
        # trailing axes stay full
        assert all(w_ext[c] == u_ext[c] == m and start[c] == 0
                   for c in range(1, d))

        # eliminated vertex is the marker: kdiff[0 : 3L+1],
        # outputs [L : 3L+1]
        w_ext, u_ext, start = trunc._peel_geometry(1, 2)
        assert w_ext[0] == L + 1 and u_ext[0] == m
        assert start[0] == 0
        assert w_ext[0] + u_ext[0] - 1 == 3 * L + 1


def _marker_factors(L, d, nu, spectator=False):
    """Hand-built factor list with vertex 1 = marker, peel partner 3.

    ``spectator=True`` adds a dense factor on {1, 2} so the peel of
    vertex 1 carries a spectator axis (chunkable)."""
    m = 2 * L + 1
    A = np.eye(d)
    gen = ds._conv_kernel_diff(float(nu), A, L, d)
    h_ext = ds._axis_extents(L, d, True)
    f_ext = ds._axis_extents(L, d, False)
    h_org = ds._axis_origins(L, d, True)
    f_org = ds._axis_origins(L, d, False)
    n_half = int(np.prod(h_ext))
    n_full = m ** d

    factors = [
        {"scope": (1, 3), "gen": gen, "ext": (h_ext, f_ext),
         "off": ds._gen_offsets(L, d, h_org, f_org), "d": d,
         "conv_nu": float(nu)},
        {"scope": (1,), "gen": gen, "ext": (h_ext,),
         "off": tuple(int(h_org[c] + 2 * L) for c in range(d)), "d": d},
        {"scope": (3,), "gen": gen, "ext": (f_ext,),
         "off": tuple(int(f_org[c] + 2 * L) for c in range(d)), "d": d},
    ]
    if spectator:
        rng = np.random.default_rng(11)
        factors.append({"scope": (1, 2),
                        "tensor": rng.random((n_half, n_full)) + 0.5})
    return factors, n_full, n_half


class TestZ2FoldAtTheExecutor:
    """Gate (a): eliminating the MARKER by peel must reproduce the
    dense marker elimination (which applies the Z2 weights) exactly to
    round-off — asserted at ``_eliminate_all``, the executor, not on a
    hand-symmetrised phi."""

    @pytest.mark.parametrize("d,L", [(1, 4), (2, 3)])
    def test_w_marker_peel_matches_dense(self, d, L, monkeypatch):
        m = 2 * L + 1
        nu = d + 1.5
        z2 = _z2_weights(m, d)

        def run():
            factors, n_full, n_half = _marker_factors(L, d, nu)
            out = ds._eliminate_all(
                factors, [1, 3], marker=1,
                n_full=n_full, n_half=n_half, z2_weights=z2,
                A=np.eye(d), L=L, d=d,
            )
            r = 1.0
            for f in out:
                r *= float(np.asarray(f["tensor"]))
            return r

        seen = _marker_peel_counter(monkeypatch)
        monkeypatch.setattr(ds, "_FFT_MARGIN", 0.0)
        monkeypatch.setattr(ds, "_FFT_MIN_BAG", 0)
        monkeypatch.setattr(ds, "_USE_CONV", True)
        v_on = run()
        assert seen["marker_w"] >= 1, "the w = marker peel never ran"
        assert seen["fft"] >= 1

        monkeypatch.setattr(ds, "_USE_CONV", False)
        v_off = run()
        assert abs(v_on - v_off) <= 1e-15 * abs(v_off)

    @pytest.mark.parametrize("d,L", [(1, 4), (2, 3)])
    def test_u_marker_peel_matches_dense(self, d, L, monkeypatch):
        """The mirror case: peeling TOWARD the marker (u = marker).
        No weights are involved at this step — they apply when the
        marker itself is eliminated — but the sub-range/slice geometry
        is the mirror image and deserves its own executor gate."""
        m = 2 * L + 1
        nu = d + 1.5
        z2 = _z2_weights(m, d)
        gen_kw = dict(marker=1, z2_weights=z2, A=np.eye(d), L=L, d=d)

        def run():
            # kernel {1, 3} again, but eliminate 3 first: bucket
            # {kernel(1,3), pin(3)} peels toward u = 1 = marker.
            factors, n_full, n_half = _marker_factors(L, d, nu)
            out = ds._eliminate_all(
                factors, [3, 1],
                n_full=n_full, n_half=n_half, **gen_kw,
            )
            r = 1.0
            for f in out:
                r *= float(np.asarray(f["tensor"]))
            return r

        seen = _marker_peel_counter(monkeypatch)
        monkeypatch.setattr(ds, "_FFT_MARGIN", 0.0)
        monkeypatch.setattr(ds, "_FFT_MIN_BAG", 0)
        monkeypatch.setattr(ds, "_USE_CONV", True)
        v_on = run()
        assert seen["marker_u"] >= 1, "the u = marker peel never ran"

        monkeypatch.setattr(ds, "_USE_CONV", False)
        v_off = run()
        assert abs(v_on - v_off) <= 1e-15 * abs(v_off)


class TestMarkerPeelSentinel:
    """Gate (d): ``_fft_conv_step`` fires on marker-touching steps of
    the real engines — impossible before the peel was offset-aware,
    when ``_conv_partner`` refused them."""

    def test_u_marker_fires_in_the_vacuum_engine(self, monkeypatch):
        seen = _marker_peel_counter(monkeypatch)
        monkeypatch.setattr(ds, "_FFT_MARGIN", 0.0)
        monkeypatch.setattr(ds, "_FFT_MIN_BAG", 0)
        fft_before = seen["fft"]
        ds.direct_sum_zero_momentum(PATH, np.full(2, 2.5), np.eye(1), 6,
                                    root=0)
        assert seen["marker_u"] >= 1
        assert seen["fft"] > fft_before

    def test_w_marker_fires_in_the_open_terminal_engine(self,
                                                        monkeypatch):
        seen = _marker_peel_counter(monkeypatch)
        monkeypatch.setattr(ds, "_FFT_MARGIN", 0.0)
        monkeypatch.setattr(ds, "_FFT_MIN_BAG", 0)
        em = {(0, 1): 2.5, (1, 2): 2.5}
        ds._direct_sum_open_terminal(em, np.eye(1), 6, 0, 2, 1)
        assert seen["marker_w"] >= 1
        assert seen["fft"] >= 1


class TestChunkedMarkerPeel:
    """The chunked peel is bit-identical on marker-touching steps, in
    both marker roles, with the Z2 fold applied per chunk."""

    @pytest.mark.parametrize("role", ["w_marker", "u_marker"])
    @pytest.mark.parametrize("chunk", [1, 3])
    def test_bit_identical(self, role, chunk, monkeypatch):
        d, L = 1, 4
        nu = 2.5
        m = 2 * L + 1
        z2 = _z2_weights(m, d)
        elimination = [1] if role == "w_marker" else [3]

        def run(force_chunk):
            monkeypatch.setattr(ds, "_FORCE_CHUNK", force_chunk)
            factors, n_full, n_half = _marker_factors(
                L, d, nu, spectator=(role == "w_marker"))
            if role == "u_marker":
                # spectator on the eliminated FULL vertex 3 instead
                rng = np.random.default_rng(11)
                factors.append({"scope": (2, 3),
                                "tensor": rng.random((n_full, n_full))
                                + 0.5})
            out = ds._eliminate_all(
                factors, elimination, marker=1,
                n_full=n_full, n_half=n_half, z2_weights=z2,
                A=np.eye(d), L=L, d=d,
            )
            (res,) = [f for f in out if len(f["scope"]) == 2]
            return np.asarray(res["tensor"])

        seen = _marker_peel_counter(monkeypatch)
        monkeypatch.setattr(ds, "_FFT_MARGIN", 0.0)
        monkeypatch.setattr(ds, "_FFT_MIN_BAG", 0)
        base = run(None)
        got = run(chunk)
        key = "marker_w" if role == "w_marker" else "marker_u"
        assert seen[key] >= 2, f"{role} peel did not run in both arms"
        assert np.array_equal(base, got)

    def test_u_marker_output_axis_is_the_half_axis(self, monkeypatch):
        """Peeling toward the marker relabels the summed axis to the
        marker's HALF axis — the output length must be n_half, not
        n_full (deriving it from the generator would be wrong by
        (2 - 1/(2L+1))^d)."""
        d, L = 1, 4
        z2 = _z2_weights(2 * L + 1, d)
        monkeypatch.setattr(ds, "_FFT_MARGIN", 0.0)
        monkeypatch.setattr(ds, "_FFT_MIN_BAG", 0)
        factors, n_full, n_half = _marker_factors(L, d, 2.5)
        out = ds._eliminate_all(
            factors, [3], marker=1,
            n_full=n_full, n_half=n_half, z2_weights=z2,
            A=np.eye(d), L=L, d=d,
        )
        # The lazy pin factor shares scope (1,); pick the peel result,
        # which carries a dense tensor.
        (res,) = [f for f in out
                  if f["scope"] == (1,) and f.get("tensor") is not None]
        assert np.asarray(res["tensor"]).shape == (n_half,)


class TestLadderRefs:
    """Gate (c): the shipped ladders against values frozen before the
    offset-aware peel, at 1e-12 relative, with the Richardson fit
    residual re-checked (a basis/executor desync fits the wrong power
    while the solve still succeeds and the value stays plausible)."""

    @pytest.mark.parametrize(
        "rec", _REFS, ids=[r["name"] for r in _REFS])
    def test_ladder(self, rec):
        edges = [tuple(e) for e in rec["edges"]]
        nu_vec = np.full(len(edges), rec["nu"])
        A = np.eye(rec["d"])
        Ls = tuple(rec["L_list"])
        K = rec["n_correction_terms"]

        S = np.array([
            float(np.real(complex(ds.direct_sum_zero_momentum(
                edges, nu_vec, A, L)))) for L in Ls
        ])
        v = complex(ds.direct_sum_extrapolated(
            edges, nu_vec, A, L_list=Ls, n_correction_terms=K))
        ref = float.fromhex(rec["value_hex"])
        assert abs(float(np.real(v)) - ref) <= 1e-12 * abs(ref)

        # Raw rungs against their frozen values: the d = 1 ladder is
        # gate-closed (bit-identical before and after the offset-aware
        # peel); d >= 2 moved <= 4e-16 measured.
        S_ref = np.array([float.fromhex(h) for h in rec["S_hex"]])
        assert np.max(np.abs(S - S_ref)) <= 1e-13 * np.max(np.abs(S_ref))

        # The residual is a PHYSICAL basis-truncation quantity (1e-11
        # to 4e-7 at d = 1), not round-off, so "small in absolute
        # terms" is the wrong gate: it is compared against its own
        # frozen baseline instead.  A basis/executor desync inflates
        # it by orders of magnitude; factor 2 is far outside numerical
        # drift and far inside any real desync.  Only an overdetermined
        # fit has a residual: the d >= 2 ladders have as many rungs as
        # basis functions, the solve is exact for any exponent, and
        # their residual is round-off.
        if len(Ls) <= K + 1:
            return
        edge_map = ds._collapse_multi_edges(np.asarray(edges), nu_vec)
        V = max(max(u, w) for u, w in edge_map) + 1
        root_eff = ds._pick_root(edge_map, V)
        alpha = rec["d"] - ds._min_free_cut_nu(edge_map, root_eff)
        Lf = np.array(Ls, dtype=float)
        design = np.stack([np.ones_like(Lf)] +
                          [Lf ** (alpha - k) for k in range(K)], axis=1)
        coef, res, *_ = np.linalg.lstsq(design, S, rcond=None)
        resid = (float(np.sqrt(res[0])) if len(res)
                 else float(np.linalg.norm(design @ coef - S)))
        resid_ref = float.fromhex(rec["residual_hex"])
        assert resid <= 2.0 * resid_ref + 1e-13 * np.max(np.abs(S_ref)), (
            f"fit residual {resid:.3e} vs frozen {resid_ref:.3e} — "
            f"basis/executor desync"
        )
