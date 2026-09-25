# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Executor gate: ``hybrid._dense_core`` delegates to the shared executor.

The hybrid's private dense-contraction loop was replaced by
``_contract.eliminate`` on a ``TorusTruncation`` — a *value change*
(different contraction association; the FFT peel is now reachable from
the hybrid core), made while production never reached the dense core
(all 299/299 observed production hybrid calls then SP-reduced to a
(2, 1) residue; the docstring of
``tests/fixtures/_freeze_hybrid_s7_refs.py`` says why that no longer
holds).  The gate is against values frozen before the swap
(``tests/fixtures/hybrid_s7_refs.json``):

* **core tensor M, peel off** — per-entry **<= 4 ulp**.  This is the
  pure executor-swap reassociation; measured max 3 ulp.  Per-entry ulp
  is meaningful on M because every entry is a same-sign kernel sum.
* **core tensor M, default (peel on)** — **<= 8 ulp of the largest
  entry**.  The FFT peel's reassociation is norm-wise, so entries far
  below the max carry relatively larger noise (measured up to 29
  per-entry ulp on M entries orders of magnitude below the peak, at a
  max *relative* move of 7.6e-16).  Gating those per entry would gate
  FFT round-off, not the swap.
* **public surface (both peel states)** — **<= 8 ulp of the largest
  entry**.  Grid mode applies ``fftn`` to M; small grid entries arise by
  cancellation, so per-entry ulp there is meaningless even for the
  dense path (measured 32 per-entry ulp at 2.6 ulp of scale).

The fixture set is self-validating: the two dense branches (old
pairwise einsum vs the executor's axis-sliced loop) are bit-identical
on a 2-factor bucket eliminating the leading axis, so a test that only
exercised that case would certify the equivalence exactly where it
cannot fail.  A recorder asserts the sweep hits a >= 3-factor bucket, a
bucket whose eliminated vertex is the *last* axis of the union scope,
and at least one fired peel (zero with the kill switch off).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

import gzl.tensor_network as tn
from gzl.hybrid import (
    HybridCoreTooLargeError,
    _dense_core,
    _sp_reduce,
    hybrid_zeta,
)
from gzl.tensor_network import (
    TorusTruncation,
    _edge_kernel_torus,
    graph_zeta_general_at_zero,
)


_REFS = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "hybrid_s7_refs.json")
    .read_text()
)["records"]

_IDS = [r["name"] for r in _REFS]


@pytest.fixture(autouse=True)
def _hold_the_historical_pin(monkeypatch):
    r"""Pin at the CALLER's source, as the fixtures were recorded.

    These references were frozen to gate ONE change: the hybrid's
    private dense loop delegating to the shared executor.  The
    planner pin (``hybrid.USE_PLANNER_PIN``) is a later and separate
    change that picks a different vertex to eliminate around, which is
    value-exact by translation invariance but reassociates the sum — it
    moves ``core7_d1_vacuum`` by 1.4 ulp on macOS/arm64 and past the
    8-ulp gate on Linux/x86_64.

    Letting it drift in here would silently turn an executor-swap gate
    into a two-variable comparison, so it is held fixed.  The pin has its
    own value-exactness gate in ``tests/test_hybrid_pin.py``, which is
    where that change belongs.
    """
    monkeypatch.setattr("gzl.hybrid.USE_PLANNER_PIN", False)


def _entry_ulp(a, b) -> int:
    """Max per-entry ulp distance between two float64 arrays."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ia = a.view(np.int64).copy()
    ib = b.view(np.int64).copy()
    # map the IEEE-754 bit patterns to a lexicographic integer scale
    ia[ia < 0] = np.int64(-(2 ** 63)) - ia[ia < 0]
    ib[ib < 0] = np.int64(-(2 ** 63)) - ib[ib < 0]
    return int(np.max(np.abs(ia - ib)))


def _scale_ulp(got, ref) -> float:
    """Largest absolute move in units of the largest entry's ulp."""
    got = np.asarray(got, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return float(np.max(np.abs(got - ref))
                 / np.spacing(np.max(np.abs(ref))))


def _from_hex(hex_list, shape):
    return np.array([float.fromhex(h) for h in hex_list]).reshape(shape)


def _args(rec):
    edges = [tuple(int(x) for x in e) for e in rec["edges"]]
    nu_vec = np.full(len(edges), float(rec["nu"]))
    A = np.asarray(rec["A"], dtype=float)
    return edges, nu_vec, A, int(rec["n_points"])


def _public_of(rec) -> np.ndarray:
    edges, nu_vec, A, n = _args(rec)
    kw = {}
    if rec["terminal"] is not None:
        kw["terminal"] = int(rec["terminal"])
    if rec["momentum"] is not None:
        kw["momentum"] = np.asarray(rec["momentum"], dtype=float)
    return np.asarray(hybrid_zeta(edges, nu_vec, A, n, **kw), dtype=float)


def _core_of(rec) -> np.ndarray:
    """The position-space core tensor M, via the same reduction
    ``hybrid_zeta`` runs (mirrors the freeze script)."""
    edges, nu_vec, A, n = _args(rec)
    d = A.shape[0]
    term = int(rec["terminal"]) if rec["terminal"] is not None else 0
    kernels = [_edge_kernel_torus(float(nu_vec[i]), A, n)
               for i in range(len(edges))]
    nodes, r_edges, r_kernels, sp_scalar = _sp_reduce(
        edges, kernels, 0, term, n, d)
    M = _dense_core(nodes, r_edges, r_kernels, 0, term, n, d, A,
                    5e7, 1e12) * sp_scalar
    return np.asarray(M, dtype=float)


@pytest.mark.parametrize("rec", _REFS, ids=_IDS)
def test_core_dense_path_within_4_ulp_per_entry(rec, monkeypatch):
    """Peel off: the executor swap alone moves M by <= 4 ulp/entry."""
    monkeypatch.setattr(tn, "_USE_CONV", False)
    got = _core_of(rec)
    ref = _from_hex(rec["core_hex"], rec["core_shape"])
    assert _entry_ulp(got, ref) <= 4


@pytest.mark.parametrize("rec", _REFS, ids=_IDS)
def test_core_default_path_within_round_off_of_scale(rec):
    """Default (peel on): M moves <= 8 ulp of its largest entry."""
    got = _core_of(rec)
    ref = _from_hex(rec["core_hex"], rec["core_shape"])
    assert _scale_ulp(got, ref) <= 8.0


@pytest.mark.parametrize("use_conv", [True, False],
                         ids=["peel-on", "peel-off"])
@pytest.mark.parametrize("rec", _REFS, ids=_IDS)
def test_public_surface_within_round_off_of_scale(rec, use_conv,
                                                  monkeypatch):
    monkeypatch.setattr(tn, "_USE_CONV", use_conv)
    got = _public_of(rec)
    ref = _from_hex(rec["value_hex"], rec["shape"])
    assert got.shape == ref.shape
    assert _scale_ulp(got, ref) <= 8.0


def test_fixture_set_exercises_the_differing_branches(monkeypatch):
    """Self-validation of the sweep, in both kill-switch directions."""
    seen = {"bucket3": 0, "last_axis": 0, "peel": 0, "dense": 0}
    orig_dense = TorusTruncation.dense_step
    orig_peel = TorusTruncation.peel_step

    def _classify(bucket, v):
        union = sorted({a for f in bucket for a in f[0]})
        if len(bucket) >= 3:
            seen["bucket3"] += 1
        if union and union[-1] == v:
            seen["last_axis"] += 1

    def rec_dense(self, bucket, v, out_axes, dtype):
        _classify(bucket, v)
        seen["dense"] += 1
        return orig_dense(self, bucket, v, out_axes, dtype)

    def rec_peel(self, bucket, token, v, out_axes, dtype):
        _classify(bucket, v)
        seen["peel"] += 1
        return orig_peel(self, bucket, token, v, out_axes, dtype)

    monkeypatch.setattr(TorusTruncation, "dense_step", rec_dense)
    monkeypatch.setattr(TorusTruncation, "peel_step", rec_peel)

    for rec in _REFS:
        _public_of(rec)
    assert seen["bucket3"] >= 1, "no >= 3-factor bucket in the sweep"
    assert seen["last_axis"] >= 1, \
        "no bucket eliminating the last axis of its union scope"
    assert seen["peel"] >= 1, "the peel never fired (vacuous sweep)"
    assert seen["dense"] >= 1, "the dense branch never fired"

    # Other direction: with the kill switch off the peel may not fire.
    before = dict(seen)
    monkeypatch.setattr(tn, "_USE_CONV", False)
    for rec in _REFS:
        _public_of(rec)
    assert seen["peel"] == before["peel"], \
        "peel fired despite _USE_CONV = False"
    assert seen["dense"] > before["dense"]


def test_oversized_bucket_refused_before_contraction(monkeypatch):
    """K29 (plus a dangling source edge): every elimination order has a
    bucket with 28 output axes > 26 einsum letters.  The symbolic
    schedule pass must refuse it as HybridCoreTooLargeError — which the
    frontend can fall back on — BEFORE any bucket tensor is allocated,
    not raise NotImplementedError mid-contraction."""
    stepped = []
    orig_dense = TorusTruncation.dense_step

    def counting_dense(self, bucket, v, out_axes, dtype):
        stepped.append(v)
        return orig_dense(self, bucket, v, out_axes, dtype)

    monkeypatch.setattr(TorusTruncation, "dense_step", counting_dense)
    edges = [(0, 1)] + [(a, b) for a in range(1, 30)
                        for b in range(a + 1, 30)]
    nu_vec = np.full(len(edges), 2.5)
    with pytest.raises(HybridCoreTooLargeError):
        hybrid_zeta(edges, nu_vec, np.eye(1), 2)
    assert not stepped, "contraction started before the refusal"


def test_nu_inf_core_stays_integral_and_off_the_peel(monkeypatch):
    """peelable=False (nu = inf anywhere) densifies every factor after
    the budget guard: the NN-indicator contraction stays exact integer
    arithmetic and the FFT peel never fires — an FFT round trip would
    return 2.0 as 1.9999999999999998.  A finite-nu control asserts the
    counter is live."""
    peels = []
    orig_peel = TorusTruncation.peel_step

    def counting_peel(self, bucket, token, v, out_axes, dtype):
        peels.append(v)
        return orig_peel(self, bucket, token, v, out_axes, dtype)

    monkeypatch.setattr(TorusTruncation, "peel_step", counting_peel)
    # The prism (irreducible, 3-regular) on the triangular lattice: its
    # two triangles map onto lattice triangles, so the NN count is a
    # genuinely positive integer (168 at n = 6) — K4 or any core on a
    # hypercubic lattice would count 0 (triangle-free NN graph) and
    # assert integrality vacuously.
    PRISM = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
             (0, 3), (1, 4), (2, 5)]
    HEX = np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2.0]])

    # Vacuum: the value is a plain homomorphism-style count, exactly
    # integral.  (Grid mode would apply fftn, whose phase factors are
    # not integers — integrality is only assertable pre-transform.)
    # hybrid now REFUSES nu = inf outright.  It was exact here only
    # because the prism is SP-irreducible, so no FFT collapse runs; on a
    # block WITH an SP part it is silently wrong (99.99999999999996 for
    # 100 on a theta at d = 2), and a caller cannot tell the two classes
    # apart.  The router never sent nu = inf here in any case.
    with pytest.raises(ValueError, match="nu = inf"):
        hybrid_zeta(PRISM, np.full(9, np.inf), HEX, 6)
    assert not peels, "the refusal must happen before any contraction"

    # The value itself is unchanged and still exactly integral -- from
    # the engine that is exact for EVERY block at nu = inf.
    val = np.asarray(graph_zeta_general_at_zero(
        np.array(PRISM, dtype=int), np.full(9, np.inf), HEX, 6,
        pinned_vertex=0))
    assert np.array_equal(val, np.round(val))
    assert float(np.real(val)) == 168.0

    hybrid_zeta(PRISM, np.full(9, 2.5), HEX, 6)
    assert peels, "finite-nu control did not peel — counter is dead"
