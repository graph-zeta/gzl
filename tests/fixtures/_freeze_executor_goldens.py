# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Regenerate ``tests/fixtures/executor_goldens.json``.

A refactor of the executors that moves arithmetic between modules is
held to a *bit-identity* acceptance criterion.  That criterion needs an
instrument: a frozen set of executor inputs together with the exact
float64 bits the current tree produces for them.  This script builds
that set; :mod:`tests.test_executor_goldens` consumes it.

Each record is labelled along **two independent axes**, because one
label conflating them over-claims in whichever direction it is read.

``origin`` — hand-declared, about the *parameters*:

``captured``
    Copied verbatim from a frontend call observed by monkeypatching the
    executor names over a corpus pass (:func:`_capture_production`).

``constructed``
    Built to reach a defect class the corpus pass does not exercise
    cheaply — d = 3, a skew lattice, ``use_symmetry=False``, a forced
    FFT gate, a ``nu = inf`` kernel.  A d = 1 / orthogonal-only freeze
    is *vacuous* for the orientation defect class: the torus kernel is
    exactly even on every diagonal lattice and at every odd ``n``, so a
    reversed edge there is a no-op.  Every ``n_points`` this repo uses
    (16, 24, 32, 128, 256) is even, which is why the even-``n`` skew
    cells are mandatory rather than decorative.

``executor_reach`` — derived at freeze time (:func:`_classify`), about
the *executor*.  Never written by hand: a hand-written reachability
label drifts silently when routing changes, because the value still
matches and nothing fails.

``production``
    The frontend was observed calling this executor by name.

``production-inner``
    Not named by the frontend, but called internally by a route that
    was.  Freezing at this depth is deliberate — the Richardson
    combination sitting above ``direct_sum_zero_momentum`` would blur a
    last-bit move into its own noise.

``unreached``
    Not reached by the capture at all.  Not automatically wrong, but it
    means no shipped value depends on this record.

Tolerances are attached per record, not chosen by the consumer, and
follow the per-engine policy:

===============================  =========================================
executor                         gate
===============================  =========================================
``graph_zeta_general``           exact (it is the oracle for two other
``graph_zeta_general_at_zero``   subsystems, whose own gates have 246x
                                 and 30158x headroom and would hide a
                                 real regression)
``direct_sum_zero_momentum``     exact
``_direct_sum_open_terminal``    exact
``direct_sum_extrapolated``      1e-12 relative (a Richardson solve
``_direct_sum_extrapolated_grid``amplifies a pure reassociation ~14x;
                                 1e-12 leaves ~350x over real noise)
``hybrid_zeta``                  4e-15 relative to the largest entry.
                                 NOT per-entry ulp: grid mode ends in
                                 an ``fftn`` whose small entries arise
                                 by cancellation, so a sub-ulp-of-scale
                                 move reads as many per-entry ulp there
                                 (the dense-core executor swap, S7 in
                                 ``_freeze_hybrid_s7_refs.py``,
                                 measured 4 elementwise ulp on macOS
                                 and 8 on the Linux runner at
                                 <= 6.8e-16 of scale).
                                 4e-15 covers measured cross-env noise
                                 ~2x over and sits 11 orders below the
                                 smallest real hybrid defect (the
                                 4.9e-04 orientation defect, since fixed).
===============================  =========================================

Run from the repository root::

    python tests/fixtures/_freeze_executor_goldens.py

Re-freezing is a deliberate act: it resets the reference that any
refactor in progress is checked against, so a regenerated file belongs in
its own commit with the reason in the message.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


def fingerprint():
    """Identify the environment whose float64 bits were frozen.

    Bit-identity is a same-environment property, not a property of this
    library.  ``numpy``'s ``power``, ``cos`` and FFT all reduce
    differently across libm, BLAS and SIMD builds — measured 1 to 8 ulp
    between macOS/Accelerate and the Linux CI runner on 14 of 41
    records, including a *diagonal* d = 2 lattice and several d = 1
    cases that no BLAS-path argument predicts.  So the freeze records
    the environment, and the consumer decides which gate it is entitled
    to run.
    """
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": ".".join(str(v) for v in sys.version_info[:2]),
        "numpy": np.__version__,
    }

# Running this file BY PATH puts its own directory (tests/fixtures) on
# sys.path[0], not the repository root, so ``import gzl`` falls
# through to whatever the editable install points at -- on a machine with
# several worktrees, very likely a different checkout.  This script's
# whole job is to record what THIS tree does, so a silent resolution to
# another one makes every value and every reach label a statement about
# the wrong code.
#
# Not hypothetical: it happened here.  The other tree lacked a kwarg this
# one has, every captured call raised TypeError, the capture's
# ``except Exception: pass`` swallowed all of them, and the freeze
# recorded every executor as "unreached" -- a confident, wrong, green
# answer.  Hence this, and the liveness check in _capture_production.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import gzl                                              # noqa: E402
if not str(Path(gzl.__file__).resolve()).startswith(str(_REPO)):
    raise SystemExit(
        f"WRONG TREE: gzl resolved to {gzl.__file__}\n"
        f"            expected somewhere under {_REPO}"
    )

import gzl.direct_sum as ds
import gzl.frontend as fe
import gzl.hybrid as hy
import gzl.tensor_network as tn
from gzl import data_path

HERE = Path(__file__).resolve().parent
OUT = HERE / "executor_goldens.json"

# --- lattices -------------------------------------------------------------
A1 = [[1.0]]
A1_SCALED = [[1.3]]
A2_SQUARE = [[1.0, 0.0], [0.0, 1.0]]
A2_TRI = [[1.0, 0.5], [0.0, float(np.sqrt(3.0) / 2.0)]]
A3_CUBIC = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
A3_OBLIQUE = [[1.0, 0.3, 0.1], [0.0, 1.2, 0.2], [0.1, 0.0, 0.9]]

# --- topologies -----------------------------------------------------------
K4 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
K5 = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3],
      [1, 4], [2, 3], [2, 4], [3, 4]]
PRISM = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3],
         [0, 3], [1, 4], [2, 5]]
# K4 with a degree-1 vertex hung off it: the dangling vertex empties a
# bucket, which is the branch an earlier defect lived in and the one a
# cost-ordered schedule makes live.
K4_TAIL = K4 + [[3, 4]]
# Two distinct vertex pairs of equal collapsed nu sharing an eliminated
# vertex — the duplicate-shaped-factor case.
DIAMOND = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]
C4 = [[0, 1], [1, 2], [2, 3], [3, 0]]
# A treewidth-2 block the frontend hands to the hybrid engine (captured).
TW2_BLOCK = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]]
TW2_WIDE = [[0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4], [1, 5]]


# ---------------------------------------------------------------------------
# Bit-exact encoding
# ---------------------------------------------------------------------------

def _encode(value):
    """Encode a float/complex scalar or ndarray with no bits lost.

    ``float.hex`` round-trips exactly, which decimal repr does not
    promise across platforms.  NaN and the infinities are passed through
    as their literal names because ``float.hex`` refuses them.
    """
    arr = np.asarray(value)
    is_complex = np.iscomplexobj(arr)
    real = np.ascontiguousarray(arr.real, dtype=np.float64).reshape(-1)
    out = {
        "shape": list(arr.shape),
        "complex": bool(is_complex),
        "real_hex": [_hex(x) for x in real],
    }
    if is_complex:
        imag = np.ascontiguousarray(arr.imag, dtype=np.float64).reshape(-1)
        out["imag_hex"] = [_hex(x) for x in imag]
    return out


def _hex(x: float) -> str:
    x = float(x)
    if np.isnan(x):
        return "nan"
    if np.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x.hex()


def _decode(enc):
    """Inverse of :func:`_encode`; returns float64/complex128."""
    shape = tuple(enc["shape"])
    real = np.array([float.fromhex(h) if h not in ("nan", "inf", "-inf")
                     else float(h) for h in enc["real_hex"]],
                    dtype=np.float64)
    if enc.get("complex"):
        imag = np.array([float.fromhex(h) if h not in ("nan", "inf", "-inf")
                         else float(h) for h in enc["imag_hex"]],
                        dtype=np.float64)
        return (real + 1j * imag).reshape(shape)
    return real.reshape(shape)


# ---------------------------------------------------------------------------
# Executor dispatch — one place, shared with the test module
# ---------------------------------------------------------------------------

_MODULES = {"direct_sum": ds, "tensor_network": tn, "hybrid": hy}


def run_record(rec):
    """Evaluate one golden record and return its raw executor output.

    ``module_globals`` sets module-level tunables for the duration of
    the call and restores them afterwards.  It exists for one reason:
    the box's FFT peel is *cost-gated*, and the gate is shut at every
    d = 1 rung the ladders use, so a freeze set built only on default
    tunables would never execute the peel at d = 1 — precisely the code
    a factor-supply or chunking refactor rewrites.  A golden that cannot
    reach the branch under test certifies nothing.
    """
    overrides = rec.get("module_globals") or {}
    saved = {}
    for dotted, value in overrides.items():
        mod_name, attr = dotted.split(".", 1)
        mod = _MODULES[mod_name]
        saved[dotted] = getattr(mod, attr)
        setattr(mod, attr, value)
    try:
        return _run_record_inner(rec)
    finally:
        for dotted, value in saved.items():
            mod_name, attr = dotted.split(".", 1)
            setattr(_MODULES[mod_name], attr, value)


def _run_record_inner(rec):
    edges = np.asarray(rec["edges"], dtype=int)
    A = np.asarray(rec["A"], dtype=float)
    nu = np.asarray(rec["nu"], dtype=float)
    kw = dict(rec["kwargs"])
    name = rec["executor"]

    if name == "direct_sum_zero_momentum":
        if "momentum" in kw:
            kw["momentum"] = np.asarray(kw["momentum"], dtype=float)
        return ds.direct_sum_zero_momentum(edges, nu, A, **kw)

    if name == "_direct_sum_open_terminal":
        edge_map = ds._collapse_multi_edges(edges, nu)
        # Returns (M, term_pos).  Only ``M`` is arithmetic; ``term_pos``
        # is the integer [-L, L]^d grid, reproduced independently by the
        # test so a shape change cannot hide behind the value check.
        M, _term_pos = ds._direct_sum_open_terminal(
            edge_map, A, kw["L"], kw["root"], kw["terminal"],
            int(A.shape[0]), use_symmetry=kw.get("use_symmetry", True),
        )
        return M

    if name == "direct_sum_extrapolated":
        kw = dict(kw)
        kw["L_list"] = tuple(kw["L_list"])
        if kw.get("momentum") is not None:
            kw["momentum"] = np.asarray(kw["momentum"], dtype=float)
        return ds.direct_sum_extrapolated(edges, nu, A, **kw)

    if name == "_direct_sum_extrapolated_grid":
        kw = dict(kw)
        kw["L_list"] = tuple(kw["L_list"])
        k_grid = np.asarray(kw.pop("k_grid"), dtype=float)
        return ds._direct_sum_extrapolated_grid(edges, nu, A, k_grid, **kw)

    if name == "graph_zeta_general":
        kw = dict(kw)
        kw["terminals"] = tuple(kw.get("terminals", ()))
        if kw.get("momentum") is not None:
            kw["momentum"] = np.asarray(kw["momentum"], dtype=float)
        return tn.graph_zeta_general(edges, nu, A, kw.pop("n_points"), **kw)

    if name == "graph_zeta_general_at_zero":
        kw = dict(kw)
        return tn.graph_zeta_general_at_zero(
            edges, nu, A, kw.pop("n_points"), **kw)

    if name == "hybrid_zeta":
        kw = dict(kw)
        if kw.get("momentum") is not None:
            kw["momentum"] = np.asarray(kw["momentum"], dtype=float)
        return hy.hybrid_zeta(edges, nu, A, kw.pop("n_points"), **kw)

    raise KeyError(f"unknown executor {name!r}")


# ---------------------------------------------------------------------------
# The freeze set
# ---------------------------------------------------------------------------

EXACT = {"kind": "exact"}
REL_1E12 = {"kind": "relative", "rtol": 1e-12}
# Norm-wise, not per-entry ulp — see the tolerance table in the module
# docstring for the calibration.
REL_HYBRID = {"kind": "relative", "rtol": 4e-15}


# Forces the box's cost gate open.  The Z2 marker is the
# last eliminated vertex unconditionally (the ``_plan_marker`` pre-pass
# that margin 0 used to disable is deleted), so these records run with
# the marker active AND the gate open — marker-touching steps peel
# offset-aware, which is exactly the class these records now guard.
FORCE_BOX_PEEL = {"direct_sum._FFT_MARGIN": 0.0,
                  "direct_sum._FFT_MIN_BAG": 0}


# An executor the frontend never names directly, but which an observed
# route calls internally.  Freezing these is the point — the Richardson
# combination above them would blur a last-bit move into its own noise —
# but calling them "production" without saying how they are reached
# overstates what the capture actually saw.
_INNER_OF = {
    "direct_sum_zero_momentum": "direct_sum_extrapolated",
    "_direct_sum_open_terminal": "_direct_sum_extrapolated_grid",
}


def _classify(executor, observed):
    """Derive executor reachability from the capture, never assert it.

    A hand-written reachability label drifts the moment routing changes,
    and it drifts *silently* — the value still matches, so nothing
    fails, and the freeze set quietly starts describing a pipeline that
    no longer exists.
    """
    if executor in observed:
        return "production"
    if _INNER_OF.get(executor) in observed:
        return "production-inner"
    return "unreached"


def _rec(label, executor, edges, nu, A, kwargs, tol, origin, note="",
         module_globals=None):
    assert origin in ("captured", "constructed"), origin
    n_edges = len(edges)
    nu_vec = [float(nu)] * n_edges if np.isscalar(nu) else [float(x) for x in nu]
    return {
        "label": label,
        "executor": executor,
        "origin": origin,
        "note": note,
        "edges": [[int(u), int(v)] for u, v in edges],
        "nu": nu_vec,
        "A": [[float(x) for x in row] for row in A],
        "kwargs": kwargs,
        "module_globals": dict(module_globals) if module_globals else {},
        "tol": tol,
    }


def build_records():
    """The curated freeze set.  Order is stable; labels are the identity."""
    R = []

    # ---------------- box, raw truncation (bit-identical gate) ----------
    # These target the factor build and the peel directly: both live
    # below direct_sum_zero_momentum, and the Richardson combination
    # above it would blur a last-bit move into its own noise.
    R.append(_rec("box_raw_k4_d1_L6", "direct_sum_zero_momentum",
                  K4, 2.5, A1, {"L": 6, "root": 0}, EXACT, "constructed",
                  "the tw=3 block the frontend actually routes at d=1"))
    R.append(_rec("box_raw_k4_d1_L5_scaled", "direct_sum_zero_momentum",
                  K4, 2.5, A1_SCALED, {"L": 5, "root": 0}, EXACT, "constructed",
                  "a != 1 exercises the d=1 scalar distance path"))
    R.append(_rec("box_raw_k4_d1_L6_nosym", "direct_sum_zero_momentum",
                  K4, 2.5, A1, {"L": 6, "root": 0, "use_symmetry": False},
                  EXACT, "constructed", "marker=None: no Z2 fold, full axes"))
    R.append(_rec("box_raw_prism_d1_L5", "direct_sum_zero_momentum",
                  PRISM, 2.5, A1, {"L": 5, "root": 0}, EXACT, "constructed",
                  "6 vertices, tw=3, more elimination steps than K4"))
    R.append(_rec("box_raw_k5_d1_L4", "direct_sum_zero_momentum",
                  K5, 2.5, A1, {"L": 4, "root": 0}, EXACT, "constructed",
                  "tw=4: the widest bag the shipped ladders reach"))
    R.append(_rec("box_raw_k4tail_d1_L6", "direct_sum_zero_momentum",
                  K4_TAIL, 2.5, A1, {"L": 6, "root": 0}, EXACT, "constructed",
                  "degree-1 vertex: the empty-bucket branch"))
    R.append(_rec("box_raw_diamond_d1_L6", "direct_sum_zero_momentum",
                  DIAMOND, 2.5, A1, {"L": 6, "root": 0}, EXACT, "constructed",
                  "two equal-nu factor shapes share an eliminated vertex"))
    R.append(_rec("box_raw_k4_d2_square_L4", "direct_sum_zero_momentum",
                  K4, 3.0, A2_SQUARE, {"L": 4, "root": 0}, EXACT,
                  "constructed", ""))
    R.append(_rec("box_raw_k4_d2_tri_L4", "direct_sum_zero_momentum",
                  K4, 3.0, A2_TRI, {"L": 4, "root": 0}, EXACT, "constructed",
                  "triangular cell: a skew A the d=1 records cannot see"))
    R.append(_rec("box_raw_k4_d3_cubic_L2", "direct_sum_zero_momentum",
                  K4, 4.5, A3_CUBIC, {"L": 2, "root": 0}, EXACT, "constructed",
                  "d=3, diagonal A: gated bit-identical"))
    R.append(_rec("box_raw_k4_d3_oblique_L2", "direct_sum_zero_momentum",
                  K4, 4.5, A3_OBLIQUE, {"L": 2, "root": 0}, EXACT,
                  "constructed",
                  "d=3 oblique A: the skew counterpart of the cubic record"))
    R.append(_rec("box_raw_k4_d1_L6_finitek", "direct_sum_zero_momentum",
                  K4, 2.5, A1,
                  {"L": 6, "root": 0, "terminal": 3, "momentum": [0.3]},
                  EXACT, "constructed", "cos phase factor on the terminal"))
    R.append(_rec("box_raw_k5_d2_L3", "direct_sum_zero_momentum",
                  K5, 3.0, A2_SQUARE, {"L": 3, "root": 0}, EXACT,
                  "constructed",
                  "the auto-sliced dense class: K5's unpeelable bag-4 "
                  "bucket (3.3e6 elements) takes the axis-sliced branch — "
                  "no other record exercises it.  Exact is honest here: "
                  "the eliminated axis LEADS the bag product, and numpy's "
                  "reduction over a leading axis accumulates sequentially "
                  "— the sliced loop's own association (a trailing-axis "
                  "reduction goes pairwise and would differ; verified "
                  "both ways)"))

    # ---------------- box, peel forced open (bit-identical gate) --------
    # Without these the freeze set never executes the box FFT step at
    # d = 1: _conv_possible(2L+1, 2L+1, 1) is False for L = 4..14 and
    # every shipped d = 1 rung sits inside that window.  The peel is
    # what a factor-supply or chunking refactor rewrites, so leaving it
    # unfrozen would make the bit-identity check half vacuous.
    R.append(_rec("box_peel_k4_d1_L6", "direct_sum_zero_momentum",
                  K4, 2.5, A1, {"L": 6, "root": 0}, EXACT, "constructed",
                  "gate forced: the d=1 FFT step, unreachable by default",
                  module_globals=FORCE_BOX_PEEL))
    R.append(_rec("box_peel_prism_d1_L5", "direct_sum_zero_momentum",
                  PRISM, 2.5, A1, {"L": 5, "root": 0}, EXACT, "constructed",
                  "several peeled steps in one call",
                  module_globals=FORCE_BOX_PEEL))
    R.append(_rec("box_peel_k4_d1_L6_finitek", "direct_sum_zero_momentum",
                  K4, 2.5, A1,
                  {"L": 6, "root": 0, "terminal": 3, "momentum": [0.3]},
                  EXACT, "constructed", "peel alongside the cos phase factor",
                  module_globals=FORCE_BOX_PEEL))
    R.append(_rec("box_peel_k4_d3_cubic_L2", "direct_sum_zero_momentum",
                  K4, 4.5, A3_CUBIC, {"L": 2, "root": 0}, EXACT, "constructed",
                  "d=3 peel: 2^d padding on three axes",
                  module_globals=FORCE_BOX_PEEL))
    R.append(_rec("box_peel_k4_d3_oblique_L2", "direct_sum_zero_momentum",
                  K4, 4.5, A3_OBLIQUE, {"L": 2, "root": 0}, EXACT,
                  "constructed", "d=3 oblique A with the peel live",
                  module_globals=FORCE_BOX_PEEL))
    R.append(_rec("box_peel_open_k4_d1_L5", "_direct_sum_open_terminal",
                  K4, 2.5, A1, {"L": 5, "root": 0, "terminal": 3},
                  EXACT, "constructed", "peel with a terminal axis held open",
                  module_globals=FORCE_BOX_PEEL))

    # ---------------- box, open terminal (bit-identical gate) -----------
    R.append(_rec("box_open_k4_d1_L5", "_direct_sum_open_terminal",
                  K4, 2.5, A1, {"L": 5, "root": 0, "terminal": 3},
                  EXACT, "constructed",
                  "terminal kept open: the single-shot BZ residual"))
    R.append(_rec("box_open_k4_d2_tri_L3", "_direct_sum_open_terminal",
                  K4, 3.0, A2_TRI, {"L": 3, "root": 0, "terminal": 3},
                  EXACT, "constructed", ""))
    R.append(_rec("box_open_k4dbl_d1_L6", "_direct_sum_open_terminal",
                  K4, [2.5, 2.5, 2.5, 5.0, 2.5, 2.5], A1,
                  {"L": 6, "root": 0, "terminal": 3}, EXACT, "constructed",
                  "non-uniform nu: two distinct generators in one call"))

    # ---------------- box, extrapolated (1e-12 gate) --------------------
    R.append(_rec("box_ext_k4_d1", "direct_sum_extrapolated",
                  K4, 2.5, A1,
                  {"L_list": [4, 5, 6, 7, 8], "n_correction_terms": 3},
                  REL_1E12, "captured",
                  "the shipped d=1 ladder, verbatim"))
    R.append(_rec("box_ext_k4_d2_square", "direct_sum_extrapolated",
                  K4, 3.0, A2_SQUARE,
                  {"L_list": [3, 4, 5, 6], "n_correction_terms": 3},
                  REL_1E12, "captured", "the shipped d=2 ladder"))
    R.append(_rec("box_ext_k4_d2_tri", "direct_sum_extrapolated",
                  K4, 3.0, A2_TRI,
                  {"L_list": [3, 4, 5, 6], "n_correction_terms": 3},
                  REL_1E12, "constructed", "shipped d=2 ladder, skew cell"))
    R.append(_rec("box_ext_k4_d3", "direct_sum_extrapolated",
                  K4, 4.5, A3_CUBIC,
                  {"L_list": [2, 3], "n_correction_terms": 1},
                  REL_1E12, "constructed",
                  "d=3 truncated below the shipped (2,3,4): L=4 costs "
                  "9^9 per bag and does not belong in a unit suite"))
    # Captured verbatim from the frontend: 1qp order 7, graph 136 — a K4
    # whose 1-2 edge is doubled, so the Hadamard merge leaves a
    # NON-UNIFORM nu vector.  Every other record here is uniform-nu, and
    # a per-edge nu is the only thing that distinguishes a shared
    # generator cache keyed on nu from one keyed on the edge.
    R.append(_rec("box_ext_grid_k4dbl_d1", "_direct_sum_extrapolated_grid",
                  K4, [2.5, 2.5, 2.5, 5.0, 2.5, 2.5], A1,
                  {"k_grid": [[i / 8.0] for i in range(8)],
                   "root": 0, "terminal": 3,
                   "L_list": [4, 5, 6, 7, 8], "n_correction_terms": 3},
                  REL_1E12, "captured",
                  "the shipped finite-k single-shot BZ route, verbatim: "
                  "one elimination per L, then one cosine transform"))

    # ---------------- torus (bit-identical gate) ------------------------
    R.append(_rec("tn_k4_d1_n16_z", "graph_zeta_general",
                  K4, 2.5, A1, {"n_points": 16, "source": 0}, EXACT,
                  "constructed", ""))
    R.append(_rec("tn_k4_d1_n16_k_term", "graph_zeta_general",
                  K4, 2.5, A1,
                  {"n_points": 16, "source": 0, "terminals": [3],
                   "space": "k"},
                  EXACT, "constructed", "one free terminal, k-space"))
    R.append(_rec("tn_k4_d2_tri_n8", "graph_zeta_general",
                  K4, 3.0, A2_TRI, {"n_points": 8, "source": 0}, EXACT,
                  "constructed",
                  "even n on a skew cell: where the balanced-z axis is "
                  "asymmetric and a non-even kernel would show"))
    R.append(_rec("tn_k4_d2_square_n4_two_terms", "graph_zeta_general",
                  K4, 3.0, A2_SQUARE,
                  {"n_points": 4, "source": 0, "terminals": [2, 3],
                   "space": "k"},
                  EXACT, "constructed",
                  ">=2 externals is the route that bypasses the hybrid; "
                  "n=4 keeps the frozen (n^d)^2 grid small"))
    R.append(_rec("tn_prism_d1_n16", "graph_zeta_general",
                  PRISM, 2.5, A1, {"n_points": 16, "source": 0}, EXACT,
                  "constructed", ""))
    R.append(_rec("tn_k4_d3_n6", "graph_zeta_general",
                  K4, 4.5, A3_CUBIC, {"n_points": 6, "source": 0}, EXACT,
                  "constructed", "d=3 on the torus"))
    R.append(_rec("tn_k4_d3_oblique_n6", "graph_zeta_general",
                  K4, 4.5, A3_OBLIQUE, {"n_points": 6, "source": 0}, EXACT,
                  "constructed",
                  "d=3 oblique at even n: kernel non-evenness 4.4e-03, so "
                  "this is where a d=3 orientation slip would show; it is "
                  "exactly 0 at odd n and on every diagonal A"))
    R.append(_rec("tn_at_zero_k4_d1_n16", "graph_zeta_general_at_zero",
                  K4, 2.5, A1, {"n_points": 16, "pinned_vertex": 0}, EXACT,
                  "constructed", ""))
    # nu = inf counts nearest-neighbour homomorphisms, so the value is a
    # positive integer and any FFT round trip would return 5.999... .
    # The 4-cycle is used rather than a graph containing a triangle: no
    # triangle embeds in Z^1 with all three distances 1, so that value
    # would be an uninformative 0.
    R.append(_rec("tn_c4_d1_n32_inf", "graph_zeta_general",
                  C4, float("inf"), A1, {"n_points": 32, "source": 0},
                  EXACT, "constructed",
                  "nu=inf: exactly 6, and the peel must not fire"))
    R.append(_rec("tn_c4_d2_n8_inf", "graph_zeta_general",
                  C4, float("inf"), A2_SQUARE, {"n_points": 8, "source": 0},
                  EXACT, "constructed", "nu=inf at d=2: exactly 36"))

    # ---------------- hybrid (norm-wise 4e-15 gate) ---------------------
    R.append(_rec("hy_tw2_d1_n32", "hybrid_zeta",
                  TW2_BLOCK, 4.0, A1, {"n_points": 32}, REL_HYBRID,
                  "captured", "captured tw=2 vacuum block"))
    R.append(_rec("hy_tw2wide_d1_n32", "hybrid_zeta",
                  TW2_WIDE, 4.0, A1, {"n_points": 32}, REL_HYBRID,
                  "captured", ""))
    R.append(_rec("hy_tw2_d2_tri_n8", "hybrid_zeta",
                  TW2_BLOCK, 4.0, A2_TRI, {"n_points": 8}, REL_HYBRID,
                  "constructed",
                  "even n on the triangular cell: the orientation "
                  "defect lived exactly here"))
    R.append(_rec("hy_diamond_d1_n32_1qp", "hybrid_zeta",
                  DIAMOND, 4.0, A1,
                  {"n_points": 32, "source": 0, "terminal": 3}, REL_HYBRID,
                  "constructed", "full BZ grid, 1qp"))
    R.append(_rec("hy_tw2_d2_tri_n8_1qp", "hybrid_zeta",
                  TW2_BLOCK, 4.0, A2_TRI,
                  {"n_points": 8, "source": 0, "terminal": 1}, REL_HYBRID,
                  "constructed", "1qp grid on a skew cell at even n"))

    return R


# ---------------------------------------------------------------------------
# Production capture — evidence that the freeze set covers what ships
# ---------------------------------------------------------------------------

def _capture_production(limit=200):
    """Monkeypatch the frontend's executor names over a corpus pass.

    Returns a ``Counter`` of ``(executor, n_vertices, d)`` observed.  The
    point is not to *generate* records — the curated set above is stable
    and reviewable — but to check that no executor the pipeline reaches
    is missing from it.
    """
    from gzl.series import _load_corpus

    observed = Counter()
    failures: list[str] = []
    orig = {}
    names = ("direct_sum_extrapolated", "_direct_sum_extrapolated_grid",
             "graph_zeta_general", "graph_zeta_general_at_zero",
             "hybrid_zeta")

    def wrap(nm, fn):
        def inner(*a, **kw):
            e = np.asarray(a[0])
            observed[(nm, int(e.max()) + 1, int(np.asarray(a[2]).shape[0]))] += 1
            return fn(*a, **kw)
        return inner

    for nm in names:
        orig[nm] = getattr(fe, nm)
        setattr(fe, nm, wrap(nm, orig[nm]))
    try:
        # Both engines are swept: ``engine="hybrid"`` is the default the
        # pipeline runs and ``engine="tensor"`` is the documented
        # opt-out, so a sweep of the default alone would report
        # graph_zeta_general as unreachable when it is merely unselected.
        #
        # BOTH DENSE ROUTES are swept for the same reason, and it became
        # load-bearing once dense blocks moved to the torus.  The default
        # ``dense_engine`` is now "torus" at d <= 3, so a default-only
        # sweep would observe neither ``direct_sum_extrapolated`` nor
        # ``_direct_sum_extrapolated_grid`` and would re-freeze them as
        # "unreached" — retiring the coverage guard on two engines that
        # are still very much live (d >= 4, gate-declined blocks the slab
        # cannot take, and the MemoryError fallback).
        # ``dense_engine=None`` means "whatever ships".
        for corpus_name, order, A, npts, nu, engine, dense in (
            ("tfim_softcore_corpus_0qp.npz", 8, np.eye(1), 32, 4.0,
             "hybrid", None),
            ("tfim_softcore_corpus_0qp.npz", 8, np.eye(1), 32, 4.0,
             "tensor", None),
            ("tfim_softcore_corpus_0qp.npz", 8, np.array(A2_TRI), 16, 4.0,
             "hybrid", None),
            ("tfim_softcore_corpus_0qp.npz", 8, np.array(A2_TRI), 16, 4.0,
             "tensor", None),
            ("tfim_softcore_corpus_1qp.npz", 6, np.eye(1), 32, 4.0,
             "hybrid", None),
            ("tfim_softcore_corpus_1qp.npz", 6, np.eye(1), 32, 4.0,
             "tensor", None),
            ("tfim_softcore_corpus_0qp.npz", 8, np.eye(1), 32, 2.5,
             "hybrid", None),
            ("tfim_softcore_corpus_1qp.npz", 6, np.eye(1), 32, 2.5,
             "tensor", None),
            # Order 7 carries the first 1qp graph with a treewidth-3
            # spine block, which is the only way to reach the finite-k
            # single-shot BZ route.  Without this row the capture
            # reports _direct_sum_extrapolated_grid as unreachable when
            # it is merely absent from the lower orders.
            ("tfim_softcore_corpus_1qp.npz", 7, np.eye(1), 8, 2.5,
             "hybrid", None),
            # The box arm, forced.  Mirrors the two rows above that
            # reach the dense route, so the grid and the scalar
            # direct-sum entry points are both observed.
            ("tfim_softcore_corpus_0qp.npz", 8, np.eye(1), 32, 2.5,
             "hybrid", "direct_sum"),
            ("tfim_softcore_corpus_1qp.npz", 7, np.eye(1), 8, 2.5,
             "hybrid", "direct_sum"),
        ):
            # data_path raises on a missing corpus: a skipped corpus used to
            # freeze fewer executors without a word.
            rec = _load_corpus(data_path(corpus_name))[order]
            hop = rec.get("hopping")
            for i in range(min(limit, len(rec["edges_list"]))):
                e = np.repeat(rec["edges_list"][i],
                              rec["multiplicities_list"][i], axis=0)
                s, t = (int(hop[i][0]), int(hop[i][1])) if hop is not None \
                    else (0, 0)
                try:
                    fe.evaluate_graph(e, nu, A, source=s,
                                      terminal=(t if t != s else None),
                                      n_points=npts, engine=engine,
                                      dense_engine=dense)
                except Exception as exc:            # noqa: BLE001
                    failures.append(f"{corpus_name} O{order} #{i}: "
                                    f"{type(exc).__name__}: {exc}")
    finally:
        for nm in names:
            setattr(fe, nm, orig[nm])
    # A capture that saw NOTHING is a broken harness, not a frontend
    # that routes nothing -- and the difference is invisible downstream,
    # where it reads as "every executor is unreached".  The swallow above
    # is deliberate (individual graphs legitimately raise: nu <= d,
    # self-loops, disconnected input), but swallowing ALL of them is not
    # a result.
    if not observed:
        head = "\n  ".join(failures[:5]) or "(no calls attempted)"
        raise SystemExit(
            f"capture observed no executor at all over "
            f"{len(failures)} attempted graphs.  First failures:\n  {head}"
        )
    return observed


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture-only", action="store_true",
                    help="report what the frontend routes; write nothing")
    args = ap.parse_args()

    if args.capture_only:
        obs = _capture_production()
        for (nm, nv, d), n in sorted(obs.items()):
            print(f"  {nm:32s} V={nv} d={d}  x{n}")
        print("executors seen:", sorted({nm for nm, _, _ in obs}))
        return

    observed = _capture_production()
    seen = sorted({nm for nm, _, _ in observed})
    print("frontend-observed executors:", seen)

    records = build_records()
    for rec in records:
        rec["executor_reach"] = _classify(rec["executor"], seen)
    frozen = []
    t_all = time.time()
    for rec in records:
        t0 = time.time()
        value = run_record(rec)
        dt = time.time() - t0
        out = dict(rec)
        out["value"] = _encode(value)
        out["seconds"] = round(dt, 3)
        frozen.append(out)
        print(f"  {rec['label']:34s} {dt:7.3f}s  "
              f"shape={tuple(np.asarray(value).shape)}")
    print(f"total {time.time() - t_all:.1f}s over {len(frozen)} records")

    payload = {
        "schema": 1,
        "generated_by": "tests/fixtures/_freeze_executor_goldens.py",
        "numpy": np.__version__,
        "fingerprint": fingerprint(),
        "observed_executors": seen,
        "records": frozen,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
