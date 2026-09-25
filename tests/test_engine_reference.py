# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
r"""The frozen-reference harness: every engine, bit-exact to frozen values.

This is the value-preservation gate for engine unification.  The
reference set was generated ONCE on the pre-fold tree (before
``slab_zeta`` was folded into ``hybrid._dense_core``) by
``tests/data/_generate_engine_reference.py`` and committed; each test
here replays one call path and compares ``float.hex()`` strings --
the exact bit pattern, no tolerance -- **in the environment the file was
frozen in**.  Anywhere else the comparison falls back to the calibrated
cross-environment bound in :mod:`tests._env_gate`, because bit-identity
is a property of the toolchain and not of this library;
``test_which_gate_this_environment_runs`` says out loud which of the two
ran, so a green suite is never misread as certifying bit-identity.

WHAT A FAILURE MEANS.  A red entry says a refactor moved a shipped
number.  There are exactly two legitimate responses, and "regenerate the
JSON" is neither unless it is its own commit citing the change that
moved each value:

  1. the refactor is wrong -- fix it until the bit pattern returns;
  2. the change is a DELIBERATE, documented reassociation -- then the
     entry moves to the ``ROUNDOFF_OK`` set below in the same commit,
     with the reason, and is compared at 8 ULP instead.

The paths and what they pin:

  hybrid|...       hybrid_zeta: SP-reduce + dense core, source at 0.
  slab|...         slab_zeta with the orbit reduction.
  slab_nosym|...   slab_zeta without it -- so a reduction bug and a
                   contraction bug are distinguishable.
  tensor|...       graph_zeta_general_at_zero: the raw-graph path.
  box|...          direct_sum_extrapolated at L=(2,3,4), d <= 2.
  router_vac|...   evaluate_graph, vacuum, the full routing stack.
  router_k0|...    evaluate_graph at momentum=0 with a terminal -- the
                   finite-k dispatch at k = 0, a DIFFERENT code path.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gzl import (
    direct_sum_extrapolated,
    evaluate_graph,
    graph_zeta_general_at_zero,
    hybrid_zeta,
    slab_zeta,
)

from tests._env_gate import (
    CROSS_ENV_ULP,
    FROZEN_ENV,
    SAME_ENV,
    fingerprint,
    gate_name,
    ulp_distance,
)

_DATA = Path(__file__).parent / "data" / "engine_reference_values.json"
_REF = {k: v for k, v in json.loads(_DATA.read_text()).items()
        if not k.startswith("_")}

#: Entries a documented reassociation has moved off bit-identity.
#: Empty by construction on the pre-fold tree.  Every addition needs the
#: commit that moved it named in the comment beside it.
ROUNDOFF_OK: dict = {}
_ULP_TOL = 8

#: MACHINE-TOLERANCE CLASSES, distinct from ROUNDOFF_OK in kind: these
#: entries are bit-exact on a fixed machine+build but not ACROSS
#: machines, so elsewhere they sit within a few ULP of the frozen value
#: with no code change at all.  Two mechanisms, discovered separately by
#: this harness's own CI runs, each held at its measured ceiling:
#:
#: 1. ``box|*`` (all cells are ``np.eye``): the box's Richardson ladder
#:    runs through LAPACK ``lstsq`` and BLAS-backed Toeplitz
#:    contractions, whose kernels differ between OpenBLAS builds.
#:    Found by the harness's first CI run: exactly the five box entries
#:    moved, 1-3 ULP, on one Python's wheel.  Ceiling 8.
#:
#: 2. NON-IDENTITY METRIC CELLS (``triangular``, ``tetragonal``,
#:    ``sheared``) across all four torus families.  Found the day the
#:    harness merged: a push run on main and the next pull request's
#:    run failed 73 keys -- hybrid 18, slab 18, slab_nosym 19,
#:    tensor 18, every one on a non-identity cell, 1-11 ULP --
#:    while every ``unit``/``scaled``/``square``/``cubic`` key and
#:    every router key (the router set is all-``np.eye``) stayed
#:    bit-exact.  Three facts pin the cause on RUNNER-CPU HETEROGENEITY
#:    in the metric-dependent kernel path, not on a wheel change and not
#:    on either commit: (a) the two runs are one tree apart in code the
#:    keys never execute, yet py3.10 was green at 11:15Z and red at
#:    12:21Z with the same wheel, while py3.11 flipped the other way;
#:    (b) the four red jobs agree with each OTHER bit-identically on
#:    all 73 keys and disagree only with the frozen file -- the engines
#:    are deterministic per machine, and what varies per CPU is the
#:    A-dependent kernel table they all consume, so the families move
#:    coherently; (c) a single container reproduces all 414 frozen values
#:    bit-exactly, before and after the fold.  The 73 are a sample from
#:    the 192-key metric-cell class (48 per family), held together by
#:    the mechanism rather than the sample, so the whole class carries
#:    the tolerance: measured max 11 ULP, ceiling 24 (~2x headroom for
#:    runner classes not yet seen).  Zero tolerance everywhere else
#:    stands -- identity-metric torus values have reproduced bit-exactly
#:    on every machine this harness has touched.
#: 3. A FOREIGN TOOLCHAIN moves keys of EVERY cell type, so neither
#:    class above can absorb it and the cell is not the discriminating
#:    axis it was taken for.  Measured on macOS 26.5.2 / arm64 (M1 Max) /
#:    py3.12 / numpy 2.2.6-Accelerate, against this Linux-frozen file:
#:    88 default-suite assertions fail, 92 under ``--runslow``, all
#:    1-5 ulp; a tolerance-blind recompute of all 414 keys moves 164 of
#:    them, max 5 ulp, INCLUDING the metric cells item 2 already
#:    tolerates (25/48 triangular, 27/48 tetragonal, 19/96 sheared).
#:    The mechanism is the FFT, not the A-dependent kernel path: for
#:    every cell that fails there, ``_label_distances`` and ``pow`` are
#:    correctly rounded on every entry, so the kernel array is
#:    bit-canonical and cannot be the source; ``_USE_CONV = False``
#:    restores bit-exactness on 28 of the 86 outright and the rest run
#:    through the ungated ``np.fft.fftn`` in hybrid/slab; and swapping
#:    numpy's pocketfft for scipy's on that one machine reproduces the
#:    whole phenomenon (161/408 keys, 1-5 ulp, 12 of them landing back
#:    on the frozen bits exactly).  It is not a defect either way:
#:    against 50-digit exact arithmetic of the same truncated sum, over
#:    all 54 failing d = 1 keys, that machine is CLOSER to exact on 33
#:    and this file on 21 (mean 0.96 vs 1.17 ulp).  Full narrative and
#:    the reused 64-ulp calibration: :mod:`tests._env_gate`.
#:
#: So the two cell-keyed classes stay exactly as measured for the
#: environment they were measured in, and the toolchain gate wraps
#: them: off the freezing environment EVERY key -- these two classes
#: included -- is compared at ``CROSS_ENV_ULP`` instead.
_MACHINE_TOL_PREFIXES = ("box|",)
_METRIC_CELLS = ("triangular", "tetragonal", "sheared")
_METRIC_CELL_ULP_TOL = 24

#: 4. A SINGLE KEY, measured flaky ON the freezing environment itself --
#:    the one class the three above cannot express, because it is not
#:    keyed on cell, prefix or toolchain but on one value sitting astride
#:    a rounding boundary.
#:
#:    ``router_mix|K4|d2|n12`` reports 7.362463224829686
#:    (``0x1.d73298f4295b5p+2``) against the frozen 7.362463224829685
#:    (``0x1.d73298f4295b3p+2``) -- 2 ULP -- on roughly a third of
#:    GitHub-hosted py3.12 runs.  Measured on a change outside the
#:    library and its tests: 4 fails and 8 passes across 12 runs
#:    of trees whose ``gzl/`` and ``tests/`` are byte-identical,
#:    including FAIL, PASS, PASS, PASS on one unchanged commit.  Never
#:    seen failing on 3.10/3.11.  macOS computes a third value,
#:    ``...b9p+2``, deterministically, and invariantly under allocation
#:    state, scipy import and heap fragmentation -- so it is not layout.
#:
#:    The mechanism is the one items 1-3 already name.  This key's path
#:    calls ``np.linalg.lstsq`` twice (the Richardson ladder) and
#:    ``np.fft.rfftn`` six times, and ``lstsq`` is the stated reason the
#:    ``box|`` class carries ``_ULP_TOL``.  The deviation is 2 ULP
#:    against that same 8 ULP ceiling.
#:
#:    It is exempted ALONE, not as a prefix.  The other 15 ``router_mix``
#:    keys and every other ladder-bearing ``router_*`` key run the same
#:    ``lstsq`` and have never moved, so widening the class would spend
#:    the bitwise gate on ~64 entries to buy nothing; what singles this
#:    one out is the boundary, which is a property of the value and not
#:    of the route.  If a second key joins it, add it here with its own
#:    fail/pass record -- do not promote the pair to a prefix without the
#:    measurement that shows the route, rather than the value, is what
#:    varies.
_FLAKY_ON_FREEZING_ENV = {
    "router_mix|K4|d2|n12": _ULP_TOL,
}


def _machine_tol(key: str):
    """ULP ceiling for cross-machine variation, or ``None`` for bit-exact."""
    if not SAME_ENV:
        return CROSS_ENV_ULP
    if key in _FLAKY_ON_FREEZING_ENV:
        return _FLAKY_ON_FREEZING_ENV[key]
    if key.startswith(_MACHINE_TOL_PREFIXES):
        return _ULP_TOL
    parts = key.split("|")
    if len(parts) == 5 and parts[3] in _METRIC_CELLS:
        return _METRIC_CELL_ULP_TOL
    return None

CORES = {
    "K4":    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    "prism": [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
              (0, 3), (1, 4), (2, 5)],
    "K33":   [(i, j) for i in (0, 1, 2) for j in (3, 4, 5)],
    "K5":    [(i, j) for i in range(5) for j in range(i + 1, 5)],
    "K5-e":  [e for e in [(i, j) for i in range(5)
                          for j in range(i + 1, 5)] if e != (0, 4)],
    "V6E11": [(0, 1), (0, 4), (1, 2), (1, 3), (1, 5), (2, 3),
              (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)],
}
CELLS = {
    ("1", "unit"): [[1.0]],
    ("1", "scaled"): [[1.7]],
    ("2", "square"): [[1.0, 0.0], [0.0, 1.0]],
    ("2", "triangular"): [[1.0, 0.5], [0.0, 0.8660254037844386]],
    ("2", "sheared"): [[1.0, 0.37], [0.0, 1.13]],
    ("3", "cubic"): [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ("3", "tetragonal"): [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.3]],
    ("3", "sheared"): [[1.0, 0.3, 0.0], [0.0, 1.1, 0.2], [0.0, 0.0, 0.9]],
}


def _recompute(key: str) -> float:
    parts = key.split("|")
    engine = parts[0]
    if engine in ("hybrid", "slab", "slab_nosym", "tensor"):
        core, dtag, cell, ntag = parts[1:]
        d, n = int(dtag[1:]), int(ntag[1:])
        E = CORES[core]
        A = np.array(CELLS[(dtag[1:], cell)], dtype=float)
        nu = np.full(len(E), d + 0.5)
        if engine == "hybrid":
            return float(np.real(hybrid_zeta(np.array(E), nu, A, n)))
        if engine == "slab":
            return float(slab_zeta(E, nu, A, n)[0])
        if engine == "slab_nosym":
            return float(slab_zeta(E, nu, A, n, use_symmetry=False)[0])
        return float(np.real(graph_zeta_general_at_zero(
            np.array(E), nu, A, n)))
    if engine == "box":
        core, dtag, _ = parts[1:]
        d = int(dtag[1:])
        E = CORES[core]
        return float(np.real(direct_sum_extrapolated(
            np.array(E), np.full(len(E), d + 0.5), np.eye(d),
            L_list=(2, 3, 4), n_correction_terms=2)))
    if engine in ("router_nn", "router_mix"):
        # The nu = inf exponent family, which had NO frozen key at all
        # until a call-wide nearest-neighbour flag was found gating an
        # exactness shortcut: one inf edge anywhere pinned every block of
        # the call to a tiny torus and dropped n_points from its value,
        # silently and identically at every n.  ``router_nn`` is the
        # uniform case (the shortcut is legitimate there, and the answer
        # is an integer count of nearest-neighbour homomorphisms);
        # ``router_mix`` puts ONE inf edge beside finite exponents, which
        # is the class that broke and the one the 414 original keys, all
        # at nu = d + 0.5, could never have caught.
        core, dtag, ntag = parts[1:]
        d, n = int(dtag[1:]), int(ntag[1:])
        E = np.array(CORES[core], dtype=int)
        if engine == "router_nn":
            nu = np.inf
        else:
            nu = np.full(len(E), d + 0.5)
            nu[0] = np.inf
        return float(evaluate_graph(E, nu, np.eye(d), n_points=n,
                                    richardson=True))
    if engine in ("router_vac", "router_k0"):
        core, dtag, ntag = parts[1:]
        d, n = int(dtag[1:]), int(ntag[1:])
        E = np.array(CORES[core], dtype=int)
        kw = dict(n_points=n, richardson=True)
        if engine == "router_k0":
            kw.update(source=0, terminal=1, momentum=np.zeros(d))
        return float(evaluate_graph(E, d + 0.5, np.eye(d), **kw))
    raise KeyError(key)


_FAST = [k for k in sorted(_REF)
         if not (k.startswith("router") and "|d3|" in k)]
_SLOW = [k for k in sorted(_REF)
         if k.startswith("router") and "|d3|" in k]


def _machine_variable_calls(key):
    """Count the calls on ``key``'s path that this module already names as
    machine-variable: ``np.linalg.lstsq`` (items 1/4) and ``np.fft.*``
    (item 3).  Used ONLY to explain a failure, never to decide one.
    """
    counts = {"lstsq": 0, "fft": 0}
    orig_lstsq = np.linalg.lstsq
    fft_names = [n for n in ("fft", "ifft", "fftn", "ifftn",
                             "rfft", "irfft", "rfftn", "irfftn")
                 if hasattr(np.fft, n)]
    orig_fft = {n: getattr(np.fft, n) for n in fft_names}

    def _wrap(fn, slot):
        def g(*a, **k):
            counts[slot] += 1
            return fn(*a, **k)
        return g

    np.linalg.lstsq = _wrap(orig_lstsq, "lstsq")
    for n, f in orig_fft.items():
        setattr(np.fft, n, _wrap(f, "fft"))
    try:
        _recompute(key)
    except Exception:                      # pragma: no cover - diagnosis only
        pass
    finally:
        np.linalg.lstsq = orig_lstsq
        for n, f in orig_fft.items():
            setattr(np.fft, n, f)
    return counts


def _explain_small_deviation(key, ulp):
    """Say whether a failing key sits in the population that is KNOWN to
    vary by machine, so a reader does not spend hours re-deriving it.

    A red entry normally means a refactor moved a shipped number, and
    that is still the first thing to check.  But in one case a 2 ULP
    move on a key whose path runs ``lstsq`` cost most of a day and a
    false 'measured' claim committed to the tree before reruns refuted
    it: the diff touched neither ``gzl/`` nor ``tests/``, and the key
    passed 8 of 12 runs of byte-identical engine code.  This message
    exists so the next reader gets that in one line instead of
    re-deriving it.
    """
    if ulp > _ULP_TOL:
        return ""
    calls = _machine_variable_calls(key)
    if not (calls["lstsq"] or calls["fft"]):
        return ""
    return (
        f"\n\n  NOTE: {ulp:.0f} ULP is inside the {_ULP_TOL} ULP envelope this "
        f"module already tolerates for machine variation, and this key's "
        f"path calls lstsq {calls['lstsq']}x and np.fft {calls['fft']}x -- "
        f"the two mechanisms documented above (items 1, 3 and 4).\n"
        f"  Before concluding that a value MOVED: (a) check whether the "
        f"diff touches gzl/ at all, and (b) RERUN -- "
        f"router_mix|K4|d2|n12 failed 4 of 12 runs on unchanged engine "
        f"code, including FAIL/PASS/PASS/PASS on one commit.  Two CI runs "
        f"are not a measurement.\n"
        f"  If it reproduces across reruns AND the engine changed, it is "
        f"a real move: fix the refactor, or move the entry deliberately "
        f"in its own commit naming old/new hex (see AGENTS.md)."
    )


def _check(key):
    got = _recompute(key)
    want = float.fromhex(_REF[key])
    tol = _machine_tol(key)
    if tol is not None:
        ulp = ulp_distance(got, want)
        assert ulp <= tol, (
            f"{key}: {got!r} vs {want!r} = {ulp:.0f} ULP -- beyond the "
            f"cross-machine tolerance ({tol} ULP) for this key's class; "
            f"this is a value change, not machine variation")
        return
    if key in ROUNDOFF_OK:
        ulp = ulp_distance(got, want)
        assert ulp <= _ULP_TOL, (
            f"{key}: {got!r} vs {want!r} = {ulp:.0f} ULP "
            f"(documented reassociation allows {_ULP_TOL})")
    else:
        if got.hex() != _REF[key]:
            rel = abs(got - want) / max(abs(want), 1e-300)
            raise AssertionError(
                f"{key}: {got!r} != frozen {want!r} ({rel:.3e} rel)"
                + _explain_small_deviation(key, ulp_distance(got, want)))


def test_the_flaky_exemption_stays_narrow():
    """One key, at the machine-variation ceiling -- no more.

    The exemption is deliberately a SET, not a prefix: every other
    ladder-bearing ``router_*`` key runs the same ``lstsq`` and has never
    moved, so widening it would spend the bitwise gate on ~64 entries to
    buy nothing.  This test makes any widening a visible, reviewed edit
    rather than a quiet one.
    """
    assert _FLAKY_ON_FREEZING_ENV == {"router_mix|K4|d2|n12": _ULP_TOL}, (
        "the flaky-key exemption changed.  Adding a key needs its own "
        "fail/pass record in the comment above it; promoting the set to "
        "a prefix needs a measurement showing the ROUTE varies, not one "
        "value sitting astride a rounding boundary."
    )
    assert all(k in _REF for k in _FLAKY_ON_FREEZING_ENV), (
        "an exempted key is not in the frozen file -- stale entry"
    )


def test_a_small_deviation_explains_its_own_risk_class():
    """A 2 ULP red entry must say what machinery is on its path.

    This is the anti-recurrence half of the fix: the key itself is
    exempted above, but the NEXT one will not be, and the cost last time
    was not the failure -- it was reading a 2 ULP flake as 'your refactor
    moved a shipped number'.
    """
    cheap = "router_mix|K4|d1|n8"
    assert cheap in _REF
    msg = _explain_small_deviation(cheap, 2)
    assert "RERUN" in msg and "lstsq" in msg and "gzl/" in msg, msg
    # A deviation past the machine envelope is NOT excused.
    assert _explain_small_deviation(cheap, _ULP_TOL + 1) == ""


@pytest.mark.parametrize("key", _FAST)
def test_bit_exact(key):
    _check(key)


@pytest.mark.slow
@pytest.mark.parametrize("key", _SLOW)
def test_bit_exact_slow(key):
    _check(key)


def test_which_gate_this_environment_runs(capsys):
    """Say out loud which gate ran, so a green run is never misread.

    In the freezing environment every key outside the two measured
    cell-keyed classes is compared bitwise.  Anywhere else all 433 fall
    back to the calibrated bound, and a reassociation smaller than it --
    the very thing this harness exists to surface -- is invisible.  A
    green run off the freezing environment therefore does NOT certify
    bit-identity, and this line is what stops someone concluding that
    it does.
    """
    with capsys.disabled():
        print(f"\n  frozen keys run under: {gate_name()}")
        print(f"    frozen on: {FROZEN_ENV}")
        print(f"    running on: {fingerprint()}")
        if not SAME_ENV:
            print(f"    -> a <={CROSS_ENV_ULP:.0f} ulp reassociation would "
                  f"NOT be caught here; run the suite on the freezing "
                  f"environment for that.")
    assert isinstance(SAME_ENV, bool)
