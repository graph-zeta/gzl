# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
r"""Which float64 gate this environment is entitled to run.

Bit-identity is a property of the ENVIRONMENT, not of this library.
``tests/test_executor_goldens.py`` established that for its own record
set; this module is the same instrument for the repo's other frozen
float64 pins — ``tests/data/engine_reference_values.json`` and the
handful of ``float.hex()`` literals asserted inline.

WHY, MEASURED (macOS 26.5.2 / arm64 M1 Max / py3.12 / numpy 2.2.6 built
against Accelerate, against a file frozen on Linux -- see
``engine_reference_values.json``'s ``_meta.frozen_on``):

  * 88 of the default suite's assertions fail, and 4 more under
    ``--runslow`` (3 slow-gated ``router|*|d3`` keys and the slow inline
    pin in ``test_dense_torus_accuracy_gate.py``) -- 92 in all.  Every
    one is 1-5 ulp on the frozen-key set, max rel 7.7e-16; the largest
    anywhere is 8 ulp, on the ``k != 0`` cell of the ``test_slab.py``
    pin, which pytest never reported because the ``k = 0`` assertion
    above it fails first.
  * A tolerance-blind recompute of all 414 frozen keys moves 164 of
    them, max 5 ulp.  EVERY cell type moves, including the three
    ``_METRIC_CELLS`` already tolerated at 24 ulp (25/48 triangular,
    27/48 tetragonal, 19/96 sheared).  The cell is therefore NOT the
    discriminating axis it was taken for.
  * The mechanism is the FFT, not the A-dependent kernel path and not
    BLAS.  For all four cells that fail here (``unit``, ``scaled``,
    ``square``, ``cubic``) ``_label_distances`` and ``pow`` are
    correctly rounded on every entry -- the kernel array is
    bit-canonical, so it cannot be the source.  Setting
    ``tensor_network._USE_CONV = False`` restores bit-exactness on
    28 of the 86 engine-reference failures outright; the remaining 58
    run through the ``np.fft.fftn`` calls in ``hybrid``/``slab`` that
    the kill switch does not gate.  Swapping numpy's pocketfft for
    scipy's ON THIS MACHINE, with no other change, moves 161 of 408
    keys by 1-5 ulp -- the identical envelope -- and lands 12 of the
    failing keys EXACTLY on the frozen bits.  Values are invariant to
    BLAS thread count (1 vs 8), so this is a fixed per-build code path,
    not a threading artifact.
  * It is not a defect.  Evaluated against 50-digit exact arithmetic of
    the SAME truncated torus sum, over all 54 failing d = 1 keys, this
    machine is closer to exact on 33 and the frozen file on 21 (mean
    0.96 vs 1.17 ulp from exact, max 2.69 vs 2.65).  Neither
    environment is the right answer; both are roundings of a sum whose
    float64 evaluation is FFT-implementation dependent.
  * ``epsteinlib`` is ruled out: none of ``tensor_network``, ``hybrid``,
    ``slab`` or ``direct_sum`` imports it, and every epsteinlib-backed
    test in the suite passes here.

THE BOUND is not chosen, it is the one
``tests/test_executor_goldens.py`` already calibrated against this same
mechanism, reused so the repo carries ONE number: 64 ulp, which that
file measured at 8x above the cross-platform noise (1-8 ulp) and 13x
below the smallest real defect it could produce (``_USE_CONV`` off,
848 ulp).  The 5 ulp measured here, and the 11 ulp the metric-cell CI
runs measured, both sit inside that noise band.

WHAT IT GIVES UP, stated rather than hidden: a reassociation smaller
than 64 ulp is invisible to the portable gate.  It is still caught in
the freezing environment, which is where refactors happen and where
zero tolerance still stands -- ``tests/test_engine_reference.py``
announces on every run which of the two gates it ran.
"""
from __future__ import annotations

import math
import platform
import sys

import numpy as np

__all__ = ["CROSS_ENV_ULP", "FROZEN_ENV", "SAME_ENV", "fingerprint",
           "matches_freeze", "ulp_distance", "assert_pinned", "gate_name"]

#: Calibrated in ``tests/test_executor_goldens.py`` -- see module docstring.
CROSS_ENV_ULP = 64.0

#: The environment the repo's float64 pins were frozen in.  Only the OS
#: is recoverable: the freeze predates fingerprinting and recorded only
#: the path of a Linux checkout, so ``engine_reference_values.json``'s
#: ``_meta.frozen_on`` holds ``{"system": "Linux"}``.  Fields absent here
#: are WILDCARDS, so this deliberately keeps the pre-existing behaviour
#: on Linux -- bit-exact there, where every one of these values has
#: reproduced -- rather than silently relaxing the gate everywhere.  Add
#: fields (``machine``, ``python``, ``numpy``) when a freeze records
#: them; ``fingerprint()`` is the shape to record.
FROZEN_ENV = {"system": "Linux"}


def fingerprint() -> dict:
    """Identify this environment's float64 arithmetic.

    Deliberately the same four fields as
    ``tests/fixtures/_freeze_executor_goldens.fingerprint`` --
    ``tests/test_env_gate.py`` asserts the two do not drift.
    """
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": ".".join(str(v) for v in sys.version_info[:2]),
        "numpy": np.__version__,
    }


def matches_freeze(recorded: dict | None = None) -> bool:
    """True when this environment is the one ``recorded`` describes.

    Fields absent from ``recorded`` are not compared -- an incompletely
    recorded freeze narrows the bit-exact gate as far as it can and no
    further.  An empty/absent record matches NOTHING: an unidentified
    freeze cannot claim any environment as its own.
    """
    recorded = FROZEN_ENV if recorded is None else recorded
    if not recorded:
        return False
    here = fingerprint()
    return all(here.get(k) == v for k, v in recorded.items())


#: Does this environment run the bit-exact gate, or the calibrated one?
SAME_ENV = matches_freeze()


def gate_name() -> str:
    """One line naming the active gate, for tests that announce it."""
    return ("BIT-IDENTICAL (freezing environment)" if SAME_ENV else
            f"{CROSS_ENV_ULP:.0f}-ulp fallback (different environment)")


def ulp_distance(got: float, want: float) -> float:
    """Distance in units in the last place of ``want``."""
    return abs(float(got) - float(want)) / max(math.ulp(float(want)), 5e-324)


def assert_pinned(got, frozen_hex: str, what: str) -> None:
    """Assert a value still equals its frozen ``float.hex()`` pin.

    Bitwise in the freezing environment; within :data:`CROSS_ENV_ULP`
    anywhere else.  Use for every inline ``float.hex()`` literal, so a
    foreign toolchain reports the one thing that distinguishes noise
    from a value change -- the ulp distance -- instead of two opaque
    hex strings.
    """
    got = float(got)
    want = float.fromhex(frozen_hex)
    if SAME_ENV:
        assert got.hex() == frozen_hex, (
            f"{what}: {got!r} != frozen {want!r} "
            f"({ulp_distance(got, want):.0f} ulp, "
            f"{abs(got - want) / max(abs(want), 1e-300):.3e} rel) -- this IS "
            f"the freezing environment, so this is a value change")
        return
    ulp = ulp_distance(got, want)
    assert ulp <= CROSS_ENV_ULP, (
        f"{what}: {got!r} vs frozen {want!r} = {ulp:.0f} ulp -- beyond the "
        f"{CROSS_ENV_ULP:.0f}-ulp cross-environment bound; this is a value "
        f"change, not toolchain variation")
