# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The ``selftest`` command.

The test suite ships in neither the wheel nor the sdist, so somebody who
installed from PyPI has no way to ask whether the install works.  This
runs the checks that do not need the suite: the shipped data against its
manifest, and values whose exact answers are known independently of the
code that produces them.

The checks are those of ``.github/scripts/wheel_smoke.py``, which runs
the same ground in CI against a freshly built wheel.  That script keeps
its own clean-room assertions -- that ``gzl`` was imported from
site-packages rather than from a source tree -- which are a property of
how CI invokes it, not of an installation.

Nothing imports this module; see :mod:`gzl.cli` for why.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Callable, Iterable

__all__ = ["build_parser", "run"]

#: Order 3 of the chain 1qp series against Langheld et al. (2022), in
#: units of the file's own error bar.  Five sigma is a broken install,
#: not a bad day -- the quantity is deterministic, and the tolerance is
#: the Monte Carlo reference's, not ours.
_MC_MAX_PULL = 5.0

#: The file that reference comes from.
_MC_FILE = ("MC_patched/TFIM/chain/"
            "1qp_gap_series_1d_chain_tfim_k0_sigma1.0_order11.csv")


def _check_data() -> list[str]:
    """Every shipped file against its SHA-256 in the manifest."""
    from gzl._data import MANIFEST_NAME, _check_manifest, _read_manifest, data_path

    root = data_path()
    manifest = root / MANIFEST_NAME
    problems = _check_manifest(root, manifest)
    if problems:
        return problems
    rows = _read_manifest(manifest)
    if len(rows) < 100:
        return [f"only {len(rows)} manifest rows -- is the data really there?"]
    print(f"data    : {len(rows)} files under {root}, all sha256 match")
    return []


def _check_closed_form() -> list[str]:
    """Orders 1-3 of the chain 1qp series at k = 0.

    They are closed forms, so they do not move with the grid, and order
    1 is ``-2 zeta(nu)``, which at nu = 2 is ``-pi^2 / 6 * 2``.  A
    routing or truncation fault shows up here without any reference
    file.
    """
    import numpy as np

    from gzl import compute_series_coefficients

    problems = []
    coarse, fine = (
        compute_series_coefficients("tfim1qp", 2.0, "chain", n, order_max=3)
        for n in (16, 64)
    )
    c = {o: float(np.asarray(fine[o]).ravel()[0]) for o in (1, 2, 3)}
    for o in (1, 2, 3):
        lo = float(np.asarray(coarse[o]).ravel()[0])
        if lo != c[o]:
            problems.append(
                f"order {o} moved with the grid: {lo!r} at n = 16 vs "
                f"{c[o]!r} at n = 64, but orders 1-3 are closed forms")
    exact = -math.pi ** 2 / 3
    if abs(c[1] - exact) > 1e-12:
        problems.append(f"order 1 is {c[1]!r}, not -pi^2/3 = {exact!r}")
    if not problems:
        print(f"series  : c1 = {c[1]!r} (exact -pi^2/3, "
              f"|d| = {abs(c[1] - exact):.1e})")
        print("        : orders 1-3 grid-independent at n = 16 and 64")

    # The same lattice by name: "chain" IS [[1.0]], so the named-lattice
    # table must resolve to the same matrix, to the bit.
    named = compute_series_coefficients("tfim1qp", 2.0, np.array([[1.0]]), 16,
                                        order_max=3)
    for o in (1, 2, 3):
        got = float(np.asarray(named[o]).ravel()[0])
        if got != float(np.asarray(coarse[o]).ravel()[0]):
            problems.append(
                f'order {o}: "chain" and [[1.0]] disagree, {got!r} vs '
                f"{float(np.asarray(coarse[o]).ravel()[0])!r}")
    return problems


def _check_reference() -> list[str]:
    """Order 3 against the shipped Monte Carlo series."""
    import csv

    import numpy as np

    from gzl import compute_series_coefficients, data_path

    res = compute_series_coefficients("tfim1qp", 2.0, "chain", 64, order_max=3)
    value = float(np.asarray(res[3]).ravel()[0])
    with data_path(_MC_FILE).open(newline="") as fh:
        ref = {int(r["order"]): (float(r["prefactor"]), float(r["error"]))
               for r in csv.DictReader(fh)}
    mc, err = ref[3]
    pull = abs(value - mc) / err
    if pull > _MC_MAX_PULL:
        return [f"order 3 is {value!r}, {pull:.1f} sigma from the Monte "
                f"Carlo reference {mc} +- {err:.1e} (limit "
                f"{_MC_MAX_PULL:g})"]
    print(f"        : order 3 vs Langheld et al. (2022), pull {pull:.2f}")
    return []


#: name -> check.  Each returns a list of problems; empty is the pass.
_CHECKS: tuple[tuple[str, Callable[[], list[str]]], ...] = (
    ("data", _check_data),
    ("closed form", _check_closed_form),
    ("reference", _check_reference),
)


def build_parser(prog: str = "gzl selftest") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Check this installation: the shipped data against its "
                    "manifest, and values whose answers are known "
                    "independently of the code that produces them.",
    )
    return p


def run(argv: Iterable[str] | None = None, *,
        prog: str = "gzl selftest") -> int:
    """Run every check; return 0 if all pass and 1 otherwise."""
    build_parser(prog).parse_args(argv)

    problems: list[str] = []
    for name, check in _CHECKS:
        try:
            problems += check()
        except Exception as exc:                         # pragma: no cover
            problems.append(f"{name}: {type(exc).__name__}: {exc}")

    if problems:
        print(file=sys.stderr)
        for problem in problems:
            print(f"gzl selftest: {problem}", file=sys.stderr)
        print(f"\nFAILED ({len(problems)} problem(s))", file=sys.stderr)
        return 1
    print("OK")
    return 0
