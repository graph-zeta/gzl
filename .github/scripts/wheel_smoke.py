# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""An installed wheel works on its own: data present, intact, and usable.

Feed this on stdin from a directory outside any source checkout:

    cd "$RUNNER_TEMP" && "$VENV/bin/python" - < .github/scripts/wheel_smoke.py

A clean room: no source tree in the cwd or on ``sys.path``, so everything
the run touches came from the wheel.  CI also runs it BY PATH from inside
the checkout (``cd examples && python ../.github/scripts/wheel_smoke.py``),
where the script's directory -- not the repository root -- is
``sys.path[0]``, so the installed wheel is still what it imports.  On stdin
there is no ``__file__``, so nothing here may use it.

What is checked *about the install* lives in ``gzl selftest``, so that a
user who installed from PyPI can run the same checks -- the test suite
ships in neither the wheel nor the sdist.  What stays here is what only CI
can assert: that the thing being checked is the wheel and not a source
tree standing next to it.
"""
import sysconfig
from pathlib import Path

import gzl
from gzl._cli_selftest import run

pkg = Path(gzl.__file__).resolve()
site = Path(sysconfig.get_paths()["purelib"]).resolve()
assert pkg.is_relative_to(site), f"gzl imported from {pkg}, not the installed wheel in {site}"
print(f"clean room: gzl {gzl.__version__} imported from {pkg}")

rc = run([])
assert rc == 0, f"gzl selftest returned {rc}"
print("wheel smoke test passed")
