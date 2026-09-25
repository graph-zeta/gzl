# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""``pytest -m slow`` runs the slow tests.

``tests/conftest.py`` carries the ``slow`` hook.  It is loaded verbatim
into a throwaway pytest session holding one fast and one slow test, and
every invocation the docs promise is run against it.  The hook used to skip each slow test
unless ``--runslow`` was given, so ``-m slow`` selected the slow tests and
then skipped all of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_REPO = Path(__file__).resolve().parents[1]
_CONFTESTS = {
    "tests": _REPO / "tests" / "conftest.py",
}

_PROBE = """
import pytest


def test_fast():
    pass


@pytest.mark.slow
def test_slow():
    pass
"""


def _names(reports):
    return sorted(r.nodeid.rpartition("::")[2] for r in reports)


@pytest.mark.parametrize("args, passed, skipped", [
    ((), ["test_fast"], ["test_slow"]),
    (("-m", "slow"), ["test_slow"], []),
    (("--runslow",), ["test_fast", "test_slow"], []),
    (("-m", "not slow"), ["test_fast"], []),
], ids=["default", "m-slow", "runslow", "m-not-slow"])
@pytest.mark.parametrize("tree", sorted(_CONFTESTS))
def test_slow_marker_selection(pytester, tree, args, passed, skipped):
    pytester.makeconftest(_CONFTESTS[tree].read_text())
    pytester.makepyfile(test_probe=_PROBE)
    reprec = pytester.inline_run(*args)
    got_passed, got_skipped, got_failed = reprec.listoutcomes()
    assert reprec.ret == 0 and not got_failed
    assert _names(got_passed) == passed
    assert _names(got_skipped) == skipped
