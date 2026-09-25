# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

"""pytest configuration for the core gzl test suite.

Adds the ``slow`` marker, used by long-running stability sweeps that
should not run in the default ``pytest`` invocation but can be opted
into via ``pytest --runslow`` or ``pytest -m slow``.
"""

from __future__ import annotations

import re

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.slow",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (excluded from the default suite; "
        "enable with --runslow or `-m slow`).",
    )


def pytest_collection_modifyitems(config, items):
    # A mark expression that names ``slow`` (``-m slow``, ``-m "not slow"``)
    # has already chosen which slow tests run.  Skipping them on top made
    # ``pytest -m slow`` select the slow tests and then skip every one.
    if config.getoption("--runslow") or re.search(
            r"\bslow\b", config.getoption("markexpr") or ""):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow or `-m slow` to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
