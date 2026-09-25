# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The release workflow's guard, ``.github/scripts/release_guard.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CHANGELOG = """# Changelog

## [1.1.0] - 2027-01-15

Later.

## [1.0.0] - Unreleased

First public release.

* A bullet.

[1.0.0]: https://github.com/graph-zeta/gzl/releases/tag/v1.0.0
"""


@pytest.fixture(scope="module")
def guard():
    spec = importlib.util.spec_from_file_location(
        "_release_guard", ROOT / ".github" / "scripts" / "release_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_package_version_is_read_without_importing(guard):
    import gzl
    assert guard.package_version() == gzl.__version__


@pytest.mark.parametrize("version, pre", [
    ("1.0.0rc1", True), ("1.0.0a2", True), ("1.0.0b1", True),
    ("1.0.0.dev3", True), ("1.0.0", False), ("1.2.10", False),
])
def test_prerelease(guard, version, pre):
    assert guard.is_prerelease(version) is pre


def test_the_entry_of_a_release_candidate_is_its_release(guard):
    heading, body = guard.changelog_entry(CHANGELOG, "1.0.0rc1")
    assert heading == "## [1.0.0] - Unreleased"
    assert body == "First public release.\n\n* A bullet.\n"
    assert guard.changelog_entry(CHANGELOG, "1.1.0")[1] == "Later.\n"


def test_the_notes_are_unwrapped(guard):
    wrapped = ("First public release.  It implements\nthe method.\n\n"
               "* `evaluate_graph` evaluates\n  a graph zeta function.\n"
               "* A second item.\n\n```bash\ngzl info\ngzl selftest\n```\n")
    assert guard.unwrap(wrapped) == (
        "First public release.  It implements the method.\n\n"
        "* `evaluate_graph` evaluates a graph zeta function.\n"
        "* A second item.\n\n```bash\ngzl info\ngzl selftest\n```\n")


def test_a_release_candidate_may_be_unreleased(guard):
    assert guard.check("1.0.0rc1", CHANGELOG, "v1.0.0rc1") == []


def test_a_final_release_needs_a_date(guard):
    assert guard.check("1.1.0", CHANGELOG, "v1.1.0") == []
    [problem] = guard.check("1.0.0", CHANGELOG, "v1.0.0")
    assert "dated" in problem


@pytest.mark.parametrize("tag", ["v1.0.0-rc1", "1.0.0rc1", "v1.0.0"])
def test_the_tag_must_be_v_and_the_version(guard, tag):
    assert any("is not v1.0.0rc1" in p for p in guard.check("1.0.0rc1", CHANGELOG, tag))


def test_a_version_without_an_entry_is_refused(guard):
    [problem] = guard.check("2.0.0rc1", CHANGELOG, "v2.0.0rc1")
    assert "no entry '## [2.0.0]'" in problem


def test_the_shipped_changelog_has_an_entry_for_the_package(guard):
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert guard.check(guard.package_version(), changelog, None) == []
