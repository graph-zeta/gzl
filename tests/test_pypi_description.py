# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The PyPI description, ``docs/pypi.md``, is the README's head as
``tools/pypi_description.py`` renders it.  Run that script after editing
the README."""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPI = ROOT / "docs" / "pypi.md"


def _tool():
    spec = importlib.util.spec_from_file_location(
        "_pypi_description", ROOT / "tools" / "pypi_description.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_description_is_up_to_date():
    want = _tool().render((ROOT / "README.md").read_text(encoding="utf-8"))
    assert PYPI.read_text(encoding="utf-8") == want, (
        "docs/pypi.md is stale: run python tools/pypi_description.py")


def test_it_has_no_math_and_only_absolute_links():
    text = PYPI.read_text(encoding="utf-8")
    assert "$" not in text
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        assert target.startswith("https://"), target
    for source in re.findall(r'\b(?:src|srcset)="([^"]+)"', text):
        assert source.startswith("https://raw.githubusercontent.com/graph-zeta/gzl/"), source
    for repo in re.findall(r"github\.com/(graph-zeta/[^/)\s#?]+)", text):
        assert repo == "graph-zeta/gzl", repo


def test_the_package_ships_it():
    with (ROOT / "pyproject.toml").open("rb") as fh:
        readme = tomllib.load(fh)["project"]["readme"]
    assert readme == {"file": "docs/pypi.md", "content-type": "text/markdown"}
