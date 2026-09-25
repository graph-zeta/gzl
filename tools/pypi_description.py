# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Write the PyPI description, ``docs/pypi.md``, from ``README.md``.

PyPI renders Markdown but no math, and it resolves no relative links.
The description is therefore the README up to its first section with
formulas, "Quick start", with every relative link made absolute on
GitHub, every relative image source made absolute on GitHub's raw host,
and a pointer to the rest of the documentation.
``tests/test_pypi_description.py`` fails when the file is out of date.

Run from the repository root after editing the README:

    python tools/pypi_description.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = "https://github.com/graph-zeta/gzl"
#: Images come from the raw host: a /blob/ URL is a web page, not an image.
RAW = "https://raw.githubusercontent.com/graph-zeta/gzl/main"
ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
OUT = ROOT / "docs" / "pypi.md"
STOP = "\n## Quick start\n"

TAIL = f"""
The definition of the graph zeta function and the quick start continue in
the [README on GitHub]({REPO}#quick-start). For an extensive discussion of
all front-ends, general interaction kernels, the worked examples, the
engines and the full API reference, see
[DOCUMENTATION.md]({REPO}/blob/main/DOCUMENTATION.md).
"""


def _absolute(match: re.Match) -> str:
    target = match.group(1)
    if re.match(r"[a-z][a-z0-9+.-]*:", target):
        return match.group(0)
    if target.startswith("#"):
        return f"]({REPO}{target})"
    return f"]({REPO}/blob/main/{target.removeprefix('./')})"


def _absolute_src(match: re.Match) -> str:
    attribute, target = match.groups()
    if re.match(r"[a-z][a-z0-9+.-]*:", target):
        return match.group(0)
    return f'{attribute}="{RAW}/{target.removeprefix("./")}"'


def render(readme: str) -> str:
    """The PyPI description for the text of ``README.md``."""
    head, sep, _ = readme.partition(STOP)
    if not sep:
        raise ValueError("README.md has no 'Quick start' section")
    head = re.sub(r"\]\(([^)\s]+)\)", _absolute, head)
    head = re.sub(r'\b(src|srcset)="([^"]+)"', _absolute_src, head)
    return head.rstrip() + "\n" + TAIL


def main() -> int:
    OUT.write_text(render(README.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
