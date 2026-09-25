# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The release workflow's guard: tag, version, changelog and CI agree.

For a pushed tag the guard requires

* the tag to be ``v`` followed by ``gzl.__version__``, e.g. ``v1.0.0rc1``;
* a changelog entry ``## [X.Y.Z]`` for the release part of the version,
  dated for a final release (a pre-release may sit under ``Unreleased``);
* the tagged commit to be on ``main``, with a successful CI run.

Run by hand (``workflow_dispatch``), the workflow is a dry run on a
branch: only the version and the changelog entry are checked.  The guard
writes ``version`` and ``prerelease`` to ``$GITHUB_OUTPUT``.

``release_guard.py --notes VERSION`` prints the changelog entry of
``VERSION`` for the release notes, unwrapped: GitHub renders every newline
of a release description as a line break.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def package_version(root: Path = ROOT) -> str:
    """``__version__`` of the source tree, read without importing gzl."""
    tree = ast.parse((root / "gzl" / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets):
            return node.value.value
    raise ValueError("gzl/__init__.py sets no __version__")


def is_prerelease(version: str) -> bool:
    """True for a PEP 440 alpha, beta, release candidate or dev release."""
    return re.search(r"(a|b|rc)\d+$|\.dev\d+$", version) is not None


def release_part(version: str) -> str:
    """``1.0.0`` for ``1.0.0rc1``."""
    match = re.match(r"\d+(\.\d+)*", version)
    if match is None:
        raise ValueError(f"not a version: {version!r}")
    return match.group(0)


def changelog_entry(changelog: str, version: str) -> tuple[str, str]:
    """The heading and the body of the changelog entry for ``version``."""
    base = re.escape(release_part(version))
    match = re.search(rf"^## \[{base}\][^\n]*\n(.*?)(?=^## |^\[[^\]]+\]: |\Z)",
                      changelog, re.M | re.S)
    if match is None:
        raise ValueError(f"CHANGELOG.md has no entry '## [{release_part(version)}]'")
    heading = match.group(0).splitlines()[0]
    return heading, match.group(1).strip() + "\n"


def unwrap(markdown: str) -> str:
    """Join the hard-wrapped lines of each paragraph and list item.

    A line starting a list item (``*``, ``-`` or ``1.``) starts a new line;
    any other line continues the one before it.  Fenced code keeps its
    lines.
    """
    blocks = []
    for block in markdown.strip().split("\n\n"):
        if block.lstrip().startswith("```"):
            blocks.append(block)
            continue
        lines: list[str] = []
        for line in block.splitlines():
            if not lines or re.match(r"\s*([*-]|\d+\.)\s", line):
                lines.append(line.rstrip())
            else:
                lines[-1] += " " + line.strip()
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def check(version: str, changelog: str, tag: str | None) -> list[str]:
    """What is wrong with releasing ``version`` under ``tag``; empty if nothing."""
    problems = []
    if tag is not None and tag != f"v{version}":
        problems.append(f"tag {tag} is not v{version}, the package version")
    try:
        heading, _ = changelog_entry(changelog, version)
    except ValueError as err:
        problems.append(str(err))
    else:
        if not is_prerelease(version) and not re.search(r"\d{4}-\d{2}-\d{2}", heading):
            problems.append(f"a final release needs a dated changelog entry, not {heading!r}")
    return problems


def _ci_problems(sha: str) -> list[str]:
    """The tagged commit must be on main and have a successful CI run."""
    problems = []
    on_main = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"])
    if on_main.returncode != 0:
        problems.append(f"{sha[:7]} is not on main")
    listed = subprocess.run(
        ["gh", "run", "list", "--workflow", "ci.yml", "--commit", sha,
         "--json", "status,conclusion"],
        capture_output=True, text=True)
    if listed.returncode != 0:
        return problems + [f"cannot list the CI runs of {sha[:7]}: "
                           f"{listed.stderr.strip() or listed.stdout.strip()}"]
    runs = json.loads(listed.stdout)
    if any(r["conclusion"] == "success" for r in runs):
        pass
    elif any(r["status"] != "completed" for r in runs):
        problems.append(f"CI is still running on {sha[:7]}; tag it once CI has passed")
    elif runs:
        problems.append(f"CI failed on {sha[:7]}")
    else:
        problems.append(f"{sha[:7]} has no CI run (a merge carrying [skip ci] "
                        "runs none); tag a commit CI tested")
    return problems


def main(argv: list[str]) -> int:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if argv[:1] == ["--notes"]:
        sys.stdout.write(unwrap(changelog_entry(changelog, argv[1])[1]))
        return 0

    version = package_version()
    is_tag = os.environ.get("GITHUB_REF_TYPE") == "tag"
    problems = check(version, changelog, os.environ["GITHUB_REF_NAME"] if is_tag else None)
    if is_tag:
        problems += _ci_problems(os.environ["GITHUB_SHA"])
    for p in problems:
        print(f"::error::{p}")
    if problems:
        return 1

    pre = "true" if is_prerelease(version) else "false"
    print(f"gzl {version}, prerelease={pre}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"version={version}\nprerelease={pre}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
