# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""The built wheel and sdist carry exactly the package, its data and its licenses.

    python .github/scripts/check_dist.py dist/

Compares the ``gzl/`` part of both archives against
``git ls-files gzl`` -- every tracked file must ship and nothing
else may -- and checks the licence files and metadata.  An editable
install finds in-tree data whether or not the package-data globs match
it, so this is the only gate that can see a file silently dropped from
(or leaked into) the distribution.  Exit status 1 on any problem.
"""
from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
#: The legacy import name, shipped as a one-file alias beside gzl/.
ALIAS = "graph_zeta"


def tracked() -> set[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "gzl"],
                         check=True, capture_output=True, text=True).stdout
    return {line for line in out.splitlines() if line}


def main(dist: Path) -> int:
    wheels = sorted(dist.glob("gzl-*.whl"))
    sdists = sorted(dist.glob("gzl-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        print(f"expected one wheel and one sdist in {dist}, found {wheels + sdists}")
        return 1
    want = tracked()
    problems = []

    with zipfile.ZipFile(wheels[0]) as whl:
        names = {n for n in whl.namelist() if not n.endswith("/")}
        dist_info = {n.split("/", 1)[0] for n in names if ".dist-info/" in n}
        if len(dist_info) != 1:
            problems.append(f"wheel has {len(dist_info)} .dist-info directories")
        info = next(iter(dist_info), "")
        pkg = {n for n in names if n.startswith("gzl/")}
        alias = {n for n in names if n.startswith(ALIAS + "/")}
        if alias != {f"{ALIAS}/__init__.py"}:
            problems.append(f"the {ALIAS} alias package is {sorted(alias)}, "
                            f"expected exactly {ALIAS}/__init__.py")
        other = names - pkg - alias - {n for n in names if n.startswith(info + "/")}
        if other:
            problems.append(
                f"wheel carries files outside gzl/, {ALIAS}/ and {info}/: {sorted(other)[:10]}")
        for label, diff in (("missing from wheel", want - pkg), ("in wheel but untracked", pkg - want)):
            if diff:
                problems.append(f"{label}: {len(diff)} files, e.g. {sorted(diff)[:10]}")
        for lic in ("LICENSE", "NOTICE"):
            if f"{info}/licenses/{lic}" not in names:
                problems.append(f"wheel lacks {info}/licenses/{lic}")
        meta = Parser().parsestr(whl.read(f"{info}/METADATA").decode())
        if meta.get("License-Expression") != "AGPL-3.0-or-later":
            problems.append(f"METADATA License-Expression is {meta.get('License-Expression')!r}")
        if sorted(meta.get_all("License-File") or []) != ["LICENSE", "NOTICE"]:
            problems.append(f"METADATA License-File entries are {meta.get_all('License-File')}")
        print(f"wheel  {wheels[0].name}: {len(pkg)} package files, "
              f"{sum(n.startswith('gzl/data/') for n in pkg)} under gzl/data/")

    with tarfile.open(sdists[0]) as tar:
        members = [m.name for m in tar.getmembers() if m.isfile()]
        top = {m.split("/", 1)[0] for m in members}
        if len(top) != 1:
            problems.append(f"sdist has {len(top)} top-level directories")
        prefix = next(iter(top), "") + "/"
        rel = {m[len(prefix):] for m in members if m.startswith(prefix)}
        pkg = {m for m in rel if m.startswith("gzl/") and ".egg-info/" not in m}
        if f"{ALIAS}/__init__.py" not in rel:
            problems.append(f"sdist lacks the {ALIAS} alias package")
        for label, diff in (("missing from sdist", want - pkg), ("in sdist but untracked", pkg - want)):
            if diff:
                problems.append(f"{label}: {len(diff)} files, e.g. {sorted(diff)[:10]}")
        for lic in ("LICENSE", "NOTICE"):
            if lic not in rel:
                problems.append(f"sdist lacks {lic}")
        # MANIFEST.in keeps the test suite out: without conftest.py and
        # tests/data it could not run from the sdist anyway.
        leaked = sorted(m for m in rel if m.startswith("tests/"))
        if leaked:
            problems.append(f"sdist carries the test suite: {leaked[:5]}")
        print(f"sdist  {sdists[0].name}: {len(pkg)} package files")

    for p in problems:
        print("PROBLEM:", p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "dist")))
