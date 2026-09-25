# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The ``info`` command.

What a bug report needs, in one command.  ``CONTRIBUTING.md`` asks for
the `gzl` version, where it was imported from, and the `numpy` and
`epsteinlib` versions; collecting those by hand means three commands
and a guess at which checkout answered.

Nothing imports this module -- see :mod:`gzl.cli` for why that matters.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Iterable

__all__ = ["build_parser", "run"]

#: Reported in this order, which is roughly the order they break in.
_DEPENDENCIES = ("numpy", "scipy", "epsteinlib", "networkx", "h5py")


def _dependency_versions() -> list[tuple[str, str]]:
    import importlib
    import importlib.metadata

    out = []
    for name in _DEPENDENCIES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            out.append((name, "not installed"))
            continue
        version = getattr(module, "__version__", None)
        if version is None:
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                version = "unknown"
        out.append((name, str(version)))
    return out


def _origin(package_dir: Path) -> str:
    """Whether the imported package is a checkout or an installed copy.

    Which one answered is the question behind most "it worked
    yesterday" reports, and with several worktrees on one machine it is
    not guessable from the version alone.
    """
    import sysconfig

    for key in ("purelib", "platlib"):
        site = sysconfig.get_paths().get(key)
        if site and package_dir.is_relative_to(Path(site).resolve()):
            return "installed"
    if (package_dir.parent / "pyproject.toml").is_file():
        return "source checkout"
    return "unknown origin"


def _data_report() -> list[str]:
    """The shipped data directory, and whether it is intact."""
    from gzl._data import MANIFEST_NAME, _check_manifest, data_path

    try:
        root = data_path()
    except Exception as exc:                             # pragma: no cover
        return [f"data         unavailable ({exc})"]

    manifest = root / MANIFEST_NAME
    if not manifest.is_file():                           # pragma: no cover
        return [f"data         {root}", f"             no {MANIFEST_NAME}"]

    problems = _check_manifest(root, manifest)
    n_files = sum(1 for p in root.rglob("*")
                  if p.is_file() and not p.name.startswith(".")
                  and p.suffix != ".md" and p.name != MANIFEST_NAME)
    lines = [f"data         {root}"]
    if problems:
        lines.append(f"             {n_files} files, "
                     f"{len(problems)} PROBLEM(S) against {MANIFEST_NAME}")
        lines += [f"               {p}" for p in problems[:5]]
        if len(problems) > 5:
            lines.append(f"               ... and {len(problems) - 5} more")
    else:
        lines.append(f"             {n_files} files, all sha256 match")
    return lines


def build_parser(prog: str = "gzl info") -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog=prog,
        description="Versions, install location and shipped data -- what a "
                    "bug report needs.",
    )


def run(argv: Iterable[str] | None = None, *,
        prog: str = "gzl info") -> int:
    build_parser(prog).parse_args(argv)

    import gzl
    from gzl._data import CORPORA
    from gzl._lattices import LATTICES

    package_dir = Path(gzl.__file__).resolve().parent

    print(f"gzl          {gzl.__version__}")
    print(f"location     {package_dir}  ({_origin(package_dir)})")
    print(f"python       {platform.python_version()}  "
          f"({sys.executable})")
    print(f"platform     {platform.platform()}")
    print("dependencies " + " · ".join(f"{n} {v}"
                                       for n, v in _dependency_versions()))
    for line in _data_report():
        print(line)
    print(f"corpora      {', '.join(sorted(CORPORA))}")
    print(f"lattices     {', '.join(LATTICES)}")
    return 0
