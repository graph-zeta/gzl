# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The ``gzl`` command.

One umbrella command with a subcommand per job, installed by
``[project.scripts]`` in ``pyproject.toml`` and reachable as ``python
-m gzl`` as well.

Nothing in the package imports this module, and that is deliberate.
``gzl/__init__.py`` imports ``gzl.series`` eagerly, so ``python -m
gzl.series`` makes runpy execute that file a second time as
``__main__`` and warn about it.  Because nothing imports ``gzl.cli``
or ``gzl._cli_series``, the console script and ``python -m gzl`` run
exactly once by construction -- there is no ordering invariant for a
later edit to break.

Subcommands are registered in ``_SUBCOMMANDS`` and are imported
lazily, one per invocation, so ``gzl --help`` does not pay for
importing the evaluation stack.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Iterable

__all__ = ["main"]

_PROG = "gzl"

_EPILOG = """\
examples:
  gzl series --corpus tfim1qp --A cubic --n-points 8 --nu 3.5 \\
      --order-max 9 --output series.csv
  gzl series --config path/to/series.toml --progress
  gzl info
  gzl selftest

`gzl <command> --help` describes one command in full.
"""


def _series(argv: list[str]) -> int:
    from gzl._cli_series import run

    return run(argv, prog=f"{_PROG} series")


def _info(argv: list[str]) -> int:
    from gzl._cli_info import run

    return run(argv, prog=f"{_PROG} info")


def _selftest(argv: list[str]) -> int:
    from gzl._cli_selftest import run

    return run(argv, prog=f"{_PROG} selftest")


#: name -> (one-line help, handler).  The help lines are what
#: ``gzl --help`` lists, so they are kept to one short clause.
_SUBCOMMANDS: dict[str, tuple[str, Callable[[list[str]], int]]] = {
    "series": ("per-order series coefficients over a graph corpus", _series),
    "info": ("versions, install location and shipped data", _info),
    "selftest": ("check this installation against known values", _selftest),
}


def _version() -> str:
    from gzl import __version__

    return f"{_PROG} {__version__}"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=_PROG,
        description="Graph Zeta Library (GZL).",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # The subcommand owns every flag after its name; parsing stops
        # here so that `gzl series --config x` does not have to fight
        # the umbrella over an abbreviation.
        allow_abbrev=False,
    )
    p.add_argument("-V", "--version", action="store_true",
                   help="print the version and exit")
    p.add_argument(
        "command", nargs="?", choices=sorted(_SUBCOMMANDS),
        help="; ".join(f"{name}: {help_}"
                       for name, (help_, _) in sorted(_SUBCOMMANDS.items())),
    )
    p.add_argument("args", nargs=argparse.REMAINDER,
                   help="arguments for the command")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    """Dispatch to a subcommand; return a process exit code.

    0 success, 2 usage error.  A subcommand's own return value is
    passed through unchanged.
    """
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.version:
        print(_version())
        return 0
    if args.command is None:
        parser.print_help()
        return 0

    _, handler = _SUBCOMMANDS[args.command]
    return handler(args.args)


if __name__ == "__main__":
    sys.exit(main())
