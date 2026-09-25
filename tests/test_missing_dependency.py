# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""``TopologyEvaluatorUnavailableError`` means a missing dependency.

networkx and epsteinlib are dependencies of gzl.  Without epsteinlib
``import gzl`` fails, because gzl.core imports it unguarded; without
networkx the package imports and the router raises
``TopologyEvaluatorUnavailableError``, chained to the ``ImportError``.
The error used to be raised for a request no evaluator serves as well
(a block mixing ``nu = inf`` with finite exponents on a 1qp spine, now
a plain ``GraphZetaError``, pinned in test_interaction_router.py), and
its message recommended ``pip install gzl[graph_tools]``, an extra that
does not exist.

Each case runs in a fresh interpreter, because a module blocked in
``sys.modules`` must be blocked before gzl is first imported.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import gzl

ROOT = Path(__file__).resolve().parent.parent


def _run(code: str) -> str:
    """Run ``code`` in a fresh interpreter from the checkout; return stdout."""
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, check=False, cwd=ROOT,
    )
    assert out.returncode == 0, f"stdout={out.stdout!r} stderr={out.stderr!r}"
    return out.stdout.strip()


def test_without_networkx_the_router_raises_the_dependency_error():
    out = _run("""
        import sys
        sys.modules["networkx"] = None      # makes `import networkx` fail
        import gzl
        try:
            gzl.evaluate_graph([[0, 1], [1, 2], [2, 0]], 3.0, "chain",
                               n_points=8)
        except gzl.TopologyEvaluatorUnavailableError as exc:
            print(type(exc.__cause__).__name__)
            print(str(exc))
        else:
            print("no error")
    """)
    cause, message = out.split("\n", 1)
    assert cause == "ModuleNotFoundError"
    assert "networkx" in message and "reinstall gzl" in message
    assert "graph_tools" not in message


def test_without_epsteinlib_gzl_does_not_import():
    # The docstring of the error says so; keep it true.
    assert _run("""
        import sys
        sys.modules["epsteinlib"] = None
        try:
            import gzl
        except ImportError:
            print("refused")
        else:
            print("imported")
    """) == "refused"


def test_precision_mpmath_without_mpmath_says_what_to_install():
    # mpmath is only in the [dev] extra; graph_multiply's precision="mpmath"
    # used to die with a bare ModuleNotFoundError.
    out = _run("""
        import sys
        sys.modules["mpmath"] = None
        import gzl
        g = gzl.make_epstein_graph(3.5, [[1.0]], 16)
        gzl.graph_multiply(g, g)                    # float64 needs no mpmath
        try:
            gzl.graph_multiply(g, g, precision="mpmath")
        except ImportError as exc:
            print(type(exc.__cause__).__name__)
            print(str(exc))
        else:
            print("no error")
    """)
    cause, message = out.split("\n", 1)
    assert cause == "ModuleNotFoundError"
    assert "pip install mpmath" in message


def test_the_error_names_no_extra():
    doc = gzl.TopologyEvaluatorUnavailableError.__doc__
    assert "graph_tools" not in doc
    assert "networkx" in doc


def test_every_install_hint_names_an_extra_that_exists():
    """A hint ``gzl[<extra>]`` in the code or the documentation must name
    an extra that pyproject.toml defines.  The CHANGELOG is history and
    is not scanned."""
    with open(ROOT / "pyproject.toml", "rb") as fh:
        extras = set(tomllib.load(fh)["project"]["optional-dependencies"])
    paths = [*ROOT.glob("gzl/**/*.py"), *ROOT.glob("docs/**/*.py"),
             *(ROOT / name for name in ("README.md", "DOCUMENTATION.md",
                                        "CONTRIBUTING.md"))]
    named = {}
    for path in paths:
        for extra in re.findall(r"gzl\[([A-Za-z0-9_,-]+)\]",
                                path.read_text(encoding="utf-8")):
            for one in extra.split(","):
                named.setdefault(one, set()).add(path.relative_to(ROOT).as_posix())
    missing = {extra: sorted(where) for extra, where in named.items()
               if extra not in extras}
    assert not missing, f"install hints name undefined extras: {missing}"
