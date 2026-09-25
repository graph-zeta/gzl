# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Which checkout did ``import gzl`` actually load?

This repository may be worked in through several git worktrees at
once, while ``pip install -e`` can only point at *one* of them.  Every
entry point that does not put its own tree first therefore imports a
**sibling checkout** — cleanly, silently, and with no hint in the
output that it happened.

``python /path/to/tree/some/script.py`` is the common case: the
interpreter puts *the script's directory* on ``sys.path[0]``, never the
repository root, so a bare ``import gzl`` falls straight through
to the editable install's target.

The trap has fired three times during development:

* a stale editable install served an outdated ``hybrid.py`` whose
  off-grid single-k values were O(1) wrong, for months;
* a "972 of 972 values bit-identical" gate in which all 972 entries were
  the string ``EXC:TypeError`` — the foreign tree lacked the keyword
  under test, so every call raised, and comparing exceptions to
  exceptions is trivially equal.  It was cited as evidence before it
  was caught;
* a performance campaign in which the "fast" runs (13.4 s) were
  executing a sibling checkout and the "slow" ones (34 s) the tree under
  test.  That was chased through arguments, ``Path``-vs-``str``,
  module-vs-function scope, a commit bisect and two speculative fixes —
  all before anyone asked which library was loaded.

``tests/test_provenance.py`` pins the pytest half.  Per-script
``sys.path`` guards pin the scripts that carry them.  Neither can cover
an ad-hoc heredoc, a throwaway probe, or a script written next week —
so the check lives here, at import, where nothing can route around it.

What it detects, precisely: this package was loaded **from a graph-zeta
source checkout** (an editable install or a tree on ``sys.path`` — its
root holds ``pyproject.toml`` beside ``gzl/__init__.py``), and the
caller is standing **inside a different checkout** (cwd, or the
directory of the running script).  That combination has no legitimate
use in this project.  An installed package (a wheel in site-packages)
is never checked, wherever the caller stands — including a venv inside a
clone: running a clone's example or notebook against an installed
package is an ordinary run.  Standing outside any checkout is silent too.

That exemption is a trade-off, not an impossibility: a *non-editable*
install of a sibling tree (``pip install ../other-worktree``) is stale in
exactly the way this guard exists for, and it is no longer caught.  Work
across worktrees with ``pip install -e`` (or ``PYTHONPATH``), which the
check does cover.

Escape hatches, in order of preference:

* ``PYTHONPATH=$(pwd) python …`` or ``python -m …`` from the repo root —
  fix the run rather than the check;
* ``GRAPH_ZETA_PROVENANCE=warn`` — print the banner, keep going;
* ``GRAPH_ZETA_PROVENANCE=off`` — disable entirely.
"""

from __future__ import annotations

import os
import sys


class ProvenanceError(RuntimeError):
    """``import gzl`` resolved to a different checkout than the cwd."""


def _is_checkout(path: str) -> bool:
    """``path`` itself holds ``pyproject.toml`` and ``gzl/__init__.py``."""
    return (os.path.isfile(os.path.join(path, "pyproject.toml"))
            and os.path.isfile(os.path.join(path, "gzl", "__init__.py")))


def _checkout_root(start: str) -> str | None:
    """Innermost ancestor of ``start`` that looks like a graph-zeta checkout.

    A checkout is a directory holding both ``pyproject.toml`` and a
    ``gzl/__init__.py``.  Returns a realpath, or ``None``.
    """
    try:
        cur = os.path.realpath(start)
    except OSError:
        return None
    while True:
        if _is_checkout(cur):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _caller_roots() -> list[tuple[str, str]]:
    """``(label, checkout_root)`` for each place the caller is standing."""
    probes: list[tuple[str, str]] = []
    try:
        probes.append(("cwd", os.getcwd()))
    except OSError:
        pass
    # The running script's directory — this is the sys.path[0] that makes
    # `python /abs/path/script.py` pick up the wrong tree in the first place.
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 and argv0 not in ("-c", "-"):
        probes.append(("script", os.path.dirname(os.path.abspath(argv0))))
    out = []
    seen = set()
    for label, path in probes:
        root = _checkout_root(path)
        if root is not None and root not in seen:
            seen.add(root)
            out.append((label, root))
    return out


def _banner(pkg_root: str, offenders: list[tuple[str, str]]) -> str:
    lines = [
        "",
        "=" * 72,
        "gzl PROVENANCE MISMATCH — you are running a DIFFERENT checkout",
        "=" * 72,
        f"  imported from : {pkg_root}",
    ]
    for label, root in offenders:
        lines.append(f"  {label:<14}: {root}")
    lines += [
        "",
        "  A bare `import gzl` resolved to the editable install's",
        "  target, not to the tree you are standing in.  Any number this",
        "  run produces describes code you are not editing.",
        "",
        "  Fix the run:   PYTHONPATH=$(git rev-parse --show-toplevel) python ...",
        "            or:  python -m <module>   (from the repo root)",
        "  Silence it:    GRAPH_ZETA_PROVENANCE=warn   (banner, keep going)",
        "                 GRAPH_ZETA_PROVENANCE=off    (no check at all)",
        "=" * 72,
        "",
    ]
    return "\n".join(lines)


def check(pkg_file: str) -> None:
    """Raise (or warn) if the loaded package is a foreign checkout.

    ``pkg_file`` is ``gzl.__file__``.  An installed package — one
    whose root is not itself a checkout — is never checked.  Never raises
    anything but :class:`ProvenanceError`: an unexpected failure in the
    check itself must not take down an import.
    """
    mode = os.environ.get("GRAPH_ZETA_PROVENANCE", "error").strip().lower()
    if mode == "off":
        return
    try:
        pkg_root = os.path.dirname(os.path.dirname(os.path.realpath(pkg_file)))
        # The package root itself, never an ancestor walk: a venv inside a
        # clone would walk from its site-packages up to the clone root.
        # A stray top-level pyproject.toml dropped into site-packages by some
        # other distribution must not turn an installed package into a
        # "checkout" again.
        if (os.path.basename(pkg_root) in ("site-packages", "dist-packages")
                or not _is_checkout(pkg_root)):
            return
        offenders = [(label, root) for label, root in _caller_roots()
                     if root != pkg_root]
        if not offenders:
            return
        msg = _banner(pkg_root, offenders)
    except Exception:                      # never break an import over this
        return
    if mode == "warn":
        sys.stderr.write(msg)
        sys.stderr.flush()
        return
    raise ProvenanceError(msg)
