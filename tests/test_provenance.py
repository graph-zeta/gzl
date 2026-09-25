# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Is the suite testing THIS tree?

This repository is normally worked in through several git worktrees at
once, while ``pip install -e`` can only point at one of them.  So an
``import gzl`` resolves to the editable install's target unless
something puts the current tree first, and the two are routinely
different checkouts.

``pyproject.toml`` sets ``pythonpath = ["."]``, so a pytest run from the
repo root does put this tree first — this file asserts that it worked,
rather than trusting it.  The failure it guards is silent by
construction: the wrong tree imports cleanly, the suite passes, and the
result describes code nobody is editing.  It has already served
O(1)-wrong values once in this project's history.

The same trap bites ad-hoc scripts HARDER, and pytest cannot catch that
case: ``python /abs/path/script.py`` puts the SCRIPT's directory on
``sys.path``, never the repo root, so it silently imports the editable
install's tree.  That once produced a "972 of 972 values bit-identical"
gate in which all 972 entries were ``EXC:TypeError`` — the wrong tree
lacked the keyword being tested, every call raised, and comparing
exceptions to exceptions is trivially equal.

**Rule for any ad-hoc verification script: run it with the repo root on
the path** (a heredoc from the repo root, ``PYTHONPATH=$(pwd)``, or
``python -m``) **and assert two things before trusting a number** — that
``gzl.__file__`` is under the tree you are editing, and that the
comparison actually produced values rather than a uniform error.
"""

from __future__ import annotations

import pathlib

import gzl


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_imported_package_is_this_worktree():
    pkg = pathlib.Path(gzl.__file__).resolve()
    assert pkg.is_relative_to(REPO_ROOT), (
        f"gzl imported from {pkg}, which is OUTSIDE this tree "
        f"({REPO_ROOT}).  The suite is testing a different checkout — "
        f"most likely the target of an `pip install -e` in a sibling "
        f"worktree.  Re-run pytest from the repo root, or reinstall "
        f"editable here.  Do not trust any result from this run."
    )


def test_repo_root_really_is_a_checkout_of_this_project():
    """Liveness for the test above: if REPO_ROOT were wrong, the
    assertion could pass or fail for reasons having nothing to do with
    provenance."""
    assert (REPO_ROOT / "gzl" / "__init__.py").is_file()
    assert (REPO_ROOT / "pyproject.toml").is_file()


# ---------------------------------------------------------------------------
# The import-time guard (gzl/_provenance.py).
#
# The tests above pin the pytest half of the trap.  These pin the half that
# pytest cannot reach: an ad-hoc script, a heredoc, or a one-off probe
# that never puts its own tree first.  The guard runs at ``import
# gzl`` and so is the one place the trap cannot route around.
# ---------------------------------------------------------------------------

import os
import subprocess
import sys

import pytest

from gzl import _provenance


def _fake_checkout(root: pathlib.Path) -> pathlib.Path:
    """A directory that ``_checkout_root`` recognises as a graph-zeta tree."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'graph-zeta'\n")
    (root / "gzl").mkdir()
    (root / "gzl" / "__init__.py").write_text("")
    return root


def test_checkout_root_finds_the_enclosing_tree(tmp_path):
    tree = _fake_checkout(tmp_path / "wt")
    deep = tree / "applications" / "tfim_softcore"
    deep.mkdir(parents=True)
    assert _provenance._checkout_root(str(deep)) == str(tree.resolve())


def test_checkout_root_is_none_outside_any_tree(tmp_path):
    plain = tmp_path / "not-a-checkout"
    plain.mkdir()
    assert _provenance._checkout_root(str(plain)) is None


def test_check_is_silent_when_cwd_matches_the_package():
    # The live import: pytest runs from the repo root, so they agree.
    _provenance.check(gzl.__file__)


def test_check_raises_when_cwd_is_a_different_checkout(tmp_path, monkeypatch):
    other = _fake_checkout(tmp_path / "sibling")
    monkeypatch.chdir(other)
    monkeypatch.delenv("GRAPH_ZETA_PROVENANCE", raising=False)
    with pytest.raises(_provenance.ProvenanceError) as exc:
        _provenance.check(gzl.__file__)
    msg = str(exc.value)
    assert str(other.resolve()) in msg and str(REPO_ROOT) in msg


def test_warn_mode_writes_the_banner_and_continues(tmp_path, monkeypatch, capsys):
    other = _fake_checkout(tmp_path / "sibling")
    monkeypatch.chdir(other)
    monkeypatch.setenv("GRAPH_ZETA_PROVENANCE", "warn")
    _provenance.check(gzl.__file__)
    assert "PROVENANCE MISMATCH" in capsys.readouterr().err


def test_off_mode_disables_the_check(tmp_path, monkeypatch):
    other = _fake_checkout(tmp_path / "sibling")
    monkeypatch.chdir(other)
    monkeypatch.setenv("GRAPH_ZETA_PROVENANCE", "off")
    _provenance.check(gzl.__file__)          # must not raise


def test_guard_fires_on_a_real_subprocess_import(tmp_path):
    """End-to-end: the trap exactly as it happens in this repository.

    ``python /other/tree/applications/probe.py`` puts the SCRIPT's
    directory on ``sys.path[0]`` — not the checkout root — so
    ``import gzl`` falls through to the editable install, which
    is a different tree.  Nothing about the run says so, and that is
    the whole problem.  ``PYTHONPATH`` stands in for the editable
    install here so the test does not depend on where it points.
    """
    other = _fake_checkout(tmp_path / "sibling")
    scripts = other / "applications"
    scripts.mkdir()
    probe = scripts / "probe.py"
    probe.write_text("import gzl\n")
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    env.pop("GRAPH_ZETA_PROVENANCE", None)
    neutral = tmp_path / "elsewhere"
    neutral.mkdir()
    proc = subprocess.run([sys.executable, str(probe)], cwd=str(neutral),
                          env=env, capture_output=True, text=True)
    assert proc.returncode != 0, "the foreign-tree import was allowed through"
    assert "PROVENANCE MISMATCH" in proc.stderr
    # It must name the tree the SCRIPT lives in, which is the actionable half.
    assert str(other.resolve()) in proc.stderr


# ---------------------------------------------------------------------------
# An installed package is never a foreign checkout.
#
# The guard is for one developer trap: an editable install (or a tree on
# PYTHONPATH) serving a sibling checkout.  A wheel in site-packages cannot
# be that sibling, and a user who clones the repository to run an example
# or open a notebook against the installed package is doing nothing wrong
# — so the check applies only when the LOADED package root is itself a
# checkout.  Every test here pins ``sys.argv`` so the script probe sees
# what the test says, not pytest's own entry point.
# ---------------------------------------------------------------------------


def _fake_installed(site: pathlib.Path) -> str:
    """A ``gzl/__init__.py`` inside a site-packages directory."""
    (site / "gzl").mkdir(parents=True)
    init = site / "gzl" / "__init__.py"
    init.write_text("")
    return str(init)


def test_installed_package_is_silent_inside_a_checkout(tmp_path, monkeypatch):
    pkg = _fake_installed(tmp_path / "venv" / "lib" / "python3.12"
                          / "site-packages")
    clone = _fake_checkout(tmp_path / "clone")
    monkeypatch.chdir(clone)
    monkeypatch.setattr(sys, "argv", ["-c"])
    monkeypatch.delenv("GRAPH_ZETA_PROVENANCE", raising=False)
    _provenance.check(pkg)                          # must not raise


def test_stray_pyproject_in_site_packages_does_not_make_a_checkout(
        tmp_path, monkeypatch):
    """Another distribution dropping a top-level ``pyproject.toml`` into
    site-packages must not turn the installed package into a checkout."""
    site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    pkg = _fake_installed(site)
    (site / "pyproject.toml").write_text("[project]\nname = 'stray'\n")
    clone = _fake_checkout(tmp_path / "clone")
    monkeypatch.chdir(clone)
    monkeypatch.setattr(sys, "argv", ["-c"])
    monkeypatch.delenv("GRAPH_ZETA_PROVENANCE", raising=False)
    _provenance.check(pkg)                          # must not raise


def test_installed_package_is_silent_with_the_venv_inside_the_clone(
        tmp_path, monkeypatch):
    """``python examples/x.py`` from a clone whose ``.venv`` holds the wheel.

    Both probes land in the clone, and an ancestor walk from the package
    would reach the clone root too — the test is on the package root
    itself, not on its ancestors.
    """
    clone = _fake_checkout(tmp_path / "clone")
    pkg = _fake_installed(clone / ".venv" / "lib" / "python3.12"
                          / "site-packages")
    (clone / "examples").mkdir()
    script = clone / "examples" / "x.py"
    script.write_text("import gzl\n")
    monkeypatch.chdir(clone)
    monkeypatch.setattr(sys, "argv", [str(script)])
    monkeypatch.delenv("GRAPH_ZETA_PROVENANCE", raising=False)
    _provenance.check(pkg)                          # must not raise


def test_a_checkout_load_inside_another_checkout_still_raises(
        tmp_path, monkeypatch):
    """The developer trap, with both trees fake: loaded from A, standing in B."""
    a = _fake_checkout(tmp_path / "a")
    b = _fake_checkout(tmp_path / "b")
    deep = b / "applications"
    deep.mkdir()
    monkeypatch.chdir(deep)
    monkeypatch.setattr(sys, "argv", ["-c"])
    monkeypatch.delenv("GRAPH_ZETA_PROVENANCE", raising=False)
    with pytest.raises(_provenance.ProvenanceError) as exc:
        _provenance.check(str(a / "gzl" / "__init__.py"))
    msg = str(exc.value)
    assert str(a.resolve()) in msg and str(b.resolve()) in msg


def test_a_checkout_load_inside_the_same_checkout_is_silent(
        tmp_path, monkeypatch):
    a = _fake_checkout(tmp_path / "a")
    deep = a / "applications"
    deep.mkdir()
    script = deep / "probe.py"
    script.write_text("import gzl\n")
    monkeypatch.chdir(deep)
    monkeypatch.setattr(sys, "argv", [str(script)])
    monkeypatch.delenv("GRAPH_ZETA_PROVENANCE", raising=False)
    _provenance.check(str(a / "gzl" / "__init__.py"))   # must not raise
