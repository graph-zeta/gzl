# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The shipped reference data, its manifest and the NOTICE stay in step.

``gzl/data/`` ships in the wheel with a ``PROVENANCE.csv`` that has one
row and one SHA-256 per file, and every third-party source a row names
has an entry in ``NOTICE``.

A reference file that changes, appears or disappears therefore fails
here, in the same commit, instead of drifting silently.  The negative
tests at the bottom build synthetic trees and show that the gate fails
for each way it can be wrong.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

import gzl
from gzl import data_path
from gzl._data import MANIFEST_NAME, _check_manifest, _sha256

REPO = Path(__file__).resolve().parents[1]
SHIPPED = REPO / "gzl" / "data"
NOTICE = REPO / "NOTICE"


#: Sources a shipped file may carry: this project, or a NOTICE entry.
PROJECT_SOURCE = "gzl"


def _rows(manifest: Path) -> list[dict]:
    with manifest.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _notice_keys() -> set[str]:
    return set(re.findall(r"^\[([a-z0-9-]+)\]", NOTICE.read_text(encoding="utf-8"), re.M))


# ---------------------------------------------------------------------------
# data_path
# ---------------------------------------------------------------------------

class TestDataPath:

    def test_the_root_is_the_package_data_directory(self):
        root = data_path()
        assert isinstance(root, Path)
        assert root == Path(gzl.__file__).resolve().parent / "data"
        assert root.is_dir()

    def test_a_corpus_resolves_to_a_real_file(self):
        corpus = data_path("tfim_softcore_corpus_1qp.npz")
        assert isinstance(corpus, Path) and corpus.is_file()

    def test_a_directory_resolves_too(self):
        assert data_path("MC_patched/TFIM/chain").is_dir()

    @pytest.mark.parametrize("name", ["../pyproject.toml", "MC_patched/../../_data.py", "/etc/hosts"])
    def test_a_path_out_of_the_data_directory_is_refused(self, name):
        with pytest.raises(ValueError, match="relative to"):
            data_path(name)

    def test_a_missing_file_is_named(self):
        with pytest.raises(FileNotFoundError, match="no_such_corpus.npz"):
            data_path("no_such_corpus.npz")


# ---------------------------------------------------------------------------
# The shipped tree against its manifest and the NOTICE
# ---------------------------------------------------------------------------

def test_the_shipped_tree_matches_its_manifest():
    assert _check_manifest(SHIPPED, SHIPPED / MANIFEST_NAME) == []


def test_a_shipped_file_has_a_project_or_notice_source():
    allowed = _notice_keys() | {PROJECT_SOURCE}
    bad = [(r["file"], r["source"]) for r in _rows(SHIPPED / MANIFEST_NAME) if r["source"] not in allowed]
    assert bad == []


def test_the_notice_and_the_manifest_name_the_same_sources():
    # Every NOTICE entry must earn its place in the shipped manifest.
    used = {r["source"] for r in _rows(SHIPPED / MANIFEST_NAME)}
    keys = _notice_keys()
    assert keys, "NOTICE lists no [source] entries"
    assert keys - used == set(), f"NOTICE entries no file uses: {sorted(keys - used)}"
    shipped_third_party = {r["source"] for r in _rows(SHIPPED / MANIFEST_NAME)} - {PROJECT_SOURCE}
    assert shipped_third_party - keys == set(), f"shipped sources without a NOTICE entry: {sorted(shipped_third_party - keys)}"


# ---------------------------------------------------------------------------
# The gate can fail
# ---------------------------------------------------------------------------

def _tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "data"
    (root / "MC").mkdir(parents=True)
    (root / "a.csv").write_text("order,prefactor\n1,0.5\n")
    (root / "MC" / "b.dat").write_text("2\t1.25\t0.01\n")
    (root / "README.md").write_text("documents are not data\n")
    (root / ".DS_Store").write_bytes(b"\x00")
    manifest = root / MANIFEST_NAME
    with manifest.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["file", "sha256", "source", "table", "evidence", "note"])
        for rel in ("a.csv", "MC/b.dat"):
            w.writerow([rel, _sha256(root / rel), "x", "", "", ""])
    return root, manifest


def test_a_consistent_synthetic_tree_passes(tmp_path):
    root, manifest = _tree(tmp_path)
    assert _check_manifest(root, manifest) == []


def test_an_unlisted_file_fails(tmp_path):
    root, manifest = _tree(tmp_path)
    (root / "MC" / "c.csv").write_text("new\n")
    assert _check_manifest(root, manifest) == ["MC/c.csv: on disk but not listed in PROVENANCE.csv"]


def test_a_vanished_file_fails(tmp_path):
    root, manifest = _tree(tmp_path)
    (root / "a.csv").unlink()
    assert _check_manifest(root, manifest) == ["a.csv: listed in PROVENANCE.csv but not on disk"]


def test_an_edited_file_fails(tmp_path):
    root, manifest = _tree(tmp_path)
    (root / "MC" / "b.dat").write_text("2\t1.26\t0.01\n")
    assert _check_manifest(root, manifest) == ["MC/b.dat: sha256 differs from PROVENANCE.csv"]


def test_a_line_ending_change_fails(tmp_path):
    root, manifest = _tree(tmp_path)
    (root / "a.csv").write_bytes(b"order,prefactor\r\n1,0.5\r\n")
    assert _check_manifest(root, manifest) == ["a.csv: sha256 differs from PROVENANCE.csv"]


def test_a_duplicated_row_fails(tmp_path):
    root, manifest = _tree(tmp_path)
    with manifest.open("a", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerow(["a.csv", _sha256(root / "a.csv"), "x", "", "", ""])
    assert _check_manifest(root, manifest) == ["a.csv: listed twice in PROVENANCE.csv"]


def test_a_subtree_check_ignores_files_outside_it(tmp_path):
    root, manifest = _tree(tmp_path)
    (root / "generated_output.csv").write_text("not reference data\n")
    rows = [r for r in _rows(manifest) if r["file"].startswith("MC/")]
    sub_manifest = root / "MC" / MANIFEST_NAME
    with sub_manifest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    assert _check_manifest(root, sub_manifest, subtree="MC") == []
