# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Reference data shipped inside the package.

``gzl/data/`` carries what a user needs to evaluate the TFIM
corpora and to compare the series against published Monte Carlo data,
with no source checkout:

* ``tfim_softcore_corpus_0qp.npz``, ``tfim_softcore_corpus_1qp.npz`` --
  the perturbative corpora (combinatorial graphs, multiplicities,
  prefactors, hoppings), 8 403 and 22 677 graphs; every corpus argument
  also takes them by name, ``"tfim0qp"`` and ``"tfim1qp"``;
* ``full_graph_topologies.npz`` -- the 31 080-graph topology snapshot;
* ``MC_patched/TFIM/<lattice>/`` -- Monte Carlo reference series.

The corpora and the snapshot are this project's own work.  Everything
else is third-party data: it is not relicensed, and its sources are in
the distribution's ``NOTICE`` and, file by file, in
``gzl/data/PROVENANCE.csv`` -- together with a SHA-256 per file,
which :func:`_check_manifest` compares against the tree.
"""

from __future__ import annotations

import csv
import hashlib
from importlib.resources import files
from pathlib import Path, PurePosixPath

__all__ = ["data_path"]

#: Per-file provenance, one row per data file: ``file`` (relative to the
#: data root, POSIX), ``sha256``, ``source``, ``table``, ``evidence``,
#: ``note``.
MANIFEST_NAME = "PROVENANCE.csv"

#: The shipped corpora by name, for every corpus argument: a string that is
#: exactly a key here means that file of the data directory (see
#: :func:`_resolve_corpus`).
CORPORA = {
    "tfim0qp": "tfim_softcore_corpus_0qp.npz",
    "tfim1qp": "tfim_softcore_corpus_1qp.npz",
}


def _data_root() -> Path:
    # Anchor on the regular package ``gzl``, never on ``gzl.data``:
    # the data directory has no ``__init__.py``, so as a namespace package it
    # would resolve to a MultiplexedPath, not a filesystem path.
    root = files("gzl") / "data"
    if not isinstance(root, Path):
        raise RuntimeError(
            "gzl is not installed as a directory tree "
            f"(importlib.resources returned {type(root).__name__}), so its data "
            "files have no filesystem path.  Extract one with "
            "importlib.resources.as_file(importlib.resources.files('gzl') "
            "/ 'data' / <name>)."
        )
    return root


def data_path(name: str = "") -> Path:
    """Filesystem path of a data file or directory shipped with gzl.

    Parameters
    ----------
    name
        Path relative to the package's data directory, with ``/`` as the
        separator -- ``"tfim_softcore_corpus_1qp.npz"``, or
        ``"MC_patched/TFIM/chain"`` for a directory.  Empty returns the
        data directory itself.

    Returns
    -------
    pathlib.Path
        An existing file or directory, usable with ``np.load``,
        ``h5py.File``, :func:`compute_series_coefficients` or ``Path.glob``.

    Raises
    ------
    ValueError
        ``name`` is absolute or climbs out of the data directory.
    FileNotFoundError
        Nothing of that name ships with this installation.  Monte Carlo
        files without a citable source are not distributed; see
        ``gzl/data/PROVENANCE.csv`` for what is.

    Examples
    --------
    >>> import numpy as np
    >>> from gzl import compute_series_coefficients, data_path
    >>> corpus = data_path("tfim_softcore_corpus_1qp.npz")
    >>> c = compute_series_coefficients(corpus, 2.0, np.array([[1.0]]), 64,
    ...                                 order_max=4)  # doctest: +SKIP
    """
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(
            f"data_path takes a path relative to gzl's data directory, got {name!r}"
        )
    root = _data_root()
    target = root.joinpath(*rel.parts)
    if not target.exists():
        what = f"data file {name!r}" if rel.parts else "data directory"
        raise FileNotFoundError(f"gzl ships no {what} (looked in {root})")
    return target


def _resolve_corpus(corpus: str | Path) -> Path:
    """The file a corpus argument means: a shipped corpus's name, or a path.

    A ``str`` exactly equal to a key of :data:`CORPORA` -- ``"tfim0qp"``,
    ``"tfim1qp"``; lowercase, no suffix, no other spelling -- is that
    corpus, ``data_path(CORPORA[corpus])``.  Anything else, a ``Path`` of
    any spelling included, is a filesystem path and comes back as
    ``Path(corpus)``, unchecked, as it always did.  A name never shadows a
    corpus file: those carry a ``.npz`` / ``.h5`` suffix, and no name does.
    """
    if isinstance(corpus, str) and corpus in CORPORA:
        return data_path(CORPORA[corpus])
    return Path(corpus)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(manifest: Path) -> list[dict]:
    with manifest.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _check_manifest(root: Path, manifest: Path, subtree: str = "") -> list[str]:
    """Every data file under ``root / subtree`` against its manifest rows.

    A file is data unless its name starts with ``.``, it is a ``*.md``
    document, or it is the manifest itself.  Keys are paths relative to
    ``root``.  Returns one message per problem -- an unlisted file, a row
    with no file, a duplicated row, or a SHA-256 mismatch -- so an empty
    list is the pass.
    """
    root = Path(root)
    manifest = Path(manifest)
    base = root / subtree if subtree else root
    on_disk = {}
    for path in sorted(base.rglob("*")):
        rel = path.relative_to(root)
        if (not path.is_file() or path == manifest or path.suffix == ".md"
                or any(part.startswith(".") for part in rel.parts)):
            continue
        on_disk[rel.as_posix()] = path

    problems = []
    seen = set()
    for row in _read_manifest(manifest):
        key = row["file"]
        if key in seen:
            problems.append(f"{key}: listed twice in {manifest.name}")
            continue
        seen.add(key)
        path = on_disk.get(key)
        if path is None:
            problems.append(f"{key}: listed in {manifest.name} but not on disk")
        elif _sha256(path) != row["sha256"]:
            problems.append(f"{key}: sha256 differs from {manifest.name}")
    for key in sorted(set(on_disk) - seen):
        problems.append(f"{key}: on disk but not listed in {manifest.name}")
    return problems
