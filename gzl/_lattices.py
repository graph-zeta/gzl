# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""The Bravais lattices that have a name.

Wherever the library takes a lattice matrix ``A`` -- whose columns are
the primitive vectors -- it also takes one of four names:

* ``"chain"`` -- :math:`\mathbb{Z}`, ``[[1.0]]``;
* ``"square"`` -- :math:`\mathbb{Z}^2`, ``eye(2)``;
* ``"triangular"`` -- the primitive vectors :math:`(1, 0)` and
  :math:`(1/2, \sqrt{3}/2)`, the 60-degree cell;
* ``"cubic"`` -- :math:`\mathbb{Z}^3`, ``eye(3)``.

A name is exactly the matrix, to the last bit: every spelling of
:math:`\sqrt{3}/2` this repository uses is ``0x1.bb67ae8584caap-1``, the
one :data:`LATTICES` stores, so a name and its matrix share every cache
key and give bit-identical values.  (``np.cos(np.pi / 6)`` is one ulp
off and is never how the entry is built.)

The orientation of the triangular cell is load-bearing beyond that: its
zone corner is the fractional momentum :math:`(1/3, -1/3)`, which is
what the K-point comparisons against Monte Carlo data use.  In the other
common convention, columns :math:`(1, 0)` and :math:`(-1/2, \sqrt{3}/2)`,
that corner is :math:`(1/3, 1/3)` instead.

:func:`_resolve_lattice` is the one rule, in the spirit of
:func:`gzl._data._resolve_corpus`: a name becomes its matrix,
anything else is passed through untouched, so every call site keeps its
own ``np.asarray(..., dtype=float)`` line and matrix arguments reach the
engines exactly as they always did.
"""

from __future__ import annotations

import numpy as np

__all__: list[str] = []

#: The named Bravais lattices, as rows of the matrix whose COLUMNS are the
#: primitive vectors: a string that is exactly a key here means this matrix
#: wherever an ``A`` is taken (see :func:`_resolve_lattice`).
LATTICES = {
    "chain": ((1.0,),),
    "square": ((1.0, 0.0),
               (0.0, 1.0)),
    "triangular": ((1.0, 0.5),
                   (0.0, float(np.sqrt(3.0)) / 2.0)),
    "cubic": ((1.0, 0.0, 0.0),
              (0.0, 1.0, 0.0),
              (0.0, 0.0, 1.0)),
}


def _resolve_lattice(A):
    """The matrix a lattice argument means: a named lattice, or ``A`` itself.

    A ``str`` exactly equal to a key of :data:`LATTICES` -- ``"chain"``,
    ``"square"``, ``"triangular"``, ``"cubic"``; lowercase, no other
    spelling -- becomes that matrix, a fresh writable C-contiguous
    ``float64`` array on every call (epsteinlib refuses a read-only
    buffer, and a shared array a caller could mutate would be worse).
    Any other ``str`` raises ``ValueError`` listing the names: no string
    was ever a valid ``A``.  Anything else -- an array, a nested list, a
    tuple -- comes back as the same object, unchecked, for the call
    site's own normalisation to handle as it always did.
    """
    if isinstance(A, str):
        if A not in LATTICES:
            raise ValueError(
                f"Unknown lattice name: {A!r}.  Named lattices are "
                + ", ".join(repr(name) for name in LATTICES)
                + "; otherwise pass the lattice matrix itself, whose columns "
                  "are the primitive vectors."
            )
        return np.array(LATTICES[A], dtype=float)
    return A


def lattice_matrix(A, where: str = "A") -> np.ndarray:
    """The lattice argument ``A`` as a checked ``(d, d)`` float64 matrix.

    A name becomes its matrix (:func:`_resolve_lattice`).  Anything else
    must convert to a finite, square, non-singular real matrix with
    ``d >= 1``, and otherwise raises ``ValueError`` (``TypeError`` for a
    value that is not numeric at all), naming ``where``.  A singular
    matrix spans no lattice, and before this check it gave a NaN from
    epsteinlib, a ``LinAlgError`` or an ``IndexError`` depending on the
    route.  The rank is numerical: a cell whose singular values differ by
    more than about ``1 / (d * eps)`` is refused as well.

    The returned array is ``np.asarray(_resolve_lattice(A), dtype=float)``,
    exactly what every call site built before, so a valid matrix reaches
    the engines bit for bit as it did.
    """
    try:
        M = np.asarray(_resolve_lattice(A), dtype=float)
    except TypeError as exc:
        raise TypeError(
            f"{where}: A must be a lattice name or a real (d, d) matrix; "
            f"got {A!r}"
        ) from exc
    except ValueError as exc:
        if isinstance(A, str):
            raise
        raise ValueError(
            f"{where}: A must be a lattice name or a real (d, d) matrix; "
            f"got {A!r}"
        ) from exc
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] < 1:
        raise ValueError(
            f"{where}: A must be a square (d, d) matrix whose columns are "
            f"the primitive vectors, or a lattice name; got shape "
            f"{M.shape}.  The chain is [[1.0]] or 'chain'."
        )
    if not np.all(np.isfinite(M)):
        raise ValueError(f"{where}: A must be finite; got {M.tolist()}")
    if np.linalg.matrix_rank(M) < M.shape[0]:
        raise ValueError(
            f"{where}: A = {M.tolist()} is singular, so its columns span no "
            f"lattice."
        )
    return M
