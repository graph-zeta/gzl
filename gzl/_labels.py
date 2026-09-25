# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
r"""Vertex labels are names, and the vertex set is the edge support.

Every public edge-list entry point normalises labels through
:func:`relabel_to_support` before any engine work: the vertices of the
graph are exactly the labels that appear in ``edges``.  A gap label is
not a vertex — an edge list cannot express an isolated vertex, and the
infinite-lattice zeta of a graph that has one DIVERGES (the escaping
cluster cut of an isolated vertex is 0, never ``> d``), so any finite
number an engine returns for it is an artifact of that engine's
truncation: the torus's factor is ``n**d`` per gap, the box's is
``(2L+1)**d``, and the two disagree with each other as well as with the
divergent object they approximate.  Before this module, the four
engines split three ways on the same sparsely-labelled input — hybrid
and slab used the edge support, the tensor and the box multiplied in
their truncation volumes per gap, and the router refused with a
``DisconnectedGraphError`` naming a disconnection that is not there.

The relabelling is order-preserving (sorted support -> ``0..V-1``), so
a sparse graph and its hand-relabelled contiguous twin produce
IDENTICAL downstream arrays — bit-equal results, not merely close.  An
already contiguous input is returned unchanged, as the same object:
every previously reachable value is bit-identical by construction.

A ``networkx`` input is different in kind: it CAN express an isolated
node, and the frontend's ``DisconnectedGraphError`` is then the correct
refusal of a genuinely divergent sum (see
:func:`gzl.frontend.evaluate_graph`).
"""
from __future__ import annotations

import numpy as np

__all__ = ["relabel_to_support", "edge_array", "vertex_label"]


def relabel_to_support(edges_arr, refs=None):
    """Compress edge labels to ``0..V-1`` by sorted order of the support.

    Parameters
    ----------
    edges_arr : ndarray (E, 2), integer
        Validated edge list (non-empty, shape-checked by the caller).
    refs : dict[str, int | None] | None
        Caller vertex references (``source``, ``terminal``, ``root``,
        ...) to remap through the same map.  ``None`` values pass
        through untouched.  A reference whose label appears in no edge
        raises ``ValueError`` naming the reference — such a label is
        not a vertex of the graph.

    Returns
    -------
    (edges2, refs2) : the relabelled edge array and remapped refs.
        When the labels are already exactly ``0..V-1``, ``edges2 is
        edges_arr`` (the identical object) and refs are only
        membership-checked.
    """
    refs = {} if refs is None else dict(refs)
    present = np.unique(np.asarray(edges_arr, dtype=np.int64))
    if present.size and int(present[0]) < 0:
        raise ValueError(
            f"vertex labels must be non-negative integers; found "
            f"{int(present[0])}"
        )
    v_count = int(present.size)

    if v_count == 0:
        # No edges, no vertices: pass through unchanged so the caller's
        # own empty-input diagnostics fire (any non-None ref is still
        # not a vertex and raises through check() below).
        for name, label in refs.items():
            if label is not None:
                raise ValueError(
                    f"{name}={int(label)} is not a vertex of this graph: "
                    f"the edge list is empty"
                )
        return edges_arr, refs

    def check(name, label):
        if label is None:
            return None
        lab = int(label)
        i = int(np.searchsorted(present, lab))
        if i >= v_count or int(present[i]) != lab:
            raise ValueError(
                f"{name}={lab} is not a vertex of this graph: the label "
                f"appears in no edge (the vertex set is the edge support; "
                f"an isolated vertex has a divergent lattice sum and "
                f"cannot be expressed through an edge list)"
            )
        return i

    contiguous = int(present[0]) == 0 and int(present[-1]) == v_count - 1
    if contiguous:
        for name, label in refs.items():
            check(name, label)
        return edges_arr, refs

    mapped = {name: check(name, label) for name, label in refs.items()}
    edges2 = np.searchsorted(present, np.asarray(edges_arr, dtype=np.int64))
    return edges2, mapped


def _integral(arr, what, where):
    """``arr`` as int64, refusing what is not a whole number.

    Integers pass; a float passes only if it is finite and integral,
    because ``np.asarray(..., dtype=int)`` would truncate ``0.5`` to 0
    and evaluate a different graph without a word.  Booleans, complex
    numbers, strings and objects raise ``TypeError``.
    """
    if arr.dtype.kind == "b":
        raise TypeError(f"{where}: {what} must be integers, not booleans")
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64, copy=False)
    if arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)) or np.any(arr != np.rint(arr)):
            bad = arr[~np.isfinite(arr) | (arr != np.rint(arr))].reshape(-1)
            raise ValueError(
                f"{where}: {what} must be whole numbers; got {bad[0]!r}"
            )
        if arr.size and np.max(np.abs(arr)) >= 2.0 ** 63:
            raise ValueError(f"{where}: {what} out of the int64 range")
        return arr.astype(np.int64)
    raise TypeError(
        f"{where}: {what} must be integers; got dtype {arr.dtype}"
    )


def edge_array(edges, where: str = "edges") -> np.ndarray:
    """A public edge list as an ``(E, 2)`` int64 array, checked.

    ``edges`` must be array-like of shape ``(E, 2)``, or the flat form
    ``(u0, v0, u1, v1, ...)`` of even length, with whole-number labels
    (see :func:`_integral`); an empty sequence is the empty graph, shape
    ``(0, 2)``.  Any other shape raises ``ValueError``: an ``(E, 3)`` list
    used to be read as its first two columns.  Negative labels are left
    to the caller, whose error names the offending vertex.
    """
    try:
        arr = np.asarray(edges)
    except ValueError as exc:            # ragged nesting
        raise ValueError(
            f"{where}: the edge list must be an (E, 2) array of vertex "
            f"labels; {exc}"
        ) from exc
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if arr.ndim == 1 and arr.size % 2 == 0:
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"{where}: the edge list must have shape (E, 2), one row per "
            f"edge, or be its flat form of even length; got shape "
            f"{arr.shape}"
        )
    return _integral(arr, "vertex labels", where)


def vertex_label(label, name: str, where: str = "") -> int:
    """A single vertex reference (``source``, ``terminal``) as an ``int``.

    Whole numbers only, as for the edge labels: ``source=0.5`` used to be
    read as vertex 0.  A NumPy integer or an integral float is accepted.
    """
    prefix = f"{where}: " if where else ""
    if isinstance(label, np.ndarray) and label.ndim == 0:
        label = label[()]
    if isinstance(label, (bool, np.bool_)):
        raise TypeError(f"{prefix}{name} must be an integer vertex label, "
                        f"not a boolean")
    if isinstance(label, (int, np.integer)):
        return int(label)
    if isinstance(label, (float, np.floating)):
        if np.isfinite(label) and float(label).is_integer():
            return int(label)
        raise ValueError(f"{prefix}{name} must be a whole number; "
                         f"got {label!r}")
    raise TypeError(f"{prefix}{name} must be an integer vertex label; "
                    f"got {label!r}")
