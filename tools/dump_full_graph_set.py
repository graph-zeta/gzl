# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

"""One-shot script: dump the *complete* TFIM softcore graph set, 0qp
orders 2-13 and 1qp orders 1-11, into a single compressed-npz file.

This is the companion to ``tests/fixtures/_dump_tfim_graphs.py``: where
the latter snapshots just the 681 tw<=2 graphs used by the smoothness
regression test (8 KB), this dumper preserves the entire ~31 k-graph
topology corpus (~1-2 MB compressed), so the whole topology set is
available without the original per-order H5 files, which are not
distributed.

Run from the repo root with the H5 directory available:

    GZ_TFIM_H5_DIR=/path/to/softcore python tools/dump_full_graph_set.py

Produces ``gzl/data/full_graph_topologies.npz`` -- shipped package
data, pinned by SHA-256 in ``gzl/data/PROVENANCE.csv``; update that
row in the same commit as a regenerated snapshot.

The dump is pure topology — no classification at dump time, since
classifying ~31 k graphs takes far longer than just writing them.
Consumers who want the tw<=2 / tw>2 split should run
:func:`gzl.graph_from_edges_uniform` on each graph at load
time (catching :class:`gzl.NotTreewidthTwoError`) and cache
the result locally if needed.

Layout of the dumped npz:

    qp           : (N,)   int8     0 for 0qp, 1 for 1qp
    order        : (N,)   int8     order O of each graph
    name         : (N,)   <U16     "graph_<idx>" inside the H5 file
    s, t         : (N,)   int32    terminal pair (always 0,0 for 0qp)
    edges_off    : (N+1,) int64    offsets into edges_flat per graph
    edges_flat   : (M, 2) int64    all flat edge lists concatenated

For graph i, ``edges_flat[edges_off[i]:edges_off[i+1]]`` reconstructs
its flat edge list (already-expanded multigraph: ``np.repeat(edges,
multiplicities, axis=0)``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np

_H5_DIR_ENV = "GZ_TFIM_H5_DIR"
_OUT_PATH   = Path(__file__).resolve().parents[1] / "gzl" / "data" / "full_graph_topologies.npz"

# Full available range in the upstream H5 corpus.
_ORDERS_0QP = list(range(1, 14))    # 1..13
_ORDERS_1QP = list(range(1, 12))    # 1..11


def _dump_one_kind(
    h5_dir: Path,
    kind: str,                       # "0qp" or "1qp"
    orders: list[int],
) -> list[tuple[int, str, int, int, np.ndarray]]:
    """Walk the per-order H5 files for one qp-kind and collect *all*
    graphs.  Pure I/O — no classification."""
    rows: list[tuple[int, str, int, int, np.ndarray]] = []
    for order in orders:
        fp = h5_dir / f"graph_contributions_softcore_TFIM_{kind}_O{order}.h5"
        if not fp.exists():
            print(f"  {kind} O{order}: missing")
            continue
        with h5py.File(fp, "r") as f:
            okey = next(iter(f.keys()))
            n_tot = 0
            for gname in f[okey]:
                g_data = f[okey][gname]
                edges  = g_data["edges"][...]
                mults  = g_data["multiplicities"][...]
                hop    = (
                    g_data["hopping"][...] if "hopping" in g_data else None
                )
                edges_flat = np.repeat(edges, mults, axis=0).astype(np.int64)
                s, t = (
                    (int(hop[0]), int(hop[1])) if hop is not None else (0, 0)
                )
                rows.append(
                    (order, str(gname), s, t, edges_flat)
                )
                n_tot += 1
            print(f"  {kind} O{order}: {n_tot} graphs")
    return rows


def main() -> None:
    env = os.environ.get(_H5_DIR_ENV)
    if not env:
        sys.exit(f"set {_H5_DIR_ENV} to the H5 fixture directory")
    h5_dir = Path(env).expanduser()
    if not h5_dir.is_dir():
        sys.exit(f"{h5_dir} is not a directory")

    print(f"dumping 0qp orders {_ORDERS_0QP[0]}..{_ORDERS_0QP[-1]} ...")
    rows0 = _dump_one_kind(h5_dir, "0qp", _ORDERS_0QP)
    print(f"  -> {len(rows0)} 0qp graphs total\n")

    print(f"dumping 1qp orders {_ORDERS_1QP[0]}..{_ORDERS_1QP[-1]} ...")
    rows1 = _dump_one_kind(h5_dir, "1qp", _ORDERS_1QP)
    print(f"  -> {len(rows1)} 1qp graphs total\n")

    rows = (
        [(0, *r) for r in rows0]
      + [(1, *r) for r in rows1]
    )
    n = len(rows)

    qp       = np.array([r[0] for r in rows], dtype=np.int8)
    order    = np.array([r[1] for r in rows], dtype=np.int8)
    name     = np.array([r[2] for r in rows], dtype="U16")
    s_arr    = np.array([r[3] for r in rows], dtype=np.int32)
    t_arr    = np.array([r[4] for r in rows], dtype=np.int32)

    edges_lists = [r[5] for r in rows]
    sizes       = np.array([len(e) for e in edges_lists], dtype=np.int64)
    edges_off   = np.concatenate([[0], np.cumsum(sizes)])
    edges_flat  = np.concatenate(edges_lists, axis=0).astype(np.int64)

    np.savez_compressed(
        _OUT_PATH,
        qp=qp,
        order=order,
        name=name,
        s=s_arr,
        t=t_arr,
        edges_off=edges_off,
        edges_flat=edges_flat,
    )

    raw_size = _OUT_PATH.stat().st_size
    print(f"wrote {_OUT_PATH} ({raw_size:,} bytes, {n} graphs)")


if __name__ == "__main__":
    main()
