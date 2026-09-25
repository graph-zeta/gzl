# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

"""One-shot script: dump the TFIM softcore tw<=2 graph topologies from
the per-order H5 fixtures into a single compressed-npz file vendored
in this directory.

Run from the repo root with the H5 fixture directory available:

    GZ_TFIM_H5_DIR=/path/to/softcore python tests/fixtures/_dump_tfim_graphs.py

Produces ``tests/fixtures/tfim_softcore_graphs.npz`` containing both
the 0qp orders 2-10 set (401 graphs) and the 1qp orders 1-7 set
(280 graphs), classified at sigma_max=4 against
``graph_from_edges_uniform`` so that K_4-minor / treewidth>2 graphs
are filtered out.

The vendored npz is what the smoothness tests in
``tests/test_sigma_max_smoothness.py`` actually load — the H5 path
is only needed to (re)generate it.

For the *complete* graph set (0qp orders 1-13, 1qp orders 1-11, all
~31 k graphs including tw>2), see the companion dumper at
``tools/dump_full_graph_set.py``.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gzl import (
    graph_from_edges_uniform,
    NotTreewidthTwoError,
    PrefactorSingularityError,
)


_H5_DIR_ENV = "GZ_TFIM_H5_DIR"
_OUT_PATH   = Path(__file__).parent / "tfim_softcore_graphs.npz"


def _dump_one_kind(
    h5_dir: Path,
    kind: str,                       # "0qp" or "1qp"
    orders: list[int],
    A: np.ndarray,
    n_points: int,
    nu_probe: float,
) -> tuple[list[tuple[int, str, int, int, np.ndarray]], int]:
    """Walk the per-order H5 files for one qp-kind and collect tw<=2
    graphs.  Returns a list of (order, name, s, t, edges_flat) tuples
    and the number of K_4-minor graphs filtered."""
    rows: list[tuple[int, str, int, int, np.ndarray]] = []
    n_filtered = 0
    for order in orders:
        fp = h5_dir / f"graph_contributions_softcore_TFIM_{kind}_O{order}.h5"
        if not fp.exists():
            print(f"  {kind} O{order}: missing")
            continue
        with h5py.File(fp, "r") as f:
            okey = next(iter(f.keys()))
            order_n = 0
            order_filtered = 0
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
                try:
                    graph_from_edges_uniform(
                        edges_flat, nu_probe, A, n_points,
                        s=s, t=t, sigma_max=4.0,
                    )
                except NotTreewidthTwoError:
                    order_filtered += 1
                    continue
                except PrefactorSingularityError:
                    pass
                rows.append((order, str(gname), s, t, edges_flat))
                order_n += 1
            n_filtered += order_filtered
            print(f"  {kind} O{order}: {order_n} kept, {order_filtered} filtered")
    return rows, n_filtered


def main() -> None:
    env = os.environ.get(_H5_DIR_ENV)
    if not env:
        sys.exit(f"set {_H5_DIR_ENV} to the H5 fixture directory")
    h5_dir = Path(env).expanduser()
    if not h5_dir.is_dir():
        sys.exit(f"{h5_dir} is not a directory")

    A        = np.array([[1.0]])
    n_points = 16             # used only for the tw filter probe
    nu_probe = 2.5 + np.pi / 300

    print("dumping 0qp orders 2..10 ...")
    rows0, n_filt0 = _dump_one_kind(
        h5_dir, "0qp", list(range(2, 11)), A, n_points, nu_probe,
    )
    print(f"  -> {len(rows0)} 0qp graphs, {n_filt0} K_4-minor filtered\n")

    print("dumping 1qp orders 1..7 ...")
    rows1, n_filt1 = _dump_one_kind(
        h5_dir, "1qp", list(range(1, 8)), A, n_points, nu_probe,
    )
    print(f"  -> {len(rows1)} 1qp graphs, {n_filt1} K_4-minor filtered\n")

    rows = (
        [(0, *r) for r in rows0]
      + [(1, *r) for r in rows1]
    )
    n = len(rows)

    qp     = np.array([r[0] for r in rows], dtype=np.int8)
    order  = np.array([r[1] for r in rows], dtype=np.int8)
    name   = np.array([r[2] for r in rows], dtype="U16")
    s_arr  = np.array([r[3] for r in rows], dtype=np.int32)
    t_arr  = np.array([r[4] for r in rows], dtype=np.int32)

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
