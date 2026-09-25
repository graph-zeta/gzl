# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

"""Consolidate the per-order TFIM softcore HDF5 files into one corpus
file per quasiparticle sector.

This is how ``gzl/data/tfim_softcore_corpus_{0,1}qp.npz`` were built.
The input, one ``graph_contributions_softcore_TFIM_{n_qp}qp_O{n}.h5`` per
order from the pCUT graph generation, is not distributed, so the script
documents the provenance of the shipped corpora rather than being
something a user of gzl needs to run.

For each n_qp class (0 and 1) it writes two layouts that
:func:`gzl.compute_series_coefficients` reads interchangeably:

* ``tfim_softcore_corpus_{n_qp}qp.npz``, the shipped flat / CSR layout::

      orders, order_off          order boundaries in graph-index space
      order, graph_id, prefactor per graph
      hopping                    (n_graphs, 2) terminal pair, zeros for 0qp
      edges_off                  (n_graphs + 1,) offsets into edges_flat
      edges_flat, multiplicities (sum_E, 2) and (sum_E,)

  The edges of graph i are ``edges_flat[edges_off[i]:edges_off[i + 1]]``.

* ``tfim_softcore_corpus_{n_qp}qp.h5``, the per-graph subgroup layout
  ``/O{n}/graph_<id>/{edges, multiplicities, prefactor[, hopping]}``
  with gzip-compressed datasets.

Run from the repository root::

    python tools/consolidate_corpus.py --source <DIR>
        [--out <DIR>] [--gzip-level <N>] [--qp 0|1]

``--source`` defaults to the ``GZ_TFIM_H5_DIR`` environment variable.
Without ``--out`` the NPZ corpora are written into ``gzl/data`` (pinned
by SHA-256 in its ``PROVENANCE.csv``, so a regenerated corpus goes into
the same commit as its updated row) and the H5 files into the current
directory.
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np


CORPUS_VERSION = "1.0.0"
PER_ORDER_RE = re.compile(
    r"^graph_contributions_softcore_TFIM_(?P<qp>[01])qp_O(?P<order>\d+)\.h5$"
)


# ---------------------------------------------------------------------------
# Source-dir resolution
# ---------------------------------------------------------------------------

def _resolve_source_dir(arg: str | None) -> Path:
    src = arg if arg is not None else os.environ.get("GZ_TFIM_H5_DIR")
    if src is None:
        raise FileNotFoundError(
            "Pass --source <DIR> or set GZ_TFIM_H5_DIR to the directory "
            "holding the per-order H5 files."
        )
    p = Path(src).expanduser()
    if not p.is_dir():
        raise FileNotFoundError(f"source directory not found: {p}")
    return p


def _enumerate_per_order_files(
    source: Path, n_qp: int
) -> list[tuple[int, Path]]:
    """Return [(order, path)] sorted by order, for the given qp class."""
    found: list[tuple[int, Path]] = []
    for p in sorted(source.glob(f"graph_contributions_softcore_TFIM_{n_qp}qp_O*.h5")):
        m = PER_ORDER_RE.match(p.name)
        if m is None:
            continue
        if int(m.group("qp")) != n_qp:
            continue
        found.append((int(m.group("order")), p))
    found.sort()
    return found


# ---------------------------------------------------------------------------
# Copying / compression
# ---------------------------------------------------------------------------

def _natural_graph_index(name: str) -> int:
    """Sort key for graph names like 'graph_0', 'graph_10' so that 'graph_2'
    sorts before 'graph_10' (numeric ordering)."""
    m = re.match(r"^graph_(\d+)$", name)
    return int(m.group(1)) if m else -1


def _copy_graph_subgroup(
    src_grp: h5py.Group, dst_parent: h5py.Group, name: str, gzip_level: int
) -> None:
    """Re-create one graph_<id> subgroup under dst_parent with compressed
    datasets, keeping the per-graph layout that :mod:`gzl.series` reads."""
    dst_grp = dst_parent.create_group(name)
    for ds_name, ds in src_grp.items():
        data = ds[()]
        if data.shape == ():
            # Scalars: HDF5 forbids the gzip filter; store raw.
            dst_grp.create_dataset(ds_name, data=data)
        else:
            dst_grp.create_dataset(
                ds_name,
                data=data,
                compression="gzip",
                compression_opts=gzip_level,
                shuffle=True,
            )
    for k, v in src_grp.attrs.items():
        dst_grp.attrs[k] = v


def _consolidate_h5(
    n_qp: int,
    per_order: list[tuple[int, Path]],
    out_path: Path,
    gzip_level: int,
    source_subdir: str,
) -> tuple[int, int]:
    """Write the consolidated H5 (per-graph subgroup layout) for one qp class.

    Returns (n_graphs_total, n_orders).
    """
    print(f"  → {out_path.name}  [H5, per-graph subgroup layout]")
    orders = [o for o, _ in per_order]
    n_graphs_total = 0

    with h5py.File(out_path, "w", libver="latest") as out_h5:
        out_h5.attrs["corpus_version"] = CORPUS_VERSION
        out_h5.attrs["generation_date"] = datetime.date.today().isoformat()
        out_h5.attrs["n_qp"] = n_qp
        out_h5.attrs["n_orders"] = len(orders)
        out_h5.attrs["orders_min"] = min(orders) if orders else -1
        out_h5.attrs["orders_max"] = max(orders) if orders else -1
        out_h5.attrs["source_subdir"] = source_subdir
        out_h5.attrs["compression"] = f"gzip-{gzip_level}+shuffle"
        out_h5.attrs["layout"] = "per_graph_subgroup"
        out_h5.attrs["schema"] = (
            "/O{n}/graph_<id>/{edges, multiplicities, prefactor"
            + (", hopping" if n_qp == 1 else "")
            + "}"
        )

        for order, src_path in per_order:
            with h5py.File(src_path, "r") as src_h5:
                src_top_keys = list(src_h5.keys())
                if len(src_top_keys) != 1 or src_top_keys[0] != f"O{order}":
                    raise RuntimeError(
                        f"Expected one top-level group named 'O{order}' in "
                        f"{src_path}, found {src_top_keys}"
                    )
                src_order = src_h5[f"O{order}"]
                dst_order = out_h5.create_group(f"O{order}")
                graph_names = sorted(src_order.keys(), key=_natural_graph_index)
                dst_order.attrs["order"] = order
                dst_order.attrs["n_graphs"] = len(graph_names)
                for k, v in src_order.attrs.items():
                    dst_order.attrs[k] = v

                for graph_name in graph_names:
                    _copy_graph_subgroup(
                        src_order[graph_name], dst_order, graph_name, gzip_level
                    )

                n_graphs_total += len(graph_names)
                print(f"     O{order:>2}: {len(graph_names):>5} graphs")

        out_h5.attrs["n_graphs_total"] = n_graphs_total

    return n_graphs_total, len(orders)


def _consolidate_npz(
    n_qp: int,
    per_order: list[tuple[int, Path]],
    out_path: Path,
    source_subdir: str,
) -> tuple[int, int]:
    """Write the consolidated NPZ (flat / CSR layout) for one qp class.

    Returns (n_graphs_total, n_orders).
    """
    print(f"  → {out_path.name}  [NPZ, flat/CSR layout]")

    edges_chunks: list[np.ndarray] = []
    mult_chunks: list[np.ndarray] = []
    prefactor_list: list[float] = []
    hopping_list: list[np.ndarray] = []
    order_per_graph: list[int] = []
    graph_id_list: list[str] = []

    orders_present: list[int] = []
    order_off: list[int] = [0]  # boundaries between orders in graph-index space
    edges_off: list[int] = [0]  # CSR offsets into edges_flat
    cum_E = 0
    cum_G = 0

    for order, src_path in per_order:
        with h5py.File(src_path, "r") as src_h5:
            src_order = src_h5[f"O{order}"]
            graph_names = sorted(src_order.keys(), key=_natural_graph_index)
            for gname in graph_names:
                g = src_order[gname]
                e = np.asarray(g["edges"], dtype=np.int64)
                m = np.asarray(g["multiplicities"], dtype=np.int64)
                if e.ndim != 2 or e.shape[1] != 2:
                    raise RuntimeError(
                        f"{gname}: expected edges shape (E, 2), got {e.shape}"
                    )
                if m.shape != (e.shape[0],):
                    raise RuntimeError(
                        f"{gname}: edges has E={e.shape[0]} but multiplicities "
                        f"has shape {m.shape}"
                    )
                edges_chunks.append(e)
                mult_chunks.append(m)
                prefactor_list.append(float(np.asarray(g["prefactor"])))
                if n_qp == 1:
                    hopping_list.append(np.asarray(g["hopping"], dtype=np.int64))
                else:
                    hopping_list.append(np.zeros(2, dtype=np.int64))
                order_per_graph.append(order)
                graph_id_list.append(gname)
                cum_E += e.shape[0]
                edges_off.append(cum_E)
            cum_G += len(graph_names)
            orders_present.append(order)
            order_off.append(cum_G)
            print(f"     O{order:>2}: {len(graph_names):>5} graphs, "
                  f"running totals: G={cum_G}, sum_E={cum_E}")

    edges_flat = (
        np.concatenate(edges_chunks, axis=0).astype(np.int64)
        if edges_chunks
        else np.zeros((0, 2), dtype=np.int64)
    )
    multiplicities = (
        np.concatenate(mult_chunks).astype(np.int64)
        if mult_chunks
        else np.zeros((0,), dtype=np.int64)
    )
    hopping = (
        np.stack(hopping_list, axis=0).astype(np.int64)
        if hopping_list
        else np.zeros((0, 2), dtype=np.int64)
    )

    np.savez_compressed(
        out_path,
        # Corpus metadata as 0-d arrays (NPZ doesn't carry attrs).
        corpus_version=np.array(CORPUS_VERSION),
        generation_date=np.array(datetime.date.today().isoformat()),
        source_subdir=np.array(source_subdir),
        layout=np.array("flat_csr"),
        qp=np.array(n_qp, dtype=np.int8),
        # Order indexing.
        orders=np.array(orders_present, dtype=np.int64),
        order_off=np.array(order_off, dtype=np.int64),
        # Per-graph metadata.
        order=np.array(order_per_graph, dtype=np.int64),
        graph_id=np.array(graph_id_list),
        prefactor=np.asarray(prefactor_list, dtype=np.float64),
        hopping=hopping,
        # CSR-packed edge data.
        edges_off=np.array(edges_off, dtype=np.int64),
        edges_flat=edges_flat,
        multiplicities=multiplicities,
    )

    return cum_G, len(orders_present)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=None,
        help="Directory containing graph_contributions_softcore_TFIM_*qp_O*.h5. "
        "Defaults to GZ_TFIM_H5_DIR.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for both layouts.  Default: the NPZ corpora into "
        "gzl/data (shipped), the H5 into the current directory.",
    )
    parser.add_argument(
        "--gzip-level",
        type=int,
        default=4,
        help="HDF5 gzip filter level (1-9, default 4).  Higher = smaller "
        "but slower to write; reads are fast at any level.",
    )
    parser.add_argument(
        "--qp",
        action="append",
        type=int,
        choices=[0, 1],
        default=None,
        help="Restrict to a specific qp class (0 or 1).  Repeat for multiple. "
        "Default: both 0qp and 1qp.",
    )
    parser.add_argument(
        "--no-h5", action="store_true",
        help="Skip the consolidated H5 (per-graph subgroup layout) output.",
    )
    parser.add_argument(
        "--no-npz", action="store_true",
        help="Skip the consolidated NPZ (flat / CSR layout) output.",
    )
    args = parser.parse_args(argv)
    if args.no_h5 and args.no_npz:
        parser.error("--no-h5 and --no-npz cannot both be set.")

    source = _resolve_source_dir(args.source)
    # The corpora record the name of the source directory only: its full
    # path would describe the machine that built them.
    source_name = source.resolve().name
    # The NPZ corpora ship inside the package (gzl/data, pinned by
    # SHA-256 in its PROVENANCE.csv -- regenerating one means updating that
    # row in the same commit); the H5 layout is not distributed.  --out puts
    # both in one directory.
    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out).expanduser() if args.out else Path.cwd()
    npz_dir = out_dir if args.out else repo / "gzl" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)
    qp_classes = sorted(set(args.qp)) if args.qp else [0, 1]

    print(f"Source dir : {source}")
    print(f"Output dir : {out_dir} (H5), {npz_dir} (NPZ)")
    print(f"Gzip level : {args.gzip_level}")
    print(f"qp classes : {qp_classes}")
    print()

    summary: list[tuple[int, int, int, int, int, int]] = []
    for n_qp in qp_classes:
        per_order = _enumerate_per_order_files(source, n_qp)
        if not per_order:
            print(f"  ! no per-order files for {n_qp}qp under {source} — skipping")
            continue

        print(f"Consolidating {n_qp}qp ({len(per_order)} orders, "
              f"{per_order[0][0]} … {per_order[-1][0]})")
        src_total = sum(p.stat().st_size for _, p in per_order)
        h5_size = 0
        npz_size = 0
        n_graphs = 0
        n_orders = 0

        if not args.no_h5:
            h5_path = out_dir / f"tfim_softcore_corpus_{n_qp}qp.h5"
            n_graphs, n_orders = _consolidate_h5(
                n_qp, per_order, h5_path,
                gzip_level=args.gzip_level,
                source_subdir=source_name,
            )
            h5_size = h5_path.stat().st_size

        if not args.no_npz:
            npz_path = npz_dir / f"tfim_softcore_corpus_{n_qp}qp.npz"
            n_graphs, n_orders = _consolidate_npz(
                n_qp, per_order, npz_path,
                source_subdir=source_name,
            )
            npz_size = npz_path.stat().st_size

        summary.append((n_qp, n_orders, n_graphs, src_total, h5_size, npz_size))

        print(f"     → {n_graphs} graphs across {n_orders} orders")
        print(f"     → raw per-order H5 input : {src_total / 1024 / 1024:7.2f} MB")
        if h5_size:
            print(f"     → consolidated H5        : {h5_size / 1024 / 1024:7.2f} MB "
                  f"({h5_size / src_total * 100:.1f}%)")
        if npz_size:
            print(f"     → consolidated NPZ       : {npz_size / 1024 / 1024:7.2f} MB "
                  f"({npz_size / src_total * 100:.1f}%)")
        print()

    if summary:
        print("Summary:")
        print(f"  {'qp':>3} {'orders':>7} {'graphs':>8} "
              f"{'raw MB':>9} {'h5 MB':>8} {'h5 %':>6} "
              f"{'npz MB':>8} {'npz %':>6}")
        for n_qp, n_orders, n_graphs, raw, h5_sz, npz_sz in summary:
            h5_pct = h5_sz / raw * 100 if h5_sz and raw else 0.0
            npz_pct = npz_sz / raw * 100 if npz_sz and raw else 0.0
            print(f"  {n_qp:>3} {n_orders:>7d} {n_graphs:>8d} "
                  f"{raw / 1024 / 1024:>9.2f} "
                  f"{h5_sz / 1024 / 1024 if h5_sz else 0:>8.2f} "
                  f"{h5_pct:>5.1f}% "
                  f"{npz_sz / 1024 / 1024 if npz_sz else 0:>8.2f} "
                  f"{npz_pct:>5.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
