# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
r"""Freeze exact engine values from the current tree.

Run from the repo root:

    python tests/data/_generate_engine_reference.py

Writes ``engine_reference_values.json``: one entry per (engine, core,
cell, d, n, call-path) with the value stored as ``float.hex()`` -- the
exact bit pattern, not a decimal rounding.  ``tests/test_engine_reference.py``
replays every entry and compares BIT-EXACTLY.

WHY THIS EXISTS.  Engine unification (folding slab_zeta into
hybrid._dense_core as a pinning mode, then further) must be
value-preserving: bit-identical wherever the arithmetic is literally the
same, round-off only where a documented reassociation occurs.  "It
should be the same" is not an argument -- this repository shipped a
silent 4.9x accuracy regression from exactly that reasoning.  The
reference set is generated ONCE on the pre-fold tree and committed;
regenerating it on a changed tree and calling the test green is the one
way to defeat it, so: NEVER regenerate and commit in the same change
that alters an engine.  A legitimate regeneration (a deliberate,
documented value change) must be its own commit citing the change that
moved each value.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gzl import (                                       # noqa: E402
    direct_sum_extrapolated,
    evaluate_graph,
    graph_zeta_general_at_zero,
    hybrid_zeta,
    slab_zeta,
)
from tests._env_gate import fingerprint                 # noqa: E402

CORES = {
    "K4":    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    "prism": [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
              (0, 3), (1, 4), (2, 5)],
    "K33":   [(i, j) for i in (0, 1, 2) for j in (3, 4, 5)],
    "K5":    [(i, j) for i in range(5) for j in range(i + 1, 5)],
    "K5-e":  [e for e in [(i, j) for i in range(5)
                          for j in range(i + 1, 5)] if e != (0, 4)],
    "V6E11": [(0, 1), (0, 4), (1, 2), (1, 3), (1, 5), (2, 3),
              (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)],
}

# Cells chosen so every failure mode the engines have actually shown is
# armed: a cross term (the even-n reflection trap), anisotropy (partial
# orbit groups), and both parities of n.
CELLS = {
    1: [("unit", [[1.0]]), ("scaled", [[1.7]])],
    2: [("square", [[1.0, 0.0], [0.0, 1.0]]),
        ("triangular", [[1.0, 0.5], [0.0, 0.8660254037844386]]),
        ("sheared", [[1.0, 0.37], [0.0, 1.13]])],
    3: [("cubic", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ("tetragonal", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.3]]),
        ("sheared", [[1.0, 0.3, 0.0], [0.0, 1.1, 0.2], [0.0, 0.0, 0.9]])],
}
NS = {1: (7, 8), 2: (5, 6), 3: (4, 5)}


def main() -> None:
    out: dict = {"_meta": {
        "generated_on": "the tree before slab_zeta was folded into "
                        "hybrid._dense_core (pre-unification reference)",
        "frozen_on": fingerprint(),
        "format": "float.hex() of the real part",
    }}

    def put(key: str, value: float) -> None:
        out[key] = float(np.real(value)).hex()
        print(f"  {key:>60} = {float(np.real(value))!r}", flush=True)

    for core, E in CORES.items():
        Ea = np.array(E, dtype=int)
        for d, cells in CELLS.items():
            nu = np.full(len(E), d + 0.5)
            for cell, A in cells:
                Aa = np.array(A, dtype=float)
                for n in NS[d]:
                    tag = f"{core}|d{d}|{cell}|n{n}"
                    put(f"hybrid|{tag}",
                        hybrid_zeta(Ea, nu, Aa, n))
                    put(f"slab|{tag}",
                        slab_zeta(E, nu, Aa, n)[0])
                    put(f"slab_nosym|{tag}",
                        slab_zeta(E, nu, Aa, n, use_symmetry=False)[0])
                    put(f"tensor|{tag}",
                        graph_zeta_general_at_zero(Ea, nu, Aa, n))
        # The box: d <= 2 only at reference-affordable L, cubic cell.
        for d in (1, 2):
            nu = np.full(len(E), d + 0.5)
            put(f"box|{core}|d{d}|L234",
                direct_sum_extrapolated(Ea, nu, np.eye(d),
                                        L_list=(2, 3, 4),
                                        n_correction_terms=2))

    # Router paths: the full frontend, vacuum and finite-k, d = 1..3.
    for core in ("K4", "K5", "V6E11"):
        Ea = np.array(CORES[core], dtype=int)
        for d in (1, 2, 3):
            nu_s = d + 0.5
            put(f"router_vac|{core}|d{d}|n8",
                evaluate_graph(Ea, nu_s, np.eye(d), n_points=8,
                               richardson=True))
            put(f"router_k0|{core}|d{d}|n8",
                evaluate_graph(Ea, nu_s, np.eye(d), n_points=8,
                               source=0, terminal=1,
                               momentum=np.zeros(d), richardson=True))

    path = Path(__file__).with_name("engine_reference_values.json")
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"\n{len(out) - 1} values -> {path}")


if __name__ == "__main__":
    main()
