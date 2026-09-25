# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Regenerate ``tests/fixtures/hybrid_s7_refs.json``.

The change these references gate, called S7 here and in the file
names, replaces ``hybrid._dense_core``'s private execution loop with the
shared bucket-elimination executor (``_contract.eliminate`` on a
``TorusTruncation``).  That is a *value change* on the dense-core path
(different contraction association, FFT peel now reachable), gated at
**<= 4 ulp with the ulp distance recorded per case** against the values
the pre-S7 loop produced.  This file freezes those pre-S7 values.

**The frozen values are only meaningful if generated on a tree WITHOUT
the S7 change**.  Re-running this script after S7 lands would re-freeze
new-vs-new and make the gate vacuous; as a guard, the script calls
``_dense_core`` with its *pre-S7 signature*, so on a post-S7 tree it
fails loudly with a ``TypeError`` instead of silently producing a
vacuous baseline.  Regenerate only to move the baseline deliberately
(check out the pre-S7 tree), and say so in the commit message.

Two value layers per record:

``core_hex``
    The position-space core output ``M`` (``_sp_reduce`` +
    ``_dense_core``), the tensor the executor swap actually changes.
    Per-entry ulp distances are meaningful here — every entry is a same-
    sign kernel sum.  This is what the <= 4 ulp dense-path gate reads.

``value_hex``
    The public ``hybrid_zeta`` output.  Grid mode applies ``fftn`` to
    ``M``, whose small entries arise by cancellation — a 1e-15-of-scale
    move can be tens of per-entry ulps there, so the public surface is
    gated norm-wise (ulps of the largest entry), not per entry.

Unlike the executor goldens (bit-identity instrument for pure moves),
every record here is ``constructed``: at the time of freezing, the
production frontend SP-reduced all 299/299 observed hybrid calls to a
(2, 1) residue, so the dense core never fired in production and no
captured call would have exercised it.

**That premise no longer holds.**  A later routing change sent
treewidth->=3 blocks to the torus at d <= 2 (d <= 3 today), and those
are precisely the blocks whose SP reduction leaves an irreducible core.
The dense core now fires in production, so the <= 4-ulp per-entry gate
below has been promoted from a synthetic guard on an unreachable branch
to a gate on a live one.  Nothing about the records or the tolerances
changes --- what changes is how much rides on them, which is worth
knowing before anyone relaxes one.
The fixture set is chosen to hit the cases where the old and new dense
branches genuinely differ: buckets with >= 3 factors, and buckets
whose eliminated vertex is the *last* axis of the union scope — the two
branches are bit-identical only on a 2-factor bucket eliminating the
leading axis, so a fixture set without those cases would certify an
equivalence exactly where it cannot fail.

Run from the repo root::

    python tests/fixtures/_freeze_hybrid_s7_refs.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from gzl.hybrid import hybrid_zeta, _dense_core, _sp_reduce
from gzl.tensor_network import _edge_kernel_torus


K4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
K5 = [(a, b) for a in range(5) for b in range(a + 1, 5)]
PRISM = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
         (0, 3), (1, 4), (2, 5)]
CORE7V12E = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
             (1, 4), (2, 5), (3, 6), (4, 5), (4, 6), (5, 6)]

I1 = [[1.0]]
I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
# Hexagonal cell at EVEN n: the one lattice class whose torus kernel is
# genuinely not even (the balanced z-axis holds n/2 without -n/2), so
# core orientation handling is live rather than vacuous here.
HEX = [[1.0, 0.5], [0.0, float(np.sqrt(3.0) / 2.0)]]

# name, edges, nu, A, n, kwargs for hybrid_zeta
CASES = [
    ("k4_d1_vacuum", K4, 2.5, I1, 16, {}),
    ("k4_d1_grid_t1", K4, 2.5, I1, 16, {"terminal": 1}),
    ("k4_d1_singlek_offgrid", K4, 2.5, I1, 16,
     {"terminal": 1, "momentum": [0.3]}),
    ("core7_d1_vacuum", CORE7V12E, 2.5, I1, 12, {}),
    ("core7_d1_grid_t6", CORE7V12E, 2.5, I1, 12, {"terminal": 6}),
    ("prism_d1_grid_t5", PRISM, 2.5, I1, 12, {"terminal": 5}),
    ("k5_d1_vacuum", K5, 2.5, I1, 10, {}),
    ("k4_d2_hex_even_n_vacuum", K4, 3.0, HEX, 8, {}),
    ("k4_d2_hex_even_n_grid_t3", K4, 3.0, HEX, 8, {"terminal": 3}),
    ("k4_d3_vacuum", K4, 3.5, I3, 4, {}),
]


def main() -> None:
    records = []
    for name, edges, nu, A, n, kwargs in CASES:
        kwargs = dict(kwargs)
        if "momentum" in kwargs:
            kwargs["momentum"] = np.asarray(kwargs["momentum"], dtype=float)
        nu_vec = np.full(len(edges), float(nu))
        A_arr = np.asarray(A, dtype=float)
        val = hybrid_zeta(edges, nu_vec, A_arr, n, **kwargs)
        arr = np.asarray(val, dtype=float)

        # The position-space core tensor M, via the same reduction
        # hybrid_zeta runs.  NOTE the pre-S7 _dense_core signature
        # (no A, no peelable) — the deliberate post-S7 tripwire.
        d = A_arr.shape[0]
        term = kwargs.get("terminal", 0)
        kernels = [_edge_kernel_torus(float(nu_vec[i]), A_arr, n)
                   for i in range(len(edges))]
        nodes, r_edges, r_kernels, sp_scalar = _sp_reduce(
            [tuple(e) for e in edges], kernels, 0, int(term), n, d)
        M = _dense_core(nodes, r_edges, r_kernels, 0, int(term), n, d,
                        5e7, 1e12) * sp_scalar
        M = np.asarray(M, dtype=float)

        records.append({
            "name": name,
            "edges": [list(e) for e in edges],
            "nu": float(nu),
            "A": A,
            "n_points": int(n),
            "terminal": kwargs.get("terminal"),
            "momentum": ([float(k) for k in kwargs["momentum"]]
                         if "momentum" in kwargs else None),
            "shape": list(arr.shape),
            "value_hex": [x.hex() for x in arr.ravel().tolist()],
            "core_shape": list(M.shape),
            "core_hex": [x.hex() for x in M.ravel().tolist()],
        })
    out = pathlib.Path(__file__).parent / "hybrid_s7_refs.json"
    out.write_text(json.dumps({
        "comment": "Pre-S7 hybrid._dense_core values; see "
                   "_freeze_hybrid_s7_refs.py. Regenerating after S7 "
                   "makes the gate vacuous - only re-freeze to move the "
                   "baseline deliberately.",
        "records": records,
    }, indent=1) + "\n")
    print(f"froze {len(records)} records -> {out}")


if __name__ == "__main__":
    main()
