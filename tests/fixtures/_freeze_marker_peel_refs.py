# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Regenerate ``tests/fixtures/marker_peel_refs.json``.

S9 (the offset-aware marker peel) is a *value change*: the Z2 marker
becomes unconditional again (``_plan_marker``'s model is deleted) and
marker-touching elimination steps may take the FFT peel, so box values
move by re-association at the round-off scale.  The gate is
``direct_sum_extrapolated`` on the four canonical tw >= 3 blocks at
the shipped ladders, **within 1e-12 relative of the pre-S9 values
frozen here, with the Richardson fit residual re-checked** — a
basis/executor desync fits the wrong power while the least-squares
solve still succeeds and the value stays plausible.

**Generated on the pre-S9 tree**, the last one with ``_plan_marker``.
Re-running it after S9 lands re-freezes new-vs-new and the gate is
vacuous; regenerate only to move the baseline deliberately, and say so
in the commit message.

Ladders and correction terms mirror the frontend exactly:
``_DENSE_DSUM_L_LIST`` with ``K = min(3, len(L_list) - 1)``.

The committed fixture holds ten of the twelve records this writes:
k33_d3 and k5_d3 took 59-117 s each and were dropped from the suite.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

import gzl.direct_sum as ds
from gzl.direct_sum import direct_sum_extrapolated


K4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
K5 = [(a, b) for a in range(5) for b in range(a + 1, 5)]
K33 = [(a, b) for a in range(3) for b in range(3, 6)]
PRISM = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
         (0, 3), (1, 4), (2, 5)]

BLOCKS = [("K4", K4), ("PRISM", PRISM), ("K33", K33), ("K5", K5)]
LADDERS = {1: (4, 5, 6, 7, 8), 2: (3, 4, 5, 6), 3: (2, 3, 4)}


def main() -> None:
    # PRE-S9 TRIPWIRE (mirrors the signature trap in
    # _freeze_hybrid_s7_refs.py): S9 deleted _plan_marker, so a
    # post-S9 tree fails here instead of silently re-freezing
    # new-vs-new and making TestLadderRefs vacuous.  Edit this guard
    # only for a DELIBERATE re-baseline, with the reason in the
    # commit message.
    import gzl.direct_sum as _ds
    if not hasattr(_ds, "_plan_marker"):
        raise SystemExit(
            "this tree is post-S9 (_plan_marker is deleted); "
            "regenerating here would freeze new-vs-new. Check out the "
            "pre-S9 tree, or edit the tripwire for a deliberate "
            "re-baseline."
        )
    records = []
    for d, L_list in LADDERS.items():
        A = np.eye(d)
        nu = d + 1.5
        K_corr = min(3, len(L_list) - 1)
        for name, edges in BLOCKS:
            t0 = time.perf_counter()
            nu_vec = np.full(len(edges), nu)
            v = direct_sum_extrapolated(
                edges, nu_vec, A,
                L_list=L_list, n_correction_terms=K_corr,
            )
            dt = time.perf_counter() - t0
            # Raw rungs + the fit residual, so the residual gate has a
            # frozen baseline (it is a physical basis-truncation
            # quantity, not round-off).  NOTE: in the original pre-S9
            # freeze these two fields were appended in a second pass
            # from rungs verified pre-S9-equal (d=1 gate-closed hence
            # bit-identical; d>=2 moved <= 4e-16); any future
            # regeneration computes them inline here.
            S = np.array([float(np.real(complex(
                ds.direct_sum_zero_momentum(edges, nu_vec, A, L))))
                for L in L_list])
            edge_map = ds._collapse_multi_edges(np.asarray(edges), nu_vec)
            V = max(max(a, b) for a, b in edge_map) + 1
            alpha = d - ds._min_free_cut_nu(
                edge_map, ds._pick_root(edge_map, V))
            Lf = np.array(L_list, dtype=float)
            design = np.stack(
                [np.ones_like(Lf)] +
                [Lf ** (alpha - k) for k in range(K_corr)], axis=1)
            coef, res, *_ = np.linalg.lstsq(design, S, rcond=None)
            resid = (float(np.sqrt(res[0])) if len(res)
                     else float(np.linalg.norm(design @ coef - S)))
            records.append({
                "name": f"{name.lower()}_d{d}",
                "block": name,
                "edges": [list(e) for e in edges],
                "nu": float(nu),
                "d": d,
                "L_list": list(L_list),
                "n_correction_terms": K_corr,
                "value_hex": float(np.real(complex(v))).hex(),
                "S_hex": [x.hex() for x in S.tolist()],
                "residual_hex": resid.hex(),
                "seconds": round(dt, 3),
            })
            print(f"{name:6s} d={d}: {dt:7.1f}s")
    out = pathlib.Path(__file__).parent / "marker_peel_refs.json"
    out.write_text(json.dumps({
        "comment": "Pre-S9 direct_sum_extrapolated values at the shipped "
                   "ladders; see _freeze_marker_peel_refs.py.",
        "records": records,
    }, indent=1) + "\n")
    print(f"froze {len(records)} records -> {out}")


if __name__ == "__main__":
    main()
