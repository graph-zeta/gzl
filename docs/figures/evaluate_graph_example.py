# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Render the README's `evaluate_graph` quick-start example as a TikZ
figure (PDF + inline-renderable SVG), via the reusable
:mod:`docs.figures.plot_multigraph` tool.

Composite graph used in the docs example:

* **Diamond block** — 4-cycle 0-1-2-3 with chord 0-2.
* **Triangle pendant** at vertex 0 — block 0-4-5.
* **Bridge pendant** at vertex 2 — edge 2-6.

Vertices 0 and 2 are the two terminals (rendered orange); 1, 3, 4,
5, 6 are internal (rendered grey).  No vertex labels.

Run from the repo root:

    python3 docs/figures/evaluate_graph_example.py

Writes ``docs/figures/evaluate_graph_example.{pdf,svg}``.
"""

from __future__ import annotations

from pathlib import Path

from plot_multigraph import compile_tex, multigraph_to_tikz


# Topology — keep in sync with the README quick-start example.
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0), (0, 2),    # diamond (chord 0-2)
    (0, 4), (4, 5), (5, 0),                    # triangle pendant at vertex 0
    (2, 6),                                    # bridge at vertex 2
]
TERMINALS = (0, 2)


# Hand-tuned horizontal layout: triangle pendant on the left,
# diamond at the centre, bridge pendant on the right.
POS = {
    4: (-3.5,  0.7),
    5: (-3.5, -0.7),
    0: (-1.5,  0.0),
    1: ( 0.0,  1.0),
    3: ( 0.0, -1.0),
    2: ( 1.5,  0.0),
    6: ( 3.5,  0.0),
}


def main() -> None:
    here = Path(__file__).parent
    tex_path = here / "evaluate_graph_example.tex"
    tex = multigraph_to_tikz(EDGES, TERMINALS, POS)
    tex_path.write_text(tex)
    produced = compile_tex(tex_path, formats=("pdf", "svg"))
    for fmt, path in produced.items():
        print(f"wrote {path}")
    # The .tex source is intermediate per repo convention; clean up.
    if tex_path.is_file():
        tex_path.unlink()


if __name__ == "__main__":
    main()
