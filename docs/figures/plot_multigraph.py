# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Render a two-terminal multigraph as a TikZ figure, with a thin
pdflatex + pdftocairo pipeline that also produces a PDF (canonical)
and an SVG (inline-renderable on GitHub).

Visual style:

* Terminal vertices in **orange**, internal vertices in **grey**.
* No vertex labels (just filled circles).
* Parallel edges drawn as separate curved bundles, with bends
  fanning out symmetrically.

The tool is independent of the rest of `gzl` — it only needs
`networkx` for the optional spring-layout fallback.

Run as a CLI:

    python -m docs.figures.plot_multigraph \\
        --edges "0-1,1-2,2-3,3-0,0-2" \\
        --terminals 0,2 \\
        --out /tmp/g.tex --compile

Or as a library — see ``multigraph_to_tikz`` and ``compile_tex``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence




# ---------------------------------------------------------------------------
# TikZ generation
# ---------------------------------------------------------------------------

_PREAMBLE = r"""\documentclass[tikz, border=2pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{calc}
\begin{document}
\begin{tikzpicture}[
    every node/.style={circle, draw=black, line width=0.5pt, inner sep=0pt},
    every edge/.style={draw=black!55, line width=0.6pt},
]
"""

_POSTAMBLE = r"""\end{tikzpicture}
\end{document}
"""


def _bend_angles(k: int, step_deg: float) -> list[float]:
    """Symmetric fan of `k` bend angles around 0 in steps of `step_deg`.

    For `k = 1` returns `[0.0]` (straight edge).  For larger `k`,
    angles are placed symmetrically: `k = 2 -> [-step, +step]`,
    `k = 3 -> [-step, 0, +step]`, etc.  Positive angles render as
    `bend left`, negative as `bend right`.
    """
    if k <= 1:
        return [0.0]
    if k % 2 == 1:
        half = (k - 1) // 2
        return [step_deg * (i - half) for i in range(k)]
    half = k // 2
    return [step_deg * (i - half + 0.5) for i in range(k)]


def multigraph_to_tikz(
    edges: Sequence[tuple[int, int]],
    terminals: Sequence[int],
    pos: dict[int, tuple[float, float]],
    *,
    terminal_color: str = "orange",
    internal_color: str = "gray!50",
    node_size_mm: float = 4.0,
    edge_bend_step_deg: float = 8.0,
    standalone: bool = True,
) -> str:
    r"""Build a TikZ document (or just a `tikzpicture` block) for a
    two-terminal multigraph.

    Parameters
    ----------
    edges
        Iterable of vertex pairs ``(u, v)``.  Multi-edges are encoded
        by repeating the same pair.  Self-loops are rejected.
    terminals
        Iterable of terminal vertex labels (rendered in
        ``terminal_color``).  All other vertices are rendered in
        ``internal_color``.
    pos
        Required mapping ``vertex -> (x, y)``; values are used
        verbatim as TikZ coordinates.
    terminal_color, internal_color
        TikZ color names for terminals / internals.
    node_size_mm
        Node diameter in millimetres.
    edge_bend_step_deg
        Bend-angle step for parallel edges.  A bundle of ``k``
        parallels fans out symmetrically with this step.
    standalone
        ``True`` (default) ⇒ wrap with
        ``\documentclass[tikz, border=2pt]{standalone}``;
        ``False`` ⇒ return only the ``\begin{tikzpicture} … \end{tikzpicture}``
        block.

    Returns
    -------
    str
        TikZ source.
    """
    edges = [(int(u), int(v)) for u, v in edges]
    for u, v in edges:
        if u == v:
            raise ValueError(f"self-loop at vertex {u} not supported.")

    terminals = set(int(t) for t in terminals)
    vertices = set(v for e in edges for v in e) | terminals
    missing = vertices - set(pos.keys())
    if missing:
        raise ValueError(
            f"pos missing entries for vertices {sorted(missing)}."
        )

    bundles: dict[tuple[int, int], int] = defaultdict(int)
    for u, v in edges:
        key = (u, v) if u < v else (v, u)
        bundles[key] += 1

    lines = []
    if standalone:
        lines.append(_PREAMBLE)
    else:
        lines.append("\\begin{tikzpicture}\n")

    for v in sorted(vertices):
        x, y = pos[v]
        color = terminal_color if v in terminals else internal_color
        lines.append(
            f"    \\node[fill={color}, minimum size={node_size_mm:.2f}mm] "
            f"(n{v}) at ({x:.4f}, {y:.4f}) {{}};"
        )
    lines.append("")

    for (u, v), k in bundles.items():
        for angle in _bend_angles(k, edge_bend_step_deg):
            if abs(angle) < 1e-9:
                lines.append(f"    \\draw (n{u}) to (n{v});")
            elif angle > 0:
                lines.append(
                    f"    \\draw (n{u}) to[bend left={angle:.1f}] (n{v});"
                )
            else:
                lines.append(
                    f"    \\draw (n{u}) to[bend right={-angle:.1f}] (n{v});"
                )

    if standalone:
        lines.append(_POSTAMBLE)
    else:
        lines.append("\\end{tikzpicture}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Render pipeline (.tex -> .pdf -> .svg)
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, cwd=str(cwd),
        capture_output=True, text=True, check=False,
    )


def compile_tex(
    tex_path: Path,
    *,
    formats: Iterable[str] = ("pdf", "svg"),
    cleanup: bool = True,
) -> dict[str, Path]:
    """Compile a `.tex` file to PDF and/or SVG.

    Pipeline:
        pdflatex tex_path  ->  *.pdf
        pdftocairo -svg    ->  *.svg   (when 'svg' in formats)

    Parameters
    ----------
    tex_path
        Path to the `.tex` source.
    formats
        Output formats; subset of ``{"pdf", "svg"}``.  PDF is always
        produced as an intermediate even if not in ``formats``.
    cleanup
        If True (default) remove `.aux`, `.log` side products on
        success.

    Returns
    -------
    dict[str, Path]
        Mapping ``format -> path`` for each requested output.
    """
    tex_path = Path(tex_path).resolve()
    if not tex_path.is_file():
        raise FileNotFoundError(tex_path)
    if shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex not found on PATH.")

    cwd = tex_path.parent
    base = tex_path.with_suffix("")

    proc = _run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         tex_path.name],
        cwd=cwd,
    )
    if proc.returncode != 0:
        last = "\n".join(proc.stdout.splitlines()[-30:])
        raise RuntimeError(f"pdflatex failed:\n{last}")

    pdf_path = base.with_suffix(".pdf")
    out: dict[str, Path] = {"pdf": pdf_path}

    if "svg" in formats:
        if shutil.which("pdftocairo") is None:
            raise RuntimeError(
                "pdftocairo not found on PATH; install poppler "
                "(brew install poppler / apt install poppler-utils)."
            )
        svg_path = base.with_suffix(".svg")
        proc = _run(
            ["pdftocairo", "-svg", pdf_path.name, svg_path.name],
            cwd=cwd,
        )
        if proc.returncode != 0:
            last = "\n".join(
                (proc.stdout + proc.stderr).splitlines()[-30:]
            )
            raise RuntimeError(f"pdftocairo failed:\n{last}")
        out["svg"] = svg_path

    if cleanup:
        for ext in (".aux", ".log"):
            side = base.with_suffix(ext)
            if side.is_file():
                try:
                    side.unlink()
                except OSError:
                    pass

    if "pdf" not in formats and "pdf" in out:
        try:
            out.pop("pdf").unlink()
        except OSError:
            pass

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_edges(spec: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        u_s, v_s = tok.split("-")
        out.append((int(u_s), int(v_s)))
    return out


def _parse_terminals(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def _parse_pos(spec: str) -> dict[int, tuple[float, float]]:
    """Parse ``"v=x,y;v=x,y;..."``."""
    out: dict[int, tuple[float, float]] = {}
    for clause in spec.split(";"):
        clause = clause.strip()
        if not clause:
            continue
        v_s, xy = clause.split("=")
        x_s, y_s = xy.split(",")
        out[int(v_s)] = (float(x_s), float(y_s))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="plot_multigraph",
        description="Render a two-terminal multigraph as TikZ + PDF/SVG.",
    )
    p.add_argument("--edges", required=True,
                   help='Edge spec "u-v,u-v,…"; repeat pairs for parallel edges.')
    p.add_argument("--terminals", required=True,
                   help='Terminal vertex spec "s,t".')
    p.add_argument("--pos", default=None,
                   help='Optional position spec "v=x,y;…".  '
                        'If omitted, networkx.spring_layout(seed=0) is used.')
    p.add_argument("--out", required=True, type=Path,
                   help="Output .tex path.")
    p.add_argument("--compile", action="store_true",
                   help="Run pdflatex + dvisvgm to produce .pdf and .svg "
                        "next to the .tex.")
    p.add_argument("--svg-only", action="store_true",
                   help="With --compile: drop the intermediate .pdf "
                        "after producing the .svg.")
    args = p.parse_args(argv)

    edges = _parse_edges(args.edges)
    terminals = _parse_terminals(args.terminals)
    if args.pos is None:
        try:
            import networkx as nx
        except ImportError as exc:
            raise SystemExit(
                "networkx is required when --pos is not supplied; "
                "reinstall gzl, whose dependency it is, or pass --pos."
            ) from exc
        G = nx.MultiGraph()
        G.add_edges_from(edges)
        pos = nx.spring_layout(G, seed=0)
    else:
        pos = _parse_pos(args.pos)

    tex = multigraph_to_tikz(edges, terminals, pos)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    print(f"wrote {args.out}")

    if args.compile:
        formats = ("svg",) if args.svg_only else ("pdf", "svg")
        produced = compile_tex(args.out, formats=formats)
        for fmt, path in produced.items():
            print(f"wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
