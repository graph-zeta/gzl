# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""The ``series`` command-line driver.

This is the command-line half of :mod:`gzl.series`, kept in its own
module so that nothing imports it.  ``gzl/__init__.py`` imports
``gzl.series`` eagerly, which is why ``python -m gzl.series`` makes
runpy execute that file a second time as ``__main__`` and warn about
it.  Nothing imports this module, so the console script and
``python -m gzl`` run it exactly once by construction, with no
invariant left for a later edit to preserve.

Two entry points wrap the same parser:

* :func:`run` -- ``gzl series ...`` and, through
  :func:`gzl.series.main`, ``python -m gzl.series ...``.  The two
  differ only in the ``prog`` their ``--help`` prints.
* :func:`build_parser` -- the parser itself, for ``gzl --help``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

import numpy as np

from gzl._data import CORPORA, _resolve_corpus
from gzl._lattices import LATTICES, _resolve_lattice
from gzl.interaction import Interaction, is_interaction
from gzl.series import compute_series_coefficients, evaluate_corpus, nu_grid

__all__ = ["build_parser", "run"]

#: ``prog`` for the spelling that predates the console script.
_LEGACY_PROG = "python -m gzl.series"

#: ``--help`` prose.  Was ``gzl.series.__doc__``, which also described
#: the library half; only the part a CLI user needs is kept.
_DESCRIPTION = """\
Per-order series coefficients c_O(k, nu) = sum_G a_G * zeta_G(k; nu)
over a graph corpus.

The corpus is a consolidated NPZ (flat / CSR layout) or HDF5 (per-graph
subgroup layout).  Per graph it carries an integer `edges` list,
integer `multiplicities`, and a real `prefactor`; the optional
`hopping` field (s, t) marks a 1qp graph, and its absence means the
vacuum convention s = t = 0.

A TOML config, command-line flags, or both -- flags override config,
and config overrides the defaults.

`--corpus` (or `[corpus] path`) may name a shipped corpus, `tfim0qp` or
`tfim1qp`, instead of a path; without either, `tfim1qp` is used.  `--A`
(or `[lattice] A`) may likewise name a Bravais lattice -- `chain`,
`square`, `triangular`, `cubic` -- instead of spelling its matrix out.
"""


# The corpus may be left out: it defaults to _CLI_DEFAULT_CORPUS.
_TOML_REQUIRED_SECTIONS = ("lattice", "evaluation")

#: The corpus a CLI run uses when neither ``--corpus`` nor ``[corpus] path``
#: names one.  The library functions keep the corpus an explicit argument.
_CLI_DEFAULT_CORPUS = "tfim1qp"


def _read_config(config_path: Path) -> dict:
    import tomllib

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    missing = [s for s in _TOML_REQUIRED_SECTIONS if s not in data]
    if missing:
        raise ValueError(
            f"Config {config_path}: missing required sections {missing}."
        )
    # A named lattice is the VALUE of A inside [lattice]; written at the
    # top level, ``lattice = "cubic"`` passes the check above and then
    # looks like a missing A.  Say what the spelling is instead.
    for section in _TOML_REQUIRED_SECTIONS:
        if not isinstance(data[section], dict):
            example = ('[lattice]\n    A = "cubic"' if section == "lattice"
                       else f"[{section}]\n    <key> = <value>")
            raise ValueError(
                f"Config {config_path}: [{section}] must be a TOML table, "
                f"not the bare value {data[section]!r}.  Write\n"
                f"    {example}\n"
                f"rather than {section} = ... at the top level."
            )
    return data


def _parse_A(arg: str) -> np.ndarray:
    """The lattice a ``--A`` / ``[lattice] A`` string means: the name of a
    lattice -- ``"chain"``, ``"square"``, ``"triangular"``, ``"cubic"`` --
    or a matrix as a JSON / Python-literal string like ``"[[1.0]]"`` or
    ``"[[1.0, 0.0], [0.0, 1.0]]"``.  No name is valid JSON, so the two
    spellings cannot collide."""
    if arg in LATTICES:
        return _resolve_lattice(arg)
    try:
        value = json.loads(arg)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--A must name a lattice ("
            + ", ".join(repr(name) for name in LATTICES)
            + f") or be a matrix as JSON, e.g. '[[1.0]]'; got {arg!r}."
        ) from exc
    if isinstance(value, str):
        # A quoted name, '"cubic"': one rule, and one error for an
        # unknown one.
        return _resolve_lattice(value)
    A = np.array(value, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"--A must be a square matrix, got shape {A.shape}.")
    return A


def _parse_momentum_cli(arg):
    """Resolve a ``--momentum`` CLI / config value into the form expected
    by :func:`compute_series_coefficients`.

    ``None`` / empty / 'none' / 'default' → ``None`` (corpus-determined
    default: scalar for 0qp, grid for 1qp).
    Anything else is parsed as JSON.  A JSON scalar is returned as a
    Python ``float`` (which ``compute_series_coefficients`` accepts in
    d=1); a JSON list is returned as an ``np.ndarray``.
    """
    if arg is None:
        return None
    s = str(arg).strip()
    if not s or s.lower() in ("none", "default"):
        return None
    parsed = json.loads(s)
    if isinstance(parsed, (int, float)):
        return float(parsed)
    return np.asarray(parsed, dtype=float)


def _build_k_frac_table(momentum, d: int, n_points: int, trailing_shape):
    """Build the ``(prod(trailing_shape), d)`` flat k-frac table that
    indexes the flattened trailing axes of the CSV emit.

    Grid mode is identified by ``trailing_shape == (n_points,) * d``
    (no other shape can match exactly when ``n_points >= 2``), and the
    k-frac points are the standard ``arange(n_points)/n_points``
    cartesian product (C-order, matching
    ``np.meshgrid(..., indexing='ij')`` — the same convention as
    ``graph_sample`` and the topology evaluator).  Batch mode uses the
    user-supplied k-list reshaped to ``(N, d)``.  Returns an empty
    array when ``trailing_shape == ()``.
    """
    if not trailing_shape:
        return np.empty((0, d), dtype=float)
    if trailing_shape == (n_points,) * d:
        axes = [np.arange(n_points, dtype=float) / n_points for _ in range(d)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack(mesh, axis=-1).reshape(-1, d)
    arr = np.asarray(momentum, dtype=float)
    if arr.ndim == 1 and d == 1 and arr.size != d:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise AssertionError(
        f"_build_k_frac_table: unexpected momentum shape "
        f"{arr.shape} for trailing_shape {trailing_shape}."
    )


def _parse_nu_cli(arg) -> float:
    """Parse a single-ν CLI / config value; accepts a float or ``inf``."""
    if isinstance(arg, (int, float)):
        return float(arg)
    s = str(arg).strip().lower().lstrip("+")
    if s in ("inf", "np.inf", "infinity"):
        return float("inf")
    return float(arg)


def _run_per_graph(corpus_path, nus, A, n_points, momentum, order_max,
                   output_path, progress,
                   *, with_interaction: bool, routing: dict, say,
                   traceback_wanted: bool) -> int:
    """``--per-graph`` driver: dump ``evaluate_corpus`` records to CSV.

    ``with_interaction`` — the user gave an ``[interaction]`` /
    ``[[interactions]]`` table — is the ONE criterion for the extended
    schema, shared with the coefficient CSV in :func:`main`, so the two
    files agree even when every sweep point is a plain power law (the
    records then carry the label the user gave, not a float's).

    ``routing`` carries the routing settings the coefficient pass
    receives (``dense_engine``, ``sp_n_points``, ``core_grading``),
    which this path used to drop.
    A pass that cannot finish is reported the way the coefficient pass
    reports it, in one line and with exit code 1, not as a traceback.
    """
    try:
        records = evaluate_corpus(
            corpus_path, nus, A, n_points,
            momentum=momentum, order_max=order_max,
            progress=progress, **routing,
        )
    except KeyboardInterrupt:
        print("\ngzl: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        return _fail(exc, traceback_wanted=traceback_wanted)
    if not records:
        print("No records to write.", file=sys.stderr)
        return 0
    if np.ndim(records[0]["value"]) != 0:
        print("Error: --per-graph emits one scalar per graph, but the corpus / "
              "momentum produced an array value (a 1qp full-BZ grid).  Pass "
              "--momentum for a single k, or use a vacuum (0qp) corpus.",
              file=sys.stderr)
        return 2

    def _fmt_nu(x):
        return "inf" if not np.isfinite(x) else f"{x:.10f}"

    try:
        with _open_output(output_path) as f:
            w = csv.writer(f)
            header = ["order", "graph_id", "nu"]
            if with_interaction:
                header.append("interaction")
            w.writerow(header + ["value", "prefactor", "contribution",
                                 "s", "t"])
            for r in records:
                cval = complex(r["value"])
                ccon = complex(r["contribution"])
                row = [r["order"], r["graph_id"], _fmt_nu(r["nu"])]
                if with_interaction:
                    row.append(r["nu_label"])
                w.writerow(row + [f"{cval.real:.12e}",
                                  f"{r['prefactor']:.12e}",
                                  f"{ccon.real:.12e}", r["s"], r["t"]])
    except (_OutputError, OSError) as exc:
        return _fail(exc, traceback_wanted=traceback_wanted)
    # The banner already ends in a blank line; a second one only
    # separates the progress lines, when there are any.
    say(("\n" if progress else "")
        + f"Wrote {len(records)} per-graph rows to "
        + ("standard output" if str(output_path) == _STDOUT
           else str(output_path)))
    return 0


_EPILOG = """\
examples:
  A single nu, the shipped 1qp corpus, the full Brillouin zone:
    %(prog)s --corpus tfim1qp --A cubic --n-points 8 \\
        --nu 3.5 --order-max 9 --output series.csv

  A nu sweep on the chain, from a config, with per-order progress:
    %(prog)s --config series_0qp_d1.toml --progress

  One k, and the CSV on standard output:
    %(prog)s --corpus tfim1qp --A chain --n-points 64 --nu 3.0 \\
        --momentum 0.5 --output - | head

  What a config resolves to, without running it:
    %(prog)s --config series_0qp_d1.toml --dry-run

exit codes:
  0  the pass ran and the CSV was written
  1  the pass could not finish (--traceback for the stack trace)
  2  the command was used wrongly
"""


def build_parser(prog: str = _LEGACY_PROG) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=None,
                   help="TOML config (overrides defaults; CLI flags override config).")
    # ``str``, not ``Path``: only a string can be a shipped corpus's name.
    p.add_argument("--corpus", type=str, default=None,
                   help="Path to consolidated corpus (.npz, .h5), or the "
                        "name of a shipped one: "
                        + ", ".join(CORPORA)
                        + f" (default: {_CLI_DEFAULT_CORPUS}).")
    # ``str``, not a parsed matrix: only a string can be a lattice's name.
    p.add_argument("--A", type=str, default=None,
                   help="Lattice matrix as JSON, e.g. '[[1.0]]' (columns "
                        "are the primitive vectors), or the name of a "
                        "lattice: " + ", ".join(LATTICES) + ".")
    p.add_argument("--n-points", type=int, default=None,
                   dest="n_points")
    p.add_argument("--order-max", type=int, default=None, dest="order_max")
    p.add_argument("--nu-start", type=float, default=None, dest="nu_start")
    p.add_argument("--nu-end", type=float, default=None, dest="nu_end")
    p.add_argument("--nu-step", type=float, default=None, dest="nu_step")
    # The three settings of the removed k = 0 fallback cascade: accepted,
    # reported, never forwarded (see the [routing] block in `run`).
    p.add_argument("--nu-tensor-threshold", type=float, default=None,
                   dest="nu_tensor_threshold",
                   help="deprecated, has no effect")
    p.add_argument("--tw-threshold", type=int, default=None,
                   dest="tw_threshold", help="deprecated, has no effect")
    p.add_argument("--sigma-max", type=float, default=None, dest="sigma_max",
                   help="deprecated, has no effect")
    p.add_argument("--dense-engine", type=str, default=None,
                   choices=("direct_sum", "torus"), dest="dense_engine",
                   help="route for treewidth>=3 blocks; default resolves "
                        "per d inside the frontend")
    p.add_argument("--core-grading",
                   action=argparse.BooleanOptionalAction, default=None,
                   dest="core_grading",
                   help="size each dense block's irreducible core DOWN "
                        "from n_points, per block, by "
                        "kappa*n_points^(sigma_blk/sigma_core); the core "
                        "converges a full nu faster than its block, so at "
                        "a shared grid it is over-resolved and it is the "
                        "half that costs n^(tau*d).  Also suppresses the "
                        "k=0 Richardson ladder on the graded blocks")
    p.add_argument("--sp-n-points", type=int, default=None, dest="sp_n_points",
                   help="fine SP-collapse grid for dense blocks on the "
                        "torus route (must be >= n_points)")
    p.add_argument("--momentum", type=str, default=None,
                   help="External momentum in fractional BZ coords.  "
                        "Default: corpus-determined — 0qp → scalar at "
                        "k=0; 1qp → full n_points^d BZ grid.  Accepts: "
                        "'none' / 'default' (same as omitting the "
                        "flag), or a JSON literal — scalar / 1-D / "
                        "2-D array.  Examples: --momentum 0  (force "
                        "scalar at k=0 in d=1), --momentum 0.5 (k=π "
                        "in d=1), --momentum '[0.25, 0.5]' (k=(π/2, π) "
                        "in d=2), --momentum '[[0],[0.25],[0.5]]' "
                        "(3-batch in d=1).  When the result has a "
                        "k-axis the CSV emits the extended schema "
                        "``order, nu, k_index, k_frac_0, ..., "
                        "coefficient``.")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--diagnostics-csv", type=Path, default=None,
                   dest="diagnostics_csv")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--nu", type=str, default=None,
                   help="Single ν (overrides --nu-start/--nu-end/--nu-step).  "
                        "Accepts a float or 'inf' (the nearest-neighbour "
                        "kernel — emits exact lattice embedding factors).")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="print the settings the run would use -- every "
                        "value after flags have overridden config and "
                        "config the defaults -- and exit without "
                        "computing anything")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="suppress the banner, the closing summary and "
                        "--progress; errors are still reported")
    p.add_argument("--traceback", action="store_true",
                   help="on failure, raise instead of printing one line "
                        "(also GZL_TRACEBACK=1)")
    p.add_argument("--per-graph", action="store_true", dest="per_graph",
                   help="Dump per-graph values via evaluate_corpus to CSV "
                        "(order,graph_id,nu,value,prefactor,contribution,s,t) "
                        "instead of the summed series coefficients.  "
                        "Scalar-per-graph only — vacuum corpus or a single "
                        "--momentum; 1qp full-BZ-grid dumps are not supported.")
    return p


def _resolve(arg, config_section, key, default=None):
    """CLI flag wins over config value wins over default."""
    if arg is not None:
        return arg
    if config_section is not None and key in config_section:
        return config_section[key]
    return default


#: ``--output`` for standard output rather than a file.
_STDOUT = "-"


class _OutputError(Exception):
    """A CSV sink that could not be opened, with the path in the message."""


@contextlib.contextmanager
def _open_output(path: Path):
    """The CSV sink: a file, or standard output for ``--output -``.

    Standard output is never closed here -- the caller keeps using it.
    An unwritable destination is the command failing, not the library,
    so it is reported as one line rather than as a stack trace through
    ``pathlib``.
    """
    if str(path) == _STDOUT:
        yield sys.stdout
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "w", newline="")
    except OSError as exc:
        raise _OutputError(f"cannot write {path}: {exc.strerror or exc}"
                           ) from exc
    with fh:
        yield fh


def _wants_traceback(flag: bool) -> bool:
    return bool(flag) or os.environ.get("GZL_TRACEBACK", "") not in ("", "0")


def _fail(exc: BaseException, *, traceback_wanted: bool) -> int:
    """Report a run that could not finish; return the exit code.

    A stack trace is the right answer for a library and the wrong one
    for a command: the reader wants to know what went wrong, not which
    frame noticed.  ``--traceback`` (or ``GZL_TRACEBACK=1``) restores
    it for a bug report.
    """
    if traceback_wanted:
        raise exc
    name = type(exc).__name__
    print(f"gzl: error: {exc}" if str(exc) else f"gzl: error: {name}",
          file=sys.stderr)
    # The corpus pass notes which graph it was on (PEP 678).
    for note in getattr(exc, "__notes__", ()):
        print(f"gzl: ({note})", file=sys.stderr)
    print("gzl: (re-run with --traceback for the full stack trace)",
          file=sys.stderr)
    return 1


def run(argv: Iterable[str] | None = None, *,
        prog: str = _LEGACY_PROG) -> int:
    args = build_parser(prog).parse_args(argv)
    traceback_wanted = _wants_traceback(args.traceback)

    def say(*parts) -> None:
        """Status, on stderr, so ``--output -`` keeps stdout for data."""
        if not args.quiet:
            print(*parts, file=sys.stderr)

    # A malformed config is the command being used wrongly, which the
    # CLI answers with exit code 2 everywhere else.  It used to raise.
    try:
        cfg = _read_config(args.config) if args.config else {}
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:                  # a TOML syntax error
        if traceback_wanted:
            raise
        print(f"Error: {args.config}: {exc}", file=sys.stderr)
        return 2

    corpus_arg = _resolve(args.corpus, cfg.get("corpus"), "path")
    corpus_default = corpus_arg is None
    if corpus_default:
        corpus_arg = _CLI_DEFAULT_CORPUS
    # A shipped corpus may be given by name ("tfim1qp").  The banner shows
    # the file the argument resolves to; the argument itself goes on to the
    # library, which resolves it the same way and, for a bare string that
    # is neither a name nor a file, lists the names in its error.
    corpus_path = _resolve_corpus(corpus_arg)
    # Naming a corpus that is not there is the command being used wrongly,
    # like an unknown lattice, so it is exit code 2 rather than a failed
    # run.  The library opens the file lazily and raises
    # ``FileNotFoundError`` deep in the pass; checking here costs one
    # ``stat`` and reuses the library's own wording, which lists the
    # shipped names when a bare string can only have meant one.
    if not corpus_path.is_file():
        msg = f"Corpus file not found: {corpus_path}"
        if (isinstance(corpus_arg, str) and not corpus_path.suffix
                and len(corpus_path.parts) == 1 and not corpus_path.exists()):
            msg += (" (nor is it a shipped corpus name: "
                    + ", ".join(repr(name) for name in CORPORA) + ")")
        print(f"Error: {msg}", file=sys.stderr)
        return 2

    A_arg = _resolve(args.A, cfg.get("lattice"), "A")
    if A_arg is None:
        print("Error: --A or [lattice].A is required.", file=sys.stderr)
        return 2
    try:
        A = (
            _parse_A(A_arg) if isinstance(A_arg, str)
            else np.asarray(A_arg, dtype=float)
        )
    except (ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    eval_cfg = cfg.get("evaluation", {})
    n_points = _resolve(args.n_points, eval_cfg, "n_points")
    if n_points is None:
        print("Error: --n-points or [evaluation].n_points is required.",
              file=sys.stderr)
        return 2
    n_points = int(n_points)
    order_max = _resolve(args.order_max, eval_cfg, "order_max")

    # A general interaction: an [interaction] table, or an [[interactions]]
    # array of tables (a sweep), mutually exclusive with the nu spellings.
    inter_cfg = cfg.get("interaction")
    inter_list = cfg.get("interactions")
    if inter_cfg is not None and inter_list is not None:
        print("Error: give either [interaction] or [[interactions]], not both.",
              file=sys.stderr)
        return 2
    interactions = None
    if inter_cfg is not None or inter_list is not None:
        nu_spelled = [
            _resolve(args.nu, eval_cfg, "nu"),
            _resolve(args.nu_start, eval_cfg, "nu_start"),
            _resolve(args.nu_end, eval_cfg, "nu_end"),
            _resolve(args.nu_step, eval_cfg, "nu_step"),
        ]
        if any(x is not None for x in nu_spelled):
            print("Error: [interaction] / [[interactions]] cannot be combined "
                  "with nu / nu_start / nu_end / nu_step; the interaction IS "
                  "the coupling.", file=sys.stderr)
            return 2
        if inter_list is not None:
            if not isinstance(inter_list, list) or not all(
                    isinstance(t, Mapping) for t in inter_list):
                print("Error: [[interactions]] must be an array of tables: "
                      "one [[interactions]] header (double brackets) per "
                      "sweep point; a single [interactions] table is not "
                      "a sweep.", file=sys.stderr)
                return 2
            if not inter_list:
                print("Error: [[interactions]] is empty; give at least one "
                      "sweep point, or use nu / nu_start / nu_end / "
                      "nu_step.", file=sys.stderr)
                return 2
            tables = list(inter_list)
        else:
            if not isinstance(inter_cfg, Mapping):
                print("Error: [interaction] must be a table (keys label, b, "
                      "nu, and one of shells / compact / compact_npz).",
                      file=sys.stderr)
                return 2
            tables = [inter_cfg]
        # ``total`` is read by the shells form only; with compact /
        # compact_npz the table is given directly and the key was
        # silently ignored, so a config that said "subtract the tail"
        # ran on the raw table.
        for t in tables:
            if "total" in t and "shells" not in t:
                print("Error: invalid interaction table: 'total' applies to "
                      "the shells form only (compact / compact_npz give the "
                      "table directly); remove it.", file=sys.stderr)
                return 2
        try:
            interactions = [Interaction.from_config(dict(t), A) for t in tables]
        except (ValueError, TypeError, KeyError, OSError) as exc:
            print(f"Error: invalid interaction table: {exc}", file=sys.stderr)
            return 2
        # The CSV identifies a sweep point by (tail nu, label) only, so
        # two points under one label would write indistinguishable rows.
        labels = [v.default_label for v in interactions]
        dups = sorted({lab for lab in labels if labels.count(lab) > 1})
        if dups:
            print(f"Error: duplicate interaction labels {dups}; the CSV "
                  f"identifies a sweep point by its label, so give each "
                  f"[[interactions]] entry a distinct `label`.",
                  file=sys.stderr)
            return 2

    if interactions is not None:
        nus = list(interactions)
        nu_step = None
    else:
        nu_single = _resolve(args.nu, eval_cfg, "nu")
        if nu_single is not None:
            nus = np.array([_parse_nu_cli(nu_single)])
            nu_step = None
        else:
            nu_start = _resolve(args.nu_start, eval_cfg, "nu_start")
            nu_end = _resolve(args.nu_end, eval_cfg, "nu_end")
            nu_step = _resolve(args.nu_step, eval_cfg, "nu_step")
            if not (nu_start is not None and nu_end is not None and nu_step is not None):
                print("Error: provide --nu, or all of nu_start/nu_end/nu_step "
                      "(via --nu-start/--nu-end/--nu-step or [evaluation].nu_*), "
                      "or an [interaction] table.",
                      file=sys.stderr)
                return 2
            nus = nu_grid(float(nu_start), float(nu_end), float(nu_step))

    routing_cfg = cfg.get("routing", {})
    # These four tuned the k = 0 fallback cascade the library no longer
    # has.  A flag or config key that sets one is still accepted, so an
    # old config keeps running, and it is reported rather than silently
    # inert, which is how `high_tw_fallback` once sat in the shipped
    # configs.  None is forwarded.
    inert = [key for key, flag in (
        ("nu_tensor_threshold", args.nu_tensor_threshold),
        ("tw_threshold", args.tw_threshold),
        ("sigma_max", args.sigma_max),
        ("high_tw_fallback", None),
    ) if _resolve(flag, routing_cfg, key) is not None]
    if inert:
        print(f"gzl: warning: {', '.join(inert)} "
              f"{'has' if len(inert) == 1 else 'have'} no effect and will "
              f"be removed: the k = 0 fallback cascade they tuned is gone.",
              file=sys.stderr)
    dense_engine = _resolve(args.dense_engine, routing_cfg, "dense_engine")
    _sp_n = _resolve(args.sp_n_points, routing_cfg, "sp_n_points")
    sp_n_points = None if _sp_n is None else int(_sp_n)
    # Default TRUE, matching `compute_series_coefficients` and
    # `evaluate_graph`.  This read `... or False`, so with no flag and no
    # `[routing] core_grading` key -- and no shipped config sets one --
    # it resolved to `bool(None or False) == False`, and the documented
    # CLI produced a different number, and a very different cost, from
    # the same pass driven through the library.  Same failure as the
    # `--richardson` default: a wrapper quietly disagreeing with the
    # library it wraps.
    core_grading = bool(_resolve(args.core_grading, routing_cfg,
                                 "core_grading", default=True))

    momentum_arg = _resolve(args.momentum, eval_cfg, "momentum")
    try:
        momentum = _parse_momentum_cli(momentum_arg)
    except (ValueError, TypeError) as exc:
        print(f"Error: --momentum {momentum_arg!r} is not 'none', "
              f"'default', or a JSON number / array: {exc}", file=sys.stderr)
        return 2

    out_cfg = cfg.get("output", {})
    output_path = _resolve(args.output, out_cfg, "csv")
    if output_path is None:
        print("Error: --output or [output].csv is required.", file=sys.stderr)
        return 2
    output_path = Path(output_path)
    diag_path = _resolve(args.diagnostics_csv, out_cfg, "diagnostics_csv")
    diag_path = Path(diag_path) if diag_path else None

    say(f"Corpus     : {corpus_path}"
        + ("" if corpus_path == Path(corpus_arg) else
           f"  ({corpus_arg}, default)" if corpus_default else
           f"  ({corpus_arg})"))
    say(f"Lattice A  : {A.tolist()}  (d = {A.shape[0]})"
        + (f"  ({A_arg})" if isinstance(A_arg, str) and A_arg in LATTICES
           else ""))
    say(f"n_points   : {n_points}")
    say(f"order_max  : {order_max}")
    if interactions is not None:
        say("interaction: " + ", ".join(
            f"{v.default_label} (tail nu = {v.tail_exponent:g})" for v in nus))
    elif nu_step is not None:
        say(f"ν grid     : {nus[0]:.4f} ... {nus[-1]:.4f} step {float(nu_step):g} "
            f"({nus.size} points)")
    else:
        say(f"ν          : {', '.join(str(x) for x in nus.tolist())}")
    # Without --momentum the corpus decides (see `_parse_momentum`), and
    # the banner is printed before the corpus is loaded, so it states the
    # rule.  It said "none (k=0)" here, also for a 1qp pass that returns
    # the whole grid.
    if momentum is None:
        if n_points > 0:
            grid = (f"{n_points}-point" if A.shape[0] == 1
                    else "x".join([str(n_points)] * A.shape[0]))
            one_qp = f"{grid} Brillouin-zone grid"
        else:
            one_qp = "Brillouin-zone grid, needs n_points > 0"
        momentum_label = f"default (0qp corpus: k = 0, 1qp corpus: {one_qp})"
    else:
        momentum_label = str(momentum_arg)
    say(f"momentum   : {momentum_label}")
    say(f"Output     : {output_path}"
        + ("  (standard output)" if str(output_path) == _STDOUT else ""))
    if diag_path:
        say(f"Diagnostics: {diag_path}")
    say()

    if args.dry_run:
        # Everything the pass would be given, after flags have overridden
        # config and config the defaults.  Two settings have silently
        # disagreed with the library in the past -- `high_tw_fallback`
        # was parsed and never read, and `core_grading` resolved through
        # `... or False` against a library default of True -- and neither
        # was visible from outside.  Printed on stdout: it is the answer
        # to the question asked, not chatter about it.
        print("compute_series_coefficients(")
        for key, value in (
            ("corpus", corpus_arg),
            ("nu", (f"{len(nus)} interaction(s)" if interactions is not None
                    else nus.tolist())),
            ("A", A.tolist()),
            ("n_points", n_points),
            ("momentum", momentum),
            ("order_max", order_max),
            ("dense_engine", dense_engine),
            ("sp_n_points", sp_n_points),
            ("core_grading", core_grading),
        ):
            print(f"    {key} = {value!r},")
        print(")")
        print(f"-> {output_path}" + (f", {diag_path}" if diag_path else ""))
        if args.per_graph:
            print("   (--per-graph: evaluate_corpus records, "
                  "not summed coefficients)")
        return 0

    if args.per_graph:
        return _run_per_graph(
            corpus_arg, nus, A, n_points, momentum, order_max,
            output_path,
            args.progress and not args.quiet,
            with_interaction=interactions is not None,
            routing=dict(dense_engine=dense_engine,
                         sp_n_points=sp_n_points,
                         core_grading=core_grading),
            say=say,
            traceback_wanted=traceback_wanted,
        )

    t0 = time.perf_counter()
    try:
        coefficients, diag = compute_series_coefficients(
            corpus_arg, nus, A, n_points,
            momentum=momentum,
            order_max=order_max,
            dense_engine=dense_engine,
            sp_n_points=sp_n_points,
            core_grading=core_grading,
            return_diagnostics=True,
            # -q wins: asking for quiet and for progress is a
            # contradiction, and quiet is the more specific request.
            progress=args.progress and not args.quiet,
        )
    except KeyboardInterrupt:
        print("\ngzl: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        return _fail(exc, traceback_wanted=traceback_wanted)
    wall = time.perf_counter() - t0

    # Recover trailing_shape and the k-grid (for extended-schema emit).
    # ``has_hopping`` for momentum=None default must be inferred from
    # the first coefficient array's shape (cheaper than re-loading the
    # corpus): trailing dims after the n_nu axis are exactly the
    # accumulator's ``trailing_shape``.
    d = int(A.shape[0])
    sample_arr = coefficients[next(iter(coefficients))]
    # Stripping axis 0 is only correct while axis 0 IS the sweep, which
    # holds because every `nus` this function builds is a list or an
    # array -- never a bare float, which the library would answer
    # without a sweep axis.  That is an invariant of three assignments
    # far above, so check it rather than trust it: the failure it
    # guards is silent, not loud.  A missing axis makes `shape[1:]` drop
    # a real k axis, and the writer then picks the legacy three-column
    # schema and emits one k-point of a whole-Brillouin-zone pass with
    # no indication that it did.
    if sample_arr.shape[:1] != (len(nus),):
        raise AssertionError(
            f"the pass returned arrays of shape {sample_arr.shape} for "
            f"{len(nus)} sweep point(s): axis 0 is not the sweep, so the "
            f"CSV schema cannot be inferred from it."
        )
    trailing_shape = sample_arr.shape[1:]   # strip the n_nu axis
    k_frac_flat = _build_k_frac_table(momentum, d, n_points, trailing_shape)

    # Series coefficients are real-valued; we keep ``compute_series_coefficients``
    # complex-typed for internal flexibility but emit only the real part to the
    # CSV.  A non-trivial imaginary residue would indicate a routing bug — warn
    # if any coefficient has |Im| > 1e-9.
    # The CSV's ``nu`` column is the coupling for a power-law sweep, and
    # the TAIL exponent for an interaction sweep, where an ``interaction``
    # column (the label) is added right after it -- so every reader that
    # joins on a numeric ``nu`` keeps working, and legacy files are
    # byte-identical.
    def _nu_cells(nu_val):
        if interactions is not None:
            tail = float(nu_val.tail_exponent) if is_interaction(nu_val) else float(nu_val)
            label = (nu_val.default_label if is_interaction(nu_val)
                     else f"nu={tail:g}")
            return [("inf" if not np.isfinite(tail) else f"{tail:.10f}"), label]
        return [f"{nu_val:.10f}"]

    nu_header = ["nu", "interaction"] if interactions is not None else ["nu"]
    n_sweep = len(nus)
    try:
        n_rows = _write_coefficients(
            output_path, coefficients, nus, nu_header, _nu_cells,
            trailing_shape, k_frac_flat, d)
    except (_OutputError, OSError) as exc:
        return _fail(exc, traceback_wanted=traceback_wanted)
    say(("\n" if args.progress and not args.quiet else "")
        + f"Wrote {n_rows} rows to "
        + ("standard output" if str(output_path) == _STDOUT
           else str(output_path)))

    if diag_path:
        try:
            _write_diagnostics(diag_path, diag)
        except (_OutputError, OSError) as exc:
            return _fail(exc, traceback_wanted=traceback_wanted)
        say(f"Wrote diagnostics to {diag_path}")

    say(f"Total wall: {wall:.1f}s")
    return 0


def _write_coefficients(output_path, coefficients, nus, nu_header, _nu_cells,
                        trailing_shape, k_frac_flat, d) -> int:
    """Emit the coefficient CSV; return the row count."""
    n_sweep = len(nus)
    with _open_output(output_path) as f:
        w = csv.writer(f)
        if not trailing_shape:
            # Legacy schema (momentum=None / single-k).
            w.writerow(["order"] + nu_header + ["coefficient"])
            for order in sorted(coefficients.keys()):
                arr = coefficients[order]
                for j, nu_val in enumerate(nus):
                    w.writerow([order] + _nu_cells(nu_val)
                               + [f"{float(arr[j]):.12e}"])
            n_rows = len(coefficients) * n_sweep
        else:
            # Extended schema for grid / batch.  ``k_frac_flat`` has
            # shape ``(prod(trailing_shape), d)`` and indexes the
            # flattened trailing axes in C-order (matching
            # ``arr_flat = arr[j].reshape(-1)``).
            header = (["order"] + nu_header + ["k_index"]
                      + [f"k_frac_{i}" for i in range(d)]
                      + ["coefficient"])
            w.writerow(header)
            n_k = k_frac_flat.shape[0]
            for order in sorted(coefficients.keys()):
                arr = coefficients[order]   # shape (n_nu, *trailing_shape)
                for j, nu_val in enumerate(nus):
                    arr_flat = np.asarray(arr[j]).reshape(-1)
                    for idx in range(n_k):
                        row = ([order] + _nu_cells(nu_val) + [idx]
                               + [f"{k_frac_flat[idx, i]:.10f}"
                                  for i in range(d)]
                               + [f"{float(arr_flat[idx]):.12e}"])
                        w.writerow(row)
            n_rows = len(coefficients) * n_sweep * n_k
    return n_rows


def _write_diagnostics(diag_path: Path, diag: dict) -> None:
    """Emit the two stacked diagnostics tables."""
    with _open_output(diag_path) as f:
        w = csv.writer(f)
        w.writerow(["order", "n_graphs", "wall_s"])
        for order in sorted(diag["n_graphs_per_order"]):
            w.writerow([order,
                        diag["n_graphs_per_order"][order],
                        f"{diag['wall_per_order'][order]:.4f}"])
        w.writerow([])
        w.writerow(["route", "n_graphs"])
        for k, v in diag.items():
            if k.startswith("n_") and isinstance(v, int):
                w.writerow([k[2:], v])
