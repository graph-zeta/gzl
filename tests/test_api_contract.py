# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
"""The API contract of gzl 1.x, held against the code.

DOCUMENTATION.md, section "API stability", divides the public names into
a STABLE part, which semantic versioning covers, and a PROVISIONAL part,
which may change in a minor release.  This file states that division in a
form the code can be checked against:

* every name in ``gzl.__all__`` is classified, so a new export is a
  decision and not an accident;
* every parameter of a stable function is classified, and a stable
  parameter keeps its name, its kind, its position and its default;
* the stable exceptions are ``GraphZetaError`` subclasses;
* every provisional name says so in its docstring;
* the stability section names every public name in the tier it has here;
* the signatures in the Core API table are the signatures in the code.

A red entry means a change touched the contract.  Editing the tables
below is legitimate only together with a CHANGELOG entry, and for a
stable name or parameter only in a new major version.  A provisional
parameter may be added, changed or removed in a minor release, but it
must be listed here, so the decision to add one is taken consciously.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re

import pytest

import gzl

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DOCS = _ROOT / "DOCUMENTATION.md"

REQUIRED = inspect.Parameter.empty
POS = "positional"          # positional-or-keyword
KW = "keyword"              # keyword-only

# ---------------------------------------------------------------------------
# The stable tier
# ---------------------------------------------------------------------------

#: Stable functions: ``{name: (stable parameters, provisional parameters)}``.
#: A stable parameter is ``name: (kind, default)``, listed in signature
#: order, and its position among the positional parameters is part of the
#: contract.  The provisional ones select or tune the numerical method.
STABLE_FUNCTIONS = {
    "evaluate_graph": (
        {
            "graph": (POS, REQUIRED),
            "nu": (POS, None),
            "A": (POS, None),
            "source": (KW, None),
            "terminal": (KW, None),
            "momentum": (KW, None),
            "n_points": (KW, 0),
            "return_diagnostics": (KW, False),
        },
        {"richardson", "fast_cycles", "block_cache", "engine",
         "dense_engine", "sp_n_points", "core_grading", "accuracy"},
    ),
    "compute_series_coefficients": (
        {
            "corpus_path": (POS, REQUIRED),
            "nu": (POS, REQUIRED),
            "A": (POS, REQUIRED),
            "n_points": (POS, REQUIRED),
            "momentum": (KW, None),
            "order_max": (KW, None),
            "return_diagnostics": (KW, False),
            "progress": (KW, False),
        },
        {"nu_tensor_threshold", "tw_threshold", "sigma_max",
         "high_tw_fallback", "direct_sum_L_list", "direct_sum_K",
         "fast_cycles", "richardson", "engine", "dense_engine",
         "sp_n_points", "core_grading"},
    ),
    "evaluate_corpus": (
        {
            "corpus_path": (POS, REQUIRED),
            "nu": (POS, REQUIRED),
            "A": (POS, REQUIRED),
            "n_points": (POS, REQUIRED),
            "momentum": (KW, None),
            "order_max": (KW, None),
            "progress": (KW, False),
        },
        {"nu_tensor_threshold", "sigma_max", "high_tw_fallback",
         "direct_sum_L_list", "direct_sum_K", "fast_cycles", "richardson",
         "engine", "dense_engine", "sp_n_points", "core_grading"},
    ),
    "zeta_circle": (
        {"nu_vec": (POS, REQUIRED), "A": (POS, REQUIRED)},
        set(),
    ),
    "data_path": (
        {"name": (POS, "")},
        set(),
    ),
    "nu_grid": (
        {"start": (POS, REQUIRED), "end": (POS, REQUIRED),
         "step": (POS, REQUIRED)},
        set(),
    ),
}

#: Stable exceptions.  Every one is a ``GraphZetaError``, and
#: ``GraphZetaError`` is a ``ValueError``.
STABLE_EXCEPTIONS = {
    "GraphZetaError",
    "SelfLoopError",
    "DisconnectedGraphError",
    "VertexOutOfRangeError",
    "NPointsRequiredError",
    "TopologyEvaluatorUnavailableError",
    "UnsupportedLatticeSumError",
    "UnsupportedRequestError",
}

STABLE_OTHER = {"__version__"}

STABLE = set(STABLE_FUNCTIONS) | STABLE_EXCEPTIONS | STABLE_OTHER

# ---------------------------------------------------------------------------
# The provisional tier
# ---------------------------------------------------------------------------

PROVISIONAL = {
    # general interaction kernels
    "Interaction", "InteractionSupportError",
    # the semi-analytical algebra
    "GraphZeta", "PrefactorSingularityError", "cNu", "make_graph_obj",
    "make_epstein_graph", "graph_sample", "graph_sample_at", "graph_zero",
    "graph_compress", "graph_multiply", "graph_multiply_power",
    "graph_convolve", "graph_convolve_power", "graph_attach",
    "periodic_convolve_nd",
    # the constructors
    "NotSeriesParallelError", "NotTreewidthTwoError",
    "graph_from_sp", "graph_from_sp_uniform", "graph_from_tw2",
    "graph_from_tw2_uniform", "graph_from_edges", "graph_from_edges_uniform",
    # the engines
    "graph_zeta_general", "graph_zeta_general_at_zero",
    "hybrid_zeta", "HybridCoreTooLargeError",
    "slab_zeta", "SlabTooLargeError",
    "direct_sum_zero_momentum", "direct_sum_extrapolated",
}

_PROVISIONAL_NOTE = "Provisional API"


def test_the_tiers_are_disjoint():
    assert not (STABLE & PROVISIONAL)


def test_every_export_is_classified():
    exported = set(gzl.__all__)
    assert exported == STABLE | PROVISIONAL, (
        f"unclassified exports: {sorted(exported - STABLE - PROVISIONAL)}; "
        f"classified but not exported: "
        f"{sorted((STABLE | PROVISIONAL) - exported)}"
    )


def test_every_export_resolves():
    for name in gzl.__all__:
        assert getattr(gzl, name) is not None, name


@pytest.mark.parametrize("name", sorted(STABLE_FUNCTIONS))
def test_stable_signature(name):
    stable, provisional = STABLE_FUNCTIONS[name]
    params = inspect.signature(getattr(gzl, name)).parameters

    # Every parameter is classified.
    assert set(params) == set(stable) | provisional, (
        f"{name}: unclassified {sorted(set(params) - set(stable) - provisional)}"
        f", missing {sorted((set(stable) | provisional) - set(params))}"
    )
    # A stable parameter keeps its kind and its default.
    for pname, (kind, default) in stable.items():
        p = params[pname]
        got_kind = (KW if p.kind is inspect.Parameter.KEYWORD_ONLY else POS)
        assert got_kind == kind, f"{name}({pname}): kind {got_kind}"
        assert p.default == default or (
            p.default is REQUIRED and default is REQUIRED
        ), f"{name}({pname}): default {p.default!r}, contract {default!r}"
    # ... and a positional one keeps its position.
    positional = [p for p in params
                  if params[p].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional == [p for p, (k, _) in stable.items() if k == POS], (
        f"{name}: positional parameters {positional}"
    )


@pytest.mark.parametrize("name", sorted(STABLE_EXCEPTIONS))
def test_stable_exception_is_a_graph_zeta_error(name):
    cls = getattr(gzl, name)
    assert issubclass(cls, gzl.GraphZetaError)
    assert issubclass(cls, ValueError)


#: The two resource refusals.  They are MemoryErrors, so that ``except
#: MemoryError`` sees them, and deliberately NOT GraphZetaErrors: that
#: would make every ``except ValueError`` swallow a byte-budget refusal.
MEMORY_REFUSALS = {"HybridCoreTooLargeError", "SlabTooLargeError"}


def test_every_exported_exception_is_a_graph_zeta_error_or_a_memory_refusal():
    exceptions = {name for name in gzl.__all__
                  if isinstance(getattr(gzl, name), type)
                  and issubclass(getattr(gzl, name), BaseException)}
    assert MEMORY_REFUSALS <= exceptions
    for name in sorted(exceptions):
        cls = getattr(gzl, name)
        if name in MEMORY_REFUSALS:
            assert issubclass(cls, MemoryError), name
            assert not issubclass(cls, gzl.GraphZetaError), name
        else:
            assert issubclass(cls, gzl.GraphZetaError), name


@pytest.mark.parametrize("name", sorted(PROVISIONAL))
def test_provisional_name_says_so(name):
    doc = getattr(gzl, name).__doc__ or ""
    assert _PROVISIONAL_NOTE in doc, (
        f"{name}: its docstring must carry the provisional note"
    )


@pytest.mark.parametrize("name", sorted(STABLE - STABLE_OTHER))
def test_stable_name_does_not_say_provisional(name):
    doc = getattr(gzl, name).__doc__ or ""
    assert _PROVISIONAL_NOTE not in doc, name


@pytest.mark.parametrize("name", sorted(
    n for n, (_, prov) in STABLE_FUNCTIONS.items() if prov))
def test_stable_function_names_its_provisional_parameters(name):
    """A stable function's docstring says which of its parameters are
    provisional, and names each of them."""
    doc = getattr(gzl, name).__doc__ or ""
    assert "provisional" in doc.lower(), name
    _, provisional = STABLE_FUNCTIONS[name]
    missing = sorted(p for p in provisional if f"``{p}``" not in doc)
    assert not missing, f"{name}: docstring does not name {missing}"


# ---------------------------------------------------------------------------
# The documentation
# ---------------------------------------------------------------------------

def _section(text: str, heading: str) -> str:
    """``heading`` and its body, up to the next heading of the same or a
    higher level."""
    start = text.index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    nxt = re.compile(rf"^#{{1,{level}}} ", re.M).search(
        text, start + len(heading))
    return text[start:] if nxt is None else text[start:nxt.start()]


def _names_in(block: str) -> set:
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", block))


def test_stability_section_names_every_public_name_in_its_tier():
    section = _section(_DOCS.read_text(), "## API stability")
    stable_part = _section(section, "### Stable")
    provisional_part = _section(section, "### Provisional")
    for name in STABLE - STABLE_OTHER:
        assert f"`{name}`" in stable_part, f"{name} missing under Stable"
    for name in PROVISIONAL:
        assert f"`{name}`" in provisional_part, (
            f"{name} missing under Provisional")
    # ... and names nothing in the wrong tier.
    assert not (_names_in(stable_part) & PROVISIONAL)
    assert not (_names_in(provisional_part) & STABLE)


def _core_api_rows():
    """Every table row of the Core API section, as ``(name, cells)``."""
    section = _section(_DOCS.read_text(), "## Core API")
    return re.findall(r"^\| `([A-Za-z_][A-Za-z_0-9]*)` \|(.*)$", section, re.M)


def _documented_signatures():
    """``(name, signature text)`` for every row of the Core API table."""
    out = []
    for name, rest in _core_api_rows():
        m = re.match(r"\s*`([^`]*)`", rest)
        out.append((name, m.group(1) if m else None))
    return out


def test_every_core_api_row_documents_a_signature():
    """A row whose second cell does not open with a backticked signature
    would otherwise be skipped by the check below without a word."""
    rows = _documented_signatures()
    assert rows
    assert all(sig is not None for _, sig in rows), (
        [name for name, sig in rows if sig is None])


def test_the_core_api_table_covers_the_stable_functions():
    documented = {name for name, _ in _documented_signatures()}
    assert set(STABLE_FUNCTIONS) <= documented


@pytest.mark.parametrize("name,sig", _documented_signatures())
def test_documented_signature_matches_the_code(name, sig):
    """The Core API table documents a parameter only under its real name,
    with its real kind and default, and the positional parameters in the
    real order.  A parameter shown without a default must have none.
    ``...`` stands for parameters the table leaves out at the end; a
    required parameter may not be left out."""
    text = sig[len(name):] if sig.startswith(name) else sig
    text = re.sub(r",\s*\.\.\.", "", text)
    args = ast.parse(f"def f{text}: pass").body[0].args
    assert args.vararg is None and args.kwarg is None, (
        f"{name}: the table shows *args / **kwargs")
    documented = {}
    positional = args.posonlyargs + args.args
    first_default = len(positional) - len(args.defaults)
    for i, arg in enumerate(positional):
        default = (ast.literal_eval(args.defaults[i - first_default])
                   if i >= first_default else REQUIRED)
        documented[arg.arg] = (POS, default)
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        documented[arg.arg] = (
            KW, REQUIRED if default is None else ast.literal_eval(default))

    params = inspect.signature(getattr(gzl, name)).parameters
    for pname, (kind, default) in documented.items():
        assert pname in params, f"{name}: no parameter {pname!r}"
        p = params[pname]
        got_kind = KW if p.kind is inspect.Parameter.KEYWORD_ONLY else POS
        assert got_kind == kind, f"{name}({pname}): documented as {kind}"
        if default is REQUIRED:
            assert p.default is REQUIRED, (
                f"{name}({pname}): documented as required, code has "
                f"default {p.default!r}")
        else:
            assert p.default == default, (
                f"{name}({pname}): documented default {default!r}, "
                f"code {p.default!r}")
    code_positional = [p for p, q in params.items()
                       if q.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    doc_positional = [a.arg for a in positional]
    assert code_positional[:len(doc_positional)] == doc_positional, (
        f"{name}: positional order documented {doc_positional}, code "
        f"{code_positional}")
    for pname, p in params.items():
        if p.default is REQUIRED and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY):
            assert pname in documented, (
                f"{name}: required parameter {pname!r} is not documented")
