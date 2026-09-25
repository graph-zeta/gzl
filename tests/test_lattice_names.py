# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""The Bravais lattices by name: ``"chain"``, ``"square"``,
``"triangular"``, ``"cubic"``.

Every public argument that takes a lattice matrix ``A`` -- the engines,
the builders, :class:`~gzl.Interaction`, the front end, the
series pipeline and the ``python -m gzl.series`` CLI (``--A`` and
``[lattice] A``) -- resolves a name through the one rule in
:func:`gzl._lattices._resolve_lattice`.

The oracle is the matrix the name replaces, written out here as
literals rather than read back from the library, so this file can catch
a wrong table entry.  A name must be that matrix to the LAST BIT: same
cache keys, same values, same CSV byte for byte.  Every comparison here
therefore goes through :func:`_bytes` -- raw bytes and ``float.hex()``,
never ``np.array_equal``, which calls ``-0.0`` and ``0.0`` equal.

Cheap settings throughout (a triangle at ``nu = 4.5``, small grids and
boxes, ``order_max = 3``), so the whole file stays around a second.
"""

from __future__ import annotations

import inspect
import itertools
import re
from pathlib import Path

import numpy as np
import pytest

import gzl as gz
from gzl import Interaction
from gzl.core import GraphZeta
from gzl._lattices import LATTICES, _resolve_lattice
from gzl.interaction import _lattice_fingerprint
from gzl.series import main


# ---------------------------------------------------------------------------
# The oracle: literals, not a copy of the library's table
# ---------------------------------------------------------------------------

#: What each name must mean, spelled out.  Columns are the primitive
#: vectors, so the triangular cell's columns are (1, 0) and (1/2, √3/2)
#: -- the 60-degree convention, whose zone corner is the fractional
#: momentum (1/3, -1/3) the Monte Carlo comparisons use.
WANT = {
    "chain": [[1.0]],
    "square": [[1.0, 0.0],
               [0.0, 1.0]],
    "triangular": [[1.0, 0.5],
                   [0.0, 0.8660254037844386]],
    "cubic": [[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]],
}

#: The one float64 every spelling of √3/2 in this repository gives.
#: ``np.cos(np.pi / 6)`` is one ulp above it and must never build the
#: triangular entry.
SQRT3_OVER_2_HEX = "0x1.bb67ae8584caap-1"

ALL = tuple(WANT)
MATRIX = {name: np.array(rows, dtype=float) for name, rows in WANT.items()}
DIM = {name: len(rows) for name, rows in WANT.items()}

# Cheap per-lattice settings: a torus grid, a box half-width, and a
# Richardson ladder of four rungs (the fit needs K + 1 = 4).
GRID = {"chain": 16, "square": 8, "triangular": 8, "cubic": 6}
BOX = {"chain": 4, "square": 3, "triangular": 3, "cubic": 2}
LADDER = {"chain": (4, 5, 6, 7), "square": (3, 4, 5, 6),
          "triangular": (3, 4, 5, 6), "cubic": (2, 3, 4, 5)}

EDGES = np.array([[0, 1], [1, 2], [2, 0]])      # a triangle: tw 2, cheap
NU = 4.5                                        # > d on every named lattice
N_POINTS = 8
ORDER_MAX = 3


def _bytes(value) -> bytes:
    """Raw bytes of a returned value, for bit identity.

    Never ``np.array_equal``: it reports ``[-0.0] == [0.0]``, so it
    cannot see the difference a wrong sign of zero in the table would
    make.  Floats go through ``float.hex()`` for the same reason.
    """
    if isinstance(value, GraphZeta):
        return _bytes((value.aMat, value.bVec, value.nuVec, value.A,
                       value.table_magnitude))
    if isinstance(value, Interaction):
        return _bytes((value.b, value.nu, value.compact, value.lattice))
    if value is None:
        return b"None"
    if isinstance(value, (bytes, bytearray)):
        return b"b" + bytes(value)
    if isinstance(value, str):
        return b"s" + value.encode()
    if isinstance(value, (tuple, list)):
        return b"(" + b",".join(_bytes(v) for v in value) + b")"
    if isinstance(value, (bool, int, np.integer)):
        return repr(int(value)).encode()
    if isinstance(value, (float, np.floating)):
        return float(value).hex().encode()
    if isinstance(value, (complex, np.complexfloating)):
        return _bytes((complex(value).real, complex(value).imag))
    arr = np.ascontiguousarray(value)
    return f"{arr.dtype}{arr.shape}".encode() + b"|" + arr.tobytes()


def _labels(d: int) -> np.ndarray:
    """Every label in ``{-1, 0, 1}^d``, the window a radius-1 table needs."""
    return np.array(list(itertools.product((-1, 0, 1), repeat=d)))


def _triangle_multigraph(kern=None):
    import networkx as nx
    mg = nx.MultiGraph()
    for i, (u, v) in enumerate(EDGES.tolist()):
        if kern is not None and i == 0:
            mg.add_edge(u, v, nu=kern.tail_exponent, kern=kern)
        else:
            mg.add_edge(u, v, nu=NU)
    return mg


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------

class TestResolver:

    def test_the_registry_names_the_four_lattices(self):
        assert {name: [list(row) for row in rows]
                for name, rows in LATTICES.items()} == WANT

    def test_sqrt3_over_2_is_the_one_float64(self):
        # Pinned against the literal AND against the arithmetic: every
        # spelling in this repository is this float, np.cos(np.pi / 6)
        # is one ulp above it.
        assert float(WANT["triangular"][1][1]).hex() == SQRT3_OVER_2_HEX
        assert float(np.sqrt(3.0) / 2.0).hex() == SQRT3_OVER_2_HEX
        assert float(LATTICES["triangular"][1][1]).hex() == SQRT3_OVER_2_HEX
        assert float(np.cos(np.pi / 6)).hex() != SQRT3_OVER_2_HEX

    @pytest.mark.parametrize("name", ALL)
    def test_a_name_is_its_matrix(self, name):
        assert _bytes(_resolve_lattice(name)) == _bytes(MATRIX[name])

    @pytest.mark.parametrize("name", ALL)
    def test_the_array_is_float64_square_and_c_contiguous(self, name):
        A = _resolve_lattice(name)
        d = DIM[name]
        assert A.dtype == np.float64 and A.shape == (d, d)
        assert A.flags["C_CONTIGUOUS"]

    @pytest.mark.parametrize("name", ALL)
    def test_every_call_returns_a_fresh_writable_array(self, name):
        # epsteinlib refuses a read-only buffer, and a shared array a
        # caller could mutate would be worse: a name must never hand out
        # the registry's own object.
        A = _resolve_lattice(name)
        assert A.flags["WRITEABLE"]
        A[0, 0] = 99.0
        assert _bytes(_resolve_lattice(name)) == _bytes(MATRIX[name])

    @pytest.mark.parametrize("name", ALL)
    def test_the_zeros_are_positive(self, name):
        # -0.0 and +0.0 are different bytes and different cache keys.
        assert not np.signbit(_resolve_lattice(name)).any()

    @pytest.mark.parametrize("value", [
        np.eye(2),
        np.array([[1.0, 0.5], [0.0, 0.8660254037844386]]),
        [[1, 0], [0, 1]],                     # the README's int spelling
        ((1.0, 0.0), (0.0, 1.0)),
        None,
    ])
    def test_anything_that_is_not_a_string_is_returned_as_itself(self, value):
        assert _resolve_lattice(value) is value

    @pytest.mark.parametrize("bad", [
        "hex", "trig", "hexagonal",           # not aliases: exact spelling
        "Triangular", "CUBIC",                # no case folding
        " cubic", "cubic ",                   # no stripping
        "", "chain2", "[[1.0]]",
    ])
    def test_any_other_string_is_refused(self, bad):
        with pytest.raises(ValueError) as exc:
            _resolve_lattice(bad)
        msg = str(exc.value)
        assert repr(bad) in msg
        for name in ALL:
            assert repr(name) in msg

    def test_a_numpy_string_is_a_name(self):
        # np.str_ is a str subclass; a label round-tripped through an NPZ
        # comes back as one.
        assert _bytes(_resolve_lattice(np.str_("square"))) == \
            _bytes(MATRIX["square"])

    @pytest.mark.parametrize("name", ALL)
    def test_a_name_in_the_nu_slot_is_refused(self, name):
        # nu is "nu-like" and a numeric string is silently accepted as an
        # exponent, so a name in the nu slot must raise, never evaluate.
        with pytest.raises(ValueError):
            gz.evaluate_graph([[0, 1]], name, name)


# ---------------------------------------------------------------------------
# The public surface: every entry that takes a lattice
# ---------------------------------------------------------------------------

def _public_entries() -> list[str]:
    """Every public callable with an ``A`` or ``lattice`` parameter.

    Read from the package rather than listed, so a new public function
    that forgets the resolver shows up here (and in the pinned list
    below) instead of shipping unwired.  The public surface is
    ``gzl.__all__`` plus the methods of the classes it exports;
    private methods (leading underscore, ``__init__`` excepted) are not
    part of it.
    """
    found = []
    for name in gz.__all__:
        obj = getattr(gz, name)
        if inspect.isclass(obj):
            for mname, member in vars(obj).items():
                if mname.startswith("_") and mname != "__init__":
                    continue
                func = (member.__func__
                        if isinstance(member, (classmethod, staticmethod))
                        else member)
                if not callable(func):
                    continue
                try:
                    sig = inspect.signature(func)
                except (TypeError, ValueError):
                    continue
                if {"A", "lattice"} & set(sig.parameters):
                    found.append(f"{name}.{mname}")
            try:
                sig = inspect.signature(obj)
            except (TypeError, ValueError):
                sig = None
            if sig is not None and {"A", "lattice"} & set(sig.parameters):
                found.append(f"{name}.__init__")
        elif callable(obj):
            try:
                sig = inspect.signature(obj)
            except (TypeError, ValueError):
                continue
            if {"A", "lattice"} & set(sig.parameters):
                found.append(name)
    return sorted(set(found))


ENTRIES = _public_entries()

#: Public entries that deliberately do NOT resolve a name, with the reason.
UNWIRED = {
    # A frozen dataclass that normalises no field: a list A is stored
    # as-is too.  make_graph_obj is the normalising constructor.
    "GraphZeta.__init__": "raw container, normalises nothing",
}


def test_the_entry_point_list_is_deliberate():
    # The discovery above is what the bit-identity cases run on; this
    # list only makes a new (or lost) public lattice argument a visible
    # change rather than a silent gap.
    assert ENTRIES == sorted([
        "GraphZeta.__init__",
        "Interaction.__init__",
        "Interaction.fourier_terms",
        "Interaction.from_config",
        "Interaction.from_function",
        "Interaction.from_shells",
        "Interaction.from_table",
        "Interaction.from_total",
        "Interaction.lattice_sum",
        "Interaction.nearest_neighbour",
        "Interaction.sample",
        "compute_series_coefficients",
        "direct_sum_extrapolated",
        "direct_sum_zero_momentum",
        "evaluate_corpus",
        "evaluate_graph",
        "graph_from_edges",
        "graph_from_edges_uniform",
        "graph_from_sp",
        "graph_from_sp_uniform",
        "graph_from_tw2",
        "graph_from_tw2_uniform",
        "graph_zeta_general",
        "graph_zeta_general_at_zero",
        "hybrid_zeta",
        "make_epstein_graph",
        "make_graph_obj",
        "slab_zeta",
        "zeta_circle",
    ])


def test_every_entry_is_either_covered_or_named_unwired():
    covered = set(CASES) | set(UNWIRED)
    assert set(ENTRIES) <= covered, sorted(set(ENTRIES) - covered)


# ---------------------------------------------------------------------------
# One call per entry, run twice: by name and by matrix
# ---------------------------------------------------------------------------

def _shells(A, name):
    return Interaction.from_shells(A, {1.0: 0.5}, b=(1.0,),
                                   nu=(float(DIM[name]) + 1.5,))


def _kernel(name):
    """A power-law-plus-table kernel on ``name``, built from the matrix."""
    return _shells(MATRIX[name], name)


#: ``entry -> (call, lattices)``.  ``call(A, name)`` takes the lattice
#: SPELLING (a name or the matrix) and the lattice's name for its own
#: cheap settings; the two spellings must return the same bytes.  The
#: corpus entries run on the chain only -- they iterate thousands of
#: graphs, and the router beneath them is covered lattice by lattice by
#: ``evaluate_graph``.
CASES = {
    "make_graph_obj": (
        lambda A, name: gz.make_graph_obj(
            np.zeros((4,) * DIM[name]), [1.0], [NU], A), ALL),
    "make_epstein_graph": (
        lambda A, name: gz.make_epstein_graph(NU, A, 4), ALL),
    "zeta_circle": (
        lambda A, name: gz.zeta_circle([NU] * 3, A), ALL),
    "graph_from_sp": (
        lambda A, name: gz.graph_from_sp(
            _triangle_multigraph(), 0, 2, A, GRID[name]), ALL),
    "graph_from_sp_uniform": (
        lambda A, name: gz.graph_from_sp_uniform(
            _triangle_multigraph(), 0, 2, NU, A, GRID[name]), ALL),
    "graph_from_tw2": (
        lambda A, name: gz.graph_from_tw2(
            _triangle_multigraph(), 0, 2, A, GRID[name]), ALL),
    "graph_from_tw2_uniform": (
        lambda A, name: gz.graph_from_tw2_uniform(
            _triangle_multigraph(), 0, 2, NU, A, GRID[name]), ALL),
    "graph_from_edges": (
        lambda A, name: gz.graph_from_edges(
            EDGES, [NU] * 3, A, GRID[name]), ALL),
    "graph_from_edges_uniform": (
        lambda A, name: gz.graph_from_edges_uniform(
            EDGES, NU, A, GRID[name]), ALL),
    "direct_sum_zero_momentum": (
        lambda A, name: gz.direct_sum_zero_momentum(
            EDGES, [NU] * 3, A, BOX[name]), ALL),
    "direct_sum_extrapolated": (
        lambda A, name: gz.direct_sum_extrapolated(
            EDGES, [NU] * 3, A, L_list=LADDER[name]), ALL),
    "graph_zeta_general": (
        lambda A, name: gz.graph_zeta_general(
            EDGES, [NU] * 3, A, GRID[name]), ALL),
    "graph_zeta_general_at_zero": (
        lambda A, name: gz.graph_zeta_general_at_zero(
            EDGES, [NU] * 3, A, GRID[name]), ALL),
    "hybrid_zeta": (
        lambda A, name: gz.hybrid_zeta(EDGES, [NU] * 3, A, GRID[name]), ALL),
    "slab_zeta": (
        lambda A, name: gz.slab_zeta(
            [(0, 1), (1, 2), (2, 0)], [NU] * 3, A, GRID[name]), ALL),
    "evaluate_graph": (
        lambda A, name: gz.evaluate_graph(
            EDGES.ravel(), NU, A, n_points=GRID[name]), ALL),
    "compute_series_coefficients": (
        lambda A, name: sorted(gz.compute_series_coefficients(
            "tfim1qp", 3.0, A, N_POINTS, order_max=ORDER_MAX,
            momentum=0.0).items()), ("chain",)),
    "evaluate_corpus": (
        lambda A, name: [(r["graph_id"], r["value"], r["contribution"])
                         for r in gz.evaluate_corpus(
                             "tfim1qp", 3.0, A, N_POINTS, order_max=ORDER_MAX,
                             momentum=0.0)], ("chain",)),
    "Interaction.__init__": (
        lambda A, name: Interaction(
            compact=Interaction.nearest_neighbour(MATRIX[name]).compact,
            lattice=A).lattice, ALL),
    "Interaction.from_table": (
        lambda A, name: Interaction.from_table(
            Interaction.nearest_neighbour(MATRIX[name]).compact,
            lattice=A).lattice, ALL),
    "Interaction.from_function": (
        lambda A, name: Interaction.from_function(
            lambda x: np.exp(-np.linalg.norm(x, axis=-1) ** 2), A, 1.2), ALL),
    "Interaction.from_total": (
        lambda A, name: Interaction.from_total(
            lambda x: np.exp(-np.linalg.norm(x, axis=-1) ** 2), A, 1.2,
            (1.0,), (float(DIM[name]) + 1.5,)), ALL),
    "Interaction.from_shells": (_shells, ALL),
    "Interaction.nearest_neighbour": (
        lambda A, name: Interaction.nearest_neighbour(A, 0.5), ALL),
    "Interaction.from_config": (
        lambda A, name: Interaction.from_config(
            {"label": "V", "b": [1.0], "nu": [float(DIM[name]) + 1.5],
             "shells": {"1.0": 0.5}}, A), ALL),
    "Interaction.sample": (
        lambda A, name: _kernel(name).sample(_labels(DIM[name]), A), ALL),
    "Interaction.fourier_terms": (
        lambda A, name: _kernel(name).fourier_terms(A), ALL),
    "Interaction.lattice_sum": (
        lambda A, name: _kernel(name).lattice_sum(A), ALL),
}


@pytest.mark.parametrize("entry", sorted(CASES))
def test_a_name_and_its_matrix_are_bit_identical(entry):
    call, lattices = CASES[entry]
    for name in lattices:
        by_name = _bytes(call(name, name))
        by_matrix = _bytes(call(MATRIX[name], name))
        assert by_name == by_matrix, (entry, name)


# ---------------------------------------------------------------------------
# The branches a float-only test would not reach
# ---------------------------------------------------------------------------

class TestKernelBranches:
    """Every entry with a separate Interaction path has TWO branches; a
    float-only check passes while the kernel one is still broken."""

    def test_zeta_circle_with_an_interaction_edge(self):
        # The Interaction branch forwards A before zeta_circle's own
        # np.array line, so the resolver has to be its first statement.
        V = _kernel("triangular")
        by_name = _bytes(gz.zeta_circle([V, 5.0, 5.0], "triangular"))
        by_matrix = _bytes(gz.zeta_circle([V, 5.0, 5.0],
                                          MATRIX["triangular"]))
        assert by_name == by_matrix

    @pytest.mark.parametrize("builder", ["graph_from_sp", "graph_from_tw2"])
    def test_sp_builders_with_an_interaction_edge(self, builder):
        # These never normalise A themselves: the raw value reaches the
        # per-edge leaf, and the Interaction leaf is a different leaf
        # from the power-law one.
        V = _kernel("triangular")
        build = getattr(gz, builder)
        mg = _triangle_multigraph(kern=V)
        by_name = _bytes(build(mg, 0, 2, "triangular", 8))
        by_matrix = _bytes(build(mg, 0, 2, MATRIX["triangular"], 8))
        assert by_name == by_matrix

    @pytest.mark.parametrize("compact_only", [True, False])
    def test_a_kernel_product_on_both_branches(self, compact_only):
        # V * W is a _KernelProduct, which forwards the raw A to each
        # factor on the non-compact branch and resolves it itself on the
        # purely compact one.
        A, name = MATRIX["triangular"], "triangular"
        if compact_only:
            product = (Interaction.nearest_neighbour(A)
                       * Interaction.nearest_neighbour(A, 0.5))
        else:
            product = _kernel(name) * Interaction.power_law(1.0, 5.0)
        labels = _labels(2)
        for call in (lambda X: product.sample(labels, X),
                     lambda X: product.fourier_terms(X),
                     lambda X: product.lattice_sum(X)):
            assert _bytes(call(name)) == _bytes(call(A))

    def test_evaluate_graph_in_multigraph_positional_form(self):
        # evaluate_graph(mg, "square") lands the name in nu, and the
        # MultiGraph swap moves it to A: the resolver must run after it.
        mg = _triangle_multigraph()
        assert _bytes(gz.evaluate_graph(mg, "square", n_points=8)) == \
            _bytes(gz.evaluate_graph(mg, MATRIX["square"], n_points=8))

    def test_a_kernel_built_by_name_evaluates_on_the_matrix(self):
        # The stored fingerprint is the matrix's, not the name's, so
        # _check_lattice passes either way round.
        A, name = MATRIX["triangular"], "triangular"
        by_name, by_matrix = _shells(name, name), _shells(A, name)
        assert by_name.lattice == by_matrix.lattice == _lattice_fingerprint(A)
        labels = _labels(2)
        assert _bytes(by_name.sample(labels, A)) == \
            _bytes(by_matrix.sample(labels, name))

    def test_an_unknown_name_is_refused_by_the_constructor(self):
        # Resolved OUTSIDE the constructor's try, so the list of names
        # survives instead of becoming the generic TypeError.
        tab = Interaction.nearest_neighbour(MATRIX["square"]).compact
        with pytest.raises(ValueError) as exc:
            Interaction(compact=tab, lattice="hexagonal")
        assert "'triangular'" in str(exc.value)


# ---------------------------------------------------------------------------
# A shared block cache serves both spellings
# ---------------------------------------------------------------------------

def test_a_block_cache_is_shared_between_the_two_spellings():
    # Every cache key is built from the normalised matrix, so a name and
    # its matrix hit the SAME entries: the second call adds nothing.
    cache: dict = {}
    first = gz.evaluate_graph(EDGES.ravel(), NU, "square", n_points=8,
                              block_cache=cache)
    size = len(cache)
    assert size > 0
    second = gz.evaluate_graph(EDGES.ravel(), NU, MATRIX["square"],
                               n_points=8, block_cache=cache)
    assert len(cache) == size
    assert _bytes(first) == _bytes(second)


# ---------------------------------------------------------------------------
# The series CLI
# ---------------------------------------------------------------------------

CUBIC_JSON = "[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]"


def _cli_flags(A: str, out_csv: Path) -> list[str]:
    return ["--corpus", "tfim1qp", "--A", A, "--n-points", "4",
            "--order-max", str(ORDER_MAX), "--nu", str(NU),
            "--output", str(out_csv)]


def _write_config(path: Path, A: str, out_csv: Path) -> Path:
    path.write_text(
        "[corpus]\n"
        'path = "tfim1qp"\n'
        "\n"
        "[lattice]\n"
        f"A = {A}\n"
        "\n"
        "[evaluation]\n"
        "n_points = 4\n"
        f"order_max = {ORDER_MAX}\n"
        f"nu = {NU}\n"
        "\n"
        "[output]\n"
        f'csv = "{out_csv.as_posix()}"\n'
    )
    return path


class TestCli:

    def test_the_A_flag_takes_a_name(self, tmp_path, capsys):
        by_json = tmp_path / "by_json.csv"
        by_name = tmp_path / "by_name.csv"

        assert main(_cli_flags(CUBIC_JSON, by_json)) == 0
        banner_json = capsys.readouterr().err
        assert main(_cli_flags("cubic", by_name)) == 0
        banner_name = capsys.readouterr().err

        assert by_name.read_bytes() == by_json.read_bytes()
        assert len(by_json.read_text().splitlines()) > 1
        # The run log names the matrix used, and the name it came from; a
        # matrix argument's banner line is what it always was.
        assert f"Lattice A  : {MATRIX['cubic'].tolist()}  (d = 3)  (cubic)\n" \
            in banner_name
        assert f"Lattice A  : {MATRIX['cubic'].tolist()}  (d = 3)\n" \
            in banner_json

    def test_the_config_takes_a_name(self, tmp_path):
        by_json = tmp_path / "cfg_json.csv"
        by_name = tmp_path / "cfg_name.csv"
        assert main(["--config", str(_write_config(
            tmp_path / "json.toml", CUBIC_JSON, by_json))]) == 0
        assert main(["--config", str(_write_config(
            tmp_path / "name.toml", '"cubic"', by_name))]) == 0
        assert by_name.read_bytes() == by_json.read_bytes()
        assert len(by_json.read_text().splitlines()) > 1

    def test_the_json_spelling_of_a_name_is_the_same_name(self, tmp_path):
        # '"cubic"' json-decodes to the str 'cubic': one rule, not two.
        by_bare = tmp_path / "bare.csv"
        by_quoted = tmp_path / "quoted.csv"
        assert main(_cli_flags("cubic", by_bare)) == 0
        assert main(_cli_flags('"cubic"', by_quoted)) == 0
        assert by_quoted.read_bytes() == by_bare.read_bytes()

    @pytest.mark.parametrize("bad", ["hexagonal", "cubik", '"hex"'])
    # A bad lattice is the command used wrongly: exit code 2 and the
    # message on stderr, like every other refusal.  It used to raise.
    # The messages themselves are what these pin, and they are unchanged.

    def test_an_unknown_name_lists_the_names(self, tmp_path, capsys, bad):
        assert main(_cli_flags(bad, tmp_path / "out.csv")) == 2
        msg = capsys.readouterr().err
        for name in ALL:
            assert repr(name) in msg

    def test_a_non_square_matrix_keeps_its_message(self, tmp_path, capsys):
        assert main(_cli_flags("[[1.0, 0.0]]", tmp_path / "out.csv")) == 2
        assert "must be a square matrix" in capsys.readouterr().err

    def test_a_top_level_lattice_value_names_the_right_spelling(self,
                                                                tmp_path,
                                                                capsys):
        # `lattice = "cubic"` at the top level passes the required-section
        # check (the key is there) and then looks like a missing A.
        cfg = tmp_path / "flat.toml"
        cfg.write_text('lattice = "cubic"\n\n[evaluation]\nn_points = 4\n')
        assert main(["--config", str(cfg)]) == 2
        msg = capsys.readouterr().err
        assert "[lattice]" in msg and 'A = "cubic"' in msg


# ---------------------------------------------------------------------------
# The benchmark and the example configs
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]

#: Spellings of a cell that now HAS a name.  A script writing one of
#: these keeps a second copy of the library's table, which is how the
#: triangular cell came to be spelled five different ways in this tree;
#: the point of the names is that there is one.  ``np.eye(d)`` with a
#: variable is a hypercubic cell for whatever ``d`` the script was asked
#: for and is not a named lattice, so only literal 1 / 2 / 3 match.
_BY_HAND = {
    "np.eye(1|2|3)":   re.compile(r"np\.eye\(\s*[123]\s*[,)]"),
    "the chain cell":  re.compile(
        r"np\.(?:array|asarray)\(\s*\[\[\s*1\.0*\s*\]\]"),
    "sqrt(3)":         re.compile(r"sqrt\(\s*3"),
}


@pytest.mark.parametrize("path", sorted((REPO / "examples").glob("*.py")),
                         ids=lambda p: p.stem)
def test_no_example_spells_a_named_lattice_by_hand(path):
    src = path.read_text(encoding="utf-8")
    for what, pattern in _BY_HAND.items():
        hit = pattern.search(src)
        assert hit is None, (
            f"{path.name} spells {what} at offset {hit.start()}: "
            f"{src[hit.start():hit.start() + 60]!r}.  Pass the lattice's "
            f"name ({', '.join(ALL)})."
        )


def test_the_benchmark_keeps_no_lattice_table_of_its_own():
    # mc_comparison.py used to carry its own {name: (d, matrix)} table
    # under the name LATTICES.  It now reads the library's, so an
    # identifier LATTICES with a DIFFERENT shape must not come back --
    # `LATTICES[name][0]` would silently return a matrix row, not d.
    src = (REPO / "benchmarks" / "mc_comparison.py").read_text(
        encoding="utf-8")
    assert "from gzl._lattices import LATTICES" in src
    assert "\nLATTICES" not in src
    for what, pattern in _BY_HAND.items():
        assert pattern.search(src) is None, what


def test_the_outside_a_checkout_gate_exercises_a_name(monkeypatch, capsys):
    """The installed wheel must ship the table.

    ``gzl selftest`` is the only gate that runs outside a checkout --
    CI reaches it both through the console script and, from a clean
    room, through ``.github/scripts/wheel_smoke.py``.  Break what
    ``"chain"`` resolves to and the gate must notice; otherwise a wheel
    that shipped without the table would pass it.
    """
    from gzl import cli
    import gzl._lattices as lattices

    assert "from gzl._cli_selftest import run" in (
        REPO / ".github" / "scripts" / "wheel_smoke.py").read_text(
            encoding="utf-8")

    assert cli.main(["selftest"]) == 0
    capsys.readouterr()

    monkeypatch.setitem(lattices.LATTICES, "chain", ((2.0,),))
    assert cli.main(["selftest"]) == 1
    capsys.readouterr()


@pytest.mark.parametrize("config",
                         sorted((REPO / "examples").glob("*.toml")),
                         ids=lambda p: p.stem)
def test_the_example_configs_name_their_lattice(config):
    import tomllib
    with config.open("rb") as fh:
        data = tomllib.load(fh)
    assert data["lattice"]["A"] in LATTICES
