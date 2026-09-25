# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""``import graph_zeta`` still works, and it is the SAME library as ``gzl``.

The danger in a rename is not that the old spelling stops working; it is
that it keeps working as a second copy.  Two module objects mean two
block caches, two ``lru_cache``s and two sets of classes, so
``isinstance`` fails across the spellings and a cached block computed
under one name is invisible to the other.  These tests pin that the
alias is an alias.

Each subprocess starts a fresh interpreter, because ``sys.modules`` and
the meta-path finder are process-wide and the rest of the suite has
already imported ``gzl``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _run(code: str, *, warn: str = "ignore::DeprecationWarning") -> str:
    """Run ``code`` in a fresh interpreter, return its stdout."""
    out = subprocess.run(
        [sys.executable, "-W", warn, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0, f"stdout={out.stdout!r} stderr={out.stderr!r}"
    return out.stdout.strip()


class TestItIsOneLibraryNotTwo:

    def test_the_module_object_is_the_same(self):
        assert _run("""
            import graph_zeta, gzl
            print(graph_zeta is gzl)
        """) == "True"

    def test_submodules_are_the_same_object(self):
        assert _run("""
            import graph_zeta.series, gzl.series
            print(graph_zeta.series is gzl.series)
        """) == "True"

    def test_classes_are_the_same_so_isinstance_holds(self):
        assert _run("""
            from graph_zeta import GraphZeta as Old
            from gzl import GraphZeta as New
            import numpy as np
            from gzl import make_epstein_graph
            g = make_epstein_graph(3.5, "chain", 8)
            print(Old is New, isinstance(g, Old))
        """) == "True True"

    def test_the_caches_are_shared(self):
        # A block cache primed through one spelling must serve the other.
        assert _run("""
            import graph_zeta, gzl
            cache = {}
            edges = [[0, 1], [1, 2], [0, 2]]
            graph_zeta.evaluate_graph(edges, 3.5, "chain", block_cache=cache)
            primed = len(cache)
            gzl.evaluate_graph(edges, 3.5, "chain", block_cache=cache)
            print(primed > 0, len(cache) == primed)
        """) == "True True"

    def test_a_value_is_identical_through_either_spelling(self):
        assert _run("""
            import graph_zeta, gzl
            edges = [[0, 1], [1, 2], [0, 2]]
            a = graph_zeta.evaluate_graph(edges, 3.5, "square")
            b = gzl.evaluate_graph(edges, 3.5, "square")
            print(a.hex() == b.hex())
        """) == "True"


class TestEveryImportShape:

    @pytest.mark.parametrize("code", [
        "import graph_zeta; print(graph_zeta.data_path('').name)",
        "import graph_zeta.series as s; print(s.__name__.split('.')[-1])",
        "from graph_zeta.series import compute_series_coefficients as f; print(f.__name__[:7])",
        "from graph_zeta import evaluate_graph; print(evaluate_graph.__module__.split('.')[0])",
        "from graph_zeta._lattices import LATTICES; print(len(LATTICES))",
    ])
    def test_cold_import_shapes_work(self, code):
        # Each of these is the FIRST import in its interpreter, so the
        # finder has to be installed by the parent package itself.
        assert _run(code) != ""

    def test_the_cli_runs_under_the_old_module_path(self, tmp_path):
        out = subprocess.run(
            [sys.executable, "-W", "ignore::DeprecationWarning",
             "-m", "graph_zeta.series", "--help"],
            capture_output=True, text=True,
        )
        assert out.returncode == 0 and "--corpus" in out.stdout

    def test_pickles_cross_the_two_spellings(self):
        assert _run("""
            import pickle, graph_zeta, gzl, sys
            g = graph_zeta.make_epstein_graph(3.5, "chain", 8)
            blob = pickle.dumps(g)
            del sys.modules["graph_zeta"]
            back = pickle.loads(blob)
            print(type(back).__module__.split('.')[0], isinstance(back, gzl.GraphZeta))
        """) == "gzl True"


class TestItSaysSo:

    def test_importing_the_old_name_warns_once_and_names_the_new_one(self):
        assert _run("""
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                import graph_zeta                      # noqa: F401
            msgs = [str(x.message) for x in w
                    if issubclass(x.category, DeprecationWarning)]
            print(len(msgs), "gzl" in msgs[0], "2.0" in msgs[0])
        """, warn="default") == "1 True True"

    def test_gzl_itself_warns_about_nothing(self):
        assert _run("""
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                import gzl                             # noqa: F401
            print([str(x.message) for x in w
                   if issubclass(x.category, DeprecationWarning)])
        """, warn="default") == "[]"
