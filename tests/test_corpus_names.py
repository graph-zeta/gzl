# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""The shipped corpora by name: ``"tfim0qp"`` and ``"tfim1qp"``.

Every corpus argument -- :func:`compute_series_coefficients`,
:func:`evaluate_corpus` and the ``python -m gzl.series`` CLI
(``--corpus`` and ``[corpus] path``) -- resolves a name through the one
rule in :func:`gzl._data._resolve_corpus`.

The oracle is the path spelling the name replaces,
``data_path("tfim_softcore_corpus_{0,1}qp.npz")``: a name must give the
same file, so the same values bit for bit and the same CSV byte for byte,
and every path argument must keep meaning what it meant.  Cheap settings
throughout (``order_max = 3``, ``n_points = 8``, d = 1), as in
``tests/test_series_cli.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gzl import compute_series_coefficients, data_path, evaluate_corpus
from gzl._data import CORPORA, _resolve_corpus
from gzl.series import main


FILES = {
    "tfim0qp": "tfim_softcore_corpus_0qp.npz",
    "tfim1qp": "tfim_softcore_corpus_1qp.npz",
}
A_1D = np.eye(1)
N_POINTS = 8
ORDER_MAX = 3
NU = 3.0

# 1qp at an explicit k = 0 (a scalar per order); 0qp at its own default.
KWARGS = {
    "tfim0qp": {},
    "tfim1qp": {"momentum": 0.0},
}


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------

class TestResolver:

    def test_the_registry_names_the_two_corpora(self):
        assert {name: CORPORA.get(name) for name in FILES} == FILES

    @pytest.mark.parametrize("name", sorted(CORPORA))
    def test_every_registered_name_ships(self, name):
        assert data_path(CORPORA[name]).is_file()

    @pytest.mark.parametrize("name", sorted(FILES))
    def test_a_name_is_the_shipped_file(self, name):
        got = _resolve_corpus(name)
        assert got == data_path(FILES[name])
        assert got.is_file()

    @pytest.mark.parametrize("name", sorted(FILES))
    def test_a_path_is_returned_as_itself(self, name):
        p = data_path(FILES[name])
        assert _resolve_corpus(p) == p
        assert _resolve_corpus(str(p)) == p

    @pytest.mark.parametrize("arg", [
        "some/dir/corpus.npz",       # relative, with a directory
        "corpus.h5",                 # relative, bare file
        "/abs/corpus.npz",           # absolute
        "tfim1qp.npz",               # a name with a suffix is a path
        "TFIM1QP",                   # names are exact: no case folding
        " tfim1qp",                  # ... and no stripping
    ])
    def test_any_other_string_is_a_path(self, arg):
        assert _resolve_corpus(arg) == Path(arg)

    def test_a_path_object_is_never_a_name(self):
        assert _resolve_corpus(Path("tfim1qp")) == Path("tfim1qp")


# ---------------------------------------------------------------------------
# Library entry points
# ---------------------------------------------------------------------------

class TestLibrary:

    @pytest.mark.parametrize("name", sorted(FILES))
    def test_compute_series_coefficients_is_bit_identical(self, name):
        by_name = compute_series_coefficients(
            name, NU, A_1D, N_POINTS, order_max=ORDER_MAX, **KWARGS[name])
        by_path = compute_series_coefficients(
            data_path(FILES[name]), NU, A_1D, N_POINTS, order_max=ORDER_MAX,
            **KWARGS[name])
        assert sorted(by_name) == sorted(by_path)
        assert by_path
        for order in by_path:
            assert np.array_equal(by_name[order], by_path[order]), order

    @pytest.mark.parametrize("name", sorted(FILES))
    def test_evaluate_corpus_is_bit_identical(self, name):
        by_name = evaluate_corpus(
            name, NU, A_1D, N_POINTS, order_max=ORDER_MAX, **KWARGS[name])
        by_path = evaluate_corpus(
            data_path(FILES[name]), NU, A_1D, N_POINTS, order_max=ORDER_MAX,
            **KWARGS[name])
        assert len(by_name) == len(by_path) > 0
        for got, want in zip(by_name, by_path):
            assert got.keys() == want.keys()
            for key in want:
                assert np.array_equal(got[key], want[key]), (key, want)

    @pytest.mark.parametrize("call", [compute_series_coefficients,
                                      evaluate_corpus])
    @pytest.mark.parametrize("bare", ["tfim2qp", "TFIM1QP"])
    def test_an_unknown_bare_string_lists_the_names(self, call, bare):
        with pytest.raises(FileNotFoundError) as exc:
            call(bare, NU, A_1D, N_POINTS, order_max=ORDER_MAX)
        msg = str(exc.value)
        assert msg.startswith(f"Corpus file not found: {bare} ")
        assert "'tfim0qp'" in msg and "'tfim1qp'" in msg

    @pytest.mark.parametrize("missing", [
        "no_such_dir/no_such_corpus.npz",
        "no_such_corpus.npz",
        Path("no_such_corpus.npz"),
        Path("tfim2qp"),
    ])
    def test_a_missing_path_keeps_its_message(self, missing):
        # Path-shaped arguments (a Path, a suffix, a directory) could not
        # have meant a name: their error is exactly what it always was.
        with pytest.raises(FileNotFoundError) as exc:
            compute_series_coefficients(
                missing, NU, A_1D, N_POINTS, order_max=ORDER_MAX)
        assert str(exc.value) == f"Corpus file not found: {Path(missing)}"

    def test_an_existing_directory_keeps_its_message(self, tmp_path,
                                                     monkeypatch):
        # A bare string that is a directory named something real, not a
        # mistyped name: no list of names.
        (tmp_path / "somedir").mkdir()
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError) as exc:
            compute_series_coefficients(
                "somedir", NU, A_1D, N_POINTS, order_max=ORDER_MAX)
        assert str(exc.value) == "Corpus file not found: somedir"


# ---------------------------------------------------------------------------
# The series CLI
# ---------------------------------------------------------------------------

def _cli_flags(corpus: str, out_csv: Path) -> list[str]:
    return ["--corpus", corpus, "--A", "[[1.0]]",
            "--n-points", str(N_POINTS), "--order-max", str(ORDER_MAX),
            "--nu", str(NU), "--output", str(out_csv)]


def _write_config(path: Path, corpus: str, out_csv: Path) -> Path:
    path.write_text(
        "[corpus]\n"
        f'path = "{corpus}"\n'
        "\n"
        "[lattice]\n"
        "A = [[1.0]]\n"
        "\n"
        "[evaluation]\n"
        f"n_points = {N_POINTS}\n"
        f"order_max = {ORDER_MAX}\n"
        f"nu = {NU}\n"
        "\n"
        "[output]\n"
        f'csv = "{out_csv.as_posix()}"\n'
    )
    return path


class TestCli:

    def test_corpus_flag_takes_a_name(self, tmp_path, capsys):
        by_path = tmp_path / "by_path.csv"
        by_name = tmp_path / "by_name.csv"
        shipped = data_path(FILES["tfim0qp"])

        assert main(_cli_flags(str(shipped), by_path)) == 0
        banner_path = capsys.readouterr().err
        assert main(_cli_flags("tfim0qp", by_name)) == 0
        banner_name = capsys.readouterr().err

        assert by_name.read_bytes() == by_path.read_bytes()
        assert len(by_path.read_text().splitlines()) > 1
        # The run log names the file used, and the name it came from; a
        # path argument's banner line is what it always was.
        assert f"Corpus     : {shipped}  (tfim0qp)\n" in banner_name
        assert f"Corpus     : {shipped}\n" in banner_path

    def test_config_path_takes_a_name(self, tmp_path, capsys):
        # A 1qp corpus at its default momentum: the full-BZ extended schema.
        by_path = tmp_path / "by_path.csv"
        by_name = tmp_path / "by_name.csv"
        shipped = data_path(FILES["tfim1qp"])
        cfg_path = _write_config(tmp_path / "by_path.toml",
                                 shipped.as_posix(), by_path)
        cfg_name = _write_config(tmp_path / "by_name.toml", "tfim1qp", by_name)

        assert main(["--config", str(cfg_path)]) == 0
        assert main(["--config", str(cfg_name)]) == 0
        banner_name = capsys.readouterr().err

        assert by_name.read_bytes() == by_path.read_bytes()
        assert by_path.read_text().splitlines()[0].startswith(
            "order,nu,k_index,k_frac_0,")
        assert f"Corpus     : {shipped}  (tfim1qp)\n" in banner_name

    def test_an_unknown_name_lists_the_names(self, tmp_path, capsys):
        # Exit code 2: naming a corpus that is not there is the command
        # used wrongly, like an unknown lattice.  It used to raise.
        assert main(_cli_flags("tfim2qp", tmp_path / "out.csv")) == 2
        err = capsys.readouterr().err
        assert "'tfim0qp'" in err and "'tfim1qp'" in err
