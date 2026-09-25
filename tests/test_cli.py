# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""The ``gzl`` command: dispatch, and the two subcommands that are not
``series``.

``series`` itself is covered by :mod:`tests.test_series_cli` and
:mod:`tests.test_series_cli_paths`, which drive ``gzl.series.main`` --
the same function ``gzl series`` reaches.  What is tested here is that
the umbrella routes to it unchanged, that ``info`` reports the things a
bug report needs, and that ``selftest`` is a check that can fail.

That last one is the point of ``TestSelftestCanFail``.  A self-test
which passes because it never looks is worse than none, so each of its
three checks is broken on purpose and must be caught.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import gzl
from gzl import cli


class TestUmbrella:
    """Dispatch, version and help."""

    def test_version_flag(self, capsys):
        assert cli.main(["--version"]) == 0
        assert capsys.readouterr().out.strip() == f"gzl {gzl.__version__}"

    def test_bare_command_prints_help(self, capsys):
        assert cli.main([]) == 0
        out = capsys.readouterr().out
        assert "usage: gzl" in out
        for name in ("series", "info", "selftest"):
            assert name in out

    def test_unknown_subcommand_is_a_usage_error(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["nosuchthing"])
        assert exc.value.code == 2
        assert "nosuchthing" in capsys.readouterr().err

    def test_every_registered_subcommand_has_help(self, capsys):
        for name in cli._SUBCOMMANDS:
            with pytest.raises(SystemExit) as exc:
                cli.main([name, "--help"])
            assert exc.value.code == 0
            out = capsys.readouterr().out
            assert out.startswith(f"usage: gzl {name}"), (name, out[:80])

    def test_nothing_imports_the_cli_modules(self):
        """``import gzl`` must not pull them in.

        This is what makes ``python -m gzl`` and the console script run
        the driver exactly once; if a future edit imports either module
        from the package, runpy starts warning again.
        """
        code = ("import sys, gzl; "
                "print([m for m in sys.modules "
                "if m in ('gzl.cli', 'gzl._cli_series', "
                "'gzl._cli_info', 'gzl._cli_selftest')])")
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "[]", out.stdout


class TestSeriesDispatch:
    """``gzl series`` is the same driver, through the same parser."""

    ARGS = ["--corpus", "tfim0qp", "--A", "chain", "--n-points", "8",
            "--order-max", "3", "--nu", "3.0"]

    def test_matches_the_legacy_entry_point(self, tmp_path):
        from gzl.series import main as legacy

        umbrella = tmp_path / "umbrella.csv"
        direct = tmp_path / "direct.csv"
        assert cli.main(["series", *self.ARGS,
                         "--output", str(umbrella)]) == 0
        assert legacy([*self.ARGS, "--output", str(direct)]) == 0
        assert umbrella.read_text() == direct.read_text()

    def test_usage_errors_pass_through(self, tmp_path):
        """A subcommand's exit code is the command's exit code."""
        assert cli.main(["series", "--corpus", "tfim0qp", "--A", "chain",
                         "--n-points", "8", "--nu", "3.0"]) == 2


class TestInfo:
    def test_reports_what_a_bug_report_needs(self, capsys):
        assert cli.main(["info"]) == 0
        out = capsys.readouterr().out
        assert gzl.__version__ in out
        for field in ("gzl", "location", "python", "dependencies",
                      "data", "corpora", "lattices"):
            assert field in out
        # CONTRIBUTING asks for these two by name.
        assert "numpy" in out and "epsteinlib" in out
        assert "tfim0qp" in out and "tfim1qp" in out
        assert "triangular" in out

    def test_names_the_package_that_answered(self, capsys):
        """With several worktrees on one machine, which one answered is
        the question behind most "it worked yesterday" reports."""
        from pathlib import Path

        cli.main(["info"])
        out = capsys.readouterr().out
        assert str(Path(gzl.__file__).resolve().parent) in out


class TestSelftest:
    def test_passes_on_this_checkout(self, capsys):
        assert cli.main(["selftest"]) == 0
        out = capsys.readouterr().out
        assert out.rstrip().endswith("OK")
        assert "all sha256 match" in out


class TestSelftestCanFail:
    """Each check, broken on purpose.

    A self-test that cannot fail is not evidence.  These pin that every
    check actually looks at what it claims to look at, and that a
    failure is reported on stderr with a non-zero exit code.
    """

    def test_a_wrong_closed_form_is_caught(self, monkeypatch, capsys):
        import numpy as np

        real = gzl.compute_series_coefficients

        def wrong(*args, **kwargs):
            out = real(*args, **kwargs)
            return {o: np.asarray(v) * 1.5 for o, v in out.items()}

        monkeypatch.setattr(gzl, "compute_series_coefficients", wrong)
        assert cli.main(["selftest"]) == 1
        err = capsys.readouterr().err
        assert "FAILED" in err
        assert "-pi^2/3" in err

    def test_a_grid_dependent_order_is_caught(self, monkeypatch, capsys):
        """Orders 1-3 are closed forms; if one moves with n_points,
        something is routing them through a truncation."""
        import numpy as np

        real = gzl.compute_series_coefficients
        seen: list[int] = []

        def drifting(corpus, nu, A, n_points, **kwargs):
            out = real(corpus, nu, A, n_points, **kwargs)
            seen.append(n_points)
            bump = 1.0 + 1e-9 * len(seen)
            return {o: np.asarray(v) * bump for o, v in out.items()}

        monkeypatch.setattr(gzl, "compute_series_coefficients", drifting)
        assert cli.main(["selftest"]) == 1
        assert "moved with the grid" in capsys.readouterr().err

    def test_a_corrupt_data_file_is_caught(self, monkeypatch, tmp_path,
                                           capsys):
        """The manifest check must read the bytes, not just the names."""
        import shutil

        from gzl._data import MANIFEST_NAME, data_path
        import gzl._data as _data

        root = tmp_path / "data"
        shutil.copytree(data_path(), root)
        victim = next(p for p in sorted(root.rglob("*.csv"))
                      if p.name != MANIFEST_NAME)
        victim.write_bytes(victim.read_bytes() + b"\n# tampered\n")

        monkeypatch.setattr(_data, "_data_root", lambda: root)
        assert cli.main(["selftest"]) == 1
        err = capsys.readouterr().err
        assert "FAILED" in err
        assert victim.name in err

    def test_a_reference_disagreement_is_caught(self, monkeypatch, capsys):
        """Order 3 against the shipped Monte Carlo series."""
        import numpy as np

        import gzl._cli_selftest as selftest

        real = gzl.compute_series_coefficients

        def only_order_3_wrong(*args, **kwargs):
            out = real(*args, **kwargs)
            if 3 in out and kwargs.get("order_max") == 3:
                out = dict(out)
                out[3] = np.asarray(out[3]) * 2.0
            return out

        # Break order 3 only, and only far enough to clear the Monte
        # Carlo error bar -- so this is the reference check firing, not
        # the closed-form one.
        monkeypatch.setattr(selftest, "_CHECKS",
                            (("reference", selftest._check_reference),))
        monkeypatch.setattr(gzl, "compute_series_coefficients",
                            only_order_3_wrong)
        assert cli.main(["selftest"]) == 1
        err = capsys.readouterr().err
        assert "sigma from the Monte" in err


class TestStreams:
    """stdout carries the CSV; everything else is stderr.

    Without that split ``--output -`` is useless, because the banner
    and ``--progress`` would land in the middle of the data.
    """

    ARGS = ["series", "--corpus", "tfim0qp", "--A", "chain",
            "--n-points", "8", "--order-max", "3", "--nu", "3.0"]

    def test_output_dash_writes_the_csv_to_stdout(self, tmp_path, capsys):
        assert cli.main([*self.ARGS, "--output", "-"]) == 0
        captured = capsys.readouterr()
        assert captured.out.splitlines()[0] == "order,nu,coefficient"
        assert "Corpus" in captured.err

        to_file = tmp_path / "f.csv"
        assert cli.main([*self.ARGS, "--output", str(to_file)]) == 0
        capsys.readouterr()
        # ``newline=""``: csv.writer emits CRLF, and ``read_text()`` would
        # translate it away, so the comparison has to be byte-for-byte or
        # it is not a comparison of what the two sinks wrote.
        with open(to_file, newline="") as fh:
            assert captured.out == fh.read()

    def test_progress_does_not_land_in_the_data(self, capsys):
        assert cli.main([*self.ARGS, "--progress", "--output", "-"]) == 0
        captured = capsys.readouterr()
        for line in captured.out.splitlines():
            assert line.count(",") == 2, line
        assert "O 2:" in captured.err

    def test_quiet_leaves_stderr_empty(self, capsys):
        assert cli.main([*self.ARGS, "-q", "--output", "-"]) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out.splitlines()[0] == "order,nu,coefficient"

    def test_quiet_wins_over_progress(self, capsys):
        assert cli.main([*self.ARGS, "-q", "--progress",
                         "--output", "-"]) == 0
        assert capsys.readouterr().err == ""


class TestDryRun:
    """``--dry-run`` prints what the pass would be given, and runs none."""

    ARGS = ["series", "--corpus", "tfim0qp", "--A", "chain",
            "--n-points", "8", "--order-max", "3", "--nu", "3.0"]

    def test_writes_nothing_and_returns_zero(self, tmp_path, capsys):
        out = tmp_path / "never.csv"
        assert cli.main([*self.ARGS, "--dry-run", "--output", str(out)]) == 0
        assert not out.exists()
        assert "compute_series_coefficients(" in capsys.readouterr().out

    def test_shows_the_settings_that_have_disagreed_before(self, tmp_path,
                                                           capsys):
        """`core_grading` once differed silently from the library the CLI
        wraps.  Whatever it resolves to, --dry-run must say so."""
        assert cli.main([*self.ARGS, "--dry-run",
                         "--output", str(tmp_path / "x.csv")]) == 0
        out = capsys.readouterr().out
        assert "core_grading = True" in out

    def test_the_cascade_settings_are_reported_and_not_passed(self, tmp_path,
                                                              capsys):
        """The four settings of the removed k = 0 fallback cascade are
        still accepted, as flags and as config keys, so an old config
        runs.  They are reported on stderr and left out of the call."""
        cfg = tmp_path / "c.toml"
        cfg.write_text('[lattice]\nA = "chain"\n\n[evaluation]\n'
                       "n_points = 8\norder_max = 3\nnu = 3.0\n"
                       '\n[routing]\nhigh_tw_fallback = "direct_sum"\n'
                       f'\n[output]\ncsv = "{(tmp_path / "o.csv").as_posix()}"\n')
        assert cli.main(["series", "--corpus", "tfim0qp",
                         "--config", str(cfg), "--sigma-max", "2.5",
                         "--nu-tensor-threshold", "3", "--tw-threshold", "3",
                         "--dry-run"]) == 0
        captured = capsys.readouterr()
        assert ("gzl: warning: nu_tensor_threshold, tw_threshold, "
                "sigma_max, high_tw_fallback have no effect") in captured.err
        for key in ("nu_tensor_threshold", "tw_threshold", "sigma_max",
                    "high_tw_fallback"):
            assert key not in captured.out

    def test_no_cascade_setting_no_warning(self, tmp_path, capsys):
        assert cli.main([*self.ARGS, "--dry-run",
                         "--output", str(tmp_path / "x.csv")]) == 0
        assert "warning" not in capsys.readouterr().err

    def test_reports_the_resolved_value_not_the_config_value(self, tmp_path,
                                                             capsys):
        cfg = tmp_path / "c.toml"
        cfg.write_text('[lattice]\nA = "chain"\n\n[evaluation]\n'
                       "n_points = 8\norder_max = 3\nnu = 3.0\n"
                       "\n[routing]\nsp_n_points = 12\n"
                       f'\n[output]\ncsv = "{(tmp_path / "o.csv").as_posix()}"\n')
        assert cli.main(["series", "--corpus", "tfim0qp",
                         "--config", str(cfg), "--sp-n-points", "24",
                         "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "sp_n_points = 24" in out      # the flag, not the config's 12
        assert "sp_n_points = 12" not in out


class TestExitCodes:
    """0 ran, 1 could not finish, 2 used wrongly."""

    ARGS = ["series", "--corpus", "tfim0qp", "--A", "chain",
            "--n-points", "8", "--order-max", "3"]

    def test_a_refused_pass_is_one_not_a_traceback(self, tmp_path, capsys):
        """nu <= d is refused by the frontend, deep inside the pass."""
        assert cli.main([*self.ARGS, "--nu", "0.5",
                         "--output", str(tmp_path / "o.csv")]) == 1
        err = capsys.readouterr().err
        assert err.rstrip().endswith(
            "gzl: (re-run with --traceback for the full stack trace)")
        assert "gzl: error:" in err
        assert "Traceback" not in err

    def test_an_unwritable_output_is_one(self, capsys):
        assert cli.main([*self.ARGS, "--nu", "3.0",
                         "--output", "/nonexistent-root/x.csv"]) == 1
        assert "cannot write" in capsys.readouterr().err

    def test_traceback_flag_restores_the_stack(self, tmp_path):
        # At nu = 0.5 the first refusal is the order-2 double edge, one
        # bundle of exponent 1.0 = d: a bridge at its pole.
        with pytest.raises(gzl.UnsupportedLatticeSumError) as exc:
            cli.main([*self.ARGS, "--nu", "0.5", "--traceback",
                      "--output", str(tmp_path / "o.csv")])
        assert "pole" in str(exc.value)

    def test_traceback_env_var_restores_the_stack(self, tmp_path,
                                                  monkeypatch):
        monkeypatch.setenv("GZL_TRACEBACK", "1")
        with pytest.raises(Exception):
            cli.main([*self.ARGS, "--nu", "0.5",
                      "--output", str(tmp_path / "o.csv")])

    @pytest.mark.parametrize("bad, needle", [
        (["--A", "nonsense"], "--A must"),
        (["--A", "chain", "--corpus", "tfim2qp"], "shipped corpus name"),
        (["--A", "chain", "--momentum", "{oops"], "--momentum"),
    ])
    def test_a_bad_argument_is_two(self, tmp_path, capsys, bad, needle):
        args = ["series", "--n-points", "8", "--order-max", "3",
                "--nu", "3.0", *bad, "--output", str(tmp_path / "o.csv")]
        if "--corpus" not in bad:
            args = [*args, "--corpus", "tfim0qp"]
        assert cli.main(args) == 2
        assert needle in capsys.readouterr().err
