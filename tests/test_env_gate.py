# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit
r"""The toolchain gate itself.

:mod:`tests._env_gate` decides whether the repo's frozen ``float.hex()``
pins are compared bitwise or against a calibrated ulp bound.  A gate
that cannot refuse certifies nothing, so the refusal is exercised here
in both directions, and the fingerprint is pinned against the one
``tests/fixtures/_freeze_executor_goldens.py`` already writes into its
payload -- two definitions of "same environment" that silently drift
apart would let one gate open while the other stays shut.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests import _env_gate
from tests._env_gate import (
    CROSS_ENV_ULP,
    FROZEN_ENV,
    assert_pinned,
    fingerprint,
    matches_freeze,
    ulp_distance,
)


def _freezer():
    spec = importlib.util.spec_from_file_location(
        "_freeze_executor_goldens",
        Path(__file__).parent / "fixtures" / "_freeze_executor_goldens.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_fingerprint_does_not_drift_from_the_goldens_one():
    """Both gates must mean the same thing by 'this environment'."""
    assert fingerprint() == _freezer().fingerprint()


def test_an_unidentified_freeze_claims_no_environment():
    """An empty record must match NOTHING.

    The tempting reading -- "no constraints recorded, so everything
    matches" -- would silently promote every machine to the bit-exact
    gate and reintroduce the spurious failures this module exists to
    remove.
    """
    assert matches_freeze({}) is False


def test_absent_fields_are_wildcards_and_present_ones_bind():
    here = fingerprint()
    assert matches_freeze({"system": here["system"]})
    assert matches_freeze(dict(here))
    assert not matches_freeze({**here, "numpy": "0.0.0-not-a-release"})
    assert not matches_freeze({"system": "NotAnOperatingSystem"})


def test_the_shipped_record_is_the_shape_fingerprint_writes():
    """A typo'd key would be a wildcard, i.e. a silently wider gate."""
    assert set(FROZEN_ENV) <= set(fingerprint())
    assert FROZEN_ENV, "an empty record disables the bit-exact gate entirely"


class TestAssertPinned:
    """The refusal, in both gate states."""

    PIN = "0x1.0000000000000p+0"          # exactly 1.0

    def test_the_exact_value_passes_under_either_gate(self, monkeypatch):
        for same in (True, False):
            monkeypatch.setattr(_env_gate, "SAME_ENV", same)
            assert_pinned(1.0, self.PIN, "unit")

    def test_a_few_ulp_passes_off_the_freezing_environment(self, monkeypatch):
        monkeypatch.setattr(_env_gate, "SAME_ENV", False)
        import math
        assert_pinned(1.0 + 8 * math.ulp(1.0), self.PIN, "unit")

    def test_a_few_ulp_is_refused_ON_the_freezing_environment(
            self, monkeypatch):
        """Zero tolerance still stands where refactors happen."""
        monkeypatch.setattr(_env_gate, "SAME_ENV", True)
        import math
        with pytest.raises(AssertionError, match="value change"):
            assert_pinned(1.0 + math.ulp(1.0), self.PIN, "unit")

    def test_beyond_the_bound_is_refused_everywhere(self, monkeypatch):
        import math
        over = 1.0 + (CROSS_ENV_ULP + 1) * math.ulp(1.0)
        for same in (True, False):
            monkeypatch.setattr(_env_gate, "SAME_ENV", same)
            with pytest.raises(AssertionError):
                assert_pinned(over, self.PIN, "unit")

    def test_the_bound_is_the_calibrated_one(self):
        """Tied to test_executor_goldens' calibration, not chosen here."""
        from tests.test_executor_goldens import CROSS_ENV_ULP as goldens_bound
        assert CROSS_ENV_ULP == goldens_bound


def test_ulp_distance_survives_zero_and_sign():
    assert ulp_distance(0.0, 0.0) == 0.0
    assert ulp_distance(-1.0, -1.0) == 0.0
    assert ulp_distance(1.0, 0.0) > 0.0
