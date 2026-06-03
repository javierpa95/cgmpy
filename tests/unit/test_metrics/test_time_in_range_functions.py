"""Tests for pure time-in-range functions."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from cgmpy.metrics.time_in_range import data_completeness, tar, tbr, tir


def test_tir_all_in_range():
    glucose = pd.Series([100.0, 110.0, 120.0])
    assert tir(glucose, 70, 180) == 100.0


def test_tir_partial_in_range():
    glucose = pd.Series([50.0, 60.0, 120.0, 200.0, 250.0])
    assert tir(glucose, 70, 180) == 20.0  # only 120 is in [70,180]


def test_tir_none_in_range():
    glucose = pd.Series([50.0, 60.0, 200.0, 250.0])
    assert tir(glucose, 70, 180) == 0.0  # all outside


def test_tir_empty():
    assert tir(pd.Series([], dtype=float), 70, 180) == 0.0


def test_tir_boundary_inclusive():
    glucose = pd.Series([70.0, 180.0, 69.0, 181.0])
    assert tir(glucose, 70, 180) == 50.0  # 70 and 180 are in range


def test_tar_all_above():
    glucose = pd.Series([200.0, 250.0, 300.0])
    assert tar(glucose, 180) == 100.0


def test_tar_none_above():
    glucose = pd.Series([100.0, 110.0, 120.0])
    assert tar(glucose, 180) == 0.0


def test_tar_mixed():
    glucose = pd.Series([100.0, 200.0, 150.0, 300.0])
    assert tar(glucose, 180) == 50.0


def test_tar_empty():
    assert tar(pd.Series([], dtype=float), 180) == 0.0


def test_tbr_all_below():
    glucose = pd.Series([50.0, 60.0, 65.0])
    assert tbr(glucose, 70) == 100.0


def test_tbr_none_below():
    glucose = pd.Series([100.0, 110.0, 120.0])
    assert tbr(glucose, 70) == 0.0


def test_tbr_mixed():
    glucose = pd.Series([50.0, 100.0, 60.0, 120.0])
    assert tbr(glucose, 70) == 50.0


def test_tbr_empty():
    assert tbr(pd.Series([], dtype=float), 70) == 0.0


def test_data_completeness_full():
    start = datetime(2024, 1, 1, 0, 0)
    timestamps = pd.Series([start + timedelta(minutes=5 * i) for i in range(288)])
    glucose = pd.Series([100.0] * 288)
    pct = data_completeness(glucose, timestamps, 5)
    assert 99.0 <= pct <= 100.0


def test_data_completeness_empty():
    assert data_completeness(pd.Series([], dtype=float), pd.Series([], dtype=object), 5) == 0.0


def test_data_completeness_single_reading():
    start = datetime(2024, 1, 1, 0, 0)
    timestamps = pd.Series([start])
    glucose = pd.Series([100.0])
    pct = data_completeness(glucose, timestamps, 5)
    assert pct == 100.0  # 1 reading out of 1 expected


def test_backward_compatibility():
    """Pure functions return same values as GlucoseAnalysis methods on real data."""
    from cgmpy import GlucoseAnalysis, GlucoseData

    data = GlucoseAnalysis(GlucoseData("tests/fixtures/synthetic/sine_24h.csv"))

    # TIR
    old_tir = data.TIR()
    new_tir = tir(data.glucose, data.targets.target_low, data.targets.target_high)
    assert old_tir == pytest.approx(new_tir)

    # TAR180
    old_tar = data.TAR180()
    new_tar = tar(data.glucose, 180)
    assert old_tar == pytest.approx(new_tar)

    # TBR70
    old_tbr = data.TBR70()
    new_tbr = tbr(data.glucose, 70)
    assert old_tbr == pytest.approx(new_tbr)
