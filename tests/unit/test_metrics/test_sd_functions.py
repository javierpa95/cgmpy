"""Tests for pure SD/CV functions in cgmpy.metrics.variability.sd."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from cgmpy.metrics.variability.sd import (
    _get_segment_mask,
    cv_from_sd_mean,
    cv_global,
    mean_global,
    sd_between_timepoints,
    sd_daily_mean,
    sd_global,
    sd_interaction,
    sd_same_timepoint,
    sd_same_timepoint_adjusted,
    sd_segment,
    sd_within_day,
    sd_within_series,
    sdw,
)


def _make_multi_day_df(n_days: int) -> pd.DataFrame:
    """Build a deterministic multi-day synthetic dataset (5-min samples)."""
    start = datetime(2024, 1, 1, 0, 0)
    n = 288 * n_days
    times = [start + timedelta(minutes=5 * i) for i in range(n)]
    hours = np.arange(n) * 5 / 60.0
    glucose = 140 + 25 * np.sin(hours / 24 * 2 * np.pi) + np.random.default_rng(42).normal(0, 3, n)
    return pd.DataFrame({"time": times, "glucose": glucose})


# ── Simple: sd_global, mean_global ──────────────────────────


def test_sd_global():
    glucose = pd.Series([100.0, 110.0, 120.0])
    result = sd_global(glucose)
    assert result == pytest.approx(glucose.std())


def test_sd_global_constant():
    glucose = pd.Series([100.0, 100.0, 100.0])
    assert sd_global(glucose) == 0.0


def test_mean_global():
    glucose = pd.Series([100.0, 110.0, 120.0])
    assert mean_global(glucose) == 110.0


# ── sd_within_day, sdw ──────────────────────────────────────


def test_sd_within_day_returns_dict():
    df = _make_multi_day_df(3)
    result = sd_within_day(df["glucose"], df["time"])
    assert isinstance(result, dict)
    assert "sd" in result
    assert "mean" in result
    assert "valid_days" in result


def test_sdw_returns_float():
    df = _make_multi_day_df(2)
    result = sdw(df["glucose"], df["time"])
    assert isinstance(result, float)
    assert result >= 0.0


def test_sd_within_day_constant():
    start = datetime(2024, 1, 1, 0, 0)
    n = 288 * 2
    times = pd.Series([start + timedelta(minutes=5 * i) for i in range(n)])
    glucose = pd.Series([100.0] * n)
    result = sd_within_day(glucose, times)
    assert result["sd"] == pytest.approx(0.0)


# ── sd_daily_mean ───────────────────────────────────────────


def test_sd_daily_mean_returns_dict():
    df = _make_multi_day_df(3)
    result = sd_daily_mean(df["glucose"], df["time"])
    assert isinstance(result, dict)
    assert "sd" in result and "mean" in result
    assert "valid_days" in result and "total_days" in result
    assert result["valid_days"] >= 2


def test_sd_daily_mean_constant():
    start = datetime(2024, 1, 1, 0, 0)
    n = 288 * 3
    times = pd.Series([start + timedelta(minutes=5 * i) for i in range(n)])
    glucose = pd.Series([100.0] * n)
    result = sd_daily_mean(glucose, times)
    assert result["sd"] == pytest.approx(0.0)


# ── sd_between_timepoints ───────────────────────────────────


def test_sd_between_timepoints_returns_dict():
    df = _make_multi_day_df(2)
    result = sd_between_timepoints(df["glucose"], df["time"])
    for key in ("sd", "mean", "valid_timepoints", "total_timepoints", "median_count", "min_count", "max_count"):
        assert key in result
    assert isinstance(result["sd"], float | np.floating)


def test_sd_between_timepoints_grouping():
    df = _make_multi_day_df(2)
    result = sd_between_timepoints(
        df["glucose"], df["time"],
        filter_outliers=False,
        group_by_intervals=True,
        interval_minutes=5,
    )
    assert isinstance(result, dict)
    assert "sd" in result


# ── sd_same_timepoint ───────────────────────────────────────


def test_sd_same_timepoint_returns_dict():
    df = _make_multi_day_df(3)
    result = sd_same_timepoint(df["glucose"], df["time"])
    assert isinstance(result, dict)
    assert "sd" in result and "mean" in result
    assert "total_timepoints" in result


def test_sd_same_timepoint_no_filter():
    df = _make_multi_day_df(2)
    result = sd_same_timepoint(df["glucose"], df["time"], filter_outliers=False)
    assert isinstance(result["sd"], float | np.floating)


# ── sd_same_timepoint_adjusted ──────────────────────────────


def test_sd_same_timepoint_adjusted_returns_dict():
    df = _make_multi_day_df(2)
    result = sd_same_timepoint_adjusted(df["glucose"], df["time"])
    assert "sd" in result and "mean" in result
    assert isinstance(result["sd"], float | np.floating)


# ── _get_segment_mask, sd_segment ──────────────────────────


def test_get_segment_mask_returns_bool_series():
    df = _make_multi_day_df(1)
    mask = _get_segment_mask(df["time"], "08:00", 8)
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool
    assert mask.sum() > 0


def test_get_segment_mask_midnight_crossing():
    df = _make_multi_day_df(2)
    mask = _get_segment_mask(df["time"], "22:00", 8)
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool


def test_sd_segment_returns_dict():
    df = _make_multi_day_df(2)
    result = sd_segment(df["glucose"], df["time"], "08:00", 8)
    assert "sd" in result and "mean" in result
    assert isinstance(result["sd"], float | np.floating)


def test_sd_segment_crosses_midnight():
    df = _make_multi_day_df(2)
    result = sd_segment(df["glucose"], df["time"], "22:00", 8)
    assert "sd" in result and "mean" in result


def test_sd_segment_empty_returns_zero():
    start = datetime(2024, 1, 1, 12, 0)
    n = 144
    times = pd.Series([start + timedelta(minutes=5 * i) for i in range(n)])
    glucose = pd.Series([120.0] * n)
    result = sd_segment(glucose, times, "00:00", 6)
    assert result["sd"] == 0.0
    assert result["mean"] == 0.0


# ── sd_within_series ────────────────────────────────────────


def test_sd_within_series_returns_dict():
    df = _make_multi_day_df(2)
    result = sd_within_series(df["glucose"], df["time"], hours=1)
    assert "sd" in result and "mean" in result
    assert "windows_analyzed" in result
    assert result["windows_analyzed"] > 0


def test_sd_within_series_24h():
    df = _make_multi_day_df(3)
    result = sd_within_series(df["glucose"], df["time"], hours=24)
    assert result["sd"] >= 0.0


# ── sd_interaction ──────────────────────────────────────────


def test_sd_interaction_returns_dict():
    df = _make_multi_day_df(2)
    result = sd_interaction(df["glucose"], df["time"])
    assert "sd" in result and "mean" in result
    assert isinstance(result["sd"], float | np.floating)


# ── cv_global, cv_from_sd_mean ──────────────────────────────


def test_cv_global():
    glucose = pd.Series([100.0, 110.0, 120.0])
    expected = (glucose.std() / glucose.mean()) * 100
    assert cv_global(glucose) == pytest.approx(expected)


def test_cv_global_zero_mean():
    assert cv_global(pd.Series([0.0, 0.0, 0.0])) == 0.0


def test_cv_from_sd_mean():
    assert cv_from_sd_mean(10.0, 100.0) == pytest.approx(10.0)
    assert cv_from_sd_mean(0.0, 100.0) == pytest.approx(0.0)
    assert cv_from_sd_mean(10.0, 0.0) == pytest.approx(0.0)


# ── Known values ────────────────────────────────────────────


def test_sd_known_values():
    glucose = pd.Series([100.0, 110.0, 120.0])
    assert sd_global(glucose) == pytest.approx(10.0)


def test_cv_known_values():
    glucose = pd.Series([100.0, 110.0, 120.0])
    expected = (10.0 / 110.0) * 100
    assert cv_global(glucose) == pytest.approx(expected)


# ── Empty data ──────────────────────────────────────────────


def test_sd_global_empty():
    glucose = pd.Series([], dtype=float)
    assert np.isnan(sd_global(glucose))


def test_mean_global_empty():
    glucose = pd.Series([], dtype=float)
    assert np.isnan(mean_global(glucose))


def test_sd_within_day_empty():
    glucose = pd.Series([], dtype=float)
    timestamps = pd.Series([], dtype="datetime64[ns]")
    result = sd_within_day(glucose, timestamps)
    assert result["sd"] == 0.0


# ── Backward compatibility ──────────────────────────────────


@pytest.fixture
def sine_24h_glucose():
    df = _make_multi_day_df(3)
    return df["glucose"], df["time"]


def test_backward_compatibility_sd_global():
    """Pure function matches the mixin's sd_total().sd"""
    from cgmpy import GlucoseAnalysis

    gm = GlucoseAnalysis("tests/fixtures/synthetic/sine_24h.csv")
    result = gm.sd_total()
    assert sd_global(gm.data["glucose"]) == pytest.approx(result["sd"])


def test_backward_compatibility_mean_global():
    from cgmpy import GlucoseAnalysis

    gm = GlucoseAnalysis("tests/fixtures/synthetic/sine_24h.csv")
    result = gm.sd_total()
    assert mean_global(gm.data["glucose"]) == pytest.approx(result["mean"])


def test_backward_compatibility_sd_within_day():
    from cgmpy import GlucoseAnalysis

    gm = GlucoseAnalysis("tests/fixtures/synthetic/sine_24h.csv")
    method_result = gm.sd_within_day()
    func_result = sd_within_day(gm.data["glucose"], gm.data["time"])
    assert func_result["sd"] == pytest.approx(method_result["sd"])
    assert func_result["mean"] == pytest.approx(method_result["mean"])


def test_backward_compatibility_sdw():
    from cgmpy import GlucoseAnalysis

    gm = GlucoseAnalysis("tests/fixtures/synthetic/sine_24h.csv")
    assert sdw(gm.data["glucose"], gm.data["time"]) == pytest.approx(gm.sdw())


def test_backward_compatibility_sd_daily_mean():
    from cgmpy import GlucoseAnalysis

    gm = GlucoseAnalysis("tests/fixtures/synthetic/sine_24h.csv")
    method_result = gm.sd_daily_mean()
    func_result = sd_daily_mean(gm.data["glucose"], gm.data["time"])
    # Both will produce NaN with single-day data; assert matching NaN status
    assert (
        func_result["sd"] == method_result["sd"]
        or (np.isnan(func_result["sd"]) and np.isnan(method_result["sd"]))
    )


def test_backward_compatibility_sd_between_timepoints():
    from cgmpy import GlucoseAnalysis

    gm = GlucoseAnalysis("tests/fixtures/synthetic/sine_24h.csv")
    method_result = gm.sd_between_timepoints()
    func_result = sd_between_timepoints(gm.data["glucose"], gm.data["time"])
    assert func_result["sd"] == pytest.approx(method_result["sd"])


def test_backward_compatibility_sd_same_timepoint():
    from cgmpy import GlucoseAnalysis

    gm = GlucoseAnalysis("tests/fixtures/synthetic/sine_24h.csv")
    method_result = gm.sd_same_timepoint()
    func_result = sd_same_timepoint(gm.data["glucose"], gm.data["time"])
    assert func_result["sd"] == pytest.approx(method_result["sd"])


def test_backward_compatibility_sd_same_timepoint_adjusted():
    from cgmpy import GlucoseAnalysis

    def _same_or_both_nan(a, b):
        return a == b or (np.isnan(a) and np.isnan(b))

    gm = GlucoseAnalysis("tests/fixtures/synthetic/sine_24h.csv")
    method_result = gm.sd_same_timepoint_adjusted()
    func_result = sd_same_timepoint_adjusted(gm.data["glucose"], gm.data["time"])
    assert _same_or_both_nan(func_result["sd"], method_result["sd"])


def test_backward_compatibility_sd_segment():
    from cgmpy import GlucoseAnalysis

    gm = GlucoseAnalysis("tests/fixtures/synthetic/sine_24h.csv")
    method_result = gm.sd_segment("08:00", 8)
    func_result = sd_segment(gm.data["glucose"], gm.data["time"], "08:00", 8)
    assert func_result["sd"] == pytest.approx(method_result["sd"])


def test_backward_compatibility_sd_segment_midnight():
    from cgmpy import GlucoseAnalysis

    gm = GlucoseAnalysis("tests/fixtures/synthetic/sine_24h.csv")
    method_result = gm.sd_segment("22:00", 8)
    func_result = sd_segment(gm.data["glucose"], gm.data["time"], "22:00", 8)
    assert func_result["sd"] == pytest.approx(method_result["sd"])


def test_backward_compatibility_sd_within_series():
    from cgmpy import GlucoseAnalysis

    gm = GlucoseAnalysis("tests/fixtures/synthetic/sine_24h.csv")
    method_result = gm.sd_within_series(hours=1)
    func_result = sd_within_series(gm.data["glucose"], gm.data["time"], hours=1)
    assert func_result["sd"] == pytest.approx(method_result["sd"])


def test_backward_compatibility_sd_interaction():
    from cgmpy import GlucoseAnalysis

    gm = GlucoseAnalysis("tests/fixtures/synthetic/sine_24h.csv")
    method_result = gm.sd_interaction()
    func_result = sd_interaction(gm.data["glucose"], gm.data["time"])
    assert func_result["sd"] == pytest.approx(method_result["sd"])
    assert func_result["mean"] == pytest.approx(method_result["mean"])


def test_backward_compatibility_cv_global():
    from cgmpy import GlucoseAnalysis

    gm = GlucoseAnalysis("tests/fixtures/synthetic/sine_24h.csv")
    method_result = gm.calculate_all_cv_metrics()
    pure_result = cv_global(gm.data["glucose"])
    assert pure_result == pytest.approx(method_result["CVT"])
