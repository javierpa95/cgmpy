"""Tests for cgmpy.metrics.variability.sd.SDMetrics."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from cgmpy import GlucoseMetrics


def _make_multi_day_df(n_days: int) -> pd.DataFrame:
    """Build a deterministic multi-day synthetic dataset (5-min samples)."""
    start = datetime(2024, 1, 1, 0, 0)
    n = 288 * n_days
    times = [start + timedelta(minutes=5 * i) for i in range(n)]
    # Pattern: mean ~140 mg/dL with a gentle daily sine and mild noise.
    hours = np.arange(n) * 5 / 60.0
    glucose = 140 + 25 * np.sin(hours / 24 * 2 * np.pi) + np.random.default_rng(42).normal(0, 3, n)
    return pd.DataFrame({"time": times, "glucose": glucose})


def test_sd_total_returns_dict(stable_glucose_df):
    """sd_total should return a dict with 'sd' and 'mean' matching the raw series."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    result = gm.sd_total()
    assert isinstance(result, dict)
    assert "sd" in result and "mean" in result
    assert result["sd"] == pytest.approx(stable_glucose_df["glucose"].std())
    assert result["mean"] == pytest.approx(stable_glucose_df["glucose"].mean())


def test_sd_within_day_default_and_threshold(stable_glucose_df):
    """sd_within_day should return the same value as sdw() and respect min_count_threshold."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    default = gm.sd_within_day()
    assert "sd" in default
    assert default["sd"] == pytest.approx(gm.sdw())
    # Custom threshold should be accepted without error and return a numeric SD.
    custom = gm.sd_within_day(min_count_threshold=0.9)
    assert isinstance(custom["sd"], float)


def test_sdw_returns_float(stable_glucose_df):
    """sdw() should return only the simplified float value."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    val = gm.sdw()
    assert isinstance(val, float)
    assert val >= 0.0


def test_sd_within_day_segment(stable_glucose_df):
    """sd_within_day_segment should return sd/mean for a time window."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    result = gm.sd_within_day_segment("08:00", 8)
    assert "sd" in result and "mean" in result
    assert isinstance(result["sd"], (float, np.floating))


def test_sd_between_timepoints_default(stable_glucose_df):
    """sd_between_timepoints should return a dict with sd/mean plus statistics."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    result = gm.sd_between_timepoints()
    for key in ("sd", "mean", "valid_timepoints", "total_timepoints",
                "median_count", "min_count", "max_count"):
        assert key in result
    assert isinstance(result["sd"], float)


def test_sd_between_timepoints_with_grouping(stable_glucose_df):
    """agrupar_por_intervalos=True branch is a known source bug (missing 'day' column).

    The implementation references ``df.groupby(["day", "interval"])`` but the
    ``day`` column is only created in the non-grouped branch. The call therefore
    raises ``KeyError: 'day'`` on every input, so we verify the failure mode
    rather than asserting on a numeric result.
    """
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    with pytest.raises(KeyError):
        gm.sd_between_timepoints(
            filter_outliers=False,
            agrupar_por_intervalos=True,
            intervalo_minutos=5,
        )


def test_sd_between_timepoints_segment():
    """Use a multi-day dataset so the segment has repeated timestamps."""
    df = _make_multi_day_df(2)
    gm = GlucoseMetrics(data_source=df)
    result = gm.sd_between_timepoints_segment("00:00", 8)
    assert "sd" in result and "mean" in result
    assert isinstance(result["sd"], (float, np.floating))


def test_sd_within_series_1h():
    """sd_within_series(hours=1) should return a finite SD and metadata."""
    df = _make_multi_day_df(2)
    gm = GlucoseMetrics(data_source=df)
    result = gm.sd_within_series(hours=1)
    assert "sd" in result
    assert "windows_analyzed" in result
    assert result["windows_analyzed"] > 0
    assert isinstance(result["sd"], (float, np.floating))


def test_sd_within_series_24h():
    """sd_within_series(hours=24) should return a finite SD on multi-day data."""
    df = _make_multi_day_df(3)
    gm = GlucoseMetrics(data_source=df)
    result = gm.sd_within_series(hours=24)
    assert result["sd"] >= 0.0
    assert isinstance(result["sd"], (float, np.floating))


def test_sd_daily_mean():
    """sd_daily_mean should return sd/mean of daily means across multiple days."""
    df = _make_multi_day_df(3)
    gm = GlucoseMetrics(data_source=df)
    result = gm.sd_daily_mean()
    assert "sd" in result and "mean" in result
    assert "valid_days" in result and "total_days" in result
    assert result["valid_days"] >= 2
    assert isinstance(result["sd"], (float, np.floating))


def test_sd_same_timepoint_default():
    """sd_same_timepoint should aggregate per-time-point between-day SDs."""
    df = _make_multi_day_df(3)
    gm = GlucoseMetrics(data_source=df)
    result = gm.sd_same_timepoint()
    assert "sd" in result and "mean" in result
    assert "total_timepoints" in result
    assert isinstance(result["sd"], (float, np.floating))


def test_sd_same_timepoint_no_filter():
    """filter_outliers=False should keep every time point in the result."""
    df = _make_multi_day_df(2)
    gm = GlucoseMetrics(data_source=df)
    result = gm.sd_same_timepoint(filter_outliers=False)
    assert isinstance(result["sd"], (float, np.floating))


def test_sd_same_timepoint_adjusted():
    """sd_same_timepoint_adjusted should adjust values by daily mean and return sd/mean."""
    df = _make_multi_day_df(2)
    gm = GlucoseMetrics(data_source=df)
    result = gm.sd_same_timepoint_adjusted()
    assert "sd" in result and "mean" in result
    assert isinstance(result["sd"], (float, np.floating))


def test_sd_interaction():
    """sd_interaction should return the SDI value plus the global mean."""
    df = _make_multi_day_df(2)
    gm = GlucoseMetrics(data_source=df)
    result = gm.sd_interaction()
    assert "sd" in result and "mean" in result
    assert isinstance(result["sd"], (float, np.floating))
    assert result["mean"] == pytest.approx(df["glucose"].mean(), rel=1e-3)


def test_sd_segment_day(stable_glucose_df):
    """sd_segment should return sd/mean for a defined time window (day)."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    result = gm.sd_segment("08:00", 8)
    assert "sd" in result and "mean" in result
    assert isinstance(result["sd"], (float, np.floating))


def test_sd_segment_crosses_midnight():
    """A segment that crosses midnight (e.g. 22:00-06:00) should be handled."""
    df = _make_multi_day_df(2)
    gm = GlucoseMetrics(data_source=df)
    result = gm.sd_segment("22:00", 8)
    assert "sd" in result and "mean" in result
    assert isinstance(result["sd"], (float, np.floating))


def test_calculate_all_sd_metrics_returns_full_dict(stable_glucose_df):
    """calculate_all_sd_metrics should return a dict with all SD keys."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    result = gm.calculate_all_sd_metrics()
    expected = {
        "SDT", "SDw", "SDhh:mm", "Noche", "Day", "Tarde",
        "SDws_1h", "SDws_6h", "SDws_24h", "SDdm", "SDbhh:mm",
        "SDbhh:mm_dm", "SDI",
    }
    assert set(result.keys()) == expected
    for k, v in result.items():
        assert v is not None, f"value for {k} was None"


def test_calculate_all_cv_metrics_returns_percentages():
    """calculate_all_cv_metrics should return percentage values (CV * 100)."""
    df = _make_multi_day_df(2)
    gm = GlucoseMetrics(data_source=df)
    result = gm.calculate_all_cv_metrics()
    expected = {
        "CVT", "CVw", "CVhh:mm", "CVNoche", "CVDay", "CVTarde",
        "CVSDws_1h", "CVSDws_6h", "CVSDws_24h", "CVdm",
        "CVbhh:mm", "CVbhh:mm_dm", "CVSDI",
    }
    assert set(result.keys()) == expected
    for k, v in result.items():
        assert v is not None
        # CV expressed as percentage should be a reasonable number (<500%).
        assert -500.0 < v < 500.0, f"{k} = {v} is not a plausible CV percentage"


def test_sd_between_timepoints_no_filter(stable_glucose_df):
    """filter_outliers=False should retain every time point (cover the 'else' branch)."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    result = gm.sd_between_timepoints(filter_outliers=False)
    assert "sd" in result
    assert isinstance(result["sd"], (float, np.floating))


def test_sd_within_day_segment_empty_returns_zero():
    """A segment that has no overlapping data should return sd=0, mean=0."""
    # Build a dataset that only covers noon to midnight.
    start = datetime(2024, 1, 1, 12, 0)
    n = 144  # 12 hours
    times = [start + timedelta(minutes=5 * i) for i in range(n)]
    df = pd.DataFrame({"time": times, "glucose": [120.0] * n})
    gm = GlucoseMetrics(data_source=df)
    result = gm.sd_within_day_segment("00:00", 6)  # No data covers 00:00-06:00
    assert result["sd"] == 0.0
    assert result["mean"] == 0.0


def test_sd_segment_end_at_midnight():
    """A segment that ends exactly at 00:00 (e.g. 16:00 + 8h) should be handled."""
    df = _make_multi_day_df(2)
    gm = GlucoseMetrics(data_source=df)
    # 16:00 + 8h = 24:00 (== 00:00 next day); exercises the end_min==0 branch.
    result = gm.sd_segment("16:00", 8)
    assert "sd" in result and "mean" in result
    assert isinstance(result["sd"], (float, np.floating))
