"""Tests for `cgmpy.agata.adapter.prepare_data_for_agata`.

These tests are skipped if the optional `py_agata` dependency is missing.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Skip the whole module if py_agata is not installed.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("py_agata") is None,
    reason="py_agata optional dependency is not installed",
)

from cgmpy.agata.adapter import prepare_data_for_agata  # noqa: E402
from cgmpy.data.core import GlucoseData  # noqa: E402


@pytest.fixture
def one_day_glucose_df() -> pd.DataFrame:
    """A clean 1-day dataset at exact 5-min intervals."""
    start = datetime(2024, 1, 1, 0, 0)
    times = [start + timedelta(minutes=5 * i) for i in range(288)]
    rng = np.random.default_rng(42)
    glucose = 120 + 20 * np.sin(np.linspace(0, 4 * np.pi, 288)) + rng.normal(0, 4, 288)
    return pd.DataFrame({"time": times, "glucose": glucose})


@pytest.fixture
def unaligned_glucose_df() -> pd.DataFrame:
    """A 1-day dataset with unaligned start time and slightly irregular sampling."""
    start = datetime(2024, 1, 1, 0, 3)  # not aligned to 5-min boundary
    times = [start + timedelta(minutes=5 * i + (i % 3)) for i in range(288)]
    rng = np.random.default_rng(7)
    glucose = 110 + 25 * np.sin(np.linspace(0, 4 * np.pi, 288)) + rng.normal(0, 5, 288)
    return pd.DataFrame({"time": times, "glucose": glucose})


class TestPrepareDataForAgata:
    """Tests for the adapter that prepares data for py_agata."""

    def test_returns_dataframe_with_expected_columns(
        self, one_day_glucose_df: pd.DataFrame
    ) -> None:
        """The adapter returns a DataFrame with `t` and `glucose` columns."""
        gd = GlucoseData(data_source=one_day_glucose_df)
        result = prepare_data_for_agata(gd)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["t", "glucose"]

    def test_time_column_is_datetime(self, one_day_glucose_df: pd.DataFrame) -> None:
        """The `t` column is datetime-typed."""
        gd = GlucoseData(data_source=one_day_glucose_df)
        result = prepare_data_for_agata(gd)
        assert pd.api.types.is_datetime64_any_dtype(result["t"])

    def test_homogeneous_5min_grid(self, one_day_glucose_df: pd.DataFrame) -> None:
        """The result is on a homogeneous 5-minute grid."""
        gd = GlucoseData(data_source=one_day_glucose_df)
        result = prepare_data_for_agata(gd)
        deltas = result["t"].diff().dropna().dt.total_seconds() / 60
        assert (deltas == 5.0).all()

    def test_unaligned_data_is_floored_to_grid(self, unaligned_glucose_df: pd.DataFrame) -> None:
        """Unaligned timestamps are floored to the 5-minute grid."""
        gd = GlucoseData(data_source=unaligned_glucose_df)
        result = prepare_data_for_agata(gd)
        # All minutes in the floored grid are multiples of 5
        minutes = result["t"].dt.minute.unique()
        assert all(m % 5 == 0 for m in minutes)

    def test_custom_resample_freq(self, one_day_glucose_df: pd.DataFrame) -> None:
        """A non-default resample frequency is honored."""
        gd = GlucoseData(data_source=one_day_glucose_df)
        result = prepare_data_for_agata(gd, resample_freq="15min")
        deltas = result["t"].diff().dropna().dt.total_seconds() / 60
        assert (deltas == 15.0).all()

    def test_preserves_glucose_range(self, one_day_glucose_df: pd.DataFrame) -> None:
        """Glucose values in the result fall within the original range (modulo NaNs)."""
        gd = GlucoseData(data_source=one_day_glucose_df)
        result = prepare_data_for_agata(gd)
        non_null = result["glucose"].dropna()
        original = one_day_glucose_df["glucose"]
        assert non_null.min() >= original.min() - 1e-6
        assert non_null.max() <= original.max() + 1e-6
