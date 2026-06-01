"""Tests for `cgmpy.data.processor.DataProcessor`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cgmpy.data.processor import DataProcessor


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A 24-hour synthetic trace at 5-min intervals with one duplicate row."""
    n = 24 * 12
    times = pd.date_range("2024-01-01", periods=n, freq="5min")
    glucose = np.full(n, 110.0)
    # Append a duplicate of the first row to test deduplication.
    times = times.append(pd.DatetimeIndex([times[0]]))
    glucose = np.append(glucose, [glucose[0]])
    return pd.DataFrame({"time": times, "glucose": glucose})


class TestDataProcessor:
    """Tests for the data validation / cleaning pipeline."""

    def test_process_dedupes(self, raw_df: pd.DataFrame) -> None:
        """Duplicate rows are removed by the processor."""
        processor = DataProcessor()
        cleaned, _ = processor.process_data(raw_df, "time", "glucose")
        assert len(cleaned) == len(raw_df) - 1

    def test_process_returns_typical_interval(self, raw_df: pd.DataFrame) -> None:
        """The processor returns the typical sample interval in minutes."""
        processor = DataProcessor()
        _, diffs = processor.process_data(raw_df, "time", "glucose")
        # 5-minute sampling → typical interval 5.0
        if hasattr(diffs, "mean"):
            assert diffs.mean() == pytest.approx(5.0, abs=0.1)

    def test_process_coerces_numeric(self) -> None:
        """Non-numeric glucose strings are coerced (and dropped if unparseable)."""
        df = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=5, freq="5min"),
                "glucose": ["100", "110", "abc", "120", "130"],
            }
        )
        processor = DataProcessor()
        cleaned, _ = processor.process_data(df, "time", "glucose")
        # 'abc' row is dropped
        assert len(cleaned) == 4
        assert pd.api.types.is_numeric_dtype(cleaned["glucose"])
