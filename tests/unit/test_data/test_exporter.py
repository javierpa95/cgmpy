"""Tests for `cgmpy.data.exporter.DataExporter`."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cgmpy.data.exporter import DataExporter


@pytest.fixture
def small_glucose_df() -> pd.DataFrame:
    """A small 24h synthetic glucose trace at 5 min intervals."""
    n = 24 * 12
    times = pd.date_range("2024-01-01", periods=n, freq="5min")
    glucose = np.full(n, 110, dtype=int)
    return pd.DataFrame({"time": times, "glucose": glucose})


@pytest.fixture
def unsorted_glucose_df() -> pd.DataFrame:
    """A DataFrame whose 'time' column is not monotonically increasing."""
    times = pd.to_datetime(["2024-01-01 02:00", "2024-01-01 00:00", "2024-01-01 01:00"])
    glucose = np.array([110, 100, 105], dtype=int)
    return pd.DataFrame({"time": times, "glucose": glucose})


@pytest.fixture
def df_with_duplicates() -> pd.DataFrame:
    """DataFrame containing a duplicate timestamp row."""
    times = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:05"])
    glucose = np.array([100, 110, 120], dtype=int)
    return pd.DataFrame({"time": times, "glucose": glucose})


class TestDataExporterParquet:
    """Round-trip tests for the Parquet writers."""

    def test_to_parquet_creates_file(self, small_glucose_df: pd.DataFrame, tmp_path: Path) -> None:
        """`to_parquet` creates a file on disk."""
        out = tmp_path / "out.parquet"
        DataExporter().to_parquet(small_glucose_df, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_to_parquet_roundtrip(self, small_glucose_df: pd.DataFrame, tmp_path: Path) -> None:
        """Data read back from Parquet matches the original (modulo dtype)."""
        out = tmp_path / "rt.parquet"
        DataExporter().to_parquet(small_glucose_df, str(out))
        back = pd.read_parquet(out)
        assert len(back) == len(small_glucose_df)
        assert list(back.columns) == ["time", "glucose"]
        # Glucose values are preserved
        assert (back["glucose"].astype(int) == small_glucose_df["glucose"]).all()

    def test_to_parquet_sorts_unsorted_input(
        self, unsorted_glucose_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """When `sort=True`, output is sorted by timestamp."""
        out = tmp_path / "sorted.parquet"
        DataExporter().to_parquet(unsorted_glucose_df, str(out), sort=True)
        back = pd.read_parquet(out)
        assert back["time"].is_monotonic_increasing

    def test_to_parquet_no_sort(self, unsorted_glucose_df: pd.DataFrame, tmp_path: Path) -> None:
        """When `sort=False`, the original order is preserved."""
        out = tmp_path / "unsorted.parquet"
        DataExporter().to_parquet(unsorted_glucose_df, str(out), sort=False)
        back = pd.read_parquet(out)
        assert not back["time"].is_monotonic_increasing


class TestDataExporterCsv:
    """Tests for `to_csv`."""

    def test_to_csv_creates_file(self, small_glucose_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        DataExporter().to_csv(small_glucose_df, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_to_csv_roundtrip(self, small_glucose_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "rt.csv"
        DataExporter().to_csv(small_glucose_df, str(out))
        back = pd.read_csv(out)
        assert len(back) == len(small_glucose_df)
        assert "time" in back.columns
        assert "glucose" in back.columns

    def test_to_csv_with_semicolon_separator(
        self, small_glucose_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Custom separator is respected."""
        out = tmp_path / "semi.csv"
        DataExporter().to_csv(small_glucose_df, str(out), separator=";")
        back = pd.read_csv(out, sep=";")
        assert len(back) == len(small_glucose_df)


class TestDataExporterExcel:
    """Tests for `to_excel`."""

    def test_to_excel_creates_file(self, small_glucose_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "out.xlsx"
        DataExporter().to_excel(small_glucose_df, str(out))
        assert out.exists()

    def test_to_excel_roundtrip_with_openpyxl(
        self, small_glucose_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Round-trip an Excel file using openpyxl."""
        out = tmp_path / "rt.xlsx"
        DataExporter().to_excel(small_glucose_df, str(out), sheet_name="custom")
        back = pd.read_excel(out, sheet_name="custom", engine="openpyxl")
        assert len(back) == len(small_glucose_df)
        assert "glucose" in back.columns


class TestAppendToParquet:
    """Tests for `append_to_parquet`."""

    def test_append_creates_file_when_missing(
        self, small_glucose_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """If the file doesn't exist, it is created and the row count returned."""
        out = tmp_path / "new.parquet"
        added = DataExporter().append_to_parquet(small_glucose_df, str(out))
        assert out.exists()
        assert added == len(small_glucose_df)

    def test_append_extends_existing_file(self, tmp_path: Path) -> None:
        """Appending new (non-overlapping) data grows the file."""
        # First batch
        df1 = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=12, freq="5min"),
                "glucose": np.full(12, 110, dtype=int),
            }
        )
        # Non-overlapping second batch
        df2 = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01 02:00", periods=12, freq="5min"),
                "glucose": np.full(12, 120, dtype=int),
            }
        )
        out = tmp_path / "growing.parquet"
        exporter = DataExporter()
        exporter.to_parquet(df1, str(out))
        added = exporter.append_to_parquet(df2, str(out))
        assert added == len(df2)
        back = pd.read_parquet(out)
        assert len(back) == len(df1) + len(df2)
        assert back["time"].is_monotonic_increasing

    def test_append_keep_new_overrides_existing(self, tmp_path: Path) -> None:
        """`handle_duplicates="keep_new"` drops the existing duplicated rows."""
        df_existing = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=3, freq="5min"),
                "glucose": np.array([100, 100, 100], dtype=int),
            }
        )
        # Overlapping at the second timestamp, with a different glucose
        df_new = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01 00:05", periods=2, freq="5min"),
                "glucose": np.array([200, 200], dtype=int),
            }
        )
        out = tmp_path / "kn.parquet"
        exporter = DataExporter()
        exporter.to_parquet(df_existing, str(out))
        exporter.append_to_parquet(df_new, str(out), handle_duplicates="keep_new")
        back = pd.read_parquet(out).sort_values("time").reset_index(drop=True)
        # No duplicates at the combined timestamp set
        assert not back["time"].duplicated().any()
        # The new values (200) should be present for the overlapping timestamps
        assert back.loc[back["time"] == pd.Timestamp("2024-01-01 00:05"), "glucose"].iloc[0] == 200

    def test_append_keep_old_preserves_existing(self, tmp_path: Path) -> None:
        """`handle_duplicates="keep_old"` discards conflicting new rows."""
        df_existing = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=3, freq="5min"),
                "glucose": np.array([100, 100, 100], dtype=int),
            }
        )
        df_new = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01 00:05", periods=2, freq="5min"),
                "glucose": np.array([200, 200], dtype=int),
            }
        )
        out = tmp_path / "ko.parquet"
        exporter = DataExporter()
        exporter.to_parquet(df_existing, str(out))
        exporter.append_to_parquet(df_new, str(out), handle_duplicates="keep_old")
        back = pd.read_parquet(out).sort_values("time").reset_index(drop=True)
        assert not back["time"].duplicated().any()
        # Old values (100) should remain at the overlapping timestamps
        assert back.loc[back["time"] == pd.Timestamp("2024-01-01 00:05"), "glucose"].iloc[0] == 100


class TestOptimizeDataTypes:
    """Unit tests for the internal optimization helpers."""

    def test_optimize_uses_int16_for_small_range(self) -> None:
        """Glucose values in [-32768, 32767] are stored as int16."""
        df = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=3, freq="5min"),
                "glucose": [80.0, 110.0, 200.0],
            }
        )
        optimized = DataExporter()._optimize_data_types(df)
        assert optimized["glucose"].dtype == np.int16

    def test_optimize_uses_int32_when_outside_int16(self) -> None:
        """Glucose values exceeding int16 are stored as int32."""
        df = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=3, freq="5min"),
                "glucose": [80.0, 110.0, 40_000.0],  # exceeds int16 max
            }
        )
        optimized = DataExporter()._optimize_data_types(df)
        assert optimized["glucose"].dtype == np.int32

    def test_optimize_converts_time_column_to_datetime(self) -> None:
        """A string 'time' column is coerced to datetime."""
        df = pd.DataFrame(
            {
                "time": ["2024-01-01 00:00", "2024-01-01 00:05"],
                "glucose": [100, 110],
            }
        )
        optimized = DataExporter()._optimize_data_types(df)
        assert pd.api.types.is_datetime64_any_dtype(optimized["time"])


class TestRemoveDuplicates:
    """Tests for `_remove_duplicates`."""

    def test_removes_duplicate_timestamps(self, df_with_duplicates: pd.DataFrame) -> None:
        """Duplicate timestamps are dropped, keeping the first occurrence."""
        cleaned = DataExporter()._remove_duplicates(df_with_duplicates)
        assert len(cleaned) == 2
        assert not cleaned["time"].duplicated().any()

    def test_returns_same_when_no_duplicates(self, small_glucose_df: pd.DataFrame) -> None:
        """If there are no duplicates, the DataFrame is returned unchanged."""
        cleaned = DataExporter()._remove_duplicates(small_glucose_df)
        assert len(cleaned) == len(small_glucose_df)


class TestHandleDuplicates:
    """Tests for the in-memory `_handle_duplicates` helper."""

    def _make_pair(self):
        existing = pd.DataFrame(
            {
                "time": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:05"]),
                "glucose": [100, 100],
            }
        )
        new = pd.DataFrame(
            {
                "time": pd.to_datetime(["2024-01-01 00:05", "2024-01-01 00:10"]),
                "glucose": [200, 200],
            }
        )
        return existing, new

    def test_keep_new_drops_existing_overlap(self) -> None:
        existing, new = self._make_pair()
        existing_out, new_out = DataExporter()._handle_duplicates(existing, new, "keep_new")
        # Existing row at 00:05 should be removed
        assert pd.Timestamp("2024-01-01 00:05") not in set(existing_out["time"])
        # New data unchanged
        assert len(new_out) == len(new)

    def test_keep_old_drops_new_overlap(self) -> None:
        existing, new = self._make_pair()
        existing_out, new_out = DataExporter()._handle_duplicates(existing, new, "keep_old")
        # New row at 00:05 should be removed
        assert pd.Timestamp("2024-01-01 00:05") not in set(new_out["time"])
        # Existing unchanged
        assert len(existing_out) == len(existing)

    def test_average_strategy_does_not_alter_data(self) -> None:
        """The 'average' branch is not yet implemented but must not crash."""
        existing, new = self._make_pair()
        existing_out, new_out = DataExporter()._handle_duplicates(existing, new, "average")
        # The current implementation leaves data untouched for unknown strategies
        assert len(existing_out) == len(existing)
        assert len(new_out) == len(new)


class TestPrepareNewData:
    """Tests for the `_prepare_new_data` helper."""

    def test_prepare_coerces_string_time_to_datetime(self) -> None:
        df = pd.DataFrame(
            {
                "time": ["2024-01-01 00:00", "2024-01-01 00:05"],
                "glucose": [100, 110],
            }
        )
        prepared = DataExporter()._prepare_new_data(df)
        assert pd.api.types.is_datetime64_any_dtype(prepared["time"])

    def test_prepare_returns_copy(self) -> None:
        """The original DataFrame is not mutated."""
        df = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=2, freq="5min"),
                "glucose": [100.0, 110.0],
            }
        )
        prepared = DataExporter()._prepare_new_data(df)
        assert prepared is not df

    def test_prepare_uses_int16_when_possible(self) -> None:
        """Float glucose in int16 range is downcast to int16."""
        df = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=3, freq="5min"),
                "glucose": [80.0, 110.0, 200.0],
            }
        )
        prepared = DataExporter()._prepare_new_data(df)
        assert prepared["glucose"].dtype == np.int16


class TestLogSaveInfo:
    """Smoke tests for the logging helper."""

    def test_log_save_info_does_not_crash(
        self, small_glucose_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """The logging helper produces no exceptions on a real saved file."""
        out = tmp_path / "logged.parquet"
        exporter = DataExporter(logger=logging.getLogger("test_exporter"))
        exporter.to_parquet(small_glucose_df, str(out))
        # Calling _log_save_info directly should also work
        exporter._log_save_info(str(out), small_glucose_df, save_time=0.01)
