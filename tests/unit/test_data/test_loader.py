"""Tests for `cgmpy.data.loader.DataLoader`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cgmpy.data.loader import DataLoader

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "data"


class TestDataLoader:
    """Smoke tests for DataLoader with the bundled fixtures."""

    def test_load_dm_csv(self) -> None:
        """The T1D fixture loads without errors and has glucose + time columns."""
        loader = DataLoader()
        df = loader.load_from_csv(str(FIXTURES / "dm.csv"))
        assert not df.empty
        assert "glucose" in df.columns
        # The 'time' column is normalized to a standard name.
        assert any(c.lower() in ("time", "timestamp", "datetime", "date") for c in df.columns)

    def test_load_nodm_csv(self) -> None:
        """The non-diabetic fixture loads."""
        loader = DataLoader()
        df = loader.load_from_csv(str(FIXTURES / "nodm.csv"))
        assert not df.empty

    def test_load_pregnancy_csv(self) -> None:
        """The pregnancy fixture loads."""
        loader = DataLoader()
        df = loader.load_from_csv(str(FIXTURES / "pregnancy.csv"))
        assert not df.empty

    def test_load_from_dataframe(self, stable_glucose_df: pd.DataFrame) -> None:
        """Loading from a DataFrame returns a usable DataFrame."""
        loader = DataLoader()
        df = loader.load_from_dataframe(stable_glucose_df)
        assert len(df) == len(stable_glucose_df)
        assert "glucose" in df.columns

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """A missing CSV raises a clear error (not a generic pandas crash)."""
        loader = DataLoader()
        missing = tmp_path / "does_not_exist.csv"
        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            loader.load_from_csv(str(missing))

    def test_glucose_values_are_numeric(self) -> None:
        """After load, glucose column is numeric (non-numeric rows are dropped)."""
        loader = DataLoader()
        df = loader.load_from_csv(str(FIXTURES / "dm.csv"))
        assert pd.api.types.is_numeric_dtype(df["glucose"])
