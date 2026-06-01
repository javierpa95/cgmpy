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
        df = loader.load_from_source(
            str(FIXTURES / "dm.csv"), date_col="time", glucose_col="glucose"
        )
        assert not df.empty
        assert "glucose" in df.columns
        # The 'time' column is preserved under the name we requested.
        assert "time" in df.columns

    def test_load_nodm_csv(self) -> None:
        """The non-diabetic fixture loads."""
        loader = DataLoader()
        df = loader.load_from_source(
            str(FIXTURES / "nodm.csv"), date_col="time", glucose_col="glucose"
        )
        assert not df.empty
        assert "glucose" in df.columns

    def test_load_pregnancy_csv(self) -> None:
        """The pregnancy fixture loads."""
        loader = DataLoader()
        df = loader.load_from_source(
            str(FIXTURES / "pregnancy.csv"), date_col="time", glucose_col="glucose"
        )
        assert not df.empty
        assert "glucose" in df.columns

    def test_load_from_dataframe(self, stable_glucose_df: pd.DataFrame) -> None:
        """Loading from a DataFrame returns a usable DataFrame."""
        loader = DataLoader()
        df = loader.load_from_source(stable_glucose_df, date_col="time", glucose_col="glucose")
        assert len(df) == len(stable_glucose_df)
        assert "glucose" in df.columns

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """A missing CSV raises a clear error (not a generic pandas crash)."""
        loader = DataLoader()
        missing = tmp_path / "does_not_exist.csv"
        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            loader.load_from_source(str(missing), date_col="time", glucose_col="glucose")

    def test_glucose_values_are_numeric(self) -> None:
        """After load, glucose column is numeric (the loader casts via usecols)."""
        loader = DataLoader()
        df = loader.load_from_source(
            str(FIXTURES / "dm.csv"), date_col="time", glucose_col="glucose"
        )
        assert pd.api.types.is_numeric_dtype(df["glucose"])
