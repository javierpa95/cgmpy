"""Regression tests for edge-case guards in `cgmpy.agata.adapter`.

These tests are skipped if the optional `py_agata` dependency is missing.
They do not exercise py_agata itself; they only verify the adapter
raises the new empty-data errors from `cgmpy.errors`.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime

import pandas as pd
import pytest

# Skip the whole module if py_agata is not installed.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("py_agata") is None,
    reason="py_agata optional dependency is not installed",
)

from cgmpy.agata.adapter import prepare_data_for_agata  # noqa: E402
from cgmpy.data.core import GlucoseData  # noqa: E402
from cgmpy.errors import EmptyDataError  # noqa: E402


def _empty_glucose_df() -> pd.DataFrame:
    """A well-formed but zero-row DataFrame with the expected schema."""
    return pd.DataFrame(
        {
            "time": pd.Series(dtype="datetime64[ns]"),
            "glucose": pd.Series(dtype="float64"),
        }
    )


def _single_row_glucose_df() -> pd.DataFrame:
    """A DataFrame with exactly one (time, glucose) row."""
    return pd.DataFrame({"time": [datetime(2024, 1, 1, 12, 0)], "glucose": [120.0]})


class TestAdapterEdgeCases:
    """Edge-case guards on `prepare_data_for_agata`."""

    def test_empty_glucose_data_raises(self) -> None:
        """An empty input must raise EmptyDataError, not a pandas error."""
        gd = GlucoseData(data_source=_empty_glucose_df())
        assert len(gd.data) == 0  # sanity check on the fixture
        with pytest.raises(EmptyDataError):
            prepare_data_for_agata(gd)

    def test_single_row_does_not_crash(self) -> None:
        """A single-row input yields a 1-row DataFrame with non-NaN glucose."""
        gd = GlucoseData(data_source=_single_row_glucose_df())
        result = prepare_data_for_agata(gd)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert list(result.columns) == ["t", "glucose"]
        assert not result["glucose"].isna().any()
