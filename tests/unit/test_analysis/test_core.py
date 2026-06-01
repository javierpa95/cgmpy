"""Tests for `cgmpy.analysis.core.GlucoseAnalysis`."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cgmpy.analysis.core import GlucoseAnalysis

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "data"


@pytest.fixture
def two_day_glucose_df() -> pd.DataFrame:
    """A 2-day synthetic trace with realistic variability (5-min intervals)."""
    start = datetime(2024, 1, 1, 0, 0)
    n = 288 * 2
    times = [start + timedelta(minutes=5 * i) for i in range(n)]
    rng = np.random.default_rng(42)
    glucose = 130 + 30 * np.sin(np.linspace(0, 6 * np.pi, n)) + rng.normal(0, 5, n)
    return pd.DataFrame({"time": times, "glucose": glucose})


class TestGlucoseAnalysisInit:
    """Constructor tests for the GlucoseAnalysis facade."""

    def test_init_with_dataframe(self, stable_glucose_df: pd.DataFrame) -> None:
        """`GlucoseAnalysis(data_source=df)` constructs successfully."""
        ga = GlucoseAnalysis(data_source=stable_glucose_df)
        assert ga is not None
        assert not ga.data.empty
        assert "time" in ga.data.columns
        assert "glucose" in ga.data.columns

    def test_init_with_csv_file(self) -> None:
        """`GlucoseAnalysis` loads a CSV file end-to-end."""
        ga = GlucoseAnalysis(data_source=str(FIXTURES / "dm.csv"))
        assert ga is not None
        assert len(ga.data) > 0

    def test_init_with_date_range(self, two_day_glucose_df: pd.DataFrame) -> None:
        """`start_date` / `end_date` filter the loaded data."""
        ga = GlucoseAnalysis(
            data_source=two_day_glucose_df,
            start_date="2024-01-01 00:00",
            end_date="2024-01-01 23:55",
        )
        assert ga.data["time"].min() >= pd.Timestamp("2024-01-01 00:00")
        assert ga.data["time"].max() <= pd.Timestamp("2024-01-01 23:55")

    def test_inherits_modular_glucose_data(self, stable_glucose_df: pd.DataFrame) -> None:
        """GlucoseAnalysis inherits the data-handling facade."""
        from cgmpy.data.core import ModularGlucoseData

        ga = GlucoseAnalysis(data_source=stable_glucose_df)
        assert isinstance(ga, ModularGlucoseData)


class TestGlucoseAnalysisInfo:
    """Tests for the `info()` and quality-metric methods."""

    def test_info_returns_expected_keys(self, two_day_glucose_df: pd.DataFrame) -> None:
        """`info()` returns a dict with the canonical keys."""
        ga = GlucoseAnalysis(data_source=two_day_glucose_df)
        info = ga.info()
        assert isinstance(info, dict)
        for key in ("n_records", "start_date", "end_date", "typical_interval", "completeness"):
            assert key in info

    def test_info_record_count(self, two_day_glucose_df: pd.DataFrame) -> None:
        """`info()['n_records']` matches len(data)."""
        ga = GlucoseAnalysis(data_source=two_day_glucose_df)
        assert ga.info()["n_records"] == len(ga.data)

    def test_get_data_quality_metrics(self, two_day_glucose_df: pd.DataFrame) -> None:
        """`get_data_quality_metrics()` returns the expected dict."""
        ga = GlucoseAnalysis(data_source=two_day_glucose_df)
        quality = ga.get_data_quality_metrics()
        assert isinstance(quality, dict)
        for key in ("total_gaps", "mean_interval", "min_glucose", "max_glucose"):
            assert key in quality

    def test_typical_interval_is_five_minutes(self, two_day_glucose_df: pd.DataFrame) -> None:
        """5-min sampling resolves to a 5-minute typical interval."""
        ga = GlucoseAnalysis(data_source=two_day_glucose_df)
        assert ga.get_typical_interval() == pytest.approx(5.0, abs=0.1)


class TestGlucoseAnalysisMetrics:
    """Tests for the metric helpers exposed via inheritance."""

    def test_time_statistics_returns_dict(self, two_day_glucose_df: pd.DataFrame) -> None:
        """`time_statistics()` returns a dict with TIR/TBR/TAR keys."""
        ga = GlucoseAnalysis(data_source=two_day_glucose_df)
        stats = ga.time_statistics()
        assert isinstance(stats, dict)
        assert "TIR" in stats
        assert "TBR_total" in stats
        assert "TAR_total" in stats

    def test_time_range_summary(self, two_day_glucose_df: pd.DataFrame) -> None:
        """`time_range_summary()` returns the standard + pregnancy ranges."""
        ga = GlucoseAnalysis(data_source=two_day_glucose_df)
        summary = ga.time_range_summary()
        assert isinstance(summary, dict)
        assert "standard_ranges" in summary
        assert "pregnancy_ranges" in summary

    def test_tir_is_percentage(self, two_day_glucose_df: pd.DataFrame) -> None:
        """TIR returns a value in [0, 100]."""
        ga = GlucoseAnalysis(data_source=two_day_glucose_df)
        tir = ga.TIR()
        assert 0.0 <= tir <= 100.0

    def test_stable_glucose_has_high_tir(self, stable_glucose_df: pd.DataFrame) -> None:
        """A constant glucose of 100 mg/dL yields TIR == 100."""
        ga = GlucoseAnalysis(data_source=stable_glucose_df)
        assert ga.TIR() == pytest.approx(100.0)


class TestGlucoseAnalysisDataAccess:
    """Tests for the read-only data-access helpers."""

    def test_get_raw_data_returns_copy(self, stable_glucose_df: pd.DataFrame) -> None:
        """`get_raw_data()` returns a DataFrame copy, not a view."""
        ga = GlucoseAnalysis(data_source=stable_glucose_df)
        raw = ga.get_raw_data()
        assert isinstance(raw, pd.DataFrame)
        # Mutating the copy must not mutate the underlying data
        raw.loc[raw.index[0], "glucose"] = 999.0
        assert ga.data.loc[ga.data.index[0], "glucose"] != 999.0

    def test_get_glucose_values_is_series(self, stable_glucose_df: pd.DataFrame) -> None:
        """`get_glucose_values()` returns a pandas Series."""
        ga = GlucoseAnalysis(data_source=stable_glucose_df)
        values = ga.get_glucose_values()
        assert isinstance(values, pd.Series)
        assert len(values) == len(ga.data)

    def test_get_timestamps_is_series(self, stable_glucose_df: pd.DataFrame) -> None:
        """`get_timestamps()` returns a pandas Series of timestamps."""
        ga = GlucoseAnalysis(data_source=stable_glucose_df)
        ts = ga.get_timestamps()
        assert isinstance(ts, pd.Series)
        assert pd.api.types.is_datetime64_any_dtype(ts)


class TestGlucoseAnalysisFilter:
    """Tests for the inherited filter helpers."""

    def test_filter_by_date_range(self, two_day_glucose_df: pd.DataFrame) -> None:
        """`filter_by_date_range()` returns a new instance restricted to the range."""
        ga = GlucoseAnalysis(data_source=two_day_glucose_df)
        filtered = ga.filter_by_date_range("2024-01-01 06:00", "2024-01-01 18:00")
        assert isinstance(filtered, GlucoseAnalysis)
        assert filtered.data["time"].min() >= pd.Timestamp("2024-01-01 06:00")
        assert filtered.data["time"].max() <= pd.Timestamp("2024-01-01 18:00")
        # Original is untouched
        assert len(ga.data) == len(two_day_glucose_df)


def _stub_basic_stats() -> dict:
    """Synthetic basic-statistics dict used to exercise the report builders."""
    return {
        "GMI": 6.5,
        "Mean": 130.0,
        "Median": 128.0,
        "Std": 20.0,
        "CV": 15.0,
    }


def _stub_variability() -> dict:
    """Synthetic variability dict used to exercise the report builders."""
    return {
        "MAGE": 50.0,
        "MODD": 10.0,
        "CONGA": 25.0,
        "SD_total": 20.0,
        "SD_within_day": 15.0,
        "SD_between_day": 8.0,
    }


class TestFlattenReport:
    """Tests for the private `_flatten_report()` helper."""

    def test_flatten_simple_report(self, stable_glucose_df: pd.DataFrame) -> None:
        """A nested report is flattened with `section_key` keys."""
        ga = GlucoseAnalysis(data_source=stable_glucose_df)
        report = {
            "basic_info": {"n_records": 288, "completeness": 100.0},
            "basic_metrics": {"GMI": 5.7, "Mean": 100.0},
        }
        flat = ga._flatten_report(report)
        assert isinstance(flat, pd.DataFrame)
        assert "basic_info_n_records" in flat.columns
        assert "basic_metrics_GMI" in flat.columns
        assert flat.iloc[0]["basic_info_n_records"] == 288
        assert flat.iloc[0]["basic_metrics_GMI"] == pytest.approx(5.7)

    def test_flatten_preserves_scalar_sections(self, stable_glucose_df: pd.DataFrame) -> None:
        """Non-dict values at the top level pass through unchanged."""
        ga = GlucoseAnalysis(data_source=stable_glucose_df)
        report = {"basic_info": {"n_records": 10}, "total_count": 42}
        flat = ga._flatten_report(report)
        assert flat.iloc[0]["total_count"] == 42


class TestReportBuilders:
    """Tests for `get_comprehensive_report` / `get_summary_string` / `export_report`.

    These methods compose results from sibling metric helpers that are not part of
    the `GlucoseAnalysis` MRO. We stub the missing methods on the instance so the
    composition logic in `analysis/core.py` can be exercised directly.
    """

    @staticmethod
    def _patch_missing_helpers(ga: GlucoseAnalysis) -> None:
        """Attach the helpers the builders expect onto the instance."""
        ga.basic_statistics_summary = _stub_basic_stats  # type: ignore[attr-defined]
        ga.calculate_all_variability_metrics = _stub_variability  # type: ignore[attr-defined]

    def test_get_comprehensive_report_assembles_sections(
        self, stable_glucose_df: pd.DataFrame
    ) -> None:
        """`get_comprehensive_report()` returns the documented section keys."""
        ga = GlucoseAnalysis(data_source=stable_glucose_df)
        self._patch_missing_helpers(ga)
        report = ga.get_comprehensive_report()
        assert isinstance(report, dict)
        for key in (
            "basic_info",
            "basic_metrics",
            "time_statistics",
            "variability_metrics",
            "data_quality",
        ):
            assert key in report

    def test_get_summary_string_contains_key_metrics(self, stable_glucose_df: pd.DataFrame) -> None:
        """`get_summary_string()` renders the report header for the populated sections.

        Bug fixed in v0.5.1: the function used to read legacy keys
        (``TIR_tight``, ``TBR70`` ...) from `time_statistics()` and raise
        ``KeyError``. It now calls the individual time-in-range methods
        directly. The helper that patches in basic_statistics_summary /
        calculate_all_variability_metrics is no longer required, but we keep
        the fixture for stability.
        """
        ga = GlucoseAnalysis(data_source=stable_glucose_df)
        self._patch_missing_helpers(ga)
        text = ga.get_summary_string()
        # Header lines from the four sections must be present.
        assert "DATA:" in text
        assert "BASIC METRICS:" in text
        assert "TIME IN RANGE:" in text
        assert "VARIABILITY:" in text
        # The legacy keys are now rendered as labels, not as dict lookups.
        assert "TIR tight" in text
        assert "TBR70" in text
        assert "TBR55" in text
        assert "TAR180" in text
        assert "TAR250" in text

    def test_export_report_json(self, stable_glucose_df: pd.DataFrame, tmp_path) -> None:
        """`export_report(format='json')` writes a valid JSON file."""
        import json

        ga = GlucoseAnalysis(data_source=stable_glucose_df)
        self._patch_missing_helpers(ga)
        out = tmp_path / "report.json"
        ga.export_report(str(out), format="json")
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert "basic_info" in loaded

    def test_export_report_csv(self, stable_glucose_df: pd.DataFrame, tmp_path) -> None:
        """`export_report(format='csv')` writes a flat CSV via `_flatten_report`."""
        ga = GlucoseAnalysis(data_source=stable_glucose_df)
        self._patch_missing_helpers(ga)
        out = tmp_path / "report.csv"
        ga.export_report(str(out), format="csv")
        assert out.exists()
        loaded = pd.read_csv(out)
        assert "basic_info_n_records" in loaded.columns

    def test_export_report_unsupported_format(
        self, stable_glucose_df: pd.DataFrame, tmp_path
    ) -> None:
        """An unsupported format raises ValueError."""
        ga = GlucoseAnalysis(data_source=stable_glucose_df)
        self._patch_missing_helpers(ga)
        out = tmp_path / "report.xyz"
        with pytest.raises(ValueError, match="Unsupported format"):
            ga.export_report(str(out), format="xyz")
