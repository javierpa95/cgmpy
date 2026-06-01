"""Tests for `cgmpy.agata.metrics` (Agata bridge).

These tests are skipped if the optional `py_agata` dependency is missing.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("py_agata") is None,
        reason="py_agata optional dependency is not installed",
    ),
    # py_agata emits RuntimeWarnings on empty event arrays / all-NaN slices when
    # the synthetic dataset has no hypo/hyper events. They are harmless here.
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
]

from cgmpy.agata.metrics import (  # noqa: E402
    AgataAnalysis,
    analyze_one_arm,
    analyze_with_agata,
    summarize_agata_results,
)
from cgmpy.data.core import ModularGlucoseData  # noqa: E402

EXPECTED_TOP_LEVEL_KEYS = {
    "variability",
    "time_in_ranges",
    "risk",
    "glycemic_transformation",
    "data_quality",
}


@pytest.fixture
def week_glucose_df() -> pd.DataFrame:
    """A 7-day synthetic dataset at 5-min intervals."""
    start = datetime(2024, 1, 1, 0, 0)
    n = 288 * 7
    times = [start + timedelta(minutes=5 * i) for i in range(n)]
    rng = np.random.default_rng(123)
    glucose = 130 + 30 * np.sin(np.linspace(0, 14 * np.pi, n)) + rng.normal(0, 5, n)
    return pd.DataFrame({"time": times, "glucose": glucose})


class TestAnalyzeWithAgata:
    """Tests for the functional `analyze_with_agata()` entry point."""

    def test_returns_dict_with_top_level_sections(self, week_glucose_df: pd.DataFrame) -> None:
        """The full analysis returns the expected top-level sections."""
        gd = ModularGlucoseData(data_source=week_glucose_df)
        results = analyze_with_agata(gd)
        assert isinstance(results, dict)
        assert EXPECTED_TOP_LEVEL_KEYS.issubset(results.keys())

    def test_variability_section_has_metrics(self, week_glucose_df: pd.DataFrame) -> None:
        """The variability section exposes mean/median/std glucose."""
        gd = ModularGlucoseData(data_source=week_glucose_df)
        results = analyze_with_agata(gd)
        var = results["variability"]
        assert "mean_glucose" in var
        assert "median_glucose" in var
        assert "std_glucose" in var

    def test_diabetes_target_is_default(self, week_glucose_df: pd.DataFrame) -> None:
        """The default `glycemic_target='diabetes'` runs without error."""
        gd = ModularGlucoseData(data_source=week_glucose_df)
        results = analyze_with_agata(gd, glycemic_target="diabetes")
        assert isinstance(results, dict)


class TestAgataAnalysisClass:
    """Tests for the `AgataAnalysis` OO wrapper."""

    def test_init_sets_glycemic_target(self, week_glucose_df: pd.DataFrame) -> None:
        """The constructor stores the `glycemic_target` on the instance."""
        analyzer = AgataAnalysis(data_source=week_glucose_df, glycemic_target="diabetes")
        assert analyzer.glycemic_target == "diabetes"

    def test_run_returns_dict(self, week_glucose_df: pd.DataFrame) -> None:
        """`.run()` returns the full nested results dictionary."""
        analyzer = AgataAnalysis(data_source=week_glucose_df)
        results = analyzer.run()
        assert isinstance(results, dict)
        assert EXPECTED_TOP_LEVEL_KEYS.issubset(results.keys())

    def test_run_with_summary_returns_flat_dict(self, week_glucose_df: pd.DataFrame) -> None:
        """`.run(summary=True)` returns a flat key→value dict."""
        analyzer = AgataAnalysis(data_source=week_glucose_df)
        summary = analyzer.run(summary=True)
        assert isinstance(summary, dict)
        assert all(not isinstance(v, dict) for v in summary.values())
        # A few summary keys we expect after flattening:
        assert any(k.startswith("variability_") for k in summary)
        assert any(k.startswith("time_in_ranges_") for k in summary)

    def test_inherits_modular_glucose_data(self, week_glucose_df: pd.DataFrame) -> None:
        """`AgataAnalysis` inherits the data-handling facade."""
        analyzer = AgataAnalysis(data_source=week_glucose_df)
        assert isinstance(analyzer, ModularGlucoseData)
        assert len(analyzer.data) > 0


class TestAnalyzeOneArm:
    """Tests for `analyze_one_arm()`."""

    def test_returns_dict_for_arm_of_two(self, week_glucose_df: pd.DataFrame) -> None:
        """A two-subject arm analysis returns a dict with the expected sections."""
        gd1 = ModularGlucoseData(data_source=week_glucose_df.copy())
        df2 = week_glucose_df.copy()
        df2["time"] = df2["time"] + pd.Timedelta(days=14)
        gd2 = ModularGlucoseData(data_source=df2)
        results = analyze_one_arm([gd1, gd2])
        assert isinstance(results, dict)
        # `analyze_one_arm` reports aggregated stats, no per-subject `events`.
        for key in ("variability", "time_in_ranges", "risk", "data_quality"):
            assert key in results

    def test_classmethod_with_summary(self, week_glucose_df: pd.DataFrame) -> None:
        """`AgataAnalysis.analyze_one_arm(..., summary=True)` returns a flat dict.

        For per-arm analysis, py_agata aggregates each metric across subjects, so the
        flattened values are dicts of summary statistics (mean, median, prc_25, ...).
        We only assert top-level flatness and the presence of variability_ keys.
        """
        gd1 = ModularGlucoseData(data_source=week_glucose_df.copy())
        df2 = week_glucose_df.copy()
        df2["time"] = df2["time"] + pd.Timedelta(days=14)
        gd2 = ModularGlucoseData(data_source=df2)
        summary = AgataAnalysis.analyze_one_arm([gd1, gd2], summary=True)
        assert isinstance(summary, dict)
        assert any(k.startswith("variability_") for k in summary)
        assert any(k.startswith("time_in_ranges_") for k in summary)


class TestSummarizeAgataResults:
    """Tests for the lightweight `summarize_agata_results()` helper."""

    def test_flattens_nested_dict(self, week_glucose_df: pd.DataFrame) -> None:
        """The flat summary has prefixed keys matching the nested categories."""
        gd = ModularGlucoseData(data_source=week_glucose_df)
        nested = analyze_with_agata(gd)
        flat = summarize_agata_results(nested)
        assert isinstance(flat, dict)
        assert all(not isinstance(v, dict) for v in flat.values())
        # At least one variability_* key must be present
        assert any(k.startswith("variability_") for k in flat)
