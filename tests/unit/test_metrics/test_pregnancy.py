"""Tests for `cgmpy.metrics.pregnancy.PregnancyAnalysis`."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from cgmpy.metrics.pregnancy import PregnancyAnalysis


@pytest.fixture
def pregnancy_synthetic_df() -> pd.DataFrame:
    """A synthetic 30-week pregnancy trace at 15-min intervals."""
    start = datetime(2024, 1, 1, 0, 0)
    n_days = 210  # 30 weeks
    n = n_days * 24 * 4  # 15-min intervals
    times = [start + timedelta(minutes=15 * i) for i in range(n)]
    rng = np.random.default_rng(0)
    glucose = 100 + 10 * np.sin(np.linspace(0, 60 * np.pi, n)) + rng.normal(0, 4, n)
    return pd.DataFrame({"time": times, "glucose": glucose})


@pytest.fixture
def pregnancy_delivery_date(pregnancy_synthetic_df: pd.DataFrame) -> str:
    """Delivery date 10 weeks past last record -> week 30 gestation."""
    last_time = pregnancy_synthetic_df["time"].iloc[-1]
    delivery = last_time + timedelta(weeks=10)
    return delivery.strftime("%Y-%m-%d")


@pytest.fixture
def pregnancy_analysis(
    pregnancy_synthetic_df: pd.DataFrame,
    pregnancy_delivery_date: str,
) -> PregnancyAnalysis:
    """A ready-to-use PregnancyAnalysis instance."""
    return PregnancyAnalysis(
        data_source=pregnancy_synthetic_df,
        delivery_date=pregnancy_delivery_date,
        week=30,
        day=0,
    )


class TestPregnancyAnalysisInit:
    """Constructor tests."""

    def test_init_succeeds(self, pregnancy_analysis: PregnancyAnalysis) -> None:
        """The constructor builds an instance and exposes pregnancy attributes."""
        pa = pregnancy_analysis
        assert pa is not None
        assert not pa.data.empty

    def test_trimester_wrappers_present(self, pregnancy_analysis: PregnancyAnalysis) -> None:
        """t1, t2, t3 are exposed (some may be None when no data)."""
        pa = pregnancy_analysis
        assert hasattr(pa, "t1")
        assert hasattr(pa, "t2")
        assert hasattr(pa, "t3")
        non_empty = [t for t in (pa.t1, pa.t2, pa.t3) if t is not None]
        assert len(non_empty) >= 1


class TestSummaryByTrimester:
    """Tests for `summary_by_trimester()`."""

    def test_returns_three_keys(self, pregnancy_analysis: PregnancyAnalysis) -> None:
        """The summary has T1, T2, T3 keys."""
        summary = pregnancy_analysis.summary_by_trimester()
        assert isinstance(summary, dict)
        assert set(summary.keys()) == {"T1", "T2", "T3"}

    def test_populated_trimester_has_metrics(self, pregnancy_analysis: PregnancyAnalysis) -> None:
        """At least one populated trimester contains the simplified-metric keys."""
        summary = pregnancy_analysis.summary_by_trimester()
        populated = [v for v in summary.values() if v is not None]
        assert len(populated) >= 1
        first = populated[0]
        for key in ("GMI", "Mean", "Median", "SD", "CV", "TIR"):
            assert key in first


class TestCalculateAllMetrics:
    """Tests for `calculate_all_metrics()`."""

    def test_nested_returns_expected_sections(self, pregnancy_analysis: PregnancyAnalysis) -> None:
        """The nested output has gestation, overall, trimesters sections."""
        result = pregnancy_analysis.calculate_all_metrics(flatten=False)
        assert isinstance(result, dict)
        for key in ("gestation", "overall", "trimesters"):
            assert key in result

    def test_gestation_section_contents(self, pregnancy_analysis: PregnancyAnalysis) -> None:
        """The gestation block contains weeks/days/conception/delivery."""
        gestation = pregnancy_analysis.calculate_all_metrics(flatten=False)["gestation"]
        assert gestation["weeks"] == 30
        assert gestation["days"] == 0
        assert "conception" in gestation
        assert "delivery" in gestation

    def test_flatten_produces_flat_dict(self, pregnancy_analysis: PregnancyAnalysis) -> None:
        """`flatten=True` returns a single-level dict with prefixed keys."""
        flat = pregnancy_analysis.calculate_all_metrics(flatten=True)
        assert isinstance(flat, dict)
        assert all(not isinstance(v, dict) for v in flat.values())
        prefixes = {k.split("_")[0] for k in flat}
        assert "gest" in prefixes
        assert "total" in prefixes
        assert any(p in prefixes for p in ("t1", "t2", "t3"))


class TestPregnancyAnalysisStr:
    """Tests for the `__str__` representation."""

    def test_str_contains_header_and_metrics(self, pregnancy_analysis: PregnancyAnalysis) -> None:
        """The string output mentions the gestational diabetes report header."""
        s = str(pregnancy_analysis)
        assert isinstance(s, str)
        assert "PREGNANCY ANALYSIS REPORT" in s
        assert "Trimester Breakdown" in s

    def test_str_includes_overall_gmi(self, pregnancy_analysis: PregnancyAnalysis) -> None:
        """The string output includes an overall GMI value."""
        s = str(pregnancy_analysis)
        assert "GMI" in s
        assert "TIR" in s


class TestPregnancyMetricsBehavior:
    """Sanity checks on the simplified metric flow under pregnancy targets."""

    def test_all_simplified_uses_pregnancy_keys(
        self, pregnancy_analysis: PregnancyAnalysis
    ) -> None:
        """The `all_simplified()` flow includes pregnancy-specific TAR/TBR keys."""
        simplified = pregnancy_analysis.all_simplified()
        assert isinstance(simplified, dict)
        assert "TAR140" in simplified
        assert "TBR63" in simplified
        assert "TAR180" not in simplified
        assert "TBR70" not in simplified
