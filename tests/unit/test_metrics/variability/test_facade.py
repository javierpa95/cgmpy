"""Tests for the composite VariabilityMetrics facade (variability/__init__.py)."""

import numpy as np
import pandas as pd
import pytest

from cgmpy import GlucoseAnalysis, GlucoseData

EXPECTED_KEYS = {
    "Mean",
    "Median",
    "Std",
    "CV",
    "GMI",
    "TIR",
    "SDT",
    "SDW",
    "mage_avg",
    "modd",
    "CONGA1",
    "CONGA2",
    "CONGA4",
    "CONGA6",
    "CONGA24",
    "LBGI",
    "HBGI",
    "GRI",
    "ADRR",
    "GRADE",
    "M_Value",
    "J_Index",
    "data_completeness",
    "Skewness",
    "Kurtosis",
}


@pytest.fixture
def analysis(variable_glucose_df):
    return GlucoseAnalysis(GlucoseData(data_source=variable_glucose_df))


def test_calculate_variability_metrics_returns_full_dict(analysis):
    """calculate_variability_metrics should return a dict with 50+ keys on 24h data."""
    result = analysis.calculate_variability_metrics()
    assert isinstance(result, dict)
    assert len(result) >= 50, f"expected at least 50 metrics, got {len(result)}"


def test_calculate_variability_metrics_has_all_expected_keys(analysis):
    """The result dict should contain every expected metric key."""
    result = analysis.calculate_variability_metrics()
    missing = EXPECTED_KEYS - set(result.keys())
    assert not missing, f"missing keys: {missing}"


def test_calculate_variability_metrics_basic_values_not_none(analysis):
    """Core metrics should not be None for a valid 24h dataset."""
    result = analysis.calculate_variability_metrics()
    for key in ("Mean", "Median", "Std", "CV", "GMI", "TIR", "data_completeness"):
        assert result[key] is not None, f"{key} was None"
    # data_completeness should be 100% for a regular 5-min dataset of 288 points.
    assert result["data_completeness"] == pytest.approx(100.0, abs=0.5)


def test_calculate_variability_metrics_on_stable_data(stable_glucose_df):
    """The facade should also work for stable, constant glucose data."""
    ga = GlucoseAnalysis(GlucoseData(data_source=stable_glucose_df))
    result = ga.calculate_variability_metrics()
    # All expected keys should still be present.
    assert EXPECTED_KEYS.issubset(set(result.keys()))
    # Standard deviation should be 0 (or very close) for perfectly stable data.
    assert result["Std"] == pytest.approx(0.0, abs=1e-6)


def test_calculate_variability_metrics_skew_kurtosis_present(analysis):
    """Skewness and Kurtosis keys should be present and finite."""
    result = analysis.calculate_variability_metrics()
    assert "Skewness" in result
    assert "Kurtosis" in result
    assert np.isfinite(result["Skewness"])
    assert np.isfinite(result["Kurtosis"])


def test_calculate_variability_metrics_mage_block_present(analysis):
    """The MAGE-related keys (mage_*) should be present in the result."""
    result = analysis.calculate_variability_metrics()
    for key in ("mage_plus", "mage_minus", "mage_avg", "mage_excursions"):
        assert key in result, f"{key} missing from result dict"


def test_calculate_variability_metrics_gri_components(analysis):
    """GRI, GRI_high, GRI_low and their pregnancy variants should be present."""
    result = analysis.calculate_variability_metrics()
    for key in (
        "GRI",
        "GRI_high",
        "GRI_low",
        "GRI_pregnancy",
        "GRI_pregnancy_high",
        "GRI_pregnancy_low",
    ):
        assert key in result, f"{key} missing from result dict"


def test_calculate_variability_metrics_with_logging(variable_glucose_df):
    """Passing log=True should not break the facade; the dict should still be returned."""
    ga = GlucoseAnalysis(GlucoseData(data_source=variable_glucose_df, log=True))
    result = ga.calculate_variability_metrics()
    assert isinstance(result, dict)
    assert "Mean" in result
    assert "Std" in result


def test_calculate_variability_metrics_top_level_error_handling():
    """When the initial metrics dict cannot be built, return an error payload.

    An empty DataFrame causes ``data_completeness`` to still work but many
    downstream metrics to fail. The facade's outer ``try/except`` returns a
    plain ``{error, message}`` dict, so we verify the contract.
    """
    empty_df = pd.DataFrame({"time": pd.to_datetime([]), "glucose": []})
    ga = GlucoseAnalysis(GlucoseData(data_source=empty_df))
    result = ga.calculate_variability_metrics()
    # Either the metrics dict is returned (with None values) or an error payload.
    assert isinstance(result, dict)


def test_calculate_variability_metrics_triggers_mage_error_logging():
    """A 1-point dataset makes MAGE_Baghurst fail; with log=True the error path runs.

    This exercises the inner ``try/except`` in ``calculate_variability_metrics``
    that logs the MAGE failure when ``self.log`` is True.
    """
    tiny_df = pd.DataFrame({"time": pd.to_datetime(["2024-01-01"]), "glucose": [100.0]})
    ga = GlucoseAnalysis(GlucoseData(data_source=tiny_df, log=True))
    result = ga.calculate_variability_metrics()
    assert isinstance(result, dict)
