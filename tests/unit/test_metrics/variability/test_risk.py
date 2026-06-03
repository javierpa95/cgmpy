"""Tests for cgmpy.metrics.variability.risk.RiskMetrics."""

import numpy as np
import pandas as pd
import pytest

from cgmpy import GlucoseAnalysis, GlucoseData


@pytest.fixture
def analysis(stable_glucose_df):
    return GlucoseAnalysis(GlucoseData(data_source=stable_glucose_df))


def test_m_value_default(analysis):
    """M_Value should return a finite float rounded to 2 decimals."""
    val = analysis.M_Value()
    assert isinstance(val, float)
    assert np.isfinite(val)
    assert val == round(val, 2)


def test_m_value_with_custom_reference():
    """M_Value should accept a custom reference_glucose and change the result."""
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=48, freq="5min"),
            "glucose": [120] * 48,
        }
    )
    ga_default = GlucoseAnalysis(GlucoseData(data_source=df))
    ga_custom = GlucoseAnalysis(GlucoseData(data_source=df))
    v_default = ga_default.M_Value()
    v_custom = ga_custom.M_Value(reference_glucose=180)
    # Different reference values should change the M-Value.
    assert v_default != v_custom


def test_j_index(analysis):
    """J-index should equal 0.001 * (mean + sd) ** 2."""
    expected = (
        0.001 * (analysis.data["glucose"].mean() + analysis.data["glucose"].std()) ** 2
    )
    assert analysis.j_index() == pytest.approx(expected, rel=1e-4)


def test_grade_default_unit(analysis):
    """GRADE() default unit (mg/dL) should return a dict with the four components."""
    result = analysis.GRADE()
    assert set(result.keys()) == {"grade_score", "hypo_percent", "eu_percent", "hyper_percent"}
    # Percentages must sum to ~100% (within rounding).
    total = result["hypo_percent"] + result["eu_percent"] + result["hyper_percent"]
    assert 95.0 <= total <= 100.0
    assert isinstance(result["grade_score"], float)


def test_grade_mmol_l_unit():
    """GRADE(unit='mmol/L') should also work; values are internally scaled by 18."""
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=48, freq="5min"),
            "glucose": [5.5] * 48,  # 5.5 mmol/L ≈ 99 mg/dL
        }
    )
    ga = GlucoseAnalysis(GlucoseData(data_source=df))
    result = ga.GRADE(unit="mmol/L")
    assert "grade_score" in result
    assert isinstance(result["grade_score"], float)


def test_grade_invalid_unit_raises(analysis):
    """An invalid unit string should raise ValueError."""
    with pytest.raises(ValueError):
        analysis.GRADE(unit="g/L")


def test_lbgi(analysis):
    """LBGI should return a non-negative float (penalty for low values)."""
    val = analysis.LBGI()
    assert isinstance(val, float)
    assert val >= 0.0


def test_hbgi(analysis):
    """HBGI should return a non-negative float (penalty for high values)."""
    val = analysis.HBGI()
    assert isinstance(val, float)
    assert val >= 0.0


def test_gri_default(variable_glucose_df):
    """GRI() with default ranges should return a dict with GRI and components."""
    ga = GlucoseAnalysis(GlucoseData(data_source=variable_glucose_df))
    result = ga.GRI()
    assert "GRI" in result
    assert result["is_pregnancy"] is False
    assert result["validated"] is True
    assert "components" in result
    assert "derived_metrics" in result
    # Sum of components should be <= 100% (each is a percentage of time).
    c = result["components"]
    total = c["VLow"] + c["Low"] + c["VHigh"] + c["High"]
    assert 0.0 <= total <= 100.0


def test_gri_pregnancy(variable_glucose_df):
    """GRI(pregnancy=True) should use pregnancy-specific ranges and flag unvalidated."""
    ga = GlucoseAnalysis(GlucoseData(data_source=variable_glucose_df))
    result = ga.GRI(pregnancy=True)
    assert result["is_pregnancy"] is True
    assert result["validated"] is False
    assert "GRI" in result


def test_adrr_classification(analysis):
    """ADRR should return a dict with 'adrr' and a 'risk_category' in {Low, Moderate, High}."""
    result = analysis.ADRR()
    assert "adrr" in result
    assert "risk_category" in result
    assert result["risk_category"] in {"Low", "Moderate", "High"}
    assert "components" in result
    assert "hypo_risk" in result["components"]
    assert "hyper_risk" in result["components"]
