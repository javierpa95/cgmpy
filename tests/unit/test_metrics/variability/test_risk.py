"""Tests for cgmpy.metrics.variability.risk.RiskMetrics."""

import numpy as np
import pandas as pd
import pytest

from cgmpy import GlucoseMetrics


def test_m_value_default(stable_glucose_df):
    """M_Value should return a finite float rounded to 2 decimals."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    val = gm.M_Value()
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
    gm_default = GlucoseMetrics(data_source=df)
    gm_custom = GlucoseMetrics(data_source=df)
    v_default = gm_default.M_Value()
    v_custom = gm_custom.M_Value(reference_glucose=180)
    # Different reference values should change the M-Value.
    assert v_default != v_custom


def test_j_index(stable_glucose_df):
    """J-index should equal 0.001 * (mean + sd) ** 2."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    expected = 0.001 * (stable_glucose_df["glucose"].mean()
                        + stable_glucose_df["glucose"].std()) ** 2
    assert gm.j_index() == pytest.approx(expected, rel=1e-4)


def test_grade_default_unit(stable_glucose_df):
    """GRADE() default unit (mg/dL) should return a dict with the four components."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    result = gm.GRADE()
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
    gm = GlucoseMetrics(data_source=df)
    result = gm.GRADE(unit="mmol/L")
    assert "grade_score" in result
    assert isinstance(result["grade_score"], float)


def test_grade_invalid_unit_raises(stable_glucose_df):
    """An invalid unit string should raise ValueError."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    with pytest.raises(ValueError):
        gm.GRADE(unit="g/L")


def test_lbgi(stable_glucose_df):
    """LBGI should return a non-negative float (penalty for low values)."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    val = gm.LBGI()
    assert isinstance(val, float)
    assert val >= 0.0


def test_hbgi(stable_glucose_df):
    """HBGI should return a non-negative float (penalty for high values)."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    val = gm.HBGI()
    assert isinstance(val, float)
    assert val >= 0.0


def test_gri_default(variable_glucose_df):
    """GRI() with default ranges should return a dict with GRI and components."""
    gm = GlucoseMetrics(data_source=variable_glucose_df)
    result = gm.GRI()
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
    gm = GlucoseMetrics(data_source=variable_glucose_df)
    result = gm.GRI(pregnancy=True)
    assert result["is_pregnancy"] is True
    assert result["validated"] is False
    assert "GRI" in result


def test_adrr_classification(stable_glucose_df):
    """ADRR should return a dict with 'adrr' and a 'risk_category' in {Low, Moderate, High}."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    result = gm.ADRR()
    assert "adrr" in result
    assert "risk_category" in result
    assert result["risk_category"] in {"Low", "Moderate", "High"}
    assert "components" in result
    assert "hypo_risk" in result["components"]
    assert "hyper_risk" in result["components"]
