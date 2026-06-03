"""Tests for pure risk metric functions."""
import pandas as pd
import numpy as np
import pytest
from cgmpy.metrics.variability.risk import (
    m_value, j_index, grade, lbgi, hbgi, gri, adrr,
)

def test_m_value_constant():
    """M-Value should be 0 when all values equal the reference."""
    glucose = pd.Series([90.0, 90.0, 90.0])
    assert m_value(glucose, 90) == 0.0

def test_m_value_zero_glucose():
    """Should handle zero glucose without crashing."""
    glucose = pd.Series([0.0, 90.0, 180.0])
    result = m_value(glucose, 90)
    assert isinstance(result, float)
    assert result >= 0

def test_m_value_empty():
    """Should return 0.0 for empty series."""
    assert m_value(pd.Series([], dtype=float), 90) == 0.0

def test_j_index_known():
    """J-index: 0.001 * (mean + sd)^2"""
    # mean=100, sd=10 → 0.001 * 110^2 = 12.1
    result = j_index(100.0, 10.0)
    assert result == pytest.approx(12.1)

def test_lbgi_hbgi_sanity():
    """LBGI and HBGI should return finite floats."""
    glucose = pd.Series(np.linspace(50, 300, 100))
    lb = lbgi(glucose)
    hb = hbgi(glucose)
    assert np.isfinite(lb)
    assert np.isfinite(hb)
    assert lb >= 0
    assert hb >= 0

def test_lbgi_hbgi_zero_glucose():
    """Should handle zero glucose without crashing."""
    glucose = pd.Series([0.0, 100.0, 200.0])
    assert np.isfinite(lbgi(glucose))
    assert np.isfinite(hbgi(glucose))

def test_lbgi_hbgi_empty():
    """Should return 0.0 for empty series."""
    assert lbgi(pd.Series([], dtype=float)) == 0.0
    assert hbgi(pd.Series([], dtype=float)) == 0.0

def test_grade_returns_dict():
    """GRADE should return dict with expected keys."""
    glucose = pd.Series(np.linspace(70, 200, 50))
    result = grade(glucose)
    assert isinstance(result, dict)
    assert "grade_score" in result
    assert "hypo_percent" in result
    assert "eu_percent" in result
    assert "hyper_percent" in result

def test_grade_sums_to_100():
    """Hypo + eu + hyper percentages should sum to ~100%."""
    glucose = pd.Series(np.linspace(40, 300, 100))
    result = grade(glucose)
    total = result["hypo_percent"] + result["eu_percent"] + result["hyper_percent"]
    assert total == pytest.approx(100.0, abs=1.0)

def test_gri_returns_dict():
    """GRI should return expected structure."""
    glucose = pd.Series(np.linspace(50, 350, 288))
    result = gri(glucose, pregnancy=False)
    assert isinstance(result, dict)
    assert "GRI" in result
    assert "components" in result
    assert "derived_metrics" in result

def test_adrr_returns_dict():
    """ADRR should return expected structure."""
    import pandas as pd
    from datetime import datetime, timedelta
    start = datetime(2024, 1, 1, 0, 0)
    timestamps = pd.Series([start + timedelta(minutes=5*i) for i in range(288)])
    glucose = pd.Series(np.linspace(70, 250, 288))
    result = adrr(glucose, timestamps)
    assert isinstance(result, dict)
    assert "adrr" in result
    assert "risk_category" in result

def test_backward_compatibility():
    """Pure functions match GlucoseAnalysis results on real data."""
    from cgmpy import GlucoseAnalysis, GlucoseData
    import pytest
    data = GlucoseAnalysis(GlucoseData("tests/fixtures/synthetic/sine_24h.csv"))
    
    lb_old = data.LBGI()
    lb_new = lbgi(data.glucose)
    assert lb_old == pytest.approx(lb_new)
    
    hb_old = data.HBGI()
    hb_new = hbgi(data.glucose)
    assert hb_old == pytest.approx(hb_new)
    
    m_old = data.M_Value()
    m_new = m_value(data.glucose)
    assert m_old == pytest.approx(m_new)
    
    j_old = data.j_index()
    j_new = j_index(data.mean(), data.sd())
    assert j_old == pytest.approx(j_new)
