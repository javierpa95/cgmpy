"""Tests for pure MODD, CONGA, and Lability functions."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest
from cgmpy.metrics.variability.modd import modd
from cgmpy.metrics.variability.conga import conga
from cgmpy.metrics.variability.lability import lability_index

# --- MODD ---

def test_modd_constant():
    """MODD should be 0 for constant daily data."""
    start = datetime(2024, 1, 1, 0, 0)
    timestamps = pd.Series([start + timedelta(minutes=5*i) for i in range(288)])
    glucose = pd.Series([100.0] * 288)
    result = modd(glucose, timestamps, days=1)
    assert result["value"] == pytest.approx(0.0, abs=1e-6)

def test_modd_invalid_days():
    """MODD should raise for days outside [1,6]."""
    start = datetime(2024, 1, 1, 0, 0)
    timestamps = pd.Series([start + timedelta(minutes=5*i) for i in range(288)])
    glucose = pd.Series([100.0] * 288)
    with pytest.raises(ValueError):
        modd(glucose, timestamps, days=0)
    with pytest.raises(ValueError):
        modd(glucose, timestamps, days=7)

# --- CONGA ---

def test_conga_constant():
    """CONGA should be 0 for constant data."""
    start = datetime(2024, 1, 1, 0, 0)
    timestamps = pd.Series([start + timedelta(minutes=5*i) for i in range(288)])
    glucose = pd.Series([100.0] * 288)
    result = conga(glucose, timestamps, hours=1)
    assert result["value"] == pytest.approx(0.0, abs=1e-6)

def test_conga_returns_dict():
    """CONGA should return dict with value and n."""
    start = datetime(2024, 1, 1, 0, 0)
    timestamps = pd.Series([start + timedelta(minutes=5*i) for i in range(288)])
    glucose = pd.Series(np.linspace(70, 200, 288))
    result = conga(glucose, timestamps, hours=4)
    assert isinstance(result, dict)
    assert "value" in result
    assert "n" in result

# --- Lability ---

def test_lability_constant():
    """Lability should be 0 for constant data."""
    start = datetime(2024, 1, 1, 0, 0)
    timestamps = pd.Series([start + timedelta(minutes=5*i) for i in range(288)])
    glucose = pd.Series([100.0] * 288)
    result = lability_index(glucose, timestamps, interval=1)
    assert isinstance(result, dict)
    assert result["mean_li"] == pytest.approx(0.0, abs=1e-6)

def test_lability_invalid_interval():
    """Lability should raise for interval <= 0."""
    start = datetime(2024, 1, 1, 0, 0)
    timestamps = pd.Series([start + timedelta(minutes=5*i) for i in range(288)])
    glucose = pd.Series([100.0] * 288)
    with pytest.raises(ValueError):
        lability_index(glucose, timestamps, interval=0)
    with pytest.raises(ValueError):
        lability_index(glucose, timestamps, interval=-1)

# --- Backward compatibility ---

def test_backward_compatibility():
    """Pure functions match GlucoseAnalysis results on real data."""
    from cgmpy import GlucoseAnalysis, GlucoseData

    gm = GlucoseAnalysis(GlucoseData("tests/fixtures/synthetic/sine_24h.csv"))

    # MODD
    old_modd = gm.MODD(days=1)
    new_modd = modd(gm.glucose, gm.timestamps, days=1)
    assert old_modd["value"] == pytest.approx(new_modd["value"])

    # CONGA
    old_conga = gm.CONGA(hours=4)
    new_conga = conga(gm.glucose, gm.timestamps, hours=4)
    assert old_conga["value"] == pytest.approx(new_conga["value"])

    # Lability
    old_li = gm.Lability_index(interval=1)
    new_li = lability_index(gm.glucose, gm.timestamps, interval=1)
    assert old_li["mean_li"] == pytest.approx(new_li["mean_li"])
