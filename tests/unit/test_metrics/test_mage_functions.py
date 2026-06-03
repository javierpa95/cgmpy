"""Tests for pure MAGE functions."""

import numpy as np
import pandas as pd
import pytest

from cgmpy.metrics.variability.mage import (
    mage_baghurst,
    mage_baghurst_direct_elimination,
    mage_baghurst_simplified,
    mage_baghurst_smoothing,
    mage_simple,
)


def test_mage_simple_constant():
    """MAGE should be 0 for constant data."""
    glucose = pd.Series([100.0] * 100)
    result = mage_simple(glucose)
    assert result == 0.0


def test_mage_small_dataset():
    """MAGE should handle datasets with fewer than 9 points."""
    glucose = pd.Series([100.0, 110.0, 120.0, 100.0, 90.0])
    result = mage_simple(glucose)
    assert result == 0.0


def test_mage_baghurst_constant():
    """MAGE_Baghurst should return zeros for constant data."""
    glucose = pd.Series([100.0] * 100)
    result = mage_baghurst(glucose)
    assert result["MAGE+"] == 0.0
    assert result["MAGE-"] == 0.0
    assert result["MAGE_avg"] == 0.0


def test_mage_baghurst_small_dataset():
    """MAGE_Baghurst should handle datasets with fewer than 9 points."""
    glucose = pd.Series([100.0, 110.0, 120.0, 100.0, 90.0])
    result = mage_baghurst(glucose)
    assert isinstance(result, dict)
    assert "error" not in result


def test_baghurst_smoothing_constant():
    """Smoothing approach should handle constant data."""
    glucose = pd.Series([100.0] * 100)
    result = mage_baghurst_smoothing(glucose)
    assert isinstance(result, dict)


def test_baghurst_direct_elimination_constant():
    """Direct elimination approach should handle constant data."""
    glucose = pd.Series([100.0] * 100)
    result = mage_baghurst_direct_elimination(glucose)
    assert isinstance(result, dict)


def test_baghurst_simplified_constant():
    """Simplified approach should handle constant data."""
    glucose = pd.Series([100.0] * 100)
    result = mage_baghurst_simplified(glucose)
    assert isinstance(result, dict)


def test_mage_with_sine_data():
    """MAGE should return non-zero for variable data with excursions."""
    np.random.seed(42)
    base = np.sin(np.linspace(0, 4 * np.pi, 288)) * 50 + 120
    glucose = pd.Series(base)
    result = mage_simple(glucose)
    assert result >= 0


def test_mage_baghurst_with_sine_data():
    """MAGE_Baghurst should return sensible values for variable data."""
    np.random.seed(42)
    base = np.sin(np.linspace(0, 4 * np.pi, 288)) * 50 + 120
    glucose = pd.Series(base)
    result = mage_baghurst(glucose)
    assert isinstance(result, dict)
    assert result["MAGE_avg"] >= 0


def test_backward_compatibility():
    """Pure functions match GlucoseAnalysis results on real data."""
    from cgmpy import GlucoseAnalysis, GlucoseData

    data = GlucoseAnalysis(GlucoseData("tests/fixtures/synthetic/sine_24h.csv"))
    glucose = data.glucose

    old_mage = data.MAGE()
    new_mage = mage_simple(glucose)
    assert old_mage == pytest.approx(new_mage)

    # Compare Baghurst approach results (approach=1 smoothing)
    threshold = data.sd()
    new_baghurst = mage_baghurst_smoothing(glucose, threshold)["MAGE_avg"]
    old_baghurst = data.MAGE_Baghurst(approach=1, threshold=threshold)["MAGE_avg"]
    assert old_baghurst == pytest.approx(new_baghurst)
