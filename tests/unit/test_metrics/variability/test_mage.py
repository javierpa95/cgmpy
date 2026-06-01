"""Tests for cgmpy.metrics.variability.mage.MAGEMetrics."""

import numpy as np
import pandas as pd
import pytest

from cgmpy import GlucoseMetrics


def test_mage_simple(variable_glucose_df):
    """MAGE() should return a non-negative float on variable data."""
    gm = GlucoseMetrics(data_source=variable_glucose_df)
    val = gm.MAGE()
    assert isinstance(val, float)
    assert val >= 0.0
    assert np.isfinite(val)


def test_mage_simple_constant_returns_zero(stable_glucose_df):
    """Constant glucose has no peaks/nadirs, so MAGE() should be 0."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    val = gm.MAGE()
    assert val == 0.0


def test_mage_baghurst_approach_1(variable_glucose_df):
    """MAGE_Baghurst(approach=1) should return the documented dictionary."""
    gm = GlucoseMetrics(data_source=variable_glucose_df)
    result = gm.MAGE_Baghurst(threshold_sd=1, approach=1, plot=False)
    expected_keys = {
        "MAGE+", "MAGE-", "MAGE_avg", "SD_used", "threshold", "num_excursions",
    }
    assert set(result.keys()) == expected_keys
    # SD_used should equal gm.sd() (within rounding).
    assert result["SD_used"] == pytest.approx(gm.sd(), abs=0.5)
    # threshold should equal SD_used (threshold_sd=1).
    assert result["threshold"] == pytest.approx(result["SD_used"], abs=0.5)


def test_mage_baghurst_approach_2_known_bug(variable_glucose_df):
    """MAGE_Baghurst(approach=2) crashes with ``IndexError`` on many inputs.

    The implementation indexes ``glucose[turning_points[0]]`` without
    checking that ``turning_points`` is non-empty, which can happen for
    monotonically-tending data. The test documents the bug rather than
    asserting on a numeric result so the regression is detected when fixed.
    """
    gm = GlucoseMetrics(data_source=variable_glucose_df)
    with pytest.raises(IndexError):
        gm.MAGE_Baghurst(threshold_sd=1, approach=2, plot=False)


def test_mage_baghurst_approach_3(variable_glucose_df):
    """MAGE_Baghurst(approach=3) should also return the documented dictionary."""
    gm = GlucoseMetrics(data_source=variable_glucose_df)
    result = gm.MAGE_Baghurst(threshold_sd=1, approach=3, plot=False)
    assert "MAGE+" in result and "MAGE-" in result
    assert "MAGE_avg" in result
    assert isinstance(result["num_excursions"], int)
