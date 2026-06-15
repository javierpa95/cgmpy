"""Tests for MAGE variability metrics via the GlucoseAnalysis facade."""

import numpy as np
import pandas as pd
import pytest

from cgmpy import GlucoseAnalysis, GlucoseData


@pytest.fixture
def analysis(variable_glucose_df):
    return GlucoseAnalysis(GlucoseData(data_source=variable_glucose_df))


def test_mage_simple(analysis):
    """MAGE() should return a non-negative float on variable data."""
    val = analysis.MAGE()
    assert isinstance(val, float)
    assert val >= 0.0
    assert np.isfinite(val)


def test_mage_simple_constant_returns_zero(stable_glucose_df):
    """Constant glucose has no peaks/nadirs, so MAGE() should be 0."""
    ga = GlucoseAnalysis(GlucoseData(data_source=stable_glucose_df))
    val = ga.MAGE()
    assert val == 0.0


def test_mage_baghurst_approach_1(analysis):
    """MAGE_Baghurst(approach=1) should return the documented dictionary."""
    result = analysis.MAGE_Baghurst(threshold_sd=1, approach=1, plot=False)
    expected_keys = {
        "MAGE+",
        "MAGE-",
        "MAGE_avg",
        "SD_used",
        "threshold",
        "num_excursions",
        "turning_points",
    }
    assert set(result.keys()) == expected_keys
    # SD_used should equal analysis.sd() (within rounding).
    assert result["SD_used"] == pytest.approx(analysis.sd(), abs=0.5)
    # threshold should equal SD_used (threshold_sd=1).
    assert result["threshold"] == pytest.approx(result["SD_used"], abs=0.5)


def test_mage_baghurst_approach_2_no_indexerror(analysis):
    """MAGE_Baghurst(approach=2) must not raise IndexError on real CGM data.

    Bug fixed in v0.5.1: the implementation indexed
    ``glucose[turning_points[0]]`` without checking that ``turning_points``
    was non-empty, which could happen for monotonically-tending data.
    The function now returns a well-formed dict in every case.
    """
    result = analysis.MAGE_Baghurst(threshold_sd=1, approach=2, plot=False)
    assert isinstance(result, dict)
    assert "MAGE_avg" in result
    assert "num_excursions" in result


def test_mage_baghurst_handles_tiny_dataset():
    """Bug fixed in v0.5.1: MAGE_Baghurst raised IndexError / ValueError
    on datasets smaller than the smoothing window (9 points)."""
    from datetime import datetime, timedelta

    df = pd.DataFrame(
        {
            "time": [datetime(2024, 1, 1) + timedelta(minutes=5 * i) for i in range(4)],
            "glucose": [100, 100, 100, 100],
        }
    )
    ga = GlucoseAnalysis(GlucoseData(data_source=df))
    result = ga.MAGE_Baghurst(threshold_sd=1, approach=1, plot=False)
    assert isinstance(result, dict)
    assert result["num_excursions"] == 0
    assert result["MAGE_avg"] == 0.0


def test_mage_baghurst_handles_constant_data():
    """Bug fixed in v0.5.1: with SD=0 (constant glucose) the threshold
    is 0 and the algorithm would register a fake excursion of magnitude 0.
    The function now short-circuits to a zeroed result."""
    from datetime import datetime, timedelta

    df = pd.DataFrame(
        {
            "time": [datetime(2024, 1, 1) + timedelta(minutes=5 * i) for i in range(288)],
            "glucose": [120] * 288,
        }
    )
    ga = GlucoseAnalysis(GlucoseData(data_source=df))
    result = ga.MAGE_Baghurst(threshold_sd=1, approach=2, plot=False)
    assert result["num_excursions"] == 0
    assert result["MAGE_avg"] == 0.0


def test_mage_baghurst_approach_3(analysis):
    """MAGE_Baghurst(approach=3) should also return the documented dictionary."""
    result = analysis.MAGE_Baghurst(threshold_sd=1, approach=3, plot=False)
    assert "MAGE+" in result and "MAGE-" in result
    assert "MAGE_avg" in result
    assert isinstance(result["num_excursions"], int)
