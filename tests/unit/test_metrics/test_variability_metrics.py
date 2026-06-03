import numpy as np
import pandas as pd
import pytest

from cgmpy import GlucoseAnalysis, GlucoseData


@pytest.fixture
def analysis(stable_glucose_df):
    return GlucoseAnalysis(GlucoseData(data_source=stable_glucose_df))


def test_sd_total(stable_glucose_df):
    """Test total standard deviation calculation."""
    ga = GlucoseAnalysis(GlucoseData(data_source=stable_glucose_df))
    result = ga.sd_total()
    assert "sd" in result
    assert "mean" in result
    assert result["sd"] == pytest.approx(stable_glucose_df["glucose"].std())


def test_sd_within_day(analysis):
    """Test within-day standard deviation calculation."""
    result = analysis.sd_within_day()
    assert "sd" in result
    assert result["sd"] == pytest.approx(analysis.sdw())


def test_modd_calculation(variable_glucose_df):
    """Test MODD (Mean Of Daily Differences) calculation with variable glucose."""
    # Two days of variable glucose shifted by 1 day.
    df1 = variable_glucose_df.copy()
    df2 = variable_glucose_df.copy()
    df2["time"] = df2["time"] + pd.Timedelta(days=1)
    df_combined = pd.concat([df1, df2])

    ga = GlucoseAnalysis(GlucoseData(data_source=df_combined))
    result = ga.MODD()
    assert "value" in result
    # With two days of identical-but-shifted variable glucose, MODD is small
    # but finite. Constant glucose would produce nan, so we use variable data.
    assert result["value"] is not None
    assert result["value"] >= 0.0


def test_conga_calculation(analysis):
    """Test CONGA calculation."""
    result = analysis.CONGA(hours=1)
    assert "value" in result
    assert isinstance(result["value"], float | np.float64 | type(None))


def test_mage_calculation(variable_glucose_df):
    """Test MAGE calculation."""
    ga = GlucoseAnalysis(GlucoseData(data_source=variable_glucose_df))
    mage_val = ga.MAGE()
    assert isinstance(mage_val, float | np.float64)
    assert mage_val >= 0


def test_variability_summary(analysis):
    """Test the variability summary via GlucoseAnalysis."""
    result = analysis.calculate_variability_metrics()
    assert isinstance(result, dict)
    # Should contain basic variability keys
    assert "Std" in result
    assert "CV" in result
    # Should contain excursion metrics
    assert "mage_avg" in result
