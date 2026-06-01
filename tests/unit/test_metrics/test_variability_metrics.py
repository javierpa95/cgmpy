import pytest
import pandas as pd
import numpy as np
from cgmpy import GlucoseMetrics

def test_sd_total(stable_glucose_df):
    """Test total standard deviation calculation."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    result = gm.sd_total()
    assert "sd" in result
    assert "mean" in result
    assert result["sd"] == pytest.approx(stable_glucose_df["glucose"].std())

def test_sd_within_day(stable_glucose_df):
    """Test within-day standard deviation calculation."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    result = gm.sd_within_day()
    assert "sd" in result
    assert result["sd"] == pytest.approx(gm.sdw())

def test_modd_calculation(stable_glucose_df):
    """Test MODD (Mean Of Daily Differences) calculation."""
    # Create two days of identical data to expect MODD ~ 0
    df1 = stable_glucose_df.copy()
    df2 = stable_glucose_df.copy()
    df2["time"] = df2["time"] + pd.Timedelta(days=1)
    df_combined = pd.concat([df1, df2])
    
    gm = GlucoseMetrics(data_source=df_combined)
    result = gm.MODD()
    assert "value" in result
    # With identical days, MODD should be very small
    assert result["value"] == pytest.approx(0.0, abs=1e-1)

def test_conga_calculation(stable_glucose_df):
    """Test CONGA calculation."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    result = gm.CONGA(hours=1)
    assert "value" in result
    assert isinstance(result["value"], (float, np.float64, type(None)))

def test_mage_calculation(variable_glucose_df):
    """Test MAGE calculation."""
    gm = GlucoseMetrics(data_source=variable_glucose_df)
    mage_val = gm.MAGE()
    assert isinstance(mage_val, (float, np.float64))
    assert mage_val >= 0

def test_variability_summary(stable_glucose_df):
    """Test the variability summary dictionary."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    summary = gm.variability_summary()
    assert "basic_variability" in summary
    assert "sd_total" in summary["basic_variability"]
    assert "cv" in summary["basic_variability"]
    assert "excursion_metrics" in summary

