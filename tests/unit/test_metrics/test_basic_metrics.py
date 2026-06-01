import pytest
import pandas as pd
import numpy as np
from cgmpy import GlucoseMetrics

def test_basic_metrics_initialization(stable_glucose_df):
    """Test that GlucoseMetrics correctly initializes and provides basic metrics."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    assert not gm.data.empty
    assert gm.mean() == pytest.approx(100.0, abs=1.0)

def test_mean_calculation(stable_glucose_df):
    """Test mean glucose calculation."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    expected_mean = stable_glucose_df["glucose"].mean()
    assert gm.mean() == pytest.approx(expected_mean)

def test_median_calculation(stable_glucose_df):
    """Test median glucose calculation."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    expected_median = stable_glucose_df["glucose"].median()
    assert gm.median() == pytest.approx(expected_median)

def test_sd_calculation(stable_glucose_df):
    """Test standard deviation calculation."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    expected_sd = stable_glucose_df["glucose"].std()
    assert gm.sd() == pytest.approx(expected_sd)

def test_cv_calculation(variable_glucose_df):
    """Test coefficient of variation calculation."""
    gm = GlucoseMetrics(data_source=variable_glucose_df)
    mean = variable_glucose_df["glucose"].mean()
    sd = variable_glucose_df["glucose"].std()
    expected_cv = (sd / mean) * 100
    assert gm.cv() == pytest.approx(expected_cv)

def test_gmi_calculation(stable_glucose_df):
    """Test GMI calculation."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    mean = stable_glucose_df["glucose"].mean()
    expected_gmi = round(3.31 + (0.02392 * mean), 2)
    assert gm.gmi() == pytest.approx(expected_gmi)

def test_percentile_calculation(variable_glucose_df):
    """Test percentile calculations."""
    gm = GlucoseMetrics(data_source=variable_glucose_df)
    assert gm.percentile(50) == pytest.approx(gm.median())
    assert gm.percentile(25) < gm.percentile(75)

def test_distribution_analysis(stable_glucose_df):
    """Test the complete distribution analysis dictionary."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    analysis = gm.distribution_analysis()
    assert "mean" in analysis
    assert "percentiles" in analysis
    assert "IQR" in analysis["percentiles"]
    assert analysis["mean"] == gm.mean()

def test_calculate_all_metrics(stable_glucose_df):
    """Test calculate_all_metrics method."""
    gm = GlucoseMetrics(data_source=stable_glucose_df)
    metrics = gm.calculate_all_metrics()
    assert "GMI" in metrics
    assert "Mean" in metrics
    assert metrics["Mean"] == gm.mean()
