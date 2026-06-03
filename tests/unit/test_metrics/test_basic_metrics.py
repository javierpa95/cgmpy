import pytest

from cgmpy import GlucoseAnalysis, GlucoseData


@pytest.fixture
def analysis(stable_glucose_df):
    return GlucoseAnalysis(GlucoseData(data_source=stable_glucose_df))


def test_basic_metrics_initialization(stable_glucose_df):
    """Test that GlucoseAnalysis correctly initializes and provides basic metrics."""
    ga = GlucoseAnalysis(GlucoseData(data_source=stable_glucose_df))
    assert not ga.data.empty
    assert ga.mean() == pytest.approx(100.0, abs=1.0)


def test_mean_calculation(analysis):
    """Test mean glucose calculation."""
    expected_mean = analysis.data["glucose"].mean()
    assert analysis.mean() == pytest.approx(expected_mean)


def test_median_calculation(analysis):
    """Test median glucose calculation."""
    expected_median = analysis.data["glucose"].median()
    assert analysis.median() == pytest.approx(expected_median)


def test_sd_calculation(analysis):
    """Test standard deviation calculation."""
    expected_sd = analysis.data["glucose"].std()
    assert analysis.sd() == pytest.approx(expected_sd)


def test_cv_calculation(variable_glucose_df):
    """Test coefficient of variation calculation."""
    ga = GlucoseAnalysis(GlucoseData(data_source=variable_glucose_df))
    mean = variable_glucose_df["glucose"].mean()
    sd = variable_glucose_df["glucose"].std()
    expected_cv = (sd / mean) * 100
    assert ga.cv() == pytest.approx(expected_cv)


def test_gmi_calculation(analysis):
    """Test GMI calculation."""
    mean = analysis.data["glucose"].mean()
    expected_gmi = round(3.31 + (0.02392 * mean), 2)
    assert analysis.gmi() == pytest.approx(expected_gmi)


def test_percentile_calculation(variable_glucose_df):
    """Test percentile calculations."""
    ga = GlucoseAnalysis(GlucoseData(data_source=variable_glucose_df))
    assert ga.percentile(50) == pytest.approx(ga.median())
    assert ga.percentile(25) < ga.percentile(75)


def test_distribution_analysis(analysis):
    """Test the complete distribution analysis dictionary."""
    da = analysis.distribution_analysis()
    assert "mean" in da
    assert "percentiles" in da
    assert "IQR" in da["percentiles"]
    assert da["mean"] == analysis.mean()


def test_calculate_all_metrics(analysis):
    """Test calculate_all_metrics method."""
    metrics = analysis.calculate_all_metrics()
    assert "GMI" in metrics
    assert "Mean" in metrics
    assert metrics["Mean"] == analysis.mean()
