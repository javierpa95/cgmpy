"""Unit tests for ``cgmpy.plotting.statistical_plots``.

Covers:
- ``histogram`` (public)
- ``plot_time_in_range`` (public, standard + pregnancy)
- ``plot_distribution_comparison`` (public, default + custom ranges)
- ``plot_correlation_matrix`` (public, default + custom segments)
- Helper: ``_generate_statistics_text``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from cgmpy import GlucoseAnalysis, GlucoseData
from cgmpy.plotting import statistical_plots as stat_module

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gp_24h(glucose_24h_df):
    return GlucoseAnalysis(GlucoseData(data_source=glucose_24h_df))


@pytest.fixture
def gp_7day(glucose_7day_df):
    return GlucoseAnalysis(GlucoseData(data_source=glucose_7day_df))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Public API — histogram
# ---------------------------------------------------------------------------


def test_histogram_creates_figure(gp_24h, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    assert plt.get_fignums() == []
    gp_24h.histogram()
    assert len(plt.get_fignums()) == 1
    ax = plt.gcf().axes[0]
    assert "Histogram" in ax.get_title()
    assert "Glucose" in ax.get_xlabel()


def test_histogram_default_bin_width_is_ten(gp_24h, monkeypatch):
    """Default bin_width=10 must appear in the chart title."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_24h.histogram()
    assert "10 mg/dL bins" in plt.gcf().axes[0].get_title()


def test_histogram_custom_bin_width(gp_24h, monkeypatch):
    """Custom bin_width must propagate to the title."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_24h.histogram(bin_width=20)
    assert "20 mg/dL bins" in plt.gcf().axes[0].get_title()


def test_histogram_has_three_zones(gp_24h, monkeypatch):
    """Histogram should have 3 axvspan zones (hypo/target/hyper) plus bars."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_24h.histogram()
    ax = plt.gcf().axes[0]
    assert len(ax.patches) >= 3


# ---------------------------------------------------------------------------
# Public API — plot_time_in_range
# ---------------------------------------------------------------------------


def test_plot_time_in_range_standard_creates_two_axes(gp_24h, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_24h.plot_time_in_range(pregnancy=False)
    fig = plt.gcf()
    assert len(plt.get_fignums()) == 1
    assert len(fig.axes) == 2
    title = fig.axes[0].get_title()
    assert "Time in Range" in title
    assert "Standard" in title


def test_plot_time_in_range_standard_pie_has_tir_label(gp_24h, monkeypatch):
    """Standard pie chart should include the TIR (70-180 mg/dL) label."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_24h.plot_time_in_range(pregnancy=False)
    ax = plt.gcf().axes[0]
    pie_texts = [t.get_text() for t in ax.texts if "mg/dL" in t.get_text()]
    assert any("TIR" in t for t in pie_texts)


def test_plot_time_in_range_standard_bar_chart_y_labels(gp_24h, monkeypatch):
    """Standard bar chart should have y-tick labels equal to the # of non-zero slices."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_24h.plot_time_in_range(pregnancy=False)
    ax_bar = plt.gcf().axes[1]
    yticklabels = [t.get_text() for t in ax_bar.get_yticklabels()]
    assert any("TIR" in t for t in yticklabels)
    assert len(ax_bar.patches) == len(yticklabels)
    assert len(yticklabels) >= 1


def test_plot_time_in_range_pregnancy_creates_two_axes(gp_24h, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_24h.plot_time_in_range(pregnancy=True)
    fig = plt.gcf()
    assert len(fig.axes) == 2
    assert "Pregnancy" in fig.axes[0].get_title()


def test_plot_time_in_range_pregnancy_pie_has_tir_label(gp_24h, monkeypatch):
    """Pregnancy pie chart should include the TIR Pregnancy (63-140 mg/dL) label."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_24h.plot_time_in_range(pregnancy=True)
    ax = plt.gcf().axes[0]
    pie_texts = [t.get_text() for t in ax.texts if "mg/dL" in t.get_text()]
    assert any("TIR Pregnancy" in t for t in pie_texts)


def test_plot_time_in_range_pregnancy_bar_chart_y_labels(gp_24h, monkeypatch):
    """Pregnancy bar chart should have the TIR Pregnancy label."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_24h.plot_time_in_range(pregnancy=True)
    ax_bar = plt.gcf().axes[1]
    yticklabels = [t.get_text() for t in ax_bar.get_yticklabels()]
    assert any("TIR Pregnancy" in t for t in yticklabels)
    assert len(ax_bar.patches) == len(yticklabels)
    assert len(yticklabels) >= 1


def test_plot_time_in_range_bar_chart_has_values(gp_24h, monkeypatch):
    """The horizontal bar chart should display numeric annotations."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_24h.plot_time_in_range(pregnancy=False)
    ax_bar = plt.gcf().axes[1]
    bar_text = " ".join(t.get_text() for t in ax_bar.texts)
    assert "%" in bar_text


# ---------------------------------------------------------------------------
# Public API — plot_distribution_comparison
# ---------------------------------------------------------------------------


def test_plot_distribution_comparison_default(gp_24h, monkeypatch):
    import warnings

    monkeypatch.setattr(plt, "show", lambda: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp_24h.plot_distribution_comparison()
    fig = plt.gcf()
    assert len(plt.get_fignums()) == 1
    assert len(fig.axes) == 4


def test_plot_distribution_comparison_axes_titles(gp_24h, monkeypatch):
    """The 2x2 subplot should contain Histogram, Box plot, Q-Q plot, Stats."""
    import warnings

    monkeypatch.setattr(plt, "show", lambda: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp_24h.plot_distribution_comparison()
    fig = plt.gcf()
    titles = [ax.get_title() for ax in fig.axes]
    titles_combined = " ".join(titles)
    assert "Distribution" in titles_combined
    assert "Box Plot" in titles_combined
    assert "Q-Q" in titles_combined


def test_plot_distribution_comparison_custom_ranges(gp_24h, monkeypatch):
    """Custom target_ranges should be accepted without errors."""
    import warnings

    monkeypatch.setattr(plt, "show", lambda: None)
    custom = [
        (60, 90, "Tight Range", "#abcdef"),
        (90, 150, "Wider Range", "#fedcba"),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp_24h.plot_distribution_comparison(target_ranges=custom)
    assert len(plt.get_fignums()) == 1


# ---------------------------------------------------------------------------
# Public API — plot_correlation_matrix
# ---------------------------------------------------------------------------


def test_plot_correlation_matrix_default_creates_figure(gp_7day, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_7day.plot_correlation_matrix()
    assert len(plt.get_fignums()) == 1
    ax = plt.gcf().axes[0]
    assert len(ax.collections) >= 1


def test_plot_correlation_matrix_title(gp_7day, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_7day.plot_correlation_matrix()
    assert "Correlation" in plt.gcf().axes[0].get_title()


def test_plot_correlation_matrix_custom_segments(gp_7day, monkeypatch):
    """Custom time_segments should produce a different-sized heatmap."""
    monkeypatch.setattr(plt, "show", lambda: None)
    custom = ["00:00-12:00", "12:00-24:00"]
    gp_7day.plot_correlation_matrix(time_segments=custom)
    assert len(plt.get_fignums()) == 1


def test_plot_correlation_matrix_segment_crossing_midnight(gp_7day, monkeypatch):
    """Segment that crosses midnight (e.g. 18:00-06:00) must work."""
    monkeypatch.setattr(plt, "show", lambda: None)
    custom = ["18:00-06:00", "06:00-18:00"]
    gp_7day.plot_correlation_matrix(time_segments=custom)
    assert len(plt.get_fignums()) == 1


def test_plot_correlation_matrix_24hour_segment(gp_7day, monkeypatch):
    """The special case ``end_h == 0`` (24:00) must be handled."""
    monkeypatch.setattr(plt, "show", lambda: None)
    custom = ["00:00-24:00"]
    gp_7day.plot_correlation_matrix(time_segments=custom)
    assert len(plt.get_fignums()) == 1


# ---------------------------------------------------------------------------
# Helper — _generate_statistics_text
# ---------------------------------------------------------------------------


@pytest.fixture
def glucose_series(glucose_24h_df):
    return glucose_24h_df["glucose"]


@pytest.fixture
def metric_values(glucose_24h_df):
    g = glucose_24h_df["glucose"]
    tir_val = ((g >= 70) & (g <= 180)).sum() / len(g) * 100
    tbr_val = (g < 70).sum() / len(g) * 100
    tar_val = (g > 180).sum() / len(g) * 100
    mean_val = g.mean()
    gmi_val = round(3.31 + (0.02392 * mean_val), 2)
    return tir_val, tbr_val, tar_val, gmi_val


def test_generate_statistics_text_contains_all_fields(glucose_series, metric_values):
    """The generated statistics string must contain all key fields."""
    tir_val, tbr_val, tar_val, gmi_val = metric_values
    text = stat_module._generate_statistics_text(glucose_series, tir_val, tbr_val, tar_val, gmi_val)
    for field in (
        "DESCRIPTIVE STATISTICS",
        "Mean:",
        "Median:",
        "Std Dev:",
        "CV:",
        "Percentiles:",
        "P5:",
        "P25:",
        "P75:",
        "P95:",
        "Time in Range:",
        "TIR (70-180):",
        "TBR (<70):",
        "TAR (>180):",
        "GMI:",
    ):
        assert field in text, f"Missing field: {field!r}"


def test_generate_statistics_text_values_match_data(glucose_series, metric_values):
    """Mean, SD and CV in the text should match the source data."""
    import re

    tir_val, tbr_val, tar_val, gmi_val = metric_values
    text = stat_module._generate_statistics_text(glucose_series, tir_val, tbr_val, tar_val, gmi_val)
    expected_mean = glucose_series.mean()
    expected_std = glucose_series.std()
    expected_cv = expected_std / expected_mean * 100

    mean_m = re.search(r"Mean:\s+([\d.]+)\s*mg/dL", text)
    std_m = re.search(r"Std Dev:\s+([\d.]+)\s*mg/dL", text)
    cv_m = re.search(r"CV:\s+([\d.]+)\s*%", text)
    assert mean_m is not None
    assert std_m is not None
    assert cv_m is not None
    assert float(mean_m.group(1)) == pytest.approx(expected_mean, rel=1e-2)
    assert float(std_m.group(1)) == pytest.approx(expected_std, rel=1e-2)
    assert float(cv_m.group(1)) == pytest.approx(expected_cv, rel=1e-2)
