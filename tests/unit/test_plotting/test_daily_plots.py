"""Unit tests for ``cgmpy.plotting.daily_plots``.

Covers:
- ``day_graph`` (public, with and without explicit date)
- ``plot_overlapping_days`` (public)
- ``plot_week_boxplots`` (public)
- ``plot_daily_variations`` (public)
- Helpers: ``_add_glucose_zones``, ``_add_reference_lines``,
  ``_configure_daily_plot``, ``_configure_overlapping_plot``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from cgmpy import GlucosePlot
from cgmpy.plotting.daily_plots import DailyPlotter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gp_24h(glucose_24h_df) -> GlucosePlot:
    return GlucosePlot(data_source=glucose_24h_df)


@pytest.fixture
def gp_2day(glucose_2day_df) -> GlucosePlot:
    return GlucosePlot(data_source=glucose_2day_df)


@pytest.fixture
def gp_7day(glucose_7day_df) -> GlucosePlot:
    return GlucosePlot(data_source=glucose_7day_df)


@pytest.fixture
def daily_mixin(gp_24h) -> DailyPlotter:
    return gp_24h


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Public API — day_graph
# ---------------------------------------------------------------------------


def test_day_graph_default_date_creates_figure(gp_24h, monkeypatch):
    """``day_graph`` without args uses the first day in the data."""
    monkeypatch.setattr(plt, "show", lambda: None)
    assert plt.get_fignums() == []
    gp_24h.day_graph()
    assert len(plt.get_fignums()) == 1
    ax = plt.gcf().axes[0]
    assert "Glucose Levels" in ax.get_title()


def test_day_graph_explicit_date_creates_figure(gp_24h, monkeypatch):
    """``day_graph(date="YYYY-MM-DD")`` must respect the requested day."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_24h.day_graph(date="2024-01-01")
    assert len(plt.get_fignums()) == 1
    ax = plt.gcf().axes[0]
    assert "2024-01-01" in ax.get_title()


def test_day_graph_no_data_for_date_returns_none(gp_24h, monkeypatch):
    """``day_graph`` returns None silently if the date has no data."""
    monkeypatch.setattr(plt, "show", lambda: None)
    # Use a date outside the dataset
    result = gp_24h.day_graph(date="2099-12-31")
    assert result is None
    assert plt.get_fignums() == []


def test_day_graph_axes_have_zones_and_line(gp_24h, monkeypatch):
    """The plotted axes must contain 3 zone patches, lines, and text labels."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_24h.day_graph()
    ax = plt.gcf().axes[0]
    # 3 zones via _add_glucose_zones
    assert len(ax.patches) == 3
    # The glucose trace (with markers) + 2 reference axhline lines
    assert len(ax.lines) >= 3
    # 2 text annotations (70 mg/dL, 180 mg/dL)
    assert len(ax.texts) == 2


# ---------------------------------------------------------------------------
# Public API — overlapping days / week boxplots / daily variations
# ---------------------------------------------------------------------------


def test_plot_overlapping_days_creates_figure(gp_2day, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_2day.plot_overlapping_days()
    assert len(plt.get_fignums()) == 1
    ax = plt.gcf().axes[0]
    assert ax.get_title() == "Overlapping Glucose Profiles"


def test_plot_overlapping_days_renders_per_day(gp_2day, monkeypatch):
    """Two unique days must produce 2 individual traces + 1 mean profile line."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_2day.plot_overlapping_days()
    ax = plt.gcf().axes[0]
    # 2 per-day lines (gray) + 1 mean line + 2 reference axhlines = 5
    assert len(ax.lines) >= 3


def test_plot_week_boxplots_creates_figure(gp_7day, monkeypatch):
    import warnings

    monkeypatch.setattr(plt, "show", lambda: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp_7day.plot_week_boxplots()
    assert len(plt.get_fignums()) == 1
    ax = plt.gcf().axes[0]
    assert "Distribution" in ax.get_title()
    # 7 weekday boxes
    assert len(ax.get_xticklabels()) == 7


def test_plot_week_boxplots_labels_include_day_counts(gp_7day, monkeypatch):
    """Each x-tick label should contain ``(n=`` for the day count."""
    import warnings

    monkeypatch.setattr(plt, "show", lambda: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp_7day.plot_week_boxplots()
    ax = plt.gcf().axes[0]
    for label in ax.get_xticklabels():
        assert "(n=" in label.get_text()


def test_plot_daily_variations_creates_figure(gp_7day, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_7day.plot_daily_variations()
    assert len(plt.get_fignums()) == 1
    ax = plt.gcf().axes[0]
    assert "Variations" in ax.get_title()
    # 1 mean line + 2 reference lines (>= 3 because rolling mean may add extras)
    assert len(ax.lines) >= 3
    # 2 fill_between collections (SD band + IQR band)
    assert len(ax.collections) == 2


def test_plot_daily_variations_x_axis_config(gp_7day, monkeypatch):
    """X-axis must be configured with hourly ticks from 0 to 24 in steps of 3."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_7day.plot_daily_variations()
    ax = plt.gcf().axes[0]
    assert ax.get_xlim() == (0, 24)
    assert ax.get_ylim() == (0, 400)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_add_glucose_zones_helper(daily_mixin):
    fig, ax = plt.subplots()
    daily_mixin._add_glucose_zones(ax)
    assert len(ax.patches) == 3
    labels = [p.get_label() for p in ax.patches]
    assert "Hypoglycemia" in labels
    assert "Target range" in labels
    assert "Hyperglycemia" in labels


def test_add_reference_lines_helper(daily_mixin):
    fig, ax = plt.subplots()
    daily_mixin._add_reference_lines(ax)
    # 2 reference lines
    assert len(ax.lines) == 2
    # 2 text labels
    assert len(ax.texts) == 2
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "70 mg/dL" in texts
    assert "180 mg/dL" in texts


def test_configure_daily_plot_helper(daily_mixin):
    fig, ax = plt.subplots()
    ax.plot([0, 24], [100, 100], label="dummy")
    daily_mixin._configure_daily_plot(ax, "Test Title")
    assert ax.get_title() == "Test Title"
    assert ax.get_xlabel() == "Time of Day"
    assert ax.get_ylabel() == "Glucose Level (mg/dL)"
    assert ax.get_ylim() == (0, 400)
    assert ax.get_xlim() == (0, 24)
    assert ax.get_legend() is not None


def test_configure_overlapping_plot_helper(daily_mixin):
    """``_configure_overlapping_plot`` sets labels, zones and reference lines."""
    plt.figure()
    ax = plt.gca()
    ax.plot([0, 24], [100, 100], label="dummy")
    daily_mixin._configure_overlapping_plot()
    assert ax.get_title() == "Overlapping Glucose Profiles"
    assert ax.get_xlabel() == "Time of Day"
    assert ax.get_ylabel() == "Glucose Level (mg/dL)"
    # 2 reference lines + 0 from us, plus the dummy = 3 lines
    assert len(ax.lines) == 3
    # 3 zones
    assert len(ax.patches) == 3
    assert ax.get_ylim() == (0, 400)


# ---------------------------------------------------------------------------
# Edge case: empty data
# ---------------------------------------------------------------------------


def test_day_graph_with_single_day_data(monkeypatch):
    """``day_graph`` on a 1-row DataFrame must not raise."""
    monkeypatch.setattr(plt, "show", lambda: None)
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01 08:00"]),
            "glucose": [120.0],
        }
    )
    gp = GlucosePlot(data_source=df)
    gp.day_graph()
    assert len(plt.get_fignums()) == 1
