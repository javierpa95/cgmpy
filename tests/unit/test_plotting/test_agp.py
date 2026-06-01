"""Unit tests for ``cgmpy.plotting.agp``.

Covers:
- ``plot_agp`` (public)
- ``generate_week_agp`` (combined / separate branches)
- Helpers: ``_add_glucose_zones``, ``_configure_agp_plot``,
  ``_plot_percentiles``, ``_calculate_day_percentiles``,
  ``_calculate_full_day_percentiles``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cgmpy import GlucosePlot
from cgmpy.plotting.agp import AGPPlotter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gp_24h(glucose_24h_df) -> GlucosePlot:
    """Single-day GlucosePlot instance."""
    return GlucosePlot(data_source=glucose_24h_df)


@pytest.fixture
def gp_7day(glucose_7day_df) -> GlucosePlot:
    """7-day GlucosePlot instance spanning every weekday."""
    return GlucosePlot(data_source=glucose_7day_df)


@pytest.fixture
def agp_mixin(gp_24h) -> AGPPlotter:
    """Expose the AGPPlotter mixin for helper-level unit tests."""
    return gp_24h


@pytest.fixture(autouse=True)
def _close_figures():
    """Ensure no figures leak between tests."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def test_plot_agp_creates_figure(gp_24h, monkeypatch):
    """``plot_agp`` should produce a matplotlib Figure."""
    monkeypatch.setattr(plt, "show", lambda: None)
    assert plt.get_fignums() == []
    gp_24h.plot_agp()
    assert len(plt.get_fignums()) == 1


def test_plot_agp_with_custom_smoothing(gp_24h, monkeypatch):
    """Custom smoothing_window must be accepted without error."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_24h.plot_agp(smoothing_window=5)
    assert len(plt.get_fignums()) == 1


def test_plot_agp_axes_have_zones_and_lines(gp_24h, monkeypatch):
    """The created axes must contain 3 axhspan zones + 2 reference lines."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_24h.plot_agp()
    fig = plt.gcf()
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    # 3 zones (axhspan adds 2 Patches per call) + 2 reference lines
    assert len(ax.patches) >= 3
    assert len(ax.lines) >= 2


def test_plot_agp_configures_titles_and_legend(gp_24h, monkeypatch):
    """Title, x-label, y-label, and legend must be set."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_24h.plot_agp()
    ax = plt.gcf().axes[0]
    assert ax.get_title() == "Ambulatory Glucose Profile (AGP)"
    assert ax.get_xlabel() == "Time of Day"
    assert ax.get_ylabel() == "Glucose Level (mg/dL)"
    assert ax.get_legend() is not None


def test_generate_week_agp_combined_creates_figure(gp_7day, monkeypatch):
    """``generate_week_agp(combined=True)`` should produce exactly one Figure."""
    monkeypatch.setattr(plt, "show", lambda: None)
    assert plt.get_fignums() == []
    gp_7day.generate_week_agp(combined=True)
    assert len(plt.get_fignums()) == 1
    ax = plt.gcf().axes[0]
    assert "Day of Week" in ax.get_title()


def test_generate_week_agp_combined_has_seven_day_lines(gp_7day, monkeypatch):
    """Each of the 7 weekdays should appear in the legend (n=1 day each)."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_7day.generate_week_agp(combined=True)
    ax = plt.gcf().axes[0]
    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    for day in (
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ):
        assert any(day in lbl for lbl in labels), f"Missing {day} in {labels}"


def test_generate_week_agp_separate_creates_seven_subplots(gp_7day, monkeypatch):
    """``generate_week_agp(combined=False)`` should yield 7 subplots + a suptitle."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp_7day.generate_week_agp(combined=False)
    fig = plt.gcf()
    assert len(fig.axes) == 7
    assert fig._suptitle is not None
    assert "Day of Week" in fig._suptitle.get_text()


def test_generate_week_agp_combined_with_two_days(glucose_2day_df, monkeypatch):
    """Combined week plot with only 2 unique days should still succeed."""
    monkeypatch.setattr(plt, "show", lambda: None)
    gp = GlucosePlot(data_source=glucose_2day_df)
    gp.generate_week_agp(combined=True)
    assert len(plt.get_fignums()) == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_add_glucose_zones_helper(agp_mixin):
    """``_add_glucose_zones`` should add 3 spans + 2 horizontal lines."""
    fig, ax = plt.subplots()
    agp_mixin._add_glucose_zones(ax)
    # Each axhspan adds 1 Patch; 3 zones expected.
    assert len(ax.patches) == 3
    # Two reference lines (70 and 180).
    assert len(ax.lines) == 2


def test_configure_agp_plot_helper(agp_mixin):
    """``_configure_agp_plot`` must apply title, labels, ticks, grid, ylim."""
    fig, ax = plt.subplots()
    # The legend call inside _configure_agp_plot requires labelled artists.
    ax.plot([0, 24], [100, 100], label="dummy")
    agp_mixin._configure_agp_plot(ax, "Custom Title")
    assert ax.get_title() == "Custom Title"
    assert ax.get_xlabel() == "Time of Day"
    assert ax.get_ylabel() == "Glucose Level (mg/dL)"
    assert ax.get_ylim() == (0, 400)
    # 9 ticks every 3h from 0..24 inclusive
    xticks = ax.get_xticks()
    assert list(xticks) == [0, 3, 6, 9, 12, 15, 18, 21, 24]
    assert ax.get_legend() is not None
    assert ax.grid is not None


def test_plot_percentiles_helper(agp_mixin):
    """``_plot_percentiles`` should add 1 line + 2 fill_between collections."""
    # Build a tiny synthetic percentile DataFrame.
    idx = np.arange(0, 24, 0.5)
    percentiles = pd.DataFrame(
        {
            0.05: 80 + np.random.default_rng(0).normal(0, 1, len(idx)),
            0.25: 95 + np.random.default_rng(1).normal(0, 1, len(idx)),
            0.5: 110 + np.random.default_rng(2).normal(0, 1, len(idx)),
            0.75: 125 + np.random.default_rng(3).normal(0, 1, len(idx)),
            0.95: 150 + np.random.default_rng(4).normal(0, 1, len(idx)),
        },
        index=idx,
    )
    fig, ax = plt.subplots()
    agp_mixin._plot_percentiles(ax, percentiles)
    # 1 line for median
    assert len(ax.lines) == 1
    # 2 fill_between collections (IQR + 5-95%)
    assert len(ax.collections) == 2


def test_calculate_day_percentiles_returns_three_columns(agp_mixin):
    """``_calculate_day_percentiles`` returns DataFrame with columns 0.25/0.5/0.75."""
    df = pd.DataFrame(
        {
            "time_decimal": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "glucose": [100, 110, 120, 105, 115, 125],
        }
    )
    out = agp_mixin._calculate_day_percentiles(df, smoothing_window=3)
    assert list(out.columns) == [0.25, 0.5, 0.75]
    assert len(out) == 3


def test_calculate_full_day_percentiles_returns_five_columns(agp_mixin):
    """``_calculate_full_day_percentiles`` returns columns 0.05..0.95."""
    df = pd.DataFrame(
        {
            "time_decimal": [0.0, 0.5, 1.0] * 3,
            "glucose": [100, 110, 120, 105, 115, 125, 95, 108, 122],
        }
    )
    out = agp_mixin._calculate_full_day_percentiles(df, smoothing_window=3)
    assert list(out.columns) == [0.05, 0.25, 0.5, 0.75, 0.95]
    assert len(out) == 3


def test_calculate_day_percentiles_smoothing_window_one(agp_mixin):
    """smoothing_window=1 must still work and not raise."""
    df = pd.DataFrame(
        {
            "time_decimal": [0.0, 0.25, 0.5, 0.75, 1.0],
            "glucose": [100, 105, 110, 115, 120],
        }
    )
    out = agp_mixin._calculate_day_percentiles(df, smoothing_window=1)
    assert out.loc[0.0, 0.5] == pytest.approx(100.0)
