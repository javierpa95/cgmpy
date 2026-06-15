"""
Module for daily glucose data plots.

This module contains functions to generate charts related to daily patterns:
- Specific day plots
- Overlapping multiple days
- Boxplots by day of week
- Daily variation analysis

Every public function accepts an optional ``ax`` and returns the resulting
``Figure`` without calling ``plt.show()`` — display is left to the caller
(e.g. the :class:`~cgmpy.analysis.core.GlucoseAnalysis` facade), which makes
the plots composable inside dashboards.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._constants import (
    GLUCOSE_AXIS_MAX,
    GLUCOSE_AXIS_MIN,
    TARGET_HIGH,
    TARGET_LOW,
)
from ._utils import add_glucose_zones, resolve_axes

logger = logging.getLogger(__name__)


def day_graph(data: pd.DataFrame, date: str | None = None, ax: Axes | None = None) -> Figure | None:
    """Builds the glucose chart for a specific day.

    Args:
        data: Glucose DataFrame with ``time`` and ``glucose`` columns.
        date: Optional date in 'YYYY-MM-DD' format.
              If not provided, the first day of the DataFrame is used.
        ax: Optional axis to draw into. If ``None`` a new figure is created.

    Returns:
        The matplotlib ``Figure``, or ``None`` if the date has no data.
    """
    if date is None:
        date = data["time"].dt.date.min()
    else:
        date = pd.to_datetime(date).date()

    day_data = data[data["time"].dt.date == date].copy()

    if day_data.empty:
        logger.info("No data for date %s", date)
        return None

    day_data["hours"] = day_data["time"].dt.hour + day_data["time"].dt.minute / 60.0

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    fig, ax, created = resolve_axes(ax, figsize=(16, 9))

    add_glucose_zones(ax)

    ax.plot(
        day_data["hours"],
        day_data["glucose"],
        label="Glucose",
        color="#3366CC",
        linewidth=2,
        marker="o",
        markersize=4,
    )

    _add_reference_lines(ax)

    _configure_daily_plot(ax, f"Glucose Levels - {date}")

    if created:
        fig.tight_layout()
    return fig


def plot_overlapping_days(data: pd.DataFrame, ax: Axes | None = None) -> Figure:
    """Builds a chart with the glucose profiles of multiple overlapping days.

    Args:
        data: Glucose DataFrame with ``time`` and ``glucose`` columns.
        ax: Optional axis to draw into. If ``None`` a new figure is created.

    Returns:
        The matplotlib ``Figure`` containing the plot.
    """
    data_copy = data.copy()
    data_copy["time_decimal"] = data_copy["time"].dt.hour + data_copy["time"].dt.minute / 60.0
    data_copy["date"] = data_copy["time"].dt.date

    fig, ax, created = resolve_axes(ax, figsize=(12, 8))

    mean_profile = (
        data_copy.groupby("time_decimal")["glucose"]
        .mean()
        .rolling(window=15, center=True, min_periods=1)
        .mean()
    )

    dates = data_copy["date"].unique()
    for d in dates:
        day_data = data_copy[data_copy["date"] == d]
        ax.plot(
            day_data["time_decimal"],
            day_data["glucose"],
            color="gray",
            alpha=0.2,
            linewidth=1,
        )

    ax.plot(
        mean_profile.index,
        mean_profile.values,
        color="black",
        linewidth=2,
        label="Mean profile",
    )

    _configure_overlapping_plot(ax)

    if created:
        fig.tight_layout()
    return fig


def plot_week_boxplots(data: pd.DataFrame, ax: Axes | None = None) -> Figure:
    """Builds a boxplot chart of the glucose distribution by day of the week.

    Args:
        data: Glucose DataFrame with ``time`` and ``glucose`` columns.
        ax: Optional axis to draw into. If ``None`` a new figure is created.

    Returns:
        The matplotlib ``Figure`` containing the plot.
    """
    data_copy = data.copy()
    data_copy["weekday"] = data_copy["time"].dt.day_name()
    data_copy["date"] = data_copy["time"].dt.date

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    unique_days = data_copy.groupby("weekday")["date"].nunique()

    labels = [f"{day}\n(n={unique_days.get(day, 0)} days)" for day in day_order]

    fig, ax, created = resolve_axes(ax, figsize=(12, 8))

    ax.axhspan(GLUCOSE_AXIS_MIN, TARGET_LOW, color="#ffcccb", alpha=0.2, label="Hypoglycemia")
    ax.axhspan(TARGET_LOW, TARGET_HIGH, color="#90ee90", alpha=0.2, label="Target range")
    ax.axhspan(TARGET_HIGH, GLUCOSE_AXIS_MAX, color="#ffcccb", alpha=0.2, label="Hyperglycemia")

    sns.boxplot(
        x="weekday",
        y="glucose",
        data=data_copy,
        order=day_order,
        whis=1.5,
        medianprops={"color": "red", "linewidth": 1.5},
        flierprops={"marker": "o", "markerfacecolor": "gray", "markersize": 4},
        ax=ax,
    )

    ax.axhline(y=TARGET_LOW, color="red", linestyle="--", linewidth=1)
    ax.axhline(y=TARGET_HIGH, color="red", linestyle="--", linewidth=1)

    ax.set_title("Glucose Distribution by Day of Week", fontsize=14, pad=20)
    ax.set_xlabel("Day of Week", fontsize=12)
    ax.set_ylabel("Glucose Level (mg/dL)", fontsize=12)

    ax.set_xticks(range(len(day_order)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(GLUCOSE_AXIS_MIN, GLUCOSE_AXIS_MAX)

    ax.legend(title="Ranges", bbox_to_anchor=(1.05, 1), loc="upper left")

    if created:
        fig.tight_layout()
    return fig


def plot_daily_variations(data: pd.DataFrame, ax: Axes | None = None) -> Figure:
    """Builds a chart of the average daily variations with confidence bands.

    Args:
        data: Glucose DataFrame with ``time`` and ``glucose`` columns.
        ax: Optional axis to draw into. If ``None`` a new figure is created.

    Returns:
        The matplotlib ``Figure`` containing the plot.
    """
    data_copy = data.copy()
    data_copy["time_decimal"] = data_copy["time"].dt.hour + data_copy["time"].dt.minute / 60.0

    hourly_stats = (
        data_copy.groupby("time_decimal")["glucose"]
        .agg(
            [
                "mean",
                "std",
                "count",
                lambda x: np.percentile(x, 25),
                lambda x: np.percentile(x, 75),
            ]
        )
        .reset_index()
    )

    hourly_stats.columns = ["time_decimal", "mean", "std", "count", "p25", "p75"]

    window_size = 15
    for col in ["mean", "std", "p25", "p75"]:
        hourly_stats[col] = (
            hourly_stats[col].rolling(window=window_size, center=True, min_periods=1).mean()
        )

    fig, ax, created = resolve_axes(ax, figsize=(14, 8))

    add_glucose_zones(ax)

    ax.plot(
        hourly_stats["time_decimal"],
        hourly_stats["mean"],
        color="blue",
        linewidth=2,
        label="Mean",
    )

    ax.fill_between(
        hourly_stats["time_decimal"],
        hourly_stats["mean"] - hourly_stats["std"],
        hourly_stats["mean"] + hourly_stats["std"],
        alpha=0.3,
        color="blue",
        label="± 1 SD",
    )

    ax.fill_between(
        hourly_stats["time_decimal"],
        hourly_stats["p25"],
        hourly_stats["p75"],
        alpha=0.2,
        color="green",
        label="Interquartile range",
    )

    ax.set_xlabel("Time of Day", fontsize=12)
    ax.set_ylabel("Glucose Level (mg/dL)", fontsize=12)
    ax.set_title("Average Daily Glucose Variations", fontsize=14)

    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.set_xlim(0, 24)
    ax.set_ylim(GLUCOSE_AXIS_MIN, GLUCOSE_AXIS_MAX)

    ax.axhline(y=TARGET_LOW, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(y=TARGET_HIGH, color="red", linestyle="--", linewidth=1, alpha=0.7)

    ax.grid(True, alpha=0.3)
    ax.legend()

    if created:
        fig.tight_layout()
    return fig


def _add_reference_lines(ax):
    """Adds reference lines to the chart."""
    ax.axhline(y=TARGET_LOW, color="#FF6666", linestyle="--", linewidth=1)
    ax.axhline(y=TARGET_HIGH, color="#FF6666", linestyle="--", linewidth=1)
    ax.text(24, TARGET_LOW + 2, f"{TARGET_LOW} mg/dL", va="bottom", ha="right", color="#FF6666")
    ax.text(24, TARGET_HIGH + 2, f"{TARGET_HIGH} mg/dL", va="bottom", ha="right", color="#FF6666")


def _configure_daily_plot(ax, title: str):
    """Configures common elements of the daily chart."""
    ax.set_xlabel("Time of Day", fontsize=12, fontweight="bold")
    ax.set_ylabel("Glucose Level (mg/dL)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")

    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(GLUCOSE_AXIS_MIN, GLUCOSE_AXIS_MAX)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, linestyle=":", alpha=0.6)


def _configure_overlapping_plot(ax: Axes | None = None) -> None:
    """Configures the overlapping days chart."""
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel("Time of Day", fontsize=12)
    ax.set_ylabel("Glucose Level (mg/dL)", fontsize=12)
    ax.set_title("Overlapping Glucose Profiles", fontsize=14)

    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])

    ax.axhline(y=TARGET_LOW, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=TARGET_HIGH, color="red", linestyle="--", alpha=0.5)

    ax.axhspan(GLUCOSE_AXIS_MIN, TARGET_LOW, facecolor="#ffcccb", alpha=0.2)
    ax.axhspan(TARGET_LOW, TARGET_HIGH, facecolor="#90ee90", alpha=0.2)
    ax.axhspan(TARGET_HIGH, GLUCOSE_AXIS_MAX, facecolor="#ffcccb", alpha=0.2)

    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(GLUCOSE_AXIS_MIN, GLUCOSE_AXIS_MAX)
