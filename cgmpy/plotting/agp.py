"""
Module for ambulatory glucose profile (AGP) plots.

This module contains functions to generate ambulatory profiles:
- Standard AGP
- AGP by day of week
- Helper functions for percentile calculation
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._constants import GLUCOSE_AXIS_MAX, GLUCOSE_AXIS_MIN
from ._utils import add_glucose_zones, resolve_axes


def plot_agp(data: pd.DataFrame, smoothing_window: int = 15, ax: Axes | None = None) -> Figure:
    """Builds the enhanced Ambulatory Glucose Profile (AGP).

    Args:
        data: Glucose DataFrame with ``time`` and ``glucose`` columns.
        smoothing_window: Smoothing window in minutes (default 15).
        ax: Optional axis to draw into. If ``None`` a new figure is created.

    Returns:
        The matplotlib ``Figure`` containing the plot.
    """
    data_copy = data.copy()
    data_copy["time_decimal"] = (
        data_copy["time"].dt.hour + data_copy["time"].dt.minute / 60.0
    ).round(2)

    percentiles = data_copy.groupby("time_decimal")["glucose"].agg(
        [
            lambda x: np.percentile(x, 5),
            lambda x: np.percentile(x, 25),
            lambda x: np.percentile(x, 50),
            lambda x: np.percentile(x, 75),
            lambda x: np.percentile(x, 95),
        ]
    )

    percentiles.columns = [0.05, 0.25, 0.5, 0.75, 0.95]

    for col in percentiles.columns:
        percentiles[col] = (
            percentiles[col].rolling(window=smoothing_window, center=True, min_periods=1).mean()
        )

    percentiles = percentiles.sort_index()

    fig, ax, created = resolve_axes(ax, figsize=(14, 8))

    add_glucose_zones(ax)

    _plot_percentiles(ax, percentiles)

    _configure_agp_plot(ax, "Ambulatory Glucose Profile (AGP)")

    if created:
        fig.tight_layout()
    return fig


def generate_week_agp(
    data: pd.DataFrame, smoothing_window: int = 15, combined: bool = True
) -> Figure:
    """Builds the Ambulatory Glucose Profile (AGP) by day of week.

    Args:
        data: Glucose DataFrame with ``time`` and ``glucose`` columns.
        smoothing_window: Smoothing window in minutes (default 15).
        combined: If True, displays all days in a single chart.
                 If False, displays a subplot for each day.

    Returns:
        The matplotlib ``Figure`` containing the plot.
    """
    data_copy = data.copy()
    data_copy["time_decimal"] = (
        data_copy["time"].dt.hour + data_copy["time"].dt.minute / 60.0
    ).round(2)
    data_copy["weekday"] = data_copy["time"].dt.day_name()

    if combined:
        return _plot_combined_week_agp(data_copy, smoothing_window)
    return _plot_separate_week_agp(data_copy, smoothing_window)


def _plot_percentiles(ax, percentiles):
    """Plots the percentile lines."""
    ax.plot(
        percentiles.index,
        percentiles[0.5],
        label="Median",
        color="blue",
        linewidth=2,
    )

    ax.fill_between(
        percentiles.index,
        percentiles[0.25],
        percentiles[0.75],
        color="blue",
        alpha=0.3,
        label="Interquartile Range",
    )

    ax.fill_between(
        percentiles.index,
        percentiles[0.05],
        percentiles[0.95],
        color="lightblue",
        alpha=0.2,
        label="Percentiles 5-95%",
    )


def _configure_agp_plot(ax, title: str):
    """Configures common elements of the AGP chart."""
    ax.set_xlabel("Time of Day", fontsize=12)
    ax.set_ylabel("Glucose Level (mg/dL)", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold")

    ax.legend(title="Legend", loc="upper left", fontsize=10)

    ax.grid(True, linestyle=":", alpha=0.6)

    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])

    ax.set_ylim(GLUCOSE_AXIS_MIN, GLUCOSE_AXIS_MAX)


def _plot_combined_week_agp(data_copy, smoothing_window: int):
    """Plots combined AGP for all weekdays."""
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEEAD",
        "#D4A5A5",
        "#9B59B6",
    ]

    fig, ax = plt.subplots(figsize=(15, 8))

    add_glucose_zones(ax)

    for day, color in zip(days, colors, strict=False):
        day_data = data_copy[data_copy["weekday"] == day]

        if not day_data.empty:
            percentiles = _calculate_day_percentiles(day_data, smoothing_window)

            ax.plot(
                percentiles.index,
                percentiles[0.5],
                label=f"{day} (n={len(day_data['time'].dt.date.unique())} days)",
                color=color,
                linewidth=2,
            )

            ax.fill_between(
                percentiles.index,
                percentiles[0.25],
                percentiles[0.75],
                color=color,
                alpha=0.1,
            )

    ax.set_title(
        "Ambulatory Glucose Profile (AGP) by Day of Week",
        fontsize=14,
        pad=20,
    )
    ax.set_xlabel("Time of Day", fontsize=12)
    ax.set_ylabel("Glucose Level (mg/dL)", fontsize=12)
    ax.set_ylim(GLUCOSE_AXIS_MIN, GLUCOSE_AXIS_MAX)
    ax.grid(True, linestyle=":", alpha=0.6)

    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])

    ax.legend(
        title="Days of the week",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=10,
    )

    fig.tight_layout()
    return fig


def _plot_separate_week_agp(data_copy, smoothing_window: int):
    """Plots separate AGP for each weekday."""
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    fig, axes = plt.subplots(7, 1, figsize=(15, 20), sharex=True)
    fig.suptitle(
        "Ambulatory Glucose Profile (AGP) by Day of Week",
        fontsize=16,
        fontweight="bold",
        y=0.92,
    )

    for ax, day in zip(axes, days, strict=False):
        day_data = data_copy[data_copy["weekday"] == day]

        if not day_data.empty:
            percentiles = _calculate_full_day_percentiles(day_data, smoothing_window)

            add_glucose_zones(ax)

            _plot_percentiles(ax, percentiles)

            ax.set_title(
                f"{day} (n={len(day_data['time'].dt.date.unique())} days)",
                fontsize=12,
                pad=10,
            )
            ax.set_ylabel("Glucose (mg/dL)", fontsize=10)
            ax.set_ylim(GLUCOSE_AXIS_MIN, GLUCOSE_AXIS_MAX)
            ax.grid(True, linestyle=":", alpha=0.6)

    axes[-1].set_xlabel("Time of Day", fontsize=12)
    axes[-1].set_xticks(range(0, 25, 3))
    axes[-1].set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])

    fig.tight_layout()
    return fig


def _calculate_day_percentiles(dia_data, smoothing_window: int):
    """Calculate percentiles for data from a specific day (25, 50, 75)."""
    percentiles = dia_data.groupby("time_decimal")["glucose"].agg(
        [
            lambda x: np.percentile(x, 25),
            lambda x: np.percentile(x, 50),
            lambda x: np.percentile(x, 75),
        ]
    )

    percentiles.columns = [0.25, 0.5, 0.75]

    for col in percentiles.columns:
        percentiles[col] = (
            percentiles[col].rolling(window=smoothing_window, center=True, min_periods=1).mean()
        )

    return percentiles


def _calculate_full_day_percentiles(dia_data, smoothing_window: int):
    """Calculate full percentiles for data from a specific day (5, 25, 50, 75, 95)."""
    percentiles = dia_data.groupby("time_decimal")["glucose"].agg(
        [
            lambda x: np.percentile(x, 5),
            lambda x: np.percentile(x, 25),
            lambda x: np.percentile(x, 50),
            lambda x: np.percentile(x, 75),
            lambda x: np.percentile(x, 95),
        ]
    )

    percentiles.columns = [0.05, 0.25, 0.5, 0.75, 0.95]

    for col in percentiles.columns:
        percentiles[col] = (
            percentiles[col].rolling(window=smoothing_window, center=True, min_periods=1).mean()
        )

    return percentiles
