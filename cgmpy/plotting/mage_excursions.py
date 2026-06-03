"""MAGE excursion visualization module.

Extracted from ``cgmpy/metrics/variability/mage.py`` to decouple
plotting from metric computation.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)


def plot_mage_excursions(
    glucose: np.ndarray,
    times: np.ndarray,
    turning_points_approaches: dict[int, list[int]],
    threshold: float,
    threshold_sd: int = 1,
    mean_glucose: float = 0,
) -> plt.Figure:
    """Plot MAGE excursions for all three approaches.

    Args:
        glucose: Glucose values array.
        times: Corresponding timestamps array.
        turning_points_approaches: Dict mapping approach number to list of
            turning point indices.
        threshold: The threshold value used.
        threshold_sd: Number of SDs (for display).
        mean_glucose: Mean glucose (for threshold lines).

    Returns:
        The matplotlib Figure.
    """
    unique_days = pd.Series(times).dt.normalize().unique()

    if len(unique_days) == 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    plt.ion()

    excursions_by_approach: dict[int, list[dict[str, Any]]] = {}
    for approach_num in [1, 2, 3]:
        tp = turning_points_approaches.get(approach_num, [])
        excursions_approach: list[dict[str, Any]] = []
        if len(tp) > 1:
            last_val = glucose[tp[0]]
            last_point = tp[0]
            for point in tp[1:]:
                curr_val = glucose[point]
                diff = abs(curr_val - last_val)
                if diff >= threshold:
                    excursions_approach.append(
                        {
                            "start_point": last_point,
                            "end_point": point,
                            "start": last_val,
                            "end": curr_val,
                            "type": "up" if curr_val > last_val else "down",
                            "magnitude": diff,
                        }
                    )
                last_val = curr_val
                last_point = point
        excursions_by_approach[approach_num] = excursions_approach

    current_day_index = 0

    def update_plot(day_index: int) -> None:
        for ax in axs:
            ax.clear()

        current_day = unique_days[day_index]
        next_day = current_day + timedelta(days=1)

        day_mask = (times >= current_day) & (times < next_day)
        day_times = times[day_mask]
        day_glucose = glucose[day_mask]

        if len(day_times) == 0:
            return

        for i, approach_num in enumerate([1, 2, 3]):
            ax = axs[i]
            ax.plot(day_times, day_glucose, "b-", label="Glucosa")

            approach_turning_points = turning_points_approaches.get(approach_num, [])
            approach_excursions = excursions_by_approach.get(approach_num, [])

            day_turning_points = [tp for tp in approach_turning_points if day_mask[tp]]

            excursion_points: set[int] = set()
            day_excursions: list[dict[str, Any]] = []
            for exc in approach_excursions:
                start_point = exc["start_point"]
                end_point = exc["end_point"]
                if day_mask[start_point] or day_mask[end_point]:
                    if day_mask[start_point]:
                        excursion_points.add(start_point)
                    if day_mask[end_point]:
                        excursion_points.add(end_point)
                    day_excursions.append(exc)

            significant_points = [tp for tp in day_turning_points if tp in excursion_points]
            non_significant_points = [tp for tp in day_turning_points if tp not in excursion_points]

            for tp in non_significant_points:
                ax.plot(times[tp], glucose[tp], "bo", markersize=6)
            for tp in significant_points:
                ax.plot(times[tp], glucose[tp], "ro", markersize=8)
            for exc in day_excursions:
                start_point = exc["start_point"]
                end_point = exc["end_point"]
                if day_mask[start_point] and day_mask[end_point]:
                    color = "green" if exc["type"] == "up" else "red"
                    ax.plot(
                        [times[start_point], times[end_point]],
                        [glucose[start_point], glucose[end_point]],
                        color=color,
                        linewidth=2.5,
                        alpha=0.7,
                    )

            excursions_up = [e["magnitude"] for e in day_excursions if e["type"] == "up"]
            excursions_down = [e["magnitude"] for e in day_excursions if e["type"] == "down"]

            mage_plus = np.mean(excursions_up) if excursions_up else 0
            mage_minus = np.mean(excursions_down) if excursions_down else 0
            mage_avg = (
                np.mean(excursions_up + excursions_down)
                if (excursions_up or excursions_down)
                else 0
            )

            approach_name = (
                "Suavizado"
                if approach_num == 1
                else "Direct elimination"
                if approach_num == 2
                else "Improved smoothing"
            )
            ax.set_title(
                f"MAGE Baghurst - Approach {approach_num} ({approach_name}) - "
                f"{current_day.strftime('%d/%m/%Y')}\n"
                f"MAGE+: {mage_plus:.1f}, MAGE-: {mage_minus:.1f}, "
                f"MAGE: {mage_avg:.1f}, Excursions: {len(day_excursions)}"
            )
            ax.set_ylabel("Glucose (mg/dL)")
            ax.grid(True)
            ax.axhline(
                y=mean_glucose + threshold,
                color="g",
                linestyle="--",
                label=f"Threshold (+{threshold_sd} SD)",
            )
            ax.axhline(
                y=mean_glucose - threshold,
                color="g",
                linestyle="--",
                label=f"Threshold (-{threshold_sd} SD)",
            )

            custom_lines = [
                Line2D([0], [0], color="b", marker="o", linestyle="None", markersize=6),
                Line2D([0], [0], color="r", marker="o", linestyle="None", markersize=8),
                Line2D([0], [0], color="green", linewidth=2.5),
                Line2D([0], [0], color="red", linewidth=2.5),
                Line2D([0], [0], color="g", linestyle="--"),
            ]
            ax.legend(
                custom_lines,
                [
                    "Turning points",
                    "Excursion points",
                    "Positive excursion",
                    "Negative excursion",
                    "Threshold (+-1 SD)",
                ],
            )

        axs[2].set_xlabel("Time")

        for ax in axs:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        fig.suptitle(
            f"Day {day_index + 1} of {len(unique_days)} - "
            "Press left/right arrows to navigate, q to quit",
            fontsize=12,
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        fig.canvas.draw_idle()

    def on_key(event: Any) -> None:
        nonlocal current_day_index
        if event.key == "right" and current_day_index < len(unique_days) - 1:
            current_day_index += 1
            update_plot(current_day_index)
        elif event.key == "left" and current_day_index > 0:
            current_day_index -= 1
            update_plot(current_day_index)
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    update_plot(current_day_index)

    logger.info("Navigation:")
    logger.info("  <- Left arrow: Previous day")
    logger.info("  -> Right arrow: Next day")
    logger.info("  q: Quit")

    plt.show(block=True)

    return fig
