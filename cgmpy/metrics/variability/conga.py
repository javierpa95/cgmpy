"""CONGA (Continuous Overlapping Net Glycemic Action) metric."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._base import VariabilityBase

if TYPE_CHECKING:
    pass


class CONGAMetrics(VariabilityBase):
    """Mixin providing CONGA calculation for glucose data."""

    if TYPE_CHECKING:
        data: pd.DataFrame
        typical_interval: float

    def CONGA(self, hours: int = 4, max_gap_minutes: float | None = None) -> dict:
        """
        Calculates CONGA (Continuous Overlapping Net Glycemic Action).

        CONGA measures intraday glycemic variability by calculating the standard deviation
        of differences between current values and values 'n' hours earlier.

        :param hours: Number of hours for the time interval (n).
        :param max_gap_minutes: Maximum allowed gap in minutes between measurements to
                           consider a comparison valid. If None, uses 2 times the typical interval.
        :return: Dictionary with CONGA value and related statistics.
        :reference: McDonnell CM, et al. Diabetes Technol Ther. 2005;7(2):243-9.
                   DOI: 10.1089/dia.2005.7.243
        """
        # Create a sorted copy of the data by time
        df = self.data.sort_values("time").copy()

        # Compute the interval in minutes
        interval_minutes = self.typical_interval  # Ya está en minutos

        # If max_gap_minutes is not specified, use 2 times the typical interval
        if max_gap_minutes is None:
            max_gap_minutes = 2 * interval_minutes

        # Compute how many intervals correspond to 'hours' hours
        n_intervals = int((hours * 60) / interval_minutes)

        if n_intervals <= 0:
            raise ValueError(f"The interval of {hours} hours is too small for the available data")

        # Compute differences between current values and values from 'n' hours ago
        # but accounting for possible disconnections

        # Method 1: Use shift but verify the actual time difference
        df["time_n_hours_ago"] = df["time"].shift(n_intervals)
        df["glucose_n_hours_ago"] = df["glucose"].shift(n_intervals)

        # Compute the actual time difference in minutes
        df["time_diff_minutes"] = (df["time"] - df["time_n_hours_ago"]).dt.total_seconds() / 60

        # Compute glucose difference only if the time difference is close to the target
        target_diff_minutes = hours * 60
        df["valid_comparison"] = (
            df["time_diff_minutes"] >= target_diff_minutes - max_gap_minutes
        ) & (df["time_diff_minutes"] <= target_diff_minutes + max_gap_minutes)

        # Compute the difference only for valid comparisons
        df["difference"] = np.where(
            df["valid_comparison"], df["glucose"] - df["glucose_n_hours_ago"], np.nan
        )

        # Drop rows with missing values or invalid comparisons
        valid_data = df.dropna(subset=["difference"])

        if len(valid_data) == 0:
            return {
                "value": None,
                "n_observations": 0,
                "mean_difference": None,
                "abs_mean_difference": None,
                "std": None,
                "hours": hours,
                "max_gap_minutes": max_gap_minutes,
            }

        # Compute CONGA as the standard deviation of the differences
        conga_value = valid_data["difference"].std()

        # Compute additional statistics
        mean_diff = valid_data["difference"].mean()
        abs_mean_diff = valid_data["difference"].abs().mean()

        # Information about disconnections
        total_comparisons = len(df.dropna(subset=["glucose_n_hours_ago"]))
        valid_comparisons = len(valid_data)
        invalid_comparisons = total_comparisons - valid_comparisons

        return {
            "value": conga_value,
            "n_observations": len(valid_data),
            "mean_difference": mean_diff,
            "abs_mean_difference": abs_mean_diff,
            "hours": hours,
            "max_gap_minutes": max_gap_minutes,
            "total_comparisons": total_comparisons,
            "valid_comparisons": valid_comparisons,
            "invalid_comparisons": invalid_comparisons,
            "percent_valid": (valid_comparisons / total_comparisons * 100)
            if total_comparisons > 0
            else 0,
        }
