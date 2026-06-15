"""CONGA (Continuous Overlapping Net Glycemic Action) metric."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


# ──────────────────────────────────────────────
# Pure function
# ──────────────────────────────────────────────


def conga(
    glucose: pd.Series, timestamps: pd.Series, hours: int = 4, max_gap_minutes: float | None = None
) -> dict:
    """Continuous Overlapping Net Glycemic Action.

    Args:
        glucose: Glucose values.
        timestamps: Corresponding timestamps.
        hours: Number of hours for the interval (1, 2, 4, 6, or 24).
        max_gap_minutes: Maximum allowed gap in minutes.

    Returns:
        dict with "value" and "n" keys.

    Reference:
        McDonnell CM, et al. 2005, DOI: 10.1089/dia.2005.7.243
    """
    df = pd.DataFrame({"time": timestamps, "glucose": glucose}).sort_values("time")
    interval_minutes = df["time"].diff().dt.total_seconds().median() / 60.0

    if max_gap_minutes is None:
        max_gap_minutes = 2 * interval_minutes

    n_intervals = int((hours * 60) / interval_minutes)

    if n_intervals <= 0:
        return {"value": 0.0, "n": 0}

    target_diff_minutes = hours * 60
    df["time_n_hours_ago"] = df["time"].shift(n_intervals)
    df["glucose_n_hours_ago"] = df["glucose"].shift(n_intervals)
    df["time_diff_minutes"] = (df["time"] - df["time_n_hours_ago"]).dt.total_seconds() / 60.0
    df["valid_comparison"] = (df["time_diff_minutes"] >= target_diff_minutes - max_gap_minutes) & (
        df["time_diff_minutes"] <= target_diff_minutes + max_gap_minutes
    )
    df["difference"] = np.where(
        df["valid_comparison"], df["glucose"] - df["glucose_n_hours_ago"], np.nan
    )
    valid_data = df.dropna(subset=["difference"])

    if len(valid_data) == 0:
        return {"value": 0.0, "n": 0}

    conga_value = float(valid_data["difference"].std())

    return {
        "value": conga_value,
        "n": len(valid_data),
    }
