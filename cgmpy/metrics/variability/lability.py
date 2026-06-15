"""Lability Index and variability summaries."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..units import MGDL_TO_MMOLL

if TYPE_CHECKING:
    pass


# ──────────────────────────────────────────────
# Pure function
# ──────────────────────────────────────────────


def lability_index(
    glucose: pd.Series, timestamps: pd.Series, interval: int = 1, period: str = "week"
) -> dict:
    """Lability Index.

    Args:
        glucose: Glucose values.
        timestamps: Corresponding timestamps.
        interval: Hours between measurements (must be > 0).
        period: 'week' or 'month'.

    Returns:
        dict with LI values and statistics.

    Raises:
        ValueError: If interval <= 0.

    Reference:
        DOI: 10.2337/diabetes.53.4.955
    """
    if interval <= 0:
        raise ValueError(f"interval must be positive, got {interval}")

    data_copy = pd.DataFrame({"time": timestamps, "glucose": glucose}).copy()
    data_copy["time_rounded"] = data_copy["time"].dt.floor("h")
    data_copy["week"] = data_copy["time"].dt.isocalendar().week

    weekly_li = []

    for _week, group in data_copy.groupby("week"):
        group = group.sort_values("time_rounded")
        glucose_diffs = group["glucose"].shift(-interval) - group["glucose"]
        li_values = (glucose_diffs**2) / interval
        li_week = li_values.dropna().sum()
        weekly_li.append(li_week)

    mean_li = np.mean(weekly_li) if weekly_li else 0
    mean_li_mmol = mean_li / (MGDL_TO_MMOLL**2)
    mean_li_por_hora = mean_li / 168
    typical_change_per_hour = math.sqrt(mean_li_por_hora)

    return {
        "weekly_values": weekly_li,
        "mean_li": mean_li,
        "mean_li_mmol": mean_li_mmol,
        "std_li": np.std(weekly_li) if len(weekly_li) > 1 else 0,
        "n_weeks": len(weekly_li),
        "typical_change_per_hour": typical_change_per_hour,
    }
