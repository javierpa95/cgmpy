"""Lability Index and variability summaries."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..units import MGDL_TO_MMOLL
from ._base import VariabilityBase

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


# ──────────────────────────────────────────────
# Mixin class (delegates to pure function)
# ──────────────────────────────────────────────


class LabilityMetrics(VariabilityBase):
    """Mixin providing Lability Index and variability summary calculations.

    Cross-mixin calls: :meth:`Variability` and :meth:`variability_summary`
    call methods provided by other mixins (e.g. :class:`SDMetrics`,
    :class:`MODDMetrics`, :class:`RiskMetrics`).
    """

    if TYPE_CHECKING:
        data: pd.DataFrame
        typical_interval: float

        def sd(self) -> float: ...
        def mean(self) -> float: ...
        def cv(self) -> float: ...
        def sd_total(self) -> dict: ...
        def sd_within_day(self, min_count_threshold: float = 0.5) -> dict: ...
        def MAGE(self) -> float: ...
        def MODD(self, days: int = 1) -> dict: ...
        def CONGA(self, hours: int = 4, max_gap_minutes: float | None = None) -> dict: ...
        def j_index(self) -> float: ...
        def LBGI(self) -> float: ...
        def HBGI(self) -> float: ...
        def M_Value(self, reference_glucose: int = 90) -> dict: ...

    def Lability_index(self, interval: int = 1, period: str = "week") -> dict:
        return lability_index(self.data["glucose"], self.data["time"], interval, period)

    def Variability(self) -> dict[str, Any]:
        """
        Calculates all variability metrics.
        :return: A JSON string with all variability metrics.
        """
        variability_metrics = {
            "CONGA1": self.CONGA(hours=1),
            "CONGA2": self.CONGA(hours=2),
            "CONGA4": self.CONGA(hours=4),
            "CONGA6": self.CONGA(hours=6),
            "CONGA24": self.CONGA(hours=24),
            "MODD": self.MODD(days=1),
            "J_index": self.j_index(),
            "LBGI": self.LBGI(),
            "HBGI": self.HBGI(),
            "MAGE": self.MAGE(),
            "M_value": self.M_Value(),
            "LI_week": self.Lability_index(interval=1, period="week"),
        }
        return variability_metrics

    def variability_summary(self) -> dict[str, Any]:
        """
        Complete summary of all variability metrics.

        Returns:
            dict: Complete summary of variability metrics
        """
        return {
            "basic_variability": {"sd_total": self.sd_total(), "cv": self.cv()},
            "excursion_metrics": {
                "mage": self.MAGE(),
            },
            "inter_day_variability": {
                "modd_1day": self.MODD(1),
                "modd_2days": self.MODD(2),
            },
            "intra_day_variability": {
                "conga_1h": self.CONGA(1),
                "conga_2h": self.CONGA(2),
                "conga_4h": self.CONGA(4),
            },
            "lability": {"lability_index": self.Lability_index()},
        }
