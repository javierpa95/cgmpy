"""
Module for pregnancy-specific glucose analysis.
"""

from typing import Any

import pandas as pd

from ..analysis.core import GlucoseAnalysis
from ..data.core import GlucoseData
from ..data.pregnancy_data import PregnancyData


class PregnancyAnalysis:
    """Analysis of CGM data during pregnancy (pregestational and gestational).

    Provides trimester-specific metrics and reports for pregnant women
    with diabetes.

    Args:
        data_source: A file path or DataFrame with glucose data.
        delivery_date: Estimated delivery date in 'YYYY-MM-DD' format.
        week: Gestation week at delivery.
        day: Additional days (default 0).
        **kwargs: Additional arguments passed to ``PregnancyData``.
    """

    def __init__(
        self,
        data_source: str | pd.DataFrame | GlucoseData,
        delivery_date: str,
        week: int,
        day: int = 0,
        **kwargs: Any,
    ):
        if isinstance(data_source, GlucoseData):
            # Use the already-loaded GlucoseData directly
            self._pregnancy_data = PregnancyData(
                data_source=data_source.data,
                delivery_date=delivery_date,
                week=week,
                day=day,
                target_type="pregnancy",
                **kwargs,
            )
            self._analysis = GlucoseAnalysis(data_source)
        else:
            self._pregnancy_data = PregnancyData(
                data_source=data_source,
                delivery_date=delivery_date,
                week=week,
                day=day,
                **kwargs,
            )
            self._analysis = GlucoseAnalysis(
                GlucoseData(
                    data_source=self._pregnancy_data.data,
                    target_type="pregnancy",
                )
            )

        def _wrap_trimester(df: pd.DataFrame):
            if len(df) == 0:
                return None
            return GlucoseAnalysis(GlucoseData(data_source=df, target_type="pregnancy"))

        self.t1 = _wrap_trimester(self._pregnancy_data.trimesters["first_trimester"])
        self.t2 = _wrap_trimester(self._pregnancy_data.trimesters["second_trimester"])
        self.t3 = _wrap_trimester(self._pregnancy_data.trimesters["third_trimester"])

    # ── Delegate basic accessors ────────────────────────────────────────

    @property
    def data(self) -> pd.DataFrame:
        return self._pregnancy_data.data

    def gmi(self) -> float:
        return self._analysis.gmi()

    def TIR(self) -> float:
        return self._analysis.TIR()

    def data_completeness(self) -> int:
        return self._analysis.data_completeness()

    def sd_within_day(self) -> dict:
        return self._analysis.sd_within_day()

    def sd_daily_mean(self) -> dict:
        return self._analysis.sd_daily_mean()

    def MAGE(self) -> float:
        return self._analysis.MAGE()

    def MODD(self, days: int = 1) -> dict:
        return self._analysis.MODD(days)

    def CONGA(self, hours: int = 4, max_gap_minutes: float | None = None) -> dict:
        return self._analysis.CONGA(hours, max_gap_minutes)

    def LBGI(self) -> float:
        return self._analysis.LBGI()

    def HBGI(self) -> float:
        return self._analysis.HBGI()

    def ADRR(self) -> dict:
        return self._analysis.ADRR()

    def GRI(self, pregnancy: bool = False) -> dict:
        return self._analysis.GRI(pregnancy)

    def j_index(self) -> float:
        return self._analysis.j_index()

    def get_weeks_days(self) -> tuple[int, int]:
        return self._pregnancy_data.get_weeks_days()

    # ── Pregnancy-specific methods ──────────────────────────────────────

    def summary_by_trimester(self) -> dict[str, Any]:
        """Simplified comparative summary."""

        def _trimester_metrics(ta):
            if ta is None:
                return None
            basic = ta.calculate_all_metrics()
            return {
                "DataCompleteness": ta.data_completeness(),
                "GMI": basic.get("GMI"),
                "Mean": basic.get("Mean"),
                "Median": basic.get("Median"),
                "SD": basic.get("Std"),
                "CV": basic.get("CV"),
                "TIR": ta.TIR(),
                "TIR_tight": ta.TIR_tight(),
                "TAR140": ta.TAR140(),
                "TAR250": ta.TAR250(),
                "TBR63": ta.TBR63(),
                "TBR55": ta.TBR55(),
                "SDw": ta.sd_within_day().get("sd"),
                "SDdm": ta.sd_daily_mean().get("sd"),
                "MAGE": ta.MAGE(),
                "MODD": ta.MODD().get("value"),
                "CONGA4": ta.CONGA(hours=4).get("value"),
                "LBGI": ta.LBGI(),
                "HBGI": ta.HBGI(),
                "ADRR": ta.ADRR().get("adrr"),
                "GRI": ta.GRI(pregnancy=True).get("GRI"),
                "J-Index": ta.j_index(),
            }

        return {
            "T1": _trimester_metrics(self.t1),
            "T2": _trimester_metrics(self.t2),
            "T3": _trimester_metrics(self.t3),
        }

    def all_simplified(self) -> dict:
        """Simplified metrics dictionary for overall pregnancy data."""
        basic = self._analysis.calculate_all_metrics()
        simplified = {
            "DataCompleteness": self.data_completeness(),
            "GMI": basic.get("GMI"),
            "Mean": basic.get("Mean"),
            "Median": basic.get("Median"),
            "SD": basic.get("Std"),
            "CV": basic.get("CV"),
            "TIR": self.TIR(),
            "TIR_tight": self._analysis.TIR_tight(),
            "TAR140": self._analysis.TAR_total(),
            "TAR250": self._analysis.TAR250(),
            "TBR63": self._analysis.TBR_total(),
            "TBR55": self._analysis.TBR55(),
            "SDw": self.sd_within_day().get("sd"),
            "SDdm": self.sd_daily_mean().get("sd"),
            "MAGE": self.MAGE(),
            "MODD": self.MODD().get("value"),
            "CONGA4": self.CONGA(hours=4).get("value"),
            "LBGI": self.LBGI(),
            "HBGI": self.HBGI(),
            "ADRR": self.ADRR().get("adrr"),
            "GRI": self.GRI(pregnancy=True).get("GRI"),
            "J-Index": self.j_index(),
        }
        return simplified

    def calculate_all_metrics(self, flatten: bool = False) -> dict[str, Any]:
        """Complete analysis summary.

        Args:
            flatten: If True, returns a flat dictionary with prefixes
                    (total_, t1_, t2_, t3_, gest_) suitable for CSV/DataFrames.
        """
        w, d = self.get_weeks_days()
        results = {
            "gestation": {
                "weeks": w,
                "days": d,
                "conception": self._pregnancy_data.conception_date.isoformat(),
                "delivery": self._pregnancy_data.delivery_date.isoformat(),
            },
            "overall": self.all_simplified(),
            "trimesters": self.summary_by_trimester(),
        }

        if not flatten:
            return results

        flat = {}
        for k, v in results["gestation"].items():
            flat[f"gest_{k}"] = v
        for k, v in results["overall"].items():
            flat[f"total_{k}"] = v
        for t_key, t_metrics in results["trimesters"].items():
            if t_metrics:
                for k, v in t_metrics.items():
                    flat[f"{t_key.lower()}_{k}"] = v
            else:
                template = self.all_simplified().keys()
                for k in template:
                    flat[f"{t_key.lower()}_{k}"] = None

        return flat

    def __str__(self) -> str:
        w, d = self.get_weeks_days()
        output = [
            f"=== PREGNANCY ANALYSIS REPORT ({w}+{d} weeks) ===",
            f"Overall GMI: {self.gmi():.1f}% | TIR (63-140): {self.TIR():.1f}%",
            "\nTrimester Breakdown:",
        ]

        summary = self.summary_by_trimester()
        for t_label, metrics in summary.items():
            if metrics:
                output.append(
                    f"  {t_label}: GMI {metrics['GMI']:.1f}% | TIR {metrics['TIR']:.1f}% | CV {metrics['CV']:.1f}%"
                )
            else:
                output.append(f"  {t_label}: (No data available)")

        return "\n".join(output)
