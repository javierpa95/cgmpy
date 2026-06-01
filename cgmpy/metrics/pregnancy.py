"""
Module for simplified gestational diabetes analysis.
"""

from typing import Any

import pandas as pd

from ..data.pregnancy_data import PregnancyData
from . import ModularGlucoseMetrics


class GestationalDiabetes(PregnancyData, ModularGlucoseMetrics):
    """
    Unified class for gestational diabetes analysis.
    Inherits data processing from PregnancyData and calculation logic from ModularGlucoseMetrics.
    """

    def __init__(
        self, data_source: str | pd.DataFrame, delivery_date: str, week: int, day: int = 0, **kwargs
    ):
        # 1. Initialize data and trimesters via PregnancyData
        # (Passes target_type="pregnancy" automatically from PregnancyData)
        super().__init__(
            data_source=data_source, delivery_date=delivery_date, week=week, day=day, **kwargs
        )

        # 2. Local import to avoid circular dependency
        from .. import GlucoseMetrics

        # 3. Create metric wrappers for each trimester
        self.t1 = self._wrap_trimester(self.trimesters["first_trimester"], GlucoseMetrics)
        self.t2 = self._wrap_trimester(self.trimesters["second_trimester"], GlucoseMetrics)
        self.t3 = self._wrap_trimester(self.trimesters["third_trimester"], GlucoseMetrics)

    def _wrap_trimester(self, df: pd.DataFrame, cls) -> None | Any:
        if len(df) == 0:
            return None
        return cls(data_source=df, target_type="pregnancy")

    def summary_by_trimester(self) -> dict[str, Any]:
        """Simplified comparative summary."""
        return {
            "T1": self.t1.all_simplified() if self.t1 else None,
            "T2": self.t2.all_simplified() if self.t2 else None,
            "T3": self.t3.all_simplified() if self.t3 else None,
        }

    def calculate_all_metrics(self, flatten: bool = False) -> dict[str, Any]:
        """
        Complete analysis summary.

        Args:
            flatten (bool): If True, returns a flat dictionary with prefixes
                           (total_, t1_, t2_, t3_, gest_) suitable for CSV/DataFrames.
        """
        w, d = self.get_weeks_days()
        results = {
            "gestation": {
                "weeks": w,
                "days": d,
                "conception": self.conception_date.isoformat(),
                "delivery": self.delivery_date.isoformat(),
            },
            "overall": self.all_simplified(),
            "trimesters": self.summary_by_trimester(),
        }

        if not flatten:
            return results

        # Flattening logic
        flat = {}
        # Gestation
        for k, v in results["gestation"].items():
            flat[f"gest_{k}"] = v
        # Overall
        for k, v in results["overall"].items():
            flat[f"total_{k}"] = v
        # Trimesters
        for t_key, t_metrics in results["trimesters"].items():
            if t_metrics:
                for k, v in t_metrics.items():
                    flat[f"{t_key.lower()}_{k}"] = v
            else:
                # Fill with None if trimester is empty to keep consistent columns
                # We can take keys from all_simplified template
                template = self.all_simplified().keys()
                for k in template:
                    flat[f"{t_key.lower()}_{k}"] = None

        return flat

    def __str__(self) -> str:
        w, d = self.get_weeks_days()
        output = [
            f"=== GESTATIONAL DIABETES REPORT ({w}+{d} weeks) ===",
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
