"""Variability metrics package.

This package splits the original ``variability.py`` module into one
file per metric family (SD, MAGE, MODD, CONGA, Lability, Risk).
For convenience, :class:`VariabilityMetrics` is re-exported here as
a composite mixin combining every family — so existing code that
imports it continues to work unchanged.

Individual mixin classes are also exported for users who only need
a subset of metrics::

    from cgmpy.metrics.variability import SDMetrics, MAGEMetrics
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._base import VariabilityBase
from .conga import CONGAMetrics
from .lability import LabilityMetrics
from .mage import MAGEMetrics
from .modd import MODDMetrics
from .risk import RiskMetrics
from .sd import SDMetrics


class VariabilityMetrics(
    SDMetrics,
    MAGEMetrics,
    MODDMetrics,
    CONGAMetrics,
    LabilityMetrics,
    RiskMetrics,
):
    """Composite mixin combining all variability metric families.

    This is a mixin class: it must be combined with
    :class:`~cgmpy.data.core.ModularGlucoseData` to be useful, e.g.::

        class MyGlucose(ModularGlucoseData, VariabilityMetrics):
            pass
    """

    def calculate_variability_metrics(self) -> dict:
        """
        Compute a full battery of variability metrics in a single call.

        Aggregates the results of MAGE (Baghurst + simple), MODD, CONGA,
        Lability, ADRR, GRI, GRADE, HBGI, LBGI, M-Value, and J-Index.
        Each individual call is wrapped in a try/except so that a failure
        in one family does not abort the whole computation.

        :return: Dictionary with all computed metrics.
        """
        try:
            metrics = {
                "data_completeness": self.data_completeness(),
                "Mean": self.mean(),
                "Median": self.median(),
                "Std": self.sd(),
                "CV": self.cv(),
                "GMI": self.gmi(),
                "TIR": self.TIR(),
                "TIR_tight": self.TIR_tight(),
                "TIR_pregnancy": self.TIR_pregnancy(),
                "TAR180": self.TAR180(),
                "TAR250": self.TAR250(),
                "TAR140": self.TAR140(),
                "TBR70": self.TBR70(),
                "TBR63": self.TBR63(),
                "TBR55": self.TBR55(),
                "Skewness": float(self.data["glucose"].skew()),
                "Kurtosis": float(self.data["glucose"].kurtosis()),
            }

            sd_metrics = {
                "SDT": self.sd_total().get("sd"),
                "SDW": self.sd_within_day().get("sd"),
                "SD_timepoints": self.sd_between_timepoints().get("sd"),
                "SD_night": self.sd_segment("00:00", 8).get("sd"),
                "SD_day": self.sd_segment("08:00", 8).get("sd"),
                "SD_evening": self.sd_segment("16:00", 8).get("sd"),
                "SD_1h": self.sd_within_series(hours=1).get("sd"),
                "SD_6h": self.sd_within_series(hours=6).get("sd"),
                "SD_24h": self.sd_within_series(hours=24).get("sd"),
                "SD_daily_mean": self.sd_daily_mean().get("sd"),
                "SD_same_timepoint": self.sd_same_timepoint().get("sd"),
                "SD_same_timepoint_adj": self.sd_same_timepoint_adjusted().get("sd"),
                "SD_interaction": self.sd_interaction().get("sd"),
            }
            metrics.update(sd_metrics)

            conga_metrics = {
                "CONGA1": self.CONGA(hours=1).get("value"),
                "CONGA2": self.CONGA(hours=2).get("value"),
                "CONGA4": self.CONGA(hours=4).get("value"),
                "CONGA6": self.CONGA(hours=6).get("value"),
                "CONGA24": self.CONGA(hours=24).get("value"),
            }
            metrics.update(conga_metrics)

            try:
                mage_results = self.MAGE_Baghurst()
                metrics.update(
                    {
                        "mage_plus": mage_results.get("MAGE+"),
                        "mage_minus": mage_results.get("MAGE-"),
                        "mage_avg": mage_results.get("MAGE_avg"),
                        "mage_sd": mage_results.get("SD_used"),
                        "mage_threshold": mage_results.get("threshold"),
                        "mage_excursions": mage_results.get("num_excursions"),
                    }
                )
            except Exception as e:
                if getattr(self, "log", False):
                    self.logger.error("Error calculating MAGE: %s", e)

            try:
                modd_result = self.MODD()
                metrics.update(
                    {
                        "modd": modd_result.get("value"),
                        "modd_sd": modd_result.get("std"),
                    }
                )
            except Exception as e:
                if getattr(self, "log", False):
                    self.logger.error("Error calculating MODD: %s", e)

            try:
                lgbi = self.LBGI()
                hbgi = self.HBGI()
                adrr = self.ADRR()
                gri = self.GRI()
                gri_pregnancy = self.GRI(pregnancy=True)
                grade = self.GRADE()
                m_value = self.M_Value()
                j_index = self.j_index()

                risk_metrics = {
                    "LBGI": lgbi,
                    "HBGI": hbgi,
                    "ADRR": adrr.get("adrr") if isinstance(adrr, dict) else adrr,
                    "GRI": gri.get("GRI") if isinstance(gri, dict) else gri,
                    "GRI_high": gri.get("derived_metrics", {}).get("hyper_component", 0),
                    "GRI_low": gri.get("derived_metrics", {}).get("hypo_component", 0),
                    "GRI_pregnancy": gri_pregnancy.get("GRI")
                    if isinstance(gri_pregnancy, dict)
                    else gri_pregnancy,
                    "GRI_pregnancy_high": gri_pregnancy.get("derived_metrics", {}).get(
                        "hyper_component", 0
                    ),
                    "GRI_pregnancy_low": gri_pregnancy.get("derived_metrics", {}).get(
                        "hypo_component", 0
                    ),
                    "GRADE": grade.get("grade_score") if isinstance(grade, dict) else grade,
                    "M_Value": m_value if not isinstance(m_value, dict) else m_value.get("M_Value"),
                    "J_Index": j_index,
                }

                metrics.update(risk_metrics)

            except Exception as e:
                if getattr(self, "log", False):
                    self.logger.error("General error calculating risk metrics: %s", e)
                import traceback

                traceback.print_exc()

            return metrics
        except Exception as e:
            return {"error": str(e), "message": "Error calculating metrics"}


__all__ = [
    "CONGAMetrics",
    "LabilityMetrics",
    "MAGEMetrics",
    "MODDMetrics",
    "RiskMetrics",
    "SDMetrics",
    "VariabilityBase",
    "VariabilityMetrics",
]
