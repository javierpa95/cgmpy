"""Glycemic quality and risk metrics (LBGI, HBGI, GRI, GRADE, ADRR, M-Value, J-Index)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ._base import VariabilityBase

if TYPE_CHECKING:
    pass


class RiskMetrics(VariabilityBase):
    """Mixin providing glycemic risk and quality metrics.

    Cross-mixin calls: :meth:`GRI` calls :meth:`TBR`, :meth:`TAR` and
    :meth:`calculate_time_in_range` which are provided by
    :class:`~cgmpy.metrics.time_in_range.TimeInRangeMetrics`.
    These work at runtime because :class:`VariabilityMetrics` inherits
    from all the mixins.
    """

    if TYPE_CHECKING:
        data: pd.DataFrame

        def mean(self) -> float: ...
        def sd(self) -> float: ...
        def TBR(self, threshold: float) -> float: ...
        def TAR(self, threshold: float) -> float: ...
        def calculate_time_in_range(self, low: float, high: float) -> float: ...

    def M_Value(self, reference_glucose: int = 90) -> float:
        """
        Calculates M-Value according to Schlichtkrull's definition and Service's consideration.

        M-Value is a hybrid between:
        1. Mean blood glucose deviation
        2. Glycemic variability

        Special features:
        - Gives more weight to hypoglycemia than hyperglycemia
        - Uses 90 mg/dL as historical reference value. Original paper used 120 mg/dL
        - Combines mean deviation and fluctuation amplitude

        Formula: M = (1/n) sum |10 * log10(BG/120)|^3 + W/20
        (The correction factor can be omitted when there are more than 24 data points)

        :param reference_glucose: Reference value (default 90 mg/dL - updated from docstring default).
        :return: M-Value rounded to 2 decimals.
        :reference: 10.1111/j.0954-6820.1965.tb01810.x
        :reference: 10.2337/db12-1396
        """
        glucose_values = self.data["glucose"].values

        M_BS_values = np.abs(10 * np.log10(glucose_values / reference_glucose)) ** 3
        M_BS_mean = np.mean(M_BS_values)
        return round(float(M_BS_mean), 2)

    def j_index(self) -> float:
        """
        Calculates J-index.

        :return: J-index value.
        :reference: DOI: 10.1055/s-2007-979906
        """
        return 0.001 * (self.mean() + self.sd()) ** 2

    def GRADE(self, unit: str = "mg/dL") -> dict[str, float]:
        """
        Calculates GRADE (Glycaemic Risk Assessment in Diabetes Engine).

        :param unit: Unit of the glucose data, either "mg/dL" or "mmol/L".
        :return: Dictionary with GRADE components (score, hypo/eu/hyper percentages).
        :reference: DOI: 10.1111/j.1464-5491.2007.02119.x
        """
        df = self.data.copy()

        if unit.lower() == "mg/dl":
            df["glucose_value"] = df["glucose"]
        elif unit.lower() == "mmol/l":
            df["glucose_value"] = df["glucose"] * 18
        else:
            raise ValueError("The unit must be 'mg/dL' or 'mmol/L'")

        hypo_threshold = 70
        hyper_threshold = 140

        df["hypo"] = df["glucose_value"] < hypo_threshold
        df["eu"] = (df["glucose_value"] >= hypo_threshold) & (
            df["glucose_value"] <= hyper_threshold
        )
        df["hyper"] = df["glucose_value"] > hyper_threshold

        glucose_values = df["glucose_value"].values

        grade_values = np.zeros_like(glucose_values, dtype=float)

        valid_mask = (glucose_values >= 37) & (glucose_values <= 630)

        with np.errstate(invalid="ignore", divide="ignore"):
            glucose_mmol = glucose_values[valid_mask] / 18
            log_log_values = np.log10(np.log10(glucose_mmol))
            grade_values[valid_mask] = 425 * (log_log_values + 0.16) ** 2

        invalid_mask = ~valid_mask | ~np.isfinite(grade_values)
        grade_values[invalid_mask] = 50

        df["grade"] = grade_values

        grade_total = df["grade"].sum()
        grade_hypo = df.loc[df["hypo"], "grade"].sum()
        grade_eu = df.loc[df["eu"], "grade"].sum()
        grade_hyper = df.loc[df["hyper"], "grade"].sum()

        hypo_percent = (grade_hypo / grade_total) * 100 if grade_total > 0 else 0
        eu_percent = (grade_eu / grade_total) * 100 if grade_total > 0 else 0
        hyper_percent = (grade_hyper / grade_total) * 100 if grade_total > 0 else 0

        grade_score = df["grade"].mean()

        return {
            "grade_score": float(grade_score),
            "hypo_percent": float(hypo_percent),
            "eu_percent": float(eu_percent),
            "hyper_percent": float(hyper_percent),
        }

    def LBGI(self) -> float:
        """
        Calculates Low Blood Glucose Index (LBGI).

        :return: LBGI value.
        :reference: DOI: 10.2337/db12-1396
        """
        glucose_values = self.data["glucose"].values

        f_bg = 1.509 * ((np.log(glucose_values)) ** 1.084 - 5.381)
        r_bg = 10 * f_bg**2
        rl_bg = np.where(f_bg < 0, r_bg, 0)

        return float(np.mean(rl_bg))

    def HBGI(self) -> float:
        """
        Calculates High Blood Glucose Index (HBGI).

        :return: HBGI value.
        :reference: DOI: 10.2337/db12-1396
        """
        glucose_values = self.data["glucose"].values

        f_bg = 1.509 * ((np.log(glucose_values)) ** 1.084 - 5.381)
        r_bg = 10 * f_bg**2
        rh_bg = np.where(f_bg > 0, r_bg, 0)

        return float(np.mean(rh_bg))

    def GRI(self, pregnancy: bool = False) -> dict[str, Any]:
        """
        Calculates Glucose Risk Index (GRI).

        GRI combines time in different glucose ranges, giving different weights
        to hypoglycemia and hyperglycemia.

        GRI = (3.0 * VLow) + (2.4 * Low) + (1.6 * VHigh) + (0.8 * High)

        Standard ranges:
        - VLow: <54 mg/dL
        - Low: 54-70 mg/dL
        - VHigh: >250 mg/dL
        - High: 180-250 mg/dL

        Pregnancy ranges (Experimental, not clinically validated):
        - VLow: <55 mg/dL
        - Low: 55-63 mg/dL
        - VHigh: >250 mg/dL
        - High: 140-250 mg/dL

        :param pregnancy: If True, uses specific ranges for pregnancy.
        :return: Dictionary with GRI and its components.
        :reference: DOI: 10.1016/j.diabres.2013.03.006 (Standard)
        """
        # NOTE: GRI was originally validated for non-pregnant adults.
        # The use of pregnancy-specific targets here is experimental and NOT clinically validated.

        if pregnancy:
            vlow_threshold = 55
            low_range = (55, 63)
            high_range = (140, 250)
            vhigh_threshold = 250
        else:
            vlow_threshold = 54
            low_range = (54, 70)
            high_range = (180, 250)
            vhigh_threshold = 250

        vlow = self.TBR(vlow_threshold)
        low = self.calculate_time_in_range(*low_range)
        vhigh = self.TAR(vhigh_threshold)
        high = self.calculate_time_in_range(*high_range)

        gri = (3.0 * vlow) + (2.4 * low) + (1.6 * vhigh) + (0.8 * high)

        hypo_component = vlow + (0.8 * low)
        hyper_component = vhigh + (0.5 * high)

        tir = 100 - (vlow + low + vhigh + high)

        return {
            "GRI": round(gri, 2),
            "is_pregnancy": pregnancy,
            "validated": not pregnancy,
            "components": {
                "VLow": round(vlow, 2),
                "Low": round(low, 2),
                "VHigh": round(vhigh, 2),
                "High": round(high, 2),
            },
            "derived_metrics": {
                "hypo_component": round(hypo_component, 2),
                "hyper_component": round(hyper_component, 2),
                "TIR": round(tir, 2),
            },
        }

    def ADRR(self) -> dict[str, Any]:
        """
        Calculates Average Daily Risk Range (ADRR).

        ADRR is a variability measure that:
        1. Is equally sensitive to hypoglycemia and hyperglycemia.
        2. Uses logarithmic transformation to normalize the scale.

        :return: Dictionary with ADRR and related statistics.
        :reference: DOI: 10.1177/193229681300700529
        """
        daily_readings = self.data.groupby(self.data["time"].dt.date)

        def transform_bg(bg_values):
            return 1.509 * ((np.log(bg_values)) ** 1.084 - 5.381)

        daily_risks = []
        daily_hypo_risks = []
        daily_hyper_risks = []

        for _date, day_data in daily_readings:
            bg_values = day_data["glucose"].values
            transformed = transform_bg(bg_values)

            rl = np.where(transformed < 0, 10 * transformed**2, 0)
            rh = np.where(transformed > 0, 10 * transformed**2, 0)
            lr = np.max(rl) if len(rl) > 0 else 0
            hr = np.max(rh) if len(rh) > 0 else 0

            daily_risks.append(lr + hr)
            daily_hypo_risks.append(lr)
            daily_hyper_risks.append(hr)

        adrr = np.mean(daily_risks)

        if adrr < 20:
            risk_category = "Low"
        elif adrr < 40:
            risk_category = "Moderate"
        else:
            risk_category = "High"

        hypo_risk = np.mean(daily_hypo_risks)
        hyper_risk = np.mean(daily_hyper_risks)

        return {
            "adrr": round(float(adrr), 2),
            "risk_category": risk_category,
            "components": {
                "hypo_risk": round(float(hypo_risk), 2),
                "hyper_risk": round(float(hyper_risk), 2),
            },
        }
