"""Glycemic quality and risk metrics (LBGI, HBGI, GRI, GRADE, ADRR, M-Value, J-Index)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from cgmpy.metrics.time_in_range import tar, tbr, tir
from cgmpy.metrics.units import MGDL_TO_MMOLL

if TYPE_CHECKING:
    pass

# GRADE categorisation boundaries (mg/dL). These are fixed by the GRADE
# definition (Hill et al., Diabet. Med. 2007; DOI 10.1111/j.1464-5491.2007.02119.x),
# NOT patient-specific targets, so they live as named constants here.
_GRADE_HYPO_MGDL = 70
_GRADE_HYPER_MGDL = 140


# ──────────────────────────────────────────────
# Pure functions
# ──────────────────────────────────────────────


def m_value(glucose: pd.Series, reference_glucose: int = 90) -> float:
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

    Args:
        glucose: Glucose values.
        reference_glucose: Reference value (default 90 mg/dL - updated from docstring default).

    Returns:
        M-Value rounded to 2 decimals.
        Returns 0.0 if no valid glucose values (prevents log10(0)).

    References:
        10.1111/j.0954-6820.1965.tb01810.x
        10.2337/db12-1396
    """
    glucose_values = glucose.values

    # Guard against zero/negative glucose (np.log10(0) = -inf)
    valid = glucose_values > 0
    if not valid.any():
        return 0.0
    glucose_values = glucose_values[valid]

    M_BS_values = np.abs(10 * np.log10(glucose_values / reference_glucose)) ** 3
    M_BS_mean = np.mean(M_BS_values)
    return round(float(M_BS_mean), 2)


def j_index(mean_glucose: float, sd_glucose: float) -> float:
    """
    Calculates J-index.

    Args:
        mean_glucose: Pre-computed mean glucose.
        sd_glucose: Pre-computed standard deviation of glucose.

    Returns:
        J-index value.

    References:
        DOI: 10.1055/s-2007-979906
    """
    return 0.001 * (mean_glucose + sd_glucose) ** 2


def grade(glucose: pd.Series, unit: str = "mg/dL") -> dict[str, float]:
    """
    Calculates GRADE (Glycaemic Risk Assessment in Diabetes Engine).

    Args:
        glucose: Glucose values.
        unit: Unit of the glucose data, either "mg/dL" or "mmol/L".

    Returns:
        Dictionary with GRADE components (score, hypo/eu/hyper percentages).

    References:
        DOI: 10.1111/j.1464-5491.2007.02119.x
    """
    if len(glucose) == 0:
        return {"grade_score": 0.0, "hypo_percent": 0.0, "eu_percent": 0.0, "hyper_percent": 0.0}

    if unit.lower() == "mg/dl":
        glucose_value = glucose.copy()
    elif unit.lower() == "mmol/l":
        glucose_value = glucose * MGDL_TO_MMOLL
    else:
        raise ValueError("The unit must be 'mg/dL' or 'mmol/L'")

    hypo_threshold = _GRADE_HYPO_MGDL
    hyper_threshold = _GRADE_HYPER_MGDL

    hypo_mask = glucose_value < hypo_threshold
    eu_mask = (glucose_value >= hypo_threshold) & (glucose_value <= hyper_threshold)
    hyper_mask = glucose_value > hyper_threshold

    glucose_values = glucose_value.values

    grade_values = np.zeros_like(glucose_values, dtype=float)

    valid_mask = (glucose_values >= 37) & (glucose_values <= 630)

    with np.errstate(invalid="ignore", divide="ignore"):
        glucose_mmol = glucose_values[valid_mask] / MGDL_TO_MMOLL
        log_log_values = np.log10(np.log10(glucose_mmol))
        grade_values[valid_mask] = 425 * (log_log_values + 0.16) ** 2

    invalid_mask = ~valid_mask | ~np.isfinite(grade_values)
    grade_values[invalid_mask] = 50

    grade_total = grade_values.sum()
    grade_hypo = grade_values[hypo_mask.values].sum()
    grade_eu = grade_values[eu_mask.values].sum()
    grade_hyper = grade_values[hyper_mask.values].sum()

    hypo_percent = (grade_hypo / grade_total) * 100 if grade_total > 0 else 0
    eu_percent = (grade_eu / grade_total) * 100 if grade_total > 0 else 0
    hyper_percent = (grade_hyper / grade_total) * 100 if grade_total > 0 else 0

    grade_score = grade_values.mean()

    return {
        "grade_score": float(grade_score),
        "hypo_percent": float(hypo_percent),
        "eu_percent": float(eu_percent),
        "hyper_percent": float(hyper_percent),
    }


def lbgi(glucose: pd.Series) -> float:
    """
    Calculates Low Blood Glucose Index (LBGI).

    Args:
        glucose: Glucose values.

    Returns:
        LBGI value.
        Returns 0.0 if no valid glucose values (prevents log(0)).

    References:
        DOI: 10.2337/db12-1396
    """
    glucose_values = glucose.values

    # Guard against zero/negative glucose (np.log(0) = -inf)
    valid = glucose_values > 0
    if not valid.any():
        return 0.0
    glucose_values = glucose_values[valid]

    f_bg = 1.509 * ((np.log(glucose_values)) ** 1.084 - 5.381)
    r_bg = 10 * f_bg**2
    rl_bg = np.where(f_bg < 0, r_bg, 0)

    return float(np.mean(rl_bg))


def hbgi(glucose: pd.Series) -> float:
    """
    Calculates High Blood Glucose Index (HBGI).

    Args:
        glucose: Glucose values.

    Returns:
        HBGI value.
        Returns 0.0 if no valid glucose values (prevents log(0)).

    References:
        DOI: 10.2337/db12-1396
    """
    glucose_values = glucose.values

    # Guard against zero/negative glucose (np.log(0) = -inf)
    valid = glucose_values > 0
    if not valid.any():
        return 0.0
    glucose_values = glucose_values[valid]

    f_bg = 1.509 * ((np.log(glucose_values)) ** 1.084 - 5.381)
    r_bg = 10 * f_bg**2
    rh_bg = np.where(f_bg > 0, r_bg, 0)

    return float(np.mean(rh_bg))


def gri(glucose: pd.Series, pregnancy: bool = False) -> dict[str, Any]:
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

    Args:
        glucose: Glucose values.
        pregnancy: If True, uses specific ranges for pregnancy.

    Returns:
        Dictionary with GRI and its components.

    References:
        DOI: 10.1016/j.diabres.2013.03.006 (Standard)
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

    vlow = tbr(glucose, vlow_threshold)
    low = tir(glucose, *low_range)
    vhigh = tar(glucose, vhigh_threshold)
    high = tir(glucose, *high_range)

    gri_value = (3.0 * vlow) + (2.4 * low) + (1.6 * vhigh) + (0.8 * high)

    hypo_component = vlow + (0.8 * low)
    hyper_component = vhigh + (0.5 * high)

    tir_value = 100 - (vlow + low + vhigh + high)

    return {
        "GRI": round(gri_value, 2),
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
            "TIR": round(tir_value, 2),
        },
    }


def adrr(glucose: pd.Series, timestamps: pd.Series) -> dict[str, Any]:
    """
    Calculates Average Daily Risk Range (ADRR).

    ADRR is a variability measure that:
    1. Is equally sensitive to hypoglycemia and hyperglycemia.
    2. Uses logarithmic transformation to normalize the scale.

    Args:
        glucose: Glucose values.
        timestamps: Corresponding timestamps.

    Returns:
        Dictionary with ADRR and related statistics.

    References:
        DOI: 10.1177/193229681300700529
    """
    daily_readings = pd.DataFrame({"time": timestamps, "glucose": glucose}).groupby(
        timestamps.dt.date
    )

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

    adrr_val = np.mean(daily_risks)

    if adrr_val < 20:
        risk_category = "Low"
    elif adrr_val < 40:
        risk_category = "Moderate"
    else:
        risk_category = "High"

    hypo_risk = np.mean(daily_hypo_risks)
    hyper_risk = np.mean(daily_hyper_risks)

    return {
        "adrr": round(float(adrr_val), 2),
        "risk_category": risk_category,
        "components": {
            "hypo_risk": round(float(hypo_risk), 2),
            "hyper_risk": round(float(hyper_risk), 2),
        },
    }
