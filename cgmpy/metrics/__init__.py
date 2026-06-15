"""Glucose metrics as composable pure functions.

This package exposes the full low-level metric API. Every function takes a
glucose ``pandas.Series`` (and timestamps where relevant) and returns a value
or a dict, so it can be used standalone — the same style as ``scipy.stats`` or
``sklearn.metrics``::

    from cgmpy.metrics import tir, gmi, mage_baghurst, lbgi, get_targets

    tir(glucose_series, low=70, high=180)
    mage_baghurst(glucose_series)

The high-level :class:`~cgmpy.analysis.core.GlucoseAnalysis` facade is the
convenience layer that composes these functions for a loaded dataset.

Submodules:
    - ``basic``: mean, median, sd, cv, gmi, percentile
    - ``time_in_range``: tir, tar, tbr, data_completeness
    - ``variability``: SD/CV, MAGE, MODD, CONGA, Lability, Risk (LBGI, HBGI, GRI, ...)
    - ``units``: glucose unit conversion (mg/dL <-> mmol/L)
    - ``targets``: ``GlucoseTargets`` and the ``get_targets`` factory
    - ``validation``: glucose-range validation
"""

from .basic import cv, gmi, mean, median, percentile, sd
from .targets import GlucoseTargets, get_targets
from .time_in_range import data_completeness, tar, tbr, time_in_range, tir
from .units import GlucoseUnit, convert, to_mg_per_dl, to_mmol_per_l
from .validation import ValidationReport, validate_glucose_range
from .variability import (
    adrr,
    conga,
    cv_from_sd_mean,
    cv_global,
    grade,
    gri,
    hbgi,
    j_index,
    lability_index,
    lbgi,
    m_value,
    mage_baghurst,
    mage_baghurst_direct_elimination,
    mage_baghurst_simplified,
    mage_baghurst_smoothing,
    mage_simple,
    mean_global,
    modd,
    sd_between_timepoints,
    sd_daily_mean,
    sd_global,
    sd_interaction,
    sd_same_timepoint,
    sd_same_timepoint_adjusted,
    sd_segment,
    sd_within_day,
    sd_within_series,
    sdw,
)

__all__ = [
    "GlucoseTargets",
    "GlucoseUnit",
    "ValidationReport",
    "adrr",
    "conga",
    "convert",
    "cv",
    "cv_from_sd_mean",
    "cv_global",
    "data_completeness",
    "get_targets",
    "gmi",
    "grade",
    "gri",
    "hbgi",
    "j_index",
    "lability_index",
    "lbgi",
    "m_value",
    "mage_baghurst",
    "mage_baghurst_direct_elimination",
    "mage_baghurst_simplified",
    "mage_baghurst_smoothing",
    "mage_simple",
    "mean",
    "mean_global",
    "median",
    "modd",
    "percentile",
    "sd",
    "sd_between_timepoints",
    "sd_daily_mean",
    "sd_global",
    "sd_interaction",
    "sd_same_timepoint",
    "sd_same_timepoint_adjusted",
    "sd_segment",
    "sd_within_day",
    "sd_within_series",
    "sdw",
    "tar",
    "tbr",
    "time_in_range",
    "tir",
    "to_mg_per_dl",
    "to_mmol_per_l",
    "validate_glucose_range",
]
