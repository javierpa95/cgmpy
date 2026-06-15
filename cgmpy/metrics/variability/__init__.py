"""Glycemic variability metrics as pure functions.

This package splits the variability metrics into one file per metric family
(SD, MAGE, MODD, CONGA, Lability, Risk). Every metric is a **pure function**
that takes a glucose ``pandas.Series`` (and timestamps where relevant) and
returns a value or a dict — the same low-level API style as ``scipy.stats``::

    from cgmpy.metrics.variability import mage_baghurst, lbgi, conga

    mage = mage_baghurst(glucose_series)
    risk = lbgi(glucose_series)

The high-level :class:`~cgmpy.analysis.core.GlucoseAnalysis` facade composes
these same functions for users who work from a loaded dataset.
"""

from .conga import conga
from .lability import lability_index
from .mage import (
    mage_baghurst,
    mage_baghurst_direct_elimination,
    mage_baghurst_simplified,
    mage_baghurst_smoothing,
    mage_simple,
)
from .modd import modd
from .risk import adrr, grade, gri, hbgi, j_index, lbgi, m_value
from .sd import (
    cv_from_sd_mean,
    cv_global,
    mean_global,
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
    "adrr",
    "conga",
    "cv_from_sd_mean",
    "cv_global",
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
    "mean_global",
    "modd",
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
]
