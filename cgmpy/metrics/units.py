"""Glucose unit conversion utilities.

All internal computation in CGMPy uses **mg/dL**. This module provides
the tools to convert user input to mg/dL and back.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


MGDL_TO_MMOLL = 18.0156


class GlucoseUnit(Enum):
    MG_DL = "mg/dL"
    MMOLL = "mmol/L"


def to_mmol_per_l(value: float | pd.Series | np.ndarray) -> float | pd.Series | np.ndarray:
    return value / MGDL_TO_MMOLL


def to_mg_per_dl(value: float | pd.Series | np.ndarray) -> float | pd.Series | np.ndarray:
    return value * MGDL_TO_MMOLL


def convert(value: float | pd.Series, from_unit: GlucoseUnit, to_unit: GlucoseUnit):
    if from_unit == to_unit:
        return value
    if from_unit == GlucoseUnit.MG_DL:
        return to_mmol_per_l(value)
    return to_mg_per_dl(value)
