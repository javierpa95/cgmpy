"""
Glucose data validation utilities.

This module provides functions to validate that glucose readings are within
physiologically plausible ranges, including the detection of impossible
values, sensor errors, and unit-conversion mistakes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .targets import GlucoseTargets

logger = logging.getLogger(__name__)


DEFAULT_ABSOLUTE_LOW_MG_DL: float = 39.0
DEFAULT_ABSOLUTE_HIGH_MG_DL: float = 600.0


@dataclass(frozen=True)
class ValidationReport:
    """Result of a glucose data validation pass.

    Attributes:
        n_total: Total number of glucose readings inspected.
        n_valid: Number of readings inside the acceptable range.
        n_below: Number of readings below the acceptable range.
        n_above: Number of readings above the acceptable range.
        n_null: Number of null/NaN readings.
        min_glucose: Minimum observed glucose value.
        max_glucose: Maximum observed glucose value.
        low_threshold: Lower bound used for the validation (mg/dL).
        high_threshold: Upper bound used for the validation (mg/dL).
        warnings: List of human-readable warning messages.
    """

    n_total: int
    n_valid: int
    n_below: int
    n_above: int
    n_null: int
    min_glucose: float
    max_glucose: float
    low_threshold: float
    high_threshold: float
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when every non-null reading is inside the acceptable range."""
        return self.n_below == 0 and self.n_above == 0 and self.n_null == 0

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "n_below": self.n_below,
            "n_above": self.n_above,
            "n_null": self.n_null,
            "min_glucose": self.min_glucose,
            "max_glucose": self.max_glucose,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "warnings": list(self.warnings),
            "is_valid": self.is_valid,
        }


def _resolve_thresholds(
    targets: GlucoseTargets | None,
) -> tuple[float, float]:
    """Derive (low, high) thresholds from targets, or fall back to absolutes."""
    if targets is None:
        return DEFAULT_ABSOLUTE_LOW_MG_DL, DEFAULT_ABSOLUTE_HIGH_MG_DL

    low = targets.tir_low
    high = targets.tir_high

    if targets.very_low is not None:
        low = min(low, targets.very_low)
    if targets.very_high is not None:
        high = max(high, targets.very_high)

    return float(low), float(high)


def validate_glucose_range(
    data: pd.DataFrame,
    targets: GlucoseTargets | None = None,
    warn: bool = True,
) -> ValidationReport:
    """Validate that all glucose readings are inside a plausible range.

    Args:
        data: DataFrame with a `glucose` column (mg/dL).
        targets: Optional :class:`GlucoseTargets` used to derive
            clinically meaningful bounds. If ``None``, absolute physiological
            extremes (39 - 600 mg/dL) are used.
        warn: When ``True`` (default), warnings are logged at WARNING
            level. Set to ``False`` for silent validation (e.g. in tests).

    Returns:
        :class:`ValidationReport` describing the findings.
    """
    if "glucose" not in data.columns:
        raise ValueError("DataFrame must contain a 'glucose' column")

    glucose = data["glucose"]
    low, high = _resolve_thresholds(targets)

    n_total = len(glucose)
    n_null = int(glucose.isna().sum())
    valid_mask = glucose.between(low, high, inclusive="both")
    n_below = int((glucose < low).sum())
    n_above = int((glucose > high).sum())
    n_valid = int(valid_mask.sum())

    if n_total == 0:
        min_val = float("nan")
        max_val = float("nan")
    else:
        min_val = float(glucose.min(skipna=True))
        max_val = float(glucose.max(skipna=True))

    warnings: list[str] = []
    if n_below > 0:
        warnings.append(
            f"{n_below} reading(s) below {low} mg/dL "
            f"(min observed: {min_val if min_val == min_val else 'n/a'})"
        )
    if n_above > 0:
        warnings.append(
            f"{n_above} reading(s) above {high} mg/dL "
            f"(max observed: {max_val if max_val == max_val else 'n/a'})"
        )
    if n_null > 0:
        warnings.append(f"{n_null} null reading(s) found")

    if warn and warnings:
        for w in warnings:
            logger.warning("Glucose validation: %s", w)

    return ValidationReport(
        n_total=n_total,
        n_valid=n_valid,
        n_below=n_below,
        n_above=n_above,
        n_null=n_null,
        min_glucose=min_val,
        max_glucose=max_val,
        low_threshold=low,
        high_threshold=high,
        warnings=warnings,
    )
