"""
Glucose metrics and statistics module.

This module contains the functions to calculate:
- Basic metrics: mean, median, percentiles, GMI
- Time in range: TIR, TAR, TBR
- Variability: SD, CV, MAGE, MODD, CONGA
- Advanced metrics: GRADE, GRI, M-Value, J-Index
"""

from .targets import GlucoseTargets, get_targets
from .validation import ValidationReport, validate_glucose_range


__all__ = [
    "ValidationReport",
    "validate_glucose_range",
]
