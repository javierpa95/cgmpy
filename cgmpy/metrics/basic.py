"""
Basic metrics module for glucose data.

This module contains fundamental metrics for glucose data analysis:
- Basic descriptive statistics (mean, median, percentiles)
- Standard deviation and coefficient of variation
- Glucose Management Index (GMI)
- Distribution analysis
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def mean(glucose: pd.Series) -> float:
    """Mean glucose value.

    Args:
        glucose: Glucose values in mg/dL.

    Returns:
        Mean glucose in mg/dL.
    """
    return glucose.mean()


def median(glucose: pd.Series) -> float:
    """Median glucose value.

    Args:
        glucose: Glucose values in mg/dL.

    Returns:
        Median glucose in mg/dL.
    """
    return glucose.median()


def sd(glucose: pd.Series) -> float:
    """Standard deviation of glucose values.

    Args:
        glucose: Glucose values in mg/dL.

    Returns:
        Standard deviation in mg/dL.
    """
    return glucose.std()


def cv(glucose: pd.Series) -> float:
    """Coefficient of variation (CV).

    Returns 0.0 if mean is 0 to prevent division by zero.

    Args:
        glucose: Glucose values in mg/dL.

    Returns:
        Coefficient of variation in percentage.
    """
    mean_val = mean(glucose)
    if mean_val == 0:
        return 0.0
    return (sd(glucose) / mean_val) * 100


def gmi(mean_glucose: float) -> float:
    """Glucose Management Index (GMI) — an HbA1c estimation based on CGM data.

    Reference: DOI: 10.2337/dc18-1581

    Args:
        mean_glucose: Mean glucose in mg/dL.

    Returns:
        GMI (HbA1c estimation in %).
    """
    return round(3.31 + (0.02392 * mean_glucose), 2)


def percentile(glucose: pd.Series, p: float) -> float:
    """Calculate a glucose percentile.

    Args:
        glucose: Glucose values in mg/dL.
        p: Percentile to calculate (0-100).

    Returns:
        Percentile value in mg/dL.
    """
    return glucose.quantile(p / 100)
