"""
Time in Range metrics module for glucose data.

This module contains metrics related to time in different ranges:
- Time in Range (TIR)
- Time Above Range (TAR)
- Time Below Range (TBR)
- Specific time statistics
"""

import pandas as pd

# =============================================================================
# Pure functions
# =============================================================================


def tir(glucose: pd.Series, low: float = 70, high: float = 180) -> float:
    """Time In Range (TIR): percentage of readings between low and high.

    Args:
        glucose: Glucose values in mg/dL.
        low: Lower bound (default 70 mg/dL).
        high: Upper bound (default 180 mg/dL).

    Returns:
        Percentage of readings in range [0, 100]. Returns 0.0 if glucose is empty.

    Reference:
        Battelino et al. 2019, DOI: 10.2337/dci19-0028
    """
    if len(glucose) == 0:
        return 0.0
    in_range = ((glucose >= low) & (glucose <= high)).sum()
    return (in_range / len(glucose)) * 100


def tar(glucose: pd.Series, threshold: float) -> float:
    """Time Above Range (TAR): percentage of readings above threshold.

    Args:
        glucose: Glucose values in mg/dL.
        threshold: Upper threshold (e.g. 180 or 250 mg/dL).

    Returns:
        Percentage of readings above threshold [0, 100]. Returns 0.0 if glucose
        is empty.
    """
    if len(glucose) == 0:
        return 0.0
    above = (glucose > threshold).sum()
    return (above / len(glucose)) * 100


def tbr(glucose: pd.Series, threshold: float) -> float:
    """Time Below Range (TBR): percentage of readings below threshold.

    Args:
        glucose: Glucose values in mg/dL.
        threshold: Lower threshold (e.g. 70 or 54 mg/dL).

    Returns:
        Percentage of readings below threshold [0, 100]. Returns 0.0 if glucose
        is empty.
    """
    if len(glucose) == 0:
        return 0.0
    below = (glucose < threshold).sum()
    return (below / len(glucose)) * 100


def data_completeness(
    glucose: pd.Series,
    timestamps: pd.Series,
    expected_interval_minutes: float = 5,
) -> float:
    """Calculate the percentage of expected readings that are present.

    Args:
        glucose: Glucose values.
        timestamps: Corresponding timestamps.
        expected_interval_minutes: Expected interval between readings in minutes
            (default 5).

    Returns:
        Percentage of expected data [0, 100]. Returns 0.0 if no data.
    """
    if len(timestamps) == 0:
        return 0.0
    actual_minutes = (timestamps.max() - timestamps.min()).total_seconds() / 60.0
    expected_readings = actual_minutes / expected_interval_minutes + 1
    return (len(glucose) / expected_readings) * 100


def time_in_range(glucose: pd.Series, low: float, high: float) -> float:
    """Alias for :func:`tir`. Returns percentage of values in [low, high].

    Kept for backward compatibility with existing code that calls this name.
    """
    return tir(glucose, low, high)



