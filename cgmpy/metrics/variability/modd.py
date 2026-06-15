"""MODD (Mean Of Daily Differences) metric."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass


# ──────────────────────────────────────────────
# Pure function
# ──────────────────────────────────────────────


def modd(glucose: pd.Series, timestamps: pd.Series, days: int = 1) -> dict:
    """Mean Of Daily Differences.

    Args:
        glucose: Glucose values.
        timestamps: Corresponding timestamps.
        days: Number of days apart (1-6).

    Returns:
        dict with "value", "std", "days" keys.

    Raises:
        ValueError: If days not in [1, 6].
    """
    if not 1 <= days <= 6:
        raise ValueError(f"days must be between 1 and 6, got {days}")

    df = pd.DataFrame({"time": timestamps, "glucose": glucose})
    target_delta = pd.Timedelta(days=days)
    df_indexed = df.set_index("time")
    df_shifted = df_indexed.shift(1, freq=target_delta)
    merged = df_indexed.join(df_shifted, lsuffix="_current", rsuffix="_past", how="inner")

    if merged.empty:
        return {"value": 0.0, "std": 0.0, "days": days}

    abs_diffs = (merged["glucose_current"] - merged["glucose_past"]).abs()

    return {
        "value": float(abs_diffs.mean()),
        "std": float(abs_diffs.std()) if len(abs_diffs) > 1 else 0.0,
        "days": days,
    }
