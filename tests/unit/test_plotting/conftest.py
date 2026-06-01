"""Shared fixtures for plotting tests."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def glucose_24h_df() -> pd.DataFrame:
    """Single-day (24h) glucose DataFrame with moderate variation."""
    start = datetime(2024, 1, 1, 0, 0)
    times = [start + timedelta(minutes=5 * i) for i in range(288)]
    rng = np.random.default_rng(seed=42)
    glucose = 120 + 30 * np.sin(np.linspace(0, 4 * np.pi, 288)) + rng.normal(0, 3, 288)
    return pd.DataFrame({"time": times, "glucose": glucose})


@pytest.fixture
def glucose_7day_df() -> pd.DataFrame:
    """7-day glucose DataFrame with daily patterns."""
    start = datetime(2024, 1, 1, 0, 0)  # 2024-01-01 is a Monday
    n = 7 * 288
    times = [start + timedelta(minutes=5 * i) for i in range(n)]
    rng = np.random.default_rng(seed=123)
    glucose = (
        120
        + 30 * np.sin(np.linspace(0, 4 * np.pi, n))
        + 5 * np.sin(np.linspace(0, 14 * np.pi, n))
        + rng.normal(0, 4, n)
    )
    return pd.DataFrame({"time": times, "glucose": glucose})


@pytest.fixture
def glucose_2day_df() -> pd.DataFrame:
    """2-day glucose DataFrame for overlapping-day plots."""
    start = datetime(2024, 1, 1, 0, 0)
    n = 2 * 288
    times = [start + timedelta(minutes=5 * i) for i in range(n)]
    rng = np.random.default_rng(seed=7)
    glucose = 110 + 25 * np.sin(np.linspace(0, 4 * np.pi, n)) + rng.normal(0, 3, n)
    return pd.DataFrame({"time": times, "glucose": glucose})
