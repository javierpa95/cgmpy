import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def stable_glucose_df():
    """Generates a DataFrame with very stable glucose levels around 100 mg/dL."""
    start_time = datetime(2024, 1, 1, 12, 0)
    times = [start_time + timedelta(minutes=5 * i) for i in range(288)]  # 24 hours
    # Use integers to represent typical CGM data
    glucose = [100 for _ in range(288)]
    return pd.DataFrame({"time": times, "glucose": glucose})

@pytest.fixture
def variable_glucose_df():
    """Generates a highly variable glucose DataFrame with highs and lows."""
    start_time = datetime(2024, 1, 1, 12, 0)
    times = [start_time + timedelta(minutes=5 * i) for i in range(288)]
    # Create a pattern with significant swings
    glucose = 120 + 50 * np.sin(np.linspace(0, 4 * np.pi, 288)) + np.random.normal(0, 5, 288)
    return pd.DataFrame({"time": times, "glucose": glucose})

@pytest.fixture
def glucose_df_with_gaps():
    """Generates a glucose DataFrame with a 2-hour gap in the middle."""
    start_time = datetime(2024, 1, 1, 12, 0)
    times_part1 = [start_time + timedelta(minutes=5 * i) for i in range(100)]
    times_part2 = [start_time + timedelta(minutes=5 * i) for i in range(124, 288)]
    
    times = times_part1 + times_part2
    glucose = [110] * len(times)
    return pd.DataFrame({"time": times, "glucose": glucose})

@pytest.fixture
def mock_csv_file(tmp_path, stable_glucose_df):
    """Creates a temporary CSV file with glucose data."""
    csv_file = tmp_path / "test_glucose.csv"
    stable_glucose_df.to_csv(csv_file, index=False)
    return str(csv_file)
