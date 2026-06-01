import pytest
import pandas as pd
import os
from cgmpy import ModularGlucoseData

def test_load_from_dataframe(stable_glucose_df):
    """Test initializing ModularGlucoseData from a pandas DataFrame."""
    data = ModularGlucoseData(data_source=stable_glucose_df)
    assert len(data.data) == len(stable_glucose_df)
    assert data.typical_interval > 0

def test_load_from_csv(mock_csv_file):
    """Test initializing ModularGlucoseData from a CSV file."""
    data = ModularGlucoseData(data_source=mock_csv_file)
    assert not data.data.empty
    assert len(data.data) == 288

def test_date_range_filtering(stable_glucose_df):
    """Test filtering by date range during initialization."""
    start_date = stable_glucose_df["time"].iloc[0] + pd.Timedelta(hours=2)
    end_date = stable_glucose_df["time"].iloc[0] + pd.Timedelta(hours=10)
    
    data = ModularGlucoseData(
        data_source=stable_glucose_df,
        start_date=start_date,
        end_date=end_date
    )
    
    assert data.data["time"].min() >= start_date
    assert data.data["time"].max() <= end_date
    assert len(data.data) < len(stable_glucose_df)

def test_data_info_and_quality(stable_glucose_df):
    """Test that basic info and quality metrics are available."""
    data = ModularGlucoseData(data_source=stable_glucose_df)
    info = data.info()
    assert "n_records" in info
    assert "typical_interval" in info
    
    quality = data.get_data_quality_metrics()
    assert "total_gaps" in quality

def test_export_data(stable_glucose_df, tmp_path):
    """Test exporting data to CSV and Parquet."""
    data = ModularGlucoseData(data_source=stable_glucose_df)
    
    csv_out = tmp_path / "exported.csv"
    data.to_csv(csv_out)
    assert os.path.exists(csv_out)
    
    parquet_out = tmp_path / "exported.parquet"
    data.to_parquet(parquet_out)
    assert os.path.exists(parquet_out)
