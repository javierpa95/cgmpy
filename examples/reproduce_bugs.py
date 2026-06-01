import pandas as pd
import numpy as np
import sys
import os

# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from datetime import datetime, timedelta
from cgmpy.metrics.variability import VariabilityMetrics
from cgmpy.metrics.basic import BasicMetrics

# Create a dummy class that inherits from the metrics classes
class MockGlucoseData(VariabilityMetrics, BasicMetrics):
    def __init__(self):
        # Create dummy data: 2 days of data, 5-minute intervals
        dates = pd.date_range(start="2023-01-01", periods=288*2, freq="5min")
        glucose = np.random.normal(100, 20, size=len(dates))
        self.data = pd.DataFrame({"time": dates, "glucose": glucose})

def test_variability_bug():
    print("Testing VariabilityMetrics bug...")
    data = MockGlucoseData()
    try:
        # This calls _get_segment_data internally
        result = data.sd_within_day_segment(start_time="08:00", duration_hours=4)
        print(f"Success! Result: {result}")
    except AttributeError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {type(e).__name__}: {e}")

def test_basic_bug():
    print("\nTesting BasicMetrics bug...")
    data = MockGlucoseData()
    try:
        # This calls internal summary method
        print(str(data))
    except AttributeError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_variability_bug()
    test_basic_bug()
