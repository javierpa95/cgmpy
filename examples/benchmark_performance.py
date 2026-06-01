import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cgmpy import ModularGlucoseMetrics

def generate_large_dataset(days=30):
    start_time = datetime(2024, 1, 1, 0, 0)
    minutes = days * 24 * 60
    times = [start_time + timedelta(minutes=5 * i) for i in range(minutes // 5)]
    glucose = np.random.normal(120, 20, len(times)).astype(int)
    return pd.DataFrame({"time": times, "glucose": glucose})

def benchmark():
    df = generate_large_dataset(days=365) # 1 year of data
    print(f"Dataset size: {len(df)} records")
    
    # Simple class that inherits from ModularGlucoseMetrics
    class BenchMetrics(ModularGlucoseMetrics):
        def __init__(self, data):
            self.data = data
            self.typical_interval = 5.0
            self.log = False

    metrics = BenchMetrics(df)
    
    # Benchmark sd_within_day_segment
    start = time.time()
    for _ in range(10):
        metrics.sd_within_day_segment("00:00", 8)
    end = time.time()
    print(f"sd_within_day_segment (10 calls): {end - start:.4f}s")
    
    # Benchmark MODD
    start = time.time()
    metrics.MODD()
    end = time.time()
    print(f"MODD: {end - start:.4f}s")
    
    # Benchmark CONGA
    start = time.time()
    metrics.CONGA()
    end = time.time()
    print(f"CONGA: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark()
