"""Performance benchmark: how long does each metric take on large inputs?

Generates a synthetic 30-day CGM trace at 5-minute intervals (~8 640 samples)
and times every metric in `cgmpy.metrics`.

Run from the project root:

    python examples/04_performance/benchmark.py

Reports timings to stdout. Useful for spotting regressions when you change
the implementation of a metric.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from cgmpy import ModularGlucoseData
from cgmpy.metrics import ModularGlucoseMetrics


def _make_synthetic(n_days: int = 30, sample_minutes: int = 5) -> pd.DataFrame:
    """Generate a synthetic CGM DataFrame with realistic noise + meal spikes."""
    n = (24 * 60 // sample_minutes) * n_days
    start = datetime(2024, 1, 1, 0, 0)
    times = [start + timedelta(minutes=sample_minutes * i) for i in range(n)]
    rng = np.random.default_rng(42)
    base = 110.0
    # Add three meal spikes per day at 8, 13, 20h
    glucose = np.full(n, base, dtype=float)
    for day in range(n_days):
        for hour, peak in ((8, 60), (13, 70), (20, 50)):
            i = day * 288 + hour * 12
            if i + 24 < n:
                glucose[i : i + 24] += peak * np.exp(-np.linspace(0, 3, 24))
    glucose += rng.normal(0, 5, n)
    return pd.DataFrame({"time": times, "glucose": np.round(glucose, 1)})


def _time_it(label: str, fn) -> float:
    start = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"  {label:30s} {elapsed_ms:8.2f} ms")
    return elapsed_ms


def main() -> None:
    print("Generating synthetic 30-day CGM at 5-min intervals...")
    df = _make_synthetic()
    print(f"  {len(df):,} samples\n")

    data = ModularGlucoseData(df)
    metrics = ModularGlucoseMetrics(data)

    print("Timing each metric:")
    _time_it("basic.mean", lambda: metrics.basic().mean())
    _time_it("basic.median", lambda: metrics.basic().median())
    _time_it("basic.gmi", lambda: metrics.basic().gmi())
    _time_it("basic.std", lambda: metrics.basic().std())
    _time_it("time_in_range.tir", lambda: metrics.time_in_range().tir())
    _time_it("variability.cv", lambda: metrics.variability().cv())
    _time_it("variability.mage", lambda: metrics.variability().mage())
    _time_it("variability.modd", lambda: metrics.variability().modd())
    _time_it("variability.lbgi_hbgi", lambda: metrics.variability().lbgi_hbgi())


if __name__ == "__main__":
    main()
