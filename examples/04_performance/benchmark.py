"""Performance benchmark: how long does each metric take on large inputs?

Generates a synthetic 30-day CGM trace at 5-minute intervals (~8 640 samples)
and times every metric in `cgmpy.metrics`.

Run from the project root:

    python examples/04_performance/benchmark.py

Reports timings to stdout. Useful for spotting regressions when you change
the implementation of a metric.

The mixin design of CGMPy means a metric method is called directly on the
data class (e.g. ``gd.mean()``), not on a sub-namespace. We use
``GlucoseMetrics`` which combines ``ModularGlucoseData`` with
``ModularGlucoseMetrics`` so every metric is reachable as a flat method.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from cgmpy import GlucoseMetrics


def _make_synthetic(n_days: int = 30, sample_minutes: int = 5) -> pd.DataFrame:
    """Generate a synthetic CGM DataFrame with realistic noise + meal spikes."""
    n = (24 * 60 // sample_minutes) * n_days
    start = datetime(2024, 1, 1, 0, 0)
    times = [start + timedelta(minutes=sample_minutes * i) for i in range(n)]
    rng = np.random.default_rng(42)
    base = 110.0
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
    fn()
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"  {label:30s} {elapsed_ms:8.2f} ms")
    return elapsed_ms


def main() -> None:
    print("Generating synthetic 30-day CGM at 5-min intervals...")
    df = _make_synthetic()
    print(f"  {len(df):,} samples\n")

    gm = GlucoseMetrics(df)

    print("Timing each metric:")
    _time_it("basic.mean", lambda: gm.mean())
    _time_it("basic.median", lambda: gm.median())
    _time_it("basic.gmi", lambda: gm.gmi())
    _time_it("basic.std", lambda: gm.sd())
    _time_it("basic.cv", lambda: gm.cv())
    _time_it("time_in_range.tir", lambda: gm.TIR())
    _time_it("time_in_range.tir_tight", lambda: gm.TIR_tight())
    _time_it("variability.sd_total", lambda: gm.sd_total())
    _time_it("variability.sd_within_day", lambda: gm.sd_within_day())
    _time_it("variability.mage", lambda: gm.MAGE())
    _time_it("variability.mage_baghurst", lambda: gm.MAGE_Baghurst())
    _time_it("variability.modd", lambda: gm.MODD())
    _time_it("variability.conga_4h", lambda: gm.CONGA(hours=4))
    _time_it("risk.lbgi", lambda: gm.LBGI())
    _time_it("risk.hbgi", lambda: gm.HBGI())
    _time_it("risk.grade", lambda: gm.GRADE())


if __name__ == "__main__":
    main()
