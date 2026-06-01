# Quickstart

This page walks you through your first CGMPy analysis in **~5 minutes**.

## 1. Install

If you haven't already:

```bash
pip install cgmpy
# or, for the full development setup:
pip install cgmpy[dev,docs,agata]
```

See [Installation](installation.md) for details.

## 2. Get a CSV

CGMPy ships with a small synthetic dataset in
`tests/fixtures/data/dm.csv`. You can also
[download a sample CGM export](#sample-data) from any device and try with
that — see [Data formats](data-formats.md) for the expected columns.

## 3. Run the high-level facade

The simplest entry point is `GlucoseAnalysis`. It wraps loading, metric
computation, and plotting in one class.

```python
from pathlib import Path
from cgmpy import GlucoseAnalysis

# Adjust this to your CSV path
CSV_PATH = Path("tests/fixtures/data/dm.csv")

analysis = GlucoseAnalysis(str(CSV_PATH))

# 1. Human-readable summary
print(analysis.get_summary_string())

# 2. Programmatic access to every metric
report = analysis.get_comprehensive_report()
print(f"Time in Range: {report['time_in_range']['tir']:.1f} %")
print(f"Mean glucose:  {report['basic']['mean']:.1f} mg/dL")
print(f"GMI:           {report['basic']['gmi']:.1f} %")

# 3. Render the AGP dashboard
analysis.plot_comprehensive_dashboard()
```

You should see something like:

```
=== GlucoseAnalysis Summary ===
Records:    1 728 (24h)
Mean:       142.3 mg/dL
GMI:        6.7 %
TIR (70-180): 64.5 %
TAR (>180):    28.0 %
TBR (<70):     7.5 %
...
```

## 4. Use the modular API

If you need finer control, drop down to the modular classes.

```python
from cgmpy.data import ModularGlucoseData
from cgmpy.metrics import ModularGlucoseMetrics
from cgmpy.metrics.targets import get_targets

# Load
data = ModularGlucoseData(str(CSV_PATH))

# Filter
filtered = data.filter_by_date_range("2024-01-01", "2024-01-31")

# Compute metrics with custom cutoffs
targets = get_targets("diabetes")  # or "pregnancy"
metrics = ModularGlucoseMetrics(data, targets=targets)

print(metrics.time_in_range().tir())  # 64.5
print(metrics.variability().cv())     # 32.1
print(metrics.variability().mage())   # 95.4
```

## 5. Plot

`AGPPlotter`, `DailyPlotter`, and `StatisticalPlotter` are available
standalone, or via the facade:

```python
from cgmpy.plotting import AGPPlotter

plotter = AGPPlotter(data=data)
plotter.plot_agp(save_path="agp.png")
```

## 6. Cross-validate with AGATA

```python
from cgmpy import AgataAnalysis

agata = AgataAnalysis(data_source=str(CSV_PATH))
agata_results = agata.run()
```

See [AGATA integration](../user-guide/agata-integration.md) for details.

## Where to go next

- [User Guide → Loading Data](../user-guide/loading-data.md) — every way to ingest CGM data.
- [User Guide → Computing Metrics](../user-guide/computing-metrics.md) — the full metric reference.
- [API Reference](../api/data.md) — function-level documentation.
- [Examples](https://github.com/javierpa95/cgmpy/tree/main/examples) — runnable scripts.

## Sample data

If you do not have a CSV at hand, the repo includes three anonymized
synthetic datasets:

| File | Size | Profile |
|------|------|---------|
| `tests/fixtures/data/dm.csv` | 1 728 rows | Type 1 Diabetes |
| `tests/fixtures/data/nodm.csv` | 1 440 rows | Non-diabetic subject |
| `tests/fixtures/data/pregnancy.csv` | 4 320 rows | Pregnancy trace |

> **Never replace these with real data** — see
> [Security policy](https://github.com/javierpa95/cgmpy/blob/main/SECURITY.md).
