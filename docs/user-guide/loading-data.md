# Loading Data

CGMPy provides several ways to ingest CGM data, from a one-line helper to
fully controlled per-device loaders.

## One-line loader

The `GlucoseAnalysis` facade takes a path and does everything:

```python
from cgmpy import GlucoseAnalysis

analysis = GlucoseAnalysis("data.csv")
```

Internally it:

1. Detects the file format (CSV / Parquet / DataFrame).
2. Picks the right `DataLoader` (generic or device-specific).
3. Processes and validates the data (numeric coercion, dedup, gap detection).
4. Computes every metric.
5. Makes the data available for plotting.

## Modular loaders

If you want to control the pipeline step by step, use the modular classes.

### `ModularGlucoseData`

```python
from cgmpy.data import ModularGlucoseData

# From CSV
data = ModularGlucoseData("data.csv")

# From Parquet (fast)
data = ModularGlucoseData("data.parquet")

# From a DataFrame
import pandas as pd
data = ModularGlucoseData(pd.read_csv("data.csv"))
```

### `DataLoader` (low-level)

For maximum control, use `DataLoader` directly:

```python
from cgmpy.data import DataLoader, DataProcessor, DataAnalyzer

loader = DataLoader()
processor = DataProcessor()
analyzer = DataAnalyzer()

raw = loader.load_from_csv("data.csv", time_col="time", glucose_col="glucose")
processed, diffs = processor.process_data(raw, "time", "glucose")
interval = analyzer.calculate_typical_interval(diffs)
print(f"Typical interval: {interval} min")
```

## Device-specific loaders

| Device              | Class                 | Notes                                          |
|---------------------|-----------------------|------------------------------------------------|
| Dexcom Clarity      | `Dexcom`              | Auto-detected from header.                     |
| FreeStyle Libre     | `Libreview`           | Pass `header=2` to skip metadata.              |
| Medtronic CareLink  | `MedtronicCarelink`   |                                                |
| Tandem t:slim       | `TandemDiabetes`      |                                                |

Auto-detection:

```python
from cgmpy.data import detect_device_type, create_specialized_loader

device = detect_device_type("data.csv")  # 'dexcom', 'libreview', ...
loader = create_specialized_loader("data.csv")
```

## Filtering

```python
data = ModularGlucoseData("data.csv")

# By date range
jan = data.filter_by_date_range("2024-01-01", "2024-01-31")

# By glucose range (e.g., physiological sanity)
normal = data.filter_by_glucose_range(40, 400)
```

## Inspection

```python
print(data)  # __str__ shows a summary

info = data.info(include_disconnections=True)
print(info["n_records"], info["typical_interval"])

quality = data.get_data_quality_metrics()
print(quality["total_gaps"], quality["max_gap_hours"])
```

## Exporting

```python
data.to_parquet("optimized.parquet")  # recommended for big data
data.to_csv("cleaned.csv")
data.to_excel("report.xlsx")
```

The Parquet writer preserves dtypes, uses snappy compression, and is
~10× faster to re-read than CSV.

## See also

- [Data formats](../getting-started/data-formats.md) — required columns and aliases.
- [API reference → Data](../api/data.md) — every class and method.
