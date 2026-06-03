# Loading Data

CGMPy provides several ways to ingest CGM data, from a one-line helper to
fully controlled per-device loaders. See
[Supported devices](supported-devices.md) for the list of auto-detected
formats.

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

### `GlucoseData`

```python
from cgmpy import GlucoseData

# From CSV
data = GlucoseData("data.csv")

# From Parquet (fast)
data = GlucoseData("data.parquet")

# From a DataFrame
import pandas as pd
data = GlucoseData(pd.read_csv("data.csv"))
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
data = GlucoseData("data.csv")

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

## Troubleshooting

All CGMPy-raised errors inherit from `cgmpy.errors.CGMPyError`. The
subclasses below are the ones you will encounter most often in the loading
path.

### `ColumnNotFoundError`

Your CSV is missing the time column (or whichever column CGMPy expected for
the auto-detected device). Either it is not a recognized device export, or
you need to pass `date_col=...` and `glucose_col=...` to `GlucoseData`.

```python
from cgmpy import CGMPyError, ColumnNotFoundError, GlucoseData

try:
    data = GlucoseData("export.csv")
except ColumnNotFoundError as e:
    print(f"Missing: {e.column}; available: {e.available}")
    data = GlucoseData(
        "export.csv",
        date_col="Datetime",
        glucose_col="Sensor Glucose (mg/dL)",
    )
```

### `InvalidCSVFormatError`

The CSV could not be parsed. Try specifying `delimiter=';'` (some locales
export with semicolons), or check that the file is not corrupted / not
actually a text file saved with the `.csv` extension.

```python
from cgmpy import InvalidCSVFormatError, GlucoseData

try:
    data = GlucoseData("export.csv", delimiter=";")
except InvalidCSVFormatError as e:
    print(f"Cannot parse {e.file_path}: {e.reason}")
```

### `DeviceDetectionError`

We could not auto-detect the device. The exception message lists the
columns that were found. Pass the loader class explicitly
(e.g. `Dexcom("file.csv")`), or use `GlucoseData` with
`date_col=` / `glucose_col=`.

```python
from cgmpy import Dexcom, DeviceDetectionError, GlucoseData

try:
    data = GlucoseData("export.csv")
except DeviceDetectionError as e:
    print(f"Found columns: {e.columns_found}")
    data = Dexcom("export.csv")  # try Dexcom explicitly
```

### `EmptyDataError`

No rows remained after filtering. Check that your `start_date` /
`end_date` range actually overlaps with the file, and that the file is
not itself empty.

```python
from cgmpy import EmptyDataError

try:
    jan = data.filter_by_date_range("2024-01-01", "2024-01-31")
except EmptyDataError as e:
    print(f"Empty after: {e.context}")
```

### `AgataNotInstalledError`

You need `py_agata` for the AGATA comparison. Install the optional
`[agata]` extra:

```bash
pip install 'cgmpy[agata]'
```

```python
from cgmpy import AgataNotInstalledError

try:
    analysis.run_agata_comparison()
except AgataNotInstalledError:
    print("Run: pip install 'cgmpy[agata]'")
```

For the full error hierarchy, see `cgmpy.errors` in the
[API reference](../api/data.md#exceptions).

## See also

- [Supported devices](supported-devices.md) — auto-detected formats.
- [Data formats](../getting-started/data-formats.md) — required columns and aliases.
- [API reference → Data](../api/data.md) — every class and method.
