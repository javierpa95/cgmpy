# Data Formats

CGMPy is permissive about input formats but expects a minimum schema.

## Required columns

| Column     | Type             | Description                                      |
|------------|------------------|--------------------------------------------------|
| `time`     | ISO 8601 string  | Timestamp of the glucose reading.                |
| `glucose`  | float (mg/dL)    | Glucose value. NaN is allowed; dropped by loader. |

CGMPy accepts several **aliases** for these columns and tries to detect
the format automatically:

| Canonical name | Accepted aliases                          |
|----------------|-------------------------------------------|
| `time`         | `timestamp`, `datetime`, `date`, `t`      |
| `glucose`      | `glucose_value`, `glucose_mg_dl`, `sg`    |

## Standard CSV header

The simplest format a CGMPy loader accepts is:

```csv
time,glucose
2024-01-01 00:00:00,120
2024-01-01 00:05:00,118
2024-01-01 00:10:00,121
...
```

## Device-specific formats

CGMPy has built-in loaders for common CGM vendors. They handle the
vendor's quirks (multi-line headers, BOM, US dates, etc.).

### Dexcom Clarity

```csv
Device,Serial Number,Device Timestamp,Record Type,Glucose Value (mg/dL)
Dexcom,G12345,2024-01-01 00:00:00,0,118
...
```

Loaded via `from cgmpy import Dexcom; Dexcom("dexcom_export.csv")`.

### FreeStyle Libre (Libreview)

```csv
Device,Serial Number,Device Timestamp,Record Type,Historic Glucose mg/dL
...
```

Loaded via `from cgmpy import Libreview; Libreview("libreview_export.csv", header=2)`.

The `header=2` argument tells the loader to skip the Libre multi-line
header (the first two lines are patient metadata).

### Medtronic CareLink

```python
from cgmpy import MedtronicCarelink
data = MedtronicCarelink("medtronic_export.csv")
```

### Tandem t:slim

```python
from cgmpy import TandemDiabetes
data = TandemDiabetes("tandem_export.csv")
```

## Parquet

CGMPy reads and writes Parquet natively via `pyarrow`. Parquet is
recommended for large datasets (millions of rows) because it is
~10× faster to read than CSV.

```python
from cgmpy import GlucoseData

# Write
data = GlucoseData("input.csv")
data.to_parquet("output.parquet")

# Read
data2 = GlucoseData("output.parquet")
```

## DataFrame

You can also pass a `pandas.DataFrame` directly:

```python
import pandas as pd
from cgmpy import GlucoseData

df = pd.DataFrame(
    {
        "time": pd.date_range("2024-01-01", periods=288, freq="5min"),
        "glucose": [110 + i % 30 for i in range(288)],
    }
)
data = GlucoseData(df)
```

## Units

CGMPy uses **mg/dL** internally. To convert from mmol/L:

```python
data["glucose_mg_dl"] = data["glucose_mmol_l"] * 18.0182
```

A future release may add automatic unit detection.

## Time zones

Timestamps without a timezone are assumed to be **local time** of the
device. If your export is in UTC, set it explicitly:

```python
import pandas as pd
data = pd.read_csv(
    "export.csv",
    parse_dates=["time"],
)
data["time"] = data["time"].dt.tz_localize("UTC")
```

## Data quality expectations

CGMPy's processor will:

- **Drop** rows with non-numeric glucose values.
- **Drop** duplicate (time, glucose) pairs.
- **Warn** about gaps > 60 minutes.
- **Not impute** missing values by default. Pass `impute=True` if you
  want linear interpolation over small gaps.

## See also

- [Loading Data](../user-guide/loading-data.md) — full reference.
- [Adding a new device loader](https://github.com/javierpa95/cgmpy/blob/main/CONTRIBUTING.md#adding-a-new-device-loader) — for contributors.
- [Security policy](https://github.com/javierpa95/cgmpy/blob/main/SECURITY.md) — what to do with real exports.
