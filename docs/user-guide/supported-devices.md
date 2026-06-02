# Supported Devices

CGMPy auto-detects four CGM device export formats. This page lists them and
explains how to export data from each. If your device is not on the list, see
the [Custom format](#dont-see-your-device) section at the bottom.

## Dexcom Clarity

Dexcom exports CGM data from the **Clarity** web portal.

- **How to export:** Sign in to [clarity.dexcom.com](https://clarity.dexcom.com),
  open **Reports**, choose **Export**, and download the CSV.
- **Column names (header):**
  - `Marca temporal (AAAA-MM-DDThh:mm:ss)` (timestamp)
  - `Nivel de glucosa (mg/dL)` (glucose)
- **Loader class:** `Dexcom`

```python
from cgmpy import Dexcom

data = Dexcom("dexcom_clarity_export.csv")
```

## FreeStyle Libreview

Abbott's FreeStyle Libre / Libre 2 / Libre 3 sensors report through the
**Libreview** cloud portal.

- **How to export:** Sign in to [www.libreview.com](https://www.libreview.com),
  open the **Glucose History** report, and click **Download CSV**.
- **Column names (header, row 2):**
  - `Sello de tiempo del dispositivo` (timestamp)
  - `Historial de glucosa mg/dL` (glucose)
- **Loader class:** `Libreview`

```python
from cgmpy import Libreview

data = Libreview("libreview_history.csv")  # header=2 by default
```

## Medtronic CareLink

Medtronic pumps and sensors upload to the **CareLink** portal.

- **How to export:** Sign in to [carelink.minimed.eu](https://carelink.minimed.eu),
  open **Reports**, select a date range, and **Export** the CSV.
- **Column names (header):**
  - `Fecha y hora` (timestamp)
  - `Valor del sensor (mg/dL)` (glucose)
- **Loader class:** `MedtronicCarelink`

```python
from cgmpy import MedtronicCarelink

data = MedtronicCarelink("carelink_export.csv")
```

## Tandem t:slim (Source)

Tandem pumps upload sensor data to the **Tandem Source** platform.

- **How to export:** Sign in to [source.tandemdiabetes.com](https://source.tandemdiabetes.com),
  choose a date range, and download the **CSV** report.
- **Column names (header):**
  - `Timestamp` (timestamp)
  - `CGM Glucose Value (mg/dL)` (glucose)
- **Loader class:** `TandemDiabetes`

```python
from cgmpy import TandemDiabetes

data = TandemDiabetes("tandem_source_export.csv")
```

## Don't see your device?

CGMPy falls back to a generic loader that only needs two columns: a timestamp
and a glucose value. Pass their names explicitly with `date_col=` and
`glucose_col=`:

```python
from cgmpy.data import ModularGlucoseData

data = ModularGlucoseData(
    "my_export.csv",
    date_col="Datetime",
    glucose_col="Sensor Glucose (mg/dL)",
)
```

For a full list of accepted column aliases, see
[Data formats](../getting-started/data-formats.md). To request a built-in
loader for a new device, open an issue on GitHub.

## See also

- [Loading data](loading-data.md) — the full loading pipeline.
- [Data formats](../getting-started/data-formats.md) — required columns and
  aliases.
- [API reference → Data](../api/data.md) — every loader and method.
