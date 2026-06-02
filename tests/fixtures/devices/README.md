# Device-format fixtures

Small, hand-validated CSV files that match the exact export format of the
four CGM devices supported by `cgmpy.data.specialized`. These are designed
for **unit tests that need a known ground truth** — every value is
deterministic, byte-for-byte reproducible, and every metric is
hand-computable.

> **Do not** add new values to these files by hand. If a value needs to
> change, regenerate them with
> [`scripts/generate_fixtures_v052.py`](../../../scripts/generate_fixtures_v052.py)
> and update the expected metric table below.

## Reference: column names

| Device             | Date column                                          | Glucose column                    |
|--------------------|------------------------------------------------------|-----------------------------------|
| Dexcom             | `Marca temporal (AAAA-MM-DDThh:mm:ss)`               | `Nivel de glucosa (mg/dL)`        |
| Libreview          | `Sello de tiempo del dispositivo`                    | `Historial de glucosa mg/dL`      |
| Medtronic CareLink | `Fecha y hora`                                       | `Valor del sensor (mg/dL)`        |
| Tandem Diabetes    | `Timestamp`                                          | `CGM Glucose Value (mg/dL)`       |

These names must stay in lockstep with
[`cgmpy/data/specialized.py`](../../../cgmpy/data/specialized.py). If the
loader changes a column name, the fixtures and this README must change
in the same commit.

## Constant-120 fixtures

| File                              | Device     | n  | Duration | Interval | Glucose | Size (B) |
|-----------------------------------|------------|----|----------|----------|---------|----------|
| `dexcom_constant_120.csv`         | Dexcom     | 288 | 24h      | 5 min    | 120     | 6 974    |
| `libreview_constant_120.csv`      | Libreview  | 288 | 24h      | 5 min    | 120     | 13 102   |
| `medtronic_constant_120.csv`      | Medtronic  | 288 | 24h      | 5 min    | 120     | 6 950    |
| `tandem_constant_120.csv`         | Tandem     | 288 | 24h      | 5 min    | 120     | 6 948    |

Timestamps start at `2024-01-01 00:00:00` and step +5 min. Each file is
**288 readings of constant 120 mg/dL**, so every metric has a known,
trivial expected value.

### Ground truth (applies to all four constant fixtures)

| Metric                | Expected value        | Why |
|-----------------------|-----------------------|-----|
| `n`                   | 288                   | 24h × 12 readings/h |
| `mean`                | 120.0 mg/dL           | constant signal |
| `median`              | 120.0 mg/dL           | constant signal |
| `sd` (sample, ddof=1) | 0.0 mg/dL             | no variance |
| `cv`                  | 0.0 %                 | sd/mean = 0 |
| `min`, `max`          | 120, 120              | constant |
| `TIR` (70-180)        | 100.0 %               | 120 ∈ [70, 180] |
| `TAR_total` (>180)    | 0.0 %                 | 120 ≤ 180 |
| `TBR_total` (<70)     | 0.0 %                 | 120 ≥ 70 |
| `GMI`                 | 6.18 %                | `round(3.31 + 0.02392·120, 2)` — see note below |
| `data_completeness`   | 100 %                 | 288 expected, 288 real, 5-min interval |

**GMI note.** CGMPy's `BasicMetrics.gmi()`
([`cgmpy/metrics/basic.py:82`](../../../cgmpy/metrics/basic.py)) uses
**GMI = round(3.31 + 0.02392·mean, 2)** (Beck et al. 2019,
DOI:10.2337/dc18-1581). This is the **Glucose Management Indicator**,
not the older **eA1c = (mean + 46.7) / 28.7**. For mean = 120 mg/dL the
two formulas give:

- GMI (this library)  = `round(3.31 + 0.02392·120, 2)` = **6.18 %**
- eA1c (DCCT/ADAG)    = `(120 + 46.7) / 28.7`          = **5.81 %**

Tests must assert against **6.18**, not 5.81.

### Format-specific notes

- **Dexcom**: timestamps use the ISO `YYYY-MM-DDTHH:MM:SS` form (note
  the `T` separator, no space).
- **Libreview**: file has **2 banner rows** above the real header (so
  the loader must use `header=2`). Date format is `DD-MM-YYYY HH:MM`
  (no seconds). Auxiliary columns (`Dispositivo`, `Numero de serie`,
  `Tipo de registro`) are included to match a real Libreview export.
- **Medtronic CareLink**: timestamps use `YYYY-MM-DD HH:MM:SS`.
- **Tandem**: timestamps use `YYYY-MM-DD HH:MM:SS`; column names are
  English.

## Edge cases (`edge_cases/`)

| File                       | Format  | n  | Description |
|----------------------------|---------|----|-------------|
| `empty.csv`                | Dexcom  | 0  | Header only, no data rows. Exercises the empty-input path. |
| `single_row.csv`           | Dexcom  | 1  | Exactly 1 data row at `100` mg/dL. Exercises the n=1 edge case. |
| `all_nan_glucose.csv`      | Dexcom  | 12 | Glucose column is entirely NaN. Exercises NaN handling. |
| `with_gap.csv`             | Dexcom  | 12 | 12 rows with a **30-minute gap** between 00:25 and 00:55. 6 rows from 00:00-00:25, 6 rows from 00:55-01:20. |
| `out_of_range_high.csv`    | Dexcom  | 12 | 11 rows of 120 mg/dL, **one value of 700 mg/dL** at 00:30 (above the 600 mg/dL ceiling). |
| `out_of_range_low.csv`     | Dexcom  | 12 | 11 rows of 120 mg/dL, **one value of 20 mg/dL** at 00:30 (below the 39 mg/dL floor). |

All edge cases use the **Dexcom** column format because that is the
loader most heavily tested. Edge cases for other device formats can be
added here on demand.

## Cross-references

- **Clinical ground truth** for hand-computed metrics (mean, sd, cv,
  GMI, TIR, TAR, TBR) lives in
  [`tests/clinical/test_basic_metrics_reference.py`](../../clinical/test_basic_metrics_reference.py).
- **In-memory fixtures** (DataFrames, not files) for unit tests that
  do not need a real loader are in
  [`tests/conftest.py`](../../conftest.py).
- The fixtures in [`tests/fixtures/data/`](../data/) (`dm.csv`,
  `nodm.csv`, `pregnancy.csv`) are kept for legacy tests; they are
  large and use stochastic noise, so they are **not** suitable for
  asserting on exact metric values.
