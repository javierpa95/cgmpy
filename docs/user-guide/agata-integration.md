# AGATA Integration

[AGATA](https://github.com/gcappon/agata) is a Python library for glucose
data analysis, used as the **reference implementation** by much of the
CGM research community. CGMPy ships with first-class AGATA support so
that you can:

1. **Cross-validate** your CGMPy results against AGATA.
2. **Use both** libraries side-by-side from the same Python code.
3. **Migrate** from AGATA to CGMPy (or vice versa) with minimal effort.

## Installation

AGATA is an **optional** dependency of CGMPy. Install it with:

```bash
pip install cgmpy[agata]
```

This pulls in the `agata` package and registers the integration classes.

## The `AgataAnalysis` wrapper

```python
from cgmpy import AgataAnalysis

agata = AgataAnalysis(data_source="data.csv")
results = agata.run()
# results is a dict with the same structure AGATA returns
```

`AgataAnalysis` is a thin wrapper that:

- Loads the data through the same `DataLoader` pipeline.
- Calls AGATA on the loaded data.
- Returns the AGATA result dict unchanged (so you can use the existing
  AGATA documentation and examples).

## Cross-validation

The most common use case is comparing CGMPy and AGATA on the same data:

```python
from cgmpy import AgataAnalysis, GlucoseAnalysis

agata = AgataAnalysis(data_source="data.csv").run()
cgm = GlucoseAnalysis("data.csv").get_comprehensive_report()

# Compare metric by metric
print("Mean glucose — AGATA:", agata["variability"]["mean_glucose"])
print("Mean glucose — CGMPy:", cgm["basic"]["mean"])
```

A complete side-by-side script is in
[`examples/03_agata_comparison/comparison.py`](https://github.com/javierpa95/cgmpy/blob/main/examples/03_agata_comparison/comparison.py).

## Parity testing

CGMPy's CI runs AGATA parity tests when the `agata` optional dependency
is installed. The test markers:

```python
import pytest

@pytest.mark.agata
def test_mean_matches_agata(synthetic_cgm):
    """CGMPy mean() matches AGATA's mean_glucose() to within 1e-6."""
    ...
```

Run parity tests with:

```bash
pip install cgmpy[agata]
pytest -v -m agata
```

## When to use which

| Use case                                  | Library     |
|-------------------------------------------|-------------|
| Cutting-edge / experimental metrics      | AGATA       |
| Pregnancy-specific workflows              | **CGMPy**   |
| Tight integration with pandas / numpy     | **CGMPy**   |
| Standalone, well-tested, single source    | **CGMPy**   |
| Publication-grade reproducibility          | **CGMPy** + AGATA cross-check |

## Naming and units

CGMPy mirrors AGATA's function names and units where possible, to make
migration painless. Where the names differ, the CGMPy name is documented
in the API reference.

| Concept             | AGATA                 | CGMPy                       |
|---------------------|-----------------------|-----------------------------|
| Mean glucose        | `mean_glucose`        | `basic.mean()`              |
| Standard deviation  | `std_glucose`         | `basic.std()`               |
| Coefficient of var. | `cv_glucose`          | `basic.cv()`                |
| Time in target      | `time_in_target`      | `time_in_range.tir()`       |
| LBGI                | `lbgi`                | `variability.lbgi()`        |
| GRI                 | `gri`                 | `variability.gri()`         |

## See also

- [Example: 03_agata_comparison](https://github.com/javierpa95/cgmpy/blob/main/examples/03_agata_comparison/comparison.py).
- [AGATA documentation](https://agata.readthedocs.io/).
- [AGATA GitHub](https://github.com/gcappon/agata).
