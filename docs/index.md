# CGMPy

Welcome to **CGMPy** — a modular Python library for **Continuous Glucose
Monitoring (CGM)** data analysis.

Whether you are a clinician tracking a patient's glycemic control, a
researcher running cohort analyses, or a developer building diabetes
software, CGMPy gives you a clean, well-tested foundation.

## ✨ What can CGMPy do?

- **Load** CGM data from CSV, Parquet, or in-memory `pandas.DataFrame`.
  Auto-detect device format (Dexcom, FreeStyle Libre, Tandem, Medtronic).
- **Validate** glucose time series: range checks, gap detection, interval
  regularity.
- **Compute** the full set of consensus clinical metrics:
  TIR, TAR, TBR, GMI, CV, MAGE, MODD, CONGA, J-Index, LBGI, HBGI, GRI, ADRR.
- **Analyze pregnancy** with tighter cutoffs and the `PregnancyAnalysis`
  class.
- **Visualize** Ambulatory Glucose Profiles (AGP), daily traces, and
  statistical dashboards.
- **Compare** results against the [AGATA](https://github.com/gcappon/agata)
  reference library.

## 🚀 Five lines to a full report

```python
from cgmpy import GlucoseAnalysis

analysis = GlucoseAnalysis("my_cgm.csv")
print(analysis.get_summary_string())
analysis.plot_comprehensive_dashboard()
```

## 📚 Where to start

| If you want to …                              | Read                                           |
|-----------------------------------------------|------------------------------------------------|
| Install CGMPy                                 | [Getting Started → Installation](getting-started/installation.md) |
| See a complete example                        | [Getting Started → Quickstart](getting-started/quickstart.md) |
| Understand which data formats are supported   | [Getting Started → Data Formats](getting-started/data-formats.md) |
| Compute specific clinical metrics             | [User Guide → Computing Metrics](user-guide/computing-metrics.md) |
| Render plots and dashboards                   | [User Guide → Visualization](user-guide/visualization.md) |
| Run pregnancy-specific analysis               | [User Guide → Pregnancy Analysis](user-guide/pregnancy-analysis.md) |
| Cross-validate with AGATA                     | [User Guide → AGATA Integration](user-guide/agata-integration.md) |
| Look up a function signature                  | [API Reference](api/data.md)                   |
| Contribute to CGMPy                           | [Contributing](https://github.com/javierpa95/cgmpy/blob/main/CONTRIBUTING.md) |

## 🛡️ Clinical disclaimer

CGMPy is a **research and analysis tool**. It is **not a medical device** and
must not be used as a substitute for professional medical advice. Always
validate clinical interpretations with a qualified healthcare provider.

## 📜 License

CGMPy is released under the **MIT License**. See
[LICENSE](https://github.com/javierpa95/cgmpy/blob/main/LICENSE) for details.
