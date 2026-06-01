# CGMPy — Continuous Glucose Monitoring Analysis

[![CI](https://github.com/javierpa95/cgmpy/actions/workflows/ci.yml/badge.svg)](https://github.com/javierpa95/cgmpy/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/cgmpy.svg)](https://pypi.org/project/cgmpy/)
[![Python versions](https://img.shields.io/pypi/pyversions/cgmpy.svg)](https://pypi.org/project/cgmpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](docs/index.md)
[![Status: Beta](https://img.shields.io/badge/status-beta-orange.svg)](https://pypi.org/project/cgmpy/)

> A modular Python library for analyzing **Continuous Glucose Monitoring (CGM)** data, with a strong focus on clinical metrics, pregnancy-specific workflows, and reproducible research.

---

## What is CGMPy?

CGMPy provides a clean, type-hinted Python API for:

- **Loading** CGM data from CSV, Parquet, or in-memory `pandas.DataFrame` with automatic device detection (Dexcom Clarity, FreeStyle Libre / Libreview, Tandem, Medtronic CareLink).
- **Processing & validating** glucose time series (glucometric ranges, gap detection, interval normalization).
- **Computing clinical metrics** following international consensus:
  - **Basic statistics**: mean, median, GMI, glucose CV.
  - **Time in range (TIR)**: TIR, TAR, TBR with customizable targets (diabetes vs. pregnancy).
  - **Glycemic variability**: SD, CV, MAGE, MODD, CONGA, J-Index, LBGI, HBGI, GRI.
  - **Pregnancy-specific metrics** for gestational diabetes.
- **Visualizing** ambulatory glucose profiles (AGP), daily trends, and statistical summaries.
- **Comparing** results against the [AGATA](https://github.com/gcappon/agata) reference implementation for validation.

The library is **modular**: every component (loader, processor, analyzer, exporter, plotter, metrics) can be used independently or composed through the high-level `GlucoseAnalysis` facade.

---

## Installation

```bash
# Stable release
pip install cgmpy

# With AGATA integration (optional, requires agata)
pip install cgmpy[agata]

# With documentation / development tools
pip install cgmpy[dev,docs]
```

CGMPy requires **Python ≥ 3.10**.

### From source

```bash
git clone https://github.com/javierpa95/cgmpy.git
cd cgmpy
pip install -e .[dev]
```

---

## Quickstart

```python
from cgmpy import GlucoseAnalysis

# One-line analysis
analysis = GlucoseAnalysis("path/to/glucose.csv")
report = analysis.get_comprehensive_report()
print(f"Time in Range: {report['time_in_range']['tir']}%")

# Generate the standard Ambulatory Glucose Profile
analysis.plot_comprehensive_dashboard()
```

### Lower-level modular API

```python
from cgmpy.data import ModularGlucoseData
from cgmpy.metrics import ModularGlucoseMetrics
from cgmpy.metrics.targets import get_targets

data = ModularGlucoseData("cgm_data.csv")
data.filter_by_date_range("2024-01-01", "2024-01-31")

# Compute metrics with pregnancy-specific cutoffs
targets = get_targets("pregnancy")
metrics = ModularGlucoseMetrics(data, targets=targets)
print(metrics.time_in_range())
print(metrics.variability().mage())
```

---

## Features

| Domain              | Capabilities                                                                              |
|---------------------|-------------------------------------------------------------------------------------------|
| **Data ingestion**  | CSV, Parquet, DataFrame. Auto delimiter, header detection, device-specific loaders.      |
| **Validation**      | Glucose range check, time interval regularity, gap analysis, data quality scoring.        |
| **Basic metrics**   | Mean, median, GMI, SD, IQR, % time in any custom range.                                  |
| **Time in range**   | TIR, TAR (level 1/2), TBR (level 1/2) with international consensus cutoffs.              |
| **Variability**     | CV, MAGE, MODD, CONGA, J-Index, LBGI, HBGI, GRI, MAG, ADRR.                               |
| **Pregnancy**       | Dedicated `GestationalDiabetes` metrics, pregnancy glucose targets (63–140 mg/dL).       |
| **Visualization**   | AGP (Ambulatory Glucose Profile), daily traces, statistical dashboards.                   |
| **AGATA parity**    | `AgataAnalysis` wrapper for side-by-side comparison with the AGATA reference.             |
| **Export**          | To Parquet (optimized) / CSV / Excel, with deduplication and type coercion.               |

---

## Documentation

- [Getting Started](docs/getting-started/installation.md)
- [User Guide](docs/user-guide/loading-data.md)
- [API Reference](docs/api/data.md)
- [Development Guide](docs/development/setup.md)
- [Roadmap](ROADMAP.md)
- [Changelog](CHANGELOG.md)

---

## Clinical & Research Use

CGMPy is built to support clinical research workflows in diabetes and gestational diabetes. If you use CGMPy in a publication, please cite the version you used (see `CITATION.cff`).

> **Important — Not a medical device.** CGMPy is a research and analysis tool. It is not a medical device and must not be used as a substitute for professional medical advice. Always validate clinical interpretations with a qualified healthcare provider.

---

## Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs, proposing features, and submitting pull requests. By participating, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

Quick links:

- [Good first issues](https://github.com/javierpa95/cgmpy/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- [Help wanted](https://github.com/javierpa95/cgmpy/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
- [Discussion forum](https://github.com/javierpa95/cgmpy/discussions)

---

## Security & Privacy

CGMPy can process sensitive health data. Please review [SECURITY.md](SECURITY.md) for:

- How to report vulnerabilities.
- The strict policy on committing Protected Health Information (PHI).
- How to anonymize datasets before sharing examples or bug reports.

**Never commit real patient data, real CGM exports with personal identifiers, or any HIPAA/GDPR-protected data.** Examples and tests must use synthetic or fully anonymized data only.

---

## License

CGMPy is released under the **MIT License**. See [LICENSE](LICENSE) for details.

Copyright © 2024–2026 Javier Peñate Arrieta.

---

## Acknowledgments

- The clinical metric definitions follow international consensus guidelines (Battelino et al., *Diabetes Care* 2019; Bergenstal et al., *J. Diabetes Sci. Technol.* 2013).
- The [AGATA](https://github.com/gcappon/agata) library is the reference implementation used for cross-validation.
- Inspired by R packages like [`iglu`](https://github.com/irinagaina/iglu) and Python packages in the diabetes research ecosystem.
