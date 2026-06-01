# System Overview

CGMPy is a **Python library** for analyzing Continuous Glucose Monitoring
(CGM) data. It is **library-first** (not a service, not a CLI), designed
to be imported into research pipelines, Jupyter notebooks, and clinical
software.

## Who uses CGMPy

- **Clinical researchers** running cohort analyses.
- **Diabetes clinicians** generating per-patient reports.
- **Medical-device companies** integrating CGM analytics into their apps.
- **Data scientists** building predictive models on top of CGM data.

## What CGMPy is **not**

- **Not a medical device** — does not provide medical advice.
- **Not a service** — no hosted API, no auth, no data storage.
- **Not a UI** — plots are static PNG/PDF/SVG, not interactive dashboards.
- **Not a real-time system** — it is a batch analytics library, not a
  streaming pipeline.

## Design principles

1. **Modularity** — every component (loader, processor, analyzer, plotter)
   can be used independently. The public API is a facade on top of the
   modular layers.
2. **Backward compatibility** — the public API in `cgmpy/__init__.py` is
   stable. New versions do not break existing code unless part of a
   documented deprecation cycle.
3. **English-only** — code, comments, docstrings, documentation, commit
   messages, and PR descriptions are in English. This maximizes the
   pool of potential contributors.
4. **Medical-data aware** — explicit policies on PHI handling, anonymization,
   and security.
5. **AI-friendly, human-controlled** — the project ships an OpenCode
   agent harness (`AGENTS.md` + `.opencode/`) so AI assistants can help,
   but humans remain the decision-makers.
6. **Reproducible** — every metric is implemented with vectorized
   operations and has a reference value in the literature.
7. **Tested** — every public function has a unit test, every new metric
   has a clinical regression test when a published reference exists.

## Data flow

```
                    ┌──────────────────┐
                    │  External source │
                    │  (CSV / Parquet  │
                    │   / DataFrame)   │
                    └────────┬─────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │  cgmpy.data.loader      │
                │  - format detection     │
                │  - column normalization │
                │  - device-specific      │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │  cgmpy.data.processor   │
                │  - numeric coercion     │
                │  - deduplication        │
                │  - gap detection        │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │  cgmpy.data.analyzer    │
                │  - quality metrics      │
                │  - typical interval     │
                │  - summary              │
                └────────────┬────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
   ┌────────────────────────┐   ┌──────────────────────────┐
   │  cgmpy.metrics.*       │   │  cgmpy.plotting.*        │
   │  - basic               │   │  - AGP                   │
   │  - time_in_range       │   │  - daily traces          │
   │  - variability         │   │  - statistical summary   │
   │  - pregnancy           │   │                          │
   └────────────┬───────────┘   └────────────┬─────────────┘
                │                            │
                └────────────┬───────────────┘
                             ▼
                ┌─────────────────────────┐
                │  cgmpy.analysis         │
                │  GlucoseAnalysis facade │
                │  (orchestrates the      │
                │   whole pipeline)       │
                └─────────────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Outputs      │
                    │  - JSON report   │
                    │  - Plots (PNG)   │
                    │  - Exported DF   │
                    └──────────────────┘
```

## Where things live

| Concern                | Module / File                                  |
|------------------------|------------------------------------------------|
| Public API             | `cgmpy/__init__.py`                            |
| Data loaders           | `cgmpy/data/loader.py`, `cgmpy/data/specialized.py` |
| Validation             | `cgmpy/data/processor.py`                      |
| Quality analysis       | `cgmpy/data/analyzer.py`                       |
| Export                 | `cgmpy/data/exporter.py`                       |
| Pregnancy              | `cgmpy/data/pregnancy_data.py`, `cgmpy/metrics/pregnancy.py` |
| Basic metrics          | `cgmpy/metrics/basic.py`                       |
| TIR / TAR / TBR        | `cgmpy/metrics/time_in_range.py`               |
| Variability            | `cgmpy/metrics/variability.py`                 |
| Targets                | `cgmpy/metrics/targets.py`                     |
| Plots                  | `cgmpy/plotting/*`                             |
| High-level facade      | `cgmpy/analysis/core.py`                       |
| AGATA wrapper          | `cgmpy/agata/metrics.py`                       |

## See also

- [Architecture (developer guide)](../development/architecture.md).
- [ADRs](decisions/index.md).
- [`AGENTS.md`](https://github.com/javierpa95/cgmpy/blob/main/AGENTS.md).
