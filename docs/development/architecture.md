# Architecture

CGMPy follows a **facade + modular** design: a small public API at the
top, a deeper modular structure beneath, each layer independently testable.

## Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Public API (cgmpy/__init__.py)                             │
│  GlucoseAnalysis, GlucoseData, GlucoseMetrics, GlucosePlot  │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼────────────────────┐
        ▼                  ▼                    ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│  cgmpy.data  │   │ cgmpy.metrics│   │  cgmpy.plotting  │
│              │   │              │   │                  │
│ loader       │   │ basic        │   │ agp              │
│ processor    │   │ time_in_range│   │ daily_plots      │
│ analyzer     │   │ variability  │   │ statistical_plots│
│ exporter     │   │ pregnancy    │   │                  │
│ specialized  │   │ targets      │   │                  │
│ core         │   │              │   │                  │
│ pregnancy    │   │              │   │                  │
└──────────────┘   └──────────────┘   └──────────────────┘
        │                  │                    │
        └──────────────────┼────────────────────┘
                           ▼
                  ┌──────────────────┐
                  │  cgmpy.analysis  │
                  │  GlucoseAnalysis │
                  │  (orchestrator)  │
                  └──────────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │   cgmpy.agata    │
                  │  (cross-check    │
                  │  vs. reference)  │
                  └──────────────────┘
```

## Data flow

A typical CGMPy session looks like:

```
   ┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
   │  CSV    │────▶│  Loader  │────▶│Processor │────▶│ Analyzer │
   │  /DF    │     │          │     │          │     │          │
   └─────────┘     └──────────┘     └──────────┘     └────┬─────┘
                                                           │
                       ┌───────────────────────────────────┘
                       ▼
                ┌─────────────┐         ┌──────────┐
                │   Metrics   │────────▶│  Plotter │
                │  layer      │         │          │
                └──────┬──────┘         └──────────┘
                       │
                       ▼
                ┌─────────────┐
                │  Report     │
                │  (JSON-like │
                │   dict)     │
                └─────────────┘
```

## Key design decisions

CGMPy's main design decisions are recorded as
[ADRs](../architecture/decisions/index.md). The most important ones:

- **Modular facade pattern** — public API exposes both the
  high-level `GlucoseAnalysis` and the underlying modular classes, so
  users can choose their level of abstraction.
- **Backward compatibility** — the v0.2 API (`GlucoseData`,
  `GlucoseMetrics`, `GlucosePlot`, `Dexcom`, `Libreview`) is preserved
  through aliases, so existing code continues to work.
- **NumPy / pandas first** — every metric is implemented with vectorized
  operations over `pandas.Series`, not Python loops.
- **AGATA parity** — the `agata` optional dependency is the reference
  implementation; CGMPy cross-validates against it where the API overlaps.
- **English only** — code, comments, docstrings, documentation,
  commit messages, and PR descriptions are in English.

## Module responsibilities

| Module                  | Responsibility                                            |
|-------------------------|-----------------------------------------------------------|
| `cgmpy/data/loader.py`  | File I/O and format detection.                            |
| `cgmpy/data/processor.py` | Validation, type coercion, dedup.                       |
| `cgmpy/data/analyzer.py`| Gap detection, quality scoring, summaries.                 |
| `cgmpy/data/exporter.py`| Parquet / CSV / Excel export.                             |
| `cgmpy/data/specialized.py` | Vendor-specific loaders.                              |
| `cgmpy/data/pregnancy_data.py` | Pregnancy-window trimming.                         |
| `cgmpy/data/core.py`    | `ModularGlucoseData` facade (wires the above).            |
| `cgmpy/metrics/basic.py`| Mean, median, GMI, SD, CV, IQR, percentiles.              |
| `cgmpy/metrics/time_in_range.py` | TIR, TAR (1/2), TBR (1/2).                    |
| `cgmpy/metrics/variability.py` | CV, MAGE, MODD, CONGA, J-Index, LBGI, HBGI, GRI, ADRR. |
| `cgmpy/metrics/pregnancy.py` | GestationalDiabetes-specific metrics.                |
| `cgmpy/metrics/targets.py` | GlucoseTargets dataclass + helpers.                   |
| `cgmpy/plotting/agp.py` | Ambulatory Glucose Profile.                                |
| `cgmpy/plotting/daily_plots.py` | Daily trace plots.                            |
| `cgmpy/plotting/statistical_plots.py` | Histograms, TIR breakdown.                |
| `cgmpy/analysis/core.py`| `GlucoseAnalysis` facade (load + metrics + plot).         |
| `cgmpy/agata/metrics.py`| `AgataAnalysis` wrapper around the AGATA library.         |
| `cgmpy/utils/`          | Helpers (date parsing, range validation, …).              |

## Public API stability

Symbols exported from `cgmpy/__init__.py` are **stable**. To change one:

1. Add a `DeprecationWarning` in the same release.
2. Document the change in `CHANGELOG.md`.
3. Remove the old symbol after 1 minor version.

## See also

- [System overview](../architecture/system-overview.md)
- [ADRs](../architecture/decisions/index.md)
