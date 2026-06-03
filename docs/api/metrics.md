# API Reference — Metrics

The metrics layer computes clinical statistics from a CGM time series.

## High-level facade

The `GlucoseAnalysis` facade (in `cgmpy.analysis.core`) provides all
metrics via methods:

```python
from cgmpy import GlucoseAnalysis

analysis = GlucoseAnalysis("data.csv")
analysis.mean()
analysis.TIR()
analysis.cv()
analysis.gmi()
analysis.MAGE()
# ... see the API reference for the full list.
```

See `cgmpy.analysis.core.GlucoseAnalysis` for the full method list.

## Pure functions

All metric calculations are also available as standalone functions:

| Module | Functions |
|--------|-----------|
| `cgmpy.metrics.basic` | `mean`, `median`, `sd`, `cv`, `gmi`, `percentile` |
| `cgmpy.metrics.time_in_range` | `tir`, `tar`, `tbr`, `data_completeness` |
| `cgmpy.metrics.variability.sd` | `sd_global`, `sd_within_day`, `sd_between_timepoints`, ... |
| `cgmpy.metrics.variability.mage` | `mage_simple`, `mage_baghurst` |
| `cgmpy.metrics.variability.modd` | `modd` |
| `cgmpy.metrics.variability.conga` | `conga` |
| `cgmpy.metrics.variability.lability` | `lability_index` |
| `cgmpy.metrics.variability.risk` | `j_index`, `lbgi`, `hbgi`, `gri`, `adrr`, `m_value`, `grade` |

## Targets

::: cgmpy.metrics.targets.GlucoseTargets
    options:
      show_root_heading: true
      members:
        - standard
        - pregnancy

::: cgmpy.metrics.targets.get_targets
    options:
      show_root_heading: true

## Pregnancy

::: cgmpy.metrics.pregnancy.PregnancyAnalysis
    options:
      show_root_heading: true
      members:
        - summary_by_trimester
        - all_simplified
        - calculate_all_metrics
