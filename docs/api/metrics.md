# API Reference — Metrics

The metrics layer computes clinical statistics from a CGM time series.

## High-level

::: cgmpy.metrics.ModularGlucoseMetrics
    options:
      show_root_heading: true
      members:
        - basic
        - time_in_range
        - variability

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

## Basic metrics

::: cgmpy.metrics.basic.BasicMetrics
    options:
      show_root_heading: true
      members:
        - mean
        - median
        - std
        - cv
        - gmi
        - iqr
        - percentile
        - distribution_analysis

## Time in range

::: cgmpy.metrics.time_in_range.TimeInRangeMetrics
    options:
      show_root_heading: true
      members:
        - tir
        - tar1
        - tar2
        - tbr1
        - tbr2

## Variability

::: cgmpy.metrics.variability.VariabilityMetrics
    options:
      show_root_heading: true
      members:
        - cv
        - sd
        - mage
        - mage_plus
        - mage_minus
        - modd
        - conga
        - j_index
        - lbgi
        - hbgi
        - gri
        - adrr

## Pregnancy

::: cgmpy.metrics.pregnancy.GestationalDiabetes
    options:
      show_root_heading: true
      members:
        - compute_all
        - time_in_range_per_meal
        - glycemia_risk_index
        - overnight_metrics
        - daytime_metrics
