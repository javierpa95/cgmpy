# API Reference — Plotting

The plotting layer generates static visualizations of CGM data.

## Plotters

::: cgmpy.plotting.agp.AGPPlotter
    options:
      show_root_heading: true
      members:
        - plot_agp

::: cgmpy.plotting.daily_plots.DailyPlotter
    options:
      show_root_heading: true
      members:
        - plot_daily_traces

::: cgmpy.plotting.statistical_plots.StatisticalPlotter
    options:
      show_root_heading: true
      members:
        - plot_glucose_histogram
        - plot_tir_breakdown

## High-level facade

::: cgmpy.analysis.core.GlucoseAnalysis
    options:
      show_root_heading: true
      members:
        - plot_comprehensive_dashboard
        - get_summary_string
        - get_comprehensive_report
