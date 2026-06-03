# API Reference — Plotting

The plotting layer generates static visualizations of CGM data.

## Module-level plot functions

::: cgmpy.plotting.agp
    options:
      show_root_heading: true
      members:
        - plot_agp
        - generate_week_agp

::: cgmpy.plotting.daily_plots
    options:
      show_root_heading: true
      members:
        - day_graph
        - plot_overlapping_days
        - plot_week_boxplots
        - plot_daily_variations

::: cgmpy.plotting.statistical_plots
    options:
      show_root_heading: true
      members:
        - histogram
        - plot_time_in_range
        - plot_distribution_comparison
        - plot_correlation_matrix

## High-level facade

::: cgmpy.analysis.core.GlucoseAnalysis
    options:
      show_root_heading: true
      members:
        - plot_agp
        - plot_daily
        - plot_overlapping_days
        - plot_week_boxplots
        - plot_daily_variations
        - histogram
        - plot_time_in_range
        - plot_distribution_comparison
        - plot_correlation_matrix
        - generate_week_agp
        - plot_variability_dashboard
        - plot_glucose_statistics
        - plot_comprehensive_dashboard
        - get_summary_string
        - get_comprehensive_report
