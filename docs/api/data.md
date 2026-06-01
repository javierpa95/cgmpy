# API Reference — Data

The data layer handles loading, processing, analysis, and exporting of
CGM time series.

## High-level

::: cgmpy.data.ModularGlucoseData
    options:
      show_root_heading: true
      show_root_toc_entry: true
      members: []

## Loaders

::: cgmpy.data.loader.DataLoader
    options:
      show_root_heading: true
      members:
        - load_from_csv
        - load_from_parquet
        - load_from_dataframe

## Processors

::: cgmpy.data.processor.DataProcessor
    options:
      show_root_heading: true
      members:
        - process_data
        - validate_columns
        - coerce_types
        - deduplicate

## Analyzers

::: cgmpy.data.analyzer.DataAnalyzer
    options:
      show_root_heading: true
      members:
        - calculate_typical_interval
        - detect_gaps
        - get_data_quality_metrics
        - info

## Exporters

::: cgmpy.data.exporter.DataExporter
    options:
      show_root_heading: true
      members:
        - to_parquet
        - to_csv
        - to_excel

## Device-specific loaders

::: cgmpy.data.specialized.Dexcom
    options:
      show_root_heading: true

::: cgmpy.data.specialized.Libreview
    options:
      show_root_heading: true

::: cgmpy.data.specialized.MedtronicCarelink
    options:
      show_root_heading: true

::: cgmpy.data.specialized.TandemDiabetes
    options:
      show_root_heading: true

## Pregnancy

::: cgmpy.data.pregnancy_data.PregnancyData
    options:
      show_root_heading: true

::: cgmpy.data.pregnancy_data.PregnancyDataHandler
    options:
      show_root_heading: true
      members:
        - trim_to_pregnancy_window
