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

## Exceptions

All data-loading errors raise a subclass of `cgmpy.errors.DataError`, which
inherits from both `CGMPyError` and `ValueError`. Catch `CGMPyError` to
handle any CGMPy-raised error. Common ones:

- `ColumnNotFoundError` — required column missing (has `.column`, `.available`)
- `InvalidCSVFormatError` — CSV cannot be parsed (has `.file_path`, `.reason`)
- `DeviceDetectionError` — auto-detect failed (has `.file_path`, `.columns_found`)
- `GlucoseRangeError` — values outside 39-600 mg/dL (has `.n_invalid`, `.total`)
- `EmptyDataError` — no rows left after filtering

For the full hierarchy (including `MetricError`, `AgataIntegrationError`,
`AgataNotInstalledError`, `ConfigurationError`), see
[`cgmpy.errors`](https://github.com/javierpa95/cgmpy/blob/main/cgmpy/errors.py)
in the source tree.

## Pregnancy

::: cgmpy.data.pregnancy_data.PregnancyData
    options:
      show_root_heading: true

::: cgmpy.data.pregnancy_data.PregnancyDataHandler
    options:
      show_root_heading: true
      members:
        - trim_to_pregnancy_window
