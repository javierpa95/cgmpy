# Changelog

All notable changes to CGMPy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note**: This CHANGELOG is generated and maintained by [release-please](https://github.com/googleapis/release-please)
> starting from v0.4.0. Earlier entries below are the historical record.

---

## [Unreleased]

### Added

- Open source documentation: English `README.md`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, `ROADMAP.md`.
- Comprehensive `pyproject.toml` metadata (classifiers, project URLs, optional
  dependencies for `dev`, `docs`, and `agata`).
- `.editorconfig` and `.gitattributes` for cross-platform consistency.
- Strengthened `.gitignore`.

### Changed

- `LICENSE` populated with the full MIT text.

---

## [0.3.0] — 2026-06-01

### Added

- **Modular data layer**:
  - `cgmpy.data.loader.DataLoader` — load from CSV, Parquet, DataFrame.
  - `cgmpy.data.processor.DataProcessor` — validation, type coercion, dedup.
  - `cgmpy.data.analyzer.DataAnalyzer` — basic info, gap analysis, quality.
  - `cgmpy.data.exporter.DataExporter` — Parquet/CSV/Excel export.
  - `cgmpy.data.specialized.{Dexcom,Libreview}` — device-specific loaders.
  - `cgmpy.data.pregnancy_data.{PregnancyData, PregnancyDataHandler}`.
- **Modular metrics layer**:
  - `cgmpy.metrics.basic.BasicMetrics`.
  - `cgmpy.metrics.time_in_range.TimeInRangeMetrics`.
  - `cgmpy.metrics.variability.VariabilityMetrics`.
  - `cgmpy.metrics.pregnancy.GestationalDiabetes`.
  - `cgmpy.metrics.targets.GlucoseTargets` dataclass with
    `GlucoseTargets.standard()` and `GlucoseTargets.pregnancy()` factories.
- **Initial test suite**:
  - `tests/conftest.py` with glucose DataFrame fixtures.
  - `tests/unit/test_basic_metrics.py`.
  - `tests/unit/test_variability_metrics.py`.
  - `tests/integration/test_data_loading.py`.
- **New examples**:
  - `examples/benchmark_performance.py`.
  - `examples/reproduce_bugs.py`.
  - Updated `examples/comparison_agata_cgmpy.py` and
    `examples/repro_pregnancy.py`.

### Changed

- Backward-compatible public API; legacy class names (`GlucoseData`,
  `GlucoseMetrics`, `GlucosePlot`, `GlucoseAnalysis`) preserved.
- `cgmpy/metrics/__init__.py` re-exports the modular metric classes and
  exposes `ModularGlucoseMetrics` as a high-level facade.

---

## [0.2.0] — 2026-05

### Added

- AGATA library wrapper (`cgmpy.agata.AgataAnalysis`).
- Side-by-side comparison tooling against AGATA reference values.
- Improved glucose CSV parsing (numeric coercion, delimiter detection).
- Initial `examples/` directory.
- `PregnancyData` and gestational diabetes analysis.

---

## [0.1.0] — 2024

### Added

- Initial release.
- CGM CSV loader.
- Mean, median, GMI, SD.
- Simple matplotlib plots.
- Proof-of-concept pregnancy metrics.

---

[Unreleased]: https://github.com/javierpa95/cgmpy/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/javierpa95/cgmpy/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/javierpa95/cgmpy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/javierpa95/cgmpy/releases/tag/v0.1.0
