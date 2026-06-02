# Changelog

All notable changes to CGMPy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note**: This CHANGELOG is generated and maintained by [release-please](https://github.com/googleapis/release-please)
> starting from v0.4.0. Earlier entries below are the historical record.

---

## [Unreleased]

No changes yet. The next release will be **v0.6.0** (MAGE deep refactor + deprecation policy + mypy on public API), per `ROADMAP.md`.

---

## [0.5.2] — 2026-06-02

Hardening of the data-ingestion-to-AGATA pipeline. No breaking changes. All four CGM device loaders are now in the public API, all data-path and AGATA-path errors are now raised as CGMPy-specific subclasses (catchable as `CGMPyError` or, for backward compatibility, as `ValueError`), and the test suite is anchored on deterministic synthetic fixtures with hand-computed expected values.

### Added
- **`cgmpy.errors`**: new `CGMPyError` base + 12 specialised subclasses (`DataError` → `ColumnNotFoundError`, `InvalidCSVFormatError`, `DeviceDetectionError`, `GlucoseRangeError`, `EmptyDataError`; `AgataIntegrationError` → `AgataNotInstalledError`; `MetricError` → `InsufficientDataError`; `ConfigurationError`). All carry their context as instance attributes (`.column`, `.file_path`, `.columns_found`, `.n_invalid`, …) so callers can react programmatically.
- **`cgmpy-info` CLI**: new console-script entry point (`pyproject.toml [project.scripts]`) that reports the installed CGMPy version, the running Python interpreter, and the status of the optional `[agata]`, `[docs]`, and `[dev]` extras. Supports `--json` for machine-readable consumption in CI.
- **`MedtronicCarelink` and `TandemDiabetes` in the public API**: previously importable only via the private `cgmpy.data.specialized` path. All four device loaders are now in `cgmpy.__init__` and `cgmpy.__all__`.
- **13 deterministic synthetic CSV fixtures** under `tests/fixtures/` (4 device-format constant-120 datasets, 6 edge-case CSVs, 1 sine-24h synthetic). All values are reproducible byte-for-byte; the expected metric values are documented in two `README.md` files alongside the data. A generator script (`scripts/generate_fixtures_v052.py`) can re-create them.
- **122 new tests** across 4 new test files: `tests/unit/test_data/test_errors.py` (41), `tests/unit/test_data/test_synthetic_metrics.py` (67), `tests/unit/test_agata/test_metrics_errors.py` (15), and `tests/integration/test_data_to_agata.py` (22, gated by `pytest.importorskip("py_agata")`).
- **3 new docs pages**: `docs/user-guide/supported-devices.md` (clinician-friendly device list), `docs/user-guide/cli.md` (cgmpy-info reference), and a new "Troubleshooting" section in `docs/user-guide/loading-data.md`. The `docs/api/data.md` page now has an "Exceptions" section.

### Changed
- **`cgmpy.data.loader.DataLoader`**: unparseable CSVs and PyArrow failures now raise `InvalidCSVFormatError` (with the original parser error in `.reason` and a delimiter hint). Missing columns in a DataFrame raise `ColumnNotFoundError`.
- **`cgmpy.data.specialized.detect_device_type`** now returns `None` (not the string `"unknown"`) when no device format matches. **`create_specialized_loader`** now raises `DeviceDetectionError` (with the first 5 file columns in `.columns_found`) instead of silently falling back to a generic `ModularGlucoseData` with no columns configured.
- **`cgmpy.data.processor.DataProcessor.process_data`** gains a `strict_glucose_range: bool = False` parameter. When `True`, out-of-physiological-range glucose values (default 39–600 mg/dL) raise `GlucoseRangeError`; when `False` (default), the previous warn-only behaviour is preserved.
- **`cgmpy.agata.adapter.prepare_data_for_agata`** now raises `EmptyDataError` at 5 distinct points (empty input, empty after timestamp normalisation, NaT bounds, empty time range, all-NaN glucose on the regular grid) where it previously crashed with a cryptic pandas error or returned a useless DataFrame.
- **`cgmpy.agata.metrics.{analyze_one_arm,analyze_with_agata}`** now raise `AgataNotInstalledError` (with a `pip install 'cgmpy[agata]'` hint) instead of a bare `ImportError`. The redundant no-op `try/except Exception: raise` blocks in both functions are removed.
- **`py_agata` pinned to `==0.0.8`** in `pyproject.toml` for reproducible AGATA parity comparisons across environments.

### Notes for reviewers
- `DataError` deliberately inherits from both `CGMPyError` and `ValueError` so that pre-v0.5.2 code using `except ValueError:` continues to work. New code should prefer `except CGMPyError:` or the more specific subclasses.
- The `detect_device_type` heuristic still does not handle Libreview's 2-row banner header — that path requires `Libreview(file, header=2)`. A regression test (`test_detect_device_type_libreview_returns_none`) documents the limitation.
- The `cgmpy-info` CLI is brand new in v0.5.2. Pinning `py_agata==0.0.8` is a behavioural change for anyone upgrading from `>=0.0.1`; the AGATA parity tests in `tests/unit/test_agata/` will need to be re-validated when the pin is bumped.

---

## [0.5.1] — 2026-06-01

Bug-fix sweep over the 6 latent bugs surfaced by the v0.5.0 test expansion.
No public API changes. Coverage: 81.18% → 81.66%. Tests: 310 → 329.

### Added
- **`cgmpy.metrics.validation`**: new `validate_glucose_range()` function and `ValidationReport` dataclass to flag glucose readings outside physiologically plausible ranges (default band 39-600 mg/dL; tightens to clinical targets when a `GlucoseTargets` is supplied). Exported from `cgmpy.metrics`.
- **Clinical reference tests** in `tests/clinical/test_basic_metrics_reference.py`: hand-computed expected values for mean, median, SD, CV, GMI, TIR, TAR, TBR, and data-completeness on a synthetic 24h dataset.
- **Per-family variability mixin classes**: `SDMetrics`, `MAGEMetrics`, `MODDMetrics`, `CONGAMetrics`, `LabilityMetrics`, `RiskMetrics` are now individually importable from `cgmpy.metrics.variability` for users who only need a subset of metrics. `VariabilityBase` is the shared type-stub mixin.
- **Massively expanded test suite**: 310 tests (up from 42), global coverage 30% → **81%**. New tests live in `tests/unit/test_metrics/variability/` (4 files, 47 tests), `tests/unit/test_plotting/` (3 files, 50 tests), `tests/unit/test_data/test_exporter.py` (25), `tests/unit/test_data/test_specialized.py` (18), `tests/unit/test_utils/test_date_utils.py` (37), `tests/unit/test_analysis/test_core.py` (23), `tests/unit/test_metrics/test_pregnancy.py` (12), `tests/unit/test_data/test_pregnancy_data.py` (21), `tests/unit/test_agata/test_adapter.py` (6), `tests/unit/test_agata/test_metrics.py` (10). Test files use `matplotlib.use('Agg')` for plotting and `pytest.importorskip("py_agata")` for the optional agata integration.
- **v0.5.1 regression tests** in `tests/unit/test_v051_regressions.py` (17 tests across 6 classes) covering the bug-fix sweep below.

### Changed
- **`cgmpy/metrics/variability.py` (single 2034-line file) is now a package** `cgmpy/metrics/variability/` with one file per metric family: `_base.py` (50), `sd.py` (679), `mage.py` (709), `modd.py` (76), `conga.py` (115), `lability.py` (130), `risk.py` (281), `__init__.py` (187). The public `VariabilityMetrics` class is re-exported from the package as a composite mixin combining all six families, so existing code (`from cgmpy.metrics.variability import VariabilityMetrics`) keeps working unchanged.
- **Internalised `GlucoseData` alias in `cgmpy.data.__init__`**: removed duplicate `GlucoseData = ModularGlucoseData` binding. The class is still exported from `cgmpy` as a subclass of `ModularGlucoseData`.
- **Replaced `print()` calls with `logger` calls in library code**: `cgmpy/metrics/__init__.py` (16 print calls in `all()` and `all_simplified()`), `cgmpy/metrics/variability.py` (8 print calls in MAGE_Baghurst navigation and error handlers), `cgmpy/data/exporter.py` (18 print calls in `to_parquet`/`to_csv`/`to_excel`/`_log_save_info`/`append_to_parquet`), `cgmpy/agata/metrics.py`. `ModularGlucoseData` and its subclasses now log via `self.logger`.
- **Translated code, docstrings, and comments to English** across `cgmpy/`, including ~100 Spanish comments in `cgmpy/metrics/variability.py`, all of `cgmpy/data/`, `cgmpy/plotting/`, `cgmpy/analysis/`, `cgmpy/agata/`, `cgmpy/utils/date_utils.py`, and the top-level `cgmpy/__init__.py`.
- **Translation keys for the MAGE visualisation and segment dictionaries** (`Día`→`Day`, `CVDía`→`CVDay`, `Puntos de inflexión`→`Turning points`, `Excursión positiva/negativa`→`Positive/Negative excursion`, `Eliminación directa`→`Direct elimination`).
- **User-facing error message in `variability.py`**: `"El intervalo de {hours} horas es demasiado pequeño para los datos disponibles"` → `"The interval of {hours} hours is too small for the available data"`.
- **Navigation hint for the MAGE interactive plot**: now logged in English via `self.logger.info`.
- **`_create_filtered_instance` in `cgmpy/data/core.py`**: replaced manual `__new__` + `setattr` loop with `copy.copy(self)` (a single line), reducing the method from ~40 lines to ~14 while keeping identical behaviour.
- **`pyproject.toml` `fail_under`**: bumped from 25% (placeholder) to **80%** (real coverage is 81.66%).

### Removed
- **Dead code in `cgmpy/data/core.py`**: an orphaned `for attr in [...]` block after `return new_instance` in `_create_filtered_instance`.
- **Absolute import in `cgmpy/metrics/time_in_range.py`**: replaced `from cgmpy.metrics.targets import GlucoseTargets` with the relative `from .targets import GlucoseTargets`.
- **`cgmpy/metrics/variability_OLD.py`**: the 2034-line monolithic file was replaced by the `cgmpy/metrics/variability/` package; the temporary backup was deleted once the package passed the full test suite.

### Fixed
- **Glucose validation** is now wired into `DataProcessor._convert_data_types`; impossible sensor values (e.g. < 39 or > 600 mg/dL) generate a `WARNING`-level log entry and are surfaced via `processor._last_validation_report`.
- **`GRADE` return value mismatch** in `calculate_variability_metrics`: the old code read `grade.get("total")`, but `GRADE()` actually returns a dict with key `"grade_score"`. The risk-metrics aggregator now reads `"grade_score"` to avoid silently returning `None`.
- **`GlucosePlot` facade**: now mixes in `BasicMetrics` and `TimeInRangeMetrics` so `StatisticalPlotter` (which calls `self.gmi()`, `self.TIR()`, `self.TBR70()`, `self.TAR180()`) works through the public facade. Pre-fix: `AttributeError` on every `_generate_statistics_text` / `plot_time_in_range` / `plot_distribution_comparison` call.
- **`MAGE_Baghurst` crashes on small / constant datasets**: added a top-level guard that returns a well-formed zeroed dict when `len(glucose) < 9` (less than the 9-point smoothing window) or `sd == 0` (constant glucose → `threshold == 0` would otherwise register every pair of points as a fake excursion of magnitude 0). Pre-fix: `IndexError` or `ValueError: attempt to get argmax of an empty sequence` depending on the chosen approach.
- **`sd_between_timepoints(agrupar_por_intervalos=True)`**: added `df["day"] = df["time"].dt.date` before the `groupby(["day", "interval"])` call. Pre-fix: `KeyError: 'day'` on every invocation of the grouped path.
- **`cgmpy/data/specialized.py` `__str__` methods**: `Dexcom`, `Libreview`, `MedtronicCarelink`, `TandemDiabetes` now read `info['completeness']` (the actual key returned by `DataAnalyzer.get_basic_info`). Pre-fix: `KeyError: 'data_completeness'` on every `str(loader)` call.
- **`GlucoseAnalysis` MRO**: now mixes in `BasicMetrics` so `self.calculate_all_metrics()` resolves. Without this, every `get_comprehensive_report()` / `get_summary_string()` call raised `AttributeError`.
- **`GlucoseAnalysis.get_comprehensive_report`**: now calls `self.calculate_all_metrics()` and `self.calculate_variability_metrics()` (the methods that actually exist). Pre-fix: called `self.basic_statistics_summary()` and `self.calculate_all_variability_metrics()` (both `AttributeError`).
- **`GlucoseAnalysis.get_summary_string`**: the TIME IN RANGE section now calls the individual methods (`self.TIR()`, `self.TIR_tight()`, `self.TBR70()`, `self.TBR55()`, `self.TAR140()`, `self.TAR180()`, `self.TAR250()`) directly instead of reading legacy keys (`TIR_tight`, `TBR70`, `TBR55`, `TAR140`, `TAR180`, `TAR250`) that `time_statistics()` no longer emits. Pre-fix: `KeyError` on every call.

---

## [0.5.0] — 2026-06-01

### Added

- **OpenCode agent harness** (`AGENTS.md` + `.opencode/`) with 9
  agents (architect + 6 specialists + 2 execution agents), 5 skills,
  4 slash-commands, and 5 rule files.
- **GitHub Actions CI/CD**: matrix testing (3 OS × 3 Python), coverage
  upload to Codecov, `release-please` for version bumps, CodeQL,
  close-stale, line-ending check, PR title standards.
- **MkDocs documentation site** (`mkdocs.yml` + `docs/`): Material
  theme, mkdocstrings auto-API, sections for getting-started,
  user-guide, API, development, architecture (with ADRs), and legal
  (privacy + GDPR).
- **PyPI prep**: `MANIFEST.in`, `cgmpy/py.typed` (PEP 561),
  cross-platform `build-dist.{sh,ps1}` and `publish-{test,prod}.{sh,ps1}`
  scripts.
- **VSCode workspace** (`.vscode/`): `settings.json` (ruff, format,
  Python interpreter, EOL), `extensions.json` (16 recommendations),
  `launch.json` (6 debug configs), `tasks.json` (13 task shortcuts).
- **Devcontainer** (`.devcontainer/`): reproducible Python 3.11
  environment with `uv` and pre-configured VSCode extensions.
- Open source documentation: English `README.md`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md` (Covenant 2.1), `SECURITY.md`, `ROADMAP.md`.
- `commitlint` with Conventional Commits and an enumerated set of
  allowed scopes; `.pre-commit-config.yaml` with ruff, interrogate,
  commitlint, and a local docs-sync hook.
- `Makefile` with cross-platform targets (`help`, `test-fast`,
  `lint-fix`, `docs-serve`, `build`, `publish-test`, etc.).
- `CODEOWNERS`, `dependabot.yml`, issue and PR templates.
- `.editorconfig` and `.gitattributes` for cross-platform consistency.

### Changed

- Examples reorganized into numbered folders:
  `examples/01_quickstart/`, `02_pregnancy/`, `03_agata_comparison/`,
  `04_performance/`.
- Test suite reorganized into `tests/{unit,integration,clinical}/`
  with new test files (`test_targets.py`, `test_loader.py`,
  `test_processor.py`, `test_data_pipeline.py`).
- `LICENSE` populated with the full MIT text.
- Strengthened `.gitignore`.
- Comprehensive `pyproject.toml` metadata (classifiers, project URLs,
  optional dependencies for `dev`, `docs`, and `agata`).
- `cgmpy/__version__` bumped to `0.5.0`.

### Fixed

- All 156 ruff lint warnings in `cgmpy/`: replaced `typing.Dict` /
  `typing.List` / `typing.Tuple` with PEP 585 generics, replaced
  `Union[X, Y]` and `Optional[X]` with PEP 604 `X | Y`, fixed
  implicit optional, replaced `os.path.*` with `pathlib.Path`,
  removed unused imports, sorted `__all__` lists, fixed `dict(...)`
  literals, collapsible-`if`, ambiguous `×` characters in docstrings.

### Security

- `.github/workflows/publish-pypi.yml` **disabled** (renamed to
  `.disabled`) until the maintainer is ready to publish to PyPI.
  Re-enable instructions are in the file header.
- `SECURITY.md` documents PHI policies, vulnerability reporting
  email, and supported versions.
- `detect-secrets` and `bandit` integrated into the local pre-commit
  hooks and CI.

---

## [0.3.0] — 2026-06-01

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
