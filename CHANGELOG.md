# Changelog

All notable changes to CGMPy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note**: This CHANGELOG is generated and maintained by [release-please](https://github.com/googleapis/release-please)
> starting from v0.4.0. Earlier entries below are the historical record.

---

## [Unreleased]

### Added
- **`cgmpy.metrics.validation`**: new `validate_glucose_range()` function and `ValidationReport` dataclass to flag glucose readings outside physiologically plausible ranges (default band 39-600 mg/dL; tightens to clinical targets when a `GlucoseTargets` is supplied). Exported from `cgmpy.metrics`.
- **Clinical reference tests** in `tests/clinical/test_basic_metrics_reference.py`: hand-computed expected values for mean, median, SD, CV, GMI, TIR, TAR, TBR, and data-completeness on a synthetic 24h dataset.
- **Per-family variability mixin classes**: `SDMetrics`, `MAGEMetrics`, `MODDMetrics`, `CONGAMetrics`, `LabilityMetrics`, `RiskMetrics` are now individually importable from `cgmpy.metrics.variability` for users who only need a subset of metrics. `VariabilityBase` is the shared type-stub mixin.
- **Massively expanded test suite**: 310 tests (up from 42), global coverage 30% → **81%**. New tests live in `tests/unit/test_metrics/variability/` (4 files, 47 tests), `tests/unit/test_plotting/` (3 files, 50 tests), `tests/unit/test_data/test_exporter.py` (25), `tests/unit/test_data/test_specialized.py` (18), `tests/unit/test_utils/test_date_utils.py` (37), `tests/unit/test_analysis/test_core.py` (23), `tests/unit/test_metrics/test_pregnancy.py` (12), `tests/unit/test_data/test_pregnancy_data.py` (21), `tests/unit/test_agata/test_adapter.py` (6), `tests/unit/test_agata/test_metrics.py` (10). Test files use `matplotlib.use('Agg')` for plotting and `pytest.importorskip("py_agata")` for the optional agata integration.

### Changed
- **`cgmpy/metrics/variability.py` (single 2034-line file) is now a package** `cgmpy/metrics/variability/` with one file per metric family: `_base.py` (50), `sd.py` (679), `mage.py` (709), `modd.py` (76), `conga.py` (115), `lability.py` (130), `risk.py` (281), `__init__.py` (187). The public `VariabilityMetrics` class is re-exported from the package as a composite mixin combining all six families, so existing code (`from cgmpy.metrics.variability import VariabilityMetrics`) keeps working unchanged.
- **Internalised `GlucoseData` alias in `cgmpy.data.__init__`**: removed duplicate `GlucoseData = ModularGlucoseData` binding. The class is still exported from `cgmpy` as a subclass of `ModularGlucoseData`.
- **Replaced `print()` calls with `logger` calls in library code**: `cgmpy/metrics/__init__.py` (16 print calls in `all()` and `all_simplified()`), `cgmpy/metrics/variability.py` (8 print calls in MAGE_Baghurst navigation and error handlers), `cgmpy/data/exporter.py` (18 print calls in `to_parquet`/`to_csv`/`to_excel`/`_log_save_info`/`append_to_parquet`), `cgmpy/agata/metrics.py`. `ModularGlucoseData` and its subclasses now log via `self.logger`.
- **Translated code, docstrings, and comments to English** across `cgmpy/`, including ~100 Spanish comments in `cgmpy/metrics/variability.py`, all of `cgmpy/data/`, `cgmpy/plotting/`, `cgmpy/analysis/`, `cgmpy/agata/`, `cgmpy/utils/date_utils.py`, and the top-level `cgmpy/__init__.py`.
- **Translation keys for the MAGE visualisation and segment dictionaries** (`Día`→`Day`, `CVDía`→`CVDay`, `Puntos de inflexión`→`Turning points`, `Excursión positiva/negativa`→`Positive/Negative excursion`, `Eliminación directa`→`Direct elimination`).
- **User-facing error message in `variability.py`**: `"El intervalo de {hours} horas es demasiado pequeño para los datos disponibles"` → `"The interval of {hours} hours is too small for the available data"`.
- **Navigation hint for the MAGE interactive plot**: now logged in English via `self.logger.info`.
- **`_create_filtered_instance` in `cgmpy/data/core.py`**: replaced manual `__new__` + `setattr` loop with `copy.copy(self)` (a single line), reducing the method from ~40 lines to ~14 while keeping identical behaviour.

### Removed
- **Dead code in `cgmpy/data/core.py`**: an orphaned `for attr in [...]` block after `return new_instance` in `_create_filtered_instance`.
- **Absolute import in `cgmpy/metrics/time_in_range.py`**: replaced `from cgmpy.metrics.targets import GlucoseTargets` with the relative `from .targets import GlucoseTargets`.
- **`cgmpy/metrics/variability_OLD.py`**: the 2034-line monolithic file was replaced by the `cgmpy/metrics/variability/` package; the temporary backup was deleted once the package passed the full test suite.

### Fixed
- **Glucose validation** is now wired into `DataProcessor._convert_data_types`; impossible sensor values (e.g. < 39 or > 600 mg/dL) generate a `WARNING`-level log entry and are surfaced via `processor._last_validation_report`.
- **`GRADE` return value mismatch** in `calculate_variability_metrics`: the old code read `grade.get("total")`, but `GRADE()` actually returns a dict with key `"grade_score"`. The risk-metrics aggregator now reads `"grade_score"` to avoid silently returning `None`.

### Known bugs (uncovered during the new test work; not fixed in this release)
The expanded test suite surfaced six latent bugs that should be tracked separately. They are captured by regression tests that document the current incorrect behaviour:
- `MAGE_Baghurst(approach=2)` raises `IndexError` on small/variable datasets because `glucose[turning_points[0]]` is dereferenced without checking the list is non-empty (`cgmpy/metrics/variability/mage.py:372`).
- `sd_between_timepoints(agrupar_por_intervalos=True)` raises `KeyError: 'day'` because the `day` column is only created in the non-grouped branch (`cgmpy/metrics/variability/sd.py:185`).
- `GlucosePlot` does not mix in `ModularGlucoseMetrics`, so `StatisticalPlotter` (which calls `self.TIR()`, `self.TBR70()`, `self.TAR180()`, `self.gmi()`) raises `AttributeError` when used through the public facade. Either add `ModularGlucoseMetrics` to the `GlucosePlot` MRO or refactor the plotters to use composition.
- `cgmpy/data/specialized.py` `__str__` methods reference `info['data_completeness']` but `DataAnalyzer.get_basic_info()` returns the key as `completeness`. Calling `str(Dexcom(...))` raises `KeyError`.
- `cgmpy/analysis/core.py` `get_comprehensive_report` / `get_summary_string` / `export_report` / `plot_comprehensive_dashboard` call `self.basic_statistics_summary()` and `self.calculate_variability_metrics()`, neither of which exists in the inheritance chain.
- `cgmpy/analysis/core.py:132` reads legacy keys (`TIR_tight`, `TBR70`, `TBR55`, `TAR140`, `TAR180`, `TAR250`) that the current `time_statistics()` no longer emits.

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
