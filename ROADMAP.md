# ROADMAP — CGMPy

> Continuous Glucose Monitoring analysis library for Python.

## Current Status

> **Maintenance mode — repo cleanup complete (June 2026).**
>
> The 5-commit cleanup sweep (ghost files, CI security, docs site,
> legal consolidation, ADRs) is done. See [Recently Completed](#recently-completed-june-2026).
>
> **No active development focus.** The next planned work is the MAGE
> refactor, but it will not be started until the maintainer explicitly
> requests it. AI agents must not propose or initiate MAGE work
> unprompted (see AGENTS.md §12).

---

## Recently Completed (June 2026)

### v0.5.1 — Bug Fixes & API Stability

All six latent bugs surfaced by the v0.5 test expansion are fixed and
covered by regression tests in `tests/unit/test_v051_regressions.py`.

- **`GlucosePlot` facade** — now mixes in `BasicMetrics` and
  `TimeInRangeMetrics` so `StatisticalPlotter` works through the public
  facade. (`cgmpy/__init__.py:56`)
- **`MAGE_Baghurst` small/constant datasets** — top-level guard returns
  a well-formed zeroed dict when `len(glucose) < 9` or `sd == 0`.
  (`cgmpy/metrics/variability/mage.py`)
- **`sd_between_timepoints(agrupar_por_intervalos=True)`** — added
  `df["day"] = df["time"].dt.date` before the `groupby`. (`sd.py:185`)
- **`specialized.py.__str__` key mismatch** — `Dexcom`, `Libreview`,
  `MedtronicCarelink`, `TandemDiabetes` now read `info['completeness']`
  (the actual key). (`cgmpy/data/specialized.py`)
- **`GlucoseAnalysis` MRO** — now mixes in `BasicMetrics`; the report
  methods call `calculate_all_metrics()` and `calculate_variability_metrics()`
  (the names that actually exist).
- **`GlucoseAnalysis.get_summary_string` time-in-range keys** — the
  section now calls the individual methods instead of reading legacy
  keys.

Other:

- `pyproject.toml` `fail_under` raised from 25 (placeholder) to **80**
  (real coverage is 81.66%).
- 17 new regression tests in `tests/unit/test_v051_regressions.py`.
  Three pre-existing tests that documented the bugs as "expected
  failure" were inverted to assert the fixed behaviour.
- Auto-fixed 13 pre-existing UP038 isinstance lint warnings.
- Tests: 310 → 329 passing. Coverage: 81.18% → 81.66%.

### v0.4 → v0.5 — Modernisation Sprint

> Massive refactor and translation pass that took the library from "hobby
> project" to "competent open-source package". 42 → 310 tests, 30% → 81% line
> coverage, 100% of library code English, zero `print()` calls, monolith
> split into a package.

- **Internationalisation**
  - Translated ~100 Spanish comments in `cgmpy/metrics/variability.py` and all
    Spanish docstrings / module docs / error messages to English.
  - MAGE visualisation labels (`Día` → `Day`, `Puntos de inflexión` →
    `Turning points`, `Excursión positiva/negativa` → `Positive/Negative
    excursion`, `Eliminación directa` → `Direct elimination`).
  - User-facing error string in `CONGA` translated to English.
  - 2 Spanish chars remaining are intentional: the author name (`__author__`)
    and the explanatory comment `# Ya está en minutos`.

- **Logging discipline** — replaced 45+ `print()` calls with `self.logger`:
  - `cgmpy/metrics/__init__.py` (16 calls in `all()` / `all_simplified()`)
  - `cgmpy/metrics/variability.py` (8 calls in MAGE_Baghurst navigation)
  - `cgmpy/data/exporter.py` (18 calls in Parquet / CSV / Excel exporters)
  - `cgmpy/agata/metrics.py` (3 calls in error paths)

- **Mechanical cleanup**
  - Deleted dead code in `cgmpy/data/core.py:_create_filtered_instance`
    (orphaned loop after `return new_instance`).
  - Fixed absolute import in `cgmpy/metrics/time_in_range.py` (now relative
    `from .targets import GlucoseTargets`).
  - Removed duplicate `GlucoseData = ModularGlucoseData` alias in
    `cgmpy/data/__init__.py`.
  - `cgmpy/data/core.py:_create_filtered_instance` refactored to use
    `copy.copy(self)` instead of manual `__new__` + `setattr` loop.

- **`cgmpy/metrics/validation` — new module**
  - `validate_glucose_range(data, targets=None, warn=True)` → `ValidationReport`
    dataclass flagging glucose readings outside physiologically plausible
    bounds (default 39–600 mg/dL; tightens to clinical targets when supplied).
  - Exported from `cgmpy.metrics`.
  - Wired into `DataProcessor._convert_data_types`; impossible values
    generate a `WARNING` log entry and are surfaced via
    `processor._last_validation_report`.
  - **Bonus fix:** `calculate_variability_metrics` was reading `grade.get("total")`
    but `GRADE()` actually returns `grade_score` — aggregator now reads the
    right key, fixing a silent `None` return.

- **Variability subpackage** — `cgmpy/metrics/variability.py` (2034 lines)
  replaced with a package of one file per metric family:
  - `_base.py` (50) — `VariabilityBase` mixin + type stubs
  - `sd.py` (679) — `SDMetrics` (14 methods, the SD/CV battery)
  - `mage.py` (709) — `MAGEMetrics` (`MAGE`, `MAGE_Baghurst`)
  - `modd.py` (76) — `MODDMetrics`
  - `conga.py` (115) — `CONGAMetrics`
  - `lability.py` (130) — `LabilityMetrics` (Lability Index + summaries)
  - `risk.py` (281) — `RiskMetrics` (LBGI, HBGI, GRI, GRADE, ADRR, M-Value, J-Index)
  - `__init__.py` (187) — composite `VariabilityMetrics` + `calculate_variability_metrics`
  - `VariabilityMetrics` is re-exported from the package as a multiple-inheritance
    composite, so the public API (`from cgmpy.metrics.variability import
    VariabilityMetrics`) is unchanged.
  - Individual mixins (`SDMetrics`, `MAGEMetrics`, etc.) are also importable
    for users who only need a subset.

- **Test suite expansion** — 42 → 310 tests, 30% → 81% line coverage.
  - `tests/unit/test_metrics/variability/` (47 tests, 4 files)
  - `tests/unit/test_plotting/` (50 tests, 3 files, `matplotlib.use('Agg')`)
  - `tests/unit/test_data/test_exporter.py` (25)
  - `tests/unit/test_data/test_specialized.py` (18)
  - `tests/unit/test_utils/test_date_utils.py` (37)
  - `tests/unit/test_analysis/test_core.py` (23)
  - `tests/unit/test_metrics/test_pregnancy.py` (12)
  - `tests/unit/test_data/test_pregnancy_data.py` (21)
  - `tests/unit/test_agata/test_adapter.py` (6, optional dep)
  - `tests/unit/test_agata/test_metrics.py` (10, optional dep)
  - `tests/clinical/test_basic_metrics_reference.py` (10, hand-computed)

- **Project hygiene**
  - `pyproject.toml` `fail_under` adjusted to 25 (the real coverage was 25–30%;
    the previous 70% was a lie). `CHANGELOG.md` `[Unreleased]` documents all
    the changes.
  - LF line endings enforced everywhere; one stray CRLF in
    `cgmpy/utils/__init__.py` was fixed.

---

## Roadmap

### v0.6.0 — MAGE Refactor & Type-Strict API (deferred — user-request only)

- [ ] **Split `MAGE_Baghurst`** — the 645-line function with 3 approaches
      becomes three focused methods (`mage_baghurst_smoothing`,
      `mage_baghurst_direct_elimination`, `mage_baghurst_simplified`).
- [ ] **Move interactive matplotlib code** out of `variability/mage.py` and
      into `cgmpy/plotting/mage_excursions.py` so the metric module contains
      only pure computation.
- [ ] **Add a deprecation policy** — public symbols can be deprecated with
      a `DeprecationWarning` and a 2-release grace period before removal.
- [ ] **Run `mypy --strict cgmpy/` on the public API** — add type hints where
      they are missing (mostly in plotting and analysis modules).
- [ ] **Drop Python 3.10 support** if the test matrix shows it adds no
      signal — keep 3.11+ as a hard floor.

### v0.7.0 — Documentation Overhaul (target: August 2026)

- [ ] **Regenerate `docs/api/`** with the new variability subpackage structure
      (one page per metric family: `sd.md`, `mage.md`, `modd.md`, `conga.md`,
      `lability.md`, `risk.md`, `variability.md`).
- [ ] **Tutorial notebooks** under `examples/notebooks/`:
      `01_quickstart.ipynb`, `02_agp_plot.ipynb`, `03_pregnancy.ipynb`,
      `04_agata_comparison.ipynb`, `05_validation.ipynb`.
- [ ] **Architecture diagrams** in `docs/architecture/` — a Mermaid graph of
      the mixin composition (`ModularGlucoseData` + 4 plotters + 4 metric
      mixins + AGATA wrapper + analysis facade).
- [ ] **Decision records** — formalise the mixin-composition decision and the
      one-file-per-metric-family decision as ADRs in
      `docs/architecture/decisions/`.
- [ ] **Glossary** — `docs/user-guide/glossary.md` with the meaning of every
      metric (Mean, GMI, MAGE, CONGA, GRI, etc.) and its clinical reference.

### v0.8.0 — Clinical Research Features (target: October 2026)

- [ ] **Cohort analysis** — multiple subjects in one DataFrame, per-subject
      metrics, group summaries, between-group comparison helpers.
- [ ] **Time-windowed metrics** — TIR per week, per month, per trimester;
      rolling-window MAGE / CV.
- [ ] **Statistical testing** — paired comparisons (before/after, control /
      intervention), confidence intervals for proportions, equivalence /
      non-inferiority tests for sensor accuracy studies.
- [ ] **Report generation** — PDF / HTML reports for clinical visits using
      `jinja2` templates + `weasyprint` (or `playwright` for HTML).
- [ ] **More clinical regression tests** — reference metrics from published
      datasets (OhioT1DM, REPLACE-BG, JDRF, etc.).
- [ ] **FHIR interoperability** — import / export to FHIR `Observation`
      resources (read-only at first).

### v0.9.0 — Multi-modal & International (target: December 2026)

- [ ] **More device loaders** — Eversense, Medtronic 780G, Tandem t:slim X2,
      Insulet Omnipod 5.
- [ ] **Insulin & meal integration** — load CHO, bolus, basal alongside
      glucose; compute insulin-on-board and meal-impact metrics.
- [ ] **Real-time / streaming** — process incoming CGM data (e.g. from a
      Nightscout REST API) with a streaming-friendly data structure.
- [ ] **Internationalisation** — units (mg/dL ↔ mmol/L) as a first-class
      `GlucoseUnit` enum; translated error messages and report templates
      (es, en, fr, de, pt).
- [ ] **Optional web dashboard** — FastAPI + HTMX minimal front-end for
      clinicians who do not use Python.

### v1.0.0 — Production Ready (target: Q1 2027)

- [ ] **Strict mypy** — `mypy --strict cgmpy/` passes with zero errors.
- [ ] **95% test coverage** including branch coverage, with a CI-enforced
      `fail_under` of 90.
- [ ] **Performance benchmarks** — regression test that fails CI if a metric
      gets >20% slower than the v1.0 baseline. 1M-row dataset under 5s
      for the full battery.
- [ ] **Stable API promise** — public surface (`cgmpy/__init__.py`,
      `cgmpy.metrics`, `cgmpy.data`, `cgmpy.plotting`, `cgmpy.analysis`,
      `cgmpy.agata`) is frozen for 12 months.
- [ ] **DOI via Zenodo** for the v1.0.0 release.
- [ ] **Security audit** — third-party review of the agata integration,
      data-loader input handling, and CSV/Excel parsers.

---

## Backlog (distant future, no target)

- [ ] Wheel / source distribution via **cibuildwheel** for all major platforms
      (Linux x86_64 + aarch64, macOS Intel + Apple Silicon, Windows).
- [ ] Type stubs (`.pyi`) for the public API to give downstream type-checkers
      faster feedback.
- [ ] Performance profiling dashboard in CI — `viztracer` flame graphs posted
      as PR comments.
- [ ] **Glucose unit policy** — accept mixed-unit data, normalise internally
      to a single canonical unit, expose unit conversion in public API.
- [ ] **Docker image** for a notebook environment (binder / Codespaces).
- [ ] **DuckDB backend** for >1M-row datasets where pandas becomes slow.
- [ ] **Plugin system** — let third parties register custom metrics under
      a `cgmpy.metrics.contrib` namespace.
- [ ] **Federated analysis** — compute metrics across a fleet of CSV files
      without loading everything into memory.

---

## Out of scope (deliberately not planned)

- **Web front-end as the primary interface.** CGMPy is a Python library first;
  a thin web front-end is an *optional* convenience, not a product.
- **Native mobile app.** The Nightscout / xDrip ecosystem already covers this.
- **Cloud-hosted SaaS.** CGMPy is local-only by design (medical data privacy);
  the project will not run a hosted service.
- **Non-CGM diabetes data** (BGM fingersticks, HbA1c lab values). These are
  useful inputs but out of scope for the core library; they belong in a
  separate `cgmpy-bgm` companion if ever built.

---

## How to propose a roadmap change

Open a [Discussion](../../discussions) or a [feature request](../../issues/new?template=feature_request.md)
with the `roadmap` label. Roadmap items are prioritised by:

1. **Clinical relevance** — does it support a real research / clinical use case?
2. **Community demand** — how many users have asked for it?
3. **Implementation cost** — how much maintenance overhead does it add?
4. **Strategic alignment** — does it move CGMPy toward the long-term vision?

For solo development (no team), items are also gated on:
5. **Solo maintainability** — can one person realistically own, test, and
   document the change in <1 week of focused work?

---

_Last updated: June 2026_
