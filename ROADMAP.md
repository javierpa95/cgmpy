# ROADMAP — CGMPy

## Current Focus

> **Phase 1 — Open Source Readiness (June 2026)**
>
> Consolidating CGMPy as a public open source project: AGENTS.md harness, full
> CI/CD, open source documentation (README, CONTRIBUTING, CoC, SECURITY), MIT
> license, structured docs site, and PyPI publication.

---

## ✅ Completed Phases

### v0.1 — Initial MVP

- Basic CGM CSV loader.
- Mean, median, GMI, SD.
- Simple matplotlib plots.
- Proof-of-concept for pregnancy metrics.

### v0.2 — AGATA Integration

- AGATA library wrapper (`cgmpy.agata`).
- Side-by-side comparison tooling.
- Improved glucose CSV parsing (numeric coercion, delimiter detection).
- Initial `examples/` directory.

### v0.3 — Modular Refactor (current baseline)

- Split `data/` into `loader`, `processor`, `analyzer`, `exporter`,
  `specialized`, `core`.
- Split `metrics/` into `basic`, `time_in_range`, `variability`, `pregnancy`.
- Added `metrics/targets.py` with `GlucoseTargets` dataclass (diabetes and
  pregnancy profiles).
- Added `PregnancyData` and `PregnancyDataHandler`.
- Backward-compatible public API via `cgmpy/__init__.py`.
- Initial `tests/{unit,integration}/` with conftest fixtures.

---

## 🚧 Phase 1 — Open Source Readiness (Q2 2026, in progress)

- [x] Commit modular refactor baseline (v0.3.0).
- [x] Tooling base: `.editorconfig`, `.gitattributes`, refreshed `.gitignore`,
      MIT `LICENSE`, comprehensive `pyproject.toml`.
- [x] Open source documentation: English `README.md`, `CONTRIBUTING.md`,
      `CODE_OF_CONDUCT.md`, `SECURITY.md`.
- [ ] **OpenCode Agent Harness** — `AGENTS.md`, `.opencode/agents/*`,
      `.opencode/skills/*`, `.opencode/commands/*`, `.opencode/rules/*`.
- [ ] **Git workflow** — pre-commit hooks, commitlint, lint-staged, `Makefile`.
- [ ] **CI/CD** — GitHub Actions (CI matrix, release-please, PyPI publish,
      docs deployment, CodeQL, PR standards).
- [ ] **Examples reorganization** — `examples/01_quickstart`, `02_pregnancy`,
      `03_agata_comparison`, `04_performance`, `05_reproduce_bugs`.
- [ ] **Docs site** — mkdocs-material with user guide, API reference, dev
      guide, legal section.
- [ ] **PyPI publication** — first public release on PyPI.

---

## 🔭 Phase 2 — Metric & Visualization Expansion (Q3 2026)

- [ ] **More glycemic variability metrics**: MAG, ADRR, BGRI, IGC, M-Value,
      GRADE, eA1c.
- [ ] **Interactive plots** with Plotly (AGP, daily trends, dashboards).
- [ ] **Hypoglycemia / hyperglycemia event detection** with configurable
      thresholds and durations.
- [ ] **Time-of-day analysis** (e.g., nocturnal TIR, breakfast TIR).
- [ ] **Per-day reports** — single-day CGM summaries.
- [ ] **Cross-validation reports** against AGATA for every metric, automated.
- [ ] **Performance benchmarks** for million-row datasets.

---

## 🏥 Phase 3 — Clinical Research Features (Q4 2026)

- [ ] **Cohort analysis** — multiple subjects in one DataFrame, per-subject
      metrics, group summaries.
- [ ] **Time-windowed metrics** — TIR per week, per month, per trimester.
- [ ] **Statistical testing** — paired comparisons (before/after, control/
      intervention), confidence intervals for proportions.
- [ ] **Report generation** — PDF / HTML reports for clinical visits.
- [ ] **Clinical regression tests** — reference metrics from published
      datasets (OhioT1DM, REPLACE-BG, etc.).
- [ ] **FHIR interoperability** — import / export to FHIR `Observation`
      resources.

---

## 🌍 Phase 4 — Multi-modal & Advanced Loaders (Q1 2027)

- [ ] **More device loaders**: Eversense, Medtronic 780G, Tandem t:slim X2,
      Insulet Omnipod 5.
- [ ] **Insulin & meal integration** — load CHO, bolus, basal alongside
      glucose; compute insulin metrics.
- [ ] **Real-time / streaming** — process incoming CGM data (e.g., from
      Nightscout REST API).
- [ ] **Internationalization** — units (mg/dL ↔ mmol/L), languages for
      reports (es, en, fr, de, pt).
- [ ] **Web dashboard** — optional FastAPI + HTMX / Vue front-end for
      non-Python users.

---

## 🧬 Phase 5 — ML & Digital Twin Integration (Q2 2027+)

- [ ] **Predictive alerts** — hypoglycemia / hyperglycemia forecasters.
- [ ] **Personalized baseline modeling** — what-if simulations.
- [ ] **Integration with the [Digital Twin Project](https://github.com/)**
      for metabolic simulation.
- [ ] **Federated analysis** — compute metrics without moving patient data.
- [ ] **HIPAA-compliant deployment** guidance for clinical use.

---

## Backlog

- [ ] Wheel / source distribution via cibuildwheel for all major platforms.
- [ ] Type stubs (`.pyi`) for public API.
- [ ] Performance profiling dashboard in CI.
- [ ] Translation of error messages (es, fr, de, pt).
- [ ] Docker image for a notebook environment.
- [ ] Binder / GitHub Codespaces quick start.

---

## How to propose a roadmap change

Open a [Discussion](../../discussions) or a [feature request](../../issues/new?template=feature_request.md)
with the `roadmap` label. Roadmap items are prioritized by:

1. **Clinical relevance** — does it support a real research / clinical use case?
2. **Community demand** — how many users have asked for it?
3. **Implementation cost** — how much maintenance overhead does it add?
4. **Strategic alignment** — does it move CGMPy toward the long-term vision?

---

_Last updated: June 2026_
