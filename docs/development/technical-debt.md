# Technical Debt

This page is a living log of known limitations, workarounds, and
planned refactors. It is maintained by the maintainer and the AI agents
that work on the codebase.

> **Format**: each entry has a date, a category, the affected files, a
> short description, and the proposed fix.

## Active

### 2026-06 — Type hints coverage

- **Category**: typing
- **Files**: `cgmpy/data/*.py`, `cgmpy/metrics/*.py`
- **Issue**: Many internal functions lack return-type annotations.
  `mypy` is configured permissively (`disallow_untyped_defs = false`).
- **Proposed fix**: Gradually annotate internal functions. Aim for
  ≥ 80 % coverage by v0.6. Do not block external PRs on this.

### 2026-06 — `interrogate` threshold

- **Category**: docstring coverage
- **Files**: `cgmpy/`
- **Issue**: The pre-commit `interrogate` hook is set to `fail-under = 70`,
  which is a relatively low bar. Some public functions still lack
  docstrings.
- **Proposed fix**: Raise the threshold to 80 in Phase 2 of the roadmap.
  Add a CI report that lists functions without docstrings.

### 2026-06 — Mixed language error messages

- **Category**: i18n
- **Files**: `cgmpy/data/processor.py`, `cgmpy/metrics/pregnancy.py`,
  `cgmpy/utils/date_utils.py`
- **Issue**: A few error messages and metric names are still in Spanish
  (carryover from the v0.2 codebase).
- **Proposed fix**: Audit and translate. Add a regression test that
  asserts all `raise XError("...")` messages are in English.

### 2026-06 — Single maintainer bottleneck

- **Category**: governance
- **Files**: n/a
- **Issue**: `@javierpa95` is the only CODEOWNER. PRs cannot be merged
  without a self-review, which is not ideal.
- **Proposed fix**: Recruit 1–2 co-maintainers from the open source
  community. Add them to CODEOWNERS with appropriate scope.

### 2026-06 — Docstring language

- **Category**: i18n
- **Files**: `cgmpy/data/*.py`, `cgmpy/metrics/*.py`, `cgmpy/utils/*.py`
- **Issue**: Most docstrings (Google-style) are in Spanish.
- **Proposed fix**: Translate to English in v0.6.

## Resolved

### 2026-06 — Ruff lint warnings (156 errors)

- **Category**: tooling
- **Files**: `cgmpy/`
- **Issue**: 156 ruff warnings (UP006/UP007/UP035/UP038 typing,
  RUF002 unicode in docstring, RUF010/RUF013 explicit conversion /
  implicit Optional, RUF022 unsorted `__all__`, C408 unnecessary
  `dict()` literal, PTH110/PTH113/PTH123/PTH202 pathlib, W291/W293
  whitespace, C414, SIM102 collapsible-if, F401 unused import).
- **Resolution**: Fixed in commit `style(ruff): resolve 156 lint
  warnings (typing, pathlib, formatting)` of the v0.5.0 prep.
  `ruff check .` now reports `All checks passed!`.

## How to add an entry

1. Add a new section under **Active** with today's date as the heading.
2. Fill in the four fields: Category, Files, Issue, Proposed fix.
3. When resolved, move the entry to **Resolved** with a short note on
   the resolution (commit hash, PR number, version).

## See also

- [`AGENTS.md` § 6](https://github.com/javierpa95/cgmpy/blob/main/AGENTS.md).
- [Architecture](architecture.md).
- [Git workflow](git-workflow.md).
