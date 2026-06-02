# 🤖 AGENTS.md — Guide for AI Coding Agents

> **Project:** CGMPy — Continuous Glucose Monitoring Analysis
> **Version:** 0.3.0
> **Audience:** AI coding assistants (OpenCode, GitHub Copilot, Claude Code, Cursor, Aider, etc.) and the human maintainers who direct them.

This file is the **single source of truth** for how AI agents should behave when contributing to CGMPy. If something is not documented here, **ask the human** before assuming.

---

## 1. Operating Philosophy

1. **Plan before executing.** For any change touching > 3 files or > 50 lines, present a brief plan (≤ 5 bullets) and wait for approval.
2. **Prefer small, verifiable changes.** One logical change per commit. Refactors must not be mixed with feature work.
3. **Ask, don't assume.** If requirements are ambiguous, ask. If a question is open, mark it with `TODO(human)` in code.
4. **Never break the public API.** The symbols exported from `cgmpy/__init__.py` are stable; any change must be backward compatible or part of a documented deprecation cycle.
5. **Update the docs.** See the [Documentation Golden Rule](#3-documentation-golden-rule) below.
6. **English everywhere.** Code, comments, docstrings, documentation, commit messages, and PR descriptions are in English.
7. **UTF-8 without BOM, LF line endings.** Enforced by `.editorconfig` and `.gitattributes`. Do not introduce other encodings.

---

## 2. Repository Map (Sacred)

```
cgmpy/                  ← Source code (the library itself)
├── data/               📥 Loaders, processors, exporters
│   ├── loader.py         Generic CSV/Parquet/DataFrame loaders
│   ├── processor.py      Validation, type coercion, dedup
│   ├── analyzer.py       Basic info, gap analysis, quality
│   ├── exporter.py       Parquet/CSV/Excel export
│   ├── specialized.py    Device-specific loaders (Dexcom, Libreview, ...)
│   ├── pregnancy_data.py Pregnancy-specific data handling
│   └── core.py           ModularGlucoseData facade
├── metrics/            📊 Clinical metric calculations
│   ├── basic.py          Mean, median, GMI, SD, ...
│   ├── time_in_range.py  TIR, TAR, TBR
│   ├── variability.py    CV, MAGE, MODD, CONGA, J-Index, ...
│   ├── pregnancy.py      GestationalDiabetes metrics
│   ├── targets.py        GlucoseTargets dataclass (diabetes/pregnancy)
│   └── __init__.py       Re-exports + ModularGlucoseMetrics facade
├── plotting/           📈 Visualizations
│   ├── agp.py            Ambulatory Glucose Profile
│   ├── daily_plots.py    Daily traces
│   └── statistical_plots.py Statistical summaries
├── analysis/           🧠 High-level orchestrators
│   └── core.py           GlucoseAnalysis facade
├── agata/              🤖 AGATA reference integration
│   ├── adapter.py
│   └── metrics.py
├── utils/              🛠 Helpers
└── __init__.py          🚪 Public API facade

tests/                  🧪 Test suite
├── conftest.py            Shared fixtures
├── unit/                  Per-module unit tests
├── integration/           Cross-module integration tests
├── clinical/              Reference regression tests
└── fixtures/              Synthetic / anonymized test data

examples/               💡 Usage examples (numbered)
docs/                   📚 Documentation (mkdocs)
scripts/                🔧 Dev / build / anonymization scripts
```

**Rule of thumb:** if you don't know where a file goes, ask the human. Don't invent new top-level folders.

---

## 3. Documentation Golden Rule

> **If you modify code in `cgmpy/`, you MUST verify and update the corresponding documentation before declaring the task complete.**

Git stores the history of *what* changed. Documentation stores the *why* and the *how*. Without it, the project becomes unmaintainable code.

| If you modify...                                 | You must update...                                                              |
|--------------------------------------------------|---------------------------------------------------------------------------------|
| `cgmpy/data/*.py` (loaders, parsers)             | `docs/user-guide/loading-data.md`, `docs/api/data.md`                           |
| `cgmpy/metrics/*.py` (new metric, formula change) | `docs/user-guide/computing-metrics.md`, `docs/api/metrics.md`, `CHANGELOG.md`  |
| `cgmpy/metrics/targets.py`                       | `docs/user-guide/computing-metrics.md` (targets section)                        |
| `cgmpy/plotting/*.py`                            | `docs/user-guide/visualization.md`                                              |
| `cgmpy/analysis/*.py`                            | `README.md` (quickstart if the facade changes)                                  |
| `cgmpy/agata/*.py`                               | `docs/user-guide/agata-integration.md`                                          |
| `pyproject.toml` (deps, version, classifiers)    | `docs/development/setup.md`, `CHANGELOG.md`                                     |
| Public API (anything in `cgmpy/__init__.py`)     | `docs/api/*.md`, `README.md`, `CHANGELOG.md`                                    |
| Anything user-facing (feat, fix)                 | `CHANGELOG.md` `[Unreleased]` section                                           |

**Process:**

1. **Before coding**, read the relevant docs to understand the current state.
2. **After implementing**, ask: "Did I change any behavior, API, flow, or config that a new contributor would need to know?"
3. If yes, update the docs in the **same commit** (or a follow-up commit, but before marking the task done).
4. If you add a new feature, follow the templates in `docs/user-guide/`.

---

## 4. Security — Non-Negotiable

CGMPy handles medical data. The following are **absolute prohibitions**:

1. ❌ **No hardcoded credentials, passwords, tokens, or production URLs** in any committed file.
2. ❌ **No PHI** (Protected Health Information) in tests, examples, fixtures, comments, logs, or docstrings.
3. ❌ **No real patient data** in `tests/fixtures/`, `examples/`, or anywhere in the repo. Use synthetic or fully anonymized data only.
4. ❌ **No `print(patient_id)` or `logger.info(glucose_array)`** with identifiers.
5. ❌ **No `pickle.loads()` on untrusted input** — arbitrary code execution risk.
6. ❌ **No `eval()` or `exec()` on user data** — same risk.
7. ❌ **No disable-authentication "temporarily for testing"** — fix the test, not the security.

If you discover a real vulnerability while working, **stop and report** to `javierpenatearrieta@gmail.com` per `SECURITY.md` rather than opening a public issue.

---

## 5. Coding Conventions

### Python

- **PEP 8** + **Ruff** (line length 100, see `pyproject.toml`).
- **snake_case** for functions, variables, modules, file names.
- **PascalCase** for classes, type aliases.
- **UPPER_SNAKE_CASE** for module-level constants.
- **Type hints** on all public functions and class methods.
- **Docstrings** (Google style) for all public functions, classes, and modules.
- **No `print()` in library code** — use the `logging` module.
- **No wildcard imports** (`from x import *`).
- **No `# type: ignore` without a comment** explaining why.
- **No commented-out code** — git remembers; delete it.
- **`pathlib.Path`**, not `os.path` joins.
- **f-strings**, not `%` or `.format()`.

### Imports

- Order: stdlib → third-party → first-party (`cgmpy`). Enforced by `ruff` (isort).
- No `from cgmpy import ...` inside `cgmpy/` modules — use relative imports within the package.

### Tests

- pytest, with markers (`slow`, `integration`, `clinical`, `agata`).
- Every public function must have at least one test.
- New metrics must include a **clinical regression test** when a published reference exists.
- Tests must not depend on the network or on local files outside `tests/fixtures/`.

### Commits

- **Conventional Commits** (enforced by `commitlint` via pre-commit):
  - `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, `chore:`, `ci:`, `build:`
- Subject ≤ 72 chars, imperative mood, no trailing period.
- Body explains the *why*, not the *what*.
- Footer for `BREAKING CHANGE:` and `Closes #N`.

---

## 6. Development Workflow

```
1. UNDERSTAND   — Read AGENTS.md, ROADMAP.md, relevant docs/* file
2. PLAN         — Propose a brief plan (≤ 5 bullets), wait for approval
3. BRANCH       — Create a branch from main: feat/* | fix/* | docs/* | chore/*
4. IMPLEMENT    — Small, focused commits, following the conventions above
5. TEST         — pytest + ruff check + ruff format --check
6. DOCUMENT     — Update docs/ as per the Documentation Golden Rule
7. CHANGELOG    — Add an entry under [Unreleased] if user-facing
8. PR           — Open a PR using the template, fill in the checklist
9. CI           — Wait for CI to pass, address review comments
10. MERGE       — Squash-merge, delete branch
```

---

## 7. AI Agent Roster (OpenCode)

CGMPy ships an OpenCode harness in `.opencode/`. Use it.

### Primary orchestrator: `cgmpy-architect`

- **Role:** Coordinates the SDD (Spec-Driven Development) flow.
- **File:** `.opencode/agents/cgmpy-architect.md`
- **Behavior:** Loads context (AGENTS.md, ROADMAP.md, CHANGELOG.md), analyzes the request, delegates to the right specialist agents, and **decides** PASS / FAIL on the final review.

### Specialist agents (read-only by default)

| Agent                | Domain                                       | File                                   |
|----------------------|----------------------------------------------|----------------------------------------|
| `@data-guardian`     | `cgmpy/data/*` (loaders, parsers, exporters) | `.opencode/agents/data-guardian.md`    |
| `@metrics-guardian`  | `cgmpy/metrics/*` (clinical calculations)    | `.opencode/agents/metrics-guardian.md` |
| `@plotting-guardian` | `cgmpy/plotting/*` (matplotlib / seaborn)    | `.opencode/agents/plotting-guardian.md`|
| `@agata-integrator`  | `cgmpy/agata/*` (AGATA parity)               | `.opencode/agents/agata-integrator.md` |
| `@security-guard`    | PHI, GDPR, secrets, dependencies             | `.opencode/agents/security-guard.md`   |

### Execution agents (edit in their domain)

| Agent              | Domain                                          | File                                |
|--------------------|-------------------------------------------------|-------------------------------------|
| `@test-engineer`   | Writes / maintains tests in `tests/`            | `.opencode/agents/test-engineer.md`  |
| `@docs-maintainer` | Syncs `docs/` with code changes                 | `.opencode/agents/docs-maintainer.md`|
| `@release-manager` | Version bumps, CHANGELOG, release-please        | `.opencode/agents/release-manager.md`|

### Skills (consult before acting)

| Skill                  | When to use                                                | File                                          |
|------------------------|------------------------------------------------------------|-----------------------------------------------|
| `git-advisor`          | Commits, branches, push, PRs                               | `.opencode/skills/git-advisor/SKILL.md`       |
| `docs-maintainer`      | Documentation updates triggered by code changes            | `.opencode/skills/docs-maintainer/SKILL.md`   |
| `security-guard`       | Pre-commit, sensitive data, secrets                        | `.opencode/skills/security-guard/SKILL.md`    |
| `post-coding-check`    | After every coding session, before commit                   | `.opencode/skills/post-coding-check/SKILL.md` |
| `python-lib`           | Python-library-specific conventions                        | `.opencode/skills/python-lib/SKILL.md`        |

### Commands (slash-commands in OpenCode)

| Command   | Purpose                                                  | File                              |
|-----------|----------------------------------------------------------|-----------------------------------|
| `/start`  | Load context at the beginning of a session               | `.opencode/commands/start.md`     |
| `/end`    | Persist learnings at the end of a session                | `.opencode/commands/end.md`       |
| `/test`   | Run the test suite (full or filtered)                    | `.opencode/commands/test.md`      |
| `/release`| Prepare a release (bumps version, updates CHANGELOG)     | `.opencode/commands/release.md`   |

### Configuration files

- **Base config:** `.opencode/opencode.jsonc` (default agent, permissions)
- **Per-domain rules:** `.opencode/rules/*.md`

### Typical flow

```
User: "Add a new CV-of-CV metric"
  → cgmpy-architect loads context
  → Delegates to @metrics-guardian to analyze feasibility and impact
  → Delegates to @security-guard to confirm no PHI implications
  → Delegates to @docs-maintainer to draft the spec update
  → Delegates to implementation (test-engineer + metrics-guardian in editor mode)
  → post-coding-check verifies quality
  → /end persists learnings
```

---

## 8. Pre-Commit Checklist

Before suggesting `git commit`, verify:

- [ ] No hardcoded credentials (`git grep -i "password\|secret\|api[_-]key" --cached`)
- [ ] All files are UTF-8 without BOM (no `�` U+FFFD)
- [ ] No `print()` in library code
- [ ] No new files that should be in `.gitignore`
- [ ] If `cgmpy/` was modified: docs updated per the Documentation Golden Rule
- [ ] If user-facing: `CHANGELOG.md` `[Unreleased]` updated
- [ ] On a `feat/`, `fix/`, `docs/`, or `chore/` branch — **never on `main`**
- [ ] Commit message follows Conventional Commits
- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `pytest` passes
- [ ] Pre-commit hooks all pass

---

## 9. Absolute Prohibitions

1. Creating new top-level folders without consensus.
2. Installing global dependencies without documenting them in `pyproject.toml`.
3. Changing the minimum Python version without consensus.
4. Committing real patient data (PHI), even in `git stash` or WIP branches.
5. Disabling authentication, rate limiting, or any security check "temporarily".
6. Force-pushing to `main`.
7. Amending commits that have already been pushed.
8. Adding `print()` debugging statements and forgetting to remove them.

---

## 10. Reference Documents (read before working in an area)

| Area                  | Document                                                  |
|-----------------------|-----------------------------------------------------------|
| Architecture          | `docs/architecture/system-overview.md`                    |
| Git workflow          | `docs/development/git-workflow.md`                        |
| Testing               | `docs/development/testing.md`                             |
| Release process       | `docs/development/release-process.md`                     |
| Security & PHI        | `SECURITY.md`                                             |
| Code of Conduct       | `CODE_OF_CONDUCT.md`                                      |
| Project roadmap       | `ROADMAP.md`                                              |
| Changelog             | `CHANGELOG.md`                                            |
| API reference         | `docs/api/*.md`                                           |

---

## 12. Deferred Work (do not propose proactively)

The following items are in the ROADMAP but must **not** be proposed,
started, or suggested by the agent unless the human maintainer
explicitly asks for them:

1. **MAGE refactor (v0.6.0)** — splitting `MAGE_Baghurst`, moving
   matplotlib code to plotting, `mypy --strict`. The agent must wait
   for the user to say "start MAGE" before touching any of this.
2. Any item under **v0.7.0 through v1.0.0** in ROADMAP.md — these are
   future milestones. Do not work on them or propose starting them.
   If the user asks, proceed normally.

### Rationale

The maintainer is solo and time-limited. Proactive feature proposals
create mental overhead ("should I do this now?") even when the answer
is no. By deferring roadmap items to explicit user requests, the agent
becomes a **responsive tool** rather than a **proactive distractor**.

---

## 11. Communication Style

- Be concise. Use tables and bullet lists. No essays.
- When proposing a change, give an effort estimate.
- When you detect a security risk, **alert immediately** with `🔴` and a one-line summary.
- When uncertain, ask a question rather than guessing.
- Always cite file paths in the form `path/to/file.py:line` for easy navigation.

---

_This document evolves with the project. If something is unclear, ask._
