# Documentation Rules

> Per-domain rules for the `docs-maintainer` agent and all contributors.

## The Golden Rule

> **If you modify code in `cgmpy/`, you MUST verify and update the corresponding documentation before declaring the task complete.**

Git stores the history of *what* changed. Documentation stores the *why* and the *how*.

## What to Update (Quick Reference)

| If you modify...                                 | You must update...                                                              |
|--------------------------------------------------|---------------------------------------------------------------------------------|
| `cgmpy/data/*.py` (loaders, parsers)             | `docs/user-guide/loading-data.md`, `docs/api/data.md`                           |
| `cgmpy/metrics/*.py` (new metric, formula change) | `docs/user-guide/computing-metrics.md`, `docs/api/metrics.md`, `CHANGELOG.md`  |
| `cgmpy/metrics/targets.py`                       | `docs/user-guide/computing-metrics.md` (targets section)                        |
| `cgmpy/plotting/*.py`                            | `docs/user-guide/visualization.md`                                              |
| `cgmpy/analysis/*.py`                            | `README.md` (quickstart if the facade changes)                                  |
| `cgmpy/agata/*.py`                               | `docs/user-guide/agata-integration.md`                                          |
| `pyproject.toml` (deps, version, classifiers)    | `docs/development/setup.md`, `CHANGELOG.md`                                     |
| Public API (anything in `cgmpy/__init__.py`)     | `docs/api/*.md` (auto), `README.md`, `CHANGELOG.md`                             |
| Anything user-facing (feat, fix)                 | `CHANGELOG.md` `[Unreleased]` section                                           |

## Style Guide

- **Language**: English. Markdown.
- **Whitespace**: `trim_trailing_whitespace = false` (some markdown engines
  use trailing spaces for line breaks). Enforced by `.editorconfig`.
- **Line endings**: LF. Enforced by `.gitattributes`.
- **Links**: Use **relative** paths: `[Setup](setup.md)` not
  `[Setup](/docs/development/setup.md)`.
- **Code blocks**: Must be runnable in a fresh venv. Include imports.
- **Citations**: For clinical metrics, include paper, year, and DOI.
- **"See also"** section at the end of each user-guide page.

## Docstring Style

Public functions and classes use **Google-style** docstrings:

```python
def metric(data: GlucoseData, targets: GlucoseTargets) -> float:
    """One-line summary in imperative mood.

    Optional longer description, wrapped at 100 chars.

    Args:
        data: A ``GlucoseData`` instance.
        targets: Glucose cutoffs (e.g., ``GlucoseTargets.standard()``).

    Returns:
        The percentage (0–100) spent in the target range.

    Raises:
        ValueError: If ``data`` has fewer than 24 valid samples.

    Example:
        >>> targets = GlucoseTargets.standard()
        >>> metric(data, targets)
        72.4
    """
```

`interrogate` (in `[dev]` dependencies) checks docstring coverage. See
`pyproject.toml` for the threshold.

## CHANGELOG Conventions (Keep a Changelog)

Under `## [Unreleased]`:

```markdown
### Added
- New feature description

### Changed
- Behavior change (with migration note)

### Fixed
- Bug fix

### Removed
- Deprecated feature (with migration note)

### Security
- Security fix (anonymized; do not describe the exact vulnerability)
```

Release-please (Phase 5) will move `[Unreleased]` entries into a versioned
section on release. The `[Unreleased]` section is **always at the top**.

## API Reference

`docs/api/*.md` is **auto-generated** by `mkdocstrings` from the docstrings.

- Do **not** edit it by hand.
- Improve the docstring in the source instead.
- Re-export the symbol from `cgmpy/<submodule>/__init__.py` if needed.
- Reference it in the relevant `docs/user-guide/*.md` page.

## Process

1. **Before coding**, read the relevant `docs/` files to understand the
   current state. If they don't exist yet, plan to create them.
2. **After coding**, ask: "Did I change any behavior, API, flow, or config
   that a new contributor or user would need to know?"
3. If yes, update the corresponding doc in the **same commit** (or a
   follow-up commit, but before marking the task done).
4. **Add a `[Unreleased]` entry** in `CHANGELOG.md` if the change is
   user-facing.

## What You Must Not Do

- ❌ Document a feature that doesn't exist yet.
- ❌ Edit `docs/api/*.md` by hand (it's auto-generated).
- ❌ Add new top-level files in `docs/` (use the existing categories).
- ❌ Use absolute URLs in links (use relative paths).
- ❌ Add screenshots without alt text.
- ❌ Use Spanish / non-English in `docs/` (English is the project standard).
