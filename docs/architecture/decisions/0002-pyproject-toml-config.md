# ADR 0002 — Single-source configuration with pyproject.toml

* Status: Accepted
* Date: 2026-06-02
* Deciders: @javierpa95

## Context and Problem Statement

CGMPy needs to manage build configuration, tool settings (ruff, pytest,
mypy, coverage), and project metadata across multiple files. Historically
each tool required its own config file (`setup.cfg`, `.flake8`,
`pytest.ini`, `.coveragerc`, etc.). This caused fragmentation: settings
were duplicated, contradictory, or fell out of sync when the project
grew.

We need a single source of truth that is standard, machine-readable,
and understood by the Python ecosystem tooling.

## Considered Options

* **pyproject.toml (PEP 517/518/621)** — the modern Python standard.
  All tool config goes in `[tool.*]` tables; build system is declared
  via `[build-system]`.
* **setup.cfg + tool-specific files** — traditional approach used
  before pyproject.toml became the norm.
* **pyproject.toml for build only + separate tool files** — hybrid:
  build metadata in pyproject.toml, but tool settings stay in their
  legacy locations.

## Decision Outcome

Chosen option: **pyproject.toml (full adoption)**, because:

1. It is the only option that makes `pip install -e ".[dev,docs]"` work
   without a `setup.py` or `setup.cfg` file.
2. It is natively supported by ruff, pytest, mypy, coverage, and
   setuptools via `[tool.*]` tables.
3. It eliminates five separate config files and the cognitive overhead
   of knowing which config lives where.
4. It is future-proof: the Python ecosystem is consolidating around
   pyproject.toml as the universal config file.

### Consequences

* Good, because there is exactly one place to look for project
  configuration.
* Good, because new contributors only need to understand one file.
* Good, because CI scripts can read the version from
  `pyproject.toml` via `tomllib` (stdlib in Python 3.11+).
* Bad, because not all legacy tools support pyproject.toml yet
  (e.g., some pre-commit hooks). We keep a minimal
  `.pre-commit-config.yaml` for those.

## Pros and Cons of the Options

### pyproject.toml (full adoption)

* Good, standard and supported by all modern Python tooling.
* Good, allows `[project]` table for metadata, `[build-system]` for
  build, `[tool.*]` for tool config.
* Good, supports optional dependencies (`[project.optional-dependencies]`).
* Bad, older tools (flake8, isort) did not support it at first; but
  we migrated away from those tools (ruff replaces both).

### setup.cfg + tool files

* Good, well-understood.
* Bad, fragmented: pytest.ini, mypy.ini, .coveragerc, .flake8, ...
* Bad, no standard way to declare build backend.

### Hybrid (pyproject.toml for build only)

* Good, incremental migration.
* Bad, same fragmentation problem for tool config.
* Bad, confusing: "why is some config in pyproject.toml and some not?"
