# Library Structure Rules

> Per-domain rules for the `cgmpy-architect` and the execution agents.

## Repository Map (Sacred)

```
cgmpy/                ← Source code (the library itself)
├── data/             📥 Loaders, parsers, processors, exporters
├── metrics/          📊 Clinical metric calculations
├── plotting/         📈 Visualizations
├── analysis/         🧠 High-level orchestrators
├── agata/            🤖 AGATA reference integration
├── utils/            🛠 Helpers
└── __init__.py        🚪 Public API facade

tests/                🧪 Test suite
├── conftest.py        Shared fixtures
├── unit/              Per-module unit tests
├── integration/       Cross-module integration tests
├── clinical/          Reference regression tests
└── fixtures/          Synthetic / anonymized test data

examples/             💡 Usage examples (numbered)
docs/                 📚 Documentation
scripts/              🔧 Dev / build / anonymization scripts
```

## Rules of Placement

| If you want to put...               | Goes in...                          | Does NOT go in...                  |
|-------------------------------------|-------------------------------------|------------------------------------|
| Source code                        | `cgmpy/<submodule>/`               | Root, `src/`, scattered modules    |
| New clinical metric                | `cgmpy/metrics/<category>.py`       | `cgmpy/analysis/`, `cgmpy/data/`   |
| New device loader                  | `cgmpy/data/specialized.py`         | `cgmpy/metrics/`                   |
| New plot                           | `cgmpy/plotting/<plotter>.py`       | `cgmpy/analysis/`                  |
| Unit tests                         | `tests/unit/test_<submodule>/`      | Root, `tests/test_*.py`            |
| Integration tests                  | `tests/integration/`                | `tests/unit/`                      |
| Clinical regression tests          | `tests/clinical/`                   | `tests/integration/`               |
| Synthetic / anonymized CSVs        | `tests/fixtures/`                   | Root, `examples/`, repo root       |
| Numbered examples                  | `examples/NN_<topic>/`              | `examples/<loose>.py`              |
| User guide                         | `docs/user-guide/`                  | `README.md`, scattered             |
| API reference (auto)               | `docs/api/`                         | Hand-edited                        |
| Development guide                  | `docs/development/`                 | `README.md` (only quickstart)      |
| ADRs                               | `docs/architecture/decisions/`      | `docs/architecture/*.md`           |
| Anonymization / dev scripts        | `scripts/`                          | Root, `tools/`                     |

## Top-Level Folders

The following top-level folders are **fixed** — do not create new ones without
the maintainer's explicit approval:

- `cgmpy/`
- `tests/`
- `examples/`
- `docs/`
- `scripts/`
- `.opencode/`
- `.github/`
- `config/`

If a request doesn't fit, **ask the maintainer** before creating a new folder.

## Public API

Symbols exported from `cgmpy/__init__.py` are the public API and are **stable**.

To change one (rename, signature change, removal):

1. Add a `DeprecationWarning` in the same release.
2. Document the change in `CHANGELOG.md` and `docs/development/release-process.md`.
3. After 1 minor version, remove the old symbol.

## Module Naming

- `snake_case` for modules, functions, variables.
- `PascalCase` for classes, type aliases, type variables.
- `UPPER_SNAKE_CASE` for module-level constants.
- File names match the primary class: `cgmpy/data/loader.py` → `class DataLoader`.
- One primary class per file (exceptions are documented).

## File Headers

Optional. If you add one, keep it short:

```python
"""Data loaders for CGMPy.

Supports CSV, Parquet, and in-memory DataFrame ingestion with automatic
delimiter and header detection.
"""
```

No license headers (the package is MIT — see `LICENSE`).

## Things You Must Not Do

1. ❌ Create new top-level folders.
2. ❌ Add `print()` to library code.
3. ❌ Import `cgmpy.*` absolutely inside `cgmpy/` (use relative imports).
4. ❌ Add a new device loader as a function — subclass `DataLoader`.
5. ❌ Add a new metric without a docstring and a test.
6. ❌ Add a new dependency without it appearing in `pyproject.toml`.
