# Testing

CGMPy uses **pytest** for its test suite, with **ruff** for linting and
formatting, and **mypy** (relaxed) for type checking.

## Quick reference

```bash
# Run everything
make test

# Quick smoke (skip slow + clinical + agata)
make test-fast

# Unit only
make test-unit

# Integration only
make test-integration

# With coverage
make test-coverage
```

## Test layout

```
tests/
├── conftest.py            # Shared fixtures
├── unit/                  # Fast, no I/O, no network
│   ├── test_data/
│   ├── test_metrics/
│   ├── test_plotting/
│   └── test_agata/
├── integration/           # Cross-module, may use fixtures
├── clinical/              # Slow, against published references
└── fixtures/
    └── data/              # Synthetic / anonymized CSVs
```

## Markers

Defined in `pyproject.toml` → `[tool.pytest.ini_options].markers`:

- `slow` — long-running (> 5s).
- `integration` — cross-module.
- `clinical` — clinical regression tests.
- `agata` — depends on the `agata` optional dependency.

```bash
pytest -m "not slow"           # quick smoke
pytest -m "not agata"          # CI without AGATA
pytest -m "clinical"           # only clinical
```

## Writing tests

- One test file per source module:
  `cgmpy/metrics/variability.py` → `tests/unit/test_metrics/test_variability.py`.
- Test names: `test_<unit>_<behavior>`.
- Use fixtures from `tests/conftest.py`. Add new fixtures there if reusable.
- Use `pytest.approx(expected, abs=1e-6)` for float comparisons.
- No network access, no local file paths outside `tests/fixtures/`.

Example:

```python
import pytest
from cgmpy.metrics import ModularGlucoseMetrics


def test_mean_is_close_to_known_value(stable_glucose_df):
    """The mean of a constant-100 trace is exactly 100."""
    metrics = ModularGlucoseMetrics(stable_glucose_df)
    assert metrics.basic().mean() == pytest.approx(100.0, abs=1e-6)
```

## Clinical regression tests

For each new clinical metric, add at least one test that:

1. Loads a **published reference dataset** (OhioT1DM, REPLACE-BG, etc.).
2. Computes the metric.
3. Asserts the result is within a tight tolerance of the published value.

Place these under `tests/clinical/`. Mark them `@pytest.mark.clinical`
and `@pytest.mark.slow`.

If no published reference exists, use a **known-answer synthetic test**:

```python
def test_mean_constant():
    data = make_glucose_series([100, 100, 100, 100])
    assert data.basic().mean() == pytest.approx(100.0)
```

## AGATA parity tests

```python
import pytest

@pytest.mark.agata
def test_mage_matches_agata(synthetic_cgm):
    """CGMPy MAGE matches AGATA's mage within 1e-6."""
    import agata
    from cgmpy import ModularGlucoseMetrics

    cgmpy_result = ModularGlucoseMetrics(synthetic_cgm).variability().mage()
    agata_result = agata.mage(synthetic_cgm)
    assert cgmpy_result == pytest.approx(agata_result, abs=1e-6)
```

## Coverage

CGMPy aims for **≥ 70 %** line coverage on the `cgmpy/` package, with
higher targets on the metrics layer (≥ 85 %). The CI gate is configured
in `pyproject.toml` → `[tool.coverage.report].fail_under`.

```bash
# Generate HTML report
make test-coverage
# Open htmlcov/index.html in your browser
```

## Pre-commit

Before pushing, pre-commit hooks will run:

- `ruff check` and `ruff format`.
- `interrogate` (docstring coverage ≥ 70 %).
- `commitlint` (Conventional Commits on the commit message).
- A local hook that warns if `cgmpy/` changes but `docs/` and
  `CHANGELOG.md` do not.

```bash
pre-commit run --all-files
```

## CI

GitHub Actions runs on every push and PR:

- `ci.yml` — multi-OS (ubuntu/windows/macos), multi-Python (3.10/3.11/3.12) test matrix.
- `security` job — bandit + pip-audit.
- `lint` job — ruff check + format check.
- `coverage` job — uploads to Codecov.

See `.github/workflows/ci.yml`.

## See also

- [`AGENTS.md` § Testing](https://github.com/javierpa95/cgmpy/blob/main/AGENTS.md).
- `.opencode/rules/testing.md` — agent testing rules.
- `.opencode/agents/test-engineer.md` — agent role.
