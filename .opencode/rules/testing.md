# Testing Rules

> Per-domain rules for the `test-engineer` agent and all contributors.

## Coverage Targets

| Layer              | Minimum line coverage |
|--------------------|-----------------------|
| `cgmpy/metrics/`   | 85 %                  |
| `cgmpy/data/`      | 80 %                  |
| `cgmpy/plotting/`  | 70 % (plots are visual; cover the *plumbing*) |
| `cgmpy/analysis/`  | 75 %                  |
| `cgmpy/agata/`     | 80 % (skipped if AGATA not installed) |
| Overall (`cgmpy/`) | **70 %** (CI gate)    |

See `pyproject.toml` → `[tool.coverage.report].fail_under` for the actual gate.

## Test Layout

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
└── fixtures/              # Synthetic / anonymized data
```

## Test Markers

Defined in `pyproject.toml` → `[tool.pytest.ini_options].markers`:

- `slow` — long-running (> 5s). Skip on quick checks.
- `integration` — cross-module.
- `clinical` — clinical regression tests.
- `agata` — depends on the `agata` optional dependency.

```bash
pytest -m "not slow"           # quick smoke
pytest -m "not agata"          # CI without AGATA
pytest -m "clinical"           # only clinical
```

## Conventions

- One test file per source module.
  - `cgmpy/metrics/variability.py` → `tests/unit/test_metrics/test_variability.py`.
- Test names: `test_<unit>_<behavior>`.
- Use fixtures from `tests/conftest.py` for shared data.
- Use `pytest.approx(expected, abs=1e-6)` for float comparisons.
- No network access. No files outside `tests/fixtures/`.

## Clinical Regression Tests

For each new clinical metric, add at least one test that:

1. Loads a **published reference dataset** (OhioT1DM, REPLACE-BG, etc.).
2. Computes the metric.
3. Asserts the result is within a tight tolerance of the published value.

Place these under `tests/clinical/`. Mark them `@pytest.mark.clinical` and
`@pytest.mark.slow`.

If no published reference is available, add a **known-answer synthetic test**:

```python
def test_mean_constant():
    """Mean of [100, 100, 100, 100] is exactly 100."""
    data = GlucoseSeries([100, 100, 100, 100])
    assert data.mean() == pytest.approx(100.0)
```

## AGATA Parity Tests

For each metric that has an AGATA equivalent, add a parity test:

```python
@pytest.mark.agata
def test_mage_matches_agata(synthetic_cgm):
    cgmpy_result = cgmpy.metrics.variability.mage(synthetic_cgm)
    agata_result = agata.mage(synthetic_cgm)
    assert cgmpy_result == pytest.approx(agata_result, abs=1e-6)
```

These are run in CI only when `agata` is installed.

## Pre-Merge Checklist for Tests

- [ ] All new tests pass locally.
- [ ] No real patient data in fixtures.
- [ ] Coverage for the modified module is non-decreasing.
- [ ] Test names are descriptive.
- [ ] Markers (`slow`, `integration`, `clinical`, `agata`) are applied where appropriate.
- [ ] No network access, no local file paths outside `tests/fixtures/`.

## Things You Must Not Do

- ❌ Skip a test with `@pytest.mark.skip` to make CI green — fix it.
- ❌ Add a test that depends on the network.
- ❌ Use a real CGM export in `tests/fixtures/`.
- ❌ Use `assert` outside test functions in production code (use exceptions).
- ❌ Add a test that requires interactive input.
