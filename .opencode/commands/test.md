# /test — Run Test Suite

## Description

Runs the CGMPy test suite, with optional filters.

## Usage

```
/test                    # Run all tests
/test unit               # Only unit tests
/test integration        # Only integration tests
/test clinical           # Only clinical regression tests
/test fast               # Skip slow and clinical markers
/test tests/unit/test_basic_metrics.py::test_mean  # Specific test
```

## What it does

Maps the argument to pytest invocations:

| Arg           | Command                                                |
|---------------|--------------------------------------------------------|
| (none)        | `pytest -v`                                            |
| `unit`        | `pytest tests/unit -v`                                 |
| `integration` | `pytest tests/integration -v`                          |
| `clinical`    | `pytest tests/clinical -v`                             |
| `fast`        | `pytest -m "not slow and not clinical" -v`             |
| `coverage`    | `pytest --cov=cgmpy --cov-report=term-missing -v`      |
| `<file::node>`| `pytest <file::node> -v`                               |

## When to use

- After modifying `cgmpy/`.
- Before `git commit` (quick smoke test).
- After pulling new code.
- In CI (the workflow uses these directly).

## Pre-test checklist

- [ ] `pip install -e ".[dev,agata]"` was run.
- [ ] No uncommitted changes that would affect tests.
- [ ] `tests/fixtures/` contains expected CSVs.
