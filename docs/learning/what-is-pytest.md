# What is Pytest?

## What is it?

A test runner. You write a function that checks: "if I call `mean(Series[100, 110, 120])`, do I get 110?" Pytest runs that check and tells you if it passed or failed.

More formally, pytest is a testing framework for Python. It discovers and runs test functions, collects results, and gives you a clean report of what passed, what failed, and why.

## Why we use it

When you change the GMI formula to fix a bug, how do you know you didn't break TIR? Run the tests. If they pass, you're safe.

Manual testing is slow, error-prone, and doesn't scale. As CGMPy grows, the number of things that can break grows with it. Automated tests catch regressions instantly. You make a change, run `pytest`, and get a green light (or a red one telling you exactly what you broke).

## Writing a test

```python
# In tests/unit/test_metrics/test_basic_functions.py
from cgmpy.metrics.basic import mean
import pandas as pd

def test_mean_of_three_values():
    glucose = pd.Series([100.0, 110.0, 120.0])
    assert mean(glucose) == 110.0
```

That's it. A function named `test_...` with an `assert` statement. Pytest finds it automatically, runs it, and reports whether the assertion passes or fails.

A failing test looks like this:

```python
def test_mean_of_empty_series():
    glucose = pd.Series([], dtype=float)
    assert mean(glucose) == 0.0  # What should happen here? NaN? 0?
```

This highlights something important: tests force you to decide what the correct behaviour is for edge cases. Writing a test for empty input makes you think about what your function should do in every scenario.

## Test structure in CGMPy

```
tests/
├── unit/          # Fast tests (no network, no disk)
│   ├── test_data/     # Test loaders, processors
│   ├── test_metrics/  # Test metric calculations
│   └── test_plotting/ # Test plots (using Agg backend)
├── integration/   # Tests that combine modules
├── clinical/      # Tests against published references
└── fixtures/      # Test data (synthetic CSVs)
```

- **Unit tests** are fast and isolated. They test a single function or module.
- **Integration tests** check that modules work together correctly.
- **Clinical tests** compare CGMPy's output against published reference values. If a paper says "for this input, the result is X", we write a test that asserts that.

## Fixtures — reusable test data

```python
# In tests/conftest.py
import pytest
import pandas as pd

@pytest.fixture
def stable_glucose_df():
    """Returns a DataFrame with constant 100 mg/dL glucose."""
    return pd.DataFrame({"glucose": [100.0] * 288})  # 24 hours at 5-min intervals

@pytest.fixture
def variable_glucose_series():
    """Returns a Series with typical daily variation."""
    return pd.Series(
        [100.0] * 48 +     # night: 4 hours at 100
        [120.0] * 36 +     # morning rise
        [90.0] * 24 +      # lunch dip
        [140.0] * 36 +     # afternoon
        [110.0] * 48 +     # evening
        [100.0] * 96       # night
    )
```

Fixtures let you define test data once and reuse it across many tests. They can be as simple or as complex as you need. Pytest handles setup and teardown automatically.

Using a fixture in a test is just a parameter name:

```python
def test_mean_of_stable_glucose(stable_glucose_df):
    result = mean(stable_glucose_df["glucose"])
    assert result == 100.0
```

## Running tests

```bash
pytest                          # all tests
pytest -m "not slow"            # quick smoke test
pytest tests/unit/test_metrics/ # specific folder
pytest -k "test_mean"           # by name
```

Markers like `slow` let you categorise tests. Run the full suite before a release, or just the fast ones while you're iterating.

## What is coverage?

Coverage tells you what percentage of your code is exercised by tests. CGMPy aims for 80%+. Think of it as: "does every line of this medical software actually get tested?"

```bash
pytest --cov=cgmpy --cov-report=term-missing
```

This runs the tests and shows you which lines of CGMPy were never executed. Lines in red are untested — they are potential hiding spots for bugs. Coverage doesn't guarantee correctness (you can have 100% coverage and still have bugs), but low coverage guarantees that plenty of code has never been run in a test.

## Why it's good for learners

Tests are the best documentation. Reading a test for `MAGE()` tells you exactly what the function should do. It shows you:

- What inputs it expects
- What output it produces
- How edge cases are handled
- What the function's contract actually is

A test suite is an executable specification. When the code and the docs disagree, the tests are usually right (because they actually run and fail if they're wrong).

If you're new to the codebase, start by reading the tests. They will teach you how every function is supposed to behave.
