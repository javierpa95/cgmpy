# Pure Functions: The CGMPy Design Philosophy

A pure function is like a vending machine: you put in the same coins, you get the same snack, every time. It doesn't remember what you bought last time, it doesn't play music, and it certainly doesn't write to your diary. It just takes an input and returns an output, period.

## What is a pure function?

A function is **pure** when it satisfies two rules:

1. **Same input always gives the same output** — no hidden state, no random numbers, no reading from a file that could change.
2. **No side effects** — it does not modify any external variable, write to disk, print to the console, or mutate its arguments.

```python
# Pure
def add(a: int, b: int) -> int:
    return a + b

# Impure (depends on external state)
current_user: str = "alice"
def greet() -> str:
    return f"Hello, {current_user}"

# Impure (side effect)
def log_add(a: int, b: int) -> int:
    print(f"adding {a} and {b}")
    return a + b
```

A pure function has no `self` parameter — it doesn't belong to an object. It stands alone, asks for exactly what it needs, and returns exactly what it promises.

## Why pure functions for science?

Scientific code needs to be *trustworthy*. Pure functions help because:

### Testable

No setup, no mocking, no object instantiation:

```python
def mean(glucose: pd.Series) -> float:
    return glucose.mean()

# Test — one line
assert mean(pd.Series([100, 110, 120])) == 110.0
```

Compare this to a mixin method that requires a full object with `self.data` properly wired:

```python
# Testing a mixin requires:
obj = SomeBigClass()
obj.data = pd.DataFrame({"glucose": [100, 110, 120]})
assert obj.mean() == 110.0
```

### Composable

Pure functions chain naturally:

```python
result = gmi(mean(glucose))
```

No intermediate object, no method chaining gymnastics.

### Parallelizable

No shared state means no race conditions. You can call `mean()` on a hundred threads with different data and never worry about corrupted state.

### Documentable

The function signature is the documentation:

```python
def mean(glucose: pd.Series) -> float:
    """Mean glucose value."""
```

You know exactly what goes in and what comes out. No `self` magic, no hidden attributes.

### No MRO surprises

Method Resolution Order (MRO) is the bane of multiple inheritance. With pure functions, there is no inheritance — you call `mean(series)`, not `obj.mean()`. No diamond problem, no confusing super() chains.

## Code example — before vs after

CGMPy is migrating from mixin-based classes to pure functions. Here's what that looks like:

```python
# BEFORE — mixin pattern
class BasicMetrics:
    """Requires self.data to be a DataFrame with a 'glucose' column."""

    def mean(self) -> float:
        return self.data["glucose"].mean()

    def gmi(self) -> float:
        return 12.71 * self.mean() + 3.16
```

```python
# AFTER — pure functions
import pandas as pd

def mean(glucose: pd.Series) -> float:
    """Mean glucose value.

    Args:
        glucose: A series of glucose readings in mg/dL.

    Returns:
        The arithmetic mean of the glucose values.
    """
    return glucose.mean()


def gmi(glucose: pd.Series) -> float:
    """Glucose Management Indicator (GMI).

    Args:
        glucose: A series of glucose readings in mg/dL.

    Returns:
        Estimated GMI percentage.
    """
    return 12.71 * mean(glucose) + 3.16
```

## Real CGMPy transition

The old mixin still works, but now it delegates internally:

```python
# The mixin now wraps the pure function
from cgmpy.metrics import mean as pure_mean

class BasicMetrics:
    """Convenience wrapper for object-oriented users."""

    def mean(self) -> float:
        return pure_mean(self.data["glucose"])

    def gmi(self) -> float:
        return pure_mean(self.data["glucose"])
```

This means:
- New contributors only need to learn the pure functions.
- Existing code using `obj.mean()` continues to work.
- The pure functions live in `cgmpy/metrics/` and are the single source of truth.

## When NOT to use pure functions

Pure functions are not a silver bullet. Sometimes state is the right tool:

| Situation | Recommended approach | Example |
|---|---|---|
| **Lazy loading / caching** | Class with state | A `Loader` that caches parsed files |
| **Expensive initialisation** | Class with state | A database connection pool |
| **Plotting** | Stateful (matplotlib) | `matplotlib.pyplot.figure()` |
| **I/O** | Side effects are the point | File writers, HTTP clients |

CGMPy uses classes for orchestration (`GlucoseAnalysis`) and data loading (`GlucoseData`), but the leaf-level computations are pure functions. This is the *functional core, imperative shell* pattern.

## Comparison with other scientific libraries

CGMPy is not inventing a new paradigm — it is aligning with the wider scientific Python ecosystem:

| Library | Example | Pure? |
|---|---|---|
| NumPy | `np.mean(array)` | Yes |
| SciPy | `scipy.stats.ttest_ind(a, b)` | Yes |
| scikit-learn | `sklearn.metrics.accuracy_score(y_true, y_pred)` | Yes |
| Pandas | `pd.Series([1, 2, 3]).mean()` | No (method on object) |
| CGMPy (new) | `mean(glucose: pd.Series) -> float` | Yes |

Notice that SciPy and scikit-learn already use this pattern. Even Pandas offers standalone functions in `pd.api` (e.g. `pd.api.types.is_numeric_dtype`). CGMPy is following the same path: the library provides standalone, testable, composable functions, with object-oriented wrappers where ergonomics demand them.

In short: **pure functions are the atoms of CGMPy**. Classes are the molecules. Build with atoms, compose with molecules.
