# Type Hints for Science

## What are type hints?

Type hints were introduced in Python 3.5 via [PEP 484](https://peps.python.org/pep-0484/). They are optional annotations that tell you (and tools) what type a variable, parameter, or return value should be.

They are **NOT** enforced at runtime. Python ignores them entirely when running your code. They exist for developers and tools — mypy, pyright, pytype, and your IDE's autocomplete.

```python
# This runs fine — Python does not check types at runtime
def mean(glucose: pd.Series) -> float:
    return glucose.mean()

mean("hello")  # RuntimeError: 'str' object has no attribute 'mean'
```

The error above is a runtime error, not a type error. A type checker like mypy would have caught it *before* you ran the code.

## Why type hints for scientific code?

### A function signature IS documentation

```python
def mean(glucose: pd.Series) -> float:
```

Without reading a single line of the docstring or body, you know:
- **Input:** a Pandas Series
- **Output:** a single float

Compare with:

```python
def mean(glucose):
    ...
```

What is `glucose`? A list? A NumPy array? A DataFrame column? You have to read the body or the docstring to find out. Type hints make the contract explicit.

### Catch bugs before runtime

```python
def gmi(mean_glucose: float) -> float:
    return round(3.31 + 0.02392 * mean_glucose, 2)

gmi("hello")  # ❌ mypy: Argument 1 has incompatible type "str"; expected "float"
```

This fails at mypy time, not at 3am when your batch job crashes.

### IDE autocomplete

VS Code, PyCharm, and other editors use type hints to show available methods. When you type `glucose.` on a `pd.Series`, you get Series methods in the dropdown. Without the hint, the IDE has no idea what `glucose` is.

### Refactoring confidence

Renaming a parameter? Changing a return type? `mypy` tells you every caller that needs updating. No manual hunting through the codebase.

## Example: before and after

```python
# WITHOUT type hints (what does this function need? what does it return?)
def gmi(m):
    return round(3.31 + 0.02392 * m, 2)


# WITH type hints (crystal clear)
def gmi(mean_glucose: float) -> float:
    """Glucose Management Index.

    Reference: DOI: 10.2337/dc18-1581
    """
    return round(3.31 + 0.02392 * mean_glucose, 2)
```

The second version tells you everything: `gmi` takes a single float (mean glucose in mg/dL) and returns a float (the GMI percentage). You could use this function correctly without ever reading the docstring.

## Common type hints for scientific Python

```python
# Basic types
def cv(sd: float, mean: float) -> float: ...

# Pandas
def mean(glucose: pd.Series) -> float: ...

# NumPy
def transform(values: np.ndarray) -> np.ndarray: ...

# Optional
def tir(glucose: pd.Series, low: float = 70, high: float | None = None) -> float: ...

# Dictionaries (for complex returns)
def report() -> dict[str, float]: ...

# Union types (Python 3.10+ syntax)
def load(path: str | Path) -> GlucoseData: ...

# Type aliases (for complex types)
GlucoseArray: TypeAlias = pd.Series | np.ndarray
```

### Python version note

The `float | None` and `str | Path` syntax requires Python 3.10+. For Python 3.9 and earlier, use `Optional[float]` and `Union[str, Path]` from `typing`. CGMPy targets Python 3.10+, so the new syntax is preferred throughout the codebase.

## What mypy checks

```python
def mean(glucose: pd.Series) -> float:
    return glucose.mean()

mean(pd.Series([100, 110, 120]))  # ✅ OK
mean("hello")                      # ❌ mypy: Argument 1 has incompatible type "str"; expected "pd.Series"
```

mypy also catches:

```python
def gmi(mean_glucose: float) -> float:
    return round(3.31 + 0.02392 * mean_glucose, 2)

result: str = gmi(120.0)  # ❌ mypy: Incompatible types in assignment (expected "str", got "float")
```

mypy checks both **callers** and **implementations**. If you annotate a return type as `float` but return a `str`, mypy will tell you.

## Gradual typing — CGMPy's approach

Not all code needs types immediately. CGMPy uses `disallow_untyped_defs = false` in its mypy configuration, which means you can add types gradually.

- **Start with the public API** — every function exported from `cgmpy.metrics` has full type hints.
- **Then internal helper functions** — as you touch a file, add types.
- **Pure functions are easy to type** — they take arguments and return values with no hidden state.

This is called **gradual typing**: adding type annotations incrementally as the codebase evolves. A partially typed codebase is still better than an untyped one because mypy checks the parts that are annotated.

```python
# Partially typed — mypy checks inner(), but not outer()
def outer(x):
    return inner(x)

def inner(x: float) -> float:
    return x * 2
```

## Common pitfalls in scientific typing

### Pitfall 1: pd.Series vs np.ndarray

They are different types with different methods. Be specific.

```python
# BAD — what is this?
def bad(glucose):
    ...

# BAD — lies to the caller
def bad(glucose: np.ndarray) -> float:
    # Actually expects a pd.Series for .mean()
    ...

# GOOD — accurate
def good(glucose: pd.Series) -> float:
    return glucose.mean()
```

If your function truly accepts both, use a type alias:

```python
GlucoseArray: TypeAlias = pd.Series | np.ndarray

def mean(glucose: GlucoseArray) -> float:
    return np.mean(glucose)
```

### Pitfall 2: Float vs int

Python's `int` is a subtype of `float` for type checking purposes, but many scientific functions should declare `float` as the return type even when the result is mathematically an integer.

```python
def cv(glucose: pd.Series) -> float:
    """Coefficient of variation as a percentage."""
    result = glucose.std() / glucose.mean() * 100
    return result  # Always a float, even if 0.0
```

Declaring `float` is safer than `int` because:
- Division always produces floats in Python 3 (`100` returns `float`)
- Callers should not assume integer precision

### Pitfall 3: None handling

```python
# BAD — caller might pass None and find out at runtime
def m_value(glucose: pd.Series, reference: int = 90) -> float:
    ...

# GOOD — explicit about what can be None
def m_value(glucose: pd.Series, reference: int | None = None) -> float:
    """M-value, a measure of glycemic stability.

    Args:
        glucose: A series of glucose readings in mg/dL.
        reference: Target glucose value. Defaults to 90 mg/dL if None.
    """
    ref = reference if reference is not None else 90
    ...
```

If a parameter can be `None`, say so in the type hint. Otherwise, mypy assumes it is always the declared type.

### Pitfall 4: Any and ignorance

```python
# BAD — escapes type checking entirely
from typing import Any

def mean(glucose: Any) -> Any:
    return glucose.mean()
```

`Any` tells mypy to skip all checks on this value. It is useful during migration but should not be permanent. Use specific types wherever possible.

## How the CGMPy refactor improves type safety

### Before — mixin pattern

```python
class BasicMetrics:
    """Requires self.data to be a DataFrame with a 'glucose' column.

    mypy has no idea what self.data is at this point.
    """

    def mean(self) -> float:
        return self.data["glucose"].mean()
        # ❌ mypy: Cannot access attribute "glucose" for "Any"
```

With the mixin pattern, `self.data` could be anything — a DataFrame, a dict, a string. mypy cannot verify that `"glucose"` is a valid column or that `.mean()` exists. This is because `self.data` is typically defined in a subclass or injected at runtime, and mypy does not have enough information.

### After — pure functions

```python
def mean(glucose: pd.Series) -> float:
    """Mean glucose value.

    Args:
        glucose: A series of glucose readings in mg/dL.

    Returns:
        The arithmetic mean of the glucose values.
    """
    return glucose.mean()
    # ✅ mypy: pd.Series has .mean() -> float
```

The pure function pattern is **inherently more type-safe** because:

1. **Explicit parameters** — every input is declared in the function signature.
2. **Specific types** — `pd.Series` tells mypy exactly what methods are available.
3. **Return type** — `-> float` lets mypy check that callers use the result correctly.
4. **No self** — no hidden state, no attributes defined in distant subclasses.

### What mypy verifies after the refactor

```python
from cgmpy.metrics.basic import mean

# ✅ mypy can verify this entire chain
def report(glucose: pd.Series) -> dict[str, float]:
    return {
        "mean": mean(glucose),
        "gmi": gmi(mean_glucose=mean(glucose)),
        "cv": cv(glucose),
    }
```

Before the refactor, mypy would see `self.data["glucose"]` as `Any` and give up. After the refactor, every function has a typed signature and mypy can verify the entire computation graph.

### Running mypy on CGMPy

```bash
# From the project root
mypy cgmpy

# Check a specific module
mypy cgmpy/metrics/basic.py

# Show what errors are suppressed
mypy cgmpy --show-error-codes
```

If you are adding types to a file, run `mypy` on just that file first:

```bash
mypy cgmpy/metrics/variability.py  # Iterate until clean
```

## Summary

| Before (mixin) | After (pure function with types) |
|---|---|
| `def mean(self) -> float` | `def mean(glucose: pd.Series) -> float` |
| `self.data` is `Any` | `glucose` is `pd.Series` |
| mypy cannot check column access | mypy verifies `.mean()` exists |
| Hidden dependencies on `self` | All inputs are explicit |
| Fragile inheritance chains | Standalone, composable, typed |

Type hints turn your function signatures into executable documentation. They cost nothing at runtime and save hours of debugging. In scientific code where correctness matters most, they are not optional — they are essential.
