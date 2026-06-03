# What is Mypy? (Type Hints)

## What is it?

Python doesn't care if you pass a patient name where a glucose value goes. It will crash at runtime. Mypy checks this BEFORE the code runs, while you're still writing it.

Type hints are optional annotations in Python 3.5+ that tell you (and mypy) what type a variable, parameter, or return value should be. Mypy is a static type checker that reads these annotations and finds bugs before you run a single line of code.

## Why we use it

In a CGM library, passing `"patient_123"` to `mean()` instead of a Series of numbers should be impossible. Mypy makes it impossible.

Consider this scenario: you're writing a report and accidentally pass a string where a number is expected. Without mypy, you find out when the script crashes at 3am during a batch job. With mypy, you get a red squiggly line in your editor the moment you type it.

## Type hints 101

```python
# Without type hints (mypy can't help)
def gmi(m):
    return round(3.31 + 0.02392 * m, 2)

# With type hints (mypy checks this)
def gmi(mean_glucose: float) -> float:
    return round(3.31 + 0.02392 * mean_glucose, 2)
```

The first version is a mystery. What is `m`? A number? A string? A Series? You have to read the function body or docstring to guess. The second version tells you everything: it takes a `float` (mean glucose in mg/dL) and returns a `float` (GMI percentage).

## Real CGMPy example

```python
from cgmpy.metrics.basic import mean

mean("100, 110, 120")  # mypy: Argument 1 has incompatible type "str"
mean(pd.Series([100, 110, 120]))  # mypy: OK
```

Mypy catches this at type-check time, not at runtime. You fix it before the code ever runs.

It also checks return values:

```python
def gmi(mean_glucose: float) -> float:
    return round(3.31 + 0.02392 * mean_glucose, 2)

result: str = gmi(120.0)  # mypy: Incompatible types in assignment (expected "str", got "float")
```

Mypy checks both callers and implementations. If you declare a return type of `float` but return a `str`, mypy tells you.

## Gradual typing

CGMPy uses gradual typing. Not everything is typed yet. We started with the most important functions first.

```python
# Partially typed — mypy checks inner(), but not outer()
def outer(x):
    return inner(x)

def inner(x: float) -> float:
    return x * 2
```

This approach lets us add types incrementally without blocking development. A partially typed codebase is still better than an untyped one — mypy checks everything it can see.

## How to use it

```bash
mypy cgmpy/           # check the library
mypy cgmpy/ --strict  # even stricter (future goal)
mypy cgmpy/metrics/variability.py  # check a single file
```

If you're adding types to a file, run mypy on just that file first to iterate quickly.

## Common errors

| Error | Meaning |
|-------|---------|
| `Incompatible return type` | Function returns something different from declared type |
| `Missing type parameters` | Need to specify types in generic (e.g. `dict[str, float]` not just `dict`) |
| `Argument 1 has incompatible type "str"; expected "float"` | Passed a string where a number is expected |
| `Cannot access attribute "glucose" for "Any"` | Variable is untyped so mypy can't verify attribute access |

In scientific computing, `Missing type parameters` is the most common one you'll hit. Instead of writing `dict` as the return type, write `dict[str, float]` to tell mypy what keys and values look like.

## Why it's good for learners

Type hints are like labelled axes on a clinical chart. They tell you exactly what goes where. When you read a function signature:

```python
def tir(glucose: pd.Series, low: float = 70, high: float = 180) -> float: ...
```

You know immediately: glucose readings go in, a percentage comes out. The `low` and `high` thresholds are numbers with sensible defaults. You don't need to read the docstring to understand the contract.

This is especially valuable in a library like CGMPy where getting the types wrong means wrong clinical results. Type hints are your first line of defence against silly mistakes.
