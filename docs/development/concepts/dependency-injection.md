# Dependency Injection: Give It What It Needs

Here is a short program. What does it do?

```python
def calculate() -> float:
    # 50 lines of math
    return result
```

You have no idea. It could be anything. Now compare:

```python
def calculate(glucose: pd.Series) -> float:
    # 50 lines of math
    return result
```

You still don't know *exactly* what it does, but you know what it needs: a series of glucose values. That is dependency injection in its simplest form — making dependencies explicit in the function signature.

## What is Dependency Injection (DI)?

Dependency Injection means: **give a function what it needs, don't let it grab things itself.**

Instead of reaching outward (reading a global variable, accessing `self.data`, importing a module at the top level), the function receives its dependencies as arguments.

```python
# BAD: implicit dependency
def mean() -> float:
    return SESSION["glucose"].mean()  # needs SESSION to exist somewhere

# GOOD: injected dependency
def mean(glucose: pd.Series) -> float:
    return glucose.mean()             # explicit, clear, testable
```

This sounds trivial — it is just passing arguments, after all. But in practice, codebases accumulate implicit dependencies constantly: mixin classes that assume `self.data` exists, functions that read from module-level globals, methods that fetch from a database connection stored in a config object. DI is the discipline of making those dependencies visible.

## The problem in CGMPy (before refactor)

The old mixin pattern was the main offender:

```python
# BEFORE: grab from self (hidden dependency)
class BasicMetrics:
    def mean(self) -> float:
        return self.data["glucose"].mean()  # WHERE does self.data come from? Magic!
```

`self.data` is an implicit dependency. The method assumes it exists, but nothing in the signature tells you. To test `mean()`, you have to construct a full object that provides `self.data`:

```python
# To test this:
data = GlucoseData("file.csv")  # need to create full object first
# OR mock:
obj.data = pd.DataFrame(...)  # fragile, breaks if attribute name changes
```

Neither approach is ideal. The first requires instantiating a complex class. The second couples your test to the internal attribute name — if someone renames `self.data` to `self._df`, the test silently breaks.

## The solution (after refactor)

```python
# AFTER: dependency is explicit
def mean(glucose: pd.Series) -> float:
    return glucose.mean()

# To test: ONE LINE, no setup needed
assert mean(pd.Series([100, 110, 120])) == 110.0
```

The dependency (`glucose`) is a parameter. Testing does not require a `GlucoseData` object, a mock, or any setup. You call the function with plain data and assert the result.

## Different forms of DI in Python

DI is a spectrum of patterns, not a single technique. Here are the forms you will find in CGMPy.

### Function parameters (CGMPy's approach)

The simplest and most Pythonic form:

```python
def tir(glucose: pd.Series, low: float, high: float) -> float:
    ...

def gmi(glucose: pd.Series) -> float:
    return 12.71 * glucose.mean() + 3.16
```

Every dependency is a function parameter. This is the default for all leaf-level computations in CGMPy.

### Constructor injection

When a class needs a dependency throughout its lifecycle, inject it in `__init__`:

```python
class GlucoseAnalysis:
    def __init__(self, data: GlucoseData) -> None:
        self._data = data  # injected here

    def mean(self) -> float:
        return mean(self._data.get_glucose_values())
```

The caller provides `GlucoseData` at construction time. Every method on `GlucoseAnalysis` uses the same data object. This is useful when the same dependency is used across multiple operations.

### Method injection

When a dependency changes per-call, pass it to the specific method:

```python
class Exporter:
    def to_csv(self, data: GlucoseData, path: str) -> None:
        data.df.to_csv(path)  # data injected per call

    def to_parquet(self, data: GlucoseData, path: str) -> None:
        data.df.to_parquet(path)
```

`data` is not stored on `self` — it is passed fresh each time. This makes `Exporter` stateless and reusable across different data sources.

### Which form to use?

| Situation | DI form | Why |
|---|---|---|
| Leaf-level computation | Function parameter | Simplest, most testable, no class needed |
| Orchestration facade | Constructor injection | Same data across many methods |
| Utility / stateless service | Method injection | No state to manage, call with any data |
| Plugin / config system | Framework (optional) | Dynamic resolution of dependencies |

## Benefits for scientific code

Explicit dependencies are not just an aesthetic preference. They have concrete effects on code quality.

| Aspect | Without DI | With DI |
|---|---|---|
| Testing | Create full object, mock internals | Pass a Series, get result |
| Clarity | Where does `self.data` come from? | All inputs are in the signature |
| Type checking | Needs TYPE_CHECKING hacks | Works out of the box with mypy |
| Parallelism | Shared state issues | No shared state — process safely |
| Reusability | Only works in the class hierarchy | Works anywhere, with any data |

**Testing** is the biggest win. A function with injected dependencies can be tested in isolation with synthetic data. You do not need to load a real CSV, mock a database, or instantiate a complex object.

**Clarity** follows closely. When you read `mean(glucose, timestamps)` you know exactly what the function operates on. When you read `obj.mean()`, you have to look at the class definition, then at the parent class, then at the mixin, to understand what `self` contains.

**Parallelism** matters for CGMPy because some operations (bootstrapping, Monte Carlo simulations) benefit from concurrent execution. Functions that take all their data as parameters can run in threads or processes without locking or copying — there is no shared state to protect.

## Real CGMPy example: before and after

Here is a real transition from the CGMPy codebase. The original was a mixin class with `self.data` as an implicit dependency:

```python
# BEFORE (mixin)
class SDMetrics:
    def sd_within_day(self, min_count_threshold: float = 0.5) -> pd.Series:
        df = self.data[["time", "glucose"]].copy()
        df["date"] = df["time"].dt.date
        # ... 50 lines of logic
        return result
```

To test this, you needed to instantiate a class that mixes in `SDMetrics` and has `self.data` populated. A change to `self.data` (rename, restructure) would silently break this method.

After the refactor, the logic lives in a pure function with explicit parameters:

```python
# AFTER (pure function + delegate)
def sd_within_day(
    glucose: pd.Series,
    timestamps: pd.Series,
    min_count_threshold: float = 0.5,
) -> pd.Series:
    df = pd.DataFrame({"time": timestamps, "glucose": glucose})
    df["date"] = df["time"].dt.date
    # ... same 50 lines of logic
    return result


class SDMetrics:
    def sd_within_day(self, min_count_threshold: float = 0.5) -> pd.Series:
        return sd_within_day(
            self.data["glucose"],
            self.data["time"],
            min_count_threshold,
        )
```

Now the mixin is a thin delegate. The function is testable with two `pd.Series` objects — no class instantiation, no implicit state, no fragile attribute access.

## Common misconceptions

### "DI requires a framework"

No. Frameworks like `dependency_injector` or `inject` exist, but they are for large applications where you need dynamic resolution (e.g. "give me the current database connection"). In CGMPy, DI is just passing arguments. Python makes this trivial.

### "DI makes code slow"

A function call with arguments is the same speed as a function call with no arguments. Reading `self.data["glucose"]` is not faster than reading a `glucose` parameter — if anything, attribute lookup is slightly slower than accessing a local variable. The performance difference is negligible.

### "DI is only for OOP"

The opposite is true. CGMPy's primary DI form is function parameters, which has nothing to do with classes. Constructor and method injection are OOP patterns, but the principle — pass dependencies explicitly — applies to any programming style.

## When DI adds friction

DI is not always the best choice. Here are cases where it adds friction rather than value.

### Too many parameters

If a function takes seven or more parameters, the signature becomes unwieldy:

```python
def compute(glucose, timestamps, low, high, min_count, method, window):
    ...
```

At this point, consider grouping related parameters into a dataclass or a context object:

```python
@dataclass
class AnalysisContext:
    glucose: pd.Series
    timestamps: pd.Series
    low: float = 70
    high: float = 180
    min_count: float = 0.5
    method: str = "linear"
    window: int = 60

def compute(ctx: AnalysisContext) -> float:
    ...
```

This preserves the benefits of explicit dependencies while keeping the signature readable.

### Deep call chains

When you pass the same object through five levels of functions, it suggests an architecture issue:

```python
def plot_all(data):
    for subject in data.subjects:
        plot_subject(subject, data)  # data passed down again

def plot_subject(subject, data):
    for day in subject.days:
        plot_day(day, data.metadata)  # metadata passed down again
```

A context object or a class instance can simplify this. The distinction is that the dependency is still *injected* — it is just injected once, not threaded through every call.

### Matplotlib

Matplotlib is stateful by design. `pyplot` maintains a "current figure" and "current axes" that functions implicitly read. Trying to inject figures and axes into every matplotlib call fights the library. In this case, the stateful API is the convention, and it is fine to follow it:

```python
# This is fine — matplotlib is designed for this
plt.figure()
plt.plot(x, y)
plt.title("Glucose trace")
```

For CGMPy's plotting module, we accept that matplotlib's global state is a dependency, but we isolate it behind thin wrapper functions that are easy to test visually.
