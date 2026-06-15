# Caching Strategies in CGMPy

## The problem: redundant computation in scientific code

Scientific libraries compute the same thing repeatedly, often without realising it. Consider a typical analysis session:

```python
from cgmpy import GlucoseAnalysis

analysis = GlucoseAnalysis("patient_data.csv")

# User asks for individual metrics
analysis.mean()     # computes mean
analysis.sd()      # computes sd
analysis.TIR()     # computes TIR
analysis.cv()      # computes cv
analysis.MAGE()    # computes MAGE
analysis.get_comprehensive_report()  # computes everything again
```

The problem is hidden in the call graph. Each method delegates to a pure function that recomputes from scratch:

```python
# cgmpy/metrics/basic.py

def mean(glucose: pd.Series) -> float:
    return glucose.mean()

def sd(glucose: pd.Series) -> float:
    return glucose.std()

def cv(glucose: pd.Series) -> float:
    # Calls mean() and sd() internally!
    m = mean(glucose)
    if m == 0:
        return 0.0
    return (sd(glucose) / m) * 100
```

When you call `cv()`, it calls `mean()` and `sd()` — which is fine in isolation. But when `MAGE()` also calls `sd()`, and `report()` calls everything, the same function explodes into dozens of identical computations.

Here is the call tree for a single call to `report()`:

```
report()
├── mean()          # 1st computation
├── sd()            # 1st computation
├── TIR()
├── cv()
│   ├── mean()      # 2nd computation (already computed!)
│   └── sd()        # 2nd computation (already computed!)
├── MAGE()
│   ├── sd()        # 3rd computation (already computed!)
│   └── ...
├── distribution_analysis()
│   ├── mean()      # 3rd computation
│   ├── sd()        # 4th computation
│   ├── cv()
│   │   ├── mean()  # 4th computation
│   │   └── sd()    # 5th computation
│   └── ...
└── ...
```

`mean()` is called 4+ times. `sd()` is called 5+ times. On a dataset with 1 million rows, each call to `glucose.mean()` or `glucose.std()` takes several milliseconds. Those milliseconds add up to seconds of wasted CPU, and the user waits for no reason.

The root cause is that the pure functions — `mean()`, `sd()`, `cv()` — are stateless. They have no memory of previous results. Every call starts from zero.

## Simple caching: the `_cache` dict pattern

The fix is to introduce a cache inside the analysis object. The first time a metric is requested, compute it and store the result. Every subsequent call returns the stored value.

```python
class GlucoseAnalysis:
    """A facade over the metrics submodules, with built-in caching."""

    def __init__(self, data: GlucoseData) -> None:
        self._data = data
        self._cache: dict[str, float] = {}

    def _get_mean(self) -> float:
        if "_mean" not in self._cache:
            self._cache["_mean"] = mean(self._data.glucose)
        return self._cache["_mean"]

    def mean(self) -> float:
        return self._get_mean()

    def _get_sd(self) -> float:
        if "_sd" not in self._cache:
            self._cache["_sd"] = sd(self._data.glucose)
        return self._cache["_sd"]

    def sd(self) -> float:
        return self._get_sd()

    def cv(self) -> float:
        # No longer calls mean() and sd() directly.
        # Instead, it uses cached values — O(1) after first computation.
        m = self._get_mean()
        s = self._get_sd()
        return 0.0 if m == 0 else (s / m) * 100
```

The benefit is immediate. The first call to `mean()` is slow; the 2nd through Nth calls are a dict lookup (nanoseconds). For a full report, `mean()` goes from 4+ calls to 1 call. That is a 4x speedup for `mean()` alone. When you also cache `sd()`, `percentile()`, and `TIR()`, the total improvement is closer to 20x for the metrics that were being recomputed most heavily.

## Invalidating the cache

A cache is only safe while the underlying data does not change. If the user calls `filter_time_range()` or `resample()`, the cached metrics become stale — they were computed on the old data.

CGMPy avoids this problem by making data **immutable**:

```python
class GlucoseData:
    def filter_time_range(self, start: datetime, end: datetime) -> GlucoseData:
        # Returns a *new* GlucoseData, does not modify self
        subset = self._data[(self._data["time"] >= start) & (self._data["time"] <= end)]
        return GlucoseData(subset)  # new instance = new data
```

Because the old object is not modified, its cache remains internally consistent (still valid for the original data). The new object has an empty cache, so metrics are recomputed on the filtered data automatically:

```python
analysis = GlucoseAnalysis(data)         # full data, cache = {}
analysis.mean()                          # computes, caches

filtered = analysis.filter_time_range(   # returns new GlucoseAnalysis
    datetime(2024, 1, 1),
    datetime(2024, 1, 7),
)
filtered.mean()                          # computes on new data, cache = {}
```

**Key insight**: by making data immutable (filter returns a new instance), cache invalidation is automatic. The programmer never needs to write `self._cache.clear()`.

The same principle applies to `resample()`, `drop_outliers()`, and any other transformation:

```python
class GlucoseAnalysis:
    def filter_time_range(self, start, end):
        new_data = self._data.filter_time_range(start, end)
        return GlucoseAnalysis(new_data)  # new analysis = fresh cache

    def resample(self, freq: str):
        new_data = self._data.resample(freq)
        return GlucoseAnalysis(new_data)  # new analysis = fresh cache
```

## `functools.cache` and `@lru_cache`

Pure functions that accept hashable arguments can use Python's built-in `functools.cache` decorator:

```python
from functools import cache

# Pure functions are naturally cacheable!
@cache
def mean(glucose_tuple: tuple[float, ...]) -> float:
    """Mean glucose value, cached for identical inputs."""
    return sum(glucose_tuple) / len(glucose_tuple)
```

This works because `tuple[float, ...]` is hashable. The second time you call `mean(t)`, Python short-circuits and returns the stored result — no sum, no division, no overhead.

However, `pandas.Series` is not hashable, so you cannot use `@cache` on CGMPy's pure functions directly:

```python
@cache
def mean(glucose: pd.Series) -> float:
    # TypeError: unhashable type: 'Series'
    return glucose.mean()
```

To use `@cache` with pandas data, convert to a hashable type:

```python
@cache
def mean(glucose_bytes: bytes) -> float:
    """Mean glucose value from a serialised array.

    Call with: mean(series.to_numpy().tobytes())
    """
    import numpy as np
    arr = np.frombuffer(glucose_bytes)
    return arr.mean()
```

But this is fragile: `tobytes()` is sensitive to dtype and endianness, and the memory overhead of storing the serialised array in the cache is high.

**Verdict for CGMPy**: `@cache` is useful for leaf-level functions that accept simple hashable types (ints, strings, tuples of floats), but it is not a replacement for the `_cache` dict pattern on the facade. The `_cache` dict is simpler, faster, and more explicit about invalidation semantics.

### Pros and cons of `@cache`

| Pros | Cons |
|------|------|
| Automatic — no boilerplate | Only works with hashable arguments |
| No manual `_cache` dict | Memory grows unbounded |
| Thread-safe (CPython) | No built-in invalidation (use `@lru_cache(maxsize=N)`) |

## The `@property` + cache pattern

An alternative to writing `_get_mean()` / `_get_sd()` helper methods is to use `@property` with lazy caching:

```python
class GlucoseAnalysis:
    def __init__(self, data: GlucoseData) -> None:
        self._data = data
        self._cache: dict[str, float] = {}

    @property
    def _mean(self) -> float:
        if "__mean" not in self._cache:
            self._cache["__mean"] = mean(self._data.glucose)
        return self._cache["__mean"]

    @property
    def _sd(self) -> float:
        if "__sd" not in self._cache:
            self._cache["__sd"] = sd(self._data.glucose)
        return self._cache["__sd"]

    @property
    def _cv(self) -> float:
        if "__cv" not in self._cache:
            m = self._mean  # property calls — still cached
            s = self._sd
            self._cache["__cv"] = 0.0 if m == 0 else (s / m) * 100
        return self._cache["__cv"]

    def mean(self) -> float:
        return self._mean

    def sd(self) -> float:
        return self._sd

    def cv(self) -> float:
        return self._cv

    def report(self) -> dict:
        return {
            "mean": self.mean(),
            "sd": self.sd(),
            "cv": self.cv(),
            "gmi": gmi(self.mean()),
        }
```

The `@property` + cache pattern is ergonomic: it reads like an attribute access (`self._mean`) but still triggers lazy computation on first access. The convention of prefixing internal cached properties with `_` keeps them private.

Trade-off against the `_get_mean()` method pattern:

| Pattern | When to use |
|---------|-------------|
| `_get_mean()` method | When computation has side effects or needs arguments |
| `@property` + cache | When the cached value is an attribute-like quantity with no arguments |

In CGMPy, both are valid. The mixin classes currently use methods (`self.mean()`, `self.sd()`), so the `_get_mean()` pattern aligns better with the existing style. The `@property` approach is cleaner for new code that does not need to match legacy mixin interfaces.

## Batch computation: compute all at once

Caching avoids redundant computation, but the first call to each metric still touches the data once. If the user calls `mean()`, then `sd()`, then `cv()`, the data is traversed three times (once per metric). For a 1M-row dataset, each traversal costs time.

Batch computation solves this by computing everything in a single pass and pre-warming the cache:

```python
class GlucoseAnalysis:
    def report(self) -> dict:
        """Compute ALL metrics in one pass, caching everything.

        The cache is pre-warmed so that individual metric calls
        (mean(), sd(), cv(), etc.) return instantly without
        recomputing anything.
        """
        glucose = self._data.glucose

        # Core statistics — computed once
        m = mean(glucose)
        s = sd(glucose)
        n = len(glucose)

        # Pre-warm the cache
        self._cache["_mean"] = m
        self._cache["_sd"] = s
        self._cache["_n"] = n

        # Time in range
        tir_val = tir(glucose, self._data.targets.tir_low, self._data.targets.tir_high)
        self._cache["_tir"] = tir_val

        # Variability
        mage_val = mage(glucose)
        self._cache["_mage"] = mage_val

        # Build the report using cached values
        return {
            "n": n,
            "mean": m,
            "sd": s,
            "cv": 0.0 if m == 0 else (s / m) * 100,
            "gmi": gmi(m),
            "tir": tir_val,
            "mage": mage_val,
        }
```

After calling `report()`, every individual metric is a dict lookup. The cost of the full report is roughly the same as computing the most expensive single metric — everything else is free.

## CGMPy's caching design (the plan)

The caching architecture follows the *functional core, imperative shell* pattern described in the pure functions concept article:

```
GlucoseAnalysis                      ← the facade (imperative shell)
├── _cache: dict[str, Any]           ← manual dict cache
├── mean() → _get("_mean")           ← lazy, cached
├── sd() → _get("_sd")              ← lazy, cached
├── cv() → uses _mean + _sd         ← reuses cache
├── tir() → _get("_tir")            ← lazy, cached
├── mage() → _get("_mage")          ← lazy, cached
├── report() → computes batch       ← pre-warms cache
└── filter_time_range() → new GlucoseAnalysis → new cache (automatic invalidation)

cgmpy/metrics/basic.py              ← pure functions (functional core)
├── mean(glucose: pd.Series) -> float    ← stateless, always recomputes
├── sd(glucose: pd.Series) -> float      ← stateless, always recomputes
└── cv(glucose: pd.Series) -> float      ← stateless, always recomputes
```

The pure functions in `cgmpy/metrics/` remain stateless. They are simple, testable, and deterministic. Caching is added only at the facade level, where the `GlucoseAnalysis` object orchestrates the computation. This keeps the pure functions clean while allowing the user to benefit from caching transparently.

### Cache key naming convention

Cache keys follow a consistent convention:

| Key pattern | Example | Computed value |
|-------------|---------|----------------|
| `_mean` | `"_mean"` | Mean glucose |
| `_sd` | `"_sd"` | Standard deviation |
| `_tir` | `"_tir"` | Time in range percentage |
| `_mage` | `"_mage"` | MAGE value |
| `_percentile_95` | `"_percentile_95"` | 95th percentile |

For parameterised metrics like `percentile(p)`, the cache key includes the parameter:

```python
def percentile(self, p: float) -> float:
    key = f"_percentile_{p}"
    if key not in self._cache:
        self._cache[key] = percentile(self._data.glucose, p)
    return self._cache[key]
```

## When NOT to cache

Caching is not free. Every cached value consumes memory, and every cache lookup has a (tiny) cost. These are the situations where caching is the wrong choice:

| Situation | Reason | Example in CGMPy |
|-----------|--------|------------------|
| **One-shot computations** | The value is requested exactly once. Caching adds overhead with no benefit. | `get_data_quality_metrics()` called once during `report()` |
| **Very cheap computations** | The computation is faster than a dict lookup (microseconds). | `n = len(glucose)` |
| **Memory-constrained environments** | Cache stores every result; on embedded systems or large batch jobs this can cause OOM. | Running 10,000 analyses in a loop without clearing the cache |
| **Data changes frequently** | Cache invalidation overhead outweighs the benefit. | A real-time streaming pipeline where data arrives every second |
| **Side-effectful computations** | Caching assumes deterministic results from the same inputs. | `print()` calls, database writes, logging |
| **Large return values** | Caching a large DataFrame or figure consumes memory for the entire session. | `plot_agp()` returns a matplotlib Figure object |

The golden rule: **cache only when the same computation would be repeated on the same data**. If in doubt, measure. A simple timing test with `time.perf_counter()` will tell you whether the cache is helping or hurting:

```python
import time

# Without cache
start = time.perf_counter()
for _ in range(100):
    cv(data.glucose)
t_no_cache = time.perf_counter() - start

# With cache
analysis = GlucoseAnalysis(data)
start = time.perf_counter()
for _ in range(100):
    analysis.cv()
t_with_cache = time.perf_counter() - start

print(f"Without cache: {t_no_cache:.3f}s")   # 2.340s
print(f"With cache:    {t_with_cache:.3f}s")  # 0.001s
```

A 2000x speedup on repeated calls justifies the cache. If the speedup is less than 2x, the cache may not be worth the complexity.
