# The Facade Pattern in CGMPy

## What is the Facade Pattern?

A facade is a "simplified interface to a complex subsystem." Think of it like a restaurant menu: you order "steak" (simple), and the kitchen handles fifty steps behind the scenes (source ingredients, prep, grill, plate, sauce). The diner does not need to know about the walk-in fridge, the grill temperatures, or the plating technique.

In software, a facade object presents a single, unified API while the subsystem it delegates to can be as tangled as it likes. The user of the facade never touches the internals.

## The problem in CGMPy (before this refactor)

Before the facade, CGMPy had multiple classes with overlapping capabilities:

- `GlucoseData` -- loads and holds data
- `GlucoseMetrics` -- computes metrics from data
- `GlucosePlot` -- generates plots from data
- `GlucoseAnalysis` -- high-level orchestration

Some had `mean()`, some did not. Some had `plot_agp()`, some did not. A user had to answer the question "which class do I use?" before they could do anything useful.

```python
# BEFORE: user needs to pick the right class
data = GlucoseData("file.csv")          # has data, no metrics
metrics = GlucoseMetrics("file.csv")    # has data + metrics, no plots
plot = GlucosePlot("file.csv")          # has data + basic metrics + plots, no MAGE
```

This creates confusion, scattered documentation, and a poor onboarding experience.

## The solution: a single facade

```python
# AFTER: one class does it all
data = GlucoseData("file.csv")        # just data (simple)
analysis = GlucoseAnalysis(data)      # has everything (the facade)

analysis.mean()        # -> float
analysis.TIR()         # -> float
analysis.MAGE()        # -> float
analysis.plot_agp()    # -> plt.Figure
analysis.get_comprehensive_report()  # -> dict  (everything in one call)
```

The user learns one class. Everything else is an implementation detail.

## How the facade works internally (composition)

The facade uses **composition**, not inheritance. It holds a reference to `GlucoseData` and delegates each method call to a pure function in the appropriate submodule.

```python
class GlucoseAnalysis:
    """One class to rule them all -- the facade."""

    def __init__(self, data: GlucoseData) -> None:
        self._data = data  # composition, not inheritance

    def mean(self) -> float:
        # Delegates to the pure function
        from cgmpy.metrics.basic import mean
        return mean(self._data.glucose)

    def TIR(self) -> float:
        from cgmpy.metrics.time_in_range import tir
        return tir(
            self._data.glucose,
            self._data.targets.tir_low,
            self._data.targets.tir_high,
        )

    def plot_agp(self) -> plt.Figure:
        from cgmpy.plotting.agp import plot_agp
        return plot_agp(self._data)

    def report(self) -> dict:
        """Aggregates all metrics in one call."""
        # Each call goes to a pure function
        # No duplicate computation (cached internally)
        ...
```

Every method is a thin delegate. The real logic stays in pure functions inside `cgmpy/metrics/`, `cgmpy/plotting/`, and so on. This keeps the facade easy to maintain while giving the user a single entry point.

## Why use a facade for scientific code?

| Without Facade | With Facade |
|---|---|
| User must understand module structure | User learns one class |
| Import from five or more modules | Import one class |
| Each class has different methods | All methods in one place |
| Hard to discover API | `dir(analysis)` shows everything |
| Documentation is scattered | One doc page for the facade |

For a library like CGMPy, where the target audience includes clinicians and data scientists (not necessarily software engineers), the facade dramatically lowers the barrier to entry.

## Real-world examples

- **`sklearn.model_selection.train_test_split()`** -- a facade over shuffling, splitting, and indexing logic
- **`pd.read_csv()`** -- a facade over the CSV parser, type inference, column handling, and chunking
- **`plt.plot()`** -- a facade over figure creation, axis management, and rendering pipeline

These are not "design patterns for design's sake." They are everyday tools that thousands of scientists rely on.

## The difference between facade and wrapper

- A **wrapper** (or decorator) adds behaviour around an existing object (e.g. a logging decorator that prints before and after every call).
- A **facade** simplifies access to existing behaviour without adding new semantics.

CGMPy's `GlucoseAnalysis` is a facade, not a wrapper. It does not add cross-cutting concerns; it simply gathers the public API into one place.

## When does a facade become a god object?

Every pattern can be overused. A facade turns into a god object when:

- It grows to 5000+ lines -- split by domain
- It has 50+ unrelated methods -- split into sub-facades
- It starts implementing logic instead of delegating -- push logic back to pure functions

CGMPy's facade is roughly twenty methods and delegates every call to a submodule. It is a comfortable size and stays maintainable because the real work happens elsewhere.
