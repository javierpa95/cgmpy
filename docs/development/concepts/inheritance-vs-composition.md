# Inheritance vs Composition: Why CGMPy Switched from Mixins to Composition

> **Note.** The mixin classes shown below (`ModularGlucoseData`, `SDMetrics`,
> `VariabilityBase`, ...) are the **pre-v0.6.0** design and no longer exist in
> the codebase — they are kept here as the cautionary "before" example. The
> switch to pure functions + composition is complete; see
> [ADR 0005](../../architecture/decisions/0005-pure-functions-composition.md).

Imagine you are building a Swiss Army knife. You could forge it as one solid piece of metal — a single, monolithic class that does everything. Or you could design it as a handle that accepts attachable tools. The second approach is composition: the knife *has* tools rather than *being* a single tool.

CGMPy started with the first approach — multiple inheritance with mixins. It worked, but it was brittle. This article explains why we moved to composition, what the diamond problem looks like in practice, and when inheritance is still the right choice.

## The "is-a" vs "has-a" distinction

The core difference between inheritance and composition comes down to two relationships:

**Inheritance: "is-a"**

```python
class Animal:
    def breathe(self) -> None:
        ...

class Dog(Animal):
    """A Dog IS-A Animal."""
    def bark(self) -> None:
        ...
```

`Dog` inherits `breathe()` because every dog *is* an animal. This makes intuitive sense for taxonomies, but it creates a rigid hierarchy.

**Composition: "has-a"**

```python
class Engine:
    def start(self) -> None:
        ...

class Wheel:
    def rotate(self) -> None:
        ...

class Car:
    """A Car HAS-An Engine and HAS-A Wheels."""

    def __init__(self) -> None:
        self.engine = Engine()
        self.wheels = [Wheel() for _ in range(4)]
```

`Car` does not inherit from `Engine` or `Wheel`. It *contains* them. This is looser, more flexible, and easier to change.

**The common confusion with multiple inheritance:**

```python
# This is NOT composition — it is multiple inheritance
class Car(Engine, Wheel):  # Car IS-A Engine AND IS-A Wheel? That makes no sense
    pass
```

Multiple inheritance masquerades as composition ("I'll just inherit everything I need"), but it creates an "is-a" relationship for every parent. A car is not an engine, and `GlucoseAnalysis` is not a `BasicMetrics` — it *uses* those metrics.

## Why CGMPy started with mixins (multiple inheritance)

CGMPy's original design was elegant in its simplicity:

```python
class GlucoseAnalysis(
    ModularGlucoseData,
    BasicMetrics,
    TimeInRangeMetrics,
    VariabilityMetrics,
    AGPPlotter,
    DailyPlotter,
    StatisticalPlotter,
):
    """One class to rule them all."""
    pass
```

One class inheriting from seven mixins, each providing a set of methods. Users could write:

```python
analysis = GlucoseAnalysis("data.csv")
analysis.mean()     # from BasicMetrics
analysis.TIR()      # from TimeInRangeMetrics
analysis.MAGE()     # from VariabilityMetrics
analysis.plot_agp() # from AGPPlotter
```

It worked. The API was flat and convenient. But the convenience came at a cost.

## The diamond problem (MRO)

Take a look inside `VariabilityMetrics`:

```python
class VariabilityMetrics(
    SDMetrics,
    MAGEMetrics,
    MODDMetrics,
    CONGAMetrics,
    LabilityMetrics,
    RiskMetrics,
):
    """Composite mixin combining all variability metric families."""
    pass
```

Each of those six classes inherits from `VariabilityBase`. So the actual inheritance graph looks like this:

```
GlucoseAnalysis
 ├── ModularGlucoseData
 ├── BasicMetrics
 ├── TimeInRangeMetrics
 ├── VariabilityMetrics
 │    ├── SDMetrics ──┐
 │    ├── MAGEMetrics ─┤
 │    ├── MODDMetrics ─┤
 │    ├── CONGAMetrics ─┤── VariabilityBase
 │    ├── LabilityMetrics ─┤
 │    └── RiskMetrics ────┘
 ├── AGPPlotter
 ├── DailyPlotter
 ├── StatisticalPlotter
 └── object
```

Python's C3 linearisation resolves the MRO (Method Resolution Order) to 16 entries:

```
0: GlucoseAnalysis
1: ModularGlucoseData
2: BasicMetrics
3: TimeInRangeMetrics
4: VariabilityMetrics
5: SDMetrics
6: MAGEMetrics
7: MODDMetrics
8: CONGAMetrics
9: LabilityMetrics
10: RiskMetrics
11: VariabilityBase
12: AGPPlotter
13: DailyPlotter
14: StatisticalPlotter
15: object
```

`VariabilityBase` appears exactly once thanks to C3, but the ordering is fragile. Consider what happens when you call `analysis.mean()`:

- Python walks the MRO from index 0 upward.
- `BasicMetrics` (index 2) defines `mean()`, so that is the one you get.
- But what if someone later adds a `PlottingBase` mixin before `BasicMetrics` in the inheritance list? The resolution silently changes.

This is the practical diamond problem: **not that it crashes, but that the order of base classes silently dictates behaviour**. Adding a new mixin in the "wrong" position can shadow methods from another mixin without any warning.

```python
# Seemingly harmless reorder...
class GlucoseAnalysis(
    ModularGlucoseData,
    BasicMetrics,
    TimeInRangeMetrics,
    VariabilityMetrics,
    StatisticalPlotter,  # moved above AGPPlotter
    AGPPlotter,
    DailyPlotter,
):
    pass

# Does analysis.plot_agp() still call AGPPlotter's version?
# It depends on whether StatisticalPlotter also defines plot_agp.
# You have to check. Every. Time.
```

## Why composition won

The composition approach flips the relationship:

- Inheritance: analysis **is a** data loader, **is a** metrics calculator, **is a** plotter
- Composition: analysis **has a** data object, **uses** metric functions, **calls** plotters

Here is the current CGMPy approach:

```python
data = GlucoseData("data.csv")

# Pure functions that take data as an argument
from cgmpy.metrics.basic import mean, gmi
from cgmpy.metrics.variability.mage import mage_baghurst

result = mean(data.get_glucose_values())           # call the function directly
gmi_value = gmi(data.get_glucose_values())
mage_result = mage_baghurst(data.get_glucose_values())
```

And the facade class that wraps it all:

```python
class GlucoseAnalysis:
    """Analysis facade — HAS-A GlucoseData, not IS-A."""

    def __init__(self, data_source: str | pd.DataFrame):
        self.data = GlucoseData(data_source)

    def mean(self) -> float:
        return mean(self.data.get_glucose_values())

    def MAGE(self) -> float:
        return mage_baghurst(self.data.get_glucose_values())
```

No MRO. No diamond. Each function is independently callable, testable, and importable.

## Side-by-side code comparison

```python
# BEFORE — multiple inheritance (mixin pattern)
class GlucoseAnalysis(
    ModularGlucoseData,
    BasicMetrics,
    TimeInRangeMetrics,
    VariabilityMetrics,
    AGPPlotter,
    DailyPlotter,
    StatisticalPlotter,
):
    pass

analysis = GlucoseAnalysis("data.csv")
analysis.mean()   # Where does this come from? BasicMetrics?
                  # Or does VariabilityMetrics override it?
                  # Check the MRO...
analysis.MAGE()   # Does this exist? Depends on VariabilityMetrics being wired.
```

```python
# AFTER — composition
from cgmpy import GlucoseData
from cgmpy.metrics.basic import mean
from cgmpy.metrics.variability.mage import mage_baghurst
from cgmpy.plotting.agp import plot_agp

data = GlucoseData("data.csv")
analysis = GlucoseAnalysis(data)   # HAS-A GlucoseData, not IS-A

analysis.mean()      # delegated to basic.mean(data.glucose)
analysis.MAGE()      # delegated to mage_baghurst(data.glucose)
analysis.plot_agp()  # delegated to a plotting function
```

Each delegation is explicit. You can see exactly where each method comes from. If `mean()` breaks, you look in `cgmpy/metrics/basic.py` — not in the MRO.

## The "fragile base class" problem

When you inherit from a class, that class's internal changes can silently break your code. This is the fragile base class problem.

```python
# Mixin version: a change to BasicMetrics affects everyone
class BasicMetrics:
    def mean(self) -> float:
        # Version 1
        return self.data["glucose"].mean()

    # Version 2 — renamed internal method
    def _calculate_mean(self) -> float:
        return self.data["glucose"].mean()

    def mean(self) -> float:
        return self._calculate_mean()

# GlucoseAnalysis.mean() still works — for now.
# But what if _calculate_mean has a subtle bug?
```

With composition, the data object is a separate concern:

```python
# Composition version: GlucoseData is just a container
class GlucoseData:
    def __init__(self, data_source: str | pd.DataFrame):
        self._load(data_source)

    def get_glucose_values(self) -> pd.Series:
        return self.data["glucose"].copy()

# The metric functions are independent
def mean(glucose: pd.Series) -> float:
    return glucose.mean()

# You can refactor GlucoseData internally without touching mean()
```

The data class can change its internal representation (from CSV to Parquet, from Pandas to Polars, adding caching) without affecting any of the metric functions. The interface — `get_glucose_values()` — is the contract.

This is the fundamental shift: **inheritance binds consumers to implementation, composition binds them to interfaces**.

## When inheritance IS the right choice

Composition is not a universal replacement for inheritance. There are cases where inheritance is the correct tool:

### True specialisation

```python
class DexcomLoader(GlucoseData):
    """Dexcom IS-A specific kind of glucose data source.

    It specialises GlucoseData with Dexcom-specific column
    mapping and metadata handling, but remains a full glucose data
    object at heart.
    """

    def __init__(self, file_path: str) -> None:
        super().__init__(file_path, date_col="timestamp", glucose_col="mg/dL")
        self._parse_dexcom_metadata()
```

This is a genuine "is-a" relationship. A `DexcomLoader` *is* a `GlucoseData` with specialised loading logic. Consumers can use it wherever they use `GlucoseData`.

### Protocol / ABC definitions

```python
from typing import Protocol

class GlucoseDataSource(Protocol):
    """Defines the interface for any glucose data source."""

    def get_glucose_values(self) -> pd.Series: ...
    def get_timestamps(self) -> pd.Series: ...

class DexcomLoader:
    """Implements GlucoseDataSource — an IS-A relationship by contract."""
    ...
```

Inheritance shines when defining a contract (a Protocol or ABC) that multiple implementations satisfy. This is especially useful for dependency injection and testing.

### The rule of thumb

| Situation | Prefer | Reason |
|---|---|---|
| "X is a specialised version of Y" | Inheritance | `DexcomLoader IS-A GlucoseData` |
| "X defines a contract that Y must fulfil" | Inheritance (Protocol/ABC) | `GlucoseDataSource` as an interface |
| "X needs the behaviour of Y, Z, and W" | Composition | `GlucoseAnalysis HAS-A data, HAS-A metric functions` |
| "X needs to combine multiple independent behaviours" | Composition | Avoids the diamond problem and MRO fragility |

## Key insight

The Python community has gone through a shift. In the 2000s, frameworks like Django and Zope embraced mixins heavily. By the 2010s, the consensus had moved toward "composition over inheritance" — a principle popularised by the Gang of Four's *Design Patterns* book in the 1990s, but which took time to penetrate the Python ecosystem.

Even the Python standard library reflects this shift:

- `collections.abc` provides Protocols (contracts), not mixins
- `@dataclass` provides data containers without behavioural inheritance
- `functools.lru_cache` is a decorator (compositional), not a base class
- `pathlib.Path` uses composition internally (a path *has* a filesystem, it *is* not a filesystem)

CGMPy is following this industry best practice. The mixin classes still exist for backward compatibility, but new code should:

1. Write pure functions that take `pd.Series` or `pd.DataFrame` as arguments.
2. Use classes only for orchestration and stateful resources.
3. Prefer `has-a` over `is-a` when combining behaviours.

The result is code that is easier to test, easier to reason about, and harder to break by accident.
