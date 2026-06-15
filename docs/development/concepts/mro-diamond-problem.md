# MRO and the Diamond Problem

> **Note.** The `ModularGlucoseData` / `*Metrics` mixin hierarchy used as the
> example below is the **pre-v0.6.0** design and has since been removed in
> favour of pure functions + composition
> ([ADR 0005](../../architecture/decisions/0005-pure-functions-composition.md)).
> It is retained here purely to illustrate the diamond problem.

## What is MRO (Method Resolution Order)?

When you call `obj.method()`, Python needs to decide **which** `method` to call. In single inheritance this is simple: walk up the chain until you find it. In multiple inheritance, the class hierarchy becomes a directed graph, and Python needs a consistent algorithm to linearise it.

**MRO** = Method Resolution Order. The sequence of classes Python searches when you call a method. The first class in the MRO that defines the method wins.

## The Diamond Problem

The canonical diamond:

```
    A
   / \
  B   C
   \ /
    D
```

If `A` defines `method()`, `B` overrides it, and `C` overrides it differently, which version does `D` inherit? Before Python 2.3, the answer was "it depends on the order you list the parents" and the algorithm was a flawed depth-first search. Since Python 2.3, the answer is **C3 Linearisation**.

## C3 Linearisation: How Python Resolves It

The rule is simple: **children before parents, and respect the order of inheritance.** Given `class D(B, C):`, Python computes:

```
D -> B -> C -> A -> object
```

The first match wins. So if `B` defines `method()`, `D` uses `B`'s version regardless of what `C` does.

```python
class A:
    def method(self) -> str:
        return "A"

class B(A):
    def method(self) -> str:
        return "B"

class C(A):
    def method(self) -> str:
        return "C"

class D(B, C):
    pass

print(D.mro())
# [<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>]

print(D().method())  # "B"
```

Swap the order to `class D(C, B):` and the output flips to `"C"`. That is the danger --- **reordering parents silently changes behaviour**.

## How This Affects CGMPy

The actual MRO of `GlucosePlot`:

```python
from cgmpy import GlucosePlot
print([c.__name__ for c in GlucosePlot.__mro__])
# ['GlucosePlot', 'ModularGlucoseData', 'BasicMetrics',
#  'TimeInRangeMetrics', 'AGPPlotter', 'DailyPlotter',
#  'StatisticalPlotter', 'object']
```

Eight classes, no diamond --- but note the pattern. Each mixin (BasicMetrics, TimeInRangeMetrics, AGPPlotter, etc.) expects `self.data` to exist and `self.targets` to be available. There is no formal contract; it works by convention. If two mixins defined the same method with different behaviour, whichever appears first in the MRO would win silently.

```python
# Hypothetical collision
class PlotterMixin:
    def summary(self) -> str:
        return "plotter summary"

class StatsMixin:
    def summary(self) -> str:
        return "stats summary"

# Which summary does GlucosePlot use?
# Depends entirely on the order in the class definition.
```

## Why This Is Dangerous for Scientific Code

1. **Silent changes.** Reordering base classes changes behaviour without warning.
2. **Hard to debug.** `obj.method()` --- which class provides it? You need to examine the entire MRO.
3. **Unpredictable `super()` chains.** If every mixin uses `super().__init__()`, the order of initialisation depends on the MRO, not on explicit logic.
4. **Type checkers can't handle it.** Mypy cannot statically resolve which method wins in a diamond. This is why CGMPy uses `# type: ignore[misc]` on the `GlucosePlot` class definition and has relaxed mypy settings for metrics modules.

```
# type: ignore[misc]  <-- tells mypy: "we know this is fragile"
class GlucosePlot(
    ModularGlucoseData,
    BasicMetrics,
    TimeInRangeMetrics,
    AGPPlotter,
    DailyPlotter,
    StatisticalPlotter,
):
```

## The Fix: Composition Over Inheritance

Instead of relying on MRO magic:

```python
# Fragile: diamond inheritance
class D(B, C):
    pass
```

Use explicit delegation:

```python
# Robust: composition
class D:
    def __init__(self) -> None:
        self.b = B()
        self.c = C()

    def method(self) -> str:
        return self.b.method()  # explicit choice
```

Composition has no diamond, no MRO surprises, and mypy can check every method call. It is more typing but vastly more maintainable --- especially in a scientific library where a silent logic error means wrong clinical numbers.

## Real CGMPy MRO

Run this to see the actual MRO of the facade class:

```python
from cgmpy import GlucosePlot
for i, cls in enumerate(GlucosePlot.__mro__):
    indent = "  " * i
    print(f"{indent}{cls.__name__}")
```

Output:

```
GlucosePlot
  ModularGlucoseData
    BasicMetrics
      TimeInRangeMetrics
        AGPPlotter
          DailyPlotter
            StatisticalPlotter
              object
```

No diamond exists in the current hierarchy, but the pattern is still fragile by convention. Every mixin assumes `self.data` is a DataFrame and `self.targets` is a `GlucoseTargets` --- there is no abstract base class or protocol enforcing it. If you ever find yourself adding a new mixin to `GlucosePlot`, think twice: can you use composition instead?
