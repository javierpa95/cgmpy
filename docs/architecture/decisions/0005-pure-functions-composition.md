# ADR 0005 — Pure functions + composition for the public API

* Status: Accepted
* Date: 2026-06-15
* Deciders: @javierpa95

## Context and Problem Statement

[ADR 0003](0003-facade-mixin-pattern.md) introduced a facade + mixin pattern.
In practice the mixin layer (`ModularGlucoseData`, `ModularGlucoseMetrics`, and
the per-family `*Metrics` classes such as `SDMetrics`, `MAGEMetrics`) became a
problem:

1. **Duplicate implementations.** Each metric existed twice — once as a pure
   function and once as a thin mixin method that just delegated to it. The
   `GlucoseAnalysis` facade used the pure functions, so the mixins were dead
   code that nothing in the main flow exercised (not even their own tests,
   which already went through the facade).
2. **Cannot be used standalone.** A mixin method needs `self.data`,
   `self.typical_interval`, etc. injected via multiple inheritance with
   `GlucoseData`; you cannot call it on a bare `pandas.Series`.
3. **Hard to understand.** The diamond mixin hierarchy was the most-cited
   source of confusion for contributors (see ADR 0003 consequences).

We want the discoverability and state management of a facade **and** primitives
that are reusable from outside the library.

## Considered Options

* **Pure functions + composition** — metrics are standalone pure functions over
  a `pandas.Series`; `GlucoseAnalysis` is a thin facade that composes them.
* **Keep the mixins** — maintain both the functions and the mixin classes.
* **Facade only** — hide all functions behind the facade, no public primitives.

## Decision Outcome

Chosen option: **Pure functions + composition** — the two-layer model used by
`scipy.stats` / `sklearn.metrics`:

1. **Low-level layer:** every metric is a pure function exported from
   `cgmpy.metrics` (e.g. `tir`, `gmi`, `mage_baghurst`, `lbgi`). It takes a
   glucose `Series` (and timestamps where relevant) and returns a value/dict,
   so it works on any array without the library's data objects.
2. **High-level layer:** `GlucoseAnalysis` wraps a `GlucoseData` instance and
   composes those same functions, adding caching and convenience.

The mixin classes and `cgmpy.metrics.variability._base` were removed in v0.6.0.

### Consequences

* Good, because there is a single source of truth per metric.
* Good, because the low-level functions are reusable and easy to test directly.
* Good, because the facade stays simple (delegation, no inheritance graph).
* Neutral: no backward-compatibility shim is provided — the library is still
  pre-1.0 and the mixins had no documented external users.

## Pros and Cons of the Options

### Pure functions + composition

* Good, discoverable primitives (`from cgmpy.metrics import tir`).
* Good, no duplicate implementations, no mixin diamond.
* Bad, the facade and functions must be kept in sync by hand (mitigated by tests).

### Keep the mixins

* Bad, two parallel APIs to maintain and document.
* Bad, the confusing inheritance hierarchy remains.

### Facade only

* Good, smallest public surface.
* Bad, loses the standalone `scipy.stats`-style usage we explicitly want.
