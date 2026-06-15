# ADR 0003 — Facade + mixin pattern for the public API

* Status: Superseded by [ADR 0005](0005-pure-functions-composition.md)
* Date: 2026-06-02
* Deciders: @javierpa95

> **Superseded (v0.6.0).** The mixin layer described here
> (`ModularGlucoseData`, `ModularGlucoseMetrics`, the `*Metrics` mixin
> classes) has been removed. The public API is now **pure functions +
> composition**: standalone metric functions plus the `GlucoseAnalysis`
> facade. See [ADR 0005](0005-pure-functions-composition.md). This document
> is kept for historical context.

## Context and Problem Statement

CGMPy has three distinct user-facing concerns: loading/processing data,
computing clinical metrics, and generating plots. Early versions exposed
a flat namespace where every function was importable directly
(e.g., `cgmpy.mean_glucose(df)`). This approach had three problems:

1. **No discoverability**: users had to read the source code to find
   what functions existed.
2. **Tight coupling**: data loaders, metric calculators, and plot
   functions all depended on each other implicitly through shared
   assumptions about data format.
3. **No state management**: every call required passing the full
   DataFrame and parameters, making multi-step analysis repetitive.

We need a user-friendly API that groups operations by concern, manages
state, and remains extensible without breaking backward compatibility.

## Considered Options

* **Facade + mixin pattern** — three facade classes
  (`GlucoseData`, `GlucoseMetrics`, `GlucosePlot`) that compose
  modular mixins internally. Users interact only with the facades.
* **Service-oriented** — standalone functions in namespaced modules
  (e.g., `cgmpy.metrics.basic.mean()`). No classes, no state.
* **Monolithic class** — a single `CGM` class that does everything:
  loading, metrics, plotting.

## Decision Outcome

Chosen option: **Facade + mixin pattern**, because:

1. Facades provide a clean entry point for users: `GlucoseData(file)` →
   `GlucoseMetrics(data)` → `GlucosePlot(data)`.
2. Mixins allow internal code organization without exposing internals.
   `ModularGlucoseData` and `ModularGlucoseMetrics` are mixins; they
   exist in `cgmpy.data` and `cgmpy.metrics` but are composed by
   the facades.
3. The pattern is extensible: adding a new metric means adding a method
   to the relevant mixin; the facade picks it up automatically.
4. State is managed naturally: `GlucoseData` holds the validated
   DataFrame, `GlucoseMetrics` references it through the mixin.

### Consequences

* Good, because users only need to learn three class names.
* Good, because internal refactors (e.g., splitting a large mixin) do
   not affect the public API.
* Good, because `GlucoseAnalysis` (a higher-level facade) can compose
   the three sub-facades into a single workflow.
* Bad, because the mixin hierarchy is non-trivial to understand for
   new contributors (`ModularGlucoseMetrics` inherits from three
   separate mixin classes). Mitigated by AGENTS.md explaining the
   pattern.

## Pros and Cons of the Options

### Facade + mixin

* Good, clean public API surface (three classes).
* Good, encourages separation of concerns.
* Good, easy to test (each mixin in isolation).
* Bad, internal complexity is higher than service-oriented approach.

### Service-oriented (flat functions)

* Good, simple mental model.
* Bad, no discoverability — users must know which module to import.
* Bad, every call repeats the same DataFrame and parameter arguments.
* Bad, harder to extend without breaking imports.

### Monolithic class

* Good, single import.
* Bad, violates Single Responsibility Principle.
* Bad, impossible to test in isolation.
* Bad, grows unboundedly as features are added.
