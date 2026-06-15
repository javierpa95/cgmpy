# Deprecation Policy

## Why deprecation matters for scientific libraries

Researchers have scripts that they run for years. A paper's analysis pipeline should be reproducible long after it is published. Breaking changes erode trust in the library and undermine the scientific process.

CGMPy's policy is simple: **we never break your code without warning**.

## The deprecation cycle (2 releases)

```
Release N:     Class is marked @deprecated, still works (warning shown)
Release N+1:   Class still works, warning is louder (DeprecationWarning -> FutureWarning)
Release N+2:   Class is REMOVED
```

Users get one release to migrate and one release as a grace period before the code is removed.

## Python's warning system

Python's built-in `warnings` module is the backbone of the deprecation policy. Two categories are used:

```python
import warnings

# For internal/developer-facing deprecations (subtle)
warnings.warn(
    "GlucoseMetrics is deprecated. Use GlucoseAnalysis instead.",
    DeprecationWarning,
    stacklevel=2,
)

# For user-facing deprecations (always visible)
warnings.warn(
    "GlucoseMetrics will be removed in v0.7.0. Use GlucoseAnalysis instead.",
    FutureWarning,
    stacklevel=2,
)
```

`DeprecationWarning` is hidden by default in production code -- it is intended for developers. `FutureWarning` is always shown to end users and is the appropriate category for public API deprecations.

## How CGMPy does it

Every deprecated class or function warns at the point of use. The message includes the version in which deprecation began, the replacement API, and the version in which removal will happen.

```python
# In the deprecated class __init__
class GlucoseMetrics:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "GlucoseMetrics is deprecated since v0.6.0. "
            "Use 'GlucoseAnalysis(GlucoseData(...))' instead. "
            "GlucoseMetrics will be removed in v0.8.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
```

## The `@deprecated` decorator pattern

For functions and methods, a reusable decorator keeps the boilerplate in one place:

```python
import functools
import warnings
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

def deprecated(message: str, category: type = DeprecationWarning) -> Callable[[F], F]:
    """Decorator that marks a function or class as deprecated."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, category, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


@deprecated("Use 'mean(glucose: pd.Series)' instead.", FutureWarning)
def calculate_mean(self) -> float:
    return self.data["glucose"].mean()
```

The decorator preserves the original function's signature, name, and docstring via `functools.wraps`, so introspection tools and IDEs continue to work.

## What CGMPy is deprecating (from this refactor)

> **Note.** CGMPy is still pre-1.0, so the v0.6.0 refactor **removed** the
> legacy mixin names below outright instead of keeping deprecation aliases
> (see [ADR 0005](../../architecture/decisions/0005-pure-functions-composition.md)).
> The table illustrates how a graceful, post-1.0 deprecation *would* be
> documented; it does not imply these aliases still exist.

| Old (removed in v0.6.0)        | New (recommended)                |
|--------------------------------|----------------------------------|
| `GlucoseMetrics`               | `GlucoseAnalysis(GlucoseData(...))` |
| `GlucosePlot`                  | `GlucoseAnalysis.plot_*()`       |
| `ModularGlucoseData`           | `GlucoseData`                    |
| `ModularGlucoseMetrics`        | `cgmpy.metrics` pure functions   |

## How users should migrate

To surface all deprecation warnings during migration:

```python
import warnings
warnings.filterwarnings("default")
```

To catch accidental usage of deprecated APIs in a test suite, run pytest with:

```bash
pytest -W error::DeprecationWarning
```

This turns every `DeprecationWarning` into a test failure, making it impossible to ignore.

## Why semantic versioning matters

Version numbers communicate the degree of change. In CGMPy:

- **v0.6.0** -- may deprecate (minor bump)
- **v0.7.0** -- may remove deprecated items (minor bump, but documented in advance)
- **v1.0.0+** -- breaking changes require a major version bump only

This means that within the v0.x series, deprecation and removal can happen across minor releases as long as the two-release warning cycle is respected. After v1.0.0, breaking changes are restricted to major version bumps per standard semantic versioning.
