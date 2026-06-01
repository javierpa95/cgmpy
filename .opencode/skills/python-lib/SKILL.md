# python-lib Skill

> **When to use:** any time you write or modify Python in a library (non-script, non-application) project like CGMPy.

## Library vs. Script vs. Application

CGMPy is a **library**: code that other people `import`. This implies stricter
rules than a one-off script:

- **No side effects on import.** Importing `cgmpy` must not read files, open
  network connections, set environment variables, or print to stdout.
- **No `print()` in library code.** Use the `logging` module.
- **Lazy imports** for heavy or optional dependencies (e.g., `agata`).
- **No global mutable state** unless documented and thread-safe.

## Type Hints

- **All public functions and methods** must have type hints.
- **Internal** functions should have type hints (CI checks this).
- Use `from __future__ import annotations` at the top of each module for
  forward references.
- Use `from typing import` for collections: `list[int]`, `dict[str, Any]`,
  `tuple[float, ...]`, `Optional[T]` (or `T | None` on 3.10+).

## Docstrings (Google Style)

```python
def time_in_range(data: GlucoseData, targets: GlucoseTargets) -> float:
    """Compute the percentage of time spent in the target glucose range.

    Args:
        data: A ``GlucoseData`` instance with at least 24h of valid samples.
        targets: Glucose cutoffs (e.g., ``GlucoseTargets.standard()``).

    Returns:
        The percentage of time (0–100) spent in the target range.

    Raises:
        ValueError: If ``data`` has fewer than 24 valid samples.

    Example:
        >>> targets = GlucoseTargets.standard()
        >>> tir = time_in_range(data, targets)
        >>> 0 <= tir <= 100
        True
    """
```

- One-line summary.
- Blank line.
- Extended description (optional).
- `Args`, `Returns`, `Raises`, `Example` sections (any subset).

## Imports

- **stdlib** → **third-party** → **first-party (`cgmpy`)**.
- No wildcard imports.
- No `from cgmpy import X` inside `cgmpy/` modules — use **relative imports**:
  ```python
  from .core import ModularGlucoseData
  from ..utils.date_utils import parse_date
  ```
- Group long imports with parentheses for readability.

## Errors

- Define **custom exception classes** that inherit from a package-level base:
  ```python
  class GlucoseDataError(ValueError):
      """Base exception for cgmpy.data errors."""
  ```
- Use the **most specific** standard exception (`ValueError`, `TypeError`,
  `FileNotFoundError`) when no custom class applies.
- **Never** use bare `except:`.
- Log the traceback before re-raising: `logger.exception("...")`.

## Logging

```python
import logging

logger = logging.getLogger(__name__)

# Library code — never print()
logger.debug("Loaded %d records from %s", n_records, path)
logger.info("Computing metrics for %s", targets.name)
logger.warning("Dropped %d invalid samples", n_dropped)
logger.error("Failed to parse %s", path, exc_info=True)
```

- Module-level `logger = logging.getLogger(__name__)`.
- `logger.exception()` in `except:` blocks, not `logger.error()`.

## Public API Stability

Symbols exported from `cgmpy/__init__.py` form the **public API**. They are
**stable**. To change one:

1. Add a deprecation warning (Python `DeprecationWarning`).
2. Document the change in `CHANGELOG.md` and `docs/development/release-process.md`.
3. After 1 minor version, remove the old symbol.

## Testing

- Every public function must have at least one test.
- New metrics must have a **clinical regression test** (Phase 3+).
- Use `pytest.approx(expected, abs=1e-6)` for float comparisons.

## Performance

- **Vectorize** with NumPy / Pandas — no Python loops over rows.
- **Avoid** `df.apply(lambda x: ...)` when a vectorized expression exists.
- **Profile** with `cProfile` or `py-spy` before optimizing.

## Common Pitfalls

- ❌ `from cgmpy import X` inside `cgmpy/` modules → circular imports.
- ❌ `print(df.head())` in a function for debugging.
- ❌ `except: pass` swallowing real errors.
- ❌ Mutable default arguments: `def f(items: list = [])`.
- ❌ `==` for float comparison instead of `math.isclose()` or `pytest.approx`.
- ❌ Shadowing builtins: `def list(items): ...`.
