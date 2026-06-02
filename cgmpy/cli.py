"""Command-line interface for CGMPy.

This module exposes the :func:`main` function, which is registered as the
``cgmpy-info`` console-script entry point in ``pyproject.toml``. It prints
a human-readable (or JSON) summary of the installed CGMPy version, the
running Python interpreter, and the status of the optional dependencies
required by the ``[agata]`` and ``[dev]`` extras.

Typical usage::

    $ cgmpy-info
    CGMPy 0.5.2
    Python 3.11.9 on Linux-6.5.0-x86_64-with-glibc2.38

    Optional dependencies:
      py_agata        installed (0.0.8)
      mkdocs          not installed
      ruff            installed (0.6.9)
      pytest          installed (8.3.3)
      mypy            not installed

    $ cgmpy-info --json
    {
      "cgmpy_version": "0.5.2",
      ...
    }
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import sys
from typing import Any

# Optional dependencies that are worth reporting in `cgmpy-info`. Each tuple
# is (module-name, human-readable description, pip-extra that provides it).
_OPTIONAL_DEPS: tuple[tuple[str, str, str], ...] = (
    ("py_agata", "AGATA reference integration", "agata"),
    ("mkdocs", "Documentation build", "docs"),
    ("ruff", "Lint and format", "dev"),
    ("pytest", "Test runner", "dev"),
    ("mypy", "Static type checking", "dev"),
)


def _cgmpy_version() -> str:
    """Return the installed CGMPy version string."""
    from cgmpy import __version__

    return __version__


def _optional_dep_status() -> dict[str, dict[str, str]]:
    """Inspect each optional dependency and return a status dict.

    The shape is::

        {
            "py_agata": {"status": "installed", "version": "0.0.8", "extra": "agata"},
            "mkdocs":   {"status": "missing",   "version": "",       "extra": "docs"},
            ...
        }
    """
    result: dict[str, dict[str, str]] = {}
    for module_name, _description, extra in _OPTIONAL_DEPS:
        record: dict[str, str] = {"extra": extra}
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            record["status"] = "missing"
            record["version"] = ""
        else:
            try:
                mod = importlib.import_module(module_name)
                version = getattr(mod, "__version__", "unknown")
            except Exception as exc:
                version = f"error: {exc}"
            record["status"] = "installed"
            record["version"] = version
        result[module_name] = record
    return result


def _build_info() -> dict[str, Any]:
    """Assemble the full info dictionary that ``cgmpy-info`` reports."""
    return {
        "cgmpy_version": _cgmpy_version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "optional_dependencies": _optional_dep_status(),
    }


def _format_human(info: dict[str, Any]) -> str:
    """Render the info dict as a human-readable multi-line string."""
    lines: list[str] = [
        f"CGMPy {info['cgmpy_version']}",
        f"Python {info['python_version']} on {info['platform']}",
        "",
        "Optional dependencies:",
    ]
    for name, rec in info["optional_dependencies"].items():
        if rec["status"] == "installed":
            status = f"installed ({rec['version']})"
        else:
            status = f"missing (install with: pip install 'cgmpy[{rec['extra']}]')"
        lines.append(f"  {name:15s}  {status}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``cgmpy-info`` command.

    Args:
        argv: Optional argument list. Defaults to ``sys.argv[1:]``. Useful
            for tests that want to invoke the CLI in-process.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        prog="cgmpy-info",
        description="Display CGMPy environment and dependency information.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable summary.",
    )
    args = parser.parse_args(argv)

    info = _build_info()
    if args.json:
        print(json.dumps(info, indent=2, sort_keys=True))
    else:
        print(_format_human(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
