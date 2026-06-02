# Command-line interface

CGMPy ships a small command-line tool, `cgmpy-info`, that prints a snapshot
of the installed CGMPy environment: package version, Python interpreter, and
the status of the optional dependencies that the `[agata]`, `[docs]`, and
`[dev]` extras provide.

It is intended for sysadmins, ops engineers, support staff, and clinicians
who need a quick way to check what is installed before opening a support
ticket or running a script.

## Usage

The tool is installed automatically when you install CGMPy (no extra
dependencies). Run it from any shell:

```bash
cgmpy-info
```

Example output:

```
CGMPy 0.5.2
Python 3.11.9 on Linux-6.5.0-x86_64-with-glibc2.38

Optional dependencies:
  py_agata        installed (0.0.8)
  mkdocs          not installed
  ruff            installed (0.6.9)
  pytest          installed (8.3.3)
  mypy            not installed
```

### `--json`

For scripts and CI, use `--json` to emit a machine-readable payload that
is easy to parse with `jq` or Python:

```bash
cgmpy-info --json
```

```json
{
  "cgmpy_version": "0.5.2",
  "optional_dependencies": {
    "mkdocs":   {"extra": "docs", "status": "missing",   "version": ""},
    "mypy":     {"extra": "dev",  "status": "missing",   "version": ""},
    "py_agata": {"extra": "agata","status": "installed", "version": "0.0.8"},
    "pytest":   {"extra": "dev",  "status": "installed", "version": "8.3.3"},
    "ruff":     {"extra": "dev",  "status": "installed", "version": "0.6.9"}
  },
  "platform": "Linux-6.5.0-x86_64-with-glibc2.38",
  "python_version": "3.11.9"
}
```

## What the output tells you

| Field                     | Meaning                                                    |
|---------------------------|------------------------------------------------------------|
| `cgmpy_version`           | Installed CGMPy package version.                           |
| `python_version`          | Version of the active Python interpreter.                  |
| `platform`                | OS / kernel / arch string (from `platform.platform()`).    |
| `optional_dependencies`   | Map of module name → status. Each entry has:               |
| &nbsp;&nbsp;`status`      | `installed` or `missing`.                                  |
| &nbsp;&nbsp;`version`     | Installed version (or empty string if missing).            |
| &nbsp;&nbsp;`extra`       | The pip extra that provides the module.                    |

The reported optional modules and their extras are:

| Module     | Pip extra | Purpose                                  |
|------------|-----------|------------------------------------------|
| `py_agata` | `[agata]` | AGATA reference integration.             |
| `mkdocs`   | `[docs]`  | Build the documentation site locally.    |
| `ruff`     | `[dev]`   | Linter and formatter.                    |
| `pytest`   | `[dev]`   | Test runner.                             |
| `mypy`     | `[dev]`   | Static type checker.                     |

If a module is reported as `missing`, the human-readable output suggests
the matching `pip install 'cgmpy[extra]'` command.

## Use cases

### CI gate on `py_agata`

Use `--json` in continuous integration to assert that `py_agata` is
available before running AGATA-dependent tests:

```bash
if ! cgmpy-info --json | python -c "import json,sys; sys.exit(0 if json.load(sys.stdin)['optional_dependencies']['py_agata']['status']=='installed' else 1)"; then
  echo "py_agata is not installed; skipping AGATA tests"
  exit 0
fi
```

### Diagnose a user support ticket

When a user reports "AGATA does not work", the first thing to ask is the
output of `cgmpy-info`. It shows in one glance whether `py_agata` is
present, which Python is active, and which CGMPy version is running.

### Confirm a fresh dev environment

```bash
pip install 'cgmpy[dev,docs,agata]'
cgmpy-info            # should list every optional dep as installed
```

## See also

- [Installation](../getting-started/installation.md) — pip extras and what
  they provide.
- [Loading data](loading-data.md) — what to do once CGMPy is installed.
- [API reference → Data](../api/data.md) — programmatic access to the same
  information is available via `cgmpy.__version__` and
  `importlib.util.find_spec()`.
