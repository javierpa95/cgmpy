# Installation

CGMPy is a pure-Python library with a small set of well-known scientific
dependencies. Installation takes a single `pip` or `uv` command.

## Requirements

- **Python ≥ 3.10**
- **pip** or **uv**
- Optional: **Git** (for installing from source)

CGMPy's core dependencies:

| Package       | Why                                        |
|---------------|--------------------------------------------|
| `numpy`       | Numeric operations                         |
| `pandas`      | Tabular data manipulation                  |
| `matplotlib`  | Static plots (AGP, daily traces)           |
| `seaborn`     | Statistical plot styling                   |
| `pyarrow`     | Parquet I/O (fast binary CGM dumps)        |
| `openpyxl`    | Excel export                               |

## Stable release

```bash
# pip
pip install cgmpy

# uv
uv pip install cgmpy
```

After installation, verify:

```bash
python -c "import cgmpy; print(cgmpy.__version__)"
```

## Optional dependencies

CGMPy ships with three optional dependency groups. Install them as needed:

| Group   | What it adds                                              | Install                  |
|---------|-----------------------------------------------------------|--------------------------|
| `agata` | The [AGATA](https://github.com/gcappon/agata) reference library, used for parity tests and the `AgataAnalysis` wrapper. | `pip install cgmpy[agata]` |
| `dev`   | `pytest`, `ruff`, `mypy`, `bandit`, `pre-commit`, `build`, `twine` for development. | `pip install cgmpy[dev]` |
| `docs`  | `mkdocs`, `mkdocs-material`, `mkdocstrings` to build the documentation site. | `pip install cgmpy[docs]` |

Install everything at once:

```bash
pip install cgmpy[dev,docs,agata]
```

## From source

```bash
git clone https://github.com/javierpa95/cgmpy.git
cd cgmpy
pip install -e ".[dev]"
```

The `-e` flag installs in **editable** mode, so your local code changes
take effect immediately without reinstalling.

## Verify the install

```bash
# Library imports cleanly
python -c "from cgmpy import GlucoseAnalysis; print('OK')"

# Tests pass
pytest -q

# Lint clean
ruff check .

# Type check (relaxed — CGMPy is gradually typed)
mypy cgmpy/
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'cgmpy'`

Your virtual environment is not active. Run:

```bash
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows PowerShell
```

Then `pip install -e ".[dev]"` again.

### `ImportError: agata`

You are trying to use `AgataAnalysis` without the `agata` package.
Install with `pip install cgmpy[agata]`.

### `pyarrow` wheels fail on Windows

Try `pip install pyarrow --only-binary=:all:` or use `uv`, which usually
resolves binary wheels automatically.

### Old `numpy` / `pandas`

CGMPy requires `numpy >= 1.23` and `pandas >= 1.5`. If you are on an
older stack, upgrade: `pip install -U numpy pandas`.

## Next steps

- [Quickstart](quickstart.md) — run your first analysis in 5 minutes.
- [Data formats](data-formats.md) — what CSV columns CGMPy expects.
