# CGMPy — Usage Examples

This directory contains runnable examples that show how to use CGMPy in
real-world scenarios. Examples are **numbered** to suggest a reading order.

> **Setup**: install CGMPy with the dev dependencies before running the examples:
> ```bash
> pip install -e ".[dev,agata]"
> ```

## Index

| # | Folder                          | Topic                                                                 |
|---|---------------------------------|-----------------------------------------------------------------------|
| 1 | [`01_quickstart/`](01_quickstart/)             | End-to-end analysis: load → metrics → plot.                  |
| 2 | [`02_pregnancy/`](02_pregnancy/)              | Gestational diabetes analysis with pregnancy cutoffs.       |
| 3 | [`03_agata_comparison/`](03_agata_comparison/)    | Side-by-side parity check against the AGATA library.        |
| 4 | [`04_performance/`](04_performance/)           | Benchmark suite for every metric on synthetic 30-day data.  |

## How to run

```bash
# From the project root:
python examples/01_quickstart/basic_analysis.py
python examples/02_pregnancy/gestational_diabetes.py
python examples/03_agata_comparison/comparison.py   # requires agata
python examples/04_performance/benchmark.py
```

## Adding a new example

1. Create `examples/NN_<topic>/<script>.py` where `NN` is the next number.
2. The script must be **self-contained** and runnable from the project root
   via `python examples/NN_<topic>/<script>.py`.
3. Reference data with a path relative to the project root:
   ```python
   from pathlib import Path
   FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "data" / "my_file.csv"
   ```
4. Add a one-line entry to the table above.
5. Open a PR — the `pr-standards` workflow will validate the PR title.

## Data

The examples use the synthetic datasets in `tests/fixtures/data/`:

- `dm.csv` — anonymized Type 1 Diabetes patient.
- `nodm.csv` — anonymized non-diabetic subject.
- `pregnancy.csv` — anonymized pregnancy trace.

> **Never replace these with real data.** See [`SECURITY.md`](../../SECURITY.md)
> for the project's strict policy on Protected Health Information (PHI).
