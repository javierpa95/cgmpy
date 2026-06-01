"""Cross-validation of CGMPy metrics against the AGATA reference library.

AGATA is the de-facto reference implementation for CGM analysis in Python.
This example runs the same metric on the same data through both libraries
and reports the absolute difference.

Run from the project root:

    python examples/03_agata_comparison/comparison.py

Prerequisites:
    pip install -e ".[agata]"   # installs the agata optional dependency
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "data" / "dm.csv"


def main() -> None:
    try:
        from cgmpy import AgataAnalysis, GlucoseMetrics
    except ImportError:
        print("AGATA is not installed. Run: pip install -e .[agata]")
        return

    print("Computing metrics with AgataAnalysis and GlucoseMetrics...\n")

    # 1. Run AGATA
    try:
        agata = AgataAnalysis(data_source=str(FIXTURE))
        agata_results = agata.run()
    except Exception as exc:
        print(f"AgataAnalysis failed: {exc}")
        agata_results = {}

    # 2. Run CGMPy
    try:
        cgm = GlucoseMetrics(data_source=str(FIXTURE))
        cgm_results = cgm.all()
    except Exception as exc:
        print(f"GlucoseMetrics failed: {exc}")
        cgm_results = {}

    # 3. Side-by-side table.
    # CGMPy uses the same dict structure as `GlucoseMetrics.all()` (see
    # `cgmpy/metrics/__init__.py`); AGATA nests everything under
    # `time_in_ranges` (plural). The cgm-side keys are the actual ones
    # emitted by CGMPy.
    metric_map = [
        ("Mean glucose (mg/dL)", ("variability", "mean_glucose"), ("basic", "Mean")),
        ("Median glucose (mg/dL)", ("variability", "median_glucose"), ("basic", "Median")),
        ("Standard deviation", ("variability", "std_glucose"), ("basic", "Std")),
        ("CV (%)", ("variability", "cv_glucose"), ("basic", "CV")),
        ("GMI (%)", ("variability", "gmi"), ("basic", "GMI")),
        (
            "Time in target",
            ("time_in_ranges", "time_in_target"),
            ("time_in_range", "current_targets", "TIR"),
        ),
        (
            "TAR1 (>180)",
            ("time_in_ranges", "time_in_l1_hyperglycemia"),
            ("time_in_range", "standard_ranges", "TAR180"),
        ),
        (
            "TBR1 (<70)",
            ("time_in_ranges", "time_in_l1_hypoglycemia"),
            ("time_in_range", "standard_ranges", "TBR70"),
        ),
        ("LBGI", ("risk", "lbgi"), ("variability", "quality_metrics", "lbgi")),
        ("HBGI", ("risk", "hbgi"), ("variability", "quality_metrics", "hbgi")),
        ("GRI", ("risk", "gri"), ("variability", "quality_metrics", "gri")),
    ]

    rows = []
    for label, a_path, c_path in metric_map:
        a_val = _nested(agata_results, a_path)
        c_val = _nested(cgm_results, c_path)
        if isinstance(a_val, int | float) and isinstance(c_val, int | float):
            diff = abs(a_val - c_val)
            rows.append(
                {
                    "metric": label,
                    "agata": a_val,
                    "cgmpy": c_val,
                    "abs_diff": diff,
                }
            )
        else:
            rows.append(
                {
                    "metric": label,
                    "agata": a_val,
                    "cgmpy": c_val,
                    "abs_diff": "n/a",
                }
            )

    df = pd.DataFrame(rows)
    print("=== CGMPy vs. AGATA — metric parity ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def _nested(d: dict, path: tuple) -> object:
    """Safely descend into a nested dict."""
    current = d
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


if __name__ == "__main__":
    main()
