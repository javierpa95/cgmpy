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

    # 3. Side-by-side table
    metric_map = [
        ("Mean glucose (mg/dL)", ("variability", "mean_glucose"), ("basic", "mean")),
        ("Median glucose (mg/dL)", ("variability", "median_glucose"), ("basic", "median")),
        ("Standard deviation", ("variability", "std_glucose"), ("basic", "std")),
        ("CV (%)", ("variability", "cv_glucose"), ("basic", "cv")),
        ("GMI (%)", ("variability", "gmi"), ("basic", "gmi")),
        ("Time in target", ("time_in_ranges", "time_in_target"), ("basic", "tir")),
        ("TAR1 (>180)", ("time_in_ranges", "time_in_l1_hyperglycemia"), ("basic", "tar180")),
        ("TBR1 (<70)", ("time_in_ranges", "time_in_l1_hypoglycemia"), ("basic", "tbr70")),
        ("LBGI", ("risk", "lbgi"), ("basic", "lbgi")),
        ("HBGI", ("risk", "hbgi"), ("basic", "hbgi")),
        ("GRI", ("risk", "gri"), ("basic", "gri")),
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
    print(df.to_string(index=False, float_format="%.3f"))


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
