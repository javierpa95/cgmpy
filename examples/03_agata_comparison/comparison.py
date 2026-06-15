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
        from cgmpy import AgataAnalysis, GlucoseAnalysis
    except ImportError:
        print("AGATA is not installed. Run: pip install -e .[agata]")
        return

    print("Computing metrics with AgataAnalysis and GlucoseAnalysis...\n")

    # 1. Run AGATA
    try:
        agata = AgataAnalysis(data_source=str(FIXTURE))
        agata_results = agata.run()
    except Exception as exc:
        print(f"AgataAnalysis failed: {exc}")
        agata_results = {}

    # 2. Run CGMPy via the GlucoseAnalysis facade. We build a flat dict from
    #    the public metric methods (the pure-function results behind the facade).
    try:
        cgm = GlucoseAnalysis(str(FIXTURE))
        cgm_results = {
            "Mean": cgm.mean(),
            "Median": cgm.median(),
            "Std": cgm.sd(),
            "CV": cgm.cv(),
            "GMI": cgm.gmi(),
            "TIR": cgm.TIR(),
            "TAR180": cgm.TAR180(),
            "TBR70": cgm.TBR70(),
            "LBGI": cgm.LBGI(),
            "HBGI": cgm.HBGI(),
            "GRI": cgm.GRI().get("GRI"),
        }
    except Exception as exc:
        print(f"GlucoseAnalysis failed: {exc}")
        cgm_results = {}

    # 3. Side-by-side table. AGATA nests everything under `time_in_ranges`
    #    (plural); the cgm-side keys are the flat keys built above.
    metric_map = [
        ("Mean glucose (mg/dL)", ("variability", "mean_glucose"), ("Mean",)),
        ("Median glucose (mg/dL)", ("variability", "median_glucose"), ("Median",)),
        ("Standard deviation", ("variability", "std_glucose"), ("Std",)),
        ("CV (%)", ("variability", "cv_glucose"), ("CV",)),
        ("GMI (%)", ("variability", "gmi"), ("GMI",)),
        (
            "Time in target",
            ("time_in_ranges", "time_in_target"),
            ("TIR",),
        ),
        (
            "TAR1 (>180)",
            ("time_in_ranges", "time_in_l1_hyperglycemia"),
            ("TAR180",),
        ),
        (
            "TBR1 (<70)",
            ("time_in_ranges", "time_in_l1_hypoglycemia"),
            ("TBR70",),
        ),
        ("LBGI", ("risk", "lbgi"), ("LBGI",)),
        ("HBGI", ("risk", "hbgi"), ("HBGI",)),
        ("GRI", ("risk", "gri"), ("GRI",)),
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
