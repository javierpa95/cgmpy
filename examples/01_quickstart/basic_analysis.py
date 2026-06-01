"""Quickstart: end-to-end CGMPy workflow in ~30 lines.

This example loads a synthetic CGM file, computes all standard clinical
metrics, and generates the Ambulatory Glucose Profile (AGP) dashboard.

Run from the project root:

    python examples/01_quickstart/basic_analysis.py

Prerequisites:
    pip install -e ".[dev]"
"""

from __future__ import annotations

from pathlib import Path

from cgmpy import GlucoseAnalysis
from cgmpy.metrics.targets import get_targets

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "data" / "dm.csv"


def main() -> None:
    """Run a complete CGMPy analysis on the bundled synthetic dataset."""
    # 1. Load and analyze
    analysis = GlucoseAnalysis(str(FIXTURE))

    # 2. Human-readable summary
    print(analysis.get_summary_string())

    # 3. Programmatic access
    report = analysis.get_comprehensive_report()
    tir = report["time_in_range"]["tir"]
    print(f"\nTime in Range (TIR): {tir:.1f} %")

    # 4. Try with pregnancy-specific cutoffs
    pregnancy = get_targets("pregnancy")
    pregnancy_report = analysis.get_comprehensive_report(targets=pregnancy)
    print(
        f"Pregnancy-adjusted TIR (63-140 mg/dL): {pregnancy_report['time_in_range']['tir']:.1f} %"
    )

    # 5. Render the AGP dashboard (saved next to this file)
    out_path = Path(__file__).parent / "agp_dashboard.png"
    analysis.plot_comprehensive_dashboard(save_path=str(out_path))
    print(f"\nDashboard saved to: {out_path}")


if __name__ == "__main__":
    main()
