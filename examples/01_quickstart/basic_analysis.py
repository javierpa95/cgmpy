"""Quickstart: end-to-end CGMPy workflow in ~30 lines.

This example loads a synthetic CGM file, computes all standard clinical
metrics, and renders the human-readable summary.

Run from the project root:

    python examples/01_quickstart/basic_analysis.py

Prerequisites:
    pip install -e ".[dev]"
"""

from __future__ import annotations

from pathlib import Path

from cgmpy import GlucoseAnalysis

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "data" / "dm.csv"


def main() -> None:
    """Run a complete CGMPy analysis on the bundled synthetic dataset."""
    # 1. Load and analyse the CGM file.
    analysis = GlucoseAnalysis(str(FIXTURE))

    # 2. Human-readable summary (DATA / BASIC / TIME IN RANGE / VARIABILITY).
    print(analysis.get_summary_string())

    # 3. Programmatic access to the structured report.
    report = analysis.get_comprehensive_report()
    tir = report["time_statistics"]["TIR"]
    print(f"\nTime in Range (TIR, 70-180 mg/dL): {tir:.1f} %")

    # 4. Switch to pregnancy-specific cutoffs (Battelino et al. 2019) by
    #    swapping the `targets` attribute on the analysis instance.
    from cgmpy.metrics.targets import get_targets

    pregnancy_analysis = GlucoseAnalysis(str(FIXTURE))
    pregnancy_analysis.targets = get_targets("pregnancy")
    pregnancy_tir = pregnancy_analysis.time_statistics()["TIR"]
    print(f"Pregnancy-adjusted TIR (63-140 mg/dL): {pregnancy_tir:.1f} %")

    # 5. Render the AGP dashboard.
    out_path = Path(__file__).parent / "agp_dashboard.png"
    import matplotlib.pyplot as plt

    analysis.plot_agp()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"\nAGP saved to: {out_path}")


if __name__ == "__main__":
    main()
