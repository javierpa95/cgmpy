"""Pregnancy-specific glucose analysis with gestational diabetes cutoffs.

Demonstrates:
- Loading pregnancy-trimester data.
- Using `GlucoseTargets.pregnancy()` (TIR 63–140 mg/dL).
- Computing `GestationalDiabetes`-specific metrics.

Run from the project root:

    python examples/02_pregnancy/gestational_diabetes.py
"""

from __future__ import annotations

from pathlib import Path

from cgmpy import GestationalDiabetes, PregnancyData, PregnancyDataHandler
from cgmpy.metrics.targets import get_targets

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "data" / "pregnancy.csv"


def main() -> None:
    # 1. Load and filter the pregnancy window
    raw = PregnancyData(str(FIXTURE))
    handler = PregnancyDataHandler(raw)
    trimmed = handler.trim_to_pregnancy_window()

    # 2. Pregnancy-specific cutoffs (Battelino et al. 2019)
    targets = get_targets("pregnancy")

    # 3. Compute gestational diabetes metrics
    gdm = GestationalDiabetes(data=trimmed, targets=targets)
    metrics = gdm.compute_all()

    print("=== Gestational Diabetes Metrics ===")
    for name, value in metrics.items():
        print(f"  {name:30s} = {value:.2f}")

    # 4. Time-in-range breakdown per meal (optional)
    per_meal = gdm.time_in_range_per_meal()
    print("\n=== Time in Range per Meal ===")
    print(per_meal.to_string(index=False))


if __name__ == "__main__":
    main()
