"""Pregnancy-specific glucose analysis with gestational diabetes cutoffs.

Demonstrates:
- Loading pregnancy-trimester data with `PregnancyData`.
- Using `PregnancyAnalysis` to compute trimester-by-trimester metrics
  with the pregnancy TIR cutoffs (63-140 mg/dL, Battelino et al. 2019).
- Rendering a per-trimester summary via `calculate_all_metrics(flatten=True)`.

Run from the project root:

    python examples/02_pregnancy/gestational_diabetes.py
"""

from __future__ import annotations

from pathlib import Path

from cgmpy import PregnancyAnalysis

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "data" / "pregnancy.csv"

# The bundled fixture is a 2.5-year trace (Jul 2022 → Jan 2025). The
# pregnancy portion is roughly 2022-07-24 → 2024-04-15 (a full-term delivery
# at 38 weeks). Adjust these to your own data.
DELIVERY_DATE = "2024-04-15"
GESTATION_WEEK_AT_DELIVERY = 38


def main() -> None:
    # 1. PregnancyAnalysis uses PregnancyData internally, so it expects the
    #    same `delivery_date` and `week` arguments. It automatically:
    #      - filters the DataFrame to the conception → delivery window
    #      - splits the data into first / second / third trimester frames
    #      - wraps each trimester in a `GlucoseAnalysis` instance for
    #        per-trimester analysis.
    gdm = PregnancyAnalysis(
        data_source=str(FIXTURE),
        delivery_date=DELIVERY_DATE,
        week=GESTATION_WEEK_AT_DELIVERY,
    )

    # 2. Human-readable per-trimester summary.
    print(gdm)
    print()

    # 3. Structured per-trimester metrics (nested dict).
    nested = gdm.calculate_all_metrics(flatten=False)
    print("Overall GMI:", f"{nested['overall']['GMI']:.2f} %")
    print("Overall TIR (pregnancy cutoffs):", f"{nested['overall']['TIR']:.2f} %")
    print()

    # 4. Flat dict with `total_`, `t1_`, `t2_`, `t3_`, `gest_` prefixes —
    #    handy for writing straight to a CSV / DataFrame.
    flat = gdm.calculate_all_metrics(flatten=True)
    print("Flattened (first 10 keys):")
    for key in list(flat.keys())[:10]:
        print(f"  {key:30s} = {flat[key]}")


if __name__ == "__main__":
    main()
