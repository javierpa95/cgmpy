# Pregnancy Analysis

CGMPy ships with first-class support for **gestational diabetes (GDM)**
analysis, including tighter cutoffs and dedicated metrics.

## Why pregnancy is different

The international consensus targets for pregnancy are tighter than for
the general diabetes population:

| Cutoff                | Diabetes (mg/dL) | Pregnancy (mg/dL) |
|-----------------------|------------------|-------------------|
| Hypoglycemia L2       | < 54             | < 55              |
| Hypoglycemia L1       | 54 – 70          | 55 – 63           |
| Target range          | 70 – 180         | **63 – 140**      |
| Hyperglycemia L1      | 180 – 250        | 140 – 250         |
| Hyperglycemia L2      | > 250            | > 250             |

Using the wrong cutoffs can over- or under-estimate glycemic control.

## Loading pregnancy data

CGMPy has a dedicated `PregnancyData` class that trims the trace to
the pregnancy window:

```python
from cgmpy import PregnancyData, PregnancyDataHandler

raw = PregnancyData("pregnancy.csv")
handler = PregnancyDataHandler(raw)
trimmed = handler.trim_to_pregnancy_window()
```

If your data is already filtered, you can use the standard
`GlucoseData` instead.

## Computing gestational diabetes metrics

```python
from cgmpy import PregnancyAnalysis
from cgmpy.metrics.targets import get_targets

# Pregnancy cutoffs (63-140 mg/dL)
targets = get_targets("pregnancy")

gdm = PregnancyAnalysis(data=trimmed, targets=targets)
metrics = gdm.compute_all()

for name, value in metrics.items():
    print(f"  {name:30s} = {value:.2f}")
```

Output (example):

```
  tir                            = 78.30
  tar1                           = 18.20
  tbr1                           = 3.50
  mean_glucose                   = 112.40
  gmi                            = 5.80
  ...
```

## Per-meal TIR

Pregnancy recommendations often look at glycemic control **per meal**
(breakfast, lunch, dinner, snack). `PregnancyAnalysis` provides
`time_in_range_per_meal()`:

```python
per_meal = gdm.time_in_range_per_meal()
print(per_meal)
```

| meal     |   tir |  tar1 |  tbr1 |
|----------|-------|-------|-------|
| breakfast|  72.1 |  24.5 |   3.4 |
| lunch    |  80.4 |  17.0 |   2.6 |
| dinner   |  81.5 |  15.2 |   3.3 |
| snack    |  85.0 |  12.4 |   2.6 |

## Glycemia Risk Index (GRI) in pregnancy

The GRI is a single number summarizing hypoglycemia and hyperglycemia
risk. In pregnancy, the **pregnancy-specific GRI** is recommended:

```python
gri = gdm.glycemia_risk_index(pregnancy=True)
print(f"Pregnancy GRI: {gri:.1f}")
```

## Time-of-day analysis

```python
overnight = gdm.overnight_metrics()  # 00:00 - 06:00
daytime = gdm.daytime_metrics()      # 06:00 - 00:00
```

## Visualization

The `plot_agp` and `plot_daily` methods accept pregnancy-target-aware
data via the `GlucoseAnalysis` facade:

```python
from cgmpy import GlucoseAnalysis, GlucoseData
from cgmpy.metrics.targets import get_targets

data = GlucoseData("data.csv", target_type="pregnancy")
analysis = GlucoseAnalysis(data=data)
analysis.plot_agp()
```

## References

- Battelino T. *et al.* (2019). [Clinical Targets for Continuous Glucose
  Monitoring Data Interpretation: Recommendations From the International
  Consensus on Time in Range](https://doi.org/10.2337/dci19-0028).
  *Diabetes Care*, 42(8), 1593-1603.
- Danne T. *et al.* (2017). [International Consensus on Use of Continuous
  Glucose Monitoring](https://doi.org/10.2337/dc17-1600). *Diabetes Care*,
  40(12), 1631-1640.
- Hernández T.L. *et al.* (2022). [Guidelines for the Study of Glucose
  Metabolism in Pregnancy](https://doi.org/10.1210/endrev/bnab039).
  *Endocrine Reviews*.

## See also

- [Computing metrics](computing-metrics.md) — the general API.
- [Targets](../api/metrics.md#targets) — the full cutoff table.
- [Example: 02_pregnancy](https://github.com/javierpa95/cgmpy/tree/main/examples/02_pregnancy).
