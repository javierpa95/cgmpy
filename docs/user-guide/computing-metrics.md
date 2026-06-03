# Computing Metrics

CGMPy implements the consensus set of clinical metrics for CGM data
analysis (Battelino et al., *Diabetes Care* 2019).

## The high-level facade

The fastest way to get every metric is the `GlucoseAnalysis` facade:

```python
from cgmpy import GlucoseAnalysis

analysis = GlucoseAnalysis("data.csv")
report = analysis.get_comprehensive_report()
# report is a nested dict organized by category
```

The `report` dictionary has the following top-level keys:

- `basic` — mean, median, GMI, SD, CV.
- `time_in_range` — TIR, TAR (1/2), TBR (1/2).
- `variability` — CV, MAGE, MODD, CONGA, J-Index, LBGI, HBGI, GRI, ADRR.
- `quality` — total gaps, max gap, completeness.
- `targets` — the cutoffs that were used.

## The modular API

For finer control, use `GlucoseAnalysis`:

```python
from cgmpy import GlucoseData, GlucoseAnalysis

data = GlucoseData("data.csv")
analysis = GlucoseAnalysis(data)
```

### Basic statistics

```python
basic = analysis.basic()
print(basic.mean())     # mg/dL
print(basic.median())   # mg/dL
print(basic.gmi())      # Glucose Management Indicator, %
print(basic.std())      # mg/dL
print(basic.cv())       # %, coefficient of variation
print(basic.iqr())      # mg/dL
print(basic.percentile(25))  # Q1
print(basic.distribution_analysis())  # full dict
```

### Time in range

```python
tir = analysis.time_in_range()
print(tir.tir())    # 70-180 mg/dL (default)
print(tir.tar1())   # 180-250 mg/dL
print(tir.tar2())   # > 250 mg/dL
print(tir.tbr1())   # 54-70 mg/dL
print(tir.tbr2())   # < 54 mg/dL
```

### Variability

```python
v = analysis.variability()
print(v.cv())       # Coefficient of variation
print(v.sd())       # Standard deviation
print(v.mage())     # Mean Amplitude of Glycemic Excursions
print(v.modd())     # Mean of Daily Differences
print(v.conga())    # Continuous Overlapping Net Glycemic Action
print(v.j_index())  # J-Index
print(v.lbgi())     # Low Blood Glucose Index
print(v.hbgi())     # High Blood Glucose Index
print(v.gri())      # Glycemia Risk Index
print(v.adrr())     # Average Daily Risk Range
```

## Glucose targets

Every metric that uses cutoffs accepts a `GlucoseTargets` instance.
CGMPy ships with two profiles:

```python
from cgmpy.metrics.targets import GlucoseTargets, get_targets

# Diabetes (international consensus)
t1 = GlucoseTargets.standard()
# or
t1 = get_targets("diabetes")

# Pregnancy (tighter cutoffs)
t2 = GlucoseTargets.pregnancy()
# or
t2 = get_targets("pregnancy")
```

Pass the target to the metric or the facade:

```python
analysis = GlucoseAnalysis(data, targets=t2)
tir_pregnancy = analysis.time_in_range().tir()  # uses 63-140 mg/dL
```

### Available cutoffs

| Profile    | Hypo L2 | Hypo L1 | Target Low | Target High | Hyper L1 | Hyper L2 |
|------------|---------|---------|------------|-------------|----------|----------|
| Diabetes   | 54      | 70      | 70         | 180         | 180      | 250      |
| Pregnancy  | 55      | 63      | 63         | 140         | 140      | 250      |

### Custom targets

```python
from cgmpy.metrics.targets import GlucoseTargets

custom = GlucoseTargets(
    hypo_level2=50,
    hypo_level1=65,
    target_low=70,
    target_high=170,
    hyper_level1=170,
    hyper_level2=240,
    name="Custom",
)
```

## References

Each metric is annotated with the published formula in its docstring.
The primary reference for the consensus cutoffs is:

> Battelino T. *et al.* (2019). Clinical Targets for Continuous Glucose
> Monitoring Data Interpretation: Recommendations From the International
> Consensus on Time in Range. *Diabetes Care*, 42(8), 1593-1603.
> DOI: [10.2337/dci19-0028](https://doi.org/10.2337/dci19-0028)

Additional per-metric references are listed in the
[API reference](../api/metrics.md).

## See also

- [Pregnancy analysis](pregnancy-analysis.md) — dedicated workflows.
- [AGATA integration](agata-integration.md) — cross-validate your numbers.
- [API reference](../api/metrics.md) — function signatures.
