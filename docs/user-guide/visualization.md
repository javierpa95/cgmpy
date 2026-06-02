# Visualization

CGMPy ships with three plotters built on **matplotlib** + **seaborn**:
the **AGP** (Ambulatory Glucose Profile), **daily traces**, and
**statistical summaries**.

## Quick start

The simplest way to render every standard plot is via the facade:

```python
from cgmpy import GlucoseAnalysis

analysis = GlucoseAnalysis("data.csv")
analysis.plot_comprehensive_dashboard(save_path="dashboard.png")
```

The dashboard includes:

- AGP (5th, 25th, 50th, 75th, 95th percentiles, target band).
- Daily traces (one row per day).
- Time-in-range bar chart.
- Glucose histogram.

## Modular usage

If you want only one plot, import the relevant plotter:

```python
from cgmpy.data import ModularGlucoseData
from cgmpy.plotting import AGPPlotter, DailyPlotter, StatisticalPlotter

data = ModularGlucoseData("data.csv")
agp = AGPPlotter(data=data)
agp.plot_agp(save_path="agp.png")
```

### Ambulatory Glucose Profile (AGP)

The **AGP** is the standard one-page report of CGM data. It overlays the
5/25/50/75/95 percentiles of glucose across the 24-hour clock, plus a
target band.

```python
agp.plot_agp(
    save_path="agp.png",
    target_low=70,
    target_high=180,
    show_percentiles=(5, 25, 50, 75, 95),
    show_target_band=True,
)
```

The plot works in **headless mode** (`matplotlib.use("Agg")`) and is
tested in CI.

### Daily traces

```python
daily = DailyPlotter(data=data)
daily.plot_daily_traces(
    save_path="daily.png",
    n_rows=4,        # 4 subplots per row
    target_low=70,
    target_high=180,
)
```

### Statistical plots

```python
stats = StatisticalPlotter(data=data)
stats.plot_glucose_histogram(save_path="hist.png")
stats.plot_tir_breakdown(save_path="tir.png")
```

## Customizing

Every plotter accepts a `style` argument (a dict of matplotlib kwargs) so
you can override colors, fonts, and figure size:

```python
agp.plot_agp(
    save_path="agp_brand.png",
    style={
        "figure.figsize": (12, 6),
        "lines.linewidth": 1.5,
        "axes.facecolor": "#FAFAFA",
    },
)
```

## Colorblind-safe palettes

The default palettes are colorblind-safe (Wong 2011). If you supply
your own, please choose a colorblind-safe option such as
[ColorBrewer](https://colorbrewer2.org/) or
[Wong's palette](https://www.nature.com/articles/nmeth.1618).

## Headless rendering (CI / scripts)

When running in a script or CI without a display, set the matplotlib
backend to `Agg` **before** importing CGMPy:

```python
import matplotlib
matplotlib.use("Agg")  # noqa: E402
from cgmpy import GlucoseAnalysis
GlucoseAnalysis("data.csv").plot_comprehensive_dashboard("out.png")
```

## Saving formats

All plotters accept `save_path` with the format inferred from the
extension: `.png`, `.pdf`, `.svg`, `.jpg`. PNG is the default for the
comprehensive dashboard.

## See also

- [Computing metrics](computing-metrics.md) — what the plots show.
- [API reference → Plotting](../api/plotting.md) — function signatures.
- [Contributing → Adding a new plot](https://github.com/javierpa95/cgmpy/blob/main/CONTRIBUTING.md#adding-a-new-plot) — for contributors.
