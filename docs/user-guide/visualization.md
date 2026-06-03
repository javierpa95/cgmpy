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

If you want only one plot, use `GlucoseAnalysis`:

```python
from cgmpy import GlucoseAnalysis, GlucoseData

data = GlucoseData("data.csv")
analysis = GlucoseAnalysis(data=data)
analysis.plot_agp()
```

For direct access to individual plot functions, import from the submodules:

```python
from cgmpy.plotting import agp, daily_plots, statistical_plots

# Read data first
from cgmpy import GlucoseData
data = GlucoseData("data.csv").data

agp.plot_agp(data)
daily_plots.day_graph(data)
statistical_plots.histogram(data)
```

### Ambulatory Glucose Profile (AGP)

The **AGP** is the standard one-page report of CGM data. It overlays the
5/25/50/75/95 percentiles of glucose across the 24-hour clock, plus a
target band.

```python
analysis.plot_agp()
analysis.generate_week_agp(combined=True)     # one line per weekday
analysis.generate_week_agp(combined=False)    # one subplot per weekday
```

The plot works in **headless mode** (`matplotlib.use("Agg")`) and is
tested in CI.

### Daily traces

```python
analysis.plot_daily()                    # single day
analysis.plot_overlapping_days()         # overlay all days
analysis.plot_week_boxplots()            # weekly boxplots
analysis.plot_daily_variations()         # confidence bands
```

### Statistical plots

```python
analysis.histogram()                     # glucose histogram
analysis.plot_time_in_range()             # TIR pie + bar chart
analysis.plot_distribution_comparison()   # 2x2 statistical summary
analysis.plot_correlation_matrix()        # time-segment correlation
```

## Customizing

The plot functions use standard matplotlib. Customise by modifying the
returned figure before calling ``plt.show()`` or by setting matplotlib
parameters globally:

```python
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (12, 6), "lines.linewidth": 1.5})
analysis.plot_agp()
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

To save any plot to a file, use ``plt.savefig()`` after the plot call:

```python
analysis.plot_agp()
import matplotlib.pyplot as plt
plt.savefig("agp.png")
```

## See also

- [Computing metrics](computing-metrics.md) — what the plots show.
- [API reference → Plotting](../api/plotting.md) — function signatures.
