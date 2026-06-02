# Synthetic (non-device-bound) fixtures

Plain CSV files with a simple `time,glucose` schema. These are
**device-format agnostic** — they exist for unit tests that exercise
metric math without going through a specialized loader.

## `sine_24h.csv`

A 24-hour, 5-minute-interval dataset (288 readings) generated from:

```
g(t) = 120 + 30 * sin(2π * t / 720)
```

where `t` is minutes from the start time `2024-01-01 00:00:00`. The
period is **720 min = 12 h**, and the 24-hour window contains exactly
**2 full periods** (`288 / (720/5) = 288/144 = 2`), which makes every
metric hand-computable.

### Range

| Property | Value |
|----------|-------|
| Min      | 90.0 mg/dL (at `t = 180, 540, 900, 1260` min) |
| Max      | 150.0 mg/dL (at `t = 0, 360, 720, 1080, 1440` min) |
| Mean     | 120.0 mg/dL (exact, by symmetry of `sin` over an integer number of full periods) |

### Exact metric values

These are the values `cgmpy` will return (assuming default
`GlucoseTargets.standard()` and pandas default sample statistics,
`ddof=1`).

| Metric                          | Expected value       | How to reproduce |
|---------------------------------|----------------------|------------------|
| `n`                             | 288                  | 24h × 12 readings/h |
| `mean`                          | 120.0                | `(1/n) Σ g(t)` — exact by symmetry |
| `sd` (sample, ddof=1)           | 21.2501280996        | `np.std(g, ddof=1)` = `sqrt(450 · 288/287)` |
| `sd` (population, ddof=0)       | 21.2132034356        | `30 / sqrt(2)` (theoretical) |
| `cv`                            | 17.7084401 %         | `sd_sample / mean · 100` |
| `min`                           | 90.0                 | `120 + 30·(-1)` |
| `max`                           | 150.0                | `120 + 30·(+1)` |
| `TIR` (70-180)                  | 100.0 %              | All values in `[90, 150] ⊂ [70, 180]` |
| `TAR_total` (>180)              | 0.0 %                | Max is 150 |
| `TBR_total` (<70)               | 0.0 %                | Min is 90 |
| `GMI`                           | 6.18 %               | `round(3.31 + 0.02392·120, 2)` — see GMI note below |
| `data_completeness` (5-min int) | 100 %                | 288 expected, 288 real |

**GMI note.** CGMPy uses **GMI = round(3.31 + 0.02392·mean, 2)** (Beck
et al. 2019, DOI:10.2337/dc18-1581), not the older eA1c formula
`(mean + 46.7) / 28.7`. For mean = 120 the two differ:

- GMI (this library)  = `round(3.31 + 0.02392·120, 2)` = **6.18 %**
- eA1c (DCCT/ADAG)    = `(120 + 46.7) / 28.7`          = **5.81 %**

Assert on **6.18**.

### SD: population vs. sample

`pandas.Series.std()` defaults to **sample SD** (`ddof=1`, divisor
`n-1`). The "theoretical" SD of `30·sin(2πt/720)` over a full period is
`30/√2 = 21.2132`, but the library returns **21.2501** because of the
`n/(n-1)` Bessel correction. Both numbers are listed above; the
assertion should use the sample value (21.2501) since that is what the
library returns.

### Example assertion (pytest)

```python
import math
import pytest
from cgmpy import GlucoseMetrics

@pytest.fixture
def sine_fixture():
    return GlucoseMetrics(data_source="tests/fixtures/synthetic/sine_24h.csv")


def test_sine_mean(sine_fixture):
    assert sine_fixture.mean() == pytest.approx(120.0, abs=1e-9)


def test_sine_sd(sine_fixture):
    # pandas sample SD with ddof=1
    assert sine_fixture.sd() == pytest.approx(21.2501280996, rel=1e-6)


def test_sine_tir(sine_fixture):
    assert sine_fixture.TIR() == pytest.approx(100.0, abs=1e-9)


def test_sine_gmi(sine_fixture):
    # GMI = round(3.31 + 0.02392 * mean, 2)
    assert sine_fixture.gmi() == pytest.approx(6.18, abs=0.01)
```
