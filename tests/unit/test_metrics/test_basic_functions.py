"""Tests for pure basic metric functions."""

import numpy as np
import pandas as pd
import pytest

from cgmpy.metrics.basic import cv, gmi, mean, median, percentile, sd


def test_mean():
    assert mean(pd.Series([100.0, 110.0, 120.0])) == 110.0


def test_mean_empty_series():
    assert np.isnan(mean(pd.Series([], dtype=float)))


def test_median():
    assert median(pd.Series([100.0, 110.0, 120.0])) == 110.0


def test_median_odd():
    assert median(pd.Series([90.0, 100.0, 110.0])) == 100.0


def test_sd():
    s = pd.Series([100.0, 110.0, 120.0])
    assert sd(s) == pytest.approx(s.std())


def test_sd_constant():
    assert sd(pd.Series([100.0, 100.0, 100.0])) == 0.0


def test_cv():
    s = pd.Series([100.0, 110.0, 120.0])
    expected = (s.std() / s.mean()) * 100
    assert cv(s) == pytest.approx(expected)


def test_cv_zero_mean():
    assert cv(pd.Series([0.0, 0.0, 0.0])) == 0.0


def test_gmi():
    assert gmi(100.0) == pytest.approx(round(3.31 + 0.02392 * 100, 2))


def test_gmi_zero():
    assert gmi(0.0) == pytest.approx(round(3.31, 2))


def test_percentile():
    s = pd.Series(range(0, 100))
    assert percentile(s, 50) == pytest.approx(49.5)


def test_percentile_extremes():
    s = pd.Series([10.0, 20.0, 30.0])
    assert percentile(s, 0) == pytest.approx(10.0)
    assert percentile(s, 100) == pytest.approx(30.0)


def test_backward_compatibility():
    """Pure functions return same values as mixin methods."""
    from cgmpy import GlucoseAnalysis

    ga = GlucoseAnalysis("tests/fixtures/synthetic/sine_24h.csv")
    glucose = ga.data["glucose"]

    assert mean(glucose) == pytest.approx(ga.mean())
    assert median(glucose) == pytest.approx(ga.median())
    assert sd(glucose) == pytest.approx(ga.sd())
    assert cv(glucose) == pytest.approx(ga.cv())
    assert gmi(glucose.mean()) == pytest.approx(ga.gmi())
