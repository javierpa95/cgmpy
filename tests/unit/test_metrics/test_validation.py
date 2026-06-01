"""Tests for cgmpy.metrics.validation."""

import numpy as np
import pandas as pd
import pytest

from cgmpy.metrics.targets import get_targets
from cgmpy.metrics.validation import (
    DEFAULT_ABSOLUTE_HIGH_MG_DL,
    DEFAULT_ABSOLUTE_LOW_MG_DL,
    ValidationReport,
    validate_glucose_range,
)


def _make_df(values):
    return pd.DataFrame({"glucose": pd.array(values, dtype="float64")})


def test_validate_glucose_range_all_valid():
    df = _make_df([80, 100, 120, 140, 160])
    report = validate_glucose_range(df, warn=False)
    assert isinstance(report, ValidationReport)
    assert report.n_total == 5
    assert report.n_valid == 5
    assert report.n_below == 0
    assert report.n_above == 0
    assert report.n_null == 0
    assert report.is_valid is True
    assert report.warnings == []


def test_validate_glucose_range_detects_below():
    df = _make_df([30, 80, 100, 120])
    report = validate_glucose_range(df, warn=False)
    assert report.n_below == 1
    assert report.n_valid == 3
    assert report.is_valid is False
    assert any("below" in w for w in report.warnings)


def test_validate_glucose_range_detects_above():
    df = _make_df([80, 100, 700, 120])
    report = validate_glucose_range(df, warn=False)
    assert report.n_above == 1
    assert report.is_valid is False


def test_validate_glucose_range_detects_nulls():
    df = _make_df([80, None, 120, 140])
    report = validate_glucose_range(df, warn=False)
    assert report.n_null == 1
    assert report.is_valid is False


def test_validate_glucose_range_with_targets_uses_clinical_bounds():
    targets = get_targets("diabetes")
    df = _make_df([targets.very_low - 1, 100, 200])
    report = validate_glucose_range(df, targets=targets, warn=False)
    assert report.n_below == 1


def test_validate_glucose_range_default_thresholds_used_when_no_targets():
    df = _make_df([DEFAULT_ABSOLUTE_LOW_MG_DL + 1, 200])
    report = validate_glucose_range(df, warn=False)
    assert report.n_valid == 2
    assert report.low_threshold == DEFAULT_ABSOLUTE_LOW_MG_DL
    assert report.high_threshold == DEFAULT_ABSOLUTE_HIGH_MG_DL


def test_validate_glucose_range_empty_dataframe():
    df = _make_df([])
    report = validate_glucose_range(df, warn=False)
    assert report.n_total == 0
    assert report.n_valid == 0
    assert np.isnan(report.min_glucose)
    assert np.isnan(report.max_glucose)


def test_validate_glucose_range_missing_column_raises():
    df = pd.DataFrame({"time": [1, 2, 3]})
    with pytest.raises(ValueError, match="glucose"):
        validate_glucose_range(df, warn=False)


def test_to_dict_serializable():
    df = _make_df([80, 100])
    report = validate_glucose_range(df, warn=False)
    d = report.to_dict()
    assert d["n_total"] == 2
    assert d["is_valid"] is True
    assert d["warnings"] == []
