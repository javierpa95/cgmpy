"""Clinical reference tests for basic glucose metrics.

These tests validate the library's output against hand-computed expected
values for a small synthetic 24-hour CGM dataset. They serve as regression
tests that catch formula regressions on clinically meaningful quantities.
"""

import math

import pandas as pd
import pytest

from cgmpy import GlucoseMetrics


# ---------------------------------------------------------------------------
# Synthetic 24h dataset
# ---------------------------------------------------------------------------
# 24 readings, one per hour, 70 - 250 mg/dL. Mixed hyper/hypo/normo range
# so that every metric is exercised. The exact values are chosen so that
# all hand-computed expected values are exact integers or simple fractions.
SYNTHETIC_GLUCOSE = [
    70, 80, 95, 110, 130, 150, 165, 180, 200, 220, 250, 240,
    210, 180, 150, 130, 115, 105, 100, 95, 90, 85, 80, 75,
]


def _build_synthetic_24h() -> GlucoseMetrics:
    """Build a GlucoseMetrics instance with a known 24h dataset."""
    start = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.date_range(start=start, periods=24, freq="1h")
    df = pd.DataFrame({"time": timestamps, "glucose": SYNTHETIC_GLUCOSE})
    return GlucoseMetrics(data_source=df)


@pytest.fixture(scope="module")
def synthetic_24h() -> GlucoseMetrics:
    return _build_synthetic_24h()


# ---------------------------------------------------------------------------
# Expected values (hand-computed, see comments)
# ---------------------------------------------------------------------------
# Glucose values (n = 24):
#   [70, 80, 95, 110, 130, 150, 165, 180, 200, 220, 250, 240,
#    210, 180, 150, 130, 115, 105, 100, 95, 90, 85, 80, 75]
#
# Mean = 3305 / 24 = 137.70833... mg/dL
#
# Sorted: 70, 75, 80, 80, 85, 90, 95, 95, 100, 105, 110, 115,
#         130, 130, 150, 150, 165, 180, 180, 200, 210, 220, 240, 250
# Median (mean of 12th and 13th sorted values) = (115 + 130) / 2 = 122.5
#
# SD: pandas default uses sample SD (ddof=1, divide by n-1).
# Sum of squared deviations from mean (137.7083...) = 71,348.9583
# Variance = 71348.9583 / 23 = 3102.1286
# SD = sqrt(3102.1286) = 55.6967... mg/dL
#
# CV = SD / Mean * 100 = 55.6967 / 137.7083 * 100 = 40.4454... %
#
# GMI (Glucose Management Indicator) = round(3.31 + 0.02392 * mean, 2)
#   = round(3.31 + 0.02392 * 137.7083, 2)
#   = round(3.31 + 3.2936, 2) = round(6.6036, 2) = 6.60 %
#
# Diabetes targets (default):
#   target_low = 70, target_high = 180
# TIR uses inclusive >= 70 and <= 180, so 70 and 180 are in range.
# Values in [70, 180] (inclusive):
#   70, 80, 95, 110, 130, 150, 165, 180 (positions 0-7)
#   180, 150, 130, 115, 105, 100, 95, 90, 85, 80, 75 (positions 13-23)
#   = 8 + 11 = 19 readings
# TIR = 19 / 24 = 79.1666... %
#
# TAR_total uses strict > target_high (180), so 180 is NOT in TAR.
# Values > 180: 200, 220, 250, 240, 210 = 5 readings
# TAR_total = 5 / 24 = 20.8333... %
#
# TBR_total uses strict < target_low (70), so 70 is NOT in TBR.
# Values < 70: none
# TBR_total = 0 / 24 = 0.0 %


EXPECTED_MEAN = 3305 / 24  # 137.7083...
EXPECTED_MEDIAN = 122.5
EXPECTED_SD = math.sqrt(71348.95833333336 / 23)  # ~55.6967
EXPECTED_CV = EXPECTED_SD / EXPECTED_MEAN * 100  # ~40.4455
EXPECTED_GMI = round(3.31 + 0.02392 * EXPECTED_MEAN, 2)  # 6.60
EXPECTED_TIR_70_180 = 19 / 24 * 100  # 79.1666...
EXPECTED_TAR_TOTAL = 5 / 24 * 100  # 20.8333...
EXPECTED_TBR_TOTAL = 0.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mean_matches_hand_computed(synthetic_24h):
    """Mean glucose should match the hand-computed expected value."""
    assert synthetic_24h.mean() == pytest.approx(EXPECTED_MEAN, rel=1e-3)


def test_median_matches_hand_computed(synthetic_24h):
    """Median glucose should match the hand-computed expected value."""
    assert synthetic_24h.median() == pytest.approx(EXPECTED_MEDIAN, rel=1e-3)


def test_sd_matches_hand_computed(synthetic_24h):
    """Standard deviation should match the hand-computed expected value.

    Library uses sample SD (pandas default ddof=1, divide by n-1).
    """
    assert synthetic_24h.sd() == pytest.approx(EXPECTED_SD, rel=1e-3)


def test_cv_matches_hand_computed(synthetic_24h):
    """Coefficient of variation should match the hand-computed expected value."""
    assert synthetic_24h.cv() == pytest.approx(EXPECTED_CV, rel=1e-3)


def test_gmi_matches_hand_computed(synthetic_24h):
    """GMI (Glucose Management Indicator) should match the formula.

    Library rounds the GMI to 2 decimal places.
    """
    assert synthetic_24h.gmi() == pytest.approx(EXPECTED_GMI, abs=0.01)


def test_tir_70_180_matches_hand_computed(synthetic_24h):
    """TIR (70-180 mg/dL, inclusive on both ends) should be 19/24 on the synthetic dataset."""
    assert synthetic_24h.TIR() == pytest.approx(EXPECTED_TIR_70_180, rel=1e-3)


def test_tar_total_matches_hand_computed(synthetic_24h):
    """TAR_total (> 180 mg/dL, strict) should be 5/24 on the synthetic dataset."""
    assert synthetic_24h.TAR_total() == pytest.approx(EXPECTED_TAR_TOTAL, rel=1e-3)


def test_tbr_total_matches_hand_computed(synthetic_24h):
    """TBR_total (< 70 mg/dL, strict) should be 0% on the synthetic dataset."""
    assert synthetic_24h.TBR_total() == pytest.approx(EXPECTED_TBR_TOTAL, abs=1e-6)


def test_data_completeness(synthetic_24h):
    """24 hourly readings over a 23h span yield >100% by the library's algorithm.

    The library computes expected_data = total_minutes / typical_interval.
    With timestamps 00:00 to 23:00 (23h = 1380 min) and 60-min interval,
    expected_data = 23, but real_data = 24, so percentage = 24/23 * 100 = 104.
    """
    assert synthetic_24h.data_completeness() == 104


def test_n_records_is_24(synthetic_24h):
    """Synthetic dataset must contain exactly 24 records."""
    assert len(synthetic_24h.data) == 24
