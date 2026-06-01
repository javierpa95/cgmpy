"""Tests for `cgmpy.data.pregnancy_data.PregnancyData`."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cgmpy.data.pregnancy_data import PregnancyData, PregnancyDataHandler

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "data"


@pytest.fixture
def pregnancy_synthetic_df() -> pd.DataFrame:
    """A synthetic 30-week pregnancy trace at 15-min intervals.

    Spans ~30 weeks (210 days) starting 2024-01-01. Roughly 20,160 records.
    """
    start = datetime(2024, 1, 1, 0, 0)
    n_days = 210  # 30 weeks
    n = n_days * 24 * 4  # 15-min intervals
    times = [start + timedelta(minutes=15 * i) for i in range(n)]
    rng = np.random.default_rng(0)
    # Pregnancy-target glucose: tight around 100 mg/dL
    glucose = 100 + 10 * np.sin(np.linspace(0, 60 * np.pi, n)) + rng.normal(0, 4, n)
    return pd.DataFrame({"time": times, "glucose": glucose})


@pytest.fixture
def pregnancy_delivery_date(pregnancy_synthetic_df: pd.DataFrame) -> str:
    """Delivery date that places trimester boundaries inside the data range."""
    # Data ends ~2024-07-29 (210 days). Make delivery be 10 weeks later -> week 30.
    last_time = pregnancy_synthetic_df["time"].iloc[-1]
    delivery = last_time + timedelta(weeks=10)
    return delivery.strftime("%Y-%m-%d")


class TestCalculateDates:
    """Tests for the static `calculate_dates()` method."""

    def test_returns_expected_keys(self) -> None:
        """The dict has the four documented keys."""
        result = PregnancyData.calculate_dates("2024-10-06", week=30, day=0)
        assert isinstance(result, dict)
        for key in (
            "delivery_date",
            "conception_date",
            "first_trimester_end",
            "second_trimester_end",
            "gestation_week_decimal",
        ):
            assert key in result

    def test_conception_is_before_delivery(self) -> None:
        """The conception date precedes the delivery date."""
        result = PregnancyData.calculate_dates("2024-10-06", week=30, day=0)
        assert result["conception_date"] < result["delivery_date"]

    def test_trimester_boundaries_ordered(self) -> None:
        """conception < T1 end < T2 end < delivery."""
        result = PregnancyData.calculate_dates("2024-10-06", week=30, day=0)
        assert result["conception_date"] < result["first_trimester_end"]
        assert result["first_trimester_end"] < result["second_trimester_end"]
        assert result["second_trimester_end"] < result["delivery_date"]

    def test_gestation_week_decimal(self) -> None:
        """30 weeks + 3 days -> 30 + 3/7 weeks decimal."""
        result = PregnancyData.calculate_dates("2024-10-06", week=30, day=3)
        assert result["gestation_week_decimal"] == pytest.approx(30 + 3 / 7)

    def test_gestation_zero_days(self) -> None:
        """Default day=0 gives an integer-valued decimal week."""
        result = PregnancyData.calculate_dates("2024-10-06", week=20)
        assert result["gestation_week_decimal"] == 20.0

    def test_invalid_delivery_date_raises(self) -> None:
        """A non-parseable delivery date raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            PregnancyData.calculate_dates("not-a-date", week=30, day=0)


class TestDecimalToWeeksDays:
    """Tests for the static `decimal_to_weeks_days()` helper."""

    def test_integer_weeks(self) -> None:
        """An integer decimal week gives (week, 0)."""
        assert PregnancyData.decimal_to_weeks_days(30.0) == (30, 0)

    def test_half_week(self) -> None:
        """35.5 -> (35, 4) (35.5 * 7 = 3.5 days, rounded to 4)."""
        weeks, days = PregnancyData.decimal_to_weeks_days(35.5)
        assert weeks == 35
        assert days == 4

    def test_three_sevenths(self) -> None:
        """30 + 3/7 -> (30, 3)."""
        weeks, days = PregnancyData.decimal_to_weeks_days(30 + 3 / 7)
        assert weeks == 30
        assert days == 3

    def test_zero(self) -> None:
        """0.0 -> (0, 0)."""
        assert PregnancyData.decimal_to_weeks_days(0.0) == (0, 0)


class TestPregnancyDataInit:
    """Constructor & date-attribute tests."""

    def test_init_with_synthetic_df(
        self,
        pregnancy_synthetic_df: pd.DataFrame,
        pregnancy_delivery_date: str,
    ) -> None:
        """The constructor accepts a DataFrame source."""
        pd_obj = PregnancyData(
            data_source=pregnancy_synthetic_df,
            delivery_date=pregnancy_delivery_date,
            week=30,
            day=0,
        )
        assert pd_obj is not None
        assert not pd_obj.data.empty

    def test_date_attributes(
        self,
        pregnancy_synthetic_df: pd.DataFrame,
        pregnancy_delivery_date: str,
    ) -> None:
        """The constructor exposes the key gestational dates."""
        pd_obj = PregnancyData(
            data_source=pregnancy_synthetic_df,
            delivery_date=pregnancy_delivery_date,
            week=30,
            day=0,
        )
        assert isinstance(pd_obj.delivery_date, pd.Timestamp)
        assert isinstance(pd_obj.conception_date, pd.Timestamp)
        assert isinstance(pd_obj.first_trimester_end, pd.Timestamp)
        assert isinstance(pd_obj.second_trimester_end, pd.Timestamp)
        assert pd_obj.conception_date < pd_obj.first_trimester_end

    def test_target_type_is_pregnancy(
        self,
        pregnancy_synthetic_df: pd.DataFrame,
        pregnancy_delivery_date: str,
    ) -> None:
        """PregnancyData defaults to the pregnancy target profile."""
        pd_obj = PregnancyData(
            data_source=pregnancy_synthetic_df,
            delivery_date=pregnancy_delivery_date,
            week=30,
            day=0,
        )
        assert pd_obj.target_type == "pregnancy"
        assert pd_obj.targets.name.lower() == "pregnancy"

    def test_data_filtered_to_pregnancy_period(
        self,
        pregnancy_synthetic_df: pd.DataFrame,
        pregnancy_delivery_date: str,
    ) -> None:
        """The main dataframe is filtered to [conception_date, delivery_date]."""
        pd_obj = PregnancyData(
            data_source=pregnancy_synthetic_df,
            delivery_date=pregnancy_delivery_date,
            week=30,
            day=0,
        )
        assert pd_obj.data["time"].min() >= pd_obj.conception_date
        assert pd_obj.data["time"].max() <= pd_obj.delivery_date


class TestGetWeeksDays:
    """Tests for the instance-level `get_weeks_days()` method."""

    def test_round_weeks(
        self,
        pregnancy_synthetic_df: pd.DataFrame,
        pregnancy_delivery_date: str,
    ) -> None:
        """`get_weeks_days()` returns the (weeks, days) tuple given at construction."""
        pd_obj = PregnancyData(
            data_source=pregnancy_synthetic_df,
            delivery_date=pregnancy_delivery_date,
            week=30,
            day=2,
        )
        weeks, days = pd_obj.get_weeks_days()
        assert weeks == 30
        assert days == 2


class TestSplitTrimesters:
    """Tests for the `_split_trimesters()` / `trimesters` attribute."""

    def test_three_trimester_keys(
        self,
        pregnancy_synthetic_df: pd.DataFrame,
        pregnancy_delivery_date: str,
    ) -> None:
        """`trimesters` exposes the three expected DataFrames."""
        pd_obj = PregnancyData(
            data_source=pregnancy_synthetic_df,
            delivery_date=pregnancy_delivery_date,
            week=30,
            day=0,
        )
        assert set(pd_obj.trimesters.keys()) == {
            "first_trimester",
            "second_trimester",
            "third_trimester",
        }
        for df in pd_obj.trimesters.values():
            assert isinstance(df, pd.DataFrame)

    def test_first_trimester_in_first_window(
        self,
        pregnancy_synthetic_df: pd.DataFrame,
        pregnancy_delivery_date: str,
    ) -> None:
        """All first-trimester records lie within [conception, T1 end)."""
        pd_obj = PregnancyData(
            data_source=pregnancy_synthetic_df,
            delivery_date=pregnancy_delivery_date,
            week=30,
            day=0,
        )
        t1 = pd_obj.trimesters["first_trimester"]
        if len(t1) > 0:
            assert t1["time"].min() >= pd_obj.conception_date
            assert t1["time"].max() < pd_obj.first_trimester_end


class TestGetTrimesterData:
    """Tests for the `get_trimester_data()` window helper."""

    def test_window_returns_filtered_df(
        self,
        pregnancy_synthetic_df: pd.DataFrame,
        pregnancy_delivery_date: str,
    ) -> None:
        """`get_trimester_data(start, end)` returns rows in [start, end)."""
        pd_obj = PregnancyData(
            data_source=pregnancy_synthetic_df,
            delivery_date=pregnancy_delivery_date,
            week=30,
            day=0,
        )
        start = pd_obj.conception_date
        end = pd_obj.first_trimester_end
        result = pd_obj.get_trimester_data(start, end)
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert result["time"].min() >= start
            assert result["time"].max() < end


class TestPregnancyDataStr:
    """Tests for the `__str__` representation."""

    def test_str_contains_gestation_info(
        self,
        pregnancy_synthetic_df: pd.DataFrame,
        pregnancy_delivery_date: str,
    ) -> None:
        """The string representation mentions weeks and trimesters."""
        pd_obj = PregnancyData(
            data_source=pregnancy_synthetic_df,
            delivery_date=pregnancy_delivery_date,
            week=30,
            day=0,
        )
        s = str(pd_obj)
        assert isinstance(s, str)
        assert "Pregnancy Data Summary" in s
        assert "Gestation" in s
        assert "Trimester" in s


class TestPregnancyDataAlias:
    """The legacy alias `PregnancyDataHandler` points to the same class."""

    def test_alias_is_pregnancy_data(self) -> None:
        """`PregnancyDataHandler` is an alias of `PregnancyData`."""
        assert PregnancyDataHandler is PregnancyData


class TestPregnancyDataWithCsv:
    """Smoke test using the bundled pregnancy CSV fixture."""

    def test_loads_real_pregnancy_csv(self) -> None:
        """The real pregnancy CSV loads end-to-end."""
        # CSV spans 2022-07 to 2025-01; set a plausible delivery date inside it.
        pd_obj = PregnancyData(
            data_source=str(FIXTURES / "pregnancy.csv"),
            delivery_date="2023-04-01",
            week=40,
            day=0,
        )
        assert len(pd_obj.data) > 0
        assert pd_obj.gestation_week == 40.0
