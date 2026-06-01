"""Tests for `cgmpy.utils.date_utils`."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from cgmpy.utils.date_utils import (
    calculate_date_difference,
    format_date_for_display,
    get_date_components,
    get_quarter_dates,
    is_business_day,
    parse_date,
    validate_date_range,
)


class TestParseDate:
    """Tests for the multi-format `parse_date` parser."""

    def test_parse_iso_date_only(self) -> None:
        ts = parse_date("2024-01-01")
        assert isinstance(ts, pd.Timestamp)
        assert ts == pd.Timestamp("2024-01-01")

    def test_parse_dmy_slash_format(self) -> None:
        """Format 1: 07/10/2022 00:00 (day/month/year)."""
        ts = parse_date("07/10/2022 00:00")
        assert ts == pd.Timestamp("2022-10-07 00:00")

    def test_parse_iso_t_format(self) -> None:
        """Format 2: 2023-02-01T01:08:04."""
        ts = parse_date("2023-02-01T01:08:04")
        assert ts == pd.Timestamp("2023-02-01 01:08:04")

    def test_parse_dmy_dash_format(self) -> None:
        """Format 3: 21-03-2023 16:01 (day-month-year)."""
        ts = parse_date("21-03-2023 16:01")
        assert ts == pd.Timestamp("2023-03-21 16:01")

    def test_parse_ymd_space_format(self) -> None:
        """Format 4: 2022-07-24 00:12:00 (year-month-day)."""
        ts = parse_date("2022-07-24 00:12:00")
        assert ts == pd.Timestamp("2022-07-24 00:12:00")

    def test_parse_timestamp_input_returns_same(self) -> None:
        original = pd.Timestamp("2024-06-15 12:30:00")
        assert parse_date(original) == original

    def test_parse_datetime_input_returns_timestamp(self) -> None:
        dt = datetime.datetime(2024, 6, 15, 12, 30, 0)
        result = parse_date(dt)
        assert isinstance(result, pd.Timestamp)
        assert result == pd.Timestamp(dt)

    def test_parse_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not parse"):
            parse_date("not-a-date")

    def test_parse_unsupported_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported data type"):
            parse_date(12345)  # type: ignore[arg-type]


class TestValidateDateRange:
    """Tests for `validate_date_range`."""

    def test_returns_parsed_tuple(self) -> None:
        start, end = validate_date_range("2024-01-01", "2024-12-31")
        assert start == pd.Timestamp("2024-01-01")
        assert end == pd.Timestamp("2024-12-31")

    def test_start_after_end_raises(self) -> None:
        with pytest.raises(ValueError, match="before end date"):
            validate_date_range("2024-12-31", "2024-01-01")

    def test_both_none(self) -> None:
        start, end = validate_date_range(None, None)
        assert start is None
        assert end is None

    def test_only_start_provided(self) -> None:
        start, end = validate_date_range("2024-01-01", None)
        assert start == pd.Timestamp("2024-01-01")
        assert end is None

    def test_only_end_provided(self) -> None:
        start, end = validate_date_range(None, "2024-12-31")
        assert start is None
        assert end == pd.Timestamp("2024-12-31")


class TestFormatDateForDisplay:
    """Tests for `format_date_for_display`."""

    def test_format_timestamp(self) -> None:
        ts = pd.Timestamp("2024-06-15 12:30:00")
        assert format_date_for_display(ts) == "15/06/2024 12:30"

    def test_format_datetime(self) -> None:
        dt = datetime.datetime(2024, 6, 15, 12, 30, 0)
        assert format_date_for_display(dt) == "15/06/2024 12:30"

    def test_format_unsupported_falls_back_to_str(self) -> None:
        """For non-date inputs the function falls back to `str(...)`."""
        assert format_date_for_display("hello") == "hello"  # type: ignore[arg-type]


class TestGetDateComponents:
    """Tests for `get_date_components`."""

    def test_components_present(self) -> None:
        ts = pd.Timestamp("2024-06-15 12:30:45")
        comps = get_date_components(ts)
        assert comps["year"] == 2024
        assert comps["month"] == 6
        assert comps["day"] == 15
        assert comps["hour"] == 12
        assert comps["minute"] == 30
        assert comps["second"] == 45

    def test_weekday_for_known_date(self) -> None:
        # 2024-06-15 is a Saturday → weekday 5
        ts = pd.Timestamp("2024-06-15 00:00:00")
        comps = get_date_components(ts)
        assert comps["weekday"] == 5
        assert comps["weekday_name"] == "Saturday"

    def test_empty_dict_for_invalid_input(self) -> None:
        """Non-date types return an empty dict."""
        assert get_date_components("not-a-date") == {}  # type: ignore[arg-type]


class TestCalculateDateDifference:
    """Tests for `calculate_date_difference`."""

    def _pair(self):
        d1 = pd.Timestamp("2024-01-01 00:00:00")
        d2 = pd.Timestamp("2024-01-02 12:00:00")
        return d1, d2

    def test_days(self) -> None:
        d1, d2 = self._pair()
        assert calculate_date_difference(d1, d2, "days") == pytest.approx(1.5)

    def test_hours(self) -> None:
        d1, d2 = self._pair()
        assert calculate_date_difference(d1, d2, "hours") == pytest.approx(36.0)

    def test_minutes(self) -> None:
        d1, d2 = self._pair()
        assert calculate_date_difference(d1, d2, "minutes") == pytest.approx(36 * 60)

    def test_seconds(self) -> None:
        d1, d2 = self._pair()
        assert calculate_date_difference(d1, d2, "seconds") == pytest.approx(
            36 * 3600
        )

    def test_invalid_unit_raises(self) -> None:
        d1, d2 = self._pair()
        with pytest.raises(ValueError, match="Unsupported unit"):
            calculate_date_difference(d1, d2, "lightyears")

    def test_non_date_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="Timestamp or datetime"):
            calculate_date_difference("a", "b", "days")  # type: ignore[arg-type]

    def test_difference_is_absolute(self) -> None:
        """Order of dates does not change the magnitude of the result."""
        d1, d2 = self._pair()
        assert calculate_date_difference(d2, d1, "hours") == pytest.approx(36.0)


class TestIsBusinessDay:
    """Tests for `is_business_day`."""

    def test_monday(self) -> None:
        # 2024-06-17 is a Monday
        assert is_business_day(pd.Timestamp("2024-06-17")) is True

    def test_friday(self) -> None:
        # 2024-06-21 is a Friday
        assert is_business_day(pd.Timestamp("2024-06-21")) is True

    def test_saturday(self) -> None:
        # 2024-06-15 is a Saturday
        assert is_business_day(pd.Timestamp("2024-06-15")) is False

    def test_sunday(self) -> None:
        # 2024-06-16 is a Sunday
        assert is_business_day(pd.Timestamp("2024-06-16")) is False

    def test_invalid_returns_false(self) -> None:
        """Non-date inputs return False rather than raising."""
        assert is_business_day("not-a-date") is False  # type: ignore[arg-type]


class TestGetQuarterDates:
    """Tests for `get_quarter_dates`."""

    def test_q1(self) -> None:
        start, end = get_quarter_dates(pd.Timestamp("2024-02-10"))
        assert start == pd.Timestamp("2024-01-01")
        assert end == pd.Timestamp("2024-03-31")

    def test_q2(self) -> None:
        start, end = get_quarter_dates(pd.Timestamp("2024-05-10"))
        assert start == pd.Timestamp("2024-04-01")
        assert end == pd.Timestamp("2024-06-30")

    def test_q3(self) -> None:
        start, end = get_quarter_dates(pd.Timestamp("2024-08-10"))
        assert start == pd.Timestamp("2024-07-01")
        assert end == pd.Timestamp("2024-09-30")

    def test_q4(self) -> None:
        start, end = get_quarter_dates(pd.Timestamp("2024-11-10"))
        assert start == pd.Timestamp("2024-10-01")
        assert end == pd.Timestamp("2024-12-31")

    def test_invalid_input_raises(self) -> None:
        with pytest.raises(ValueError, match="Timestamp or datetime"):
            get_quarter_dates("not-a-date")  # type: ignore[arg-type]
