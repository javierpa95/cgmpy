"""
Date handling utilities module.

This module contains helper functions for processing
and validating dates in different formats.
"""

import datetime

import pandas as pd


def parse_date(
    date_string: str | pd.Timestamp | datetime.datetime,
) -> pd.Timestamp:
    """
    Parses a date from different common formats.

    Args:
        date_string: Date in string, Timestamp or datetime format

    Returns:
        pd.Timestamp: Parsed date

    Raises:
        ValueError: If the date cannot be parsed
    """
    # If it's already a Timestamp or datetime, return it directly
    if isinstance(date_string, pd.Timestamp | datetime.datetime):
        return pd.Timestamp(date_string)

    # If it's a string, try parsing with different formats
    if isinstance(date_string, str):
        formats = [
            "%Y-%m-%d",  # Basic ISO format: 2024-03-27
            "%d/%m/%Y %H:%M",  # Format 1: 07/10/2022 00:00
            "%Y-%m-%dT%H:%M:%S",  # Format 2: 2023-02-01T01:08:04
            "%d-%m-%Y %H:%M",  # Format 3: 21-03-2023 16:01
            "%Y-%m-%d %H:%M:%S",  # Format 4: 2022-07-24 00:12:00
        ]

        for fmt in formats:
            try:
                return pd.to_datetime(date_string, format=fmt)
            except ValueError:
                continue

        raise ValueError(f"Could not parse the date: {date_string}")

    raise ValueError(f"Unsupported data type: {type(date_string)}")


def validate_date_range(
    start_date: str | pd.Timestamp | datetime.datetime | None,
    end_date: str | pd.Timestamp | datetime.datetime | None,
) -> tuple:
    """
    Validates and parses a date range.

    Args:
        start_date: Start date (optional)
        end_date: End date (optional)

    Returns:
        tuple: (start_date_parsed, end_date_parsed)

    Raises:
        ValueError: If the dates are invalid or the range is incorrect
    """
    start_parsed = None
    end_parsed = None

    if start_date is not None:
        start_parsed = parse_date(start_date)

    if end_date is not None:
        end_parsed = parse_date(end_date)

    # Validate that start_date <= end_date if both are present
    if start_parsed is not None and end_parsed is not None and start_parsed > end_parsed:
        raise ValueError("Start date must be before end date")

    return start_parsed, end_parsed


def format_date_for_display(date: pd.Timestamp | datetime.datetime) -> str:
    """
    Formats a date for display.

    Args:
        date: Date to format

    Returns:
        str: Date formatted as string
    """
    if isinstance(date, pd.Timestamp | datetime.datetime):
        return date.strftime("%d/%m/%Y %H:%M")
    return str(date)


def get_date_components(date: pd.Timestamp | datetime.datetime) -> dict:
    """
    Extracts the components of a date.

    Args:
        date: Date to analyze

    Returns:
        dict: Dictionary with year, month, day, hour, minute, second
    """
    if isinstance(date, pd.Timestamp | datetime.datetime):
        return {
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "hour": date.hour,
            "minute": date.minute,
            "second": date.second,
            "weekday": date.weekday(),
            "weekday_name": date.strftime("%A"),
        }
    return {}


def calculate_date_difference(
    date1: pd.Timestamp | datetime.datetime,
    date2: pd.Timestamp | datetime.datetime,
    unit: str = "days",
) -> float:
    """
    Calculates the difference between two dates.

    Args:
        date1: First date
        date2: Second date
        unit: Time unit ('days', 'hours', 'minutes', 'seconds')

    Returns:
        float: Difference in the specified unit
    """
    if not isinstance(date1, pd.Timestamp | datetime.datetime) or not isinstance(
        date2, pd.Timestamp | datetime.datetime
    ):
        raise ValueError("Both dates must be Timestamp or datetime")

    diff = abs(date2 - date1)

    if unit == "days":
        return diff.total_seconds() / (24 * 3600)
    elif unit == "hours":
        return diff.total_seconds() / 3600
    elif unit == "minutes":
        return diff.total_seconds() / 60
    elif unit == "seconds":
        return diff.total_seconds()
    else:
        raise ValueError(f"Unsupported unit: {unit}")


def is_business_day(date: pd.Timestamp | datetime.datetime) -> bool:
    """
    Checks if a date is a business day (Monday to Friday).

    Args:
        date: Date to check

    Returns:
        bool: True if it is a business day, False otherwise
    """
    if isinstance(date, pd.Timestamp | datetime.datetime):
        return date.weekday() < 5  # 0-4 are Monday to Friday
    return False


def get_quarter_dates(date: pd.Timestamp | datetime.datetime) -> tuple:
    """
    Gets the start and end dates of the quarter for a given date.

    Args:
        date: Reference date

    Returns:
        tuple: (quarter_start, quarter_end)
    """
    if isinstance(date, pd.Timestamp | datetime.datetime):
        year = date.year
        quarter = (date.month - 1) // 3 + 1

        start_month = (quarter - 1) * 3 + 1
        end_month = quarter * 3

        start_date = pd.Timestamp(year, start_month, 1)
        if end_month == 12:
            end_date = pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(days=1)
        else:
            end_date = pd.Timestamp(year, end_month + 1, 1) - pd.Timedelta(days=1)

        return start_date, end_date

    raise ValueError("Date must be Timestamp or datetime")
