"""
Module for handling pregnancy-specific data.
"""

import datetime

import pandas as pd

from .core import ModularGlucoseData


class PregnancyData(ModularGlucoseData):
    """
    Class for managing pregnancy-specific data and dates.
    Inherits from ModularGlucoseData to integrate base loading and processing,
    adding trimester segmentation.
    """

    def __init__(
        self,
        data_source: str | pd.DataFrame,
        delivery_date: str,
        week: int,
        day: int = 0,
        date_col: str = "time",
        glucose_col: str = "glucose",
        delimiter: str | None = None,
        header: int = 0,
        start_date: str | datetime.datetime | None = None,
        end_date: str | datetime.datetime | None = None,
        log: bool = False,
        target_type: str = "pregnancy",
        regularize: bool = False,
        regularize_interval: int = 5,
        regularize_max_gap: int = 30,
    ):
        """
        Initializes pregnancy data.
        """
        # Inicialización base de ModularGlucoseData
        super().__init__(
            data_source=data_source,
            date_col=date_col,
            glucose_col=glucose_col,
            delimiter=delimiter,
            header=header,
            start_date=start_date,
            end_date=end_date,
            log=log,
            target_type=target_type,
            regularize=regularize,
            regularize_interval=regularize_interval,
            regularize_max_gap=regularize_max_gap,
        )

        # Specific pregnancy configuration
        self.gestational_info = self.calculate_dates(delivery_date, week, day)

        # 1. Attribute mapping for easy access
        self.delivery_date = self.gestational_info["delivery_date"]
        self.conception_date = self.gestational_info["conception_date"]
        self.gestation_week = self.gestational_info["gestation_week_decimal"]
        self.first_trimester_end = self.gestational_info["first_trimester_end"]
        self.second_trimester_end = self.gestational_info["second_trimester_end"]

        # 2. Filter the main dataframe to ensure "Overall" metrics only cover the pregnancy period
        mask = (self.data["time"] >= self.conception_date) & (
            self.data["time"] <= self.delivery_date
        )
        self.data = self.data[mask].copy()

        # 3. Recalculate time differences and typical interval after filtering
        self.time_diffs = self.data["time"].diff()
        self.typical_interval = self.analyzer.calculate_typical_interval(self.time_diffs)

        # 4. Creation of dataframes by trimester
        self.trimesters = self._split_trimesters()

    @staticmethod
    def calculate_dates(delivery_date: str, week: int, day: int = 0) -> dict:
        """
        Calculates key pregnancy dates.
        """
        delivery_date_dt = pd.to_datetime(delivery_date)
        if pd.isna(delivery_date_dt):
            raise ValueError("Invalid delivery date")

        gestation_week = week + (day / 7)
        conception_date = delivery_date_dt - pd.Timedelta(weeks=gestation_week)

        if pd.isna(conception_date):
            raise ValueError("Invalid conception date")

        first_trimester_end = conception_date + pd.Timedelta(weeks=13, days=6)
        second_trimester_end = conception_date + pd.Timedelta(weeks=27, days=6)

        return {
            "delivery_date": delivery_date_dt,
            "conception_date": conception_date,
            "first_trimester_end": first_trimester_end,
            "second_trimester_end": second_trimester_end,
            "gestation_week_decimal": gestation_week,
        }

    def _split_trimesters(self) -> dict[str, pd.DataFrame]:
        """
        Splits the main dataframe into three dataframes, one per trimester.
        """
        return {
            "first_trimester": self.get_trimester_data(
                self.conception_date, self.first_trimester_end
            ),
            "second_trimester": self.get_trimester_data(
                self.first_trimester_end, self.second_trimester_end
            ),
            "third_trimester": self.get_trimester_data(
                self.second_trimester_end, self.delivery_date
            ),
        }

    def get_trimester_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Filters the DataFrame for a date range.
        """
        return self.data[(self.data["time"] >= start_date) & (self.data["time"] < end_date)].copy()

    @staticmethod
    def decimal_to_weeks_days(weeks_decimal: float) -> tuple[int, int]:
        """
        Converts decimal weeks to (weeks, days).
        """
        weeks = int(weeks_decimal)
        days = round((weeks_decimal - weeks) * 7)
        return weeks, days

    def get_weeks_days(self) -> tuple[int, int]:
        """
        Returns gestation weeks and days in traditional format.
        """
        return self.decimal_to_weeks_days(self.gestation_week)

    def __str__(self) -> str:
        """
        Overwrites the string representation to include pregnancy-specific information.
        """
        weeks, days = self.get_weeks_days()

        # Trimester definitions for calculation
        periods = [
            ("First Trimester", self.conception_date, self.first_trimester_end, "first_trimester"),
            (
                "Second Trimester",
                self.first_trimester_end,
                self.second_trimester_end,
                "second_trimester",
            ),
            ("Third Trimester", self.second_trimester_end, self.delivery_date, "third_trimester"),
        ]

        trimester_details = []
        for label, start, end, key in periods:
            count = len(self.trimesters[key])

            # Calculate expected records
            duration_mins = (end - start).total_seconds() / 60

            if pd.isna(duration_mins) or self.typical_interval <= 0:
                expected = 0
            else:
                expected = int(duration_mins / self.typical_interval)

            coverage = (count / expected * 100) if expected > 0 else 0

            trimester_details.append(f"  * {label}: {count} records ({coverage:.1f}% coverage)")

        # Base summary from ModularGlucoseData
        base_summary = super().__str__()

        summary = [
            "Pregnancy Data Summary",
            "=====================",
            f"Gestation: {weeks} weeks and {days} days",
            f"Conception Date: {self.conception_date.strftime('%Y-%m-%d')}",
            f"Estimated Delivery: {self.delivery_date.strftime('%Y-%m-%d')}",
            "",
            "Records by Trimester:",
            *trimester_details,
            "",
            "General Data Info:",
            "-----------------",
            base_summary,
        ]

        return "\n".join(summary)


# Alias para mantener compatibilidad si fuera necesario (opcional)
PregnancyDataHandler = PregnancyData
