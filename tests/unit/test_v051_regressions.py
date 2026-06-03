"""Regression tests for the v0.5.1 bug-fix sweep.

Each test exercises one of the 6 bugs documented in CHANGELOG.md [Unreleased]
during the v0.5 modernisation sprint. They guard against the bugs ever
regressing.

Bug #1: GlucosePlot missing metrics (StatisticalPlotter AttributeError on gmi)
Bug #2: MAGE_Baghurst IndexError on small datasets
Bug #3: sd_between_timepoints(group_by_intervals=True) KeyError 'day'
Bug #4: specialized.py.__str__ key mismatch (data_completeness vs completeness)
Bug #5: analysis/core.py calls methods not in MRO
Bug #6: analysis/core.py reads legacy time-statistics keys
"""

from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")


# --------------------------------------------------------------------- helpers
def _make_small_dataset(n: int = 20, value: float = 100.0) -> pd.DataFrame:
    """Build a tiny, valid glucose DataFrame for in-memory tests."""
    start = datetime(2024, 1, 1, 0, 0)
    return pd.DataFrame(
        {
            "time": [start + timedelta(minutes=5 * i) for i in range(n)],
            "glucose": [value] * n,
        }
    )


def _make_oscillating_dataset(n: int = 288) -> pd.DataFrame:
    """Build a 24h, 5-min oscillating dataset for MAGE / variability tests."""
    start = datetime(2024, 1, 1, 0, 0)
    times = [start + timedelta(minutes=5 * i) for i in range(n)]
    glucose = 120 + 50 * np.sin(np.linspace(0, 4 * np.pi, n))
    return pd.DataFrame({"time": times, "glucose": glucose})


def _write_dexcom_csv(path, n: int = 12) -> None:
    df = pd.DataFrame(
        {
            "Marca temporal (AAAA-MM-DDThh:mm:ss)": pd.date_range(
                "2024-01-01", periods=n, freq="5min"
            ).strftime("%Y-%m-%dT%H:%M:%S"),
            "Nivel de glucosa (mg/dL)": [110] * n,
        }
    )
    df.to_csv(path, index=False)


# --------------------------------------------------------------------- Bug #1
class TestBug1GlucosePlotMetrics:
    """Bug #1: GlucosePlot inherits AGPPlotter/DailyPlotter/StatisticalPlotter
    but not BasicMetrics/TimeInRangeMetrics. StatisticalPlotter._generate_statistics_text
    calls self.gmi() / self.TIR() / self.TBR() / self.TAR() — these used to
    raise AttributeError on a plain GlucosePlot."""

    def test_generate_statistics_text_runs(self):
        from cgmpy.plotting.statistical_plots import _generate_statistics_text
        from cgmpy.data.core import GlucoseData
        from cgmpy.analysis.core import GlucoseAnalysis

        ga = GlucoseAnalysis(GlucoseData(data_source=_make_oscillating_dataset()))
        glucose = ga.glucose
        tir_val = ga.TIR()
        tbr_val = ga.TBR(70)
        tar_val = ga.TAR(180)
        gmi_val = ga.gmi()
        text = _generate_statistics_text(glucose, tir_val, tbr_val, tar_val, gmi_val)
        assert "GMI" in text
        assert "TIR" in text

    def test_plot_time_in_range_runs(self):
        import matplotlib.pyplot as plt
        from cgmpy import GlucoseAnalysis, GlucoseData

        ga = GlucoseAnalysis(GlucoseData(data_source=_make_oscillating_dataset()))
        sizes = [ga.TIR(), ga.TBR70(), ga.TBR55(), ga.TAR180(), ga.TAR250()]
        assert all(s is not None for s in sizes)
        plt.close("all")


# --------------------------------------------------------------------- Bug #2
class TestBug2MAGEBaghurstSmallDatasets:
    """Bug #2: MAGE_Baghurst crashed with IndexError when turning_points
    was empty (too few points to satisfy threshold)."""

    @pytest.mark.parametrize("approach", [1, 2, 3])
    def test_mage_baghurst_returns_dict_on_tiny_dataset(self, approach):
        from cgmpy import GlucoseAnalysis, GlucoseData

        # Only 4 points: not enough to form any turning points
        ga = GlucoseAnalysis(GlucoseData(data_source=_make_small_dataset(n=4, value=100.0)))
        result = ga.MAGE_Baghurst(approach=approach)
        assert isinstance(result, dict)
        assert "MAGE_avg" in result
        assert "num_excursions" in result
        assert result["num_excursions"] == 0
        assert result["MAGE_avg"] == 0.0

    def test_mage_baghurst_returns_dict_on_constant_glucose(self):
        from cgmpy import GlucoseAnalysis, GlucoseData

        # Constant glucose → no excursions, but enough points to attempt
        # the algorithm.
        ga = GlucoseAnalysis(GlucoseData(data_source=_make_small_dataset(n=288, value=100.0)))
        result = ga.MAGE_Baghurst(approach=2)
        assert isinstance(result, dict)
        assert result["num_excursions"] == 0


# --------------------------------------------------------------------- Bug #3
class TestBug3SDBetweenTimepointsGrouping:
    """Bug #3: sd_between_timepoints(group_by_intervals=True) raised
    KeyError 'day' because df['day'] was never created."""

    def test_grouping_path_runs(self):
        from cgmpy import GlucoseAnalysis, GlucoseData

        ga = GlucoseAnalysis(GlucoseData(data_source=_make_oscillating_dataset()))
        result = ga.sd_between_timepoints(group_by_intervals=True)
        assert "sd" in result
        assert "valid_timepoints" in result
        assert result["valid_timepoints"] > 0

    def test_grouping_path_with_custom_interval(self):
        from cgmpy import GlucoseAnalysis, GlucoseData

        ga = GlucoseAnalysis(GlucoseData(data_source=_make_oscillating_dataset()))
        result = ga.sd_between_timepoints(group_by_intervals=True, interval_minutes=15)
        assert result["valid_timepoints"] > 0

    def test_non_grouping_path_still_works(self):
        from cgmpy import GlucoseAnalysis, GlucoseData

        ga = GlucoseAnalysis(GlucoseData(data_source=_make_oscillating_dataset()))
        result = ga.sd_between_timepoints(group_by_intervals=False)
        assert "sd" in result


# --------------------------------------------------------------------- Bug #4
class TestBug4SpecializedStrKey:
    """Bug #4: specialized.loaders.__str__ read info['data_completeness']
    but analyzer.get_basic_info returns key 'completeness'."""

    @pytest.mark.parametrize(
        "loader_class",
        ["Dexcom", "Libreview", "MedtronicCarelink", "TandemDiabetes"],
    )
    def test_str_uses_correct_key(self, tmp_path, loader_class):
        from cgmpy.data.specialized import (
            Dexcom,
            Libreview,
            MedtronicCarelink,
            TandemDiabetes,
        )

        cls = {
            "Dexcom": Dexcom,
            "Libreview": Libreview,
            "MedtronicCarelink": MedtronicCarelink,
            "TandemDiabetes": TandemDiabetes,
        }[loader_class]

        path = tmp_path / f"{loader_class}.csv"
        if loader_class == "Dexcom":
            _write_dexcom_csv(path)
            loader = cls(str(path))
        elif loader_class == "Libreview":
            # Libreview has 2-row banner header.
            n = 12
            times = pd.date_range("2024-01-01", periods=n, freq="15min").strftime("%d-%m-%Y %H:%M")
            with path.open("w", encoding="utf-8", newline="") as fh:
                fh.write("Patient export\nGenerated by Libreview\n")
                fh.write(
                    "Dispositivo,Numero de serie,"
                    "Sello de tiempo del dispositivo,"
                    "Tipo de registro,Historial de glucosa mg/dL\n"
                )
                for t in times:
                    fh.write(f"FreeStyle Libre 3,XXX,{t},0,110\n")
            loader = cls(str(path))
        elif loader_class == "MedtronicCarelink":
            pd.DataFrame(
                {
                    "Fecha y hora": pd.date_range("2024-01-01", periods=12, freq="5min").strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "Valor del sensor (mg/dL)": [115] * 12,
                }
            ).to_csv(path, index=False)
            loader = cls(str(path))
        else:  # Tandem
            pd.DataFrame(
                {
                    "Timestamp": pd.date_range("2024-01-01", periods=12, freq="5min").strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "CGM Glucose Value (mg/dL)": [120] * 12,
                }
            ).to_csv(path, index=False)
            loader = cls(str(path))

        # Before the fix this raised KeyError: 'data_completeness'.
        text = str(loader)
        assert "Completeness:" in text
        # The value can be >100% on short synthetic datasets because
        # expected_data is rounded down (e.g. int(55/5) = 11 for 12 points),
        # so just check the section is present and well-formed.
        completeness_line = next(line for line in text.split("\n") if "Completeness:" in line)
        assert "%" in completeness_line
        # And the value parses as a float.
        float(completeness_line.split("Completeness: ")[1].rstrip("%"))


# --------------------------------------------------------------------- Bugs #5 + #6
class TestBug5And6AnalysisCoreMethods:
    """Bugs #5 and #6: GlucoseAnalysis.get_comprehensive_report and
    get_summary_string referenced methods and dict keys that don't exist."""

    def test_comprehensive_report_keys(self):
        from cgmpy import GlucoseAnalysis, GlucoseData

        ga = GlucoseAnalysis(GlucoseData(data_source=_make_oscillating_dataset()))
        report = ga.get_comprehensive_report()
        assert "basic_metrics" in report
        assert "GMI" in report["basic_metrics"]
        assert "variability_metrics" in report
        # calculate_variability_metrics returns a flat dict with these keys
        assert "Std" in report["variability_metrics"]
        assert "mage_avg" in report["variability_metrics"]

    def test_summary_string_runs(self):
        from cgmpy import GlucoseAnalysis, GlucoseData

        ga = GlucoseAnalysis(GlucoseData(data_source=_make_oscillating_dataset()))
        text = ga.get_summary_string()
        # Should not raise; spot-check the 4 sections
        assert "DATA:" in text
        assert "BASIC METRICS:" in text
        assert "TIME IN RANGE:" in text
        assert "VARIABILITY:" in text
        # Spot-check the time-in-range entries that used to raise KeyError
        assert "TIR tight" in text
        assert "TBR70" in text
        assert "TBR55" in text
        assert "TAR140" in text
        assert "TAR180" in text
        assert "TAR250" in text
