"""
Módulo de métricas y estadísticas de glucosa.

Este módulo contiene las clases y funciones para calcular:
- Métricas básicas: media, mediana, percentiles, GMI
- Tiempo en rango: TIR, TAR, TBR
- Variabilidad: SD, CV, MAGE, MODD, CONGA
- Métricas avanzadas: GRADE, GRI, M-Value, J-Index
"""

from typing import Any, Dict

import pandas as pd

# Importaciones que están disponibles ahora
from .basic import BasicMetrics
from .targets import GlucoseTargets, get_targets
from .time_in_range import TimeInRangeMetrics
from .variability import VariabilityMetrics

# Importaciones que estarán disponibles cuando se implemente
# from .advanced import AdvancedMetrics


# Clase combinada que integra todas las métricas modulares
class ModularGlucoseMetrics(BasicMetrics, TimeInRangeMetrics, VariabilityMetrics):
    """
    Clase que combina todas las métricas modulares.

    Esta clase permite usar las métricas de forma modular mientras
    mantiene compatibilidad con la interfaz existente.
    """

    def all(self) -> Dict[str, Any]:
        """
        Calculates all available glucose metrics with optional progress logging.
        """
        import time

        start_time = time.time()
        do_log = getattr(self, "log", False)

        try:
            if do_log:
                print("\n[Metrics] Starting complete analysis...")

            all_metrics = {}

            # 1. BASIC METRICS
            if do_log:
                print("  -> Calculating basic metrics (Mean, GMI, CV)...", end="", flush=True)
            s = time.time()
            try:
                basic_metrics = BasicMetrics.calculate_all_metrics(self)
                all_metrics["basic"] = basic_metrics
                if do_log:
                    print(f" Done ({time.time() - s:.2f}s)")
            except Exception as e:
                all_metrics["basic"] = {"error": f"Error: {str(e)}"}

            # 2. TIME IN RANGE
            if do_log:
                print("  -> Calculating Time in Range (TIR, TAR, TBR)...", end="", flush=True)
            s = time.time()
            try:
                time_metrics = self.time_range_summary()
                all_metrics["time_in_range"] = time_metrics
                if do_log:
                    print(f" Done ({time.time() - s:.2f}s)")
            except Exception as e:
                all_metrics["time_in_range"] = {"error": f"Error: {str(e)}"}

            # 3. VARIABILITY
            if do_log:
                print("  -> Calculating variability metrics (This may take a while)...")
            var_start = time.time()
            try:
                # SD Metrics
                if do_log:
                    print("     - Standard Deviations...", end="", flush=True)
                s = time.time()
                sd_metrics = {
                    "sd_total": self.sd_total(),
                    "sd_within_day": self.sd_within_day(),
                    "sd_between_timepoints": self.sd_between_timepoints(),
                    "sd_segments": {
                        "noche": self.sd_within_day_segment("00:00", 8),
                        "dia": self.sd_within_day_segment("08:00", 8),
                        "tarde": self.sd_within_day_segment("16:00", 8),
                    },
                    "sd_within_series": {
                        "1h": self.sd_within_series(hours=1),
                        "6h": self.sd_within_series(hours=6),
                        "24h": self.sd_within_series(hours=24),
                    },
                    "sd_daily_mean": self.sd_daily_mean(),
                    "sd_same_timepoint": self.sd_same_timepoint(),
                    "sd_same_timepoint_adjusted": self.sd_same_timepoint_adjusted(),
                    "sd_interaction": self.sd_interaction(),
                }
                if do_log:
                    print(f" Done ({time.time() - s:.2f}s)")

                # CV Metrics
                if do_log:
                    print("     - Coefficient of Variation...", end="", flush=True)
                s = time.time()
                cv_metrics = self.calculate_all_cv_metrics()
                if do_log:
                    print(f" Done ({time.time() - s:.2f}s)")

                # MAGE
                if do_log:
                    print("     - MAGE (Baghurst & Simple)...", end="", flush=True)
                s = time.time()
                try:
                    mage_metrics = self.MAGE_Baghurst()
                    excursion_metrics = {
                        "mage_baghurst": mage_metrics,
                        "mage_simple": self.MAGE(),
                    }
                except Exception as e:
                    excursion_metrics = {"error": str(e)}
                if do_log:
                    print(f" Done ({time.time() - s:.2f}s)")

                # Other Variability
                if do_log:
                    print("     - MODD, CONGA, Lability Index...", end="", flush=True)
                s = time.time()
                variability_metrics = {
                    "modd": self.MODD(),
                    "conga": {
                        "1h": self.CONGA(hours=1),
                        "2h": self.CONGA(hours=2),
                        "4h": self.CONGA(hours=4),
                        "6h": self.CONGA(hours=6),
                        "24h": self.CONGA(hours=24),
                    },
                    "lability_index": self.Lability_index(),
                }
                if do_log:
                    print(f" Done ({time.time() - s:.2f}s)")

                # Quality Metrics
                if do_log:
                    print("     - Quality Indices (GRI, HBGI, LBGI, GRADE)...", end="", flush=True)
                s = time.time()
                quality_metrics = {
                    "m_value": self.M_Value(),
                    "j_index": self.j_index(),
                    "grade": self.GRADE(),
                    "lbgi": self.LBGI(),
                    "hbgi": self.HBGI(),
                    "gri": self.GRI(),
                    "gri_pregnancy": self.GRI(pregnancy=True),
                    "adrr": self.ADRR(),
                }
                if do_log:
                    print(f" Done ({time.time() - s:.2f}s)")

                all_metrics["variability"] = {
                    "sd_metrics": sd_metrics,
                    "cv_metrics": cv_metrics,
                    "excursion_metrics": excursion_metrics,
                    "variability_metrics": variability_metrics,
                    "quality_metrics": quality_metrics,
                }
                if do_log:
                    print(f"  -> Variability total: {time.time() - var_start:.2f}s")

            except Exception as e:
                all_metrics["variability"] = {"error": str(e)}

            # 4. SUMMARY
            summary = {
                "total_metrics": len(all_metrics),
                "modules": list(all_metrics.keys()),
                "calculation_timestamp": pd.Timestamp.now().isoformat(),
                "data_summary": {
                    "total_readings": len(self.data),
                    "date_range": {
                        "start": self.data["time"].min().isoformat(),
                        "end": self.data["time"].max().isoformat(),
                    },
                    "data_completeness": self.data_completeness(),
                },
            }
            all_metrics["summary"] = summary

            if do_log:
                print(f"[Metrics] Analysis completed in {time.time() - start_time:.2f}s\n")

            return all_metrics

        except Exception as e:
            return {"error": f"General error: {str(e)}"}

    def all_simplified(self) -> dict:
        """
        Versión simplificada de all() que devuelve solo los valores principales.
        Calcula únicamente las métricas necesarias para mejorar el rendimiento.

        Returns:
            dict: Diccionario con métricas principales en formato plano
        """
        try:
            simplified = {}

            if len(self.data) == 0:
                print("Advertencia: No hay datos para calcular métricas.")
                return {
                    "DataCompleteness": 0,
                    "GMI": None, "Mean": None, "Median": None, "SD": None, "CV": None,
                    "TIR": 0, "TIR_tight": 0, "MAGE": None, "GRI": None
                }

            print("Calculando métricas simplificadas...")


            # 1. Basic main metrics
            basic = BasicMetrics.calculate_all_metrics(self)
            simplified.update(
                {
                    "DataCompleteness": self.data_completeness(),
                    "GMI": basic.get("GMI"),
                    "Mean": basic.get("Mean"),
                    "Median": basic.get("Median"),
                    "SD": basic.get("Std"),
                    "CV": basic.get("CV"),
                }
            )

            # 2. Main Time In Range (TIR)
            targets = self.current_targets
            is_pregnancy = targets.name.lower() == "pregnancy"

            simplified.update(
                {
                    "TIR": self.TIR(),
                    "TIR_tight": self.TIR_tight(),
                }
            )

            # Assign keys and values based on target type
            if is_pregnancy:
                simplified.update(
                    {
                        "TAR140": self.TAR_total(),
                        "TAR250": self.TAR_L2(),
                        "TBR63": self.TBR_total(),
                        "TBR55": self.TBR_L2(),
                    }
                )
            else:
                simplified.update(
                    {
                        "TAR180": self.TAR_total(),
                        "TAR250": self.TAR_L2(),
                        "TBR70": self.TBR_total(),
                        "TBR54": self.TBR_L2(),
                    }
                )

            # 3. Main Variability & Risk
            # Solo calculamos lo que se va a mostrar
            simplified.update(
                {
                    "SDw": self.sd_within_day().get("sd"),
                    "SDdm": self.sd_daily_mean().get("sd"),
                    "MAGE": self.MAGE(),
                    "MODD": self.MODD().get("value"),
                    "CONGA4": self.CONGA(hours=4).get("value"),
                    "LBGI": self.LBGI(),
                    "HBGI": self.HBGI(),
                    "ADRR": self.ADRR().get("adrr"),
                    "GRI": self.GRI(pregnancy=is_pregnancy).get("GRI"),
                    "J-Index": self.j_index(),
                }
            )

            return simplified

        except Exception as e:
            return {"error": f"Error en métricas simplificadas: {str(e)}"}

    pass


__all__ = [
    "BasicMetrics",
    "TimeInRangeMetrics",
    "VariabilityMetrics",
    "ModularGlucoseMetrics",
]
