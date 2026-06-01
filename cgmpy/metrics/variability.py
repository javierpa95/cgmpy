"""
Variability metrics module for glucose data.

This module contains metrics related to glycemic variability:
- MAGE (Mean Amplitude of Glycemic Excursions)
- MODD (Mean Of Daily Differences)
- CONGA (Continuous Overlapping Net Glycemic Action)
- Lability Index
- Specialized standard deviation metrics
"""

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


class VariabilityMetrics:
    """
    Class for glucose variability metrics.

    This class should be used as a mixin with GlucoseData.
    """

    if TYPE_CHECKING:
        data: pd.DataFrame
        typical_interval: float

        def sd(self) -> float: ...
        def mean(self) -> float: ...
        def median(self) -> float: ...
        def cv(self) -> float: ...
        def gmi(self) -> float: ...
        def data_completeness(self) -> float: ...
        def TBR(self, threshold: float) -> float: ...
        def TAR(self, threshold: float) -> float: ...
        def calculate_time_in_range(self, low: float, high: float) -> float: ...
        def TIR(self) -> float: ...
        def TIR_tight(self) -> float: ...
        def TIR_pregnancy(self) -> float: ...
        def TAR180(self) -> float: ...
        def TAR250(self) -> float: ...
        def TAR140(self) -> float: ...
        def TBR70(self) -> float: ...
        def TBR63(self) -> float: ...
        def TBR55(self) -> float: ...

    # MEDIDAS DE DESVIACIÓN

    def sd_total(self) -> dict:
        """
        Calculates total standard deviation (SDT) and global mean.
        Returns: {'sd': float, 'mean': float}
        """
        return {"sd": self.sd(), "mean": self.mean()}

    def sd_within_day(self, min_count_threshold: float = 0.5) -> dict:
        """
        Calculates within-day standard deviation (SDw).

        This metric reflects variability within each day, averaged across
        all available days.

        Optimized version for large datasets.

        :param min_count_threshold: Threshold to consider a day valid
                               (proportion of count median).
        :return: Dictionary with SDw value and related statistics.
        """
        # Crear copia eficiente con solo las columnas necesarias
        df = self.data[["time", "glucose"]].copy()

        # Extraer fecha de forma vectorizada
        df["date"] = df["time"].dt.date

        # Calcular estadísticas por día de forma vectorizada
        daily_stats = df.groupby("date").agg({"glucose": ["std", "mean", "count"]})

        # Calcular umbral para filtrar días con pocos datos
        median_count = daily_stats[("glucose", "count")].median()
        threshold = median_count * min_count_threshold

        # Guardar información sobre todos los días
        all_days_sds = daily_stats[("glucose", "std")].to_dict()
        all_days_means = daily_stats[("glucose", "mean")].to_dict()
        all_days_counts = daily_stats[("glucose", "count")].to_dict()

        # Filtrar días con suficientes datos
        valid_days = daily_stats[daily_stats[("glucose", "count")] >= threshold]

        if valid_days.empty:
            return {
                "sd": 0.0,
                "mean": 0.0,
                "valid_days": 0,
                "total_days": len(daily_stats),
                "daily_sds": {},
                "daily_means": {},
                "daily_counts": {},
                "all_days_sds": all_days_sds,
                "all_days_means": all_days_means,
                "all_days_counts": all_days_counts,
                "threshold": threshold,
            }

        # Calcular SDw (promedio de las desviaciones estándar diarias)
        sd_value = valid_days[("glucose", "std")].mean()
        mean_value = valid_days[("glucose", "mean")].mean()

        # Preparar resultado con estadísticas adicionales
        result = {
            "sd": sd_value,
            "mean": mean_value,
            "valid_days": len(valid_days),
            "total_days": len(daily_stats),
            "daily_sds": valid_days[("glucose", "std")].to_dict(),
            "daily_means": valid_days[("glucose", "mean")].to_dict(),
            "daily_counts": valid_days[("glucose", "count")].to_dict(),
            "all_days_sds": all_days_sds,
            "all_days_means": all_days_means,
            "all_days_counts": all_days_counts,
            "threshold": threshold,
        }

        return result

    def sdw(self, min_count_threshold: float = 0.5) -> float:
        """
        Calculates within-day standard deviation (SDw).
        This is a simplified method returning only the SD value.

        :param min_count_threshold: Threshold to consider a day valid
                                   (proportion of count median). Default 0.5 (50%).
        :return: SDw value (float)
        """
        return self.sd_within_day(min_count_threshold)["sd"]

    def sd_within_day_segment(self, start_time: str, duration_hours: int) -> dict[str, float]:
        """
        Calculates within-day standard deviation for a specific day segment.
        For each day, calculates the SD of the specified time segment and then
        averages these daily SDs.

        :param start_time: Start time in "HH:MM" format
        :param duration_hours: Duration of the segment in hours
        :return: Average of segment SDs for each day and average of daily means

        Example:
            sd_within_day_segment("00:00", 8)  # Average SD of night segment (00:00-08:00)
            sd_within_day_segment("08:00", 8)  # Average SD of day segment (08:00-16:00)
        """
        # Obtener los datos del segmento
        segment_data = self._get_segment_data(start_time, duration_hours)

        if segment_data.empty:
            return {"sd": 0.0, "mean": 0.0}

        # Calcular SD y media para el segmento de cada día
        daily_segment_stats = segment_data.groupby(segment_data["time"].dt.date)["glucose"].agg(
            ["std", "mean"]
        )

        return {
            "sd": daily_segment_stats["std"].mean() if not daily_segment_stats.empty else 0.0,
            "mean": daily_segment_stats["mean"].mean() if not daily_segment_stats.empty else 0.0,
        }

    def sd_between_timepoints(
        self,
        min_count_threshold: float = 0.5,
        filter_outliers: bool = True,
        agrupar_por_intervalos: bool = False,
        intervalo_minutos: int = 5,
    ) -> dict[str, Any]:
        """
        Calculates standard deviation between timepoints (SDhh:mm).
        Calculates the mean of a timestamp and then the standard deviation of those means.

        This metric measures the variability of the glucose pattern throughout the day.

        Optimized version for large datasets.

        :param min_count_threshold: Threshold to consider a timestamp valid
                               (proportion of count median). Default 0.5 (50%).
        :param filter_outliers: If True, filters timestamps with few data points.
        :param agrupar_por_intervalos: If True, groups data into regular time intervals.
        :param intervalo_minutos: Interval size in minutes for grouping (default 5 min).
        :return: Dictionary with SDhh:mm value and related statistics.
        """
        # Crear una copia de los datos con solo las columnas necesarias
        df = self.data[["time", "glucose"]].copy()

        # Extraer solo hora y minuto para reducir la carga computacional
        df["hour_min"] = df["time"].apply(lambda x: x.hour * 60 + x.minute)

        if agrupar_por_intervalos:
            # Agrupar por intervalos de tiempo para reducir la cantidad de puntos temporales
            df["interval"] = (df["hour_min"] // intervalo_minutos) * intervalo_minutos
            # Usar groupby con transform más eficiente que apply para grandes datasets
            grouped = df.groupby(["day", "interval"])
            df_agg = grouped.agg({"glucose": "mean"}).reset_index()

            # Calcular la hora y minuto a partir del intervalo para el resultado final
            df_agg["hour"] = df_agg["interval"] // 60
            df_agg["minute"] = df_agg["interval"] % 60

            # Usar estadísticas descriptivas vectorizadas
            timepoint_means = df_agg.groupby(["hour", "minute"])["glucose"].mean()
            df_final = pd.DataFrame({"mean": timepoint_means})

            # Contar número de días con datos para cada punto temporal
            timepoint_counts = df_agg.groupby(["hour", "minute"]).size()
            df_final["count"] = timepoint_counts
        else:
            # Extraer características temporales vectorizadamente
            df["hour"] = df["time"].dt.hour
            df["minute"] = df["time"].dt.minute
            df["day"] = df["time"].dt.date

            # Calcular estadísticas por hora:minuto de forma vectorizada
            grouped = df.groupby(["hour", "minute", "day"])
            daily_means = grouped["glucose"].mean().reset_index()

            # Agrupar nuevamente para obtener las medias por punto temporal
            timepoint_stats = daily_means.groupby(["hour", "minute"])

            # Calcular estadísticas de forma vectorizada
            df_final = pd.DataFrame(
                {
                    "mean": timepoint_stats["glucose"].mean(),
                    "count": timepoint_stats.size(),
                }
            )

        # Filtrar puntos con pocos datos si se solicita
        if filter_outliers:
            median_count = df_final["count"].median()
            threshold = median_count * min_count_threshold
            valid_timepoints = df_final[df_final["count"] >= threshold]
        else:
            valid_timepoints = df_final

        # Calcular SDhh:mm (la desviación estándar del patrón promedio)
        sd_value = valid_timepoints["mean"].std()
        mean_value = valid_timepoints["mean"].mean()

        # Crear el resultado como diccionario
        result = {
            "sd": sd_value,
            "mean": mean_value,
            "valid_timepoints": len(valid_timepoints),
            "total_timepoints": len(df_final),
            "median_count": df_final["count"].median(),
            "min_count": df_final["count"].min(),
            "max_count": df_final["count"].max(),
        }

        return result

    def sd_between_timepoints_segment(self, start_time: str, duration_hours: int) -> dict:
        """
        Calculates SDhh:mm for a specific day segment.

        :param start_time: Start time in "HH:MM" format
        :param duration_hours: Duration of the segment in hours
        :return: SD of the average pattern by time of day in the segment
        """
        # Filtrar el segmento primero
        segment_data = self._get_segment_data(start_time, duration_hours)

        # Agrupar por marca de tiempo "HH:MM" dentro del segmento
        time_avg = segment_data.groupby(segment_data["time"].dt.strftime("%H:%M"))["glucose"].mean()
        return {
            "sd": time_avg.std() if not time_avg.empty else 0.0,
            "mean": time_avg.mean() if not time_avg.empty else 0.0,
        }

    def sd_within_series(self, hours: int = 1) -> dict:
        """
        Calculates SDws and mean of time series.

        The fewer hours, the smaller the SDws value because
        it gives less time for glucose to vary.

        Optimized version for large datasets.

        :param hours: Window size in hours
        :return: Dictionary with average SD and mean of time series
        """
        # Crear copia eficiente con solo las columnas necesarias
        df = self.data[["time", "glucose"]].copy()

        # Asegurar que los datos estén ordenados por tiempo
        df = df.sort_values("time")

        # Convertir horas a nanosegundos para la ventana de tiempo
        # window_ns = pd.Timedelta(hours=hours).value

        # Matriz para almacenar resultados
        sd_values = []
        mean_values = []

        # Tomar muestras a intervalos regulares para reducir la carga computacional
        # Ajusta el step según el tamaño de tu dataset (mayor step = más rápido pero menos preciso)
        step = max(1, len(df) // 1000)  # Limitar a ~1000 ventanas como máximo

        for i in range(0, len(df), step):
            start_time = df.iloc[i]["time"]
            end_time = start_time + pd.Timedelta(hours=hours)

            # Filtrar los datos en la ventana de tiempo actual
            window_data = df[(df["time"] >= start_time) & (df["time"] < end_time)]

            # Solo calcular estadísticas si hay suficientes puntos en la ventana
            if len(window_data) > 2:  # Necesitamos al menos 3 puntos para una buena estimación
                sd_values.append(window_data["glucose"].std())
                mean_values.append(window_data["glucose"].mean())

        # Calcular promedios
        result = {
            "sd": np.mean(sd_values) if sd_values else 0.0,
            "mean": np.mean(mean_values) if mean_values else 0.0,
            "windows_analyzed": len(sd_values),
        }

        return result

    def sd_daily_mean(self, min_count_threshold: float = 0.5) -> dict:
        """
        Calculates standard deviation of daily means (SDdm).

        This metric reflects variability between different days.

        Optimized version for large datasets.

        :param min_count_threshold: Threshold to consider a day valid
                               (proportion of count median).
        :return: Dictionary with SDdm value and related statistics.
        """
        # Crear copia eficiente con solo las columnas necesarias
        df = self.data[["time", "glucose"]].copy()

        # Extraer fecha de forma vectorizada
        df["date"] = df["time"].dt.date

        # Calcular estadísticas por día de forma vectorizada
        daily_stats = df.groupby("date").agg({"glucose": ["mean", "count"]})

        # Calcular umbral para filtrar días con pocos datos
        median_count = daily_stats[("glucose", "count")].median()
        threshold = median_count * min_count_threshold

        # Filtrar días con suficientes datos
        valid_days = daily_stats[daily_stats[("glucose", "count")] >= threshold]

        if valid_days.empty:
            return {"sd": 0.0, "mean": 0.0}

        # Calcular SDdm (SD de las medias diarias)
        sd_value = valid_days[("glucose", "mean")].std()
        mean_value = valid_days[("glucose", "mean")].mean()

        # Preparar resultado con estadísticas adicionales
        result = {
            "sd": sd_value,
            "mean": mean_value,
            "valid_days": len(valid_days),
            "total_days": len(daily_stats),
            "daily_means": valid_days[("glucose", "mean")].to_dict(),
            "daily_counts": valid_days[("glucose", "count")].to_dict(),
        }

        return result

    def sd_same_timepoint(
        self,
        min_count_threshold: float = 0.5,
        filter_outliers: bool = True,
        agrupar_por_intervalos: bool = False,
        intervalo_minutos: int = 5,
    ) -> dict:
        """
        Calculates between-day standard deviation for each timepoint (SDbhh:mm).

        This function measures glucose variability for each specific timepoint
        across different days, reflecting day-to-day consistency.

        Optimized version for large datasets.

        :param min_count_threshold: Threshold to consider a timestamp valid
                               (proportion of count median).
        :param filter_outliers: If True, filters timestamps with few data points.
        :param agrupar_por_intervalos: If True, groups data into regular intervals.
        :param intervalo_minutos: Interval size in minutes (default 5 min).
        :return: Dictionary with SDbhh:mm value and related statistics.
        """
        # Crear copia eficiente con solo las columnas necesarias
        df = self.data[["time", "glucose"]].copy()

        # Extraer características temporales vectorizadamente
        df["hour"] = df["time"].dt.hour
        df["minute"] = df["time"].dt.minute
        df["day"] = df["time"].dt.date

        if agrupar_por_intervalos:
            # Agrupar por intervalos de tiempo
            minutes_of_day = df["hour"] * 60 + df["minute"]
            df["interval"] = (minutes_of_day // intervalo_minutos) * intervalo_minutos
            df["hour"] = df["interval"] // 60
            df["minute"] = df["interval"] % 60

        # Crear clave de tiempo para agrupación
        df["time_key"] = (
            df["hour"].astype(str).str.zfill(2) + ":" + df["minute"].astype(str).str.zfill(2)
        )

        # Agrupar por día y punto temporal
        grouped = df.groupby(["day", "time_key"])

        # Calcular promedios por día y punto temporal
        daily_means = grouped["glucose"].mean().reset_index()

        # Agrupar por punto temporal y calcular estadísticas
        timepoint_stats = daily_means.groupby("time_key")

        # Calcular estadísticas por punto temporal de forma vectorizada
        sd_por_marca = timepoint_stats["glucose"].std()
        valores_por_marca = timepoint_stats["glucose"].mean()
        conteo_por_marca = timepoint_stats["glucose"].count()

        # Calcular umbral para filtrar puntos con pocos datos
        median_count = conteo_por_marca.median()
        threshold = median_count * min_count_threshold

        # Guardar el total de puntos temporales antes del filtrado
        total_timepoints = len(sd_por_marca)

        # Filtrar puntos temporales con suficientes datos si se solicita
        if filter_outliers:
            valid_mask = conteo_por_marca >= threshold
            sd_por_marca = sd_por_marca[valid_mask]
            valores_por_marca = valores_por_marca[valid_mask]
            conteo_por_marca = conteo_por_marca[valid_mask]

        if len(sd_por_marca) == 0:
            return {
                "sd": 0.0,
                "mean": 0.0,
                "threshold": threshold,
                "total_timepoints": total_timepoints,
            }

        # Calcular SDbhh:mm (promedio ponderado de las desviaciones estándar)
        weights = conteo_por_marca / conteo_por_marca.sum()
        sd_value = (sd_por_marca * weights).sum()
        mean_value = valores_por_marca.mean()

        # Preparar resultado con estadísticas adicionales
        result = {
            "sd": sd_value,
            "mean": mean_value,
            "sd_por_marca": sd_por_marca.to_dict(),
            "valores_por_marca": valores_por_marca.to_dict(),
            "conteo_por_marca": conteo_por_marca.to_dict(),
            "threshold": threshold,
            "total_timepoints": total_timepoints,
        }

        return result

    def sd_same_timepoint_adjusted(self) -> dict:
        """
        Calculates between-day SD for each timepoint, adjusted for changes in daily means.

        The process is:
        1. Adjust glucose values: Adjusted_Glucose = Glucose - Daily_Mean + Total_Mean
        2. Calculate between-day SD for each timepoint using adjusted values
        3. Average the resulting SDs

        Returns:
            dict: Between-day SD adjusted for daily means
        """
        # Calcular la media total (Grand Mean)
        grand_mean = self.data["glucose"].mean()

        # Calcular medias diarias
        daily_means = self.data.groupby(self.data["time"].dt.date)["glucose"].transform("mean")

        # Ajustar valores de glucosa
        adjusted_glucose = self.data["glucose"] - daily_means + grand_mean

        # Crear DataFrame con valores ajustados
        adjusted_data = self.data.copy()
        adjusted_data["glucose"] = adjusted_glucose

        # Agrupar por tiempo específico del día (HH:MM)
        time_groups = adjusted_data.groupby(adjusted_data["time"].dt.strftime("%H:%M"))

        # Calcular SD para cada tiempo específico usando valores ajustados
        time_sds = time_groups["glucose"].std()

        # Promediar todas las SD
        return {
            "sd": time_sds.mean() if not time_sds.empty else 0.0,
            "mean": grand_mean if not time_sds.empty else 0.0,
        }

    def sd_interaction(self) -> dict:
        """
        Calculates standard deviation of interaction (SDI).

        SDI quantifies daily variability in the glycemic pattern,
        considering interactions between time of day and specific day.

        Optimized version for large datasets.

        :return: Dictionary with SDI value and related statistics.
        """
        # Crear copia eficiente con solo las columnas necesarias
        df = self.data[["time", "glucose"]].copy()

        # Extraer características temporales de forma vectorizada
        df["hour"] = df["time"].dt.hour
        df["minute"] = df["time"].dt.minute
        df["day"] = df["time"].dt.date
        df["time_key"] = (
            df["hour"].astype(str).str.zfill(2) + ":" + df["minute"].astype(str).str.zfill(2)
        )

        # Calcular valores necesarios para la fórmula de SDI

        # 1. Calcular la media global
        global_mean = df["glucose"].mean()

        # 2. Calcular las medias diarias
        daily_means = df.groupby("day")["glucose"].mean()

        # 3. Calcular las medias por punto temporal
        timepoint_means = df.groupby("time_key")["glucose"].mean()

        # 4. Calcular las desviaciones para cada observación
        # Usamos merge para operaciones vectorizadas eficientes
        df_temp = df.copy()

        # Convertir day_mean y timepoint_mean a DataFrames para merge
        day_mean_df = pd.DataFrame(daily_means).reset_index()
        day_mean_df.columns = ["day", "day_mean"]

        timepoint_mean_df = pd.DataFrame(timepoint_means).reset_index()
        timepoint_mean_df.columns = ["time_key", "timepoint_mean"]

        # Hacer merge de forma eficiente
        df_temp = df_temp.merge(day_mean_df, on="day")
        df_temp = df_temp.merge(timepoint_mean_df, on="time_key")

        # Calcular la interacción para cada punto
        df_temp["expected"] = (
            global_mean
            + (df_temp["day_mean"] - global_mean)
            + (df_temp["timepoint_mean"] - global_mean)
        )
        df_temp["interaction"] = df_temp["glucose"] - df_temp["expected"]

        # Calcular SDI (desviación estándar de la interacción)
        sdi = df_temp["interaction"].std()

        return {"sd": sdi, "mean": global_mean}

    def sd_segment(self, start_time: str, duration_hours: int) -> dict:
        """
        Calculates standard deviation within a specific segment (SDws).

        Useful for analyzing segments like night, day, afternoon, etc.

        :param start_time: Start time in "HH:MM".
        :param duration_hours: Duration of the segment in hours.
        :return: Standard deviation of readings within the segment.
        """
        from datetime import datetime, time, timedelta

        # Convertir start_time a objeto time
        start_hour, start_minute = map(int, start_time.split(":"))
        start = time(hour=start_hour, minute=start_minute)

        # Calcular la hora de fin (considerando cruce de medianoche)
        base = datetime(2000, 1, 1, start.hour, start.minute)
        end_dt = base + timedelta(hours=duration_hours)
        end = end_dt.time()

        # Filtrar datos según si el segmento cruza la medianoche o no
        if start <= end:
            segment_data = self.data[self.data["time"].apply(lambda dt: start <= dt.time() < end)]
        else:
            segment_data = self.data[
                self.data["time"].apply(lambda dt: dt.time() >= start or dt.time() < end)
            ]

        return {
            "sd": segment_data["glucose"].std() if not segment_data.empty else 0.0,
            "mean": segment_data["glucose"].mean() if not segment_data.empty else 0.0,
        }

    def calculate_all_sd_metrics(self) -> dict:
        """
        Calculates all available standard deviation metrics.
        """
        return {
            "SDT": self.sd_total()["sd"],
            "SDw": self.sd_within_day()["sd"],
            "SDhh:mm": self.sd_between_timepoints()["sd"],
            "Noche": self.sd_segment("00:00", 8)["sd"],
            "Día": self.sd_segment("08:00", 8)["sd"],
            "Tarde": self.sd_segment("16:00", 8)["sd"],
            "SDws_1h": self.sd_within_series(hours=1)["sd"],
            "SDws_6h": self.sd_within_series(hours=6)["sd"],
            "SDws_24h": self.sd_within_series(hours=24)["sd"],
            "SDdm": self.sd_daily_mean()["sd"],
            "SDbhh:mm": self.sd_same_timepoint()["sd"],
            "SDbhh:mm_dm": self.sd_same_timepoint_adjusted()["sd"],
            "SDI": self.sd_interaction()["sd"],
        }

    def calculate_all_cv_metrics(self) -> dict:
        """
        Calculates all available coefficient of variation metrics.
        Todo: Check if means are correct.
        """
        return {
            "CVT": self.sd_total()["sd"] / self.sd_total()["mean"] * 100,
            "CVw": self.sd_within_day()["sd"] / self.sd_within_day()["mean"] * 100,
            "CVhh:mm": self.sd_between_timepoints()["sd"]
            / self.sd_between_timepoints()["mean"]
            * 100,
            "CVNoche": self.sd_segment("00:00", 8)["sd"]
            / self.sd_segment("00:00", 8)["mean"]
            * 100,
            "CVDía": self.sd_segment("08:00", 8)["sd"] / self.sd_segment("08:00", 8)["mean"] * 100,
            "CVTarde": self.sd_segment("16:00", 8)["sd"]
            / self.sd_segment("16:00", 8)["mean"]
            * 100,
            "CVSDws_1h": self.sd_within_series(hours=1)["sd"]
            / self.sd_within_series(hours=1)["mean"]
            * 100,
            "CVSDws_6h": self.sd_within_series(hours=6)["sd"]
            / self.sd_within_series(hours=6)["mean"]
            * 100,
            "CVSDws_24h": self.sd_within_series(hours=24)["sd"]
            / self.sd_within_series(hours=24)["mean"]
            * 100,
            "CVdm": self.sd_daily_mean()["sd"] / self.sd_daily_mean()["mean"] * 100,
            "CVbhh:mm": self.sd_same_timepoint()["sd"] / self.sd_same_timepoint()["mean"] * 100,
            "CVbhh:mm_dm": self.sd_same_timepoint_adjusted()["sd"]
            / self.sd_same_timepoint_adjusted()["mean"]
            * 100,
            "CVSDI": self.sd_interaction()["sd"] / self.sd_interaction()["mean"] * 100,
        }

    def MAGE_Baghurst(self, threshold_sd: int = 1, approach: int = 1, plot: bool = False) -> dict:
        """
        Calculates MAGE using the specific algorithm from Baghurst.

        Main changes:
        1. Correct handling of edges in smoothing
        2. Search for turning points in original data between minima/maxima of smoothed profile
        3. Iterative process of eliminating invalid points
        4. Handling of excursions at the beginning/end of the dataset

        :param threshold_sd: Number of standard deviations for the threshold
        :param approach: 1 to use smoothing per original Baghurst,
                        2 for direct elimination, 3 for improved smoothing
        :param plot: If True, generates a visualization of identified peaks and valleys
        :return: Dictionary with MAGE+, MAGE- and related metrics

        Approach 1: Original Baghurst algorithm with smoothing and iterative elimination process
        Approach 2: Direct elimination of intermediate points in monotonic sequences
        Approach 3: Improved smoothing with additional turning point filtering
        """
        glucose = self.data["glucose"].values
        times = self.data["time"].values
        sd = self.sd()
        threshold = threshold_sd * sd

        # Guardar los turning points para cada enfoque si plot=True
        turning_points_approaches = {}

        # Enfoque 1: Suavizado según algoritmo original de Baghurst
        if approach == 1 or plot:
            # PASO 1: Aplicar filtro de suavizado e identificar turning points en datos suavizados
            weights = np.array([1, 2, 4, 8, 16, 8, 4, 2, 1]) / 46

            # Usar np.convolve para suavizado central (mucho más rápido)
            smoothed = np.convolve(glucose, weights, mode="same")

            # Ajustar bordes que np.convolve no maneja como el algoritmo original de Baghurst
            for i in range(min(4, len(glucose))):
                smoothed[i] = glucose[: i + 5].mean()
                if len(glucose) > i + 5:
                    smoothed[-(i + 1)] = glucose[-(i + 5) :].mean()

            # Identificar turning points en datos suavizados mediante primeras diferencias
            delta = np.diff(smoothed)
            turning_smoothed = np.where(np.diff(np.sign(delta)))[0] + 1

            # PASO 2: Identificar máximos/mínimos locales en datos originales
            turning_points_1 = []
            for i in range(len(turning_smoothed) - 1):
                start = turning_smoothed[i]
                end = turning_smoothed[i + 1]

                # Buscar máximo real en intervalo ascendente
                if smoothed[start] < smoothed[end]:
                    true_peak = np.argmax(glucose[start:end]) + start
                    turning_points_1.append(true_peak)
                # Buscar mínimo real en intervalo descendente
                else:
                    true_valley = np.argmin(glucose[start:end]) + start
                    turning_points_1.append(true_valley)

            # Añadir el primer y último punto si son extremos
            if (
                len(turning_points_1) > 0
                and turning_points_1[0] > 0
                and (
                    glucose[0] > glucose[turning_points_1[0]]
                    or glucose[0] < glucose[turning_points_1[0]]
                )
            ):
                turning_points_1.insert(0, 0)

            if (
                len(turning_points_1) > 0
                and turning_points_1[-1] < len(glucose) - 1
                and (
                    glucose[-1] > glucose[turning_points_1[-1]]
                    or glucose[-1] < glucose[turning_points_1[-1]]
                )
            ):
                turning_points_1.append(len(glucose) - 1)

            # PASO 3: Eliminar turning points asociados con excursiones no contables en ambos lados
            # Mantener aquellos cuyos máximos/mínimos adyacentes son más bajos/altos en ambos lados
            keep_iterating = True
            while keep_iterating:
                to_delete = []

                for i in range(1, len(turning_points_1) - 1):
                    current_idx = turning_points_1[i]
                    prev_idx = turning_points_1[i - 1]
                    next_idx = turning_points_1[i + 1]

                    current_val = glucose[current_idx]
                    prev_val = glucose[prev_idx]
                    next_val = glucose[next_idx]

                    # Verificar si ambas diferencias son menores que el umbral
                    if (
                        abs(current_val - prev_val) < threshold
                        and abs(current_val - next_val) < threshold
                    ):
                        # Retener si es un máximo local (ambos adyacentes más bajos)
                        is_local_max = current_val > prev_val and current_val > next_val
                        # Retener si es un mínimo local (ambos adyacentes más altos)
                        is_local_min = current_val < prev_val and current_val < next_val

                        # Si no es un máximo/mínimo local, marcar para eliminación
                        if not (is_local_max or is_local_min):
                            to_delete.append(i)

                # Si no hay más puntos para eliminar, terminar
                if not to_delete:
                    keep_iterating = False
                else:
                    # Eliminar puntos marcados
                    for idx in sorted(to_delete, reverse=True):
                        turning_points_1.pop(idx)

                # PASO 4: Eliminar observaciones que ya no son turning points
                delta_turning = np.diff([glucose[tp] for tp in turning_points_1])
                false_turning = []

                for i in range(1, len(delta_turning)):
                    # Si las diferencias tienen el mismo signo, no es un turning point
                    if np.sign(delta_turning[i - 1]) == np.sign(delta_turning[i]):
                        false_turning.append(i)

                # Eliminar puntos falsos
                for idx in sorted(false_turning, reverse=True):
                    turning_points_1.pop(idx)

            # PASO 5: Eliminar turning points con excursión contable en un solo lado
            if len(turning_points_1) >= 3:
                to_delete = []

                for i in range(1, len(turning_points_1) - 1):
                    current_idx = turning_points_1[i]
                    prev_idx = turning_points_1[i - 1]
                    next_idx = turning_points_1[i + 1]

                    current_val = glucose[current_idx]
                    prev_val = glucose[prev_idx]
                    next_val = glucose[next_idx]

                    # Verificar si sólo hay excursión significativa en un lado
                    has_sig_prev = abs(current_val - prev_val) >= threshold
                    has_sig_next = abs(current_val - next_val) >= threshold

                    if has_sig_prev != has_sig_next:  # XOR lógico - solo uno es verdadero
                        to_delete.append(i)

                # Eliminar puntos marcados
                for idx in sorted(to_delete, reverse=True):
                    turning_points_1.pop(idx)

                # Verificar de nuevo si hay puntos que ya no son turning points
                delta_turning = np.diff([glucose[tp] for tp in turning_points_1])
                false_turning = []

                for i in range(1, len(delta_turning)):
                    if np.sign(delta_turning[i - 1]) == np.sign(delta_turning[i]):
                        false_turning.append(i)

                for idx in sorted(false_turning, reverse=True):
                    turning_points_1.pop(idx)

            # PASO 6: Eliminar excursiones no contables al inicio o final
            if len(turning_points_1) >= 2:
                # Verificar excursión inicial
                if abs(glucose[turning_points_1[0]] - glucose[turning_points_1[1]]) < threshold:
                    turning_points_1.pop(0)

                # Verificar excursión final
                if (
                    len(turning_points_1) >= 2
                    and abs(glucose[turning_points_1[-1]] - glucose[turning_points_1[-2]])
                    < threshold
                ):
                    turning_points_1.pop(-1)

            # Asegurar que los puntos están ordenados y son únicos
            turning_points_1 = sorted(set(turning_points_1))
            turning_points_approaches[1] = turning_points_1

            if approach == 1:
                turning_points = turning_points_1

        # Enfoque 2: Eliminación directa
        if approach == 2 or plot:
            turning_points_2 = []
            i = 0

            # 1. Primera pasada: eliminar puntos intermedios en secuencias monótonas
            while i < len(glucose) - 2:
                if (glucose[i] <= glucose[i + 1] <= glucose[i + 2]) or (
                    glucose[i] >= glucose[i + 1] >= glucose[i + 2]
                ):
                    # El punto intermedio es parte de una secuencia monótona
                    i += 1
                else:
                    # Punto i+1 es un turning point potencial
                    turning_points_2.append(i + 1)
                    i += 2

            # Asegurar que incluimos el primer y último punto si son relevantes
            if len(turning_points_2) == 0 or turning_points_2[0] > 0:
                turning_points_2.insert(0, 0)
            if turning_points_2[-1] < len(glucose) - 1:
                turning_points_2.append(len(glucose) - 1)

            # 2. Segunda pasada: eliminar excursiones que no superan el umbral
            valid_points = []
            for i in range(1, len(turning_points_2) - 1):
                prev_val = glucose[turning_points_2[i - 1]]
                curr_val = glucose[turning_points_2[i]]
                next_val = glucose[turning_points_2[i + 1]]

                # Verificar si es un máximo o mínimo válido
                if (
                    (curr_val > prev_val and curr_val > next_val)
                    or (curr_val < prev_val and curr_val < next_val)
                ) and (
                    abs(curr_val - prev_val) >= threshold or abs(curr_val - next_val) >= threshold
                ):
                    valid_points.append(turning_points_2[i])

            # Asegurar que mantenemos puntos inicial y final si son necesarios
            if valid_points and valid_points[0] > 0:
                valid_points.insert(0, 0)
            if valid_points and valid_points[-1] < len(glucose) - 1:
                valid_points.append(len(glucose) - 1)

            turning_points_2 = valid_points
            turning_points_approaches[2] = turning_points_2

            if approach == 2:
                turning_points = turning_points_2

        # Enfoque 3: Suavizado mejorado
        if approach == 3 or plot:
            # 1. Aplicar filtro de suavizado con manejo de bordes (igual que enfoque 1)
            weights = np.array([1, 2, 4, 8, 16, 8, 4, 2, 1]) / 46
            smoothed = np.zeros_like(glucose)

            # Suavizado central
            for i in range(4, len(glucose) - 4):
                smoothed[i] = np.dot(weights, glucose[i - 4 : i + 5])

            # Manejo de bordes con media simple
            for i in range(4):
                smoothed[i] = glucose[: i + 5].mean()
                smoothed[-(i + 1)] = glucose[-(i + 5) :].mean()

            # 2. Identificar mínimos/máximos en el perfil suavizado
            delta = np.diff(smoothed)
            turning_smoothed = np.where(np.diff(np.sign(delta)))[0] + 1

            # 3. Buscar turning points reales en datos originales entre los intervalos suavizados
            # y aplicar filtrado adicional

            # Primero identificamos todos los turning points potenciales
            potential_turning_points = []
            for i in range(len(turning_smoothed) - 1):
                start = turning_smoothed[i]
                end = turning_smoothed[i + 1]

                # Buscar máximo real en intervalo ascendente
                if smoothed[start] < smoothed[end]:
                    true_peak = np.argmax(glucose[start:end]) + start
                    potential_turning_points.append((true_peak, "peak", glucose[true_peak]))
                # Buscar mínimo real en intervalo descendente
                else:
                    true_valley = np.argmin(glucose[start:end]) + start
                    potential_turning_points.append((true_valley, "valley", glucose[true_valley]))

            # Ahora procesamos los turning points para eliminar picos/valles intermedios menores
            turning_points_3 = []
            if potential_turning_points:
                # Añadir el primer punto
                turning_points_3.append(potential_turning_points[0][0])

                # Procesar el resto de puntos
                for i in range(1, len(potential_turning_points) - 1):
                    prev_point, prev_type, prev_value = potential_turning_points[i - 1]
                    curr_point, curr_type, curr_value = potential_turning_points[i]
                    next_point, next_type, next_value = potential_turning_points[i + 1]

                    # Si tenemos un patrón valle-pico-valle o pico-valle-pico
                    if curr_type == prev_type:
                        # Saltamos este punto, es redundante
                        continue

                    # Si tenemos un pico entre dos valles, verificar si es significativo
                    if (
                        curr_type == "peak"
                        and prev_type == "valley"
                        and next_type == "valley"
                        and (
                            curr_value - prev_value < threshold / 2
                            or curr_value - next_value < threshold / 2
                        )
                    ):
                        # Si el pico no es significativamente más alto que ambos valles, lo saltamos
                        continue

                    # Si tenemos un valle entre dos picos, verificar si es significativo
                    if (
                        curr_type == "valley"
                        and prev_type == "peak"
                        and next_type == "peak"
                        and (
                            prev_value - curr_value < threshold / 2
                            or next_value - curr_value < threshold / 2
                        )
                    ):
                        continue

                    # Si llegamos aquí, el punto es significativo
                    turning_points_3.append(curr_point)

                # Añadir el último punto
                turning_points_3.append(potential_turning_points[-1][0])

            # Asegurarnos de que tenemos al menos el primer y último punto
            if len(turning_points_3) == 0 and len(glucose) > 0:
                turning_points_3 = [0, len(glucose) - 1]
            elif len(turning_points_3) == 1 and len(glucose) > 1:
                if turning_points_3[0] == 0:
                    turning_points_3.append(len(glucose) - 1)
                else:
                    turning_points_3.insert(0, 0)

            turning_points_3 = np.unique(turning_points_3)
            turning_points_approaches[3] = turning_points_3

            if approach == 3:
                turning_points = turning_points_3

        # 3. Calcular excursiones válidas
        excursions = []
        last_val = glucose[turning_points[0]]
        last_point = turning_points[0]

        for point in turning_points[1:]:
            curr_val = glucose[point]
            diff = abs(curr_val - last_val)

            if diff >= threshold:
                excursions.append(
                    {
                        "start_point": last_point,
                        "end_point": point,
                        "start": last_val,
                        "end": curr_val,
                        "type": "up" if curr_val > last_val else "down",
                        "magnitude": diff,
                    }
                )
                last_val = curr_val
                last_point = point
            else:
                # Si no supera el umbral, actualizamos el último valor sin crear excursión
                last_val = curr_val
                last_point = point

        # Separar excursiones y calcular métricas
        excursions_up = [e["magnitude"] for e in excursions if e["type"] == "up"]
        excursions_down = [e["magnitude"] for e in excursions if e["type"] == "down"]

        mage_plus = np.mean(excursions_up) if excursions_up else 0
        mage_minus = np.mean(excursions_down) if excursions_down else 0
        mage_avg = (
            np.mean(excursions_up + excursions_down) if (excursions_up or excursions_down) else 0
        )

        # Generar visualización si plot=True
        if plot:
            from datetime import timedelta

            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt

            # Obtener todos los días únicos en los datos
            unique_days = pd.Series(times).dt.normalize().unique()

            # Configurar la figura y ejes - ahora con 3 subplots
            fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            plt.ion()  # Modo interactivo

            # Calcular excursiones para cada enfoque
            excursions_by_approach = {}

            for approach_num in [1, 2, 3]:
                # Usar los turning points específicos de este enfoque
                if approach_num in turning_points_approaches:
                    tp = turning_points_approaches[approach_num]

                    # Calcular excursiones para este enfoque
                    excursions_approach = []
                    if len(tp) > 1:
                        last_val = glucose[tp[0]]
                        last_point = tp[0]

                        for point in tp[1:]:
                            curr_val = glucose[point]
                            diff = abs(curr_val - last_val)

                            if diff >= threshold:
                                excursions_approach.append(
                                    {
                                        "start_point": last_point,
                                        "end_point": point,
                                        "start": last_val,
                                        "end": curr_val,
                                        "type": "up" if curr_val > last_val else "down",
                                        "magnitude": diff,
                                    }
                                )

                            # Siempre actualizamos el último valor y punto
                            last_val = curr_val
                            last_point = point

                    excursions_by_approach[approach_num] = excursions_approach

            # Función para actualizar el gráfico con un día específico
            def update_plot(day_index):
                # Limpiar los ejes
                for ax in axs:
                    ax.clear()

                # Obtener el día actual
                current_day = unique_days[day_index]
                next_day = current_day + timedelta(days=1)

                # Filtrar datos para mostrar solo el día actual
                day_mask = (times >= current_day) & (times < next_day)
                day_times = times[day_mask]
                day_glucose = glucose[day_mask]

                if len(day_times) > 0:
                    # Para cada enfoque
                    for i, approach in enumerate([1, 2, 3]):
                        ax = axs[i]
                        # Dibujar la línea de glucosa
                        ax.plot(day_times, day_glucose, "b-", label="Glucosa")

                        # Obtener turning points para este enfoque
                        approach_turning_points = turning_points_approaches.get(approach, [])

                        # Obtener excursiones para este enfoque
                        approach_excursions = excursions_by_approach.get(approach, [])

                        # Filtrar turning points para este día
                        day_turning_points = [tp for tp in approach_turning_points if day_mask[tp]]

                        # Identificar puntos involucrados en excursiones
                        excursion_points = set()
                        day_excursions = []

                        for exc in approach_excursions:
                            start_point = exc["start_point"]
                            end_point = exc["end_point"]

                            # Verificar si la excursión está en el día actual
                            if day_mask[start_point] or day_mask[end_point]:
                                if day_mask[start_point]:
                                    excursion_points.add(start_point)
                                if day_mask[end_point]:
                                    excursion_points.add(end_point)
                                day_excursions.append(exc)

                        # Clasificar turning points
                        significant_points = [
                            tp for tp in day_turning_points if tp in excursion_points
                        ]
                        non_significant_points = [
                            tp for tp in day_turning_points if tp not in excursion_points
                        ]

                        # Dibujar puntos no significativos en azul
                        for tp in non_significant_points:
                            ax.plot(times[tp], glucose[tp], "bo", markersize=6)

                        # Dibujar puntos significativos en rojo
                        for tp in significant_points:
                            ax.plot(times[tp], glucose[tp], "ro", markersize=8)

                        # Dibujar líneas para las excursiones
                        for exc in day_excursions:
                            start_point = exc["start_point"]
                            end_point = exc["end_point"]

                            # Asegurarse de que ambos puntos están en el día actual
                            if day_mask[start_point] and day_mask[end_point]:
                                # Dibujar línea gruesa de color según tipo de excursión
                                color = "green" if exc["type"] == "up" else "red"
                                ax.plot(
                                    [times[start_point], times[end_point]],
                                    [glucose[start_point], glucose[end_point]],
                                    color=color,
                                    linewidth=2.5,
                                    alpha=0.7,
                                )

                        # Calcular MAGE para este enfoque y día
                        excursions_up = [
                            e["magnitude"] for e in day_excursions if e["type"] == "up"
                        ]
                        excursions_down = [
                            e["magnitude"] for e in day_excursions if e["type"] == "down"
                        ]

                        mage_plus = np.mean(excursions_up) if excursions_up else 0
                        mage_minus = np.mean(excursions_down) if excursions_down else 0
                        mage_avg = (
                            np.mean(excursions_up + excursions_down)
                            if (excursions_up or excursions_down)
                            else 0
                        )

                        # Configurar título y etiquetas
                        approach_name = (
                            "Suavizado"
                            if approach == 1
                            else "Eliminación directa"
                            if approach == 2
                            else "Suavizado mejorado"
                        )
                        ax.set_title(
                            f"MAGE Baghurst - Enfoque {approach} ({approach_name}) - "
                            f"{current_day.strftime('%d/%m/%Y')}\n"
                            f"MAGE+: {mage_plus:.1f}, MAGE-: {mage_minus:.1f}, "
                            f"MAGE: {mage_avg:.1f}, Excursiones: {len(day_excursions)}"
                        )
                        ax.set_ylabel("Glucosa (mg/dL)")
                        ax.grid(True)
                        ax.axhline(
                            y=self.mean() + threshold,
                            color="g",
                            linestyle="--",
                            label=f"Umbral (+{threshold_sd} SD)",
                        )
                        ax.axhline(
                            y=self.mean() - threshold,
                            color="g",
                            linestyle="--",
                            label=f"Umbral (-{threshold_sd} SD)",
                        )

                        # Leyenda personalizada
                        from matplotlib.lines import Line2D

                        custom_lines = [
                            Line2D(
                                [0],
                                [0],
                                color="b",
                                marker="o",
                                linestyle="None",
                                markersize=6,
                            ),
                            Line2D(
                                [0],
                                [0],
                                color="r",
                                marker="o",
                                linestyle="None",
                                markersize=8,
                            ),
                            Line2D([0], [0], color="green", linewidth=2.5),
                            Line2D([0], [0], color="red", linewidth=2.5),
                            Line2D([0], [0], color="g", linestyle="--"),
                        ]
                        ax.legend(
                            custom_lines,
                            [
                                "Puntos de inflexión",
                                "Puntos de excursiones",
                                "Excursión positiva",
                                "Excursión negativa",
                                "Umbral (±1 SD)",
                            ],
                        )

                    # Configurar eje x para el último gráfico
                    axs[2].set_xlabel("Tiempo")

                    # Formatear eje x para mostrar horas
                    for ax in axs:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

                    # Añadir información de navegación
                    fig.suptitle(
                        f"Día {day_index + 1} de {len(unique_days)} - Presiona ← o → para navegar, q para salir",
                        fontsize=12,
                    )

                    plt.tight_layout()
                    plt.subplots_adjust(top=0.95)  # Hacer espacio para el título superior
                    fig.canvas.draw_idle()

            # Índice del día actual
            current_day_index = 0

            # Función para manejar eventos de teclado
            def on_key(event):
                nonlocal current_day_index

                if event.key == "right" and current_day_index < len(unique_days) - 1:
                    current_day_index += 1
                    update_plot(current_day_index)
                elif event.key == "left" and current_day_index > 0:
                    current_day_index -= 1
                    update_plot(current_day_index)
                elif event.key == "q":
                    plt.close(fig)

            # Conectar el evento de teclado
            fig.canvas.mpl_connect("key_press_event", on_key)

            # Mostrar el primer día
            update_plot(current_day_index)

            # Mostrar instrucciones
            print("Navegación:")
            print("  ← Flecha izquierda: Día anterior")
            print("  → Flecha derecha: Día siguiente")
            print("  q: Salir")

            # Bloquear hasta que se cierre la figura
            plt.show(block=True)

        return {
            "MAGE+": round(mage_plus, 2),
            "MAGE-": round(mage_minus, 2),
            "MAGE_avg": round(mage_avg, 2),
            "SD_used": round(sd, 2),
            "threshold": round(threshold, 2),
            "num_excursions": len(excursions),
        }

    def MAGE(self) -> float:
        """
        Calculates MAGE (Mean Amplitude of Glycemic Excursions).
        :return: MAGE value.
        """
        sd = self.sd()
        peaks_and_nadirs = self.data[
            (self.data["glucose"].shift(1) < self.data["glucose"])
            & (self.data["glucose"] > self.data["glucose"].shift(-1))
            | (self.data["glucose"].shift(1) > self.data["glucose"])
            & (self.data["glucose"] < self.data["glucose"].shift(-1))
        ].reset_index(drop=True)

        if len(peaks_and_nadirs) < 2:
            return 0.0

        excursions = []
        starts_with_peak = peaks_and_nadirs["glucose"][0] > peaks_and_nadirs["glucose"][1]

        for i in range(0, len(peaks_and_nadirs) - 1, 2):
            if starts_with_peak:
                peak, nadir = (
                    peaks_and_nadirs["glucose"][i],
                    peaks_and_nadirs["glucose"][i + 1],
                )
            else:
                nadir, peak = (
                    peaks_and_nadirs["glucose"][i],
                    peaks_and_nadirs["glucose"][i + 1],
                )
            if abs(peak - nadir) > sd:
                excursions.append(abs(peak - nadir))

        return float(np.mean(excursions)) if excursions else 0.0

    def MODD(self, days: int = 1) -> dict:
        """
        Calculates MODD (Mean Of Daily Differences) for a specific day interval.
        Optimized vectorized version.

        :param days: Number of days to calculate differences (1-6).
        :return: Dictionary with MODD value and related statistics.
        """
        if not 1 <= days <= 6:
            raise ValueError("The number of days must be between 1 and 6")

        df = self.data[["time", "glucose"]].copy()
        target_delta = pd.Timedelta(days=days)

        # Use time as index for alignment
        df_indexed = df.set_index("time")

        # Shift back to compare with 'days' ago
        try:
            # We use a frequency-based shift to align exactly by time of day
            df_shifted = df_indexed.shift(1, freq=target_delta)

            # Join to align values at the same time of day
            merged = df_indexed.join(df_shifted, lsuffix="_current", rsuffix="_past", how="inner")

            if merged.empty:
                return {
                    "value": None,
                    "n_observations": 0,
                    "std": None,
                    "correlation": None,
                }

            abs_diffs = (merged["glucose_current"] - merged["glucose_past"]).abs()

            modd_value = float(abs_diffs.mean())
            std_value = float(abs_diffs.std()) if len(abs_diffs) > 1 else 0.0
            correlation = float(merged["glucose_current"].corr(merged["glucose_past"]))

            return {
                "value": modd_value,
                "n_observations": len(abs_diffs),
                "std": std_value,
                "correlation": correlation,
            }
        except Exception as e:
            if getattr(self, "log", False):
                print(f"Error in vectorized MODD: {e}")
            return {
                "value": None,
                "n_observations": 0,
                "std": None,
                "correlation": None,
            }

    def CONGA(self, hours: int = 4, max_gap_minutes: float | None = None) -> dict:
        """
        Calculates CONGA (Continuous Overlapping Net Glycemic Action).

        CONGA measures intraday glycemic variability by calculating the standard deviation
        of differences between current values and values 'n' hours earlier.

        :param hours: Number of hours for the time interval (n).
        :param max_gap_minutes: Maximum allowed gap in minutes between measurements to
                           consider a comparison valid. If None, uses 2 times the typical interval.
        :return: Dictionary with CONGA value and related statistics.
        :reference: McDonnell CM, et al. Diabetes Technol Ther. 2005;7(2):243-9.
                   DOI: 10.1089/dia.2005.7.243
        """
        # Crear copia de datos ordenados por tiempo
        df = self.data.sort_values("time").copy()

        # Calcular el intervalo en minutos
        interval_minutes = self.typical_interval  # Ya está en minutos

        # Si no se especifica max_gap_minutes, usar 2 veces el intervalo típico
        if max_gap_minutes is None:
            max_gap_minutes = 2 * interval_minutes

        # Calcular cuántos intervalos corresponden a 'hours' horas
        n_intervals = int((hours * 60) / interval_minutes)

        if n_intervals <= 0:
            raise ValueError(
                f"El intervalo de {hours} horas es demasiado pequeño para los datos disponibles"
            )

        # Calcular diferencias entre valores actuales y valores de 'n' horas antes
        # pero teniendo en cuenta posibles desconexiones

        # Método 1: Usando shift pero verificando la diferencia de tiempo real
        df["time_n_hours_ago"] = df["time"].shift(n_intervals)
        df["glucose_n_hours_ago"] = df["glucose"].shift(n_intervals)

        # Calcular la diferencia de tiempo real en minutos
        df["time_diff_minutes"] = (df["time"] - df["time_n_hours_ago"]).dt.total_seconds() / 60

        # Calcular diferencia de glucosa solo si la diferencia de tiempo está cerca del objetivo
        target_diff_minutes = hours * 60
        df["valid_comparison"] = (
            df["time_diff_minutes"] >= target_diff_minutes - max_gap_minutes
        ) & (df["time_diff_minutes"] <= target_diff_minutes + max_gap_minutes)

        # Calcular diferencia solo para comparaciones válidas
        df["difference"] = np.where(
            df["valid_comparison"], df["glucose"] - df["glucose_n_hours_ago"], np.nan
        )

        # Eliminar filas con valores faltantes o comparaciones inválidas
        valid_data = df.dropna(subset=["difference"])

        if len(valid_data) == 0:
            return {
                "value": None,
                "n_observations": 0,
                "mean_difference": None,
                "abs_mean_difference": None,
                "std": None,
                "hours": hours,
                "max_gap_minutes": max_gap_minutes,
            }

        # Calcular CONGA como la desviación estándar de las diferencias
        conga_value = valid_data["difference"].std()

        # Calcular estadísticas adicionales
        mean_diff = valid_data["difference"].mean()
        abs_mean_diff = valid_data["difference"].abs().mean()

        # Información sobre desconexiones
        total_comparisons = len(df.dropna(subset=["glucose_n_hours_ago"]))
        valid_comparisons = len(valid_data)
        invalid_comparisons = total_comparisons - valid_comparisons

        return {
            "value": conga_value,
            "n_observations": len(valid_data),
            "mean_difference": mean_diff,
            "abs_mean_difference": abs_mean_diff,
            "hours": hours,
            "max_gap_minutes": max_gap_minutes,
            "total_comparisons": total_comparisons,
            "valid_comparisons": valid_comparisons,
            "invalid_comparisons": invalid_comparisons,
            "percent_valid": (valid_comparisons / total_comparisons * 100)
            if total_comparisons > 0
            else 0,
        }

    def Lability_index(self, interval: int = 1, period: str = "week") -> dict:
        """
        Calculates Lability Index (LI) for a specific time interval.

        :param interval: Number of hours between consecutive measurements.
        :param period: Time period to calculate LI ('week' or 'month').
        :return: Dictionary with LI values and statistics.

        DOI: 10.2337/diabetes.53.4.955
        """
        # Añadimos timing para ver dónde se gasta el tiempo

        data_copy = self.data.copy()
        data_copy["time_rounded"] = data_copy["time"].dt.floor("h")
        data_copy["week"] = data_copy["time"].dt.isocalendar().week

        weekly_li = []

        for _week, group in data_copy.groupby("week"):
            group = group.sort_values("time_rounded")

            # Versión vectorizada dentro del grupo
            glucose_diffs = group["glucose"].shift(-interval) - group["glucose"]
            li_values = (glucose_diffs**2) / interval
            li_week = li_values.dropna().sum()
            weekly_li.append(li_week)

        mean_li = np.mean(weekly_li) if weekly_li else 0
        mean_li_mmol = mean_li / (18.0**2)

        # Añadimos la interpretación clínica
        mean_li_por_hora = mean_li / 168
        cambio_tipico_por_hora = math.sqrt(mean_li_por_hora)

        return {
            "weekly_values": weekly_li,
            "mean_li": mean_li,
            "mean_li_mmol": mean_li_mmol,
            "std_li": np.std(weekly_li) if len(weekly_li) > 1 else 0,
            "n_weeks": len(weekly_li),
            # Nuevos campos de interpretación clínica
            "cambio_tipico_por_hora": cambio_tipico_por_hora,
        }

    def Variability(self) -> str:
        """
        Calculates all variability metrics.
        :return: A JSON string with all variability metrics.
        """
        variability_metrics = {
            "CONGA1": self.CONGA(hours=1),
            "CONGA2": self.CONGA(hours=2),
            "CONGA4": self.CONGA(hours=4),
            "CONGA6": self.CONGA(hours=6),
            "CONGA24": self.CONGA(hours=24),
            "MODD": self.MODD(days=1),
            "J_index": self.j_index(),
            "LBGI": self.LBGI(),
            "HBGI": self.HBGI(),
            "MAGE": self.MAGE(),
            "M_value": self.M_Value(),
            "LI_week": self.Lability_index(interval=1, period="week"),
        }
        return variability_metrics

    def variability_summary(self) -> dict[str, Any]:
        """
        Complete summary of all variability metrics.

        Returns:
            dict: Complete summary of variability metrics
        """
        return {
            "basic_variability": {"sd_total": self.sd_total(), "cv": self.cv()},
            "excursion_metrics": {
                "mage": self.MAGE(),
            },
            "inter_day_variability": {
                "modd_1day": self.MODD(1),
                "modd_2days": self.MODD(2),
            },
            "intra_day_variability": {
                "conga_1h": self.CONGA(1),
                "conga_2h": self.CONGA(2),
                "conga_4h": self.CONGA(4),
            },
            "lability": {"lability_index": self.Lability_index()},
        }

    # GLYCEMIC QUALITY MEASURES

    def M_Value(self, reference_glucose: int = 90) -> dict:
        """
        Calculates M-Value according to Schlichtkrull's definition and Service's consideration.

        M-Value is a hybrid between:
        1. Mean blood glucose deviation
        2. Glycemic variability

        Special features:
        - Gives more weight to hypoglycemia than hyperglycemia
        - Uses 90 mg/dL as historical reference value. Original paper used 120 mg/dL
        - Combines mean deviation and fluctuation amplitude

        Formula: M = (1/n)∑|10 * log10(BG/120)|³ + W/20
        (The correction factor can be omitted when there are more than 24 data points)

        :param reference_glucose: Reference value (default 90 mg/dL - updated from docstring default)
        :return: Dictionary with M-Value and components
        :reference: 10.1111/j.0954-6820.1965.tb01810.x
        :reference: 10.2337/db12-1396
        """
        # Convertir directamente a array de NumPy para operaciones más rápidas
        glucose_values = self.data["glucose"].values

        # Calcular M_BS vectorizado
        M_BS_values = np.abs(10 * np.log10(glucose_values / reference_glucose)) ** 3
        M_BS_mean = np.mean(M_BS_values)
        return round(M_BS_mean, 2)

    def j_index(self) -> float:
        """Calculates J-index.
        DOI: 10.1055/s-2007-979906
        """
        return 0.001 * (self.mean() + self.sd()) ** 2

    def GRADE(self, unit: str = "mg/dL") -> dict:
        """
        Calculates GRADE.
        :return: GRADE value.
        :reference: DOI: 10.1111/j.1464-5491.2007.02119.x
        """
        # Crear copia de los datos para no modificar los originales
        df = self.data.copy()

        # Convertir a mmol/L si es necesario para los cálculos internos
        if unit.lower() == "mg/dl":
            # Para cálculos, usamos la conversión a mmol/L
            df["glucose_value"] = df["glucose"]
        elif unit.lower() == "mmol/l":
            # Si los datos están en mmol/L, los mantenemos igual
            df["glucose_value"] = df["glucose"] * 18  # Convertir a mg/dL para clasificación
        else:
            raise ValueError("La unidad debe ser 'mg/dL' o 'mmol/L'")

        # Definir rangos para clasificación (siempre en mg/dL)
        hypo_threshold = 70  # mg/dL
        hyper_threshold = 140  # mg/dL

        # Clasificar valores según rangos (usando valores en mg/dL)
        df["hypo"] = df["glucose_value"] < hypo_threshold
        df["eu"] = (df["glucose_value"] >= hypo_threshold) & (
            df["glucose_value"] <= hyper_threshold
        )
        df["hyper"] = df["glucose_value"] > hyper_threshold

        # Vectorización para calcular GRADE
        glucose_values = df["glucose_value"].values

        # Inicializar array de resultados
        grade_values = np.zeros_like(glucose_values, dtype=float)

        # Crear máscara para valores dentro del rango válido
        valid_mask = (glucose_values >= 37) & (glucose_values <= 630)

        # Convertir mg/dL a mmol/L y calcular GRADE para valores válidos
        with np.errstate(invalid="ignore", divide="ignore"):
            # Convertir a mmol/L dividiendo por 18
            glucose_mmol = glucose_values[valid_mask] / 18
            log_log_values = np.log10(np.log10(glucose_mmol))
            grade_values[valid_mask] = 425 * (log_log_values + 0.16) ** 2

        # Asignar 50 a valores inválidos (fuera de rango o con error logarítmico)
        invalid_mask = ~valid_mask | ~np.isfinite(grade_values)
        grade_values[invalid_mask] = 50

        # Asignar resultados al dataframe
        df["grade"] = grade_values

        # Calcular componentes de GRADE
        grade_total = df["grade"].sum()
        grade_hypo = df.loc[df["hypo"], "grade"].sum()
        grade_eu = df.loc[df["eu"], "grade"].sum()
        grade_hyper = df.loc[df["hyper"], "grade"].sum()

        # Calcular porcentajes
        hypo_percent = (grade_hypo / grade_total) * 100 if grade_total > 0 else 0
        eu_percent = (grade_eu / grade_total) * 100 if grade_total > 0 else 0
        hyper_percent = (grade_hyper / grade_total) * 100 if grade_total > 0 else 0

        # Calcular GRADE score (media de todos los valores GRADE)
        grade_score = df["grade"].mean()

        # Crear diccionario de resultados
        results = {
            "grade_score": float(grade_score),
            "hypo_percent": float(hypo_percent),
            "eu_percent": float(eu_percent),
            "hyper_percent": float(hyper_percent),
        }

        return results

    def LBGI(self) -> float:
        """
        Calculates Low Blood Glucose Index (LBGI).
        :return: LBGI value.
        :reference: DOI: 10.2337/db12-1396
        """
        # Usar copia para no modificar los datos originales
        glucose_values = self.data["glucose"].values

        # Cálculos vectorizados
        f_bg = 1.509 * ((np.log(glucose_values)) ** 1.084 - 5.381)
        r_bg = 10 * f_bg**2
        rl_bg = np.where(f_bg < 0, r_bg, 0)

        return float(np.mean(rl_bg))

    def HBGI(self) -> float:
        """
        Calculates High Blood Glucose Index (HBGI).
        :return: HBGI value.
        :reference: DOI: 10.2337/db12-1396
        """
        # Use copy to avoid modifying original data
        glucose_values = self.data["glucose"].values

        # Vectorized calculations
        f_bg = 1.509 * ((np.log(glucose_values)) ** 1.084 - 5.381)
        r_bg = 10 * f_bg**2
        rh_bg = np.where(f_bg > 0, r_bg, 0)

        return float(np.mean(rh_bg))

    def GRI(self, pregnancy: bool = False) -> dict:
        """
        Calculates Glucose Risk Index (GRI).

        GRI combines time in different glucose ranges, giving different weights
        to hypoglycemia and hyperglycemia.

        GRI = (3.0 * VLow) + (2.4 * Low) + (1.6 * VHigh) + (0.8 * High)

        Standard ranges:
        - VLow: <54 mg/dL
        - Low: 54-70 mg/dL
        - VHigh: >250 mg/dL
        - High: 180-250 mg/dL

        Pregnancy ranges (Experimental, not clinically validated):
        - VLow: <55 mg/dL
        - Low: 55-63 mg/dL
        - VHigh: >250 mg/dL
        - High: 140-250 mg/dL

        :param pregnancy: If True, uses specific ranges for pregnancy.
        :return: Dictionary with GRI and its components.
        :reference: DOI: 10.1016/j.diabres.2013.03.006 (Standard)
        """
        # NOTE: GRI was originally validated for non-pregnant adults.
        # The use of pregnancy-specific targets here is experimental and NOT clinically validated.

        # Define ranges based on pregnancy status
        if pregnancy:
            vlow_threshold = 55
            low_range = (55, 63)
            high_range = (140, 250)
            vhigh_threshold = 250
        else:
            vlow_threshold = 54
            low_range = (54, 70)
            high_range = (180, 250)
            vhigh_threshold = 250

        # Calculate percentage of time in each range
        vlow = self.TBR(vlow_threshold)  # < threshold
        low = self.calculate_time_in_range(*low_range)
        vhigh = self.TAR(vhigh_threshold)  # > threshold
        high = self.calculate_time_in_range(*high_range)

        # Calculate GRI
        gri = (3.0 * vlow) + (2.4 * low) + (1.6 * vhigh) + (0.8 * high)

        # Calculate components for derived metrics
        hypo_component = vlow + (0.8 * low)
        hyper_component = vhigh + (0.5 * high)

        # Calculate TIR (Time In Range) relative to the used GRI thresholds
        tir = 100 - (vlow + low + vhigh + high)

        return {
            "GRI": round(gri, 2),
            "is_pregnancy": pregnancy,
            "validated": not pregnancy,
            "components": {
                "VLow": round(vlow, 2),
                "Low": round(low, 2),
                "VHigh": round(vhigh, 2),
                "High": round(high, 2),
            },
            "derived_metrics": {
                "hypo_component": round(hypo_component, 2),
                "hyper_component": round(hyper_component, 2),
                "TIR": round(tir, 2),
            },
        }

    def ADRR(self) -> dict:
        """
        Calculates Average Daily Risk Range (ADRR).

        ADRR is a variability measure that:
        1. Is equally sensitive to hypoglycemia and hyperglycemia.
        2. Uses logarithmic transformation to normalize the scale.

        :return: Dictionary with ADRR and related statistics.

        :reference: DOI: 10.1177/193229681300700529
        """
        # Group data by day
        daily_readings = self.data.groupby(self.data["time"].dt.date)

        # Glucose transformation function
        def transform_bg(bg_values):
            # f(BG) = 1.509 * (ln(BG)**1.084 - 5.381)
            return 1.509 * ((np.log(bg_values)) ** 1.084 - 5.381)

        # Calculate daily risks
        daily_risks = []
        daily_hypo_risks = []  # Separate list for hypoglycemia risks
        daily_hyper_risks = []  # Separate list for hyperglycemia risks

        for _date, day_data in daily_readings:
            # Transform glucose values
            bg_values = day_data["glucose"].values
            transformed = transform_bg(bg_values)

            # Separate hypo and hyper risks
            rl = np.where(transformed < 0, 10 * transformed**2, 0)  # Hypoglycemia risk
            rh = np.where(transformed > 0, 10 * transformed**2, 0)  # Hyperglycemia risk
            # Get max daily risks
            lr = np.max(rl) if len(rl) > 0 else 0  # Max hypo risk
            hr = np.max(rh) if len(rh) > 0 else 0  # Max hyper risk

            daily_risks.append(lr + hr)  # Total risk sum
            daily_hypo_risks.append(lr)  # Save hypo risk
            daily_hyper_risks.append(hr)  # Save hyper risk

        # Calculate ADRR as average of daily risks
        adrr = np.mean(daily_risks)

        # Determine risk category
        if adrr < 20:
            risk_category = "Low"
        elif adrr < 40:
            risk_category = "Moderate"
        else:
            risk_category = "High"

        # Calculate additional statistics
        hypo_risk = np.mean(daily_hypo_risks)
        hyper_risk = np.mean(daily_hyper_risks)

        return {
            "adrr": round(adrr, 2),
            "risk_category": risk_category,
            "components": {
                "hypo_risk": round(hypo_risk, 2),
                "hyper_risk": round(hyper_risk, 2),
            },
        }

    def calculate_variability_metrics(self) -> dict:
        try:
            # Basic metrics
            metrics = {
                "data_completeness": self.data_completeness(),
                "Mean": self.mean(),
                "Median": self.median(),
                "Std": self.sd(),
                "CV": self.cv(),
                "GMI": self.gmi(),
                # Time in Range
                "TIR": self.TIR(),
                "TIR_tight": self.TIR_tight(),
                "TIR_pregnancy": self.TIR_pregnancy(),
                "TAR180": self.TAR180(),
                "TAR250": self.TAR250(),
                "TAR140": self.TAR140(),
                "TBR70": self.TBR70(),
                "TBR63": self.TBR63(),
                "TBR55": self.TBR55(),
                # Distribution statistics
                "Skewness": float(self.data["glucose"].skew()),
                "Kurtosis": float(self.data["glucose"].kurtosis()),
            }

            # SD Variability metrics - these return dictionaries
            # SD Variability metrics - these return dictionaries
            sd_metrics = {
                "SDT": self.sd_total().get("sd"),
                "SDW": self.sd_within_day().get("sd"),
                "SD_timepoints": self.sd_between_timepoints().get("sd"),
                "SD_night": self.sd_segment("00:00", 8).get("sd"),
                "SD_day": self.sd_segment("08:00", 8).get("sd"),
                "SD_evening": self.sd_segment("16:00", 8).get("sd"),
                "SD_1h": self.sd_within_series(hours=1).get("sd"),
                "SD_6h": self.sd_within_series(hours=6).get("sd"),
                "SD_24h": self.sd_within_series(hours=24).get("sd"),
                "SD_daily_mean": self.sd_daily_mean().get("sd"),
                "SD_same_timepoint": self.sd_same_timepoint().get("sd"),
                "SD_same_timepoint_adj": self.sd_same_timepoint_adjusted().get("sd"),
                "SD_interaction": self.sd_interaction().get("sd"),
            }
            metrics.update(sd_metrics)

            # CONGA - returns dictionary
            conga_metrics = {
                "CONGA1": self.CONGA(hours=1).get("value"),
                "CONGA2": self.CONGA(hours=2).get("value"),
                "CONGA4": self.CONGA(hours=4).get("value"),
                "CONGA6": self.CONGA(hours=6).get("value"),
                "CONGA24": self.CONGA(hours=24).get("value"),
            }
            metrics.update(conga_metrics)

            # MAGE - returns dictionary
            try:
                mage_results = self.MAGE_Baghurst()
                metrics.update(
                    {
                        "mage_plus": mage_results.get("MAGE+"),
                        "mage_minus": mage_results.get("MAGE-"),
                        "mage_avg": mage_results.get("MAGE_avg"),
                        "mage_sd": mage_results.get("SD_used"),
                        "mage_threshold": mage_results.get("threshold"),
                        "mage_excursions": mage_results.get("num_excursions"),
                    }
                )
            except Exception as e:
                if self.log:
                    print(f"Error calculating MAGE: {e!s}")

            # MODD - returns dictionary
            try:
                modd_result = self.MODD()
                metrics.update(
                    {
                        "modd": modd_result.get("value"),
                        "modd_sd": modd_result.get("std"),
                    }
                )
            except Exception as e:
                if self.log:
                    print(f"Error calculating MODD: {e!s}")

            # Risk indices and others
            try:
                lgbi = self.LBGI()
                hbgi = self.HBGI()
                adrr = self.ADRR()
                gri = self.GRI()
                gri_pregnancy = self.GRI(pregnancy=True)
                grade = self.GRADE()
                m_value = self.M_Value()
                j_index = self.j_index()

                # Create dictionary with results, verifying the type of each value
                risk_metrics = {
                    "LBGI": lgbi,
                    "HBGI": hbgi,
                    "ADRR": adrr.get("adrr") if isinstance(adrr, dict) else adrr,
                    "GRI": gri.get("GRI") if isinstance(gri, dict) else gri,
                    "GRI_high": gri.get("derived_metrics", {}).get("hyper_component", 0),
                    "GRI_low": gri.get("derived_metrics", {}).get("hypo_component", 0),
                    "GRI_pregnancy": gri_pregnancy.get("GRI")
                    if isinstance(gri_pregnancy, dict)
                    else gri_pregnancy,
                    "GRI_pregnancy_high": gri_pregnancy.get("derived_metrics", {}).get(
                        "hyper_component", 0
                    ),
                    "GRI_pregnancy_low": gri_pregnancy.get("derived_metrics", {}).get(
                        "hypo_component", 0
                    ),
                    "GRADE": grade.get("total") if isinstance(grade, dict) else grade,
                    "M_Value": m_value if not isinstance(m_value, dict) else m_value.get("M_Value"),
                    "J_Index": j_index,
                }

                # Update the metrics dictionary
                metrics.update(risk_metrics)

            except Exception as e:
                print(f"Error general calculando métricas de riesgo: {e!s}")
                import traceback

                traceback.print_exc()

            return metrics
        except Exception as e:
            return {"error": str(e), "mensaje": "Error al calcular métricas"}

    def _get_segment_data(self, start_time: str, duration_hours: int) -> pd.DataFrame:
        """
        Helper method to get data for a specific time segment.

        :param start_time: Start time in "HH:MM" format
        :param duration_hours: Duration of the segment in hours
        :return: DataFrame with data within the specified segment
        """
        from datetime import datetime, time, timedelta

        # Convert start_time to time object
        start_hour, start_minute = map(int, start_time.split(":"))
        start = time(hour=start_hour, minute=start_minute)

        # Calculate end time (handling midnight crossing)
        base = datetime(2000, 1, 1, start.hour, start.minute)
        end_dt = base + timedelta(hours=duration_hours)
        end = end_dt.time()

        # Optimize filtering using vectorized integer comparisons
        # Object-based time comparisons (dt.time) are slow in pandas
        times = self.data["time"]
        minutes_from_midnight = times.dt.hour * 60 + times.dt.minute

        start_min = start.hour * 60 + start.minute
        # Handling end time that might be 24:00 (represented as 00:00 next day)
        end_min = end.hour * 60 + end.minute
        if end_min == 0 and duration_hours > 0:
            end_min = 1440

        if start_min < end_min:
            mask = (minutes_from_midnight >= start_min) & (minutes_from_midnight < end_min)
        else:
            # Crosses midnight (e.g. 22:00 to 06:00)
            mask = (minutes_from_midnight >= start_min) | (minutes_from_midnight < end_min)

        return self.data[mask].copy()
