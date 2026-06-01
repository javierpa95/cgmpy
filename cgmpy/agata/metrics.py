try:
    from py_agata.py_agata import Agata
except ImportError:
    # No fallamos aquí, permitimos que el paquete se importe.
    # El error ocurrirá si se intenta usar la funcionalidad.
    Agata = None

from ..data.core import ModularGlucoseData
from .adapter import prepare_data_for_agata

"""Comments
def time_in_given_range(data, th_l, th_h, include_th_l=False, include_th_h=False)

def time_in_target(data, glycemic_target='diabetes'):
    # By default, not include thresholds. I Change to True
    return time_in_given_range(data=data, th_l=th_l, th_h=th_h, include_th_l=True, include_th_h=True)

"""


def analyze_one_arm(
    data_list: list[ModularGlucoseData], glycemic_target: str = "diabetes", **kwargs
) -> dict:
    """
    Analiza un grupo (brazo) de datos de glucosa usando la librería py_agata.

    Args:
        data_list (list[ModularGlucoseData]): Lista de objetos ModularGlucoseData a analizar.
        glycemic_target (str): El objetivo glicémico a usar.

    Returns:
        dict: El diccionario de resultados devuelto por py_agata con estadísticas de grupo.
    """
    if Agata is None:
        raise ImportError(
            "La librería 'py_agata' es necesaria para esta funcionalidad. Por favor, instálala."
        )

    # 1. Preparar todos los DataFrames
    prepared_dfs = [prepare_data_for_agata(d) for d in data_list]

    try:
        # 2. Instanciar Agata y ejecutar análisis de grupo
        analyzer = Agata(glycemic_target=glycemic_target)
        results = analyzer.analyze_one_arm(prepared_dfs)

        return results

    except Exception as e:
        print(f"Error durante el análisis de grupo con py_agata: {e}")
        raise


def analyze_with_agata(
    glucose_data: ModularGlucoseData, glycemic_target: str = "diabetes", **kwargs
) -> dict:
    """
    Analiza un objeto de datos de cgmpy usando la librería py_agata.
    """
    if Agata is None:
        raise ImportError(
            "La librería 'py_agata' es necesaria para esta funcionalidad. Por favor, instálala."
        )

    # 1. Preparar los datos con el adaptador
    df_for_agata = prepare_data_for_agata(glucose_data)

    try:
        # 2. Instanciar Agata con el target correcto
        analyzer = Agata(glycemic_target=glycemic_target)

        # 3. Ejecutar el análisis
        results = analyzer.analyze_glucose_profile(df_for_agata)

        return results

    except Exception as e:
        print(f"Error durante el análisis con py_agata: {e}")
        raise


def summarize_agata_results(results: dict) -> dict:
    """
    Convierte el diccionario anidado de py_agata en un resumen plano (variable: valor),
    eliminando los arrays extensos de eventos y quedándose solo con los promedios.
    """
    summary = {}

    for category, metrics in results.items():
        if category == "events":
            # Para eventos, iteramos por tipo (hypo, hyper, etc) y nivel (l1, l2)
            for event_type, levels in metrics.items():
                if isinstance(levels, dict):
                    for level, data in levels.items():
                        if isinstance(data, dict):
                            # Construimos nombres claros: e.g. "hypo_l1_mean_duration"
                            prefix = f"{event_type}_{level}"
                            if "mean_duration" in data:
                                summary[f"{prefix}_mean_duration"] = data["mean_duration"]
                            if "events_per_week" in data:
                                summary[f"{prefix}_per_week"] = data["events_per_week"]
        else:
            # Para el resto, simplemente aplanamos un nivel
            if isinstance(metrics, dict):
                for name, value in metrics.items():
                    summary[f"{category}_{name}"] = value
            else:
                summary[category] = metrics

    return summary


# --- NUEVA CLASE ---
class AgataAnalysis(ModularGlucoseData):
    """
    Clase contenedora para analizar datos de glucosa utilizando la librería py_agata.

    Esta clase actúa como un puente orientado a objetos, permitiendo cargar datos
    y ejecutar el pipeline de análisis de py_agata de forma sencilla.
    """

    def __init__(self, *args, glycemic_target: str = "diabetes", **kwargs):
        """
        Inicializa el análisis Agata.

        Args:
            *args: Argumentos posicionales para ModularGlucoseData.
            glycemic_target (str): El objetivo glicémico ('diabetes' o 'pregnancy').
            **kwargs: Otros argumentos para ModularGlucoseData.
        """
        super().__init__(*args, **kwargs)
        self.glycemic_target = glycemic_target

    def run(self, glycemic_target: str = "diabetes", summary: bool = False, **kwargs) -> dict:
        """
        Ejecuta el pipeline de análisis completo de py_agata sobre los datos cargados.

        Args:
            glycemic_target (str): El objetivo glicémico a usar ('diabetes', 'pregnancy', etc.).
                                   Si es None, usa el valor definido al inicializar la clase.
            summary (bool): Si es True, devuelve un diccionario plano con métricas resumen.
                            Si es False, devuelve el diccionario completo anidado.
            **kwargs: Argumentos adicionales para el futuro.

        Returns:
            dict: El diccionario de resultados.
        """
        # Usar el target definido si no se pasa uno específico en la llamada
        target = glycemic_target or self.glycemic_target

        # 1. Obtener resultados crudos
        results = analyze_with_agata(self, glycemic_target=target, **kwargs)

        # 2. Resumir si se solicita
        if summary:
            return summarize_agata_results(results)

        return results

    @classmethod
    def analyze_one_arm(
        cls,
        data_list: list[ModularGlucoseData],
        glycemic_target: str = "diabetes",
        summary: bool = False,
        **kwargs,
    ) -> dict:
        """
        Analiza un grupo (brazo) de objetos ModularGlucoseData.

        Args:
            data_list (list[ModularGlucoseData]): Lista de objetos a analizar.
            glycemic_target (str): El objetivo glicémico a usar.
            summary (bool): Si es True, devuelve un resumen plano.
        """
        results = analyze_one_arm(data_list, glycemic_target=glycemic_target, **kwargs)

        if summary:
            # Nota: py_agata.analyze_one_arm suele devolver un formato similar
            return summarize_agata_results(results)

        return results
