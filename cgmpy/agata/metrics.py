try:
    from py_agata.py_agata import Agata
except ImportError:
    # No fallamos aquí, permitimos que el paquete se importe.
    # El error ocurrirá si se intenta usar la funcionalidad.
    Agata = None 

from .adapter import prepare_data_for_agata
from ..data.core import ModularGlucoseData

"""Comments
def time_in_given_range(data, th_l, th_h, include_th_l=False, include_th_h=False)

def time_in_target(data, glycemic_target='diabetes'):
    # By default, not include thresholds. I Change to True
    return time_in_given_range(data=data, th_l=th_l, th_h=th_h, include_th_l=True, include_th_h=True)

"""

def analyze_with_agata(
    glucose_data: ModularGlucoseData, 
    glycemic_target: str = 'diabetes', 
    **kwargs
) -> dict:
    """
    Analiza un objeto de datos de cgmpy usando la librería py_agata.
    (El resto de la función se mantiene igual...)
    """
    if Agata is None:
        raise ImportError("La librería 'py_agata' es necesaria para esta funcionalidad. Por favor, instálala.")
        
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
        print("Esto puede ocurrir si los datos de entrada tienen problemas que el adaptador no pudo resolver,")
        print("como grandes periodos de datos faltantes que no se pueden interpolar.")
        raise

# --- NUEVA CLASE ---
class AgataAnalysis(ModularGlucoseData):
    """
    Clase contenedora para analizar datos de glucosa utilizando la librería py_agata.

    Esta clase actúa como un puente orientado a objetos, permitiendo cargar datos
    y ejecutar el pipeline de análisis de py_agata de forma sencilla.
    """
    def run(self, glycemic_target: str = 'diabetes', **kwargs) -> dict:
        """
        Ejecuta el pipeline de análisis completo de py_agata sobre los datos cargados.

        Args:
            glycemic_target (str): El objetivo glicémico a usar ('diabetes', 'pregnancy', etc.).
                                   Se pasa directamente a la clase Agata de py_agata.
            **kwargs: Argumentos adicionales para el futuro.

        Returns:
            dict: El diccionario de resultados devuelto por py_agata.
        """
        # Llama a la función que ya has creado, pasándole la propia instancia.
        return analyze_with_agata(self, glycemic_target=glycemic_target, **kwargs)