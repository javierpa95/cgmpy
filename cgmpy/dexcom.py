from .glucose_data import GlucoseData

class Dexcom(GlucoseData):
    def __init__(self, file_path):
        """
        Inicializa la clase Dexcom con los datos de glucemia a partir de un archivo CSV.
        
        :param file_path: Ruta al archivo CSV.
        :param delimiter: Delimitador usado en el archivo CSV (por defecto ',').
        """
        # Llama al constructor de la clase base con las columnas específicas de Dexcom
        super().__init__(file_path, date_col="Marca temporal (AAAA-MM-DDThh:mm:ss)", glucose_col="Nivel de glucosa (mg/dl)")

