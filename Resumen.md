# CGMPY: Biblioteca para Procesamiento de Datos de Glucosa

## Descripción General

CGMPY es una biblioteca Python diseñada para procesar, analizar y optimizar datos de monitorización continua de glucosa (CGM) provenientes de diferentes dispositivos médicos. La biblioteca está optimizada para manejar grandes volúmenes de datos de manera eficiente, con soporte para formatos CSV y Parquet.

## Nuevas Características y Optimizaciones

Esta versión de CGMPY incluye importantes mejoras en rendimiento y funcionalidad:

- **Soporte para formato Parquet**: Implementación de carga y guardado optimizado en formato Parquet, reduciendo significativamente los tiempos de procesamiento y el tamaño de los archivos.
- **Rutas de procesamiento optimizadas**: Detección automática del origen de datos para aplicar la ruta de procesamiento más eficiente.
- **Manejo avanzado de archivos grandes**: Optimizaciones específicas para archivos de más de 10MB.
- **Detección automática de formatos**: Identificación inteligente de delimitadores, formatos de fecha y tipos de datos.
- **Soporte para Libreview**: Nueva clase especializada para procesar datos de dispositivos Freestyle Libre.
- **Análisis de desconexiones**: Funcionalidad para detectar y cuantificar períodos sin datos.
- **Medición de rendimiento**: Sistema integrado de logging para analizar tiempos de procesamiento.

## Características Principales

- **Soporte para múltiples formatos**: Procesa archivos CSV y Parquet con detección automática.
- **Optimización de rendimiento**: Implementa rutas de procesamiento optimizadas según el origen de los datos.
- **Detección automática**: Identifica delimitadores, formatos de fecha y tipos de datos.
- **Manejo inteligente de datos**: Gestiona valores nulos, duplicados y ordenación temporal.
- **Análisis de desconexiones**: Detecta y cuantifica períodos sin datos.
- **Conversión eficiente**: Permite convertir datos entre formatos con optimizaciones.
- **Soporte para dispositivos específicos**: Incluye clases especializadas para Dexcom y Libreview.

## Estructura del Código

### Clase Principal: `GlucoseData`

La clase base que implementa toda la funcionalidad para procesar datos de glucosa:

- **Inicialización flexible**: Acepta archivos CSV/Parquet o DataFrames de pandas.
- **Procesamiento optimizado**: Detecta automáticamente el formato y aplica la ruta de procesamiento más eficiente.
- **Análisis de datos**: Calcula estadísticas como intervalo típico entre mediciones y períodos de desconexión.
- **Exportación optimizada**: Permite guardar datos en formato Parquet para acceso rápido.

### Clases Especializadas

- **`Dexcom`**: Clase adaptada para procesar datos de dispositivos Dexcom con configuración predefinida.
- **`Libreview`**: Clase adaptada para procesar datos de dispositivos Freestyle Libre con configuración predefinida.

## Funcionalidades Destacadas

### Carga Inteligente de Datos

- Detección automática de formato (CSV o Parquet)
- Identificación de delimitadores en archivos CSV
- Optimización de tipos de datos para rendimiento
- Manejo eficiente de archivos grandes (>10MB)

### Procesamiento Optimizado

- Rutas de procesamiento diferenciadas según el origen de datos
- Conversión eficiente de tipos de datos
- Eliminación inteligente de duplicados
- Ordenación temporal de registros

### Análisis de Datos

- Cálculo del intervalo típico entre mediciones
- Detección de períodos de desconexión
- Estadísticas de disponibilidad de datos
- Información detallada sobre el uso de memoria

### Exportación y Persistencia

- Guardado optimizado en formato Parquet
- Compresión configurable de archivos
- Adición de nuevos datos a archivos existentes
- Estrategias configurables para manejar duplicados

## Ejemplo de Uso

```python
# Cargar datos de Dexcom
dexcom_data = Dexcom("datos_dexcom.csv", start_date="2023-01-01", end_date="2023-01-31")

# Ver información básica
print(dexcom_data)

# Obtener estadísticas detalladas incluyendo desconexiones
info_detallada = dexcom_data.info(include_disconnections=True)

# Guardar en formato optimizado
dexcom_data.to_parquet("datos_optimizados.parquet", compression="snappy")

# Cargar datos de Libreview
libre_data = Libreview("datos_libre.csv", header=2)

# Añadir nuevos datos a un archivo Parquet existente
libre_data.append_to_parquet("datos_combinados.parquet", handle_duplicates="keep_new")
```

## Optimizaciones de Rendimiento

La biblioteca implementa numerosas optimizaciones para mejorar el rendimiento:

- **Uso de NumPy para cálculos vectorizados**: Implementación de operaciones numéricas optimizadas.
- **Detección y aplicación de tipos de datos óptimos**: Conversión automática a tipos de datos eficientes (int16, float32).
- **Procesamiento diferenciado según el origen de los datos**: Rutas optimizadas para Parquet vs CSV.
- **Carga selectiva de columnas necesarias**: Lectura eficiente de solo los datos requeridos.
- **Medición y registro de tiempos de procesamiento**: Sistema de logging detallado para análisis de rendimiento.
- **Manejo optimizado de memoria**: Técnicas para reducir el consumo de memoria en grandes conjuntos de datos.

## Requisitos

- Python 3.6+
- pandas
- numpy
- pyarrow (para soporte de Parquet)
- matplotlib (para visualizaciones)
- seaborn (para visualizaciones avanzadas)

## Comparación de Rendimiento

| Operación | Formato CSV | Formato Parquet | Mejora |
|-----------|-------------|----------------|--------|
| Carga de archivo (100MB) | ~10-15s | ~1-2s | 5-10x |
| Procesamiento | ~5-8s | ~0.5-1s | 5-8x |
| Tamaño de archivo | 100% | 30-40% | 60-70% |
| Uso de memoria | 100% | 50-60% | 40-50% |

## Conclusión

Esta versión optimizada de CGMPY representa un avance significativo en el procesamiento de datos de monitorización continua de glucosa, ofreciendo un rendimiento superior y nuevas funcionalidades que facilitan el análisis de grandes volúmenes de datos. Las optimizaciones implementadas permiten trabajar con conjuntos de datos más grandes de manera más eficiente, mientras que las nuevas características de análisis proporcionan información valiosa sobre la calidad y disponibilidad de los datos. 