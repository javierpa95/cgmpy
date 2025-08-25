# Módulo de Datos Refactorizado

## Visión General

El módulo `cgmpy.data` ha sido completamente refactorizado para proporcionar una arquitectura modular y mantenible para el manejo de datos de glucosa. La refactorización divide las responsabilidades en módulos especializados mientras mantiene la compatibilidad hacia atrás.

## Arquitectura Modular

### 🔧 Módulos Especializados

#### 1. `DataLoader` (`loader.py`)
- **Responsabilidad**: Carga de datos desde diferentes fuentes
- **Funcionalidades**:
  - Carga desde archivos CSV, Parquet y DataFrames
  - Detección automática de delimitadores
  - Manejo robusto de errores
  - Soporte para diferentes formatos de encabezados

#### 2. `DataProcessor` (`processor.py`)
- **Responsabilidad**: Procesamiento y validación de datos
- **Funcionalidades**:
  - Ruta optimizada para archivos Parquet
  - Validación de columnas
  - Conversión de tipos de datos
  - Manejo de duplicados
  - Limpieza de datos nulos
  - Filtrado por fechas

#### 3. `DataAnalyzer` (`analyzer.py`)
- **Responsabilidad**: Análisis básico de datos
- **Funcionalidades**:
  - Cálculo de intervalo típico
  - Información básica de los datos
  - Análisis de desconexiones
  - Métricas de calidad de datos
  - Generación de resúmenes

#### 4. `DataExporter` (`exporter.py`)
- **Responsabilidad**: Exportación de datos
- **Funcionalidades**:
  - Exportación a Parquet optimizado
  - Exportación a CSV y Excel
  - Anexado a archivos existentes
  - Manejo de duplicados
  - Optimización de tipos de datos

### 📊 Clase Principal: `ModularGlucoseData`

La clase `ModularGlucoseData` (`core.py`) integra todos los módulos especializados proporcionando una interfaz limpia y funcional:

```python
from cgmpy.data import ModularGlucoseData

# Crear instancia
data = ModularGlucoseData('datos.csv')

# Obtener información
info = data.info()
print(f"Datos: {info['num_datos']}")
print(f"Intervalo típico: {info['intervalo_tipico']:.1f} minutos")

# Filtrar datos
filtered = data.filter_by_date_range('2024-01-01', '2024-01-31')
glucose_filtered = data.filter_by_glucose_range(70, 180)

# Exportar
data.to_parquet('datos_optimizados.parquet')
data.to_csv('datos_exportados.csv')
```

## Clases Especializadas por Dispositivo

### 🔬 Dispositivos Soportados

#### 1. `Dexcom` - Dexcom Clarity
```python
from cgmpy.data import Dexcom
dexcom = Dexcom('datos_dexcom.csv')
```

#### 2. `Libreview` - FreeStyle Libre
```python
from cgmpy.data import Libreview
libreview = Libreview('datos_libreview.csv', header=2)
```

#### 3. `MedtronicCarelink` - Medtronic CareLink
```python
from cgmpy.data import MedtronicCarelink
medtronic = MedtronicCarelink('datos_medtronic.csv')
```

#### 4. `TandemDiabetes` - Tandem Diabetes
```python
from cgmpy.data import TandemDiabetes
tandem = TandemDiabetes('datos_tandem.csv')
```

### 🤖 Detección Automática

```python
from cgmpy.data import detect_device_type, create_specialized_loader

# Detectar tipo de dispositivo
device_type = detect_device_type('datos.csv')
print(f"Dispositivo detectado: {device_type}")

# Crear cargador apropiado automáticamente
loader = create_specialized_loader('datos.csv')
```

## Compatibilidad Hacia Atrás

La refactorización mantiene **100% de compatibilidad hacia atrás**:

```python
# Código existente sigue funcionando
from cgmpy import GlucoseData, Dexcom, Libreview

# GlucoseData ahora es un alias a ModularGlucoseData
data = GlucoseData('datos.csv')
info = data.info()
```

## Beneficios de la Refactorización

### 🎯 Principio de Responsabilidad Única
- Cada módulo tiene una responsabilidad específica
- Fácil mantenimiento y testing
- Código más limpio y comprensible

### 📈 Rendimiento Mejorado
- Ruta optimizada para archivos Parquet
- Procesamiento eficiente de datos grandes
- Cálculos optimizados con NumPy

### 🔧 Extensibilidad
- Fácil agregar nuevos tipos de dispositivos
- Módulos independientes y reutilizables
- Arquitectura preparada para futuras mejoras

### 🧪 Testabilidad
- Cada módulo puede probarse independientemente
- Separación clara de responsabilidades
- Mocking más fácil para pruebas unitarias

## Ejemplos de Uso

### Uso Básico
```python
from cgmpy.data import ModularGlucoseData

# Cargar datos
data = ModularGlucoseData('datos.csv', log=True)

# Obtener información
print(data)
info = data.info(include_disconnections=True)

# Acceder a datos
raw_data = data.get_raw_data()
glucose_values = data.get_glucose_values()
timestamps = data.get_timestamps()
```

### Uso Avanzado
```python
from cgmpy.data import DataLoader, DataProcessor, DataAnalyzer

# Uso modular directo
loader = DataLoader()
processor = DataProcessor()
analyzer = DataAnalyzer()

# Procesar paso a paso
raw_data = loader.load_from_source('datos.csv', 'time', 'glucose')
processed_data, time_diffs = processor.process_data(raw_data, 'time', 'glucose')
typical_interval = analyzer.calculate_typical_interval(time_diffs)
```

### Filtrado y Análisis
```python
# Filtrar por fecha
january_data = data.filter_by_date_range('2024-01-01', '2024-01-31')

# Filtrar por glucosa
normal_range = data.filter_by_glucose_range(70, 180)

# Métricas de calidad
quality = data.get_data_quality_metrics()
print(f"Gaps detectados: {quality['total_gaps']}")
print(f"Máximo gap: {quality['max_gap_hours']:.1f} horas")
```

## Migración desde Versión Anterior

### ✅ Sin Cambios Necesarios
El código existente sigue funcionando sin modificaciones.

### 🆕 Para Aprovechar Nuevas Funcionalidades
```python
# Antes
from cgmpy import GlucoseData
data = GlucoseData('datos.csv')

# Ahora (opcional)
from cgmpy.data import ModularGlucoseData
data = ModularGlucoseData('datos.csv')

# Nuevas funcionalidades
filtered = data.filter_by_glucose_range(80, 180)
quality = data.get_data_quality_metrics()
```

## Logging y Rendimiento

```python
# Activar logging detallado
data = ModularGlucoseData('datos.csv', log=True)

# El logging muestra:
# - Tiempos de carga
# - Operaciones de procesamiento
# - Optimizaciones aplicadas
# - Uso de memoria
```

## Próximos Pasos

La refactorización del módulo de datos sienta las bases para:

1. **Refactorización de métricas** - Dividir `glucose_metrics.py`
2. **Refactorización de plotting** - Dividir `glucose_plot.py`
3. **Módulos de análisis** - Mover `glucose_pregnacy.py`
4. **Tests unitarios** - Crear suite de pruebas completa
5. **Documentación** - Completar documentación técnica

---

*Esta refactorización reduce el archivo original de 791 líneas a 6 módulos especializados, mejorando significativamente la mantenibilidad y extensibilidad del código.* 