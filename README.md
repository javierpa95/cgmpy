# CGMPy

Un paquete para analizar datos de glucosa.

## Instalación

pip install cgmpy


## Uso

```python
from cgmpy import GlucoseData

# Ejemplo de uso
g = GlucoseData('ruta/al/archivo.csv',"time","glucose")
print(g.mean())

```

# Resumen del Proyecto CGMPy

**CGMPy** es un paquete de Python diseñado para analizar datos de glucosa de manera eficiente y detallada. Este proyecto facilita el manejo y procesamiento de datos de glucosa, proporcionando diversas métricas y herramientas de visualización para usuarios en el ámbito de la salud y la investigación.

## Características Principales

- **Análisis Estadístico**:
  - Cálculo de medidas como media, mediana, desviación estándar y Glucose Management Index (GMI).
  - Evaluación de variabilidad mediante índices como CONGA, MODD y MAGE.
  
- **Tiempo en Rango (TIR)**:
  - Cálculo del tiempo que los niveles de glucosa se mantienen dentro de rangos específicos.
  - Indicadores como TAR (Tiempo Alto en Rango) y TBR (Tiempo Bajo en Rango) para evaluar riesgos de hiperglucemia e hipoglucemia.

- **Índices de Labilidad**:
  - Cálculo del índice de labilidad (LI) para evaluar la estabilidad de los niveles de glucosa en diferentes periodos (día, semana, mes).

- **Visualización de Datos**:
  - Generación de perfiles de glucosa ambulatoria (AGP) utilizando gráficos estadísticos que muestran percentiles y rangos intercuartiles.

- **Compatibilidad con Dispositivos Dexcom**:
  - Clase específica para manejar datos provenientes de dispositivos Dexcom, facilitando la integración y análisis de datos recogidos mediante estos dispositivos.

## Instalación

Puedes instalar CGMPy utilizando pip:

```bash
pip install cgmpy
```

## Uso Básico

```python
from cgmpy import GlucoseData

# Inicializar con el archivo CSV de datos de glucosa
g = GlucoseData('ruta/al/archivo.csv', "time", "glucose")

# Calcular la glucemia media
print(g.mean())
```

## Estructura del Proyecto

- **cgmpy/**: Módulo principal que contiene las clases `GlucoseData` y `Dexcom` para el análisis de datos.
- **setup.py**: Script de configuración para la instalación del paquete.
- **README.md**: Documentación básica del proyecto.
- **.gitignore**: Archivo para ignorar archivos específicos en el control de versiones.

## Dependencias

CGMPy requiere las siguientes bibliotecas de Python:

- numpy
- pandas
- matplotlib

Estas se instalan automáticamente durante la instalación del paquete.

## Contribuciones

Las contribuciones son bienvenidas. Puedes enviar issues o pull requests para mejorar el paquete y agregar nuevas funcionalidades.

## Autor

**Javier Peñate Arrieta**  
Correo electrónico: [javierpenatearrieta@gmail.com](mailto:javierpenatearrieta@gmail.com)

# License

Este proyecto está bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

# Contacto

Para cualquier consulta o sugerencia, por favor contacta al autor a través del correo electrónico proporcionado.

# Conclusión

CGMPy es una herramienta poderosa para el análisis de datos de glucosa, ofreciendo una variedad de métricas y visualizaciones que facilitan la comprensión y el seguimiento de los niveles de glucosa. Su diseño modular y extensible lo hace ideal tanto para profesionales de la salud como para investigadores en el campo.
