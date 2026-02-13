# 🩸 CGMPy: Continuous Glucose Monitoring Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Modular_Refactor-orange.svg" alt="Status">
</p>

**CGMPy** es una librería de Python potente y modular diseñada para el análisis de datos de **Monitoreo Continuo de Glucosa (CGM)**. Facilita la carga, el procesamiento y la visualización de datos, proporcionando métricas clínicas estándar y avanzadas de forma sencilla.

---

## 🏗️ Arquitectura Modular

El proyecto ha sido refactorizado recientemente siguiendo principios de diseño modular para mejorar la mantenibilidad y escalabilidad.

### 📂 Estructura del Proyecto

```text
cgmpy/
├── cgmpy/                  # 📦 Código fuente del paquete
│   ├── analysis/           # 🧠 Orquestación (Todo-en-uno: datos + métricas + plots)
│   │   └── core.py         # Clase GlucoseAnalysis (La forma más fácil de usarlo)
│   ├── data/               # 📥 Gestión de Datos (Módulo Central)
│   │   ├── core.py         # Clase ModularGlucoseData (corazón técnico)
│   │   ├── loader.py       # Cargadores (CSV, Parquet, DF)
│   │   ├── processor.py    # Limpieza y validación
│   │   └── specialized.py  # Dexcom, Libreview, etc.
│   ├── metrics/            # 📊 Lógica Clínica (Cálculos puros)
│   │   ├── basic.py        # Media, GMI, Mediana...
│   │   ├── time_in_range.py# TIR, TAR, TBR
│   │   └── variability.py  # CV, MAGE, MODD, CONGA...
│   ├── plotting/           # 📈 Visualizaciones Premium
│   │   ├── agp.py          # Perfil Ambulatorio (AGP)
│   │   └── daily_plots.py  # Tendencias diarias
│   ├── agata/              # 🤖 Integración con AGATA 
│   └── __init__.py         # 🚪 Fachada (API limpia)
├── examples/               # 💡 Proyectos de ejemplo
├── pyproject.toml          # ⚙️ Configuración (uv/pip)
└── README.md               # 📖 Esta guía
```

---

## 🚀 Inicio Rápido

CGMPy expone una "Fachada" (Facade) en su raíz para que no tengas que preocuparte por la estructura interna si solo quieres analizar tus datos rápidamente.

```python
from cgmpy import GlucoseAnalysis

# 1. Análisis completo con una sola clase
analysis = GlucoseAnalysis("datos_cgm.csv")

# 2. Obtener un reporte completo (JSON/Dict)
report = analysis.get_comprehensive_report()
print(f"TIR: {report['estadisticas_tiempo']['TIR']}%")

# 3. Generar un Dashboard visual
analysis.plot_comprehensive_dashboard()
```

---

## ✨ Características Principales

### 📊 Análisis Estadístico y Clínico
- **Métricas Estándar**: Media, Mediana, Desviación Estándar y GMI.
- **Variabilidad Glucémica**: Índices avanzados como **CV, CONGA, MODD, MAGE, J-Index, LBGI, HBGI**.
- **Tiempo en Rango (TIR)**: Cálculo preciso de TIR, TAR (hiper) y TBR (hipo) según guías internacionales.

### 📥 Carga de Datos Inteligente
- **Multi-dispositivo**: Soporte nativo y detección automática para **Dexcom Clarity**, **FreeStyle Libre (Libreview)**, Tandem y Medtronic.
- **Rendimiento**: Optimizado mediante el uso de **Apache Parquet** para el manejo eficiente de grandes volúmenes de datos.

### 📈 Visualización Avanzada
- Generación de **AGP (Ambulatory Glucose Profile)** estándar profesional.
- Gráficos diarios interactivos y análisis estadístico profundo.

---

## 🛠️ Desarrollo e Instalación

### Recomendado (usando uv o pip)
```bash
# Instalación local en modo desarrollo
pip install -e .
```

### Gestión de Dependencias
El proyecto utiliza `pyproject.toml` como única fuente de verdad para dependencias y configuración de herramientas como **Ruff** (linting) y **Pytest** (testing).

---

## 📝 Ejemplos Incluidos
En la carpeta `examples/` encontrarás datos reales (anonimizados) para probar el paquete:
- `dm.csv`: Datos de paciente con Diabetes Tipo 1.
- `nodm.csv`: Datos de sujeto sano (normoglucémico).
- `pregnancy.csv`: Análisis específico de diabetes gestacional.

---

## 👤 Autor
**Javier Peñate Arrieta**
📧 [javierpenatearrieta@gmail.com](mailto:javierpenatearrieta@gmail.com)

---

## 📜 Licencia
Este proyecto está bajo la licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.
