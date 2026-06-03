# Que es Pytest?

## Que es?

Un ejecutor de pruebas. Escribes una funcion que verifica: "si llamo a `mean(Series[100, 110, 120])`, obtengo 110?" Pytest ejecuta esa comprobacion y te dice si paso o fallo.

Mas formalmente, pytest es un framework de testing para Python. Descubre y ejecuta funciones de prueba, recopila resultados y te da un informe claro de que paso, que fallo y por que.

## Por que lo usamos

Cuando cambias la formula de GMI para corregir un error, como sabes que no has roto TIR? Ejecuta las pruebas. Si pasan, estas seguro.

Las pruebas manuales son lentas, propensas a errores y no escalan. A medida que CGMPy crece, el numero de cosas que pueden romperse crece con el. Las pruebas automatizadas detectan regresiones al instante. Haces un cambio, ejecutas `pytest`, y obtienes una luz verde (o una roja que te dice exactamente que rompiste).

## Escribir una prueba

```python
# En tests/unit/test_metrics/test_basic_functions.py
from cgmpy.metrics.basic import mean
import pandas as pd

def test_mean_of_three_values():
    glucose = pd.Series([100.0, 110.0, 120.0])
    assert mean(glucose) == 110.0
```

Eso es todo. Una funcion llamada `test_...` con una sentencia `assert`. Pytest la encuentra automaticamente, la ejecuta e informa si la asercion pasa o falla.

Una prueba fallida se ve asi:

```python
def test_mean_of_empty_series():
    glucose = pd.Series([], dtype=float)
    assert mean(glucose) == 0.0  # Que deberia pasar? NaN? 0?
```

Esto resalta algo importante: las pruebas te obligan a decidir cual es el comportamiento correcto para los casos limite. Escribir una prueba para una entrada vacia te hace pensar en lo que tu funcion deberia hacer en cada escenario.

## Estructura de pruebas en CGMPy

```
tests/
├── unit/          # Pruebas rapidas (sin red, sin disco)
│   ├── test_data/     # Pruebas de cargadores, procesadores
│   ├── test_metrics/  # Pruebas de calculo de metricas
│   └── test_plotting/ # Pruebas de graficos (usando backend Agg)
├── integration/   # Pruebas que combinan modulos
├── clinical/      # Pruebas contra referencias publicadas
└── fixtures/      # Datos de prueba (CSV sinteticos)
```

- **Unit tests** son rapidos y aislados. Prueban una unica funcion o modulo.
- **Integration tests** verifican que los modulos funcionen juntos correctamente.
- **Clinical tests** comparan la salida de CGMPy contra valores de referencia publicados. Si un articulo dice "para esta entrada, el resultado es X", escribimos una prueba que verifica eso.

## Fixtures — datos de prueba reutilizables

```python
# En tests/conftest.py
import pytest
import pandas as pd

@pytest.fixture
def stable_glucose_df():
    """Devuelve un DataFrame con glucosa constante de 100 mg/dL."""
    return pd.DataFrame({"glucose": [100.0] * 288})  # 24h a intervalos de 5 min

@pytest.fixture
def variable_glucose_series():
    """Devuelve una Serie con variacion tipica diaria."""
    return pd.Series(
        [100.0] * 48 +     # noche: 4 horas a 100
        [120.0] * 36 +     # subida matutina
        [90.0] * 24 +      # bajada de comida
        [140.0] * 36 +     # tarde
        [110.0] * 48 +     # noche
        [100.0] * 96       # madrugada
    )
```

Los fixtures te permiten definir datos de prueba una vez y reutilizarlos en multiples pruebas. Pueden ser tan simples o complejos como necesites. Pytest maneja la configuracion y limpieza automaticamente.

Usar un fixture en una prueba es solo un parametro de nombre:

```python
def test_mean_of_stable_glucose(stable_glucose_df):
    result = mean(stable_glucose_df["glucose"])
    assert result == 100.0
```

## Ejecutar pruebas

```bash
pytest                          # todas las pruebas
pytest -m "not slow"            # prueba rapida (humo)
pytest tests/unit/test_metrics/ # carpeta especifica
pytest -k "test_mean"           # por nombre
```

Los marcadores como `slow` te permiten categorizar pruebas. Ejecuta la suite completa antes de un lanzamiento, o solo las rapidas mientras iteres.

## Que es la cobertura?

La cobertura te dice que porcentaje de tu codigo es ejercitado por las pruebas. CGMPy apunta a 80%+. Piensa en ello como: "cada linea de este software medico realmente se prueba?"

```bash
pytest --cov=cgmpy --cov-report=term-missing
```

Esto ejecuta las pruebas y te muestra que lineas de CGMPy nunca se ejecutaron. Las lineas en rojo no estan probadas — son posibles escondites de errores. La cobertura no garantiza correccion (puedes tener 100% de cobertura y aun tener errores), pero una cobertura baja garantiza que gran parte del codigo nunca se ha ejecutado en una prueba.

## Por que es bueno para quienes aprenden

Las pruebas son la mejor documentacion. Leer una prueba para `MAGE()` te dice exactamente lo que la funcion deberia hacer. Te muestra:

- Que entradas espera
- Que salida produce
- Como se manejan los casos limite
- Cual es el contrato real de la funcion

Una suite de pruebas es una especificacion ejecutable. Cuando el codigo y la documentacion discrepan, las pruebas suelen tener razon (porque realmente se ejecutan y fallan si estan equivocadas).

Si eres nuevo en la base de codigo, empieza leyendo las pruebas. Te ensenaran como se supone que debe comportarse cada funcion.
