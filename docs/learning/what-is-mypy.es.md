# Que es Mypy? (Type Hints)

## Que es?

A Python no le importa si le pasas el nombre de un paciente donde deberia ir un valor de glucosa. Fallara en tiempo de ejecucion. Mypy verifica esto ANTES de que el codigo se ejecute, mientras lo estas escribiendo.

Los type hints son anotaciones opcionales introducidas en Python 3.5 que indican a ti (y a mypy) que tipo debe tener una variable, parametro o valor de retorno. Mypy es un verificador estatico de tipos que lee estas anotaciones y encuentra errores antes de ejecutar una sola linea de codigo.

## Por que lo usamos

En una libreria de MCG, pasar `"paciente_123"` a `mean()` en lugar de una Serie de numeros deberia ser imposible. Mypy lo hace imposible.

Imagina que estas escribiendo un informe y accidentalmente pasas un string donde se espera un numero. Sin mypy, te enteras cuando el script falla a las 3 de la manana durante un proceso batch. Con mypy, ves una linea roja ondulada en tu editor en el momento que lo escribes.

## Type hints 101

```python
# Sin type hints (mypy no puede ayudar)
def gmi(m):
    return round(3.31 + 0.02392 * m, 2)

# Con type hints (mypy verifica esto)
def gmi(mean_glucose: float) -> float:
    return round(3.31 + 0.02392 * mean_glucose, 2)
```

La primera version es un misterio. Que es `m`? Un numero? Un string? Una Serie? Tienes que leer el cuerpo de la funcion o el docstring para adivinarlo. La segunda version te lo dice todo: toma un `float` (glucosa media en mg/dL) y devuelve un `float` (porcentaje de GMI).

## Ejemplo real en CGMPy

```python
from cgmpy.metrics.basic import mean

mean("100, 110, 120")  # mypy: Argument 1 has incompatible type "str"
mean(pd.Series([100, 110, 120]))  # mypy: OK
```

Mypy detecta esto en tiempo de verificacion de tipos, no en tiempo de ejecucion. Lo corriges antes de que el codigo se ejecute.

Tambien verifica los valores de retorno:

```python
def gmi(mean_glucose: float) -> float:
    return round(3.31 + 0.02392 * mean_glucose, 2)

result: str = gmi(120.0)  # mypy: Incompatible types in assignment (expected "str", got "float")
```

Mypy verifica tanto a quien llama como a quien implementa. Si declaras un tipo de retorno `float` pero devuelves un `str`, mypy te lo dice.

## Tipado gradual

CGMPy usa tipado gradual. No todo esta tipado aun. Empezamos por las funciones mas importantes primero.

```python
# Parcialmente tipado — mypy verifica inner(), pero no outer()
def outer(x):
    return inner(x)

def inner(x: float) -> float:
    return x * 2
```

Este enfoque permite anadir tipos de forma incremental sin bloquear el desarrollo. Una base de codigo parcialmente tipada sigue siendo mejor que una sin tipos: mypy verifica todo lo que puede ver.

## Como usarlo

```bash
mypy cgmpy/           # verifica la libreria
mypy cgmpy/ --strict  # aun mas estricto (objetivo futuro)
mypy cgmpy/metrics/variability.py  # verifica un solo archivo
```

Si estas anadiendo tipos a un archivo, ejecuta mypy solo en ese archivo primero para iterar rapidamente.

## Errores comunes

| Error | Significado |
|-------|-------------|
| `Incompatible return type` | La funcion devuelve algo diferente al tipo declarado |
| `Missing type parameters` | Necesitas especificar tipos en un generico (ej. `dict[str, float]` no solo `dict`) |
| `Argument 1 has incompatible type "str"; expected "float"` | Pasaste un string donde se esperaba un numero |
| `Cannot access attribute "glucose" for "Any"` | La variable no tiene tipo, mypy no puede verificar el acceso a atributos |

En computacion cientifica, `Missing type parameters` es el mas comun. En lugar de escribir `dict` como tipo de retorno, escribe `dict[str, float]` para indicar a mypy como son las claves y los valores.

## Por que es bueno para quienes aprenden

Los type hints son como ejes etiquetados en un grafico clinico. Te dicen exactamente que va donde. Cuando lees una firma de funcion:

```python
def tir(glucose: pd.Series, low: float = 70, high: float = 180) -> float: ...
```

Sabes de inmediato: las lecturas de glucosa entran, un porcentaje sale. Los umbrales `low` y `high` son numeros con valores predeterminados sensatos. No necesitas leer el docstring para entender el contrato.

Esto es especialmente valioso en una libreria como CGMPy, donde equivocarse con los tipos significa resultados clinicos incorrectos. Los type hints son tu primera linea de defensa contra errores tontos.
