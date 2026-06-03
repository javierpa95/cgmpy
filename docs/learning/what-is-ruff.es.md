# Que es Ruff? (Lint + Formato)

## Que es?

Ruff es dos herramientas en una: un **linter** (encuentra problemas en tu
codigo) y un **formateador** (arregla el aspecto de tu codigo).

Imagina a un revisor de informes clinicos que te dice: "Pon los valores en
tablas, no en parrafos. Las unidades entre parentesis. Las referencias al
final." Ruff es ese revisor para el codigo Python, pero ademas puede
corregir muchos de los problemas que encuentra, automaticamente.

## Lint vs Formato

Estos dos conceptos suelen mencionarse juntos, pero hacen cosas distintas.

**Lint** encuentra errores y malas practicas antes de que causen problemas.
- Una variable sin usar que indica codigo muerto.
- Una comparacion peligrosa como `is` en lugar de `==` para cadenas.
- Una importacion que no existe (error tipografico en el nombre del modulo).
- Una importacion comodin que contamina el espacio de nombres.

**Formato** hace que el codigo tenga un aspecto consistente, sin impacto
semantico.
- Sangrado consistente (espacios, no tabulaciones).
- Saltos de linea antes o despues de operadores.
- Comillas simples o dobles (Ruff las normaliza).
- Lineas en blanco entre funciones.

Un error de lint puede indicar un error real. Un problema de formato es
cosmetico, pero la consistencia estetica importa cuando diez personas
colaboran en el mismo archivo.

## Ejemplo real en CGMPy

```python
# MAL — ruff lo detectaria
from cgmpy import *   # importacion comodin — mala practica

# BIEN
from cgmpy import GlucoseData  # importacion explicita
```

Las importaciones comodin hacen imposible saber de donde viene un nombre.
Ruff las marca con la regla F403 y sugiere importaciones explicitas.

```python
# MAL — ruff lo detectaria
def calculate_mean(values):
    total = sum(values)
    count = len(values)
    result = total / count
    return result

# BIEN (count se usa, pero si no, ruff lo marcaria como F841)
def calculate_mean(values):
    return sum(values) / len(values)
```

## Por que usamos Ruff en lugar de Black o Flake8

Historicamente, los proyectos Python usaban Flake8 para linting y Black
para formateo, ademas de isort para ordenar importaciones, y media docena
de herramientas mas. Eso significaba:
- Cinco archivos de configuracion distintos que mantener.
- Ejecucion lenta porque cada herramienta cargaba tu codigo desde el disco
  por separado.
- Conflictos ocasionales entre lo que queria una herramienta y lo que
  imponia otra.

Ruff las reemplaza todas. Es de 10 a 100 veces mas rapido porque esta
escrito en Rust, tiene correccion automatica integrada para muchas reglas,
y se configura en un solo sitio: la seccion `[tool.ruff]` de
`pyproject.toml`.

| Herramienta | Reemplazada por Ruff |
|---|---|
| Flake8 | `ruff check` |
| Black | `ruff format` |
| isort | `ruff check --fix` (regla I001) |
| autoflake | `ruff check --fix` (regla F841 y otras) |
| pyupgrade | `ruff check --fix` (reglas UP...) |

## Como usarlo

```bash
# Encuentra todos los problemas en el directorio actual
ruff check .

# Encuentra problemas y corrige lo que pueda automaticamente
ruff check . --fix

# Formatea todos los archivos Python
ruff format .

# Verifica el formato sin modificar archivos (para CI)
ruff format --check .
```

En CGMPy, rara vez necesitas ejecutar Ruff manualmente. Pre-commit lo
ejecuta por ti en cada commit. Pero cuando lo ejecutas directamente, estos
son los comandos.

## Errores comunes y que significan

| Codigo | Significado | Solucion |
|---|---|---|
| F401 | Modulo o nombre importado pero nunca usado | Elimina la importacion o usala |
| F841 | Variable asignada pero nunca usada | Elimina la asignacion o usa la variable |
| E501 | Linea demasiado larga (mas de 100 caracteres) | Divide la linea en varias |
| I001 | Importaciones en orden incorrecto | `ruff check --fix` las ordena automaticamente |
| F403 | Importacion comodin (`from x import *`) | Sustituye por importaciones explicitas |
| W291 | Espacio sobrante al final de la linea | `ruff check --fix` lo elimina |
| N802 | El nombre de la funcion deberia ser minusculas | Renombra la funcion a snake_case |
| D100 | Falta docstring en modulo publico | Anade un docstring a nivel de modulo |

Los codigos de regla indican la categoria: `F` viene de pyflakes (errores
de logica), `E` y `W` de pycodestyle (formato), `I` de isort
(importaciones), `N` de pep8-naming (convenciones de nombres), `D` de
pydocstyle (docstrings).

## Por que es util para quienes aprenden

Aprendes el estilo de Python corrigiendo los problemas que Ruff encuentra.
Es como un profesor paciente que nunca se cansa de explicar. Escribes una
linea larga y Ruff te dice que la dividas. Usas un nombre de variable
vago y Ruff te sugiere uno mejor. Cada aviso es una leccion.
