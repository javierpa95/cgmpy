# Que es Pre-commit?

## Que es?

Imagina una lista de verificacion que se ejecuta automaticamente antes de
cada `git commit`. Comprueba: espacios al final de las lineas? YAML
valido? Codigo formateado? El mensaje del commit sigue la convencion
correcta? Si alguna comprobacion falla, el commit se bloquea hasta que lo
arregles.

Esa lista es pre-commit. Es un entorno que ejecuta una serie de hooks
(pequenos programas que inspeccionan tu codigo) cada vez que escribes
`git commit`. Si un hook encuentra algo mal, lo arregla automaticamente o
te dice que tienes que corregir.

## Por que lo usamos

Diez personas escribiendo codigo Python de diez formas distintas es el
caos. Una persona olvida ejecutar el linter, otra usa tabulaciones en vez
de espacios, una tercera escribe un mensaje de commit que solo dice "wip".
Con el tiempo, el codigo se vuelve inconsistente y dificil de mantener.

Pre-commit impone la consistencia sin que nadie tenga que acordarse de
hacerlo. Las comprobaciones ocurren automaticamente, siempre. Sin fuerza
de voluntad.

## Los hooks que usamos

CGMPy utiliza los siguientes hooks de pre-commit. La configuracion
completa esta en `.pre-commit-config.yaml` en la raiz del proyecto.

| Hook | Que comprueba | Que pasa si falla |
|---|---|---|
| `trailing-whitespace` | Elimina espacios al final de lineas | Correccion automatica |
| `end-of-file-fixer` | Asegura que todo archivo termine con salto de linea | Correccion automatica |
| `check-yaml` | Valida la sintaxis YAML | Bloquea el commit |
| `check-toml` | Valida la sintaxis TOML | Bloquea el commit |
| `check-json` | Valida la sintaxis JSON | Bloquea el commit |
| `check-added-large-files` | Bloquea archivos de mas de 5 MB | Bloquea el commit |
| `check-merge-conflict` | Detecta marcadores de fusion residuales | Bloquea el commit |
| `check-case-conflict` | Bloquea nombres que solo difieren en mayusculas/minusculas | Bloquea el commit |
| `mixed-line-ending` | Fuerza saltos de linea LF | Correccion automatica |
| `detect-private-key` | Bloquea commits accidentales de claves SSH | Bloquea el commit |
| `ruff` (lint) | Encuentra violaciones de estilo y errores | Muestra errores, algunos con correccion automatica |
| `ruff-format` | Asegura que el formato del codigo sea consistente | Correccion automatica |
| `interrogate` | Verifica la cobertura de docstrings (minimo 70%) | Bloquea el commit, muestra la cobertura faltante |
| `commitlint` | Valida que el mensaje del commit siga Conventional Commits | Debes reescribir el mensaje |
| `cgmpy-docs-sync` | Verifica que la documentacion este sincronizada con el codigo | Bloquea el commit con instrucciones |

## Como usarlo

```bash
# Ejecuta las comprobaciones en los archivos preparados (mas rapido)
pre-commit run

# Ejecuta las comprobaciones en todos los archivos del repositorio
pre-commit run --all-files

# Instala los hooks para que se ejecuten automaticamente al hacer commit
pre-commit install

# Omision de emergencia — usalo con moderacion
git commit --no-verify
```

La opcion `--no-verify` es como una exencion de protocolo clinico: existe
para emergencias genuinas, pero si la usas a diario, algo esta mal.

## Ejemplo real en CGMPy

Anades una nueva funcion de metrica — por ejemplo, un indice de
variabilidad de glucosa personalizado. Escribes el codigo, lo pruebas y
funciona. Preparas el archivo y ejecutas `git commit`. Pre-commit ejecuta
interrogate, que verifica que toda funcion publica tenga un docstring. Tu
nueva funcion no lo tiene. El commit se bloquea, e interrogate te dice
exactamente que funcion carece de documentacion. Anades el docstring,
preparas de nuevo, y esta vez el commit se completa.

Sin pre-commit, ese docstring faltante habria llegado al repositorio y
se habria quedado alli hasta que alguien se diera cuenta. Con pre-commit,
se detecto antes de que llegara al codigo.

## Por que es util para quienes aprenden

No necesitas memorizar cada regla de estilo o convencion. Pre-commit te
ensena sobre la marcha. Cada vez que bloquea un commit, aprendes algo:
"Ah, necesito un docstring aqui", o "Claro, el mensaje del commit necesita
un prefijo de tipo". La herramienta es tambien el profesor.
