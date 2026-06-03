# Que es el Makefile?

## Que es?

Un Makefile es una lista de atajos. En lugar de escribir:

```
python -m pytest tests/ -x --tb=short -q --cov=cgmpy --cov-report=term --cov-report=xml
```

escribes:

```
make test-coverage
```

Un Makefile te permite definir comandos con nombre (llamados "targets") para que no tengas que memorizar comandos largos y complejos.

## Por que lo usamos

No deberias necesitar recordar 30 comandos diferentes con sus banderas. El Makefile los recuerda por ti. Si quieres ejecutar los tests, escribes `make test`. Si quieres revisar el estilo del codigo, escribes `make lint`. Un comando, una tarea.

Tambien garantiza que todos los miembros del equipo ejecuten el **mismo** comando. Adios al "en mi maquina funciona" porque alguien uso banderas diferentes de pytest.

## Los targets mas utiles

| Comando | Que hace |
|---------|----------|
| `make test` | Ejecuta todos los tests |
| `make test-fast` | Tests rapidos (omite los lentos) |
| `make test-coverage` | Tests con informe de cobertura |
| `make lint` | Revisa el estilo del codigo con ruff |
| `make lint-fix` | Corrige automaticamente problemas de estilo |
| `make typecheck` | Ejecuta mypy (verificador de tipos) |
| `make security` | Ejecuta bandit (escaneo de seguridad) |
| `make docs-serve` | Previsualiza la documentacion localmente |
| `make build` | Genera el paquete de distribucion |

## Como se lee un Makefile

Un target se ve asi:

```makefile
.PHONY: test-fast
test-fast:          # nombre del target
    pytest -m "not slow" -q  # comando (identado con tabulador)
```

- `.PHONY` le indica a Make que esto no es un archivo real -- que siempre lo ejecute.
- `test-fast:` es el nombre que escribes despues de `make`.
- La linea siguiente es el comando a ejecutar. Debe estar identada con un **tabulador**, no con espacios.

## Por que es bueno para quienes aprenden

Puedes contribuir a CGMPy sin conocer todas las herramientas del stack. Quieres verificar que tu codigo tiene el formato correcto? Ejecuta `make lint`. Quieres asegurarte de que no has roto nada? Ejecuta `make test`. El Makefile es tu red de seguridad -- ejecuta exactamente las mismas comprobaciones que CI ejecutara, asi que detectas problemas antes de que lleguen a una pull request.
