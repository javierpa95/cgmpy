# Ruta de aprendizaje: De clínico a desarrollador Python

## Introducción

Conoces los datos de CGM al dedillo. Tiempo en rango, MAGE, el perfil
glucémico ambulatorio (AGP) — te son tan familiares como tu propia
caligrafía. Puedes mirar un trazado de glucosa y detectar el fenómeno del
amanecer antes de que te tomes el café.

Ahora imagina construir las herramientas que calculan esos números. No solo
ejecutarlos en una hoja de cálculo, sino escribir el software que lo hace
de forma automática, reproducible y a escala. Para eso sirve esta ruta de
aprendizaje.

Cada paso enlaza a un artículo que enseña una herramienta, y luego te invita
a abrir el archivo de configuración real en CGMPy. La teoría se encuentra
con la práctica en cada esquina.

## La ruta

1. **Empieza aquí: what-is-pytest.md** — Escribe un test para la glucosa
   media, míralo pasar, luego rompe algo y observa cómo falla. No volverás
   a confiar en una métrica sin un test que la respalde.

2. **Dale estilo: what-is-ruff.md** — El código Python se puede escribir de
   cien formas distintas. Ruff elige una y la impone. Deja que la
   herramienta arregle el formato mientras tú te concentras en la ciencia.

3. **Atrapa errores pronto: what-is-mypy.md** — Las anotaciones de tipo
   evitan errores como pasar el nombre de un paciente donde debería ir un
   valor de glucosa. MyPy los detecta antes de ejecutar el código.

4. **Deja de repetirte: what-is-pre-commit.md** — Cuando tienes prisa, es
   fácil saltarse las comprobaciones de calidad. Pre-commit las automatiza
   antes de cada confirmación (commit) para que nunca se te olviden.

5. **Ponlo en producción: what-is-ci-cd.md** — La integración continua
   hace que cada solicitud de cambios ejecute tus tests automáticamente en
   GitHub. Se acabó eso de "en mi máquina funciona".

6. **Háblale a tu yo futuro: what-is-conventional-commits.md** — Un mensaje
   de commit como "arreglar cosas" no sirve de nada seis meses después.
   Los commits convencionales te obligan a ser claro. Tu yo del futuro te
   lo agradecerá.

7. **Un comando para gobernarlos a todos: what-is-makefile.md** — En lugar
   de recordar seis comandos distintos, un Makefile te da atajos:
   `make test`, `make lint`, `make docs`. Un comando, una tarea.

8. **Si no está documentado, no existe: what-is-mkdocs.md** — Esta web que
   estás leyendo está construida con MkDocs. Documentación que vive con el
   código y se actualiza sola.

9. **Versiona con intención: what-is-release-please.md** — Release Please
   automatiza los cambios de versión y los registros de cambios basándose
   en el historial de commits. Se acabó lo de "uy, ¿ya actualicé la
   versión?"

## La regla de oro

Lee el artículo, luego abre el proyecto y encuentra el archivo de
configuración real. Por ejemplo: lee sobre Ruff, luego abre `pyproject.toml`
y busca la sección `[tool.ruff]`. Lee sobre pre-commit, luego abre
`.pre-commit-config.yaml`. Todas las herramientas de esta ruta tienen un
archivo real en este repositorio. El artículo explica el *qué* y el *por
qué*; el archivo de configuración muestra el *cómo* para este proyecto
concreto.

Teoría sin práctica es solo cultura general. Práctica sin teoría es imitar
sin entender. Haz las dos cosas.
