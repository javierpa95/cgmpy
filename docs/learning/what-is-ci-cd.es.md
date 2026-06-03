# Que es CI/CD? (GitHub Actions)

## Que es?

CI/CD significa: cada vez que subes codigo a GitHub, un robot lo verifica. Ejecuta pruebas, revisa el formato, escanea en busca de problemas de seguridad. Si algo falla, el robot te avisa antes de que el codigo llegue a tus usuarios.

- **CI** significa Integracion Continua. Cada cambio de codigo se integra, compila y prueba automaticamente. Si alguien sube codigo roto, lo sabes en minutos.
- **CD** significa Despliegue Continuo (o Entrega Continua). Una vez que las pruebas pasan, el codigo se empaqueta y despliega automaticamente — o en el caso de CGMPy, se publica en PyPI.

## Por que lo usamos

Las pruebas manuales son lentas y propensas a errores. CI/CD hace que la calidad sea automatica. Sube codigo, obten resultados en 5 minutos.

Antes de CI/CD, el flujo de trabajo era: escribir codigo, ejecutar pruebas localmente (si te acordabas), cruzar los dedos, subir a produccion. Con CI/CD, cada subida ejecuta las mismas comprobaciones en una maquina limpia. No hay sorpresas de "funciona en mi maquina".

## El pipeline de CI de CGMPy

Esto es lo que sucede cuando subes a main:

```
1. Lint -> ruff verifica el estilo del codigo
2. Type check -> mypy verifica los tipos
3. Seguridad -> bandit escanea vulnerabilidades
4. Test (3 SO x 3 versiones Python) -> pytest en Linux, Windows, macOS
5. Cobertura -> calcula la cobertura de pruebas, la sube a Codecov
6. Docs -> compila y despliega el sitio mkdocs
7. CodeQL -> analisis de seguridad de GitHub
```

Cada paso es una puerta. Si alguna puerta falla, el pipeline se detiene y el PR recibe una cruz roja. El revisor (o el autor) sabe exactamente que salio mal sin ejecutar nada localmente.

### Que hace cada paso

1. **Lint (ruff):** Aplica un estilo de codigo consistente. Sin tabs donde deberia haber espacios, sin imports no utilizados. Es como un corrector ortografico para codigo.
2. **Type check (mypy):** Detecta errores de tipo como se describe en el articulo de mypy. Garantiza que `mean()` nunca reciba un string.
3. **Seguridad (bandit):** Busca contrasenas hardcodeadas, llamadas inseguras a `eval()`, patrones de inyeccion SQL. En una libreria de datos medicos, esto no es negociable.
4. **Test (3 SO x 3 versiones Python):** Ejecuta la suite completa de pruebas en Linux, Windows y macOS, con Python 3.10, 3.11 y 3.12. Un error en Windows se detecta antes de que llegue a un usuario.
5. **Cobertura:** Mide que porcentaje de la base de codigo es ejercitado por las pruebas.
6. **Docs:** Compila el sitio mkdocs y lo despliega. Si la documentacion tiene un enlace roto o sintaxis invalida, te enteras aqui.
7. **CodeQL:** El analisis de seguridad propio de GitHub. Busca patrones que podrian llevar a vulnerabilidades.

## Que es un workflow?

Un archivo YAML en `.github/workflows/` que le dice a GitHub que hacer. Cada workflow es un conjunto de trabajos que se ejecutan en desencadenantes especificos (push, PR, programacion).

Aqui hay una version simplificada de un workflow de CI:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest
      - run: mypy cgmpy
```

Que hace esto:

- **`on:`** — cuando se ejecuta este workflow. Aqui, en pushes y PRs a main.
- **`jobs:`** — que ejecutar. Definimos un trabajo `test`.
- **`strategy.matrix:`** — ejecuta el mismo trabajo con diferentes parametros. Aqui, tres versiones de Python.
- **`steps:`** — los comandos reales. Clonar el codigo, instalar Python, instalar dependencias, ejecutar pruebas, ejecutar mypy.

GitHub Actions proporciona la maquina virtual (runs-on). No necesitas configurar nada localmente. El archivo workflow es la configuracion completa, comprometida en el repositorio.

## Escaneo de seguridad

Bandit busca:

- Contrasenas o claves API hardcodeadas
- Vulnerabilidades de inyeccion SQL
- Uso inseguro de `eval()`, `exec()` o `pickle.loads()`
- Uso de `assert` en codigo de produccion (se puede desactivar con `-O`)
- Llamadas a funciones vulnerables conocidas

En una libreria de datos medicos, ejecutar bandit en cada subida no es negociable. Si alguien compromete accidentalmente un secreto o introduce un patron inseguro, el pipeline lo detecta.

## Por que es bueno para quienes aprenden

No necesitas recordar ejecutar 10 comandos. CI/CD los ejecuta todos por ti.

Cuando abres un pull request, los resultados de CI estan ahi mismo en la interfaz de GitHub. Marcas verdes significan que todo esta bien. Cruces rojas te senalan directamente el fallo. No necesitas configurar un entorno local con multiples versiones de Python para verificar que tus cambios funcionan en todas partes.

CI/CD convierte la calidad de una lista de verificacion manual en un proceso automatizado. Sube codigo, recibe retroalimentacion, itera. Es lo mas parecido a tener un asistente robot que revisa tu trabajo en cada commit.
