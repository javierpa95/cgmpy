# Que son los Conventional Commits?

## Que son?

Una convencion para escribir mensajes de commit en git. En lugar de "arregladas cosas" o "cambios", escribes:

```
fix(metrics): handle empty glucose series without raising ZeroDivisionError
```

Los Conventional Commits asignan a cada commit un **tipo**, un **ambito** opcional y una **descripcion** que explica que cambio y por que.

## El formato

```
<tipo>(<ambito>): <descripcion>

<cuerpo -- por que este cambio es importante>

<pie -- referencias, cambios rupturistas>
```

La linea de asunto (primera linea) debe tener 72 caracteres o menos. El cuerpo explica la razon detras del cambio. El pie puede referenciar issues o marcar cambios rupturistas (breaking changes).

## Tipos usados en CGMPy

| Tipo | Cuando usarlo | Ejemplo |
|------|---------------|---------|
| `feat` | Nueva funcionalidad | `feat(metrics): add MAGE-2 calculation` |
| `fix` | Correccion de errores | `fix(data): handle empty CSV gracefully` |
| `docs` | Documentacion | `docs: add pregnancy analysis guide` |
| `refactor` | Cambio de codigo sin cambio de comportamiento | `refactor(metrics): extract pure functions from mixins` |
| `test` | Agregar o actualizar tests | `test: add clinical reference for MAGE` |
| `chore` | Herramientas, configuracion, dependencias | `chore: update ruff to v0.15` |
| `ci` | Cambios en CI o CD | `ci: add Python 3.12 to test matrix` |

## Ejemplo real en CGMPy

Este es un commit de la historia del proyecto:

```
feat(metrics): implement MAGE_Baghurst algorithm

Adds the Baghurst smoothing approach for MAGE calculation,
with three configurable methods (smoothing, direct elimination,
simplified). Includes guard clauses for datasets with fewer
than 9 readings or zero standard deviation.

Closes #42
```

El asunto te dice que se anadio. El cuerpo te dice por que y como. El pie enlaza con el issue original.

## Por que es importante

Seis meses despues, puedes ejecutar `git log --oneline` y entender exactamente que cambio y por que. El changelog (revisa `CHANGELOG.md`) se genera automaticamente a partir de estos mensajes -- los commits `docs:` aparecen en la seccion de documentacion, los commits `fix:` en la seccion de correcciones, y asi sucesivamente.

## Como escribir un buen commit

- Linea de asunto: 72 caracteres o menos, modo imperativo ("add feature", no "added feature"), sin punto final.
- Cuerpo: explica el **por que**, no el **que**. Git ya te muestra lo que cambio -- el cuerpo debe decirte por que el cambio era necesario.
- Referencia a issues: usa `Closes #42` o `Fixes #42` para cerrar automaticamente el issue cuando se fusiona la PR.

## Por que es bueno para quienes aprenden

Te obliga a pensar sobre lo que has hecho antes de hacer el commit. Que tipo de cambio es? Es un `feat` o un `refactor`? Que ambito afecta? Responder estas preguntas convierte tus commits en una historia legible y bien organizada del proyecto -- y esa historia es invaluable cuando alguien (incluyendote a ti en el futuro) intenta entender por que se hizo un cambio.
