# Que es MkDocs?

## Que es?

MkDocs convierte archivos Markdown en un sitio web. El sitio de documentacion que estas leyendo ahora mismo fue construido con MkDocs. Escribes las paginas en Markdown simple, y MkDocs las compila en un sitio HTML limpio con navegacion, busqueda y un tema atractivo.

## Por que lo usamos

GitHub puede renderizar archivos `.md`, pero un sitio web de documentacion real es mucho mas util -- especialmente para usuarios clinicos que quizas no se sientan comodos leyendo codigo en GitHub. MkDocs nos ofrece:

- Una tabla de contenidos / barra de navegacion lateral
- Busqueda de texto completo
- Referencia de API generada automaticamente desde los docstrings de Python
- Documentacion versionada (cada version mantiene su propia documentacion)
- Un aspecto profesional con el tema Material

## Como funciona

```
docs/*.md  ──→  MkDocs  ──→  Sitio HTML
(markdown)        (build)      (directorio site/)
```

Editas los archivos Markdown en `docs/`, ejecutas `make docs-serve` para previsualizar localmente, y cuando estas satisfecho, el sitio generado se despliega en GitHub Pages.

## Caracteristicas principales

- **Tema Material** -- el tema limpio y legible que ves en este sitio
- **mkdocstrings** -- genera automaticamente la documentacion de la API desde tus docstrings de Python (estilo Google). No necesitas escribir la documentacion de la API a mano.
- **Busqueda de texto completo** -- los usuarios pueden buscar en toda la documentacion
- **Documentacion versionada (mike)** -- cada version recibe su propia version de la documentacion, asi que los usuarios que leen una version anterior ven la documentacion correcta
- **Previsualizacion en vivo** -- ejecuta `make docs-serve` y un servidor local actualiza el sitio mientras editas los archivos

## Como escribir documentacion

1. Agrega un archivo `.md` a `docs/`
2. Agregalo a la seccion de navegacion en `mkdocs.yml`
3. Ejecuta `make docs-serve` para verlo en vivo
4. Escribe tu contenido

Las paginas de referencia de la API se generan automaticamente desde los docstrings -- no necesitas escribirlas a mano. Solo agrega un docstring estilo Google a tu funcion de Python y aparecera en la documentacion de la API automaticamente.

## Por que es bueno para quienes aprenden

La documentacion es como compartes lo que has construido. MkDocs facilita la creacion de documentacion de aspecto profesional sin conocer HTML, CSS o JavaScript. Solo escribes Markdown, y la herramienta se encarga del resto. Para CGMPy, una buena documentacion es especialmente importante porque muchos usuarios son clinicos o investigadores que necesitan explicaciones claras y legibles de como funciona cada metrica.
