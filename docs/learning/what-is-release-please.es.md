# Que es Release Please?

## Que es?

Release Please es un bot de GitHub que crea una pull request para publicar una nueva version de la libreria. Lee todos los mensajes de commit desde la ultima publicacion, determina si la nueva version es un aumento major, minor o patch, actualiza la version en `pyproject.toml`, genera las entradas del changelog y abre una PR de publicacion -- todo automaticamente.

## Por que lo usamos

Las publicaciones manuales son propensas a errores. Te acordaste de actualizar el numero de version? Actualizaste el changelog? Capturaste todos los cambios rupturistas? Release Please automatiza todo el proceso para que un humano solo necesite revisar y fusionar la PR generada.

## Como determina el aumento de version

Release Please lee tus mensajes de Conventional Commit y decide el siguiente numero de version:

- Commits `fix:` -- aumento patch (0.5.1 a 0.5.2)
- Commits `feat:` -- aumento minor (0.5.2 a 0.6.0)
- `BREAKING CHANGE:` en el pie del commit -- aumento major (0.6.0 a 1.0.0)

Si hay commits tanto `fix:` como `feat:` desde la ultima publicacion, elige el aumento mas alto (minor prevalece sobre patch). Si hay un cambio rupturista, se convierte en una version major.

## El flujo de publicacion

```
1. El desarrollador fusiona conventional commits a main
2. El bot Release Please abre una pull request de publicacion
3. El mantenedor revisa el changelog generado
4. El mantenedor fusiona la pull request de publicacion
5. Release Please crea un GitHub Release y un tag de git
6. (Futuro) Publicacion automatica en PyPI
```

La PR de publicacion incluye el numero de version actualizado, el changelog para esta version y un resumen de todos los cambios agrupados por tipo.

## Archivos de configuracion

- `release-please-config.json` -- le indica a Release Please la estructura del proyecto, que archivos actualizar y como manejar el aumento de version
- `.release-please-manifest.json` -- registra la version actual del proyecto para que el bot sepa cual fue la ultima publicacion

## Por que es bueno para quienes aprenden

Release Please muestra como la automatizacion puede eliminar tareas repetitivas y propensas a errores. Te concentras en escribir codigo y buenos mensajes de commit. El bot se encarga del trabajo administrativo -- aumentos de version, generacion de changelog y publicaciones en GitHub. Es un gran ejemplo de como invertir un poco de tiempo en herramientas ahorra horas de trabajo manual durante la vida de un proyecto.
