# CGMPy

Un paquete para analizar datos de glucosa.

## Instalación

pip install cgmpy


## Uso

```python
from cgmpy import GlucoseData

# Ejemplo de uso
g = GlucoseData('ruta/al/archivo.csv',"time","glucose")
print(g.mean())

```

