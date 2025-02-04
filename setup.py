# setup.py
from setuptools import setup, find_packages

setup(
    name='cgmpy',
    version='0.1.0',
    description='A package for analyzing glucose data',
    author='Javier Peñate Arrieta',
    author_email='javierpenatearrieta@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
    ],
    url="https://github.com/javierpa95/cgmpy",

)

import numpy as np
import matplotlib.pyplot as plt

def plot_normal_distribution():
    # Crear datos para la curva
    x = np.linspace(-4, 4, 1000)
    y = (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2)
    
    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='Distribución Normal')
    plt.title('Función de Densidad Normal')
    plt.xlabel('Z-score')
    plt.ylabel('Densidad de probabilidad')
    plt.grid(True, alpha=0.3)
    
    # Añadir línea en x=0
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    
    # Añadir anotaciones
    plt.annotate('e = 2.71828... (número de Euler)\nπ = 3.14159...',
                 xy=(-3, 0.35))
    
    plt.legend()
    plt.show()

plot_normal_distribution()