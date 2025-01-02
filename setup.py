# setup.py
from setuptools import setup, find_packages

setup(
    name='cgmpy',
    version='0.1.0',
    description='Un paquete para analizar datos de glucosa',
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
