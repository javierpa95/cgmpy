# setup.py
from setuptools import setup, find_packages

setup(
    name='cgmpy',
    version='0.2.0',
    description='A package for analyzing glucose data - Version refactorizada',
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
