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
