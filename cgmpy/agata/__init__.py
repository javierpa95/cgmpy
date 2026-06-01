# cgmpy/agata/__init__.py

"""
Integration subpackage with the py_agata library.

This subpackage provides the tools needed to prepare and analyze
cgmpy data using the external py_agata library, acting as a bridge
between the two.

Public Functions:
- prepare_data_for_agata: Adapts a ModularGlucoseData object to the format required by py_agata.
- analyze_with_agata: Runs the full analysis pipeline using py_agata.
"""

# Import the functions to be exposed publicly from this subpackage
from .adapter import prepare_data_for_agata
from .metrics import AgataAnalysis, analyze_one_arm, analyze_with_agata

# Define what is imported when a user runs 'from cgmpy.agata import *'
# This is a good practice to define the public API of the subpackage.
__all__ = ["AgataAnalysis", "analyze_one_arm", "analyze_with_agata", "prepare_data_for_agata"]
