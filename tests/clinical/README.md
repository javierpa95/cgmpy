"""Clinical regression tests.

These tests validate CGMPy metrics against published reference values from
clinical literature or against the AGATA reference implementation.

Add a new file here for each metric that has a published reference.
Use `@pytest.mark.clinical` and `@pytest.mark.slow` so the suite can be
deselected with `pytest -m "not clinical"`.
"""
