from cgmpy import GlucoseData


def test_import():
    """Verify that cgmpy can be imported."""
    assert GlucoseData is not None


def test_basic_math():
    """Verify pytest is working."""
    assert 1 + 1 == 2
