"""Test module to test that all docstrings in package are complete and correct."""
from validate_python_docstring import validate_package
import embiggen


def test_docstrings():
    """Execute correctness test on all docstrings."""
    validate_package(embiggen, verbose=True)
