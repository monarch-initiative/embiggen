"""Test that proper exceptions are raised when using illegal parameters in Embedder abstract model."""
import pytest
from embiggen.embedders.embedder import Embedder


def test_illegal_arguments():
    """Check that ValueError is raised on illegal parameters."""
    with pytest.raises(ValueError):
        Embedder(0, 5)
    with pytest.raises(ValueError):
        Embedder(5, 0)


def test_not_implemented_methods():
    """Check that NotImplementedError is raised on legal parameters calling directly abstract class."""
    with pytest.raises(NotImplementedError):
        Embedder(5, 5)
