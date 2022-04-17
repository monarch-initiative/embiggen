"""Test that exceptions are raised when using illegal parameters."""
import pytest
from embiggen.embedders.tensorflow_embedders import TensorFlowEmbedder


def test_illegal_arguments():
    """Check that ValueError is raised on illegal parameters."""
    with pytest.raises(ValueError):
        TensorFlowEmbedder(0, 5)
    with pytest.raises(ValueError):
        TensorFlowEmbedder(5, 0)


def test_not_implemented_methods():
    """Check NotImplementedError is raised when calling directly abstract."""
    with pytest.raises(NotImplementedError):
        TensorFlowEmbedder(5, 5)
