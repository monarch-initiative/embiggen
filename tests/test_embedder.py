import pytest
from embiggen.embedders.embedder import Embedder


def test_illegal_arguments():
    with pytest.raises(ValueError):
        Embedder(0, 5)
    with pytest.raises(ValueError):
        Embedder(5, 0)


def test_not_implemented_methods():
    with pytest.raises(NotImplementedError):
        Embedder(5, 5)
