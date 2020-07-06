from unittest import TestCase

import pytest
from embiggen.embedders.embedder import Embedder


class TestEmbedder(TestCase):

    def test_illegal_arguments(self):
        with pytest.raises(ValueError):
            Embedder(0, 5)
        with pytest.raises(ValueError):
            Embedder(5, 0)

    def test_not_implemented_methods(self):
        with pytest.raises(NotImplementedError):
            Embedder(5, 5)
