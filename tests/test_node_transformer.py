"""Unit test class to verfy that NodeTransformer object behaves correctly."""
from unittest import TestCase
import numpy as np
import pytest
from embiggen import NodeTransformer


class TestNodeTransformer(TestCase):
    """Unit test class to verfy that NodeTransformer object behaves correctly."""

    def setUp(self):
        """Setup objects for running tests on NodeTransformer object."""
        self._embedding_size = 50
        self._nodes_number = 100
        self._embedding = np.random.random((  # pylint: disable=no-member
            self._nodes_number,
            self._embedding_size
        ))
        self._transfomer = NodeTransformer()

    def test_node_transformer(self):
        """Test to verify that node transformation returns expected shape."""
        self._transfomer.fit(self._embedding)
        sample_number = 50
        embedded_nodes = self._transfomer.transform(
            np.random.randint(0, self._nodes_number, size=sample_number)
        )
        self.assertEqual(
            embedded_nodes.shape,
            (sample_number, self._embedding_size)
        )

    def test_illegale_node_transformer(self):
        """Test that proper exception is raised when passing illegal parameters."""
        with pytest.raises(ValueError):
            self._transfomer.transform(None)
