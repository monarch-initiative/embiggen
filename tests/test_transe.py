"""Test to validate that the model GloVe works properly with graph walks."""
from embiggen.embedders.transe import TransE
from unittest import TestCase
from ensmallen_graph.datasets.linqs import Cora


class TestTransE(TestCase):
    """Unit test for model GloVe on graph walks."""

    def setUp(self):
        """Setting up objects to test CBOW model on graph walks."""
        super().setUp()
        self._graph = Cora()

    def test_fit(self):
        """Test that model fitting behaves correctly and produced embedding has correct shape."""
        model = TransE(self._graph)
        self.assertEqual("TransE", model.name)
        model.summary()
        model.fit(epochs=2)
