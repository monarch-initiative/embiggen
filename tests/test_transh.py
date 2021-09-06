"""Test to validate that the model GloVe works properly with graph walks."""
from embiggen import TransH
from unittest import TestCase
from ensmallen.datasets.linqs import Cora


class TestTransH(TestCase):
    """Unit test for model GloVe on graph walks."""

    def setUp(self):
        """Setting up objects to test CBOW model on graph walks."""
        super().setUp()
        self._graph = Cora()

    def test_fit(self):
        """Test that model fitting behaves correctly and produced embedding has correct shape."""
        model = TransH(self._graph)
        self.assertEqual("TransH", model.name)
        model.summary()
        model.fit(epochs=2, batches_per_epoch=2)
