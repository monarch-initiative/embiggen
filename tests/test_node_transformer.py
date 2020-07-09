from embiggen import NodeTransformer
from unittest import TestCase
import numpy as np
import pytest


class TestNodeTransformer(TestCase):

    def setUp(self):
        self._embedding_size = 50
        self._nodes_number = 100
        self._embedding = np.random.random(( # pylint: disable=no-member
            self._nodes_number,
            self._embedding_size
        ))
        self._transfomer = NodeTransformer()

    def test_node_transformer(self):        
        self._transfomer.fit(self._embedding)
        samples_number = 50
        embedded_nodes = self._transfomer.transform(
            np.random.randint(0, self._nodes_number, size=samples_number)
        )
        self.assertEqual(
            embedded_nodes.shape,
            (samples_number, self._embedding_size)
        )

    def test_illegale_node_transformer(self):
        with pytest.raises(ValueError):
            self._transfomer.transform(None)
