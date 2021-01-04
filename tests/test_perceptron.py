import numpy as np
from unittest import TestCase
from ensmallen_graph import StringPPI
from embiggen.link_prediction import Perceptron
from embiggen.link_prediction.layers import edge_embedding_layer


class TestPerceptron(TestCase):

    def setUp(self):
        self._string_ppi = StringPPI(verbose=False)
        self._embedding = np.random.uniform(
            size=(self._string_ppi.get_nodes_number(), 10))

    def test_training(self):
        for method in edge_embedding_layer.keys():
            model = Perceptron(
                self._embedding,
                edge_embedding_method=method
            )
            model.summary()
            model.compile()
            model.fit(self._string_ppi, batches_per_epoch=10)
