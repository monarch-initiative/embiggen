import numpy as np
from unittest import TestCase
from ensmallen_graph import StringPPI
from embiggen.link_prediction import Perceptron


class TestPerceptron(TestCase):

    def setUp(self):
        self._string_ppi = StringPPI(verbose=False)
        embedding = np.ones((self._string_ppi.get_nodes_number(), 10))
        self._model = Perceptron(embedding)

    def test_training(self):
        self._model.compile()
        self._model.fit(self._string_ppi)
