"""Setup standard unit test class for NodeSequences."""
from unittest import TestCase
from ensmallen import Graph  # pylint: disable=no-name-in-module


class TestNodeSequences(TestCase):

    def setUp(self):
        self._graph = Graph.from_csv(
            edge_path="tests/data/small_ppi.tsv",
            sources_column="subject",
            destinations_column="object",
            directed=False,
            weights_column="weight"
        )
        self._graph.enable()
        self._weights_path = "weights_path.h5"
        self._embedding_path = "embedding.npy"
        self._sequence = None

    def check_nodes_range(self, nodes):
        return all(
            node >= 0 and node < self._graph.get_nodes_number()
            for node in nodes
        )
