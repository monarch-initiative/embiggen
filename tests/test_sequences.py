from embiggen import NodeSkipGramSequence, NodeCBOWSequence
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from itertools import chain
from unittest import TestCase


class TestSequences(TestCase):

    def setUp(self):
        self._graph = EnsmallenGraph.from_csv(
            edge_path=f"tests/data/small_ppi.tsv",
            sources_column="subject",
            destinations_column="object",
            directed=False,
            weights_column="weight"
        )
        self._sequence = None

    def check_nodes_range(self, nodes):
        return all(
            node >= 0 and node < self._graph.get_nodes_number()
            for node in nodes
        )
