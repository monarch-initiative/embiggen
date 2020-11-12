"""Unit test class for GraphTransformer objects."""
from unittest import TestCase
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from embiggen import GraphTransformer, GloVe


class TestGraphTransformer(TestCase):
    """Unit test class for GraphTransformer objects."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        self._embedding_size = 50
        self._graph: EnsmallenGraph = EnsmallenGraph.from_unsorted_csv(
            edge_path=f"tests/data/small_ppi.tsv",
            sources_column="subject",
            destinations_column="object",
            directed=False,
            weights_column="weight"
        )
        self._transfomer = None
        self._embedding = GloVe(
            vocabulary_size=self._graph.get_nodes_number(),
            embedding_size=self._embedding_size
        ).get_embedding_dataframe(self._graph.get_node_names())

    def test_graph_transformer(self):
        """Test to verify that graph transformation returns expected shape."""
        self._transfomer = GraphTransformer()
        self._transfomer.fit(self._embedding)
        embedded_nodes = self._transfomer.transform(self._graph)
        self.assertEqual(
            embedded_nodes.shape,
            (self._graph.get_edges_number(), self._embedding_size)
        )
