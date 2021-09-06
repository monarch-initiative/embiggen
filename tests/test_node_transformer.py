"""Unit test class to verfy that NodeTransformer object behaves correctly."""
from unittest import TestCase
import pytest
from embiggen import NodeTransformer, GloVe
from ensmallen import Graph  # pylint: disable=no-name-in-module


class TestNodeTransformer(TestCase):
    """Unit test class to verfy that NodeTransformer object behaves correctly."""

    def setUp(self):
        """Setup objects for running tests on NodeTransformer object."""
        self._embedding_size = 50
        self._nodes_number = 100
        self._graph: Graph = Graph.from_csv(
            edge_path="tests/data/small_ppi.tsv",
            sources_column="subject",
            destinations_column="object",
            directed=False,
            weights_column="weight"
        )
        self._node_names = self._graph.get_node_names()
        self._embedding = GloVe(
            vocabulary_size=self._graph.get_nodes_number(),
            embedding_size=self._embedding_size
        ).get_embedding_dataframe(self._node_names)
        self._transfomer = NodeTransformer()

    def test_node_transformer(self):
        """Test to verify that node transformation returns expected shape."""
        self._transfomer.fit(self._embedding)
        sample_number = 50
        embedded_nodes = self._transfomer.transform(self._node_names[:sample_number])
        self.assertEqual(
            embedded_nodes.shape,
            (sample_number, self._embedding_size)
        )

    def test_illegale_node_transformer(self):
        """Test that proper exception is raised when passing illegal parameters."""
        with pytest.raises(ValueError):
            self._transfomer.transform(None)
