"""Unit test class for GraphTransformer objects."""
from unittest import TestCase
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from embiggen import GraphVisualization, GloVe
import pytest


class TestGraphVisualization(TestCase):
    """Unit test class for GraphTransformer objects."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        self._embedding_size = 50
        self._graph = EnsmallenGraph.from_csv(
            edge_path="tests/data/small_ppi.tsv",
            sources_column="subject",
            destinations_column="object",
            directed=False,
            weights_column="weight",
            edge_list_edge_types_column="edge_label"
        )
        self._embedding = GloVe(
            vocabulary_size=self._graph.get_nodes_number(),
            embedding_size=self._embedding_size
        ).get_embedding_dataframe(self._graph.get_node_names())
        self._visualization = GraphVisualization(
            self._graph,
            decomposition_method="PCA"
        )

    def test_graph_visualization(self):
        """Test graph visualization."""
        self._visualization.fit_transform_nodes(self._embedding)
        self._visualization.fit_transform_edges(self._embedding)
        self._visualization.plot_node_degrees()
        with pytest.raises(ValueError):
            self._visualization.plot_node_types()
        self._visualization.plot_edge_types()
        self._visualization.plot_edge_weights()
        with pytest.raises(ValueError):
            self._visualization.plot_edge_types(k=15)
