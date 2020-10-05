"""Unit test class for GraphTransformer objects."""
from unittest import TestCase

import numpy as np
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from embiggen import GraphVisualizations
import pytest


class TestGraphVisualization(TestCase):
    """Unit test class for GraphTransformer objects."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        self._embedding_size = 50
        self._graph = EnsmallenGraph.from_unsorted_csv(
            edge_path=f"tests/data/small_ppi.tsv",
            sources_column="subject",
            destinations_column="object",
            directed=False,
            weights_column="weight",
            edge_types_column="weight"
        )
        self._embedding = np.random.random((  # pylint: disable=no-member
            self._graph.get_nodes_number(),
            self._embedding_size
        ))
        self._visualization = GraphVisualizations()

    def test_graph_visualization(self):
        """Test graph visualization."""
        self._visualization.fit_transform_nodes(
            self._graph,
            self._embedding,
            self._graph.get_nodes_mapping(),
            n_iter=100
        )
        self._visualization.fit_transform_edges(
            self._graph,
            self._embedding,
            n_iter=100
        )
        self._visualization.plot_node_degrees(self._graph)
        with pytest.raises(ValueError):
            self._visualization.plot_node_types(self._graph)
        self._visualization.plot_edge_types(self._graph)
        self._visualization.plot_edge_weights(self._graph)
        with pytest.raises(ValueError):
            self._visualization.plot_edge_types(self._graph, k=15)
