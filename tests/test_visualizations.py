"""Unit test class for GraphTransformer objects."""
from unittest import TestCase

import numpy as np
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from embiggen import GraphVisualizations


class TestGraphVisualization(TestCase):
    """Unit test class for GraphTransformer objects."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        self._embedding_size = 50
        self._graph = EnsmallenGraph.from_csv(
            edge_path=f"tests/data/small_ppi.tsv",
            sources_column="subject",
            destinations_column="object",
            directed=False,
            weights_column="weight"
        )
        self._embedding = np.random.random((  # pylint: disable=no-member
            self._graph.get_nodes_number(),
            self._embedding_size
        ))
        self._visualization = GraphVisualizations()

    def test_graph_visualization(self):
        """Test graph visualization."""
        self._visualization.fit(
            self._embedding,
            self._graph.nodes_mapping
        )
        self._visualization.visualize(
            self._graph,
            tsne_kwargs={
                "n_iter": 100
            }
        )
