"""Unit test class for GraphTransformer objects."""
from unittest import TestCase
from ensmallen import Graph  # pylint: disable=no-name-in-module
from embiggen import GraphVisualizer
from embiggen.embedders import SPINE


class TestGraphVisualizer(TestCase):
    """Unit test class for GraphTransformer objects."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        self._visualization = GraphVisualizer(
            "Cora",
            repository="linqs",
            decomposition_method="PCA",
        )

    def test_graph_visualization(self):
        """Test graph visualization."""
        self._visualization.fit_and_plot_all("SPINE", embedding_size=5)

    def test_graph_visualization(self):
        """Test graph visualization."""
        self._visualization.fit_and_plot_all(
            SPINE(embedding_size=5).fit_transform(self._visualization._graph)
        )
