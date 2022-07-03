"""Unit test class for GraphTransformer objects."""
from unittest import TestCase
from ensmallen import Graph  # pylint: disable=no-name-in-module
from embiggen import GraphVisualizer
from embiggen.embedders import SPINE


class TestGraphVisualizer(TestCase):
    """Unit test class for GraphTransformer objects."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        pass

    def test_graph_visualization(self):
        """Test graph visualization."""
        for graph_name in ("CIO", "Usair97", "MIAPA"):
            for method in ("PCA", "TSNE", "UMAP"):
                visualization = GraphVisualizer(
                    graph_name,
                    decomposition_method=method,
                )
                visualization.fit_and_plot_all("HOPE")
                visualization.plot_dot()
                visualization.plot_edges()
                visualization.plot_nodes(annotate_nodes=True)
        