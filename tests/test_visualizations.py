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
                decomposition_kwargs = None
                if method == "TSNE":
                    decomposition_kwargs = dict(n_iter=1)
                visualization = GraphVisualizer(
                    graph_name,
                    decomposition_method=method,
                    decomposition_kwargs=decomposition_kwargs,
                    number_of_subsampled_nodes=20,
                    number_of_subsampled_edges=20,
                    number_of_subsampled_negative_edges=20
                )
                visualization.fit_and_plot_all("SPINE", embedding_size=5)
                visualization.plot_dot()
                visualization.plot_edges()
                visualization.plot_nodes(annotate_nodes=True)
                visualization.fit_and_plot_all("SPINE", embedding_size=2)
                visualization = GraphVisualizer(
                    graph_name,
                    decomposition_method=method,
                    decomposition_kwargs=decomposition_kwargs,
                )
                visualization.fit_and_plot_all("SPINE", embedding_size=5)
                visualization.plot_dot()
                visualization.plot_edges()
                visualization.plot_nodes(annotate_nodes=True)
                visualization.fit_and_plot_all("SPINE", embedding_size=2)
