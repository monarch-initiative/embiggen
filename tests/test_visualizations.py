"""Unit test class for GraphTransformer objects."""
from unittest import TestCase
from ensmallen import Graph  # pylint: disable=no-name-in-module
from embiggen import GraphVisualizer, GloVe
import pytest


class TestGraphVisualizer(TestCase):
    """Unit test class for GraphTransformer objects."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        self._embedding_size = 50
        self._visualization = GraphVisualizer(
            "Cora",
            repository="linqs",
            decomposition_method="PCA",
        )

    def test_graph_visualization(self):
        """Test graph visualization."""
        self._visualization.fit_and_plot_all("SPINE")
