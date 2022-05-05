"""Unit test class for Node-label prediction pipeline."""
from unittest import TestCase
from embiggen.pipelines import node_label_prediction
from ensmallen.datasets.linqs import Cora
import shutil
import os


class TestEvaluateNodeLabelPrediction(TestCase):
    """Unit test class for Node-label prediction pipeline."""

    def setUp(self):
        """Setup objects for running tests on Node-label prediction pipeline class."""
        self._graph = Cora()
        self._subgraph = self._graph.get_random_subgraph(
            self._graph.get_nodes_number() - 2
        ).remove_singleton_nodes()
        self._number_of_holdouts = 2

    def test_evaluate_embedding_for_edge_prediction(self):
        """Test graph visualization."""
        if os.path.exists("node_embeddings"):
            shutil.rmtree("node_embeddings")
        
        node_label_prediction(
            node_features="SPINE",
            graphs=self._graph,
        )