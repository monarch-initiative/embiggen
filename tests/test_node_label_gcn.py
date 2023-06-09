"""Unit test class for testing corner cases in node-label GCN model."""
from unittest import TestCase
from ensmallen.datasets.kgobo import HP, CIO
from embiggen.embedders.ensmallen_embedders.degree_spine import DegreeSPINE
from embiggen.node_label_prediction import KipfGCNNodeLabelPrediction


class TestNodeLabelGCN(TestCase):
    """Unit test class for node-label prediction pipeline."""

    def setUp(self):
        """Setup objects for running tests on node-label prediction pipeline class."""
        self._number_of_holdouts = 2
        self.graph = HP().remove_singleton_nodes().remove_parallel_edges()

    def test_evaluate_embedding_for_node_label_prediction(self):
        """Test graph visualization."""
        if not KipfGCNNodeLabelPrediction.is_available():
            return

        feature = DegreeSPINE(embedding_size=5)
        red = self.graph.set_all_edge_types("red")
        blue = CIO().remove_singleton_nodes().set_all_edge_types("blue")
        binary_graph = red | blue

        model: KipfGCNNodeLabelPrediction = KipfGCNNodeLabelPrediction().into_smoke_test()
        model.fit(
            binary_graph,
            node_features=feature
        )
