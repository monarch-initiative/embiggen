"""Unit test class for testing corner cases in edge GCN model."""
import os
from unittest import TestCase
from ensmallen.datasets.kgobo import HP, CIO
from embiggen.embedders.ensmallen_embedders.hyper_sketching import HyperSketching
from embiggen.embedders.ensmallen_embedders.degree_spine import DegreeSPINE
from embiggen.edge_prediction import KipfGCNEdgePrediction


class TestEdgeGCN(TestCase):
    """Unit test class for edge prediction pipeline."""

    def setUp(self):
        """Setup objects for running tests on edge prediction pipeline class."""
        self._number_of_holdouts = 2
        self.graph = HP().remove_singleton_nodes().remove_parallel_edges()

    def test_evaluate_embedding_for_edge_prediction(self):
        """Test graph visualization."""
        feature = DegreeSPINE(embedding_size=5)
        red = self.graph.set_all_edge_types("red")
        blue = CIO().remove_singleton_nodes().set_all_edge_types("blue")
        binary_graph = red | blue

        if not KipfGCNEdgePrediction.is_available():
            return
        
        model: KipfGCNEdgePrediction = KipfGCNEdgePrediction(
            epochs=1,
            use_edge_metrics=True,
            edge_embedding_methods=["Concatenate", "Hadamard", "Average"],
            verbose=True
        )

        edge_features = HyperSketching()
        edge_features.fit(binary_graph)
        
        model.fit(
            binary_graph,
            node_features=feature,
            edge_features=edge_features,
        )

        beheaded: KipfGCNEdgePrediction = model.into_beheaded_edge_model()

        _processed_edge_features = beheaded.predict_proba(
            binary_graph,
            node_features=feature,
            edge_features=edge_features,
        )

        _processed_edge_features = beheaded.predict_proba(
            binary_graph,
            node_features=feature,
            edge_features=edge_features,
        )

        if os.path.exists("test.csv"):
            os.remove("test.csv")
        _processed_edge_features_df = beheaded.predict_proba(
            binary_graph,
            node_features=feature,
            edge_features=edge_features,
            path="test.csv"
        )
        self.assertTrue(os.path.exists("test.csv"))
        os.remove("test.csv")


