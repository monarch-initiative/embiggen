"""Unit test class for testing corner cases in edge GCN model."""
import os
import pandas as pd
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
        red = self.graph.set_all_edge_types("red")
        blue = CIO().remove_singleton_nodes().set_all_edge_types("blue")
        binary_graph = red | blue
        feature: pd.DataFrame = DegreeSPINE(embedding_size=5).fit_transform(binary_graph).get_single_embedding()

        if not KipfGCNEdgePrediction.is_available():
            return
        
        model: KipfGCNEdgePrediction = KipfGCNEdgePrediction(
            epochs=1,
            use_edge_metrics=True,
            edge_embedding_methods=["Concatenate", "Hadamard", "Average"],
            verbose=True
        )

        subset_feature = feature.loc[red.get_node_names()]

        edge_features = HyperSketching()
        edge_features.fit(binary_graph)

        red_edge_features = HyperSketching()
        red_edge_features.fit(red)

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

        _processed_edge_features = beheaded.predict_proba(
            red,
            node_features=subset_feature,
            edge_features=red_edge_features,
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


