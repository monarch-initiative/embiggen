"""Unit test class for testing corner cases in edge-label GCN model."""
from tqdm.auto import tqdm
from unittest import TestCase
import pytest
import os
import numpy as np
from embiggen.edge_label_prediction import edge_label_prediction_evaluation
from embiggen import get_available_models_for_edge_label_prediction, get_available_models_for_node_embedding, get_available_models_for_edge_embedding
from embiggen.edge_label_prediction.edge_label_prediction_model import AbstractEdgeLabelPredictionModel
from embiggen.utils import AbstractEmbeddingModel
from ensmallen.datasets.kgobo import HP, CIO
from embiggen.embedders.ensmallen_embedders.hyper_sketching import HyperSketching
from embiggen.embedders.ensmallen_embedders.degree_spine import DegreeSPINE
from embiggen.feature_preprocessors import GraphConvolution
from embiggen.edge_label_prediction.edge_label_prediction_tensorflow.kipf_gcn import KipfGCNEdgeLabelPrediction


class TestEdgeLabelGCN(TestCase):
    """Unit test class for edge-label prediction pipeline."""

    def setUp(self):
        """Setup objects for running tests on edge-label prediction pipeline class."""
        self._number_of_holdouts = 2
        self.graph = HP().remove_singleton_nodes().remove_parallel_edges()

    def test_evaluate_embedding_for_edge_label_prediction(self):
        """Test graph visualization."""
        feature = DegreeSPINE(embedding_size=5)
        red = self.graph.set_all_edge_types("red")
        blue = CIO().remove_singleton_nodes().set_all_edge_types("blue")
        binary_graph = red | blue

        model: KipfGCNEdgeLabelPrediction = KipfGCNEdgeLabelPrediction(
            epochs=1,
            use_edge_metrics=True,
            verbose=True
        )
        model.fit(
            binary_graph,
            node_features=feature,
            edge_features=HyperSketching(),
        )
