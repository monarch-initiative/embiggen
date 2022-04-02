"""Unit test class for GraphTransformer objects."""
from unittest import TestCase
from ensmallen import Graph  # pylint: disable=no-name-in-module
from embiggen import GraphVisualization, GloVe
import pytest
from embiggen.pipelines import evaluate_embedding_for_edge_prediction
from plot_keras_history import plot_history
import matplotlib.pyplot as plt
from ensmallen.datasets.networkrepository import Karate
import shutil


class TestEvaluateEmbeddingForEdgePrediction(TestCase):
    """Unit test class for GraphTransformer objects."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        self._graph = Karate()

    def test_evaluate_embedding_for_edge_prediction(self):
        """Test graph visualization."""
        shutil.rmtree("node_embeddings")
        evaluate_embedding_for_edge_prediction(
            embedding_method="GloVe",
            graph=self._graph,
            model_name="Perceptron",
            number_of_holdouts=3,
            embedding_method_kwargs=dict(
                embedding_size=10,
            ),
            embedding_method_fit_kwargs=dict(
                epochs=1
            )
        )
