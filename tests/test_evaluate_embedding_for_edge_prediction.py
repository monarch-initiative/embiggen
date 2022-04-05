"""Unit test class for GraphTransformer objects."""
from unittest import TestCase
from ensmallen import Graph  # pylint: disable=no-name-in-module
from embiggen.pipelines import evaluate_embedding_for_edge_prediction
from ensmallen.datasets.networkrepository import Karate
import shutil
import os


class TestEvaluateEmbeddingForEdgePrediction(TestCase):
    """Unit test class for GraphTransformer objects."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        self._graph = Karate()
        self._subgraph = self._graph.get_random_subgraph(
            self._graph.get_nodes_number() - 2
        ).drop_singleton_nodes()
        self._number_of_holdouts = 2

    def test_evaluate_embedding_for_edge_prediction(self):
        """Test graph visualization."""
        if os.path.exists("node_embeddings"):
            shutil.rmtree("node_embeddings")
        holdouts, histories = evaluate_embedding_for_edge_prediction(
            embedding_method="GloVe",
            graph=self._graph,
            model_name="Perceptron",
            number_of_holdouts=self._number_of_holdouts,
            embedding_method_kwargs=dict(
                embedding_size=10,
            ),
            embedding_method_fit_kwargs=dict(
                epochs=1
            )
        )
        self.assertEqual(holdouts.shape[0], self._number_of_holdouts)
        self.assertEqual(len(histories), self._number_of_holdouts)

    def test_evaluate_embedding_for_edge_prediction_in_subgraph(self):
        """Test graph visualization."""
        if os.path.exists("node_embeddings"):
            shutil.rmtree("node_embeddings")
        holdouts, histories = evaluate_embedding_for_edge_prediction(
            embedding_method="GloVe",
            graph=self._graph,
            model_name="Perceptron",
            number_of_holdouts=self._number_of_holdouts,
            embedding_method_kwargs=dict(
                embedding_size=10,
            ),
            embedding_method_fit_kwargs=dict(
                epochs=1
            ),
            subgraph_of_interest_for_edge_prediction=self._subgraph
        )
        self.assertEqual(holdouts.shape[0], self._number_of_holdouts)
        self.assertEqual(len(histories), self._number_of_holdouts)