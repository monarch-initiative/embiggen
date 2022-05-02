"""Unit test class for GraphTransformer objects."""
from unittest import TestCase
from embiggen.pipelines import evaluate_embedding_for_edge_prediction
from ensmallen.datasets.linqs import Cora
import shutil
import os


class TestEvaluateEmbeddingForEdgePrediction(TestCase):
    """Unit test class for GraphTransformer objects."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        self._graph = Cora()
        self._subgraph = self._graph.get_random_subgraph(
            self._graph.get_nodes_number() - 2
        ).drop_singleton_nodes()
        self._number_of_holdouts = 2

    def test_evaluate_embedding_for_edge_prediction(self):
        """Test graph visualization."""
        if os.path.exists("node_embeddings"):
            shutil.rmtree("node_embeddings")
        for model, fit_kwargs in (("Perceptron", dict(epochs=5)), ("DecisionTreeClassifier", None)):
            holdouts = evaluate_embedding_for_edge_prediction(
                embedding_method="CBOW",
                graphs=self._graph,
                model=model,
                number_of_holdouts=self._number_of_holdouts,
                unbalance_rates = (10.0, 100.0, ),
                embedding_kwargs=dict(
                    embedding_size=10,
                    epochs=1
                ),
                classifier_fit_kwargs=fit_kwargs
            )
            self.assertEqual(holdouts.shape[0], self._number_of_holdouts*2*3)

    def test_evaluate_embedding_for_edge_prediction_in_subgraph(self):
        """Test graph visualization."""
        if os.path.exists("node_embeddings"):
            shutil.rmtree("node_embeddings")
        for model, fit_kwargs in (("Perceptron", dict(epochs=5)), ("DecisionTreeClassifier", None)):
            holdouts = evaluate_embedding_for_edge_prediction(
                embedding_method="CBOW",
                graphs=self._graph,
                model=model,
                unbalance_rates = (10.0, 100.0, ),
                number_of_holdouts=self._number_of_holdouts,
                embedding_kwargs=dict(
                    embedding_size=10,
                    epochs=1
                ),
                classifier_fit_kwargs=fit_kwargs,
                subgraph_of_interest_for_edge_prediction=self._subgraph,
            )
            self.assertEqual(holdouts.shape[0], self._number_of_holdouts*2*3)
