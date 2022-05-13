"""Unit test class for GraphTransformer objects."""
from unittest import TestCase
from embiggen import edge_prediction_evaluation, get_available_models_for_edge_prediction
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
        ).remove_singleton_nodes()
        self._number_of_holdouts = 2

    def test_evaluate_embedding_for_edge_prediction_in_subgraph(self):
        """Test graph visualization."""
        if os.path.exists("node_embeddings"):
            shutil.rmtree("node_embeddings")
        
        for _, row in get_available_models_for_edge_prediction().iterrows():
            holdouts = edge_prediction_evaluation(
                holdouts_kwargs=dict(),
                models=row.model_name,
                library_names=row.library_name,
                node_features="SPINE",
                evaluation_schema="Kfold",
                graphs=self._graph,
                number_of_holdouts=self._number_of_holdouts,
                unbalance_rates = (2.0, 3.0, ),
                verbose=False
            )
            self.assertEqual(holdouts.shape[0], self._number_of_holdouts*2*2)
