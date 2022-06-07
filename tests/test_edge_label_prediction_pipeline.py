"""Unit test class for Node-label prediction pipeline."""
from unittest import TestCase
from embiggen.edge_label_prediction import edge_label_prediction_evaluation
from embiggen import get_available_models_for_edge_label_prediction
from embiggen.embedders import SPINE
from ensmallen.datasets.string import SpeciesTree


class TestEvaluateEdgeLabelPrediction(TestCase):
    """Unit test class for edge-label prediction pipeline."""

    def setUp(self):
        """Setup objects for running tests on edge-label prediction pipeline class."""
        self._number_of_holdouts = 2
        self._graph = SpeciesTree()
    
    def test_evaluate_embedding_for_edge_label_prediction(self):
        """Test graph visualization."""
        node_embedding = SPINE(embedding_size=5).fit_transform(self._graph)

        df = get_available_models_for_edge_label_prediction()
        holdouts = edge_label_prediction_evaluation(
            holdouts_kwargs={
                "train_size": 0.8
            },
            node_features=node_embedding,
            models=df.model_name,
            library_names=df.library_name,
            graphs=self._graph,
            number_of_holdouts=self._number_of_holdouts,
            verbose=True,
            smoke_test=True
        )
        self.assertEqual(holdouts.shape[0], self._number_of_holdouts*2*df.shape[0])
