"""Unit test class for Node-label prediction pipeline."""
from unittest import TestCase
from embiggen import node_label_prediction_evaluation, get_available_models_for_node_label_prediction, SPINE
from ensmallen.datasets.linqs import Cora, get_words_data
import shutil
import os


class TestEvaluateNodeLabelPrediction(TestCase):
    """Unit test class for Node-label prediction pipeline."""

    def setUp(self):
        """Setup objects for running tests on Node-label prediction pipeline class."""
        self._graph, _ = get_words_data(Cora())
        self._number_of_holdouts = 2

    def test_evaluate_embedding_for_node_label_prediction(self):
        """Test graph visualization."""
        if os.path.exists("experiments"):
            shutil.rmtree("experiments")

        df = get_available_models_for_node_label_prediction()
        holdouts = node_label_prediction_evaluation(
            holdouts_kwargs={
                "train_size": 0.8
            },
            node_features=SPINE(embedding_size=5),
            models=df.model_name,
            library_names=df.library_name,
            graphs=self._graph,
            verbose=False,
            number_of_holdouts=self._number_of_holdouts,
            smoke_test=True
        )
        self.assertEqual(holdouts.shape[0], self._number_of_holdouts*2*df.shape[0])

        if os.path.exists("experiments"):
            shutil.rmtree("experiments")
