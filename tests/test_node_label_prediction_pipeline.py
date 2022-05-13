"""Unit test class for Node-label prediction pipeline."""
from unittest import TestCase
from embiggen import node_label_prediction_evaluation, get_available_models_for_node_label_prediction
import shutil
import os


class TestEvaluateNodeLabelPrediction(TestCase):
    """Unit test class for Node-label prediction pipeline."""

    def setUp(self):
        """Setup objects for running tests on Node-label prediction pipeline class."""
        self._number_of_holdouts = 2

    def test_evaluate_embedding_for_node_label_prediction(self):
        """Test graph visualization."""
        if os.path.exists("experiments"):
            shutil.rmtree("experiments")

        for _, row in get_available_models_for_node_label_prediction().iterrows():
            holdouts = node_label_prediction_evaluation(
                holdouts_kwargs={
                    "train_size": 0.8
                },
                node_features="SPINE",
                models=row.model_name,
                library_names=row.model_name,
                graphs="Cora",
                repositories="linqs",
                verbose=False
            )
            self.assertEqual(holdouts.shape[0], self._number_of_holdouts*2)

        if os.path.exists("experiments"):
            shutil.rmtree("experiments")
