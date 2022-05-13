"""Unit test class for Node-label prediction pipeline."""
from unittest import TestCase
from embiggen.node_label_prediction import node_label_prediction_evaluation
from ensmallen.datasets.linqs import Cora
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
        
        node_label_prediction_evaluation(
            holdouts_kwargs={
                "train_size": 0.8
            },
            node_features="SPINE",
            models="Decision Tree Classifier",
            graphs="Cora",
            repositories="linqs"
        )