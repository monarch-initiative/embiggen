"""Unit test class for Node-label prediction pipeline."""
from tqdm.auto import tqdm
from unittest import TestCase
from embiggen.edge_label_prediction import edge_label_prediction_evaluation
from embiggen import get_available_models_for_edge_label_prediction, get_available_models_for_node_embedding
from embiggen.edge_label_prediction.edge_label_prediction_model import AbstractEdgeLabelPredictionModel
from ensmallen.datasets.kgobo import PDUMDV


class TestEvaluateEdgeLabelPrediction(TestCase):
    """Unit test class for edge-label prediction pipeline."""

    def setUp(self):
        """Setup objects for running tests on edge-label prediction pipeline class."""
        self._number_of_holdouts = 2

    def test_evaluate_embedding_for_edge_label_prediction(self):
        """Test graph visualization."""
        df = get_available_models_for_edge_label_prediction()
        for evaluation_schema in AbstractEdgeLabelPredictionModel.get_available_evaluation_schemas():
            holdouts = edge_label_prediction_evaluation(
                holdouts_kwargs={
                    "train_size": 0.8
                },
                node_features="SPINE",
                models=df.model_name,
                library_names=df.library_name,
                graphs=PDUMDV().remove_singleton_nodes(),
                number_of_holdouts=self._number_of_holdouts,
                evaluation_schema=evaluation_schema,
                verbose=True,
                smoke_test=True
            )
        self.assertEqual(holdouts.shape[0],
                         self._number_of_holdouts*2*df.shape[0])

    def test_model_recreation(self):
        """Test graph visualization."""
        df = get_available_models_for_edge_label_prediction()

        for _, row in df.iterrows():
            model = AbstractEdgeLabelPredictionModel.get_model_from_library(
                model_name=row.model_name,
                task_name=AbstractEdgeLabelPredictionModel.task_name(),
                library_name=row.library_name
            )()
            AbstractEdgeLabelPredictionModel.get_model_from_library(
                model_name=row.model_name,
                task_name=AbstractEdgeLabelPredictionModel.task_name(),
                library_name=row.library_name
            )(**model.parameters())

    def test_all_embedding_models_as_feature(self):
        """Test graph visualization."""
        df = get_available_models_for_node_embedding()
        bar = tqdm(
            df.iterrows(),
            total=df.shape[0],
            leave=False,
            desc="Testing embedding methods"
        )
        for _, row in bar:
            if row.requires_edge_weights or row.requires_edge_types:
                continue

            bar.set_description(
                f"Testing embedding model {row.model_name} from library {row.library_name}")

            edge_label_prediction_evaluation(
                holdouts_kwargs=dict(train_size=0.8),
                models="Decision Tree Classifier",
                node_features=row.model_name,
                evaluation_schema="Stratified Monte Carlo",
                graphs=PDUMDV().remove_singleton_nodes(),
                number_of_holdouts=self._number_of_holdouts,
                verbose=False,
                smoke_test=True,
            )
