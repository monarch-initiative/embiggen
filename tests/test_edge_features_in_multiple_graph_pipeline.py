"""Unit test class for testing whether edge features are correctly handled in the pipeline with multiple graphs"""
from unittest import TestCase
from embiggen.edge_prediction import edge_prediction_evaluation, DecisionTreeEdgePrediction, RandomForestEdgePrediction
from ensmallen.datasets.linqs import Cora
from ensmallen.datasets.kgobo import CIO
from embiggen.embedders.ensmallen_embedders.hyper_sketching import HyperSketching


class TestEdgeFeaturesInMultipleGraphPipeline(TestCase):
    """Unit test class for testing whether edge features are correctly handled in the pipeline with multiple graphs."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        self._first_graph = Cora()
        self._second_graph = CIO()


    def test_edge_features_from_string_in_multiple_graph_pipeline(self):
        """Tests whether running from edge features defined as string works in the pipeline with multiple graphs."""
        edge_prediction_evaluation(
            holdouts_kwargs=dict(train_size=0.8),
            models=[
                "Decision Tree Classifier",
                "Random Forest Classifier"
                ],
            edge_features="HyperSketching",
            evaluation_schema="Monte Carlo",
            graphs=[
                self._first_graph,
                self._second_graph
            ],
            number_of_holdouts=2,
            smoke_test=True,
            verbose=False,
        )

    def test_edge_features_from_instance_in_multiple_graph_pipeline(self):
        """Tests whether running from edge features defined as instance works in the pipeline with multiple graphs."""
        edge_prediction_evaluation(
            holdouts_kwargs=dict(train_size=0.8),
            models=[
                "Decision Tree Classifier",
                "Random Forest Classifier"
            ],
            edge_features=HyperSketching(),
            evaluation_schema="Monte Carlo",
            graphs=[
                self._first_graph,
                self._second_graph
            ],
            number_of_holdouts=2,
            smoke_test=True,
            verbose=False,
        )

    def test_edge_features_from_instance_in_multiple_graph_pipeline_with_model_instance(self):
        """Tests whether running from edge features defined as instance works in the pipeline with multiple graphs."""
        edge_prediction_evaluation(
            holdouts_kwargs=dict(train_size=0.8),
            models=[
                RandomForestEdgePrediction(),
                DecisionTreeEdgePrediction()
            ],
            edge_features=HyperSketching(),
            evaluation_schema="Monte Carlo",
            graphs=[
                self._first_graph,
                self._second_graph
            ],
            number_of_holdouts=2,
            smoke_test=True,
            verbose=False,
        )
