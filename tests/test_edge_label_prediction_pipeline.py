"""Unit test class for Node-label prediction pipeline."""
from tqdm.auto import tqdm
from unittest import TestCase
import numpy as np
from embiggen.edge_label_prediction import edge_label_prediction_evaluation
from embiggen import get_available_models_for_edge_label_prediction, get_available_models_for_node_embedding
from embiggen.edge_label_prediction.edge_label_prediction_model import AbstractEdgeLabelPredictionModel
from embiggen.utils import AbstractEmbeddingModel
from ensmallen.datasets.kgobo import PDUMDV
from embiggen.embedders import SPINE


class TestEvaluateEdgeLabelPrediction(TestCase):
    """Unit test class for edge-label prediction pipeline."""

    def setUp(self):
        """Setup objects for running tests on edge-label prediction pipeline class."""
        self._number_of_holdouts = 2

    def test_evaluate_embedding_for_edge_label_prediction(self):
        """Test graph visualization."""
        df = get_available_models_for_edge_label_prediction()
        graph = PDUMDV().remove_singleton_nodes()
        feature = SPINE(embedding_size=5)
        for evaluation_schema in AbstractEdgeLabelPredictionModel.get_available_evaluation_schemas():
            if "Stratified" not in evaluation_schema:
                # TODO! properly test this!
                continue
            holdouts = edge_label_prediction_evaluation(
                holdouts_kwargs={
                    "train_size": 0.8
                },
                node_features=feature,
                models=df.model_name,
                library_names=df.library_name,
                graphs=graph,
                number_of_holdouts=self._number_of_holdouts,
                evaluation_schema=evaluation_schema,
                verbose=True,
                smoke_test=True
            )
        self.assertEqual(holdouts.shape[0],
                         self._number_of_holdouts*2*df.shape[0])

    def test_edge_label_prediction_models_apis(self):
        df = get_available_models_for_edge_label_prediction()
        graph = PDUMDV().remove_singleton_nodes()
        node_features = SPINE(embedding_size=10).fit_transform(graph)
        bar = tqdm(
            df.model_name,
            total=df.shape[0],
            leave=False,
        )
        for model_name in bar:
            bar.set_description(
                f"Testing API of {model_name}")
            model_class = AbstractEdgeLabelPredictionModel.get_model_from_library(model_name)
            model = model_class()
            use_edge_metrics = "use_edge_metrics" in model.parameters()
            model = model_class(
                **{
                    **model_class.smoke_test_parameters(),
                    **dict(use_edge_metrics=use_edge_metrics)
                }
            )

            model.fit(graph, node_features=node_features)
            model.predict(graph, node_features=node_features)
            model.predict_proba(graph, node_features=node_features)

    def test_evaluate_edge_label_prediction_with_node_types_features(self):
        df = get_available_models_for_edge_label_prediction()
        graph = PDUMDV().remove_singleton_nodes()
        holdouts = edge_label_prediction_evaluation(
            holdouts_kwargs=dict(train_size=0.8),
            models=df.model_name,
            library_names=df.library_name,
            node_features=SPINE(embedding_size=5),
            node_type_features=np.random.uniform(
                size=(graph.get_number_of_node_types(), 5)
            ),
            edge_features=np.random.uniform(
                size=(graph.get_number_of_directed_edges(), 5)
            ),
            evaluation_schema="Stratified Monte Carlo",
            graphs=graph,
            number_of_holdouts=self._number_of_holdouts,
            verbose=True,
            smoke_test=True
        )
        self.assertEqual(
            holdouts.shape[0],
            self._number_of_holdouts*2*df.shape[0]
        )

    def test_model_recreation(self):
        """Test graph visualization."""
        df = get_available_models_for_edge_label_prediction()

        for _, row in df.iterrows():
            model = AbstractEdgeLabelPredictionModel.get_model_from_library(
                model_name=row.model_name,
                task_name=AbstractEdgeLabelPredictionModel.task_name(),
                library_name=row.library_name
            )()
            try:
                AbstractEdgeLabelPredictionModel.get_model_from_library(
                    model_name=row.model_name,
                    task_name=AbstractEdgeLabelPredictionModel.task_name(),
                    library_name=row.library_name
                )(**model.parameters())
            except Exception as e:
                raise ValueError(
                    f"Found an error in model {row.model_name} "
                    f"implemented in library {row.library_name}."
                ) from e

    def test_all_embedding_models_as_feature(self):
        """Test graph visualization."""
        df = get_available_models_for_node_embedding()
        bar = tqdm(
            df.iterrows(),
            total=df.shape[0],
            leave=False,
            desc="Testing embedding methods"
        )
        graph = PDUMDV().remove_singleton_nodes().sort_by_decreasing_outbound_node_degree()
        for _, row in bar:
            if row.requires_edge_weights or row.requires_edge_types:
                continue

            bar.set_description(
                f"Testing {row.model_name} from library {row.library_name}")

            edge_label_prediction_evaluation(
                holdouts_kwargs=dict(train_size=0.8),
                models="Decision Tree Classifier",
                node_features=AbstractEmbeddingModel.get_model_from_library(
                    model_name=row.model_name,
                    library_name=row.library_name
                )(),
                evaluation_schema="Stratified Monte Carlo",
                graphs=graph,
                number_of_holdouts=self._number_of_holdouts,
                verbose=False,
                smoke_test=True,
            )
