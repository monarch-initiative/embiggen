"""Unit test class for Node-label prediction pipeline."""
from tqdm.auto import tqdm
from unittest import TestCase
from embiggen.node_label_prediction import node_label_prediction_evaluation
from embiggen import get_available_models_for_node_label_prediction, get_available_models_for_node_embedding
from embiggen.embedders import SPINE
from ensmallen.datasets.kgobo import MIAPA
from ensmallen.datasets.linqs import Cora, get_words_data
import shutil
import os
from embiggen.node_label_prediction.node_label_prediction_model import AbstractNodeLabelPredictionModel


class TestEvaluateNodeLabelPrediction(TestCase):
    """Unit test class for Node-label prediction pipeline."""

    def setUp(self):
        """Setup objects for running tests on Node-label prediction pipeline class."""
        self._graph, self._data = get_words_data(Cora())
        self._number_of_holdouts = 2

    def test_evaluate_embedding_for_node_label_prediction(self):
        """Test graph visualization."""
        if os.path.exists("experiments"):
            shutil.rmtree("experiments")

        df = get_available_models_for_node_label_prediction()

        for evaluation_schema in AbstractNodeLabelPredictionModel.get_available_evaluation_schemas():
            holdouts = node_label_prediction_evaluation(
                holdouts_kwargs={
                    "train_size": 0.8
                },
                node_features=[SPINE(embedding_size=5), self._data],
                models=df.model_name,
                library_names=df.library_name,
                graphs=self._graph,
                verbose=False,
                evaluation_schema=evaluation_schema,
                number_of_holdouts=self._number_of_holdouts,
                smoke_test=True
            )

        self.assertEqual(holdouts.shape[0],
                         self._number_of_holdouts*2*df.shape[0])

        if os.path.exists("experiments"):
            shutil.rmtree("experiments")

    def test_node_label_prediction_models_apis(self):
        df = get_available_models_for_node_label_prediction()
        graph = self._graph.remove_singleton_nodes()
        node_features = SPINE(embedding_size=10).fit_transform(graph)
        for model_name in tqdm(df.model_name, desc="Testing model APIs"):
            model = AbstractNodeLabelPredictionModel.get_model_from_library(
                model_name
            )().into_smoke_test()
            model.fit(graph, node_features=node_features)
            model.predict(graph, node_features=node_features)
            model.predict_proba(graph, node_features=node_features)

    def test_model_recreation(self):
        """Test graph visualization."""
        df = get_available_models_for_node_label_prediction()

        for _, row in df.iterrows():
            model = AbstractNodeLabelPredictionModel.get_model_from_library(
                model_name=row.model_name,
                task_name=AbstractNodeLabelPredictionModel.task_name(),
                library_name=row.library_name
            )()
            try:
                AbstractNodeLabelPredictionModel.get_model_from_library(
                    model_name=row.model_name,
                    task_name=AbstractNodeLabelPredictionModel.task_name(),
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
        for _, row in bar:
            if row.requires_edge_weights:
                continue

            bar.set_description(
                f"Testing {row.model_name} from library {row.library_name}")

            node_label_prediction_evaluation(
                holdouts_kwargs=dict(train_size=0.8),
                models="Decision Tree Classifier",
                node_features=row.model_name,
                graphs=MIAPA().remove_singleton_nodes().sort_by_decreasing_outbound_node_degree(),
                number_of_holdouts=self._number_of_holdouts,
                evaluation_schema="Monte Carlo",
                verbose=False,
                smoke_test=True,
            )
