"""Unit test class for Node-label prediction pipeline."""
from tqdm.auto import tqdm
from unittest import TestCase
import pytest
import numpy as np
import pandas as pd
from embiggen.node_label_prediction import node_label_prediction_evaluation
from embiggen import (
    get_available_models_for_node_label_prediction,
    get_available_models_for_node_embedding,
)
from embiggen.embedders.ensmallen_embedders.degree_spine import DegreeSPINE
from ensmallen.datasets.kgobo import MIAPA
from ensmallen import Graph
from ensmallen.datasets.linqs import Cora, get_words_data
import shutil
import os
from embiggen.utils import AbstractEmbeddingModel
from embiggen.node_label_prediction.node_label_prediction_model import (
    AbstractNodeLabelPredictionModel,
)
from embiggen.feature_preprocessors import GraphConvolution


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
        feature = DegreeSPINE(embedding_size=5)
        for (
            evaluation_schema
        ) in AbstractNodeLabelPredictionModel.get_available_evaluation_schemas():
            holdouts = node_label_prediction_evaluation(
                holdouts_kwargs={"train_size": 0.8},
                node_features=[feature, self._data],
                models=df.model_name,
                library_names=df.library_name,
                graphs=self._graph,
                verbose=False,
                evaluation_schema=evaluation_schema,
                number_of_holdouts=self._number_of_holdouts,
                smoke_test=True,
            )

        self.assertEqual(holdouts.shape[0], self._number_of_holdouts * 2 * df.shape[0])

        if os.path.exists("experiments"):
            shutil.rmtree("experiments")

    def test_node_label_prediction_models_apis(self):
        df = get_available_models_for_node_label_prediction()
        graph = self._graph.remove_singleton_nodes()
        red = self._graph.set_all_node_types("red")
        multilabel_graph = (
            Cora().remove_edge_weights().remove_edge_types() | red
        ).add_selfloops()
        binary_graph = (
            red | MIAPA().remove_edge_types().set_all_node_types("blue")
        ).add_selfloops()
        for g in (graph, multilabel_graph, binary_graph):
            node_features = DegreeSPINE(embedding_size=10).fit_transform(g)
            for model_name in tqdm(df.model_name, desc="Testing model APIs"):
                model = AbstractNodeLabelPredictionModel.get_model_from_library(
                    model_name
                )().into_smoke_test()

                if (
                    g.has_multilabel_node_types()
                    and not model.supports_multilabel_prediction()
                ):
                    with pytest.raises(ValueError):
                        model.fit(g, node_features=node_features)

                    continue
                model.fit(g, node_features=node_features)
                if model.library_name() in ("TensorFlow", "scikit-learn", "LightGBM", "CatBoost", "XGBoost"):
                    path = "model.pkl.gz"
                elif model.library_name() == "Ensmallen":
                    path = "model.json"
                else:
                    raise NotImplementedError(
                        f"The model {model.model_name()} from library {model.library_name()} "
                        f"for the task {model.task_name()} "
                        "is not currently covered by the test suite! "
                        "Please add the model to the test suite."
                    )
                if os.path.exists(path):
                    os.remove(path)
                model.dump(path)
                restored_model = model.load(path)
                
                restored_model_parameters = {
                    key: value
                    for key, value in restored_model.parameters().items()
                    if isinstance(value, (list, tuple, np.ndarray)) or pd.notna(value)
                }

                model_parameters = {
                    key: value
                    for key, value in model.parameters().items()
                    if isinstance(value, (list, tuple, np.ndarray)) or pd.notna(value)
                }

                self.assertEqual(
                    restored_model_parameters,
                    model_parameters
                )

                model.predict(g, node_features=node_features)
                model.predict_proba(g, node_features=node_features)

    def test_model_recreation(self):
        """Test graph visualization."""
        df = get_available_models_for_node_label_prediction()
        for _, row in df.iterrows():
            model = AbstractNodeLabelPredictionModel.get_model_from_library(
                model_name=row.model_name,
                task_name=AbstractNodeLabelPredictionModel.task_name(),
                library_name=row.library_name,
            )()
            try:
                AbstractNodeLabelPredictionModel.get_model_from_library(
                    model_name=row.model_name,
                    task_name=AbstractNodeLabelPredictionModel.task_name(),
                    library_name=row.library_name,
                )(**model.parameters())
            except Exception as e:
                raise ValueError(
                    f"Found an error in model {row.model_name} "
                    f"implemented in library {row.library_name}."
                ) from e

    def test_all_node_embedding_models_as_feature(self):
        """Test graph visualization."""
        df = get_available_models_for_node_embedding()
        bar = tqdm(
            df.iterrows(),
            total=df.shape[0],
            leave=False,
            desc="Testing embedding methods",
        )
        graph = Graph.generate_random_connected_graph(
            random_state=100,
            number_of_nodes=100,
            node_type="red",
        ) | Graph.generate_random_connected_graph(
            minimum_node_id=100,
            random_state=1670,
            number_of_nodes=100,
            node_type="blue",
        )

        complex_model = [
            "RotatE",
            "ComplEx",
        ]

        graph = graph.sort_by_decreasing_outbound_node_degree()

        for _, row in bar:
            if row.requires_edge_weights:
                continue

            if row.model_name in complex_model:
                continue

            bar.set_description(
                f"Testing {row.model_name} from library {row.library_name}"
            )

            node_label_prediction_evaluation(
                holdouts_kwargs=dict(train_size=0.8),
                models="Decision Tree Classifier",
                node_features=AbstractEmbeddingModel.get_model_from_library(
                    model_name=row.model_name,
                    library_name=row.library_name,
                    task_name="Node Embedding",
                )(),
                node_features_preprocessing_steps=GraphConvolution(),
                graphs=graph,
                number_of_holdouts=self._number_of_holdouts,
                evaluation_schema="Monte Carlo",
                verbose=False,
                smoke_test=True,
            )
