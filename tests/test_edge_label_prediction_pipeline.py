"""Unit test class for Node-label prediction pipeline."""
from tqdm.auto import tqdm
from unittest import TestCase
import pytest
import os
import numpy as np
import pandas as pd
from embiggen.edge_label_prediction import edge_label_prediction_evaluation
from embiggen import get_available_models_for_edge_label_prediction, get_available_models_for_node_embedding, get_available_models_for_edge_embedding
from embiggen.edge_label_prediction.edge_label_prediction_model import AbstractEdgeLabelPredictionModel
from embiggen.utils import AbstractEmbeddingModel
from ensmallen.datasets.kgobo import HP, CIO
from embiggen.embedders.ensmallen_embedders.degree_spine import DegreeSPINE
from embiggen.feature_preprocessors import GraphConvolution


class TestEvaluateEdgeLabelPrediction(TestCase):
    """Unit test class for edge-label prediction pipeline."""

    def setUp(self):
        """Setup objects for running tests on edge-label prediction pipeline class."""
        self._number_of_holdouts = 2
        self.graph = HP().remove_singleton_nodes().remove_parallel_edges()

    def test_evaluate_embedding_for_edge_label_prediction(self):
        """Test graph visualization."""
        df = get_available_models_for_edge_label_prediction()
        feature = DegreeSPINE(embedding_size=5)
        red = self.graph.set_all_edge_types("red")
        blue = CIO().remove_singleton_nodes().set_all_edge_types("blue")
        binary_graph = red | blue
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
                graphs=[self.graph, binary_graph],
                number_of_holdouts=self._number_of_holdouts,
                evaluation_schema=evaluation_schema,
                verbose=True,
                smoke_test=True
            )
        self.assertEqual(holdouts.shape[0],
                         self._number_of_holdouts*2*2*df.shape[0])

    def test_edge_label_prediction_models_apis(self):
        df = get_available_models_for_edge_label_prediction()
        graph = self.graph
        red = graph.set_all_edge_types("red")
        blue = CIO().remove_singleton_nodes().set_all_edge_types("blue")
        binary_graph = red | blue
        bar = tqdm(
            df.model_name,
            total=df.shape[0],
            leave=False,
        )
        for g in (graph, binary_graph):
            node_features = DegreeSPINE(embedding_size=10).fit_transform(g)
            edge_features=np.random.uniform(
                size=(g.get_number_of_edges(), 5)
            )
            for ef in (edge_features, None):
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

                    model.fit(
                        g,
                        node_features=node_features,
                        edge_features=ef
                    )

                    if model.library_name() in ("TensorFlow", "scikit-learn", "LightGBM", "CatBoost", "XGBoost"):
                        path = "model.pkl.gz"
                    elif model.library_name() == "Ensmallen":
                        path = "model.json"
                    else:
                        raise NotImplementedError(
                            f"The model {model.model_name()} from library {model.library_name()} "
                            "is not currently covered by the test suite!"
                        )
                    if os.path.exists(path):
                        os.remove(path)
                    
                    model.dump(path)
                    restored_model = model.load(path)

                    # We get the restored model parameters and the
                    # original ones and check that they are the same.
                    # We must be careful to remove the np.Nan values
                    # from both parameters, as they are not equal to
                    # themselves.

                    restored_model_parameters = {
                        k: v for k, v in restored_model.parameters().items()
                        if isinstance(v, (list, tuple, np.ndarray)) or pd.notna(v)
                    }

                    model_parameters = {
                        k: v for k, v in model.parameters().items()
                        if isinstance(v, (list, tuple, np.ndarray)) or pd.notna(v)
                    }

                    assert restored_model_parameters == model_parameters

                    model.predict(
                        g,
                        node_features=node_features,
                        edge_features=ef
                    )
                    model.predict_proba(
                        g,
                        node_features=node_features,
                        edge_features=ef
                    )

    def test_evaluate_edge_label_prediction_with_node_types_features(self):
        df = get_available_models_for_edge_label_prediction()
        graph = self.graph
        holdouts = edge_label_prediction_evaluation(
            holdouts_kwargs=dict(train_size=0.8),
            models=df.model_name,
            library_names=df.library_name,
            node_features=DegreeSPINE(embedding_size=5),
            node_type_features=np.random.uniform(
                size=(graph.get_number_of_node_types(), 5)
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

    def test_evaluate_edge_label_prediction_with_node_types_features(self):
        # We expect this to fail because of the wrong number of edge
        # features. Specifically, this graph is undirected, but the
        # edge features are not, so they are double the expected size.
        with pytest.raises(ValueError):
            df = get_available_models_for_edge_label_prediction()
            graph = self.graph
            edge_label_prediction_evaluation(
                holdouts_kwargs=dict(train_size=0.8),
                models=df.model_name,
                library_names=df.library_name,
                node_features=DegreeSPINE(embedding_size=5),
                edge_features=np.random.uniform(
                    size=(graph.get_number_of_directed_edges(), 5)
                ),
                evaluation_schema="Stratified Monte Carlo",
                graphs=graph,
                number_of_holdouts=self._number_of_holdouts,
                verbose=True,
                smoke_test=True
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

    def test_all_node_embedding_models_as_feature(self):
        """Test graph visualization."""
        df = get_available_models_for_node_embedding()
        bar = tqdm(
            df.iterrows(),
            total=df.shape[0],
            leave=False,
            desc="Testing node embedding methods"
        )
        graph = HP().remove_components(top_k_components=1).sort_by_decreasing_outbound_node_degree()

        for _, row in bar:
            if row.requires_edge_weights or row.requires_edge_types or row.requires_node_types:
                continue

            bar.set_description(
                f"Testing {row.model_name} from library {row.library_name}")

            edge_label_prediction_evaluation(
                holdouts_kwargs=dict(train_size=0.8),
                models="Decision Tree Classifier",
                node_features=AbstractEmbeddingModel.get_model_from_library(
                    model_name=row.model_name,
                    library_name=row.library_name,
                    task_name="Node Embedding"
                )(),
                node_features_preprocessing_steps=GraphConvolution(),
                evaluation_schema="Stratified Monte Carlo",
                graphs=graph,
                number_of_holdouts=self._number_of_holdouts,
                verbose=False,
                smoke_test=True,
            )

    def test_all_edge_embedding_models_as_feature(self):
        """Test graph visualization."""
        df = get_available_models_for_edge_embedding()
        bar = tqdm(
            df.iterrows(),
            total=df.shape[0],
            leave=False,
            desc="Testing edge embedding methods"
        )
        graph = HP().remove_components(top_k_components=1).sort_by_decreasing_outbound_node_degree()

        for _, row in bar:
            if row.requires_edge_weights or row.requires_edge_types or row.requires_node_types:
                continue

            bar.set_description(
                f"Testing {row.model_name} from library {row.library_name}")

            edge_label_prediction_evaluation(
                holdouts_kwargs=dict(train_size=0.8),
                models="Decision Tree Classifier",
                edge_features=AbstractEmbeddingModel.get_model_from_library(
                    model_name=row.model_name,
                    library_name=row.library_name,
                    task_name="Edge Embedding"
                )(),
                evaluation_schema="Stratified Monte Carlo",
                graphs=graph,
                number_of_holdouts=self._number_of_holdouts,
                verbose=False,
                smoke_test=True,
            )
