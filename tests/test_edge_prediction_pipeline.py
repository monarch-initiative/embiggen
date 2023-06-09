"""Unit test class for GraphTransformer objects."""
from tqdm.auto import tqdm
from unittest import TestCase
import pytest
import numpy as np
import pandas as pd
import os
from embiggen.edge_prediction import edge_prediction_evaluation
from embiggen import get_available_models_for_edge_prediction, get_available_models_for_node_embedding, get_available_models_for_edge_embedding
from embiggen.edge_prediction.edge_prediction_model import AbstractEdgePredictionModel
from embiggen.edge_prediction import GraphSAGEEdgePrediction
from embiggen.embedding_transformers import EdgeTransformer
from embiggen.edge_prediction.edge_prediction_ensmallen.perceptron import PerceptronEdgePrediction
from embiggen.embedders.ensmallen_embedders.degree_spine import DegreeSPINE
from embiggen.utils import AbstractEmbeddingModel
from ensmallen.datasets.linqs import Cora
from ensmallen.datasets.kgobo import CIO
from ensmallen import Graph
from embiggen.edge_prediction import DecisionTreeEdgePrediction
from embiggen.feature_preprocessors import GraphConvolution


class TestEvaluateEdgePrediction(TestCase):
    """Unit test class for GraphTransformer objects."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        self._graph = Cora().filter_from_names(
            max_node_degree=50,
        ).remove_singleton_nodes()
        self._graph_without_node_types = self._graph.remove_node_types()
        self._subgraph_of_interest = self._graph.filter_from_names(
            edge_type_names_to_keep=["Paper2Paper"]
        )
        self._number_of_holdouts = 2

    def test_model_recreation(self):
        df = get_available_models_for_edge_prediction()

        for _, row in df.iterrows():
            model = AbstractEdgePredictionModel.get_model_from_library(
                model_name=row.model_name,
                task_name=AbstractEdgePredictionModel.task_name(),
                library_name=row.library_name
            )()
            try:
                AbstractEdgePredictionModel.get_model_from_library(
                    model_name=row.model_name,
                    task_name=AbstractEdgePredictionModel.task_name(),
                    library_name=row.library_name
                )(**model.parameters())
            except Exception as e:
                raise ValueError(
                    f"Found an error in model {row.model_name} "
                    f"implemented in library {row.library_name}."
                ) from e

    def test_evaluate_edge_prediction(self):
        df = get_available_models_for_edge_prediction()
        holdouts = edge_prediction_evaluation(
            holdouts_kwargs=dict(train_size=0.9),
            models=df.model_name,
            library_names=df.library_name,
            node_features=DegreeSPINE(embedding_size=5),
            evaluation_schema="Connected Monte Carlo",
            graphs=[self._graph, self._graph_without_node_types],
            number_of_holdouts=self._number_of_holdouts,
            verbose=True,
            smoke_test=True,
        )
        self.assertEqual(
            holdouts.shape[0],
            self._number_of_holdouts*2*2*df.shape[0]
        )

    def test_evaluate_edge_prediction_with_subgraphs(self):
        df = get_available_models_for_edge_prediction()
        holdouts = edge_prediction_evaluation(
            holdouts_kwargs=dict(train_size=0.9),
            models=df.model_name,
            library_names=df.library_name,
            node_features=DegreeSPINE(embedding_size=5),
            evaluation_schema="Connected Monte Carlo",
            graphs=self._graph,
            number_of_holdouts=self._number_of_holdouts,
            verbose=True,
            smoke_test=True,
            subgraph_of_interest=self._subgraph_of_interest
        )
        self.assertEqual(
            holdouts.shape[0], self._number_of_holdouts*2*df.shape[0])

    def test_evaluate_edge_prediction_with_node_types_features(self):
        df = get_available_models_for_edge_prediction()
        holdouts = edge_prediction_evaluation(
            holdouts_kwargs=dict(train_size=0.9),
            models=df.model_name,
            library_names=df.library_name,
            node_features=DegreeSPINE(embedding_size=5),
            node_type_features=np.random.uniform(
                size=(self._graph.get_number_of_node_types(), 5)
            ),
            evaluation_schema="Connected Monte Carlo",
            graphs=self._graph,
            number_of_holdouts=self._number_of_holdouts,
            verbose=True,
            smoke_test=True
        )
        self.assertEqual(
            holdouts.shape[0],
            self._number_of_holdouts * 2 * df.shape[0]
        )

    def test_edge_prediction_models_apis(self):
        df = get_available_models_for_edge_prediction()
        graph = CIO().remove_singleton_nodes()
        multi_label_graph = graph.set_all_node_types("hu") | graph
        node_features = DegreeSPINE(embedding_size=10).fit_transform(graph)
        bar = tqdm(
            df.model_name,
            total=df.shape[0],
            leave=False,
        )
        for g in (multi_label_graph, graph):
            first_names = g.get_node_names()[:5]
            second_names = g.get_node_names()[5:10]
            for model_name in bar:
                bar.set_description(
                    f"Testing API of {model_name}"
                )
                model = AbstractEdgePredictionModel.get_model_from_library(
                    model_name
                )()
                parameters = {}
                if "use_edge_metrics" in model.parameters():
                    parameters["use_edge_metrics"] = True
                if "use_node_embedding" in model.parameters():
                    parameters["use_node_embedding"] = True
                if "use_node_type_embedding" in model.parameters():
                    parameters["use_node_type_embedding"] = True
                model_class = AbstractEdgePredictionModel.get_model_from_library(
                    model_name
                )
                node_type_features = np.random.uniform(size=(
                    g.get_number_of_node_types(),
                    7
                ))
                model = model_class(
                    **{
                        **model_class.smoke_test_parameters(),
                        **parameters
                    },
                )
                model.fit(
                    g,
                    node_features=node_features,
                    node_type_features=node_type_features
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

                # Check that the restored model is the same as the original one
                # and that the parameters are the same.
                # We must remove the NaN values from the restored model
                # because they are by definition not equal to themselves.

                restored_model_parameters = {
                    key: value
                    for key, value in restored_model.parameters().items()
                    if isinstance(value, (list, tuple, np.ndarray)) or  pd.notna(value)
                }

                model_parameters = {
                    key: value
                    for key, value in model.parameters().items()
                    if isinstance(value, (list, tuple, np.ndarray)) or  pd.notna(value)
                }

                self.assertEqual(
                    restored_model_parameters,
                    model_parameters
                )

                with pytest.raises(NotImplementedError):
                    model.fit(
                        g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        edge_features=67
                    )
                with pytest.raises(NotImplementedError):
                    model.predict(
                        g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        edge_features=67
                    )
                with pytest.raises(NotImplementedError):
                    model.predict_proba(
                        g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        edge_features=67
                    )
                for return_predictions_dataframe in (True, False):
                    model.predict(
                        g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        return_predictions_dataframe=return_predictions_dataframe
                    )
                    model.predict_proba(
                        g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        return_predictions_dataframe=return_predictions_dataframe
                    )
                    model.predict_bipartite_graph_from_edge_node_ids(
                        graph=g,
                        source_node_ids=[0, 1, 2, 3],
                        destination_node_ids=[4, 5, 6, 7],
                        support=g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        return_predictions_dataframe=return_predictions_dataframe
                    )
                    model.predict_proba_bipartite_graph_from_edge_node_ids(
                        graph=g,
                        source_node_ids=[0, 1, 2, 3],
                        destination_node_ids=[4, 5, 6, 7],
                        support=g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        return_predictions_dataframe=return_predictions_dataframe
                    )
                    model.predict_bipartite_graph_from_edge_node_names(
                        graph=g,
                        source_node_names=first_names,
                        destination_node_names=second_names,
                        support=g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        return_predictions_dataframe=return_predictions_dataframe
                    )
                    model.predict_proba_bipartite_graph_from_edge_node_names(
                        graph=g,
                        source_node_names=first_names,
                        destination_node_names=second_names,
                        support=g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        return_predictions_dataframe=return_predictions_dataframe
                    )
                    model.predict_bipartite_graph_from_edge_node_prefixes(
                        graph=g,
                        source_node_prefixes=["OIO"],
                        destination_node_prefixes=["IAO", "CIO"],
                        support=g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        return_predictions_dataframe=return_predictions_dataframe
                    )
                    model.predict_proba_bipartite_graph_from_edge_node_prefixes(
                        graph=g,
                        source_node_prefixes=["OIO"],
                        destination_node_prefixes=["IAO", "CIO"],
                        support=g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        return_predictions_dataframe=return_predictions_dataframe
                    )
                    model.predict_clique_graph_from_node_ids(
                        graph=g,
                        node_ids=[0, 1, 2],
                        support=g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        return_predictions_dataframe=return_predictions_dataframe
                    )
                    model.predict_proba_clique_graph_from_node_ids(
                        graph=g,
                        node_ids=[0, 1, 2],
                        support=g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        return_predictions_dataframe=return_predictions_dataframe
                    )
                    model.predict_clique_graph_from_node_names(
                        graph=g,
                        node_names=first_names,
                        support=g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        return_predictions_dataframe=return_predictions_dataframe
                    )
                    model.predict_proba_clique_graph_from_node_names(
                        graph=g,
                        node_names=first_names,
                        support=g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        return_predictions_dataframe=return_predictions_dataframe
                    )
                    model.predict_clique_graph_from_node_prefixes(
                        graph=g,
                        node_prefixes=["OIO:"],
                        support=g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        return_predictions_dataframe=return_predictions_dataframe
                    )
                    model.predict_proba_clique_graph_from_node_prefixes(
                        graph=g,
                        node_prefixes=["OIO:"],
                        support=g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        return_predictions_dataframe=return_predictions_dataframe
                    )
                    model.predict_clique_graph_from_node_type_names(
                        graph=g,
                        node_type_names=["biolink:NamedThing"],
                        support=g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        return_predictions_dataframe=return_predictions_dataframe
                    )
                    model.predict_proba_clique_graph_from_node_type_names(
                        graph=g,
                        node_type_names=["biolink:NamedThing"],
                        support=g,
                        node_features=node_features,
                        node_type_features=node_type_features,
                        return_predictions_dataframe=return_predictions_dataframe
                    )

    def test_tree_with_cosine(self):
        graph = CIO().remove_singleton_nodes().sort_by_decreasing_outbound_node_degree()
        for edge_embedding in EdgeTransformer.methods:
            for evaluation_schema in AbstractEdgePredictionModel.get_available_evaluation_schemas():
                model = DecisionTreeEdgePrediction(
                    edge_embedding_method=edge_embedding
                )
                holdouts = edge_prediction_evaluation(
                    holdouts_kwargs=dict(train_size=0.9),
                    models=DecisionTreeEdgePrediction(
                        **{
                            **model.parameters(),
                            **model.smoke_test_parameters(),
                        }
                    ),
                    node_features=DegreeSPINE(embedding_size=10),
                    node_features_preprocessing_steps=GraphConvolution(
                        concatenate_features=True
                    ),
                    evaluation_schema=evaluation_schema,
                    graphs=graph,
                    number_of_holdouts=self._number_of_holdouts,
                    verbose=True,
                    validation_unbalance_rates=(1.0, 2.0,),
                )
                self.assertEqual(
                    holdouts.shape[0], self._number_of_holdouts*2*2)
                self.assertTrue(set(holdouts.validation_unbalance_rate)
                                == set((1.0, 2.0)))

    def test_all_node_embedding_models_as_feature(self):
        df = get_available_models_for_node_embedding()
        bar = tqdm(
            df.iterrows(),
            total=df.shape[0],
            leave=False,
            desc="Testing embedding methods"
        )

        complex_model = [
            "RotatE",
            "ComplEx",
        ]

        for _, row in bar:
            if row.requires_node_types:
                continue
            if row.requires_edge_weights:
                graph = Graph.generate_random_connected_graph(
                    random_state=100,
                    number_of_nodes=100,
                    weight=56.0,
                    edge_type="red"
                ) | Graph.generate_random_connected_graph(
                    random_state=1670,
                    number_of_nodes=100,
                    weight=526.0,
                    edge_type="red"
                )
            else:
                graph = CIO()

            if row.model_name in complex_model:
                continue

            bar.set_description(
                f"Testing {row.model_name} from library {row.library_name}")

            embedding_model = AbstractEmbeddingModel.get_model_from_library(
                model_name=row.model_name,
                library_name=row.library_name,
                task_name="Node Embedding"
            )()

            if embedding_model.requires_nodes_sorted_by_decreasing_node_degree():
                continue

            edge_prediction_evaluation(
                holdouts_kwargs=dict(train_size=0.9),
                models=PerceptronEdgePrediction(
                    edge_embeddings="CosineSimilarity"),
                node_features=embedding_model,
                evaluation_schema="Connected Monte Carlo",
                node_features_preprocessing_steps=GraphConvolution(),
                graphs=graph.remove_singleton_nodes(),
                number_of_holdouts=self._number_of_holdouts,
                verbose=False,
                smoke_test=True,
            )

    def test_all_edge_embedding_methods(self):
        if not GraphSAGEEdgePrediction.is_available():
            return
        for edge_embedding_method in GraphSAGEEdgePrediction.get_available_edge_embedding_methods():
            edge_prediction_evaluation(
                holdouts_kwargs=dict(train_size=0.9),
                models=GraphSAGEEdgePrediction(
                    edge_embedding_method=edge_embedding_method,
                    epochs=1
                ),
                evaluation_schema="Connected Monte Carlo",
                graphs=self._graph,
                node_features="Degree-based SPINE",
                node_features_preprocessing_steps=GraphConvolution(),
                number_of_holdouts=self._number_of_holdouts,
                verbose=False
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
        for _, row in bar:
            if row.requires_node_types:
                continue
            if row.requires_edge_weights:
                graph = Graph.generate_random_connected_graph(
                    random_state=100,
                    number_of_nodes=100,
                    weight=56.0,
                    edge_type="red"
                ) | Graph.generate_random_connected_graph(
                    random_state=1670,
                    number_of_nodes=100,
                    weight=526.0,
                    edge_type="red"
                )
            else:
                graph = CIO()

            bar.set_description(
                f"Testing {row.model_name} from library {row.library_name}")

            edge_prediction_evaluation(
                holdouts_kwargs=dict(train_size=0.8),
                models="Decision Tree Classifier",
                edge_features=AbstractEmbeddingModel.get_model_from_library(
                    model_name=row.model_name,
                    library_name=row.library_name,
                    task_name="Edge Embedding"
                )(),
                evaluation_schema="Monte Carlo",
                graphs=graph,
                number_of_holdouts=self._number_of_holdouts,
                verbose=False,
                smoke_test=True,
            )
