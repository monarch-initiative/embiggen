"""Unit test class for GraphTransformer objects."""
from tabnanny import verbose
from tqdm.auto import tqdm
from unittest import TestCase
import numpy as np
from embiggen.edge_prediction import edge_prediction_evaluation
from embiggen import get_available_models_for_edge_prediction, get_available_models_for_node_embedding
from embiggen.edge_prediction.edge_prediction_model import AbstractEdgePredictionModel
from embiggen.edge_prediction.edge_prediction_tensorflow.graph_sage import GraphSAGEEdgePrediction
from embiggen.embedding_transformers import EdgeTransformer
from embiggen.embedders import SPINE
from embiggen.utils import AbstractEmbeddingModel
from ensmallen.datasets.linqs import Cora, get_words_data
from ensmallen.datasets.kgobo import CIO
from ensmallen.datasets.networkrepository import Usair97
from embiggen.edge_prediction import DecisionTreeEdgePrediction


class TestEvaluateEdgePrediction(TestCase):
    """Unit test class for GraphTransformer objects."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        self._graph, _ = get_words_data(Cora())
        self._graph = self._graph.remove_singleton_nodes()
        self._graph_without_node_types = self._graph.remove_node_types()
        self._subgraph_of_interest = self._graph.filter_from_names(
            source_node_type_name_to_keep=["Neural_Networks"]
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
            holdouts_kwargs=dict(train_size=0.8),
            models=df.model_name,
            library_names=df.library_name,
            node_features=SPINE(embedding_size=5),
            evaluation_schema="Connected Monte Carlo",
            graphs=[self._graph, self._graph_without_node_types],
            number_of_holdouts=self._number_of_holdouts,
            verbose=True,
            smoke_test=True,
        )
        self.assertEqual(
            holdouts.shape[0], self._number_of_holdouts*2*2*df.shape[0])

    def test_evaluate_edge_prediction_with_subgraphs(self):
        df = get_available_models_for_edge_prediction()
        holdouts = edge_prediction_evaluation(
            holdouts_kwargs=dict(train_size=0.8),
            models=df.model_name,
            library_names=df.library_name,
            node_features=SPINE(embedding_size=5),
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
            holdouts_kwargs=dict(train_size=0.8),
            models=df.model_name,
            library_names=df.library_name,
            node_features=SPINE(embedding_size=5),
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
        graph, _ = graph.get_node_label_holdout_graphs(0.8)
        multi_label_graph, _ = multi_label_graph.get_node_label_holdout_graphs(
            0.8)
        node_features = SPINE(embedding_size=10).fit_transform(graph)

        bar = tqdm(
            df.model_name,
            total=df.shape[0],
            leave=False,
        )
        for g in (multi_label_graph, graph):
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
                model.predict(
                    g,
                    node_features=node_features,
                    node_type_features=node_type_features
                )
                model.predict_proba(
                    g,
                    node_features=node_features,
                    node_type_features=node_type_features
                )

    def test_tree_with_cosine(self):
        graph = CIO().remove_singleton_nodes().sort_by_decreasing_outbound_node_degree()
        for edge_embedding in EdgeTransformer.methods:
            for evaluation_schema in AbstractEdgePredictionModel.get_available_evaluation_schemas():
                model = DecisionTreeEdgePrediction(
                    edge_embedding_method=edge_embedding
                )
                holdouts = edge_prediction_evaluation(
                    holdouts_kwargs=dict(train_size=0.8),
                    models=DecisionTreeEdgePrediction(
                        **{
                            **model.parameters(),
                            **model.smoke_test_parameters(),
                        }
                    ),
                    node_features=SPINE(embedding_size=10),
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

    def test_all_embedding_models_as_feature(self):
        df = get_available_models_for_node_embedding()
        bar = tqdm(
            df.iterrows(),
            total=df.shape[0],
            leave=False,
            desc="Testing embedding methods"
        )
        for _, row in bar:
            if row.requires_edge_weights:
                graph_name = Usair97
            else:
                graph_name = CIO

            bar.set_description(
                f"Testing {row.model_name} from library {row.library_name}")

            embedding_model = AbstractEmbeddingModel.get_model_from_library(
                model_name=row.model_name,
                library_name=row.library_name
            )()

            if embedding_model.requires_nodes_sorted_by_decreasing_node_degree():
                continue

            edge_prediction_evaluation(
                holdouts_kwargs=dict(train_size=0.8),
                models="Perceptron",
                node_features=embedding_model,
                evaluation_schema="Connected Monte Carlo",
                graphs=graph_name().remove_singleton_nodes(),
                number_of_holdouts=self._number_of_holdouts,
                verbose=False,
                smoke_test=True,
            )

    def test_all_edge_embedding_methods(self):
        for edge_embedding_method in GraphSAGEEdgePrediction.get_available_edge_embedding_methods():
            edge_prediction_evaluation(
                holdouts_kwargs=dict(train_size=0.8),
                models=GraphSAGEEdgePrediction(
                    edge_embedding_method=edge_embedding_method,
                    epochs=1
                ),
                evaluation_schema="Connected Monte Carlo",
                graphs=self._graph,
                node_features="SPINE",
                number_of_holdouts=self._number_of_holdouts,
                verbose=False
            )
