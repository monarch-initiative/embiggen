"""Unit test class for GraphTransformer objects."""
from tqdm.auto import tqdm
from unittest import TestCase
from embiggen.edge_prediction import edge_prediction_evaluation
from embiggen import get_available_models_for_edge_prediction, get_available_models_for_node_embedding
from embiggen.edge_prediction.edge_prediction_model import AbstractEdgePredictionModel
from embiggen.edge_prediction.edge_prediction_tensorflow.graph_sage import GraphSAGEEdgePrediction
from embiggen.embedders import SPINE
from ensmallen.datasets.linqs import Cora, get_words_data
from ensmallen.datasets.kgobo import CIO
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
            AbstractEdgePredictionModel.get_model_from_library(
                model_name=row.model_name,
                task_name=AbstractEdgePredictionModel.task_name(),
                library_name=row.library_name
            )(**model.parameters())

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
            smoke_test=True
        )
        self.assertEqual(
            holdouts.shape[0], self._number_of_holdouts*2*2*df.shape[0])

    def test_edge_prediction_models_apis(self):
        df = get_available_models_for_edge_prediction()
        graph = CIO().remove_singleton_nodes()
        node_features = SPINE(embedding_size=10).fit_transform(graph)
        for model_name in tqdm(df.model_name, desc="Testing model APIs"):
            model = AbstractEdgePredictionModel.get_model_from_library(model_name)()
            model.fit(graph, node_features=node_features)
            model.predict(graph, node_features=node_features)
            model.predict_proba(graph, node_features=node_features)
            if "use_edge_metrics" in model.parameters():
                model = AbstractEdgePredictionModel.get_model_from_library(model_name)(
                    use_edge_metrics=True
                )
                model.fit(graph, node_features=node_features)
                model.predict(graph, node_features=node_features)
                model.predict_proba(graph, node_features=node_features)
            if "use_node_embedding" in model.parameters():
                model = AbstractEdgePredictionModel.get_model_from_library(model_name)(
                    use_node_embedding=True,
                    use_node_type_embedding=True
                )
                model.fit(graph, node_features=node_features)
                model.predict(graph, node_features=node_features)
                model.predict_proba(graph, node_features=node_features)

    def test_tree_with_cosine(self):
        for evaluation_schema in AbstractEdgePredictionModel.get_available_evaluation_schemas():
            holdouts = edge_prediction_evaluation(
                holdouts_kwargs=dict(train_size=0.8),
                models=DecisionTreeEdgePrediction(
                    edge_embedding_method="CosineSimilarity"),
                node_features=SPINE(embedding_size=10),
                evaluation_schema=evaluation_schema,
                graphs="CIO",
                number_of_holdouts=self._number_of_holdouts,
                verbose=True,
                smoke_test=True,
                validation_unbalance_rates=(1.0, 2.0,),
            )
            self.assertEqual(holdouts.shape[0], self._number_of_holdouts*2*2)
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
                graph_name = "Usair97"
                repository = "networkrepository"
            else:
                graph_name = "CIO"
                repository = "kgobo"

            bar.set_description(
                f"Testing {row.model_name} from library {row.library_name}")

            edge_prediction_evaluation(
                holdouts_kwargs=dict(train_size=0.8),
                models="Perceptron",
                node_features=row.model_name,
                evaluation_schema="Connected Monte Carlo",
                graphs=graph_name,
                repositories=repository,
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
