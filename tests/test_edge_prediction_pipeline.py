"""Unit test class for GraphTransformer objects."""
from unittest import TestCase
from embiggen import edge_prediction_evaluation, get_available_models_for_edge_prediction, SPINE
from ensmallen.datasets.linqs import Cora, get_words_data
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

    def test_evaluate_edge_prediction(self):
        """Test graph visualization."""
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
        self.assertEqual(holdouts.shape[0], self._number_of_holdouts*2*2*df.shape[0])

    def test_evaluate_edge_prediction_in_subgraph(self):
        """Test graph visualization."""
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
        self.assertEqual(holdouts.shape[0], self._number_of_holdouts*2*df.shape[0])

    def test_tree_with_cosine(self):
        """Test graph visualization."""
        holdouts = edge_prediction_evaluation(
            holdouts_kwargs=dict(train_size=0.8),
            models=DecisionTreeEdgePrediction(edge_embedding_method="CosineSimilarity"),
            node_features=SPINE(embedding_size=10),
            evaluation_schema="Connected Monte Carlo",
            graphs=self._graph,
            number_of_holdouts=self._number_of_holdouts,
            verbose=True,
            smoke_test=True,
            validation_unbalance_rates=(1.0, 2.0,),
            subgraph_of_interest=self._subgraph_of_interest
        )
        self.assertEqual(holdouts.shape[0], self._number_of_holdouts*2*2)
        self.assertTrue(set(holdouts.validation_unbalance_rate) == set((1.0, 2.0)))