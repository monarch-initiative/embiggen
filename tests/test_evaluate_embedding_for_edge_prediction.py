"""Unit test class for GraphTransformer objects."""
from unittest import TestCase
from ensmallen import Graph  # pylint: disable=no-name-in-module
from embiggen import GraphVisualization, GloVe
import pytest
from embiggen.pipelines import evaluate_embedding_for_edge_prediction
from plot_keras_history import plot_history
import matplotlib.pyplot as plt

class TestEvaluateEmbeddingForEdgePrediction(TestCase):
    """Unit test class for GraphTransformer objects."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        self._embedding_size = 50
        self._graph = Graph.from_csv(
            edge_path="tests/data/small_ppi.tsv",
            sources_column="subject",
            destinations_column="object",
            directed=False,
            edge_list_edge_types_column="edge_label"
        )
        self._embedding = GloVe(
            vocabulary_size=self._graph.get_nodes_number(),
            embedding_size=self._embedding_size
        ).get_embedding_dataframe(self._graph.get_node_names())

    def test_evaluate_embedding_for_edge_prediction(self):
        """Test graph visualization."""
        evaluate_embedding_for_edge_prediction(
            embedding=self._embedding,
            graph=self._graph,
            model_name="Perceptron",
            number_of_holdouts=3,
        )