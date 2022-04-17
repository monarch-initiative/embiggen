"""Test to validate that the abstract model Node2Vec works properly."""
import pytest
from embiggen.embedders.tensorflow_embedders.node2vec import Node2Vec
from .test_node_sequences import TestNodeSequences


class TestNode2Vec(TestNodeSequences):
    """Unit test to validate that the abstract model Node2Vec works properly."""

    def setUp(self):
        """Setting up objects for testing Node2Vec abstract model."""
        super().setUp()
        self._embedding_size = 50
        self._words, self._ctxs, self._freq = self._graph.cooccurence_matrix(
            80,
            window_size=4,
            iterations=20
        )
        self._model = Node2Vec(
            vocabulary_size=self._graph.get_nodes_number(),
            embedding_size=self._embedding_size,
            model_name="Node2Vec"
        )

        with pytest.raises(ValueError):
            self._model._get_true_input_length()

        with pytest.raises(ValueError):
            self._model._get_true_output_length()

        with pytest.raises(ValueError):
            self._model._sort_input_layers(None, None)
