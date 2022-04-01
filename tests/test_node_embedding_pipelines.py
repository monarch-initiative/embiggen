"""Test to validate that the model GloVe works properly with graph walks."""
from unittest import TestCase
from embiggen.pipelines import compute_node_embedding, get_available_node_embedding_methods
from ensmallen.datasets.linqs import Cora


class TestNodeEmbeddingUtility(TestCase):
    """Unit test for model GloVe on graph walks."""

    def setUp(self):
        """Setting up objects to test CBOW model on graph walks."""
        super().setUp()
        self._graph = Cora().random_holdout(train_size=0.1)[0]

    def test_fit(self):
        """Test that model fitting behaves correctly and produced embedding has correct shape."""
        for node_embedding_method in get_available_node_embedding_methods():
            compute_node_embedding(
                self._graph,
                node_embedding_method_name=node_embedding_method,
                automatically_drop_unsupported_parameters=True,
                embedding_size=10,
                node_type_embedding_size=10,
                edge_type_embedding_size=10,
                fit_kwargs=dict(
                    epochs=1
                )
            )
