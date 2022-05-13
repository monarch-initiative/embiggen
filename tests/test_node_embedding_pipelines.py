"""Test to validate that the model GloVe works properly with graph walks."""
from unittest import TestCase
from embiggen import embed_graph, get_available_models_for_node_embedding


class TestNodeEmbeddingPipeline(TestCase):
    """Unit test for model GloVe on graph walks."""

    def setUp(self):
        """Setting up objects to test CBOW model on graph walks."""
        super().setUp()

    def test_embedding_pipeline(self):
        """Test that embed pipeline works fine in SPINE."""
        for _, row in get_available_models_for_node_embedding().iterrows():
            embed_graph(
                "Cora",
                repository="linqs",
                embedding_model=row.model_name,
                library_name=row.library_name,
                verbose=False
            )