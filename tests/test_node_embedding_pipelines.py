"""Test to validate that the model GloVe works properly with graph walks."""
from unittest import TestCase
from embiggen.embedders import embed_graph


class TestNodeEmbeddingUtility(TestCase):
    """Unit test for model GloVe on graph walks."""

    def setUp(self):
        """Setting up objects to test CBOW model on graph walks."""
        super().setUp()

    def test_spine(self):
        """Test that embed pipeline works fine in SPINE."""
        embed_graph(
            "Cora",
            library_name="Linqs",
            embedding_model="SPINE",
        )

    def test_transe(self):
        """Test that embed pipeline works fine in SPINE."""
        embed_graph(
            "Cora",
            library_name="Linqs",
            embedding_model="TransE",
        )
