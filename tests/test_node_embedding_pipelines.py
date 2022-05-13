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
            repository="linqs",
            embedding_model="SPINE",
        )

    def test_transe(self):
        """Test that embed pipeline works fine in SPINE."""
        embed_graph(
            "Cora",
            repository="linqs",
            embedding_model="TransE",
            library_name="Ensmallen"
        )

    def test_cbow_tensorflow(self):
        """Test that embed pipeline works fine in SPINE."""
        embed_graph(
            "Cora",
            repository="linqs",
            embedding_model="CBOW",
            library_name="TensorFlow"
        )

    def test_cbow_ensmallen(self):
        """Test that embed pipeline works fine in SPINE."""
        embed_graph(
            "Cora",
            repository="linqs",
            embedding_model="CBOW",
            library_name="Ensmallen"
        )

    def test_skipgram_tensorflow(self):
        """Test that embed pipeline works fine in SPINE."""
        embed_graph(
            "Cora",
            repository="linqs",
            embedding_model="SkipGram",
            library_name="TensorFlow"
        )

    def test_skipgram_ensmallen(self):
        """Test that embed pipeline works fine in SPINE."""
        embed_graph(
            "Cora",
            repository="linqs",
            embedding_model="SkipGram",
            library_name="Ensmallen"
        )
