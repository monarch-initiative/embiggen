import pytest
from embiggen.utils import EmbeddingResult
from unittest import TestCase
import numpy as np


class TestEmbeddingResult(TestCase):

    def setUp(self):
        pass

    def test_construction(self):
        nr = EmbeddingResult(
            embedding_method_name="Test",
            node_embeddings=np.random.uniform(size=(100, 10))
        )
        nr.get_all_node_embedding()
        nr.get_node_embedding_from_index(0)

        self.assertEqual("Test", nr.embedding_method_name)

        nr.dump()

        with pytest.raises(ValueError):
            nr.get_node_embedding_from_index(1)

        nr = EmbeddingResult(
            embedding_method_name="Test",
            edge_embeddings=np.random.uniform(size=(100, 10))
        )
        nr.get_all_edge_embedding()
        nr.get_edge_embedding_from_index(0)

        with pytest.raises(ValueError):
            nr.get_edge_embedding_from_index(1)

        nr = EmbeddingResult(
            embedding_method_name="Test",
            node_type_embeddings=np.random.uniform(size=(100, 10))
        )
        nr.get_all_node_type_embeddings()
        nr.get_node_type_embedding_from_index(0)

        with pytest.raises(ValueError):
            nr.get_node_type_embedding_from_index(1)

        nr = EmbeddingResult(
            embedding_method_name="Test",
            edge_type_embeddings=np.random.uniform(size=(100, 10))
        )
        nr.get_all_edge_type_embeddings()
        nr.get_edge_type_embedding_from_index(0)

        with pytest.raises(ValueError):
            nr.get_edge_type_embedding_from_index(1)

        with pytest.raises(ValueError):
            EmbeddingResult(
                embedding_method_name="Test",
                edge_type_embeddings="hu"
            )

        with pytest.raises(ValueError):
            EmbeddingResult(
                embedding_method_name="Test",
                edge_type_embeddings=np.random.uniform(size=(0, 10))
            )

        with pytest.raises(ValueError):
            EmbeddingResult(
                embedding_method_name="Test",
                edge_type_embeddings=np.full(shape=(10, 10), fill_value=np.nan)
            )

    def test_check_exceptions(self):
        node_embedding = EmbeddingResult(
            embedding_method_name="Test",
        )

        with pytest.raises(ValueError):
            node_embedding.get_all_node_embedding()

        with pytest.raises(ValueError):
            node_embedding.get_all_edge_embedding()

        with pytest.raises(ValueError):
            node_embedding.get_all_node_type_embeddings()
        
        with pytest.raises(ValueError):
            node_embedding.get_all_edge_type_embeddings()