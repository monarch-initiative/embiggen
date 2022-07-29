from typing import Dict, Any
import pytest
from embiggen.embedders import embed_graph
from embiggen.utils import AbstractEmbeddingModel
from embiggen.embedders.ensmallen_embedders.hope import HOPEEnsmallen
from unittest import TestCase


class ClassForTestAbstractEmbeddingModel(AbstractEmbeddingModel):

    def __init__(self):
        super().__init__(
            embedding_size=100
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            invalid_parameter=5,
        )

    @classmethod
    def is_stocastic(cls) -> bool:
        """Returns whether the model is stocastic and has therefore a random state."""
        return False

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "TMP"

    @classmethod
    def library_name(cls) -> str:
        """Returns name of the library."""
        return "TMP"

    @classmethod 
    def can_use_edge_weights(cls) -> bool:
        return False

    @classmethod 
    def can_use_node_types(cls) -> bool:
        return False

    @classmethod 
    def can_use_edge_types(cls) -> bool:
        return False
    
    @classmethod
    def requires_nodes_sorted_by_decreasing_node_degree(cls) -> bool:
        return False

class TestEmbedGraph(TestCase):

    def setUp(self):
        pass

    def test_embed_graph(self):
        with pytest.raises(ValueError):
            embed_graph(
                "Cora",
                embedding_model=int,
                repository="linqs"
            )

        with pytest.raises(ValueError):
            embed_graph(
                "Cora",
                embedding_model=HOPEEnsmallen(),
                repository="linqs",
                embedding_size=1000
            )
        
        with pytest.raises(ValueError):
            embed_graph(
                "Cora",
                embedding_model=ClassForTestAbstractEmbeddingModel(),
                repository="linqs",
                smoke_test=True
            )

        with pytest.raises(ValueError):
            embed_graph(
                "Cora",
                embedding_model=ClassForTestAbstractEmbeddingModel(),
                repository="linqs"
            )

        embed_graph(
            "Cora",
            embedding_model="Degree-based SPINE",
            repository="linqs"
        )

        