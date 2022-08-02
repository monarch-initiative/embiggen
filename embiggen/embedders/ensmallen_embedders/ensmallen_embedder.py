"""Module providing SocioDim implementation."""
from typing import Optional,  Dict, Any
from ensmallen import Graph
import pandas as pd
import numpy as np
from embiggen.utils.abstract_models import AbstractEmbeddingModel, abstract_class

@abstract_class
class EnsmallenEmbedder(AbstractEmbeddingModel):
    """Class implementing the Ensmallen Embedder algorithm."""

    def __init__(
        self,
        random_state: Optional[int] = None,
        embedding_size: int = 100,
        enable_cache: bool = False
    ):
        
        """Create new EnsmallenEmbedder method.

        Parameters
        --------------------------
        random_state: Optional[int] = None
            Random state to reproduce the embeddings.
        embedding_size: int = 100
            Dimension of the embedding.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        super().__init__(
            random_state=random_state,
            embedding_size=embedding_size,
            enable_cache=enable_cache,
        )

    @classmethod
    def task_name(cls) -> str:
        return "Node Embedding"

    @classmethod
    def library_name(cls) -> str:
        return "Ensmallen"

    @classmethod
    def requires_nodes_sorted_by_decreasing_node_degree(cls) -> bool:
        return False

    @classmethod
    def is_topological(cls) -> bool:
        return True