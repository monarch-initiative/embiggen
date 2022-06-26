"""Module providing SocioDim implementation."""
from typing import Optional,  Dict, Any
from ensmallen import Graph
import pandas as pd
import numpy as np
from scipy.linalg import eigh
from embiggen.utils.abstract_models import AbstractEmbeddingModel, EmbeddingResult


class SocioDimEnsmallen(AbstractEmbeddingModel):
    """Class implementing the SocioDim algorithm."""

    def __init__(
        self,
        embedding_size: int = 100,
        enable_cache: bool = False
    ):
        """Create new SocioDim method.

        Parameters
        --------------------------
        embedding_size: int = 100
            Dimension of the embedding.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache,
        )

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
        verbose: bool = True
    ) -> EmbeddingResult:
        """Return node embedding."""
        number_of_nodes = graph.get_number_of_nodes()
        embedding = eigh(
            graph.get_dense_modularity_matrix(),
            eigvals=(
                number_of_nodes-self._embedding_size,
                number_of_nodes-1
            )
        )[1]

        if return_dataframe:
            node_names = graph.get_node_names()
            embedding = pd.DataFrame(
                embedding,
                index=node_names
            )
        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            node_embeddings=embedding
        )

    @staticmethod
    def task_name() -> str:
        return "Node Embedding"

    @staticmethod
    def model_name() -> str:
        """Returns name of the model."""
        return "SocioDim"

    @staticmethod
    def library_name() -> str:
        return "Ensmallen"

    @staticmethod
    def requires_nodes_sorted_by_decreasing_node_degree() -> bool:
        return False

    @staticmethod
    def is_topological() -> bool:
        return True

    @staticmethod
    def requires_node_types() -> bool:
        return False

    @staticmethod
    def requires_edge_types() -> bool:
        return False

    @staticmethod
    def requires_edge_weights() -> bool:
        return False

    @staticmethod
    def requires_positive_edge_weights() -> bool:
        return False

    @staticmethod
    def can_use_edge_weights() -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False

    @staticmethod
    def can_use_node_types() -> bool:
        """Returns whether the model can optionally use node types."""
        return False

    @staticmethod
    def can_use_edge_types() -> bool:
        """Returns whether the model can optionally use edge types."""
        return False

    @staticmethod
    def is_stocastic() -> bool:
        """Returns whether the model is stocastic and has therefore a random state."""
        return False
