"""Module providing SocioDim implementation."""
from typing import Optional,  Dict, Any
from ensmallen import Graph
import pandas as pd
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from embiggen.utils.abstract_models import AbstractEmbeddingModel, EmbeddingResult


class SocioDimEnsmallen(AbstractEmbeddingModel):
    """Class implementing the SocioDim algorithm."""

    def __init__(
        self,
        embedding_size: int = 100,
        use_sparse_reduce: bool = True,
        enable_cache: bool = False
    ):
        """Create new SocioDim method.

        Parameters
        --------------------------
        embedding_size: int = 100
            Dimension of the embedding.
        use_sparse_reduce: bool = True
            Whether to use the sparse reduce or the dense reduce for
            the computation of eigenvectors.
            For both reduce mechanisms, we are using LAPACK implementations.
            For some currently unknown reason, their implementation using
            a sparse reduce, even on dense matrices, yields better results.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._use_sparse_reduce = use_sparse_reduce
        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache,
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return {
            **super().parameters(),
            **dict(
                use_sparse_reduce=self._use_sparse_reduce,
            )
        }

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
        verbose: bool = True
    ) -> EmbeddingResult:
        """Return node embedding."""
        if self._use_sparse_reduce:
            embedding = eigsh(
                graph.get_dense_modularity_matrix(),
                k=self._embedding_size,
                which="LM",
                return_eigenvectors=True
            )[1]
        else:
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

    @classmethod
    def task_name(cls) -> str:
        return "Node Embedding"

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "SocioDim"

    @classmethod
    def library_name(cls) -> str:
        return "Ensmallen"

    @classmethod
    def requires_nodes_sorted_by_decreasing_node_degree(cls) -> bool:
        return False

    @classmethod
    def is_topological(cls) -> bool:
        return True

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        return False

    @classmethod
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can optionally use edge types."""
        return False

    @classmethod
    def is_stocastic(cls) -> bool:
        """Returns whether the model is stocastic and has therefore a random state."""
        return False
