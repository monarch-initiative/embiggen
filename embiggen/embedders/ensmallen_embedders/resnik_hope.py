"""Module providing Resnik-based HOPE implementation."""
from typing import Optional,  Dict, Any, List
from ensmallen import Graph
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds as sparse_svds
from sklearn.utils.extmath import randomized_svd
from userinput.utils import must_be_in_set
from embiggen.embedders.ensmallen_embedders.ensmallen_embedder import EnsmallenEmbedder
from embiggen.utils import EmbeddingResult
from embiggen.similarities import DAGResnik

class ResnikHOPEEnsmallen(EnsmallenEmbedder):
    """Class implementing the Resnik-based HOPE algorithm."""

    def __init__(
        self,
        node_counts: Dict[str, float],
        embedding_size: int = 100,
        verbose: bool = False,
        enable_cache: bool = False
    ):
        """Create new Resnik-based HOPE method.

        Parameters
        --------------------------
        node_counts: Dict[str, float]
            Counts to compute the terms frequencies.
        embedding_size: int = 100
            Dimension of the embedding.
        verbose: bool = False
            Whether to show loading bars.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """        
        self._node_counts = node_counts
        self._verbose = verbose

        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return dict(
            **super().parameters(),
            **dict(
                node_counts=self._node_counts,
                metric=self._metric,
                verbose=self._verbose,
                root_node_name=self._root_node_name
            )
        )

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
    ) -> EmbeddingResult:
        """Return node embedding."""
        model = DAGResnik(self._verbose)
        model.fit(
            graph,
            node_counts=self._node_counts
        )
        matrix = model.get_pairwise_similarities(
            return_similarities_dataframe=False
        )

        U, sigmas, Vt = randomized_svd(
            matrix,
            n_components=int(self._embedding_size / 2)
        )
        
        sigmas = np.diagflat(np.sqrt(sigmas))
        left_embedding = np.dot(U, sigmas)
        right_embedding = np.dot(Vt.T, sigmas)

        if return_dataframe:
            node_names = graph.get_node_names()
            left_embedding = pd.DataFrame(
                left_embedding,
                index=node_names
            )
            right_embedding = pd.DataFrame(
                right_embedding,
                index=node_names
            )
        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            node_embeddings=[left_embedding, right_embedding]
        )

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "Resnik HOPE"

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
