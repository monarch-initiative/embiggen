"""Module providing NetMF implementation."""
from typing import Optional,  Dict, Any
from ensmallen import Graph
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from embiggen.embedders.ensmallen_embedders.ensmallen_embedder import EnsmallenEmbedder
from embiggen.utils import EmbeddingResult


class NetMFEnsmallen(EnsmallenEmbedder):
    """Class implementing the NetMF algorithm."""

    def __init__(
        self,
        embedding_size: int = 100,
        walk_length: int = 128,
        iterations: int = 10,
        window_size: int = 10,
        max_neighbours: Optional[int] = 100,
        random_state: int = 42,
        enable_cache: bool = False
    ):
        """Create new NetMF method.

        Parameters
        --------------------------
        embedding_size: int = 100
            Dimension of the embedding.
        walk_length: int = 128
            Maximal length of the walks.
        iterations: int = 10
            Number of iterations of the single walks.
        window_size: int = 10
            Window size for the local context.
            On the borders the window size is trimmed.
        max_neighbours: Optional[int] = 100
            Number of maximum neighbours to consider when using approximated walks.
            By default, None, we execute exact random walks.
            This is mainly useful for graphs containing nodes with high degrees.
        random_state: int = 42
            The random state to reproduce the training sequence.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._walk_parameters = dict(
            walk_length=walk_length,
            iterations=iterations,
            window_size=window_size,
            max_neighbours=max_neighbours,
        )

        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache,
            random_state=random_state
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **EnsmallenEmbedder.smoke_test_parameters(),
            window_size=1,
            walk_length=4,
            iterations=1,
            max_neighbours=10,
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return dict(
            **super().parameters(),
            **self._walk_parameters
        )

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
    ) -> EmbeddingResult:
        """Return node embedding."""
        edges, weights = graph.get_log_normalized_cooccurrence_coo_matrix(
            **self._walk_parameters
        )

        coo = coo_matrix(
            (weights, (edges[:, 0], edges[:, 1])),
            shape=(
                graph.get_number_of_nodes(),
                graph.get_number_of_nodes()
            ),
            dtype=np.float32
        )

        model = TruncatedSVD(
            n_components=self._embedding_size,
            random_state=self._random_state
        )
        model.fit(coo)
        embedding = model.transform(coo)

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
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "NetMF"

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
        return True
