"""Module providing Score-based SPINE implementation."""
from typing import Optional, Dict, Any
from ensmallen import Graph
import pandas as pd
import numpy as np
from ensmallen import models
from embiggen.embedders.ensmallen_embedders.ensmallen_embedder import EnsmallenEmbedder
from embiggen.utils import EmbeddingResult


class ScoreSPINE(EnsmallenEmbedder):
    """Class implementing the Score-based SPINE algorithm."""

    def __init__(
        self,
        scores: np.ndarray,
        embedding_size: int = 100,
        dtype: Optional[str] = "u8",
        maximum_depth: Optional[int] = None,
        verbose: bool = False,
        enable_cache: bool = False
    ):
        """Create new Score-based SPINE method.

        Parameters
        --------------------------
        scores: np.ndarray
            Numpy array to be used to sort the anchor nodes.
        embedding_size: int = 100
            Dimension of the embedding.
        dtype: Optional[str] = "u8"
            Dtype to use for the embedding.
            Note that an improper dtype may cause overflows.
        maximum_depth: Optional[int] = None
            Maximum depth of the shortest path.
        verbose: bool = False
            Whether to show loading bars.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._dtype = dtype
        self._verbose = verbose
        self._maximum_depth = maximum_depth
        self._scores = scores
        self._model = models.ScoreSPINE(
            embedding_size=embedding_size,
            verbose=self._verbose,
            maximum_depth=self._maximum_depth
        )

        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return {
            **super().parameters(),
            **dict(
                dtype=self._dtype
            )
        }
    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            embedding_size=5,
        )

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
    ) -> EmbeddingResult:
        """Return node embedding."""
        node_embedding = self._model.fit_transform(
            scores=self._scores,
            graph=graph,
            dtype=self._dtype,
        )
        if return_dataframe:
            node_embedding = pd.DataFrame(
                node_embedding,
                index=graph.get_node_names()
            )
        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            node_embeddings=node_embedding
        )

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "Score-based SPINE"

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        return False

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False

    @classmethod
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can optionally use edge types."""
        return False

    @classmethod
    def is_stocastic(cls) -> bool:
        """Returns whether the model is stocastic and has therefore a random state."""
        return False