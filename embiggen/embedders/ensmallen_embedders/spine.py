"""Module providing abstract Node2Vec implementation."""
from typing import Optional
from ensmallen import Graph
import numpy as np
from ensmallen import models
from .ensmallen_embedder import EnsmallenEmbedder


class SPINE(EnsmallenEmbedder):
    """Abstract class for Node2Vec algorithms."""

    def __init__(
        self,
        embedding_size: int = 100,
        dtype: Optional[str] = "u8",
        verbose: bool = True
    ):
        """Create new abstract Node2Vec method.

        Parameters
        --------------------------
        embedding_size: int = 100
            Dimension of the embedding.
        dtype: Optional[str] = "u8"
            Dtype to use for the embedding. Note that an improper dtype may cause overflows.
        verbose: bool = True
            Whether to show loading bars
        """
        self._dtype = dtype
        self._SPINE = models.SPINE(embedding_size=embedding_size)

        super().__init__(
            embedding_size=embedding_size,
            verbose=verbose
        )

    def _fit_transform(self, graph: Graph) -> np.ndarray:
        return self._SPINE.fit_transform(
            graph,
            dtype=self._dtype,
            verbose=self._verbose,
        ).T
