"""Module providing abstract Node2Vec implementation."""
from typing import Optional
from ensmallen import Graph
import numpy as np
from ensmallen import models
from .ensmallen_embedder import EnsmallenEmbedder


class WeightedSPINE(EnsmallenEmbedder):
    """Abstract class for Node2Vec algorithms."""

    def __init__(
        self,
        embedding_size: int = 100,
        use_edge_weights_as_probabilities: bool = False,
        verbose: bool = True
    ):
        """Create new abstract Node2Vec method.

        Parameters
        --------------------------
        embedding_size: int = 100
            Dimension of the embedding.
        use_edge_weights_as_probabilities: bool = False
            Whether to treat the weights as probabilities.
        verbose: bool = True
            Whether to show loading bars
        """
        self._WeightedSPINE = models.WeightedSPINE(
            embedding_size=embedding_size,
            use_edge_weights_as_probabilities=use_edge_weights_as_probabilities
        )

        super().__init__(
            embedding_size=embedding_size,
            verbose=verbose
        )

    def _fit_transform(self, graph: Graph) -> np.ndarray:
        return self._WeightedSPINE.fit_transform(
            graph,
            verbose=self._verbose,
        ).T
