"""Wrapper for NNSED model provided from the Karate Club package."""
from typing import Dict, Any
from karateclub.community_detection import NNSED
from embiggen.embedders.karateclub_embedders.abstract_karateclub_embedder import AbstractKarateClubEmbedder


class NNSEDKarateClub(AbstractKarateClubEmbedder):

    def __init__(
        self,
        embedding_size: int = 32,
        iterations: int = 10,
        noise: float = 10**-6,
        random_state: int = 42,
        ring_bell: bool = False,
        enable_cache: bool = False
    ):
        """Return a new NNSED embedding model.

        Parameters
        ----------------------
        embedding_size: int = 32
            Number of dimensions. Default is 32.
        noise: float = 10**-6
            Random noise for normalization stability. Default is 10**-6.
        random_state: int = 42
            Random state to use for the stocastic
            portions of the embedding algorithm.
        ring_bell: bool = False,
            Whether to play a sound when embedding completes.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._iterations = iterations
        self._noise = noise

        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache,
            ring_bell=ring_bell,
            random_state=random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns the parameters used in the model."""
        return dict(
            **super().parameters(),
            iterations=self._iterations,
            noise=self._noise
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **AbstractKarateClubEmbedder.smoke_test_parameters(),
            iterations=1,
        )

    def _build_model(self) -> NNSED:
        """Return new instance of the NNSED model."""
        return NNSED(
            dimensions=self._embedding_size,
            iterations=self._iterations,
            noise=self._noise,
            seed=self._random_state
        )

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model"""
        return "NNSED"

    def get_memberships(self):
        """Getting the cluster membership of nodes."""
        return self._model.get_memberships()

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

