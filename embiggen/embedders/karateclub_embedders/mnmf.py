"""Wrapper for MNMF model provided from the Karate Club package."""
from typing import Dict, Any
from karateclub.community_detection import MNMF
from embiggen.embedders.karateclub_embedders.abstract_karateclub_embedder import AbstractKarateClubEmbedder


class MNMFKarateClub(AbstractKarateClubEmbedder):

    def __init__(
        self,
        embedding_size: int = 128,
        clusters: int = 10,
        lambd: float = 0.2,
        alpha: float = 0.05,
        beta: float = 0.05,
        iterations: int = 200,
        lower_control: float = 10**-15,
        eta: float = 5.0,
        random_state: int = 42,
        ring_bell: bool = False,
        enable_cache: bool = False
    ):
        """Return a new MNMF embedding model.

        Parameters
        ----------------------
        embedding_size: int = 128
            Number of dimensions. Default is 128.
        clusters: int = 10
            Number of clusters. Default is 10.
        lambd: float = 0.2
            KKT penalty. Default is 0.2
        alpha: float = 0.05
            Clustering penalty. Default is 0.05.
        beta: float = 0.05
            Modularity regularization penalty. Default is 0.05.
        iterations: int = 200
            Number of power iterations. Default is 200.
        lower_control: float = 10**-15
            Floating point overflow control. Default is 10**-15.
        eta: float = 5.0
            Similarity mixing parameter. Default is 5.0.
        random_state: int = 42
            Random state to use for the stocastic
            portions of the embedding algorithm.
        ring_bell: bool = False,
            Whether to play a sound when embedding completes.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._clusters = clusters
        self._lambd = lambd
        self._alpha = alpha
        self._beta = beta
        self._iterations = iterations
        self._lower_control = lower_control
        self._eta = eta

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
            clusters = self._clusters,
            lambd=self._lambd,
            alpha=self._alpha,
            beta=self._beta,
            iterations=self._iterations,
            lower_control=self._lower_control,
            eta=self._eta,
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **AbstractKarateClubEmbedder.smoke_test_parameters(),
            iterations=1,
        )

    def _build_model(self) -> MNMF:
        """Return new instance of the MNMF model."""
        return MNMF(
            dimensions=self._embedding_size,
            clusters = self._clusters,
            lambd=self._lambd,
            alpha=self._alpha,
            beta=self._beta,
            iterations=self._iterations,
            lower_control=self._lower_control,
            eta=self._eta,
            seed=self._random_state
        )

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model"""
        return "MNMF"

    def get_cluster_centers(self):
        return self._model.get_cluster_centers()

    def get_memberships(self):
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

