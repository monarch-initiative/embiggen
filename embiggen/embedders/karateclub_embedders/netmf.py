"""Wrapper for NetMF model provided from the Karate Club package."""
from typing import Dict, Any
from karateclub.node_embedding import NetMF
from embiggen.embedders.karateclub_embedders.abstract_karateclub_embedder import AbstractKarateClubEmbedder


class NetMFKarateClub(AbstractKarateClubEmbedder):

    def __init__(
        self,
        embedding_size: int = 100,
        iteration: int = 10,
        order: int = 2,
        negative_samples: int = 1,
        random_state: int = 42,
        enable_cache: bool = False
    ):
        """Return a new NetMF embedding model.

        Parameters
        ----------------------
        embedding_size: int = 100
            Size of the embedding to use.
        iteration: int = 10
            Number of SVD iterations. Default is 10.
        order: int = 2
            Number of PMI matrix powers. Default is 5.
        negative_samples: int = 1
            Number of negative samples. Default is 1.
        random_state: int = 42
            Random state to use for the stocastic
            portions of the embedding algorithm.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._iteration = iteration
        self._order = order
        self._negative_samples = negative_samples
        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache,
            random_state=random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns the parameters used in the model."""
        return dict(
            **super().parameters(),
            iteration=self._iteration,
            negative_samples=self._negative_samples,
            order=self._order
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **AbstractKarateClubEmbedder.smoke_test_parameters(),
            iteration=1,
            order=2
        )

    def _build_model(self) -> NetMF:
        """Return new instance of the NetMF model."""
        return NetMF(
            dimensions=self._embedding_size,
            iteration=self._iteration,
            order=self._order,
            negative_samples=self._negative_samples,
            seed=self._random_state
        )

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model"""
        return "NetMF"

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

