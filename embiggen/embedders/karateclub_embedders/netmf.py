"""Wrapper for NetMF model provided from the Karate Club package."""
from typing import Dict, Any
from karateclub.node_embedding import NetMF
from embiggen.embedders.karateclub_embedders.abstract_karateclub_embedder import AbstractKarateClubEmbedder


class NetMFKarateClub(AbstractKarateClubEmbedder):

    def __init__(
        self,
        embedding_size: int = 128,
        iteration: int = 10,
        order: int = 2,
        negative_samples: int = 1,
        random_state: int = 42,
        enable_cache: bool = False
    ):
        """Return a new NetMF embedding model.

        Parameters
        ----------------------
        embedding_size: int = 128
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
        self._random_state = random_state
        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns the parameters used in the model."""
        return dict(
            **super().parameters(),
            random_state=self._random_state,
            iteration=self._iteration,
            negative_samples=self._negative_samples,
            order=self._order
        )

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
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

    @staticmethod
    def model_name() -> str:
        """Returns name of the model"""
        return "NetMF"

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

    def is_using_edge_weights(self) -> bool:
        """Returns whether the model is parametrized to use edge weights."""
        return False

    @staticmethod
    def can_use_node_types() -> bool:
        """Returns whether the model can optionally use node types."""
        return False

    def is_using_node_types(self) -> bool:
        """Returns whether the model is parametrized to use node types."""
        return False

    @staticmethod
    def can_use_edge_types() -> bool:
        """Returns whether the model can optionally use edge types."""
        return False

    def is_using_edge_types(self) -> bool:
        """Returns whether the model is parametrized to use edge types."""
        return False
