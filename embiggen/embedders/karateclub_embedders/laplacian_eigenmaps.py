"""Wrapper for LaplacianEigenmaps model provided from the Karate Club package."""
from typing import Dict, Any
from karateclub.node_embedding import LaplacianEigenmaps
from embiggen.embedders.karateclub_embedders.abstract_karateclub_embedder import AbstractKarateClubEmbedder


class LaplacianEigenmapsKarateClub(AbstractKarateClubEmbedder):

    def __init__(
        self,
        embedding_size: int = 100,
        random_state: int = 42,
        enable_cache: bool = False
    ):
        """Return a new LaplacianEigenmaps embedding model.

        Parameters
        ----------------------
        embedding_size: int = 100
            Size of the embedding to use.
        random_state: int = 42
            Random state to use for the stocastic
            portions of the embedding algorithm.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache,
            random_state=random_state
        )

    def _build_model(self) -> LaplacianEigenmaps:
        """Return new instance of the LaplacianEigenmaps model."""
        return LaplacianEigenmaps(
            dimensions=self._embedding_size,
            seed=self._random_state
        )

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model"""
        return "Laplacian Eigenmaps"

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

