"""Wrapper for GraRep model provided from the Karate Club package."""
from typing import Dict, Any
from karateclub.node_embedding import GraRep
from embiggen.embedders.karateclub_embedders.abstract_karateclub_embedder import AbstractKarateClubEmbedder


class GraRepKarateClub(AbstractKarateClubEmbedder):

    def __init__(
        self,
        embedding_size: int = 100,
        iteration: int = 10,
        order: int = 5,
        random_state: int = 42,
        enable_cache: bool = False
    ):
        """Return a new GraRep embedding model.

        Parameters
        ----------------------
        embedding_size: int = 100
            Size of the embedding to use.
        iteration: int = 10
            Number of SVD iterations. Default is 10.
        order: int = 5
            Number of PMI matrix powers. Default is 5.
        random_state: int = 42
            Random state to use for the stocastic
            portions of the embedding algorithm.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._iteration = iteration
        self._order = order
        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache,
            random_state=random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns the parameters used in the model."""
        return dict(
            **super().parameters(),
            iteration = self._iteration,
            order = self._order
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **AbstractKarateClubEmbedder.smoke_test_parameters(),
            iteration = 1,
            order = 2
        )

    def _build_model(self) -> GraRep:
        """Return new instance of the GraRep model."""
        return GraRep(
            dimensions=self._embedding_size,
            iteration=self._iteration,
            order=self._order,
            seed=self._random_state
        )

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model"""
        return "GraRep"

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

