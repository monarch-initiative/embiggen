"""Wrapper for RandNE model provided from the Karate Club package."""
from typing import Dict, Any, List, Union, Tuple
from karateclub.node_embedding import RandNE
from embiggen.embedders.karateclub_embedders.abstract_karateclub_embedder import AbstractKarateClubEmbedder


class RandNEKarateClub(AbstractKarateClubEmbedder):

    def __init__(
        self,
        embedding_size: int = 128,
        alphas: Union[List[float], Tuple[float]] = (0.5, 0.5),
        random_state: int = 42,
        enable_cache: bool = False
    ):
        """Return a new RandNE embedding model.

        Parameters
        ----------------------
        embedding_size: int = 128
            Size of the embedding to use.
        alphas: Union[List[float], Tuple[float]] = (0.5, 0.5)
            Smoothing weights for adjacency matrix powers. Default is [0.5, 0.5].
        random_state: int = 42
            Random state to use for the stocastic
            portions of the embedding algorithm.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._random_state = random_state
        self._alphas = alphas
        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns the parameters used in the model."""
        return dict(
            **super().parameters(),
            alphas=self._alphas,
            random_state=self._random_state
        )

    def _build_model(self) -> RandNE:
        """Return new instance of the RandNE model."""
        return RandNE(
            dimensions=self._embedding_size,
            alphas=self._alphas,
            seed=self._random_state
        )

    @staticmethod
    def model_name() -> str:
        """Returns name of the model"""
        return "RandNE"

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
