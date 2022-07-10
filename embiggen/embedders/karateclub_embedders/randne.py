"""Wrapper for RandNE model provided from the Karate Club package."""
from typing import Dict, Any, List, Union, Tuple
from karateclub.node_embedding import RandNE
from embiggen.embedders.karateclub_embedders.abstract_karateclub_embedder import AbstractKarateClubEmbedder


class RandNEKarateClub(AbstractKarateClubEmbedder):

    def __init__(
        self,
        embedding_size: int = 100,
        alphas: Union[List[float], Tuple[float]] = (0.5, 0.5),
        random_state: int = 42,
        enable_cache: bool = False
    ):
        """Return a new RandNE embedding model.

        Parameters
        ----------------------
        embedding_size: int = 100
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
        self._alphas = alphas
        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache,
            random_state=random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns the parameters used in the model."""
        return dict(
            **super().parameters(),
            alphas=self._alphas,
        )

    def _build_model(self) -> RandNE:
        """Return new instance of the RandNE model."""
        return RandNE(
            dimensions=self._embedding_size,
            alphas=self._alphas,
            seed=self._random_state
        )

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model"""
        return "RandNE"

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

