"""Wrapper for NodeSketch model provided from the Karate Club package."""
from typing import Dict, Any
from karateclub.node_embedding import NodeSketch
from embiggen.embedders.karateclub_embedders.abstract_karateclub_embedder import AbstractKarateClubEmbedder


class NodeSketchKarateClub(AbstractKarateClubEmbedder):

    def __init__(
        self,
        embedding_size: int = 100,
        iterations: int = 10,
        decay: float = 0.01,
        random_state: int = 42,
        enable_cache: bool = False
    ):
        """Return a new NodeSketch embedding model.

        Parameters
        ----------------------
        embedding_size: int = 100
            Size of the embedding to use.
        iterations: int = 10
            Number of SVD iterationss. Default is 10.
        decay: float = 0.01
            Exponential decay rate. Default is 0.01.
        random_state: int = 42
            Random state to use for the stocastic
            portions of the embedding algorithm.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._iterations = iterations
        self._decay = decay

        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache,
            random_state=random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns the parameters used in the model."""
        return dict(
            **super().parameters(),
            iterations=self._iterations,
            decay=self._decay
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **AbstractKarateClubEmbedder.smoke_test_parameters(),
            iterations=1,
        )

    def _build_model(self) -> NodeSketch:
        """Return new instance of the NodeSketch model."""
        return NodeSketch(
            dimensions=self._embedding_size,
            iterations=self._iterations,
            decay=self._decay,
            seed=self._random_state
        )

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model"""
        return "NodeSketch"

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
