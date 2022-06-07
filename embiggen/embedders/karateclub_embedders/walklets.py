"""Wrapper for Walklets model provided from the Karate Club package."""
from typing import Dict, Any
from karateclub.node_embedding import Walklets
from multiprocessing import cpu_count
from embiggen.embedders.karateclub_embedders.abstract_karateclub_embedder import AbstractKarateClubEmbedder


class WalkletsKarateClub(AbstractKarateClubEmbedder):

    def __init__(
        self,
        embedding_size: int = 128,
        walk_number: int = 10,
        walk_length: int = 80,
        window_size: int = 5,
        epochs: int = 10,
        learning_rate: float = 0.05,
        min_count: int = 1,
        random_state: int = 42,
        enable_cache: bool = False
    ):
        """Return a new Walklets embedding model.

        Parameters
        ----------------------
        embedding_size: int = 128
            Size of the embedding to use.
        walk_number: int = 10
            Number of random walks. Default is 10.
        walk_length: int = 80
            Length of random walks. Default is 80.
        window_size: int = 5
            Matrix power order. Default is 5.
        epochs: int = 10
            Number of epochs. Default is 1.
        learning_rate: float = 0.05
            HogWild! learning rate. Default is 0.05.
        min_count: int = 1
            Minimal count of node occurrences. Default is 1.
        random_state: int = 42
            Random state to use for the stocastic
            portions of the embedding algorithm.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._random_state = random_state
        self._walk_number=walk_number
        self._walk_length=walk_length
        self._workers=cpu_count()
        self._window_size=window_size
        self._epochs=epochs
        self._learning_rate=learning_rate
        self._min_count=min_count
        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns the parameters used in the model."""
        return dict(
            **super().parameters(),
            random_state=self._random_state,
            walk_number=self._walk_number,
            walk_length=self._walk_length,
            workers=self._workers,
            window_size=self._window_size,
            epochs=self._epochs,
            learning_rate=self._learning_rate,
            min_count=self._min_count,
        )

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **AbstractKarateClubEmbedder.smoke_test_parameters(),
            walk_number=1,
            walk_length=8,
            window_size=2,
            epochs=1,
        )

    def _build_model(self) -> Walklets:
        """Return new instance of the Walklets model."""
        return Walklets(
            walk_number=self._walk_number,
            walk_length=self._walk_length,
            dimensions=self._embedding_size,
            workers=self._workers,
            window_size=self._window_size,
            epochs=self._epochs,
            learning_rate=self._learning_rate,
            min_count=self._min_count,
            seed=self._random_state
        )

    @staticmethod
    def model_name() -> str:
        """Returns name of the model"""
        return "Walklets"

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
