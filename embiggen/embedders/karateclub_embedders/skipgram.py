"""Wrapper for Node2Vec CBOW model provided from the Karate Club package."""
from typing import Dict, Any
from karateclub.node_embedding import Node2Vec
from multiprocessing import cpu_count
from embiggen.embedders.karateclub_embedders.abstract_karateclub_embedder import AbstractKarateClubEmbedder


class SkipGramKarateClub(AbstractKarateClubEmbedder):

    def __init__(
        self,
        embedding_size: int = 100,
        walk_number: int = 10,
        walk_length: int = 80,
        window_size: int = 5,
        p: float = 1.0,
        q: float = 1.0,
        epochs: int = 10,
        learning_rate: float = 0.05,
        min_count: int = 1,
        random_state: int = 42,
        enable_cache: bool = False
    ):
        """Return a new Node2Vec CBOW embedding model.

        Implementation details
        ----------------------
        Even though in the original library Karate Club this
        model is reffered to as `Node2Vec`, here we refer to it
        as `CBOW` because even though it does Node2Vec sampling
        the embedding model itself is actually a CBOW model implemented
        from the Gensim library.

        Parameters
        ----------------------
        embedding_size: int = 100
            Size of the embedding to use.
        walk_number: int = 10
            Number of random walks. Default is 10.
        walk_length: int = 80
            Length of random walks. Default is 80.
        window_size: int = 5
            Matrix power order. Default is 5.
        p: float = 1.0
            Return parameter (1/p transition probability) to move towards from previous node.
        q: float = 1.0
            In-out parameter (1/q transition probability) to move away from previous node.
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
        self._walk_number=walk_number
        self._walk_length=walk_length
        self._workers=cpu_count()
        self._window_size=window_size
        self._p = p
        self._q = q
        self._epochs=epochs
        self._learning_rate=learning_rate
        self._min_count=min_count
        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache,
            random_state=random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns the parameters used in the model."""
        return dict(
            **super().parameters(),
            walk_number=self._walk_number,
            walk_length=self._walk_length,
            window_size=self._window_size,
            p=self._p,
            q=self._q,
            epochs=self._epochs,
            learning_rate=self._learning_rate,
            min_count=self._min_count,
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **AbstractKarateClubEmbedder.smoke_test_parameters(),
            walk_number=1,
            walk_length=8,
            window_size=2,
            epochs=1,
        )

    def _build_model(self) -> Node2Vec:
        """Return new instance of the Node2Vec CBOW model."""
        return Node2Vec(
            walk_number=self._walk_number,
            walk_length=self._walk_length,
            dimensions=self._embedding_size,
            workers=self._workers,
            window_size=self._window_size,
            p=self._p,
            q=self._q,
            epochs=self._epochs,
            learning_rate=self._learning_rate,
            min_count=self._min_count,
            seed=self._random_state
        )

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model"""
        return "Node2Vec SkipGram"

    @classmethod
    def requires_nodes_sorted_by_decreasing_node_degree(cls) -> bool:
        return False

    @classmethod
    def is_topological(cls) -> bool:
        return True

    @classmethod
    def requires_edge_weights(cls) -> bool:
        return False

    @classmethod
    def requires_positive_edge_weights(cls) -> bool:
        return True

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return True

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        return False

    @classmethod
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can optionally use edge types."""
        return False

