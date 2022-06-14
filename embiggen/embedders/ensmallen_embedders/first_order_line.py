"""Module providing FirstOrderLINE model implementation."""
from typing import Optional, Dict, Any, Union
from ensmallen import Graph
from ensmallen import models
import numpy as np
import pandas as pd
from embiggen.utils.abstract_models import AbstractEmbeddingModel, EmbeddingResult


class FirstOrderLINEEnsmallen(AbstractEmbeddingModel):
    """Class providing FirstOrderLINE implemeted in Rust from Ensmallen."""

    def __init__(
        self,
        embedding_size: int = 100,
        negative_samples_rate: float = 0.9,
        epochs: int = 10,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.9,
        random_state: int = 42,
        enable_cache: bool = False
    ):
        """Create new abstract Node2Vec method.

        Parameters
        --------------------------
        embedding_size: int = 100
            Dimension of the embedding.
        negative_samples_rate: float = 0.9
            Rate of negative samples per rate.
        epochs: int = 10
            Number of epochs to train the model for.
        learning_rate: float = 0.01
            The learning rate to use to train the Node2Vec model. By default 0.01.
        learning_rate_decay: float = 0.9
            Factor to reduce the learning rate for at each epoch. By default 0.9.
        random_state: int = 42
            The random state to reproduce the training sequence.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._learning_rate_decay = learning_rate_decay

        self._model_kwargs = dict(
            embedding_size=embedding_size,
            negative_samples_rate=negative_samples_rate,
            random_state=random_state,
        )

        self._model = models.FirstOrderLINE(**self._model_kwargs)

        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache
        )

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            embedding_size=5,
            epochs=1,
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return {
            **super().parameters(),
            **self._model_kwargs,
            **dict(
                epochs=self._epochs,
                learning_rate=self._learning_rate,
                learning_rate_decay=self._learning_rate_decay,
            )
        }

    @staticmethod
    def task_name() -> str:
        return "Node Embedding"

    @staticmethod
    def library_name() -> str:
        return "Ensmallen"

    def requires_nodes_sorted_by_decreasing_node_degree(self) -> bool:
        return False

    def is_topological(self) -> bool:
        return True

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
        verbose: bool = True
    ) -> EmbeddingResult:
        """Return node embedding."""
        node_embedding = self._model.fit_transform(
            graph,
            epochs=self._epochs,
            learning_rate=self._learning_rate,
            learning_rate_decay=self._learning_rate_decay,
            verbose=verbose
        )
        if return_dataframe:
            node_embedding = pd.DataFrame(
                node_embedding,
                index=graph.get_node_names()
            )
        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            node_embeddings=node_embedding
        )

    @staticmethod
    def requires_edge_weights() -> bool:
        return False

    @staticmethod
    def requires_positive_edge_weights() -> bool:
        return False

    @staticmethod
    def model_name() -> str:
        """Returns name of the model."""
        return "First Order LINE"

    @staticmethod
    def requires_node_types() -> bool:
        return False

    @staticmethod
    def requires_edge_types() -> bool:
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
