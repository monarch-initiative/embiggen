"""Module providing first-order LINE implementation."""
from typing import Dict, Any, Union
from ensmallen import Graph
import numpy as np
import pandas as pd
from ensmallen import models
from embiggen.utils.abstract_models import AbstractEmbeddingModel, EmbeddingResult


class FirstOrderLINEEnsmallen(AbstractEmbeddingModel):
    """Class implementing the first-order LINE algorithm."""

    def __init__(
        self,
        embedding_size: int = 100,
        epochs: int = 100,
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
        epochs: int = 100
            The number of epochs to run the model for, by default 10.
        learning_rate: float = 0.01
            The learning rate to update the gradient, by default 0.01.
        learning_rate_decay: float = 0.9
            Factor to reduce the learning rate for at each epoch. By default 0.9.
        random_state: int = 42
            Random state to reproduce the embeddings.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._learning_rate_decay = learning_rate_decay

        self._model = models.FirstOrderLINE(
            embedding_size=embedding_size,
            random_state=random_state
        )

        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache,
            random_state=random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return {
            **super().parameters(),
            **dict(
                epochs=self._epochs,
                learning_rate=self._learning_rate,
                learning_rate_decay=self._learning_rate_decay,
            )
        }

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            embedding_size=5,
            epochs=1
        )

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
            verbose=verbose,
        )
        if return_dataframe:
            node_embedding = pd.DataFrame(
                node_embedding,
                index=graph.get_node_names()
            )

        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            node_embeddings= node_embedding,
        )

    @classmethod
    def task_name(cls) -> str:
        return "Node Embedding"

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "First-order LINE"

    @classmethod
    def library_name(cls) -> str:
        return "Ensmallen"

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

    @classmethod
    def task_involves_edge_types(cls) -> bool:
        """Returns whether the model task involves edge types."""
        return False

    @classmethod
    def is_stocastic(cls) -> bool:
        """Returns whether the model is stocastic and has therefore a random state."""
        return True