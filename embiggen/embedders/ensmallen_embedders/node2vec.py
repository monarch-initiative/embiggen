"""Module providing abstract Node2Vec implementation."""
from typing import Optional, Union, Dict, Any
from ensmallen import Graph
import numpy as np
import pandas as pd
from ensmallen import models
from embiggen.utils.abstract_models import abstract_class, AbstractEmbeddingModel, EmbeddingResult


@abstract_class
class Node2VecEnsmallen(AbstractEmbeddingModel):
    """Abstract class for Node2Vec algorithms."""

    MODELS = {
        "cbow": models.CBOW,
        "skipgram": models.SkipGram
    }

    def __init__(
        self,
        model_name: str,
        epochs: int = 10,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.9,
        enable_cache: bool = False,
        **kwargs
    ):
        """Create new abstract Node2Vec method.

        Parameters
        --------------------------
        model_name: str
            The ensmallen graph embedding model to use.
            Can either be the SkipGram of CBOW models.
        epochs: int = 10
            Number of epochs to train the model for.
        learning_rate: float = 0.01
            The learning rate to use to train the Node2Vec model.
        learning_rate_decay: float = 0.9
            Factor to reduce the learning rate for at each epoch. By default 0.9.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        if model_name.lower() not in self.MODELS:
            raise ValueError(
                (
                    "The provided model name {} is not supported. "
                    "The supported models are `CBOW` and `SkipGram`."
                ).format(model_name)
            )

        self._epochs = epochs
        self._learning_rate = learning_rate
        self._learning_rate_decay = learning_rate_decay
        self._model_kwargs = kwargs

        self._model = Node2VecEnsmallen.MODELS[model_name.lower()](**kwargs)

        super().__init__(
            embedding_size=kwargs["embedding_size"],
            enable_cache=enable_cache
        )

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            embedding_size=5,
            epochs=1,
            window_size=1,
            walk_length=4,
            iterations=1,
            max_neighbours=10,
            number_of_negative_samples=1
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
    ) -> Union[np.ndarray, pd.DataFrame]:
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
        return True

    @staticmethod
    def can_use_edge_weights() -> bool:
        """Returns whether the model can optionally use edge weights."""
        return True

    def is_using_edge_weights(self) -> bool:
        """Returns whether the model is parametrized to use edge weights."""
        return True

    @staticmethod
    def can_use_node_types() -> bool:
        """Returns whether the model can optionally use node types."""
        return True

    def is_using_node_types(self) -> bool:
        """Returns whether the model is parametrized to use node types."""
        return self._model_kwargs["change_node_type_weight"] != 1.0

    @staticmethod
    def can_use_edge_types() -> bool:
        """Returns whether the model can optionally use edge types."""
        return True

    def is_using_edge_types(self) -> bool:
        """Returns whether the model is parametrized to use edge types."""
        return self._model_kwargs["change_edge_type_weight"] != 1.0