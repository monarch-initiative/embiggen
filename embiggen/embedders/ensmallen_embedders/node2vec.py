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
        "skipgram": models.SkipGram,
        "walkletscbow": models.WalkletsCBOW,
        "walkletsskipgram": models.WalkletsSkipGram,
    }

    def __init__(
        self,
        model_name: str,
        enable_cache: bool = False,
        **model_kwargs: Dict
    ):
        """Create new abstract Node2Vec method.

        Parameters
        --------------------------
        model_name: str
            The ensmallen graph embedding model to use.
            Can either be the SkipGram, WalkletsSkipGram, CBOW and WalkletsCBOW models.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        model_kwargs: Dict
            Further parameters to forward to the model.
        """
        if model_name.lower() not in self.MODELS:
            raise ValueError(
                (
                    "The provided model name {} is not supported. "
                    "The supported models are `CBOW` and `SkipGram`."
                ).format(model_name)
            )

        self._model_kwargs = model_kwargs

        self._model = Node2VecEnsmallen.MODELS[model_name.lower()](**model_kwargs)

        super().__init__(
            embedding_size=model_kwargs["embedding_size"],
            enable_cache=enable_cache,
            random_state=model_kwargs["random_state"]
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **AbstractEmbeddingModel.smoke_test_parameters(),
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
            **self._model_kwargs
        }

    @classmethod
    def task_name(cls) -> str:
        return "Node Embedding"

    @classmethod
    def library_name(cls) -> str:
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
        node_embeddings = self._model.fit_transform(graph)
        if not isinstance(node_embeddings, list):
            node_embeddings = [node_embeddings]
        if return_dataframe:
            node_names = graph.get_node_names()
            node_embeddings = [
                pd.DataFrame(
                    node_embedding,
                    index=node_names
                )
                for node_embedding in node_embeddings
            ]
        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            node_embeddings=node_embeddings
        )

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

    def is_using_edge_weights(self) -> bool:
        """Returns whether the model is parametrized to use edge weights."""
        return True

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        return True

    def is_using_node_types(self) -> bool:
        """Returns whether the model is parametrized to use node types."""
        return "change_node_type_weight" in self._model_kwargs and self._model_kwargs["change_node_type_weight"] != 1.0

    @classmethod
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can optionally use edge types."""
        return True

    def is_using_edge_types(self) -> bool:
        """Returns whether the model is parametrized to use edge types."""
        return "change_edge_type_weight" in self._model_kwargs and  self._model_kwargs["change_edge_type_weight"] != 1.0

    @classmethod
    def is_stocastic(cls) -> bool:
        """Returns whether the model is stocastic and has therefore a random state."""
        return True