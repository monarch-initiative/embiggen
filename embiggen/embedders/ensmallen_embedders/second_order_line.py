"""Module providing second-order LINE implementation."""
from typing import Dict, Any, Optional
from ensmallen import Graph
import numpy as np
import pandas as pd
from ensmallen import models
from embiggen.embedders.ensmallen_embedders.ensmallen_embedder import EnsmallenEmbedder
from embiggen.utils import EmbeddingResult


class SecondOrderLINEEnsmallen(EnsmallenEmbedder):
    """Class implementing the second-order LINE algorithm."""

    def __init__(
        self,
        embedding_size: int = 100,
        epochs: int = 100,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.9,
        avoid_false_negatives: bool = False,
        node_embedding_path: Optional[str] = None,
        contextual_node_embedding_path: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
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
        avoid_false_negatives: bool = False
            Whether to avoid sampling false negatives.
            This may cause a slower training.
        node_embedding_path: Optional[str] = None
            Path where to mmap and store the nodes embedding.
            This is necessary to embed large graphs whose embedding will not
            fit into the available main memory.
        contextual_node_embedding_path: Optional[str] = None
            Path where to mmap and store the contextual nodes embedding.
            This is necessary to embed large graphs whose embedding will not
            fit into the available main memory.
        random_state: int = 42
            Random state to reproduce the embeddings.
        verbose: bool = False
            Whether to show loading bars.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._kwargs = dict(
            epochs=epochs,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            avoid_false_negatives=avoid_false_negatives,
            node_embedding_path=node_embedding_path,
            contextual_node_embedding_path=contextual_node_embedding_path,
            verbose=verbose
        )

        self._model = models.SecondOrderLINE(
            embedding_size=embedding_size,
            random_state=random_state,
            **self._kwargs
        )

        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache,
            random_state=random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return dict(
            **super().parameters(),
            **self._kwargs
        )

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
    ) -> EmbeddingResult:
        """Return node embedding."""
        node_embeddings = self._model.fit_transform(graph)
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
            node_embeddings=node_embeddings,
        )

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "Second-order LINE"
    
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
    def is_stocastic(cls) -> bool:
        """Returns whether the model is stocastic and has therefore a random state."""
        return True