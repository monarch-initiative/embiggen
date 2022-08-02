"""Module providing Siamese implementation."""
from typing import Dict, Any, Optional
from ensmallen import models
from userinput.utils import must_be_in_set
from embiggen.embedders.ensmallen_embedders.ensmallen_embedder import EnsmallenEmbedder
from embiggen.utils import abstract_class


@abstract_class
class SiameseEnsmallen(EnsmallenEmbedder):
    """Class implementing the Siamese algorithm."""

    models = {
        "TransE": models.TransE,
        "TransH": models.TransH,
        "Unstructured": models.Unstructured,
        "Structured Embedding": models.StructuredEmbedding,
    }

    def __init__(
        self,
        embedding_size: int = 100,
        relu_bias: float = 1.0,
        epochs: int = 100,
        learning_rate: float = 0.1,
        learning_rate_decay: float = 0.9,
        node_embedding_path: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
        enable_cache: bool = False,
        **paths: Dict[str, str]
    ):
        """Create new abstract Node2Vec method.

        Parameters
        --------------------------
        embedding_size: int = 100
            Dimension of the embedding.
        relu_bias: float = 1.0
            Bias to use for the relu.
            In the Siamese paper it is called gamma.
        epochs: int = 100
            The number of epochs to run the model for, by default 10.
        learning_rate: float = 0.05
            The learning rate to update the gradient, by default 0.01.
        learning_rate_decay: float = 0.9
            Factor to reduce the learning rate for at each epoch. By default 0.9.
        node_embedding_path: Optional[str] = None
            Path where to mmap and store the nodes embedding.
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
            relu_bias=relu_bias,
            epochs=epochs,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            node_embedding_path=node_embedding_path,
            verbose=verbose,
            **paths
        )

        self._model_name = must_be_in_set(
            self.model_name(),
            SiameseEnsmallen.models,
            "Siamese models"
        )

        self._model = SiameseEnsmallen.models[self._model_name](
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
            **self._kwargs,
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            embedding_size=5,
            epochs=1
        )

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        return False

    @classmethod
    def is_stocastic(cls) -> bool:
        """Returns whether the model is stocastic and has therefore a random state."""
        return True
