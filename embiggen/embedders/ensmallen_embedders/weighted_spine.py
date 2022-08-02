"""Module providing abstract Node2Vec implementation."""
from typing import Dict, Any
from ensmallen import Graph
import pandas as pd
from ensmallen import models
from embiggen.embedders.ensmallen_embedders.ensmallen_embedder import EnsmallenEmbedder
from embiggen.utils import EmbeddingResult


class WeightedSPINE(EnsmallenEmbedder):
    """Abstract class for Node2Vec algorithms."""

    def __init__(
        self,
        embedding_size: int = 100,
        use_edge_weights_as_probabilities: bool = False,
        verbose: bool = False,
        enable_cache: bool = False
    ):
        """Create new abstract Node2Vec method.

        Parameters
        --------------------------
        embedding_size: int = 100
            Dimension of the embedding.
        use_edge_weights_as_probabilities: bool = False
            Whether to treat the weights as probabilities.
        verbose: bool = False
            Whether to show loading bars.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._verbose = verbose
        self._model = models.WeightedSPINE(
            embedding_size=embedding_size,
            use_edge_weights_as_probabilities=use_edge_weights_as_probabilities,
        )

        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            embedding_size=5,
        )

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
    ) -> EmbeddingResult:
        """Return node embedding."""
        node_embedding = self._model.fit_transform(
            graph,
            verbose=self._verbose,
        ).T
        if return_dataframe:
            node_embedding = pd.DataFrame(
                node_embedding,
                index=graph.get_node_names()
            )
        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            node_embeddings=node_embedding
        )

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "WeightedSPINE"

    @classmethod
    def requires_edge_weights(cls) -> bool:
        return True

    @classmethod
    def requires_positive_edge_weights(cls) -> bool:
        return True

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
        return False
