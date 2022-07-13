"""Module providing abstract Node2Vec implementation."""
from typing import Dict, Any
from ensmallen import Graph
import pandas as pd
from userinput.utils import must_be_in_set
from ensmallen import models
from embiggen.utils.abstract_models import abstract_class
from embiggen.embedders.ensmallen_embedders.ensmallen_embedder import EnsmallenEmbedder
from embiggen.utils import EmbeddingResult


@abstract_class
class Node2VecEnsmallen(EnsmallenEmbedder):
    """Abstract class for Node2Vec algorithms."""

    MODELS = {
        "CBOW": models.CBOW,
        "SkipGram": models.SkipGram,
        "WalkletsCBOW": models.WalkletsCBOW,
        "WalkletsSkipGram": models.WalkletsSkipGram,
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
        model_name = must_be_in_set(model_name, self.MODELS.keys(), "model name")
        self._model_kwargs = model_kwargs
        self._model = Node2VecEnsmallen.MODELS[model_name](**model_kwargs)

        super().__init__(
            embedding_size=model_kwargs["embedding_size"],
            enable_cache=enable_cache,
            random_state=model_kwargs["random_state"]
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **EnsmallenEmbedder.smoke_test_parameters(),
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