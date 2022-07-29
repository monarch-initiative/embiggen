"""Module providing Node-label-based WINE implementation."""
from typing import Optional, Dict, Any
from ensmallen import Graph
import pandas as pd
from ensmallen import models
from embiggen.embedders.ensmallen_embedders.ensmallen_embedder import EnsmallenEmbedder
from embiggen.utils import EmbeddingResult


class NodeLabelWINE(EnsmallenEmbedder):
    """Class implementing the Node-label-based WINE algorithm."""

    def __init__(
        self,
        dtype: Optional[str] = "u8",
        walk_length: Optional[int] = None,
        path: Optional[str] = None,
        verbose: bool = False,
        enable_cache: bool = False
    ):
        """Create new Node-label-based WINE method.

        Parameters
        --------------------------
        dtype: Optional[str] = "u8"
            Dtype to use for the embedding.
        walk_length: int = 2
            Length of the random walk.
            By default 2, to capture exclusively the immediate context.
        path: Optional[str] = None
            Path where to store the mmap-ed embedding.
            This parameter is necessary to embed very large graphs.
        verbose: bool = False
            Whether to show loading bars.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._dtype = dtype
        self._verbose = verbose
        self._walk_length = walk_length
        self._path = path
        self._model = models.NodeLabelWINE(
            verbose=self._verbose,
            walk_length=self._walk_length,
            path=self._path
        )

        super().__init__(
            enable_cache=enable_cache
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return dict(
            **{
                key: value
                for key, value in super().parameters().items()
                if key != "embedding_size"
            },
            **dict(
                dtype=self._dtype,
                walk_length=self._walk_length,
                path=self._path,
            )
        )
    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict()

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
    ) -> EmbeddingResult:
        """Return node embedding."""
        node_embedding = self._model.fit_transform(
            graph,
            dtype=self._dtype,
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

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "Node-label-based WINE"

    @classmethod
    def requires_node_types(cls) -> bool:
        """Returns whether the model requires node types."""
        return True

    @classmethod
    def get_minimum_required_number_of_node_types(cls) -> int:
        """Requires minimum number of required node types."""
        return 2

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False

    @classmethod
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can optionally use edge types."""
        return False

    @classmethod
    def is_stocastic(cls) -> bool:
        """Returns whether the model is stocastic and has therefore a random state."""
        return False