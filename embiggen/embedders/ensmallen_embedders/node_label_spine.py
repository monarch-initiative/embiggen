"""Module providing Node-label-based SPINE implementation."""
from typing import Optional, Dict, Any
from ensmallen import Graph
import pandas as pd
from ensmallen import models
from embiggen.embedders.ensmallen_embedders.ensmallen_embedder import EnsmallenEmbedder
from embiggen.utils import EmbeddingResult


class NodeLabelSPINE(EnsmallenEmbedder):
    """Class implementing the Node-label-based SPINE algorithm."""

    def __init__(
        self,
        dtype: Optional[str] = "u8",
        maximum_depth: Optional[int] = None,
        path: Optional[str] = None,
        verbose: bool = False,
        enable_cache: bool = False
    ):
        """Create new Node-label-based SPINE method.

        Parameters
        --------------------------
        dtype: Optional[str] = "u8"
            Dtype to use for the embedding.
        maximum_depth: Optional[int] = None
            Maximum depth of the shortest path.
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
        self._maximum_depth = maximum_depth
        self._path = path
        self._model = models.NodeLabelSPINE(
            verbose=self._verbose,
            maximum_depth=self._maximum_depth,
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
                maximum_depth=self._maximum_depth,
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
        return "Node-label-based SPINE"

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