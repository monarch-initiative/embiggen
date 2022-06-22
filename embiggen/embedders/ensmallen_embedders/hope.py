"""Module providing HOPE implementation."""
from typing import Optional,  Dict, Any
from ensmallen import Graph
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from embiggen.utils.abstract_models import AbstractEmbeddingModel, EmbeddingResult


class HOPEEnsmallen(AbstractEmbeddingModel):
    """Class implementing the HOPE algorithm."""

    def __init__(
        self,
        embedding_size: int = 100,
        metric: str = "Jaccard",
        root_node_name: Optional[str] = None,
        enable_cache: bool = False
    ):
        """Create new HOPE method.

        Parameters
        --------------------------
        embedding_size: int = 100
            Dimension of the embedding.
        metric: str = "Jaccard"
            The metric to use. Can either be
            `Jaccard`, for the traditional edge Jaccard,
            or alternatively the `Ancestors Jaccard` Jaccard,
            for which is mandatory to provide the root node, or
            alternatively `Adamic-Adar`.
        root_node_name: Optional[str] = None
            Root node to use when the ancestors mode for
            the Jaccard index is selected.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._metric = metric
        self._root_node_name = root_node_name

        if root_node_name is None and metric == "Ancestors Jaccard":
            raise ValueError(
                "The provided metric is `Ancestors Jaccard`, but "
                "the root node name was not provided."
            )
        if root_node_name is not None and metric != "Ancestors Jaccard":
            raise ValueError(
                "The provided metric is not `Ancestors Jaccard`, but "
                "the root node name was provided. It is unclear "
                "what to do with this parameter."
            )

        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return {
            **super().parameters(),
            **dict(
                metric=self._metric,
                root_node_name=self._root_node_name
            )
        }

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
        verbose: bool = True
    ) -> EmbeddingResult:
        """Return node embedding."""
        if self._metric == "Jaccard":
            edges, weights = graph.get_jaccard_coo_matrix()
        elif self._metric == "Ancestors Jaccard":
            edges, weights = graph.get_ancestors_jaccard_coo_matrix(
                graph.get_breadth_first_search_from_node_names(
                    src_node_name=self.root_node_name,
                    compute_predecessors=True
                )
            )
        elif self._metric == "Adamic-Adar":
            edges, weights = graph.get_adamic_adar_coo_matrix()
        else:
            raise NotImplementedError(
                f"The provided metric {self._metric} "
                "is not currently supported. The supported "
                "metrics are `Jaccard`, `Ancestors Jaccard` and "
                "`Adamic-Adar`."
            )

        coo = coo_matrix(
            (weights, (edges[:, 0], edges[:, 1])),
            shape=(
                graph.get_nodes_number(),
                graph.get_nodes_number()
            ),
            dtype=np.float32
        )

        U, sigmas, Vt = svds(coo, k=int(self._embedding_size / 2))
        sigmas = np.diagflat(np.sqrt(sigmas))
        left_embedding = np.dot(U, sigmas)
        right_embedding = np.dot(Vt.T, sigmas)

        if return_dataframe:
            node_names = graph.get_node_names()
            left_embedding = pd.DataFrame(
                left_embedding,
                index=node_names
            )
            right_embedding = pd.DataFrame(
                right_embedding,
                index=node_names
            )
        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            node_embeddings=[left_embedding, right_embedding]
        )

    @staticmethod
    def task_name() -> str:
        return "Node Embedding"

    @staticmethod
    def model_name() -> str:
        """Returns name of the model."""
        return "HOPE"

    @staticmethod
    def library_name() -> str:
        return "Ensmallen"

    @staticmethod
    def requires_nodes_sorted_by_decreasing_node_degree() -> bool:
        return False

    @staticmethod
    def is_topological() -> bool:
        return True

    @staticmethod
    def requires_node_types() -> bool:
        return False

    @staticmethod
    def requires_edge_types() -> bool:
        return False

    @staticmethod
    def requires_edge_weights() -> bool:
        return False

    @staticmethod
    def requires_positive_edge_weights() -> bool:
        return False

    @staticmethod
    def can_use_edge_weights() -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False

    @staticmethod
    def can_use_node_types() -> bool:
        """Returns whether the model can optionally use node types."""
        return False

    @staticmethod
    def can_use_edge_types() -> bool:
        """Returns whether the model can optionally use edge types."""
        return False

    @staticmethod
    def is_stocastic() -> bool:
        """Returns whether the model is stocastic and has therefore a random state."""
        return False
