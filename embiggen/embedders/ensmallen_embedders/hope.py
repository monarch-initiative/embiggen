"""Module providing HOPE implementation."""
from typing import Optional,  Dict, Any, List
from ensmallen import Graph
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds as sparse_svds
from sklearn.utils.extmath import randomized_svd
from userinput.utils import must_be_in_set
from embiggen.embedders.ensmallen_embedders.ensmallen_embedder import EnsmallenEmbedder
from embiggen.utils import EmbeddingResult


class HOPEEnsmallen(EnsmallenEmbedder):
    """Class implementing the HOPE algorithm."""

    def __init__(
        self,
        embedding_size: int = 100,
        metric: str = "Neighbours Intersection size",
        root_node_name: Optional[str] = None,
        verbose: bool = False,
        enable_cache: bool = False
    ):
        """Create new HOPE method.

        Parameters
        --------------------------
        embedding_size: int = 100
            Dimension of the embedding.
        metric: str = "Neighbours Intersection size"
            The metric to use.
            You can either use:
            - Jaccard
            - Neighbours Intersection size
            - Ancestors Jaccard
            - Ancestors size
            - Adamic-Adar
            - Adjacency
            - Laplacian
            - Left Normalized Laplacian
            - Right Normalized Laplacian
            - Symmetric Normalized Laplacian
            - Resnik
        root_node_name: Optional[str] = None
            Root node to use when the ancestors mode for
            the Jaccard index is selected.
        verbose: bool = False
            Whether to show loading bars.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        metric = must_be_in_set(metric, self.get_available_metrics(), "metric")
        ancestral_metric = ("Ancestors Jaccard", "Ancestors size")
        if root_node_name is None and metric in ancestral_metric:
            raise ValueError(
                f"The provided metric is `{metric}`, but "
                "the root node name was not provided."
            )
        if root_node_name is not None and metric not in ancestral_metric:
            raise ValueError(
                "The provided metric is not based on ancestors, but "
                f"the root node name `{root_node_name}` was provided. It is unclear "
                "what to do with this parameter."
            )

        self._metric = metric
        self._root_node_name = root_node_name
        self._verbose = verbose

        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return dict(
            **super().parameters(),
            **dict(
                metric=self._metric,
                verbose=self._verbose,
                root_node_name=self._root_node_name
            )
        )

    @classmethod
    def get_available_metrics(cls) -> List[str]:
        """Returns list of the available metrics."""
        return [
            "Jaccard",
            "Shortest Paths",
            "Neighbours Intersection size",
            "Ancestors Jaccard",
            "Ancestors size",
            "Adamic-Adar",
            "Adjacency",
            "Laplacian",
            "Modularity",
            "Left Normalized Laplacian",
            "Right Normalized Laplacian",
            "Symmetric Normalized Laplacian",
        ]

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
    ) -> EmbeddingResult:
        """Return node embedding."""
        matrix = None
        if self._metric == "Jaccard":
            edges, weights = graph.get_jaccard_coo_matrix()
        elif self._metric == "Laplacian":
            edges, weights = graph.get_laplacian_coo_matrix()
        elif self._metric == "Shortest Paths":
            matrix = graph.get_shortest_paths_matrix()
        elif self._metric == "Modularity":
            matrix = graph.get_dense_modularity_matrix()
        elif self._metric == "Left Normalized Laplacian":
            edges, weights = graph.get_left_normalized_laplacian_coo_matrix()
        elif self._metric == "Right Normalized Laplacian":
            edges, weights = graph.get_right_normalized_laplacian_coo_matrix()
        elif self._metric == "Symmetric Normalized Laplacian":
            edges, weights = graph.get_symmetric_normalized_laplacian_coo_matrix()
        elif self._metric == "Neighbours Intersection size":
            edges, weights = graph.get_neighbours_intersection_size_coo_matrix()
        elif self._metric == "Ancestors Jaccard":
            matrix = graph.get_shared_ancestors_jaccard_adjacency_matrix(
                graph.get_breadth_first_search_from_node_names(
                    src_node_name=self._root_node_name,
                    compute_predecessors=True
                ),
                verbose=self._verbose
            )
        elif self._metric == "Ancestors size":
            matrix = graph.get_shared_ancestors_size_adjacency_matrix(
                graph.get_breadth_first_search_from_node_names(
                    src_node_name=self._root_node_name,
                    compute_predecessors=True
                ),
                verbose=self._verbose
            )
        elif self._metric == "Adamic-Adar":
            edges, weights = graph.get_adamic_adar_coo_matrix()
        elif self._metric == "Adjacency":
            edges, weights = graph.get_directed_edge_node_ids(), np.ones(
                graph.get_number_of_directed_edges())

        if matrix is None:
            matrix = coo_matrix(
                (weights, (edges[:, 0], edges[:, 1])),
                shape=(
                    graph.get_number_of_nodes(),
                    graph.get_number_of_nodes()
                ),
                dtype=np.float32
            )
            
            U, sigmas, Vt = sparse_svds(
                matrix,
                k=int(self._embedding_size / 2)
            )
        else:
            U, sigmas, Vt = randomized_svd(
                matrix,
                n_components=int(self._embedding_size / 2)
            )
        
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

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "HOPE"

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
        return False
