"""Module providing abstract classes for embedding models."""
from typing import Dict, Any, Optional, Union
from ensmallen import Graph
from ensmallen.datasets import get_dataset
import warnings
from cache_decorator import Cache
from embiggen.utils.abstract_models.abstract_model import AbstractModel, abstract_class
from embiggen.utils.abstract_models.embedding_result import EmbeddingResult


@abstract_class
class AbstractEmbeddingModel(AbstractModel):
    """Class defining properties of an abstract embedding model."""

    def __init__(
        self,
        embedding_size: int,
        enable_cache: bool = False,
        random_state: Optional[int] = None
    ):
        """Create new embedding model.

        Parameters
        ---------------------
        embedding_size: int
            The dimensionality of the embedding.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        random_state: Optional[int] = None
            The random state to use if the model is stocastic.
        """
        super().__init__(random_state=random_state)
        if not isinstance(embedding_size, int) or embedding_size == 0:
            raise ValueError(
                "The embedding size should be a strictly positive integer "
                f"but {embedding_size} was provided."
            )
        self._embedding_size = embedding_size
        self._enable_cache = enable_cache

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the embedding model."""
        return dict(
            **super().parameters(),
            embedding_size=self._embedding_size
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(embedding_size=5)

    @classmethod
    def task_name(cls) -> str:
        return "Node Embedding"

    @classmethod
    def requires_nodes_sorted_by_decreasing_node_degree(cls) -> bool:
        """Returns whether this embedding requires the node degrees to be sorted."""
        raise NotImplementedError((
            "The `requires_nodes_sorted_by_decreasing_node_degree` method must be implemented "
            "in the child classes of abstract model."
        ))

    @classmethod
    def get_minimum_required_number_of_node_types(cls) -> int:
        """Requires minimum number of required node types."""
        return 0

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
    ) -> EmbeddingResult:
        """Run embedding on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        """
        raise NotImplementedError((
            "The `_fit_transform` method must be implemented "
            "in the child classes of abstract model."
        ))

    @Cache(
        cache_path="{cache_dir}/{self.model_name()}/{self.library_name()}/{graph.get_name()}/{_hash}.pkl.gz",
        cache_dir="embedding",
        enable_cache_arg_name="self._enable_cache",
    )
    def _cached_fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
    ) -> EmbeddingResult:
        """Execute embedding on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run embedding on.
        return_dataframe: bool = True
            Whether to return a pandas DataFrame with the embedding.

        Returns
        --------------------
        An embedding result, wrapping the complexity of a generic embedding.
        """
        if not graph.has_nodes():
            raise ValueError(
                f"The provided graph {graph.get_name()} is empty."
            )

        if self.requires_nodes_sorted_by_decreasing_node_degree():
            if not graph.has_nodes_sorted_by_decreasing_outbound_node_degree():
                raise ValueError(
                    f"The given graph {graph.get_name()} does not have the nodes sorted by decreasing "
                    "order, therefore the negative sampling (which follows a scale free "
                    "distribution) would not approximate well the Softmax.\n"
                    "In order to sort the given graph in such a way that the node IDs "
                    "are sorted by decreasing outbound node degrees, you can use "
                    "the Graph method `graph.sort_by_decreasing_outbound_node_degree()`."
                )

        if self.requires_node_types() and not graph.has_node_types():
            raise ValueError(
                f"The provided graph {graph.get_name()} does not have node types, but "
                f"the {self.model_name()} requires node types."
            )
        
        if self.requires_node_types() and graph.get_number_of_node_types() <= 1:
            raise ValueError(
                f"The {self.model_name()} requires the graph to have "
                f"at least {self.get_minimum_required_number_of_node_types()} node types, "
                f"but the provided one has {graph.get_number_of_node_types()} "
                "node types."
            )

        if self.requires_edge_types() and not graph.has_edge_types():
            raise ValueError(
                f"The provided graph {graph.get_name()} does not have edge types, but "
                f"the {self.model_name()} requires edge types."
            )

        if self.requires_edge_weights() and not graph.has_edge_weights():
            raise ValueError(
                f"The provided graph {graph.get_name()} does not have edge weights, but "
                f"the {self.model_name()} requires edge weights."
            )

        if self.requires_positive_edge_weights() and graph.has_edge_weights() and graph.has_negative_edge_weights():
            raise ValueError(
                f"The provided graph {graph.get_name()} has negative edge weights, but "
                f"the {self.model_name()} requires strictly positive edge weights."
            )

        if self.is_topological():
            if not graph.has_edges():
                raise ValueError(
                    f"The provided graph {graph.get_name()} does not have edges."
                )

            if graph.has_disconnected_nodes():
                warnings.warn(
                    (
                        f"Please be advised that the {graph.get_name()} graph "
                        f"contains {graph.get_number_of_disconnected_nodes()} disconnected nodes. "
                        "Consider that node embedding algorithms that only use topological information "
                        "such as CBOW, GloVe, SPINE and SkipGram are not able to provide meaningful "
                        "embeddings for these nodes, and their embedding will be generally "
                        "far away from any other node. It is also possible that all disconnected nodes "
                        "will receive a relatively similar node embedding. "
                        "Consider dropping them by using the `graph.remove_disconnected_nodes()` method."
                    )
                )

        result = self._fit_transform(
            graph=graph,
            return_dataframe=return_dataframe,
        )

        if not isinstance(result, EmbeddingResult):
            raise NotImplementedError(
                f"The embedding result produced by the {self.model_name()} method "
                f"from the library {self.library_name()} implemented in the class "
                f"called {self.__class__.__name__} does not return an Embeddingresult "
                f"but returns an object of type {type(result)}."
            )

        return result

    def fit_transform(
        self,
        graph: Union[Graph, str],
        repository: Optional[str] = None,
        version: Optional[str] = None,
        return_dataframe: bool = True,
    ) -> EmbeddingResult:
        """Execute embedding on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run embedding on.
        repository: Optional[str] = None
            The repository from where to retrieve these graphs.
            This only applies for the graph names that are available
            from the ensmallen automatic retrieval.
        version: Optional[str] = None
            The version of the graphs to be retrieved.
            When this is left to none, the retrieved version will be
            the one that has been indicated to be the most recent one.
            This only applies for the graph names that are available
            from the ensmallen automatic retrieval.
        return_dataframe: bool = True
            Whether to return a pandas DataFrame with the embedding.

        Returns
        --------------------
        An embedding result, wrapping the complexity of a generic embedding.
        """
        if isinstance(graph, str):
            graph = get_dataset(
                name=graph,
                repository=repository,
                version=version
            )()
        return self._cached_fit_transform(
            graph=graph,
            return_dataframe=return_dataframe,
        )
