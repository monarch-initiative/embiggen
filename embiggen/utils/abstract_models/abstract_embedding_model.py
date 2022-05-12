"""Module providing abstract classes for embedding models."""
from typing import Dict, Union, Any
from ensmallen import Graph
import warnings
import numpy as np
import pandas as pd
from .abstract_decorator import abstract_class
from .abstract_model import AbstractModel


@abstract_class
class AbstractEmbeddingModel(AbstractModel):
    """Class defining properties of an abstract embedding model."""

    def __init__(self, embedding_size: int):
        """Create new embedding model.

        Parameters
        ---------------------
        embedding_size: int
            The dimensionality of the embedding.
        """
        super().__init__()
        if not isinstance(embedding_size, int) or embedding_size == 0:
            raise ValueError(
                "The embedding size should be a strictly positive integer "
                f"but {embedding_size} was provided."
            )
        self._embedding_size = embedding_size

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the embedding model."""
        return {
            "embedding_size": self._embedding_size
        }

    def is_topological(self) -> bool:
        """Returns whether this embedding is based on graph topology."""
        raise NotImplementedError((
            "The `is_topological` method must be implemented "
            "in the child classes of abstract model."
        ))

    def requires_nodes_sorted_by_decreasing_node_degree(self) -> bool:
        """Returns whether this embedding requires the node degrees to be sorted."""
        raise NotImplementedError((
            "The `requires_nodes_sorted_by_decreasing_node_degree` method must be implemented "
            "in the child classes of abstract model."
        ))

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
        verbose: bool = True
    ) -> Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
        """Run embedding on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        verbose: bool
            Whether to show loading bars.
        """
        raise NotImplementedError((
            "The `_fit_transform` method must be implemented "
            "in the child classes of abstract model."
        ))

    def fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
        verbose: bool = True
    ) -> Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
        """Execute embedding on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run embedding on.
        return_dataframe: bool = True
            Whether to return a pandas DataFrame with the embedding.
        verbose: bool
            Whether to show loading bars.

        Returns
        --------------------
        It either returns a numpy array, a dataframe, or when the embedding model
        obtains multiple embeddings for different properties of the graph, such as
        node types and edge types, it returns a dictionary with the name of the embedding
        and its values.
        """
        if not graph.has_nodes():
            raise ValueError("The provided graph is empty.")

        if self.requires_nodes_sorted_by_decreasing_node_degree():
            if not graph.has_nodes_sorted_by_decreasing_outbound_node_degree():
                raise ValueError(
                    "The given graph does not have the nodes sorted by decreasing "
                    "order, therefore the negative sampling (which follows a zipfian "
                    "distribution) would not approximate well the Softmax.\n"
                    "In order to sort the given graph in such a way that the node IDs "
                    "are sorted by decreasing outbound node degrees, you can use "
                    "the Graph method `graph.sort_by_decreasing_outbound_node_degree()`."
                )

        if self.is_topological():
            if not graph.has_edges():
                raise ValueError("The provided graph does not have edges.")

            if graph.has_disconnected_nodes():
                warnings.warn(
                    (
                        "Please be advised that this graph contains {} disconnected nodes. "
                        "Consider that node embedding algorithms that only use topological information "
                        "such as CBOW, GloVe, SPINE and SkipGram are not able to provide meaningful "
                        "embeddings for these nodes, and their embedding will be generally "
                        "far away from any other node. It is also possible that all disconnected nodes "
                        "will receive a relatively similar node embedding. "
                        "Consider dropping them by using the `graph.remove_disconnected_nodes()` method."
                    ).format(graph.get_singleton_nodes_number())
                )

        return self._fit_transform(
            graph=graph,
            return_dataframe=return_dataframe,
            verbose=verbose
        )
