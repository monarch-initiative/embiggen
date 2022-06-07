"""GraphTransformer class to convert graphs to edge embeddings."""
from typing import List, Union, Optional
import pandas as pd
import numpy as np
from ensmallen import Graph  # pylint: disable=no-name-in-module

from embiggen.transformers.edge_transformer import EdgeTransformer


class GraphTransformer:
    """GraphTransformer class to convert graphs to edge embeddings."""

    def __init__(
        self,
        method: str = "Hadamard",
        aligned_node_mapping: bool = False,
    ):
        """Create new GraphTransformer object.

        Parameters
        ------------------------
        method: str = "hadamard",
            Method to use for the embedding.
            Can either be 'Hadamard', 'Sum', 'Average', 'L1', 'AbsoluteL1', 'L2' or 'Concatenate'.
        aligned_node_mapping: bool = False,
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated edge embedding
            will be meaningless.
        """
        self._transformer = EdgeTransformer(
            method=method,
            aligned_node_mapping=aligned_node_mapping,
        )
        self._aligned_node_mapping = aligned_node_mapping

    @property
    def numeric_node_ids(self) -> bool:
        """Return whether the transformer returns numeric node IDs."""
        return self._transformer.numeric_node_ids

    @property
    def method(self) -> str:
        """Return the used edge embedding method."""
        return self._transformer.method

    def fit(self, node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]):
        """Fit the model.

        Parameters
        -------------------------
        node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
            Node feature to use to fit the transformer.

        Raises
        -------------------------
        ValueError
            If the given method is None there is no need to call the fit method.
        """
        self._transformer.fit(node_feature)

    def transform(
        self,
        graph: Union[Graph, np.ndarray, List[List[str]], List[List[int]]],
        edge_features: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> np.ndarray:
        """Return edge embedding for given graph using provided method.

        Parameters
        --------------------------
        graph: Union[Graph, np.ndarray, List[List[str]], List[List[int]]],
            The graph whose edges are to embed.
            It can either be an Graph or a list of lists of edges.
        edge_features: Optional[Union[np.ndarray, List[np.ndarray]]] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the provided graph.

        Raises
        --------------------------
        ValueError,
            If embedding is not fitted.

        Returns
        --------------------------
        Numpy array of embeddings.
        """
        if isinstance(graph, Graph):
            if self._aligned_node_mapping:
                graph = (
                    graph.get_directed_source_node_ids(),
                    graph.get_directed_destination_node_ids(),
                )
            else:
                graph = graph.get_directed_edge_node_names()
        if isinstance(graph, List):
            graph = np.array(graph)
        if (
            isinstance(graph, tuple) and
            len(graph) == 2 and
            all(isinstance(e, np.ndarray) for e in graph)
        ):
            if (
                len(graph[0].shape) != 1 or
                len(graph[1].shape) != 1 or
                graph[0].shape[0] == 0 or
                graph[1].shape[0] == 0 or
                graph[0].shape[0] != graph[1].shape[0]
            ):
                raise ValueError(
                    "When providing a tuple of numpy arrays containing the source and destination "
                    "node IDs, we expect to receive two arrays both with shape "
                    "with shape (number of edges,). "
                    f"The ones you have provided have shapes {graph[0].shape} "
                    f"and {graph[1].shape}."
                )
            sources = graph[0]
            destinations = graph[1]
        elif isinstance(graph, np.ndarray):
            if len(graph.shape) != 2 or graph.shape[1] != 2 or graph.shape[0] == 0:
                raise ValueError(
                    "When providing a numpy array containing the source and destination "
                    "node IDs representing the graph edges, we expect to receive an array "
                   f"with shape (number of edges, 2). The one you have provided has shape {graph.shape}."
                )
            sources = graph[:, 0]
            destinations = graph[:, 1]
        return self._transformer.transform(sources, destinations, edge_features=edge_features)
