"""GraphTransformer class to convert graphs to edge embeddings."""
from typing import List, Union, Optional
import pandas as pd
import numpy as np
from ensmallen import Graph  # pylint: disable=no-name-in-module

from embiggen.embedding_transformers.edge_transformer import EdgeTransformer


class GraphTransformer:
    """GraphTransformer class to convert graphs to edge embeddings."""

    def __init__(
        self,
        method: str = "Hadamard",
        aligned_mapping: bool = False,
        include_both_undirected_edges: bool = True
    ):
        """Create new GraphTransformer object.

        Parameters
        ------------------------
        method: str = "hadamard"
            Method to use for the embedding.
            Can either be 'Hadamard', 'Sum', 'Average', 'L1', 'AbsoluteL1', 'L2' or 'Concatenate'.
        aligned_mapping: bool = False
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated edge embedding
            will be meaningless.
        include_both_undirected_edges: bool = True
            Whether to include both undirected edges when parsing an undirected
            graph, that is both the edge from source to destination and the edge
            from destination to source. While both edges should be included when
            training a model, as the model should learn about these simmetries
            in the graph, these edges are not necessary in the context of visualizations
            where they create redoundancy.
        """
        self._transformer = EdgeTransformer(
            method=method,
            aligned_mapping=aligned_mapping,
        )
        self._include_both_undirected_edges = include_both_undirected_edges
        self._aligned_mapping = aligned_mapping

    @property
    def method(self) -> str:
        """Return the used edge embedding method."""
        return self._transformer.method

    def fit(
        self,
        node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
        node_type_feature: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
    ):
        """Fit the model.

        Parameters
        -------------------------
        node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
            Node feature to use to fit the transformer.
        node_type_feature: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            Node type feature to use to fit the transformer.

        Raises
        -------------------------
        ValueError
            If the given method is None there is no need to call the fit method.
        """
        self._transformer.fit(
            node_feature,
            node_type_feature=node_type_feature
        )

    def transform(
        self,
        graph: Union[Graph, np.ndarray, List[List[str]], List[List[int]]],
        node_types: Optional[Union[Graph, List[Optional[List[str]]], List[Optional[List[int]]]]] = None,
        edge_features: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> np.ndarray:
        """Return edge embedding for given graph using provided method.

        Parameters
        --------------------------
        graph: Union[Graph, np.ndarray, List[List[str]], List[List[int]]],
            The graph whose edges are to embed.
            It can either be an Graph or a list of lists of edges.
        node_types: Optional[Union[Graph, List[Optional[List[str]]], List[Optional[List[int]]]]] = None,
            List of node types whose embedding is to be returned.
            This can be either a list of strings, or a graph, or if the
            aligned_mapping is setted, then this methods also accepts
            a list of ints.
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
            if self._aligned_mapping:
                if graph.is_directed() or self._include_both_undirected_edges:
                    edge_node_ids = (
                        graph.get_directed_source_node_ids(),
                        graph.get_directed_destination_node_ids(),
                    )
                else:
                    edge_node_ids = (
                        graph.get_source_node_ids(directed=False),
                        graph.get_destination_node_ids(directed=False),
                    )
            else:
                edge_node_ids = graph.get_directed_edge_node_names()
        else:
            edge_node_ids = graph

        if isinstance(edge_node_ids, List):
            edge_node_ids = np.array(edge_node_ids)
        if (
            isinstance(edge_node_ids, tuple) and
            len(edge_node_ids) == 2 and
            all(isinstance(e, np.ndarray) for e in edge_node_ids)
        ):
            if (
                len(edge_node_ids[0].shape) != 1 or
                len(edge_node_ids[1].shape) != 1 or
                edge_node_ids[0].shape[0] == 0 or
                edge_node_ids[1].shape[0] == 0 or
                edge_node_ids[0].shape[0] != edge_node_ids[1].shape[0]
            ):
                raise ValueError(
                    "When providing a tuple of numpy arrays containing the source and destination "
                    "node IDs, we expect to receive two arrays both with shape "
                    "with shape (number of edges,). "
                    f"The ones you have provided have shapes {edge_node_ids[0].shape} "
                    f"and {edge_node_ids[1].shape}."
                )
            sources = edge_node_ids[0]
            destinations = edge_node_ids[1]
        elif isinstance(edge_node_ids, np.ndarray):
            if len(edge_node_ids.shape) != 2 or edge_node_ids.shape[1] != 2 or edge_node_ids.shape[0] == 0:
                raise ValueError(
                    "When providing a numpy array containing the source and destination "
                    "node IDs representing the graph edges, we expect to receive an array "
                    f"with shape (number of edges, 2). The one you have provided has shape {edge_node_ids.shape}."
                )
            sources = edge_node_ids[:, 0]
            destinations = edge_node_ids[:, 1]

        if node_types is not None and self._transformer._transformer._node_type_feature is not None:
            if isinstance(node_types, Graph):
                if self._aligned_mapping:
                    source_node_types = [
                        node_types.get_node_type_ids_from_node_id(src)
                        for src in sources
                    ]
                    destination_node_types = [
                        node_types.get_node_type_ids_from_node_id(dst)
                        for dst in destinations
                    ]
                else:
                    source_node_types = [
                        node_types.get_node_type_names_from_node_name(src)
                        for src in sources
                    ]
                    destination_node_types = [
                        node_types.get_node_type_names_from_node_name(dst)
                        for dst in destinations
                    ]
            else:
                source_node_types, destination_node_types = node_types
        else:
            source_node_types = None
            destination_node_types = None

        return self._transformer.transform(
            sources,
            destinations,
            source_node_types=source_node_types,
            destination_node_types=destination_node_types,
            edge_features=edge_features
        )
