"""GraphTransformer class to convert graphs to edge embeddings."""
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from ensmallen import Graph  # pylint: disable=no-name-in-module

from embiggen.embedding_transformers.edge_transformer import EdgeTransformer


class GraphTransformer:
    """GraphTransformer class to convert graphs to edge embeddings."""

    def __init__(
        self,
        methods: Union[List[str], str] = "Hadamard",
        aligned_mapping: bool = False,
        include_both_undirected_edges: bool = True,
    ):
        """Create new GraphTransformer object.

        Parameters
        ------------------------
        methods: Union[List[str], str] = "Hadamard"
            Method to use for the edge embedding.
            If multiple edge embedding are provided, they
            will be Concatenated and fed to the model.
            The supported edge embedding methods are:
             * Hadamard: element-wise product
             * Sum: element-wise sum
             * Average: element-wise mean
             * L1: element-wise subtraction
             * AbsoluteL1: element-wise subtraction in absolute value
             * SquaredL2: element-wise subtraction in squared value
             * L2: element-wise squared root of squared subtraction
             * Concatenate: Concatenate of source and destination node features
             * Min: element-wise minimum
             * Max: element-wise maximum
             * L2Distance: vector-wise L2 distance - this yields a scalar
             * CosineSimilarity: vector-wise cosine similarity - this yields a scalar
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
            methods=methods,
            aligned_mapping=aligned_mapping,
        )
        self._include_both_undirected_edges = include_both_undirected_edges
        self._aligned_mapping = aligned_mapping

    def fit(
        self,
        node_feature: Union[
            pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]
        ],
        node_type_feature: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
    ):
        """Fit the model.

        Parameters
        -------------------------
        node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
            Node feature to use to fit the transformer.
        node_type_feature: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            Node type feature to use to fit the transformer.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray,
                                             List[Union[pd.DataFrame, np.ndarray]]]] = None
            Edge type feature to use to fit the transformer.

        Raises
        -------------------------
        ValueError
            If the given method is None there is no need to call the fit method.
        """
        self._transformer.fit(
            node_feature=node_feature,
            node_type_feature=node_type_feature,
            edge_type_features=edge_type_features,
        )

    def has_node_type_features(self) -> bool:
        """Return whether the transformer has a node type feature."""
        return self._transformer.has_node_type_features()

    def has_edge_type_features(self) -> bool:
        """Return whether the transformer has a edge type feature."""
        return self._transformer.has_edge_type_features()

    def is_aligned_mapping(self) -> bool:
        """Return whether the transformer has a aligned mapping."""
        return self._transformer.is_aligned_mapping()

    def transform(
        self,
        graph: Union[Graph, np.ndarray, List[List[str]], List[List[int]]],
        node_types: Optional[
            Union[Graph, List[Optional[List[str]]], List[Optional[List[int]]]]
        ] = None,
        edge_types: Optional[Union[Graph, List[str], List[int], np.ndarray]] = None,
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
            Optional edge features to be used as input Concatenated
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
            isinstance(edge_node_ids, tuple)
            and len(edge_node_ids) == 2
            and all(isinstance(e, np.ndarray) for e in edge_node_ids)
        ):
            if (
                len(edge_node_ids[0].shape) != 1
                or len(edge_node_ids[1].shape) != 1
                or edge_node_ids[0].shape[0] == 0
                or edge_node_ids[1].shape[0] == 0
                or edge_node_ids[0].shape[0] != edge_node_ids[1].shape[0]
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
            if (
                len(edge_node_ids.shape) != 2
                or edge_node_ids.shape[1] != 2
                or edge_node_ids.shape[0] == 0
            ):
                raise ValueError(
                    "When providing a numpy array containing the source and destination "
                    "node IDs representing the graph edges, we expect to receive an array "
                    f"with shape (number of edges, 2). The one you have provided has shape {edge_node_ids.shape}."
                )
            sources = edge_node_ids[:, 0]
            destinations = edge_node_ids[:, 1]

        if node_types is not None and self.has_node_type_features():
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

        assert (source_node_types is not None) == self.has_node_type_features()
        assert (destination_node_types is not None) == self.has_node_type_features()

        if isinstance(edge_types, Graph):
            edge_types.must_not_contain_unknown_edge_types()
            edge_types.must_not_be_multigraph()
            if not self.has_edge_type_features():
                raise ValueError(
                    "While the provided graph has edge types, "
                    "no edge features were provided to the graph transformer"
                )
            if self.is_aligned_mapping():
                if edge_types.is_directed() or self._include_both_undirected_edges:
                    edge_types = edge_types.get_imputed_directed_edge_type_ids(
                        imputation_edge_type_id=0
                    )
                else:
                    edge_types = edge_types.get_imputed_upper_triangular_edge_type_ids(
                        imputation_edge_type_id=0
                    )
            else:
                if edge_types.is_directed() or self._include_both_undirected_edges:
                    edge_types = edge_types.get_directed_edge_type_names()
                else:
                    edge_types = edge_types.get_upper_triangular_edge_type_names()

        assert (edge_types is not None) == self.has_edge_type_features()

        return self._transformer.transform(
            sources,
            destinations,
            source_node_types=source_node_types,
            destination_node_types=destination_node_types,
            edge_types=edge_types,
            edge_features=edge_features,
        )
