"""EdgeLabelPredictionTransformer class to convert graphs to edge embeddings to execute edge prediction."""
from typing import Tuple, Union, List, Optional
import pandas as pd
import numpy as np
import warnings
from ensmallen import Graph  # pylint: disable=no-name-in-module

from embiggen.embedding_transformers.graph_transformer import GraphTransformer


class EdgeLabelPredictionTransformer:
    """EdgeLabelPredictionTransformer class to convert graphs to edge embeddings, with edge-labels."""

    def __init__(
        self,
        method: str = "Hadamard",
        aligned_mapping: bool = False,
        one_hot_encode_labels: bool = False
    ):
        """Create new EdgeLabelPredictionTransformer object.

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
        one_hot_encode_labels: bool = False
            Whether to one-hot encode the edge labels.
        """
        self._transformer = GraphTransformer(
            method=method,
            aligned_mapping=aligned_mapping,
        )
        self._one_hot_encode_labels = one_hot_encode_labels

    def fit(
        self, 
        node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
        node_type_feature: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
    ):
        """Fit the model.

        Parameters
        -------------------------
        node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
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
        graph: Graph,
        edge_features: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        behaviour_for_unknown_edge_labels: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return edge embedding for given graph using provided method.

        Parameters
        --------------------------
        graph: Graph,
            The graph whose edges are to be embedded and edge types extracted.
            It can either be an Graph or a list of lists of edges.
        edge_features: Optional[Union[np.ndarray, List[np.ndarray]]] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the provided graph.
        behaviour_for_unknown_edge_labels: Optional[str] = None
            Behaviour to be followed when encountering edges that do not
            have a known edge type. Possible values are:
            - drop: we drop these edges
            - keep: we keep these edges
            By default, we drop these edges.
            If the behaviour has not been specified and left to None,
            a warning will be raised to notify the user of this uncertainty.

        Raises
        --------------------------
        ValueError
            If embedding is not fitted.
        ValueError
            If the graph does not have edge types.
        ValueError
            If the graph does not contain known edge types.
        ValueError
            If the graph has a single edge type.
        NotImplementedError
            If the graph is a multi-graph, which is not currently supported.

        Returns
        --------------------------
        Tuple with X and y values.
        """
        if not graph.has_edge_types():
            raise ValueError(
                "The provided graph for the edge-label prediction does "
                "not contain edge-types."
            )
        
        if not graph.has_known_edge_types():
            raise ValueError(
                "The provided graph for the edge-label prediction does "
                "not contain known edge-types, that is, it contains "
                "an edge type vocabulary but no edge has an edge type "
                "assigned to it."
            )
        
        if graph.has_homogeneous_edge_types():
            raise ValueError(
                "The provided graph for the edge-label prediction contains "
                "edges of a single type, making predictions pointless."
            )
        
        if graph.is_multigraph():
            raise NotImplementedError(
                "Multi-graphs are not currently supported by this class."
            )
        
        if graph.has_singleton_edge_types():
            warnings.warn(
                "Please do be advised that this graph contains edges with "
                "a singleton edge type, that is an edge type that appears "
                "only once in the graph. Predictions on such rare edge types "
                "will be unlikely to generalize well."
            )
        
        edge_type_counts = graph.get_edge_type_names_counts_hashmap()
        most_common_edge_type_name, most_common_count = max(
            edge_type_counts.items(),
            key=lambda x: x[1]
        )
        least_common_edge_type_name, least_common_count = min(
            edge_type_counts.items(),
            key=lambda x: x[1]
        )
        
        if most_common_count > least_common_count * 20:
            warnings.warn(
                (
                    "Please do be advised that this graph defines "
                    "an unbalanced edge-label prediction task, with the "
                    "most common edge type `{}` appearing {} times, "
                    "while the least common one, `{}`, appears only `{}` times."
                    "Do take this into account when designing the edge-label prediction model."
                ).format(
                    most_common_edge_type_name, most_common_count,
                    least_common_edge_type_name, least_common_count
                )
            )
        if graph.has_unknown_edge_types() and behaviour_for_unknown_edge_labels is None:
            warnings.warn(
                "Please be advised that the provided graph for the edge-label "
                "prediction contains edges with unknown edge types. "
                "The edges with unknown edge labels will be dropped. "
                "You may specify the behavior (and silence the warnings) "
                "for these cases by using the `behaviour_for_unknown_edge_labels` "
                "parameter."
            )
            behaviour_for_unknown_edge_labels = "drop"

        edge_embeddings = self._transformer.transform(
            graph,
            node_types=graph,
            edge_features=edge_features
        )

        if self._one_hot_encode_labels:
            if graph.has_unknown_edge_types() and behaviour_for_unknown_edge_labels == "drop":
                edge_labels = graph.get_one_hot_encoded_edge_types()
            else:
                edge_labels = graph.get_one_hot_encoded_known_edge_types()
        else:
            if graph.has_unknown_edge_types() and behaviour_for_unknown_edge_labels == "drop":
                edge_labels = graph.get_known_edge_type_ids()
                edge_embeddings = edge_embeddings[graph.get_edges_with_known_edge_types_mask()]
            else:
                edge_labels = graph.get_directed_edge_type_ids()

        return edge_embeddings, edge_labels
