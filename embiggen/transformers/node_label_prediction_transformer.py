"""NodeLabelPredictionTransformer class to convert graphs to node embeddings to execute node prediction."""
from typing import Tuple, Union, List, Optional
import pandas as pd
import numpy as np
import warnings
from ensmallen import Graph  # pylint: disable=no-name-in-module

from .node_transformer import NodeTransformer


class NodeLabelPredictionTransformer:
    """NodeLabelPredictionTransformer class to convert graphs to node embeddings, with node-labels."""

    def __init__(
        self,
        aligned_node_mapping: bool = False,
        one_hot_encode_labels: bool = False
    ):
        """Create new NodeLabelPredictionTransformer object.

        Parameters
        ------------------------
        aligned_node_mapping: bool = False
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated node embedding
            will be meaningless.
        one_hot_encode_labels: bool = False
            Whether to one-hot encode the node labels.
        """
        self._transformer = NodeTransformer(
            aligned_node_mapping=aligned_node_mapping,
        )
        self._one_hot_encode_labels = one_hot_encode_labels

    def fit(self, embedding: pd.DataFrame):
        """Fit the model.

        Parameters
        -------------------------
        embedding: pd.DataFrame,
            Embedding to use to fit the transformer.
            This is a pandas DataFrame and NOT a numpy array because we need
            to be able to remap correctly the vector embeddings in case of
            graphs that do not respect the same internal node mapping but have
            the same node set. It is possible to remap such graphs using
            Ensmallen's remap method but it may be less intuitive to users.
        """
        self._transformer.fit(embedding)

    def transform(
        self,
        graph: Graph,
        behaviour_for_unknown_node_labels: Optional[str] = None,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return node embedding for given graph using provided method.

        Parameters
        --------------------------
        graph: Graph,
            The graph whose nodes are to be embedded and node types extracted.
            It can either be an Graph or a list of lists of nodes.
        behaviour_for_unknown_node_labels: Optional[str] = None
            Behaviour to be followed when encountering nodes that do not
            have a known node type. Possible values are:
            - drop: we drop these nodes
            - keep: we keep these nodes
            By default, we drop these nodes.
            If the behaviour has not been specified and left to None,
            a warning will be raised to notify the user of this uncertainty.
        random_state: int = 42,
            The random state to use to shuffle the labels.

        Raises
        --------------------------
        ValueError
            If embedding is not fitted.
        ValueError
            If the graph does not have node types.
        ValueError
            If the graph does not contain known node types.
        ValueError
            If the graph has a single node type.
        NotImplementedError
            If the graph is a multi-graph, which is not currently supported.

        Returns
        --------------------------
        Tuple with X and y values.
        """
        if not graph.has_node_types():
            raise ValueError(
                "The provided graph for the node-label prediction does "
                "not contain node-types."
            )
        if not graph.has_known_node_types():
            raise ValueError(
                "The provided graph for the node-label prediction does "
                "not contain known node-types, that is, it contains "
                "an node type vocabulary but no node has an node type "
                "assigned to it."
            )
        if graph.has_homogeneous_node_types():
            raise ValueError(
                "The provided graph for the node-label prediction contains "
                "nodes of a single type, making predictions pointless."
            )
        if graph.has_singleton_node_types():
            warnings.warn(
                "Please do be advised that this graph contains nodes with "
                "a singleton node type, that is an node type that appears "
                "only once in the graph. Predictions on such rare node types "
                "will be unlikely to generalize well."
            )

        node_type_counts = graph.get_node_type_names_counts_hashmap()
        most_common_node_type_name, most_common_count = max(
            node_type_counts.items(),
            key=lambda x: x[1]
        )
        least_common_node_type_name, least_common_count = min(
            node_type_counts.items(),
            key=lambda x: x[1]
        )
        if most_common_count > least_common_count * 20:
            warnings.warn(
                (
                    "Please do be advised that this graph defines "
                    "an unbalanced node-label prediction task, with the "
                    "most common node type `{}` appearing {} times, "
                    "while the least common one, `{}`, appears only `{}` times. "
                    "Do take this into account when designing the node-label prediction model."
                ).format(
                    most_common_node_type_name, most_common_count,
                    least_common_node_type_name, least_common_count
                )
            )
        if graph.has_unknown_node_types() and behaviour_for_unknown_node_labels is None:
            warnings.warn(
                "Please be advised that the provided graph for the node-label "
                "prediction contains nodes with unknown node types. "
                "The nodes with unknown node labels will be dropped. "
                "You may specify the behavior (and silence the warnings) "
                "for these cases by using the `behaviour_for_unknown_node_labels` "
                "parameter."
            )
            behaviour_for_unknown_node_labels = "drop"

        node_embeddings = self._transformer.transform(
            graph,
        )

        if self._one_hot_encode_labels:
            node_labels = graph.get_one_hot_encoded_node_types()
        else:
            node_labels = graph.get_node_type_ids()

        if graph.has_unknown_node_types() and behaviour_for_unknown_node_labels == "drop":
            known_node_labels_mask = graph.get_nodes_with_known_node_types_mask()
            node_labels = node_labels[known_node_labels_mask]
            node_embeddings = node_embeddings[known_node_labels_mask]

        numpy_random_state = np.random.RandomState(  # pylint: disable=no-member
            seed=random_state
        )

        indices = numpy_random_state.permutation(node_labels.size)

        return node_embeddings[indices], node_labels[indices]
