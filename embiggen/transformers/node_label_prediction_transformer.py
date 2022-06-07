"""NodeLabelPredictionTransformer class to convert graphs to node embeddings to execute node prediction."""
from typing import Tuple, Union, List, Optional
import pandas as pd
import numpy as np
import warnings
from ensmallen import Graph  # pylint: disable=no-name-in-module

from embiggen.transformers.node_transformer import NodeTransformer


class NodeLabelPredictionTransformer:
    """NodeLabelPredictionTransformer class to convert graphs to node embeddings, with node-labels."""

    def __init__(
        self,
        aligned_node_mapping: bool = False,
    ):
        """Create new NodeLabelPredictionTransformer object.

        Parameters
        ------------------------
        aligned_node_mapping: bool = False
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated node embedding
            will be meaningless.
        """
        self._transformer = NodeTransformer(
            aligned_node_mapping=aligned_node_mapping,
        )

    def fit(self, node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]):
        """Fit the model.

        Parameters
        -------------------------
        node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
            node_feature to use to fit the transformer.
        """
        self._transformer.fit(node_feature)

    def transform(
        self,
        graph: Graph,
        behaviour_for_unknown_node_labels: Optional[str] = "warn",
        shuffle: bool = False,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return node embedding for given graph using provided method.

        Parameters
        --------------------------
        graph: Graph,
            The graph whose nodes are to be embedded and node types extracted.
            It can either be an Graph or a list of lists of nodes.
        behaviour_for_unknown_node_labels: Optional[str] = "warn"
            Behaviour to be followed when encountering nodes that do not
            have a known node type. Possible values are:
            - drop: we drop these nodes
            - keep: we keep these nodes
            By default, we drop these nodes.
            If the behaviour has not been specified and left to "warn",
            a warning will be raised to notify the user of this uncertainty.
        shuffle: bool = False
            Whether to shuffle the labels.
            In some models, this is necessary.
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

        if graph.has_unknown_node_types() and behaviour_for_unknown_node_labels =="warn":
            warnings.warn(
                "Please be advised that the provided graph for the node-label "
                "prediction contains nodes with unknown node types. "
                "The nodes with unknown node labels will be dropped. "
                "You may specify the behavior (and silence the warnings) "
                "for these cases by using the `behaviour_for_unknown_node_labels` "
                "parameter."
            )
            behaviour_for_unknown_node_labels = "drop"

        node_features = self._transformer.transform(
            graph,
        )

        if graph.has_multilabel_node_types():
            node_labels = graph.get_one_hot_encoded_node_types()
        else:
            node_labels = np.fromiter(
                (
                    np.nan if node_type_id is None else node_type_id[0]
                    for node_type_id in (graph.get_node_type_ids_from_node_id(node_id)
                    for node_id in range(graph.get_nodes_number()))
                ),
                dtype=np.float32
            )

        if graph.has_unknown_node_types() and behaviour_for_unknown_node_labels == "drop":
            known_node_labels_mask = graph.get_nodes_with_known_node_types_mask()
            node_labels = node_labels[known_node_labels_mask]
            node_features = node_features[known_node_labels_mask]

        if shuffle:
            numpy_random_state = np.random.RandomState(  # pylint: disable=no-member
                seed=random_state
            )
            indices = numpy_random_state.permutation(node_features.shape[0])

            node_features, node_labels = node_features[indices], node_labels[indices]

        return node_features, node_labels