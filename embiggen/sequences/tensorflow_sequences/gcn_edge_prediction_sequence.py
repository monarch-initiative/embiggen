"""Keras Sequence for Open-world assumption GCN."""
from typing import Tuple, List, Optional

import numpy as np
import tensorflow as tf
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence, VectorSequence
from embiggen.sequences.generic_sequences import EdgePredictionSequence


class GCNEdgePredictionSequence(Sequence):
    """Keras Sequence for running Neural Network on graph edge prediction."""

    def __init__(
        self,
        graph: Graph,
        support: Graph,
        kernel: tf.SparseTensor,
        return_node_types: bool = False,
        return_edge_types: bool = False,
        return_node_ids: bool = False,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
        use_edge_metrics: bool = False,
    ):
        """Create new Open-world assumption GCN training sequence for edge prediction.

        Parameters
        --------------------------------
        graph: Graph,
            The graph from which to sample the edges.
        support: Graph
            The graph to be used for the topological metrics.
        kernel: tf.SparseTensor
            The kernel to be used for the convolutions.
        return_node_types: bool = False
            Whether to return the node types.
        return_edge_types: bool = False
            Whether to return the edge types.
        return_node_ids: bool = False
            Whether to return the node IDs.
            These are needed when an embedding layer is used.
        node_features: List[np.ndarray]
            The node features to be used.
        node_type_features: Optional[List[np.ndarray]]
            The node type features to be used.
            For instance, these could be BERT embeddings of the
            description of the node types.
            When the graph has multilabel node types,
            we will average the features.
        edge_features: Optional[List[np.ndarray]] = None,
            The edge features to be used.
        use_edge_metrics: bool = False
            Whether to return the edge metrics.
        """
        if (
            return_node_types or
            node_type_features is not None
        ) and graph.has_unknown_node_types():
            raise ValueError(
                f"The provided graph {graph.get_name()} "
                "contains unknown node types but node types "
                "have been requested for the sequence."
            )
        
        self._graph = graph
        self._kernel = kernel
        if node_features is None:
            node_features = []
        self._node_features = [
            node_feature.astype(np.float32)
            for node_feature in node_features
        ]
        if return_node_ids:
            self._node_ids = graph.get_node_ids()
        else:
            self._node_ids = None

        if return_node_types or node_type_features is not None:
            if graph.has_multilabel_node_types():
                node_types = graph.get_one_hot_encoded_node_types()
            else:
                node_types = graph.get_single_label_node_type_ids()

        self._node_types = node_types if return_node_types else None

        if node_type_features is not None:
            if self._graph.has_multilabel_node_types():
                self._node_type_features = []
                minus_node_types = node_types - 1
                node_types_mask = node_types == 0
                for node_type_feature in node_type_features:
                    self._node_type_features.append(np.ma.array(
                        node_type_feature[minus_node_types],
                        mask=np.repeat(
                            node_types_mask,
                            node_type_feature.shape[1]
                        ).reshape((
                            *node_types.shape,
                            node_type_feature.shape[1]
                        ))
                    ).mean(axis=-2).data)
            else:
                self._node_type_features = [
                    node_type_feature[node_types]
                    for node_type_feature in node_type_features
                ]
        else:
            self._node_type_features = []

        self._sequence = EdgePredictionSequence(
            graph=graph,
            graph_used_in_training=support,
            return_node_types=False,
            return_edge_types=return_edge_types,
            use_edge_metrics=use_edge_metrics,
            batch_size=support.get_number_of_nodes()
        )

        self._edge_features = edge_features
        self._edge_features_sequences = None if edge_features is None else [
            VectorSequence(
                edge_feature,
                batch_size=graph.get_number_of_nodes(),
                shuffle=False
            )
            for edge_feature in edge_features
        ]

        self._use_edge_metrics = use_edge_metrics
        self._current_index = 0
        super().__init__(
            sample_number=graph.get_number_of_directed_edges(),
            batch_size=graph.get_number_of_nodes(),
        )

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be returned.

        Returns
        ---------------
        Return Tuple containing X and Y numpy arrays corresponding to given batch index.
        """
        values = self._sequence[idx][0]
        edge_features = None if self._edge_features_sequences is None else [
            edge_features_sequence[idx]
            for edge_features_sequence in self._edge_features_sequences
        ]
        # If necessary, we add the padding as the last batch may be
        # smaller than the required size (number of nodes).
        delta = self.batch_size - values[0].shape[0]
        if delta > 0:
            values = [
                np.pad(value, (0, delta) if len(value.shape)
                       == 1 else [(0, delta), (0, 0)])
                for value in values
                if value is not None
            ]
            if edge_features is not None:
                edge_features = [
                    np.pad(edge_feature, [(0, delta), (0, 0)])
                    for edge_feature in edge_features
                ]

        if edge_features is None:
            edge_features = []

        return (tuple([
            value
            for value in (
                *values,
                *edge_features,
                self._kernel,
                *self._node_features,
                *self._node_type_features,
                self._node_ids,
                self._node_types,
            )
            if value is not None
        ]),)
