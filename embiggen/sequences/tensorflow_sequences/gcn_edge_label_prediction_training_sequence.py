"""Keras Sequence for Open-world assumption GCN."""
from typing import List, Optional

import numpy as np
import tensorflow as tf
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import VectorSequence
from embiggen.sequences.tensorflow_sequences.gcn_edge_prediction_sequence import GCNEdgePredictionSequence


class GCNEdgeLabelPredictionTrainingSequence(GCNEdgePredictionSequence):
    """Keras Sequence for running Neural Network on graph edge-label prediction."""

    def __init__(
        self,
        graph: Graph,
        support: Graph,
        kernel: tf.SparseTensor,
        return_node_types: bool = False,
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
            For instance, these could be BERT embeddings of the
            description of the edges.
            When the graph has multilabel edges,
            we will average the features.
        use_edge_metrics: bool = False
            Whether to return the edge metrics.
        """
        super().__init__(
            graph=graph,
            support=support,
            kernel=kernel,
            return_node_types=return_node_types,
            return_edge_types=True,
            return_node_ids=return_node_ids,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            use_edge_metrics=use_edge_metrics,
        )

        self._known_edge_types_mask_sequence = VectorSequence(
            graph.get_known_edge_types_mask().astype(np.float32),
            batch_size=graph.get_number_of_nodes(),
            shuffle=False
        )

        # The index in the returned sequence that contains the
        # edge label is 2 (source and destination nodes).
        if return_node_types:
            self._edge_label_index = 4
        else:
            self._edge_label_index = 2

    def __getitem__(self, idx: int):
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be returned.

        Returns
        ---------------
        Return Tuple containing X and Y numpy arrays corresponding to given batch index.
        """
        batch = super().__getitem__(idx)[0]
        mask = self._known_edge_types_mask_sequence[idx]
        delta = self.batch_size - mask.size
        if delta > 0:
            mask = np.pad(mask, (0, delta))

        return (
            tuple([
                value
                for value in (
                    *batch[:self._edge_label_index],
                    *batch[self._edge_label_index+1:]
                )
            ]),
            batch[self._edge_label_index],
            mask
        )
