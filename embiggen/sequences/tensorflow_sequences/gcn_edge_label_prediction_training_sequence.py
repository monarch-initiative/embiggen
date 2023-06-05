"""Keras Sequence for edge-label prediction GNN and GCN."""
from typing import List, Optional, Union, Type

import numpy as np
import tensorflow as tf
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import VectorSequence
from embiggen.utils import AbstractEdgeFeature
from embiggen.sequences.tensorflow_sequences.gcn_edge_label_prediction_sequence import GCNEdgeLabelPredictionSequence


class GCNEdgeLabelPredictionTrainingSequence(GCNEdgeLabelPredictionSequence):
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
        edge_features: Optional[Union[np.ndarray, Type[AbstractEdgeFeature], List[Union[Type[AbstractEdgeFeature], np.ndarray]]]] = None,
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
            return_node_ids=return_node_ids,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            use_edge_metrics=use_edge_metrics
        )

        if graph.is_directed():
            edge_types = graph.get_imputed_directed_edge_type_ids(
                imputation_edge_type_id=0
            )
            mask = graph.get_directed_edges_with_known_edge_types_mask()
        else:
            edge_types = graph.get_imputed_upper_triangular_edge_type_ids(
                imputation_edge_type_id=0
            )
            mask = graph.get_upper_triangular_known_edge_types_mask()

        if not isinstance(edge_types, np.ndarray):
            raise RuntimeError(
                "The edge types should be a numpy array, "
                f"found {type(edge_types)} instead. "
                "This is likely an Ensmallen bug, "
                "please open an issue at "
                "the GRAPE GitHub repository."
            )

        self._edge_types = VectorSequence(
            edge_types,
            batch_size=self._batch_size,
        )

        self._mask = VectorSequence(
            mask.astype(np.uint8),
            batch_size=self._batch_size,
        )

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
        features = super().__getitem__(idx)[0]

        edge_types = self._edge_types[idx]
        mask = self._mask[idx]

        # If this last batch is smaller than the batch size, we need to pad it.
        # This is necessary because in GCNs, the batch size is fixed to the number of nodes.
        delta = self.batch_size - edge_types.shape[0]
        if delta > 0:
            edge_types = np.pad(edge_types, (0, delta))
            mask = np.pad(mask, (0, delta))

        return (
            features,
            edge_types,
            mask
        )
