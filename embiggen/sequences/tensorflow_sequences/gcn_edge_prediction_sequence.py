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
        if not graph.has_edges():
            raise ValueError(
                f"An empty instance of graph {graph.get_name()} was provided!"
            )
        self._graph = graph
        self._kernel = kernel
        if node_features is None:
            node_features = []
        self._node_features = node_features
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
                minus_node_types = node_type_feature[node_types - 1]
                node_types_mask = node_types==0
                for node_type_feature in self._node_type_features:
                    self._node_type_features.append(np.ma.array(
                        node_type_feature[minus_node_types],
                        mask=np.repeat(
                            node_types_mask,
                            node_type_feature.shape[1]
                        ).reshape((
                            *node_types.shape,
                            node_type_feature.shape[1]
                        ))
                    ).mean(axis=-1).data)
            else:
                if self._graph.has_unknown_node_types():
                    self._node_type_features = []
                    minus_node_types = node_type_feature[node_types - 1]
                    node_types_mask = node_types==0
                    for node_type_feature in self._node_type_features:
                        ntf = node_type_feature[minus_node_types]
                        # Masking the unknown values to zero.
                        ntf[node_types_mask] = 0.0
                        self._node_type_features.append(ntf)
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
            batch_size=support.get_nodes_number()
        )

        self._edge_features_sequence = None if edge_features is None else VectorSequence(
            edge_features,
            batch_size=graph.get_nodes_number(),
            shuffle=False
        )

        self._use_edge_metrics = use_edge_metrics
        self._current_index = 0
        super().__init__(
            sample_number=graph.get_number_of_directed_edges(),
            batch_size=graph.get_nodes_number(),
        )

    def __call__(self):
        """Return next batch using an infinite generator model."""
        self._current_index += 1
        return (self[self._current_index],)

    def into_dataset(self) -> tf.data.Dataset:
        """Return dataset generated out of the current sequence instance.

        Implementative details
        ---------------------------------
        This method handles the conversion of this Keras Sequence into
        a TensorFlow dataset, also handling the proper dispatching according
        to what version of TensorFlow is installed in this system.

        Returns
        ----------------------------------
        Dataset to be used for the training of a model
        """

        input_tensor_specs = []

        for _ in range(2):
            # Shapes of the source and destination node IDs
            input_tensor_specs.append(tf.TensorSpec(
                shape=(None, ),
                dtype=tf.int32
            ))

        if self._use_edge_metrics:
            # Shapes of the edge type IDs
            input_tensor_specs.append(tf.TensorSpec(
                shape=(None, self._graph.get_number_of_available_edge_metrics()),
                dtype=tf.float64
            ))

        # Shapes of the kernel
        input_tensor_specs.append(tf.SparseTensorSpec(
            shape=(None, None),
            dtype=tf.float32
        ))
        
        if self._node_features is not None:
            input_tensor_specs.extend([
                tf.TensorSpec(
                    shape=(None, node_feature.shape[1]),
                    dtype=tf.float32
                )
                for node_feature in self._node_features
            ])
            
        input_tensor_specs.extend([
            tf.TensorSpec(
                shape=(None, node_type_feature.shape[1]),
                dtype=tf.float32
            )
            for node_type_feature in self._node_type_features
        ])

        return tf.data.Dataset.from_generator(
            self,
            output_signature=(
                (
                    *input_tensor_specs,
                ),
                tf.TensorSpec(
                    shape=(None,),
                    dtype=tf.bool
                )
            )
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
        edge_features = None if self._edge_features_sequence is None else self._edge_features_sequence[idx] 
        # If necessary, we add the padding as the last batch may be
        # smaller than the required size (number of nodes).
        delta = self.batch_size - values[0].shape[0]
        if delta > 0:
            values = [
                np.pad(value, (0, delta) if len(value.shape) == 1 else [(0, delta), (0, 0)])
                for value in values
                if value is not None
            ]
            if edge_features is not None:
               edge_features = np.pad(edge_features, [(0, delta), (0, 0)])

        return (tuple([
            value
            for value in (
                *values,
                self._kernel,
                *self._node_features,
                *self._node_type_features,
                edge_features,
                self._node_ids,
                self._node_types,
            )
            if value is not None
        ]),)
