"""Keras Sequence for Open-world assumption GCN."""
from typing import Tuple, List, Optional

import numpy as np
import tensorflow as tf
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence


class GCNEdgePredictionTrainingSequence(Sequence):
    """Keras Sequence for running Neural Network on graph edge prediction."""

    def __init__(
        self,
        graph: Graph,
        kernel: tf.SparseTensor,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        return_node_ids: bool = False,
        return_node_types: bool = False,
        use_edge_metrics: bool = False,
        negative_samples_rate: float = 0.5,
        avoid_false_negatives: bool = False,
        graph_to_avoid: Graph = None,
        sample_only_edges_with_heterogeneous_node_types: bool = False,
        random_state: int = 42
    ):
        """Create new Open-world assumption GCN training sequence for edge prediction.

        Parameters
        --------------------------------
        graph: Graph,
            The graph from which to sample the edges.
        kernel: tf.SparseTensor
            The kernel to be used for the convolutions.
        support: Optional[Graph] = None
            The graph to use to compute the edge metrics.
        node_features: Optonal[List[np.ndarray]] = None
            The node features to be used.
        node_type_features: Optional[List[np.ndarray]]
            The node type features to be used.
            For instance, these could be BERT embeddings of the
            description of the node types.
            When the graph has multilabel node types,
            we will average the features.
        return_node_ids: bool = False
            Whether to return the node IDs.
            These are needed when a node embedding layer is used.
        return_node_types: bool = False,
            Whether to return the node type IDs.
            These are needed when a node type embedding layer is used.
        use_edge_metrics: bool = False
            Whether to return the edge metrics.
        negative_samples_rate: float = 0.5
            Factor of negatives to use in every batch.
            For example, with a batch size of 128 and negative_samples_rate equal
            to 0.5, there will be 64 positives and 64 negatives.
        avoid_false_negatives: bool = False,
            Whether to filter out false negatives.
            By default False.
            Enabling this will slow down the batch generation while (likely) not
            introducing any significant gain to the model performance.
        graph_to_avoid: Graph = None,
            Graph to avoid when generating the edges.
            This can be the validation component of the graph, for example.
            More information to how to generate the holdouts is available
            in the Graph package.
        sample_only_edges_with_heterogeneous_node_types: bool = False
            Whether to only sample edges between heterogeneous node types.
            This may be useful when training a model to predict between
            two portions in a bipartite graph.
        random_state: int = 42,
            The random_state to use to make extraction reproducible.
        """
        if not graph.has_edges():
            raise ValueError(
                f"An empty instance of graph {graph.get_name()} was provided!"
            )

        if support is None:
            support = graph

        self._graph = graph
        self._support = support
        self._kernel = kernel

        if node_features is None:
            node_features = []
        self._node_features = node_features
        self._negative_samples_rate = negative_samples_rate
        self._avoid_false_negatives = avoid_false_negatives
        self._graph_to_avoid = graph_to_avoid
        self._random_state = random_state
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
            if graph.has_multilabel_node_types():
                node_types = graph.get_one_hot_encoded_node_types()
            else:
                node_types = graph.get_single_label_node_type_ids()
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

        self._use_edge_metrics = use_edge_metrics
        self._sample_only_edges_with_heterogeneous_node_types = sample_only_edges_with_heterogeneous_node_types
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
        sources, _, destinations, _, edge_metrics, labels = self._graph.get_edge_prediction_mini_batch(
            (self._random_state + idx) * (1 + self.elapsed_epochs),
            return_node_types=False,
            return_edge_metrics=self._use_edge_metrics,
            batch_size=self.batch_size,
            negative_samples_rate=self._negative_samples_rate,
            avoid_false_negatives=self._avoid_false_negatives,
            sample_only_edges_with_heterogeneous_node_types=self._sample_only_edges_with_heterogeneous_node_types,
            support=self._support,
            graph_to_avoid=self._graph_to_avoid,
        )

        return (tuple([
            value
            for value in (
                sources,
                destinations,
                edge_metrics,
                self._kernel,
                *self._node_features,
                *self._node_type_features,
                self._node_ids,
                self._node_types
            )
            if value is not None
        ]), labels)
