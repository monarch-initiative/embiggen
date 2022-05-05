"""Keras Sequence for running Neural Network on graph edge prediction."""
from typing import Tuple, Union, Optional
import warnings

import numpy as np
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence
import tensorflow as tf
from ..utils import tensorflow_version_is_higher_or_equal_than


class EdgePredictionEvaluationSequence(Sequence):
    """Keras Sequence for running Neural Network on graph edge prediction."""

    def __init__(
        self,
        positive_graph: Graph,
        negative_graph: Optional[Graph] = None,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        use_edge_metrics: bool = False,
        batch_size: int = 2**10,
        filter_none_values: bool = True,
        elapsed_epochs: int = 0,
        random_state: int = 42
    ):
        """Create new EdgePredictionSequence object.

        Parameters
        --------------------------------
        positive_graph: Graph
            The graph from which to sample the positive edges.
        negative_graph: Optional[Graph] = None
            The graph from which to sample the negative edges.
        use_node_types: bool = False
            Whether to return the node types.
        use_edge_types: bool = False
            Whether to return the edge types.
        use_edge_metrics: bool = False
            Whether to return the edge metrics.
        batch_size: int = 2**10
            The batch size to use.
        filter_none_values: bool = True
            Whether to filter None values.
        elapsed_epochs: int = 0
            Number of elapsed epochs to init state of generator.
        random_state: int = 42
            The random_state to use to make extraction reproducible.

        Raises
        --------------------------
        ValueError
            If the two graphs do not share the same vocabulary and therefore
            are not compatible with each other (or the model, likely).
        ValueError
            If the positive and negative graphs overlap.
        ValueError
            If either of the provided graphs is empty.
        """
        if negative_graph is not None:
            if not positive_graph.has_compatible_node_vocabularies(negative_graph):
                raise ValueError(
                    "The two provided positive and negative graphs are "
                    "not compatible with each other, for instance they may not "
                    "share the same dictionary of nodes."
                )
            if positive_graph.overlaps(negative_graph):
                raise ValueError(
                    "The provided positive and negative graphs overlap "
                    "with each other, making the definition of true negative "
                    "and true positive within this evaluation ambigous."
                )
            if not negative_graph.has_edges():
                raise ValueError(
                    "The provided negative graph is empty, that is, it has no edges."
                )
            if positive_graph.has_selfloops() != negative_graph.has_selfloops():
                warnings.warn(
                    "Please be advides that in either the provided positive "
                    "or negative graphs there are present selfloops. "
                    "If such odd topologies were also exclusively "
                    "present during the training of this model in only either the positive "
                    "or negative edges, the model might learn that self-loops are always "
                    "positive and/or negative. This might be okay or not according "
                    "to the experimental setup you have designed."
                )

        if not positive_graph.has_edges():
            raise ValueError(
                "The provided positive graph is empty, that is, it has no edges."
            )

        self._positive_graph = positive_graph
        self._negative_graph = negative_graph
        self._random_state = random_state
        self._use_node_types = use_node_types
        self._use_edge_types = use_edge_types
        self._filter_none_values = filter_none_values
        self._use_edge_metrics = use_edge_metrics
        edges_number = positive_graph.get_number_of_directed_edges()
        if negative_graph is not None:
            edges_number += negative_graph.get_number_of_directed_edges()
        batches_per_epoch = max(
            edges_number // batch_size,
            1
        )
        self._current_index = 0
        self._batches_per_epoch = batches_per_epoch
        super().__init__(
            sample_number=edges_number,
            batch_size=batch_size,
            elapsed_epochs=elapsed_epochs
        )

    def __call__(self):
        """Return next batch using an infinite generator model."""
        self._current_index = (self._current_index + 1) % self._batches_per_epoch
        return self[self._current_index]

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

        #################################################################
        # Handling kernel creation when TensorFlow is a modern version. #
        #################################################################

        if tensorflow_version_is_higher_or_equal_than("2.5.0"):
            input_tensor_specs = []

            for _ in range(2):
                # Shapes of the source and destination node IDs
                input_tensor_specs.append(tf.TensorSpec(
                    shape=(None, ), dtype=tf.int32))
                if self._use_node_types:
                    # Shapes of the source and destination node type IDs
                    input_tensor_specs.append(tf.TensorSpec(
                        shape=(None, self._graph.get_maximum_multilabel_count()), dtype=tf.int32))

            if self._use_edge_metrics:
                # Shapes of the edge type IDs
                input_tensor_specs.append(tf.TensorSpec(shape=(
                    None, self._graph.get_number_of_available_edge_metrics()), dtype=tf.float64))

            if self._use_edge_types:
                # Shapes of the edge type IDs
                input_tensor_specs.append(tf.TensorSpec(
                    shape=(None, ), dtype=tf.int32))

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

        input_tensor_types = []
        input_tensor_shapes = []

        for _ in range(2):
            input_tensor_types.append(tf.int32,)
            input_tensor_shapes.append(tf.TensorShape([None, ]),)

            if self._use_node_types:
                input_tensor_types.append(tf.int32,)
                input_tensor_shapes.append(
                    tf.TensorShape([None, self._graph.get_maximum_multilabel_count()]),)

        if self._use_edge_metrics:
            input_tensor_types.append(tf.float64,)
            input_tensor_shapes.append(tf.TensorShape(
                [None, self._graph.get_number_of_available_edge_metrics()]),)

        if self._use_edge_types:
            input_tensor_types.append(tf.int32,)
            input_tensor_shapes.append(tf.TensorShape([None, ]),)

        return tf.data.Dataset.from_generator(
            self,
            output_types=(
                (
                    *input_tensor_types,
                ),
                tf.bool
            ),
            output_shapes=(
                (
                    *input_tensor_shapes,
                ),
                tf.TensorShape([None, ]),
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
        sources, source_node_types, destinations, destination_node_types, edge_metrics, edge_types = self._positive_graph.get_edge_prediction_chunk_mini_batch(
            idx,
            return_node_types=self._use_node_types,
            return_edge_types=False,
            return_edge_metrics=self._use_edge_metrics,
            batch_size=self.batch_size//2,
        )
        labels = np.ones_like(sources, dtype=bool)

        if self._negative_graph is not None:
            negative_sources, negative_source_node_types, negative_destinations, negative_destination_node_types, negative_edge_metrics, negative_edge_types = self._negative_graph.get_edge_prediction_chunk_mini_batch(
                idx,
                return_node_types=self._use_node_types,
                return_edge_types=False,
                return_edge_metrics=self._use_edge_metrics,
                batch_size=self.batch_size//2,
            )
            negative_labels = np.zeros_like(negative_sources, dtype=bool)

            sources = np.hstack([
                sources,
                negative_sources
            ])
            if source_node_types is not None:
                source_node_types = np.hstack([
                    source_node_types,
                    negative_source_node_types
                ])
            destinations = np.hstack([
                destinations,
                negative_destinations
            ])
            if destination_node_types is not None:
                destination_node_types = np.hstack([
                    destination_node_types,
                    negative_destination_node_types
                ])
            if edge_metrics is not None:
                edge_metrics = np.hstack([
                    edge_metrics,
                    negative_edge_metrics
                ])
            if edge_types is not None:
                edge_types = np.hstack([
                    edge_types,
                    negative_edge_types
                ])
            labels = np.hstack([
                labels,
                negative_labels
            ])

        return ((tuple([
            value
            for value in (
                sources, source_node_types, destinations, destination_node_types, edge_metrics, edge_types,
            )
            if not self._filter_none_values or value is not None
        ]), labels,), )
