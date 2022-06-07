"""Keras Sequence for running Neural Network on graph edge prediction."""
from typing import Tuple

import numpy as np
import tensorflow as tf
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence
from embiggen.utils.tensorflow_utils import tensorflow_version_is_higher_or_equal_than


class EdgePredictionTrainingSequence(Sequence):
    """Keras Sequence for running Neural Network on graph edge prediction."""

    def __init__(
        self,
        graph: Graph,
        use_node_types: bool = False,
        use_edge_metrics: bool = False,
        batch_size: int = 2**10,
        negative_samples_rate: float = 0.5,
        avoid_false_negatives: bool = False,
        graph_to_avoid: Graph = None,
        sample_only_edges_with_heterogeneous_node_types: bool = False,
        random_state: int = 42
    ):
        """Create new EdgePredictionSequence object.

        Parameters
        --------------------------------
        graph: Graph,
            The graph from which to sample the edges.
        use_node_types: bool = False,
            Whether to return the node types.
        use_edge_metrics: bool = False,
            Whether to return the edge metrics.
        batch_size: int = 2**10,
            The batch size to use.
        negative_samples_rate: float = 0.5,
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
        self._graph = graph
        self._negative_samples_rate = negative_samples_rate
        self._avoid_false_negatives = avoid_false_negatives
        self._graph_to_avoid = graph_to_avoid
        self._random_state = random_state
        self._use_node_types = use_node_types
        self._use_edge_metrics = use_edge_metrics
        self._sample_only_edges_with_heterogeneous_node_types = sample_only_edges_with_heterogeneous_node_types
        self._current_index = 0
        super().__init__(
            sample_number=graph.get_number_of_directed_edges(),
            batch_size=batch_size,
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

        #################################################################
        # Handling kernel creation when TensorFlow is a modern version. #
        #################################################################

        if tensorflow_version_is_higher_or_equal_than("2.5.0"):
            input_tensor_specs = []

            for _ in range(2):
                # Shapes of the source and destination node IDs
                input_tensor_specs.append(tf.TensorSpec(
                    shape=(self._batch_size, ),
                    dtype=tf.int32
                ))

                if self._use_node_types:
                    # Shapes of the source and destination node type IDs
                    input_tensor_specs.append(tf.TensorSpec(
                        shape=(self._batch_size,
                               self._graph.get_maximum_multilabel_count()),
                        dtype=tf.int32
                    ))

            if self._use_edge_metrics:
                # Shapes of the edge type IDs
                input_tensor_specs.append(tf.TensorSpec(
                    shape=(self._batch_size,
                           self._graph.get_number_of_available_edge_metrics()),
                    dtype=tf.float64
                ))

            return tf.data.Dataset.from_generator(
                self,
                output_signature=(
                    (
                        *input_tensor_specs,
                    ),
                    tf.TensorSpec(
                        shape=(self._batch_size,),
                        dtype=tf.bool
                    )
                )
            )

        input_tensor_types = []
        input_tensor_shapes = []

        for _ in range(2):
            input_tensor_types.append(tf.int32,)
            input_tensor_shapes.append(tf.TensorShape([self._batch_size, ]),)

            if self._use_node_types:
                input_tensor_types.append(tf.int32,)
                input_tensor_shapes.append(
                    tf.TensorShape([self._batch_size, self._graph.get_maximum_multilabel_count()]),)

        if self._use_edge_metrics:
            input_tensor_types.append(tf.float64,)
            input_tensor_shapes.append(tf.TensorShape(
                [self._batch_size, self._graph.get_number_of_available_edge_metrics()]),)

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
                tf.TensorShape([self._batch_size, ]),
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
        sources, source_node_types, destinations, destination_node_types, edge_metrics, labels = self._graph.get_edge_prediction_mini_batch(
            (self._random_state + idx) * (1 + self.elapsed_epochs),
            return_node_types=self._use_node_types,
            return_edge_metrics=self._use_edge_metrics,
            batch_size=self.batch_size,
            negative_samples_rate=self._negative_samples_rate,
            avoid_false_negatives=self._avoid_false_negatives,
            sample_only_edges_with_heterogeneous_node_types=self._sample_only_edges_with_heterogeneous_node_types,
            graph_to_avoid=self._graph_to_avoid,
        )

        return (tuple([
            value
            for value in (
                sources, source_node_types,
                destinations, destination_node_types,
                edge_metrics,
            )
            if value is not None
        ]), labels)
