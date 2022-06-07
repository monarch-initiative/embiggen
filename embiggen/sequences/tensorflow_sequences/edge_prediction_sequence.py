"""Keras Sequence for running Neural Network on graph edge prediction."""
from typing import Tuple

import numpy as np
import tensorflow as tf
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence
from embiggen.sequences.generic_sequences import EdgePredictionSequence as GenericEdgePredictionSequence
from embiggen.utils.tensorflow_utils import tensorflow_version_is_higher_or_equal_than


class EdgePredictionSequence(Sequence):
    """Keras Sequence for running Neural Network on graph edge prediction."""

    def __init__(
        self,
        graph: Graph,
        graph_used_in_training: Graph,
        use_node_types: bool,
        use_edge_metrics: bool,
        batch_size: int = 2**10,
    ):
        """Create new EdgePredictionSequence object.

        Parameters
        --------------------------------
        graph: Graph
            The graph whose edges are to be predicted.
        graph_used_in_training: Graph
            The graph that was used while training the current
            edge prediction model.
        use_node_types: bool
            Whether to return the node types.
        use_edge_metrics: bool = True
            Whether to return the edge metrics.
        batch_size: int = 2**10,
            The batch size to use.
        """
        self._sequence = GenericEdgePredictionSequence(
            graph=graph,
            graph_used_in_training=graph_used_in_training,
            use_node_types=use_node_types,
            use_edge_metrics=use_edge_metrics,
            batch_size=batch_size
        )
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
                    shape=(None, ),
                    dtype=tf.int32
                ))

                if self._use_node_types:
                    # Shapes of the source and destination node type IDs
                    input_tensor_specs.append(tf.TensorSpec(
                        shape=(None,
                               self._graph.get_maximum_multilabel_count()),
                        dtype=tf.int32
                    ))

            if self._use_edge_metrics:
                # Shapes of the edge type IDs
                input_tensor_specs.append(tf.TensorSpec(
                    shape=(None,
                           self._graph.get_number_of_available_edge_metrics()),
                    dtype=tf.float32
                ))

            return tf.data.Dataset.from_generator(
                self,
                output_signature=(
                    tuple(input_tensor_specs),
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
            input_tensor_types.append(tf.float32,)
            input_tensor_shapes.append(tf.TensorShape(
                [None, self._graph.get_number_of_available_edge_metrics()]),)

        return tf.data.Dataset.from_generator(
            self,
            output_types=(
                tuple(input_tensor_types),
            ),
            output_shapes=(
                tuple(input_tensor_shapes),
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
        return self._sequence[idx]