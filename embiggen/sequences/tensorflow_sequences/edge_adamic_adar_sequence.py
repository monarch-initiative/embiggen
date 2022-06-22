"""Keras Sequence for running Neural Network on graph edge Adamic-Adar prediction."""
from typing import Tuple

import numpy as np
import tensorflow as tf
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence
from embiggen.utils.tensorflow_utils import tensorflow_version_is_higher_or_equal_than


class EdgeAdamicAdarSequence(Sequence):
    """Keras Sequence for running Neural Network on graph edge Adamic-Adar prediction."""

    def __init__(
        self,
        graph: Graph,
        batch_size: int = 2**10,
        negative_samples_rate: float = 0.5,
        support: Graph = None,
        random_state: int = 42
    ):
        """Create new EdgePredictionSequence object.

        Parameters
        --------------------------------
        graph: Graph,
            The graph from which to sample the edges.
        batch_size: int = 2**10,
            The batch size to use.
        negative_samples_rate: float = 0.5,
            Factor of negatives to use in every batch.
            For example, with a batch size of 128 and negative_samples_rate equal
            to 0.5, there will be 64 positives and 64 negatives.
        support: Graph = None,
            The graph to be used for the topological metrics.
        random_state: int = 42,
            The random_state to use to make extraction reproducible.
        """
        if not graph.has_edges():
            raise ValueError(
                f"An empty instance of graph {graph.get_name()} was provided!"
            )
        self._graph = graph
        self._negative_samples_rate = negative_samples_rate
        self._random_state = random_state
        self._support = support
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

            return tf.data.Dataset.from_generator(
                self,
                output_signature=(
                    (
                        *input_tensor_specs,
                    ),
                    tf.TensorSpec(
                        shape=(self._batch_size,),
                        dtype=tf.float32
                    )
                )
            )

        input_tensor_types = []
        input_tensor_shapes = []

        for _ in range(2):
            input_tensor_types.append(tf.int32,)
            input_tensor_shapes.append(tf.TensorShape([self._batch_size, ]),)

        return tf.data.Dataset.from_generator(
            self,
            output_types=(
                (
                    *input_tensor_types,
                ),
                tf.float32
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
        sources, destinations, scores = self._graph.get_edge_adamic_adar_mini_batch(
            (self._random_state + idx) * (1 + self.elapsed_epochs),
            batch_size=self.batch_size,
            negative_samples_rate=self._negative_samples_rate,
            support=self._support
        )

        return (sources, destinations), scores
