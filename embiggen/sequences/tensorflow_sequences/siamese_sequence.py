"""Keras Sequence for running Neural Network on graph edge prediction."""
from typing import List

import numpy as np
import tensorflow as tf
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence
from embiggen.utils.tensorflow_utils import tensorflow_version_is_higher_or_equal_than


class SiameseSequence(Sequence):
    """Keras Sequence for running Siamese Neural Network."""

    def __init__(
        self,
        graph: Graph,
        batch_size: int = 2**10,
        random_state: int = 42
    ):
        """Create new EdgePredictionSequence object.

        Parameters
        --------------------------------
        graph: Graph,
            The graph from which to sample the triples.
        batch_size: int = 2**10,
            The batch size to use.
        random_state: int = 42,
            The random_state to use to make extraction reproducible.
        """
        self._graph = graph
        self._random_state = random_state
        self._current_index = 0
        super().__init__(
            sample_number=self._graph.get_number_of_directed_edges(),
            batch_size=batch_size,
        )

    def __call__(self):
        """Return next batch using an infinite generator model."""
        self._current_index += 1
        return (tuple(self[self._current_index]),)

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

            # For both the real and fake nodes.
            for _ in range(4):
                # Shapes of the source and destination node IDs
                input_tensor_specs.append(tf.TensorSpec(
                    shape=(self._batch_size, ),
                    dtype=tf.uint32
                ))

            # Shapes of the edge type IDs
            input_tensor_specs.append(tf.TensorSpec(
                shape=(self._batch_size,),
                dtype=tf.uint32
            ))

            return tf.data.Dataset.from_generator(
                self,
                output_signature=(tuple(input_tensor_specs),)
            )

        input_tensor_types = []
        input_tensor_shapes = []

        for _ in range(4):
            input_tensor_types.append(tf.uint32,)
            input_tensor_shapes.append(tf.TensorShape([self._batch_size, ]),)

        input_tensor_types.append(tf.uint32,)
        input_tensor_shapes.append(tf.TensorShape([self._batch_size, ]),)

        return tf.data.Dataset.from_generator(
            self,
            output_types=input_tensor_types,
            output_shapes=input_tensor_shapes
        )

    def __getitem__(self, idx: int) -> List[np.ndarray]:
        """Return batch corresponding to given index to train a Siamese network.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be returned.
        """
        random_state = (self._random_state + idx) * (1 + self.elapsed_epochs)
        return (self._graph.get_siamese_mini_batch(
            random_state,
            batch_size=self.batch_size,
            use_zipfian_sampling=True
        ),)
