"""Keras Sequence object for running CBOW and SkipGram on texts."""
from typing import Tuple, List
import tensorflow as tf
from ..utils import tensorflow_version_is_higher_or_equal_than
import numpy as np  # type: ignore
from ensmallen import preprocessing  # pylint: disable=no-name-in-module
from keras_mixed_sequence import VectorSequence

from .abstract_sequence import AbstractSequence


class Word2VecSequence(AbstractSequence):
    """Keras Sequence object for running CBOW and SkipGram on texts."""

    def __init__(
        self,
        sequences: List[np.ndarray],
        batch_size: int,
        window_size: int = 4,
        random_state: int = 42,
        elapsed_epochs: int = 0,
    ):
        """Create new Node2Vec Sequence object.

        Parameters
        -----------------------------
        sequences: List[np.ndarray]
            List of sequences of integers.
        batch_size: int
            Number of nodes to include in a single batch.
        window_size: int = 4
            Window size for the local context.
            On the borders the window size is trimmed.
        random_state: int = 42
            The random_state to use to make extraction reproducible.
        elapsed_epochs: int = 0
            Number of elapsed epochs to init state of generator.
        """

        self._sequences = VectorSequence(
            sequences,
            batch_size,
            random_state=random_state,
            elapsed_epochs=elapsed_epochs
        )
        self._current_index = 0
        super().__init__(
            window_size=window_size,
            sample_number=self._sequences.sample_number,
            batch_size=batch_size,
            elapsed_epochs=elapsed_epochs,
            random_state=random_state
        )

    def on_epoch_end(self):
        """Shuffles given sequences object."""
        super().on_epoch_end()
        self._sequences.on_epoch_end()

    def __call__(self):
        """Return next batch using an infinite generator model."""
        self._current_index = self._current_index % self._sequences.steps_per_epoch
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
            return tf.data.Dataset.from_generator(
                self,
                output_signature=(
                    (
                        tf.TensorSpec(
                            shape=(None, self._window_size*2),
                            dtype=tf.int32
                        ),
                        tf.TensorSpec(
                            shape=(None, ),
                            dtype=tf.int32
                        )
                    ),
                )
            )

        return tf.data.Dataset.from_generator(
            self,
            output_types=(
                (
                    tf.int32,
                    tf.int32
                ),
            ),
            output_shapes=(
                (
                    tf.TensorShape([None, self._window_size*2]),
                    tf.TensorShape([None, ])
                ),
            )
        )

    def __getitem__(self, idx: int) -> Tuple[Tuple[np.ndarray, np.ndarray], None]:
        """Return batch corresponding to given index.

        The return tuple of tuples is composed of an inner tuple, containing
        the words vector and the vector of vectors of the contexts.
        Depending on the order of the input_layers of the models that can
        accept these data format, one of the vectors is used as training
        input and the other one is used as the output for the NCE loss layer.

        The words vectors and contexts vectors contain numeric IDs, that
        represent the index of the words' embedding column.

        The true output value is None, since no loss function is used after
        the NCE loss, that is implemented as a layer, and this vastly improves
        the speed of the training process since it does not require to allocate
        empty vectors of considerable size for the one-hot encoding process.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be returned.

        Returns
        ---------------
        Tuple of tuples with input data.
        """
        contexts, words = preprocessing.word2vec(
            self._sequences[idx],
            window_size=self._window_size,
        )

        return (((contexts, words),),)
