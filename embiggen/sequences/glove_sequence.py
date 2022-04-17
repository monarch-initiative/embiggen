"""Abstract Keras Sequence object for running models on huge datasets."""
from typing import Union, Tuple, Dict
from keras_mixed_sequence import MixedSequence, VectorSequence
import numpy as np
import tensorflow as tf
from ..utils import tensorflow_version_is_higher_or_equal_than


class GloveSequence(MixedSequence):
    """Abstract Keras Sequence object for running models on huge datasets."""

    def __init__(
        self,
        sources: np.ndarray,
        destinations: np.ndarray,
        frequencies: np.ndarray,
        batch_size: int,
        directed: bool = False,
        shuffle: bool = True,
        elapsed_epochs: int = 0,
        random_state: int = 42
    ):
        """Create new Sequence object.

        Parameters
        -----------------------------
        sources: np.ndarray,
            Source elements.
        destinations: np.ndarray,
            Destination elements.
        frequencies: np.ndarray,
            Frequencies to be predicted by GloVe model.
        batch_size: int,
            Number of nodes to include in a single batch.
        directed: bool = False,
            Whether to randomly provide left and right inputs simmetrically
            or not.
        shuffle: bool = True,
            Whether to shuffle the values at each epoch.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        random_state: int = 42,
            Random random_state to make the sequence reproducible.
        """
        self._directed = directed
        batches_per_epoch = max(
            sources.size // batch_size,
            1
        )
        self._current_index = 0
        self._batches_per_epoch = batches_per_epoch
        super().__init__(
            x=[
                VectorSequence(
                    sources,
                    batch_size=batch_size,
                    random_state=random_state,
                    elapsed_epochs=elapsed_epochs,
                    shuffle=shuffle
                ),
                VectorSequence(
                    destinations,
                    batch_size=batch_size,
                    random_state=random_state,
                    elapsed_epochs=elapsed_epochs,
                    shuffle=shuffle
                )
            ],
            y=VectorSequence(
                frequencies,
                batch_size=batch_size,
                random_state=random_state,
                elapsed_epochs=elapsed_epochs,
                shuffle=shuffle
            )
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
            return tf.data.Dataset.from_generator(
                self,
                output_signature=(
                    (
                        tf.TensorSpec(
                            shape=(None, ),
                            dtype=tf.int32
                        ),
                        tf.TensorSpec(
                            shape=(None, ),
                            dtype=tf.int32
                        )
                    ),
                    tf.TensorSpec(
                        shape=(None, ),
                        dtype=tf.float64
                    )
                )
            )

        return tf.data.Dataset.from_generator(
            self,
            output_types=(
                (
                    tf.int32,
                    tf.int32,
                ),
                tf.float64
            ),
            output_shapes=(
                (
                    tf.TensorShape([None, ]),
                    tf.TensorShape([None, ])
                ),
                tf.TensorShape([None, ])
            )
        )

    def __getitem__(self, idx: int) -> Tuple[
        Union[np.ndarray, Dict],
        Union[np.ndarray, Dict]
    ]:
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int
            Index corresponding to batch to be rendered.

        Returns
        ---------------
        Return Tuple containing input and output batches.
        """
        (sources, destinations), frequencies = super().__getitem__(idx)
        if self._directed or np.random.randint(2, dtype=bool):
            return (((sources, destinations), frequencies,),)
        return (((destinations, sources), frequencies,),)
