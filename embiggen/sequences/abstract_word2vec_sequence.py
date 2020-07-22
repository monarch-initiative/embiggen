"""Abstract Keras Sequence object for running models on graph walks."""
from typing import List

import numpy as np  # type: ignore
from keras_mixed_sequence import VectorSequence

from .abstract_sequence import AbstractSequence


class AbstractWord2VecSequence(AbstractSequence):
    """Abstract Keras Sequence object for running models on texts."""

    def __init__(
        self,
        sequences: List[np.ndarray],
        batch_size: int,
        window_size: int = 4,
        shuffle: bool = True,
        seed: int = 42,
        elapsed_epochs: int = 0,
    ):
        """Create new Node2Vec Sequence object.

        Parameters
        -----------------------------
        sequences: List[np.ndarray],
            List of sequences of integers.
        batch_size: int,
            Number of nodes to include in a single batch.
        window_size: int = 4,
            Window size for the local context.
            On the borders the window size is trimmed.
        shuffle: bool = True,
            Wthever to shuffle the vectors.
        seed: int = 42,
            The seed to use to make extraction reproducible.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        """

        self._sequences = VectorSequence(
            sequences,
            batch_size,
            seed=seed,
            elapsed_epochs=elapsed_epochs
        )
        super().__init__(
            window_size=window_size,
            shuffle=shuffle,
            samples_number=self._sequences.samples_number,
            batch_size=batch_size,
            elapsed_epochs=elapsed_epochs
        )

    def on_epoch_end(self):
        """Shuffles given sequences object."""
        super().on_epoch_end()
        self._sequences.on_epoch_end()
