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
        support_mirrored_strategy: bool = False,
        random_state: int = 42,
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
        support_mirrored_strategy: bool = False,
            Wethever to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        random_state: int = 42,
            The random_state to use to make extraction reproducible.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        """

        self._sequences = VectorSequence(
            sequences,
            batch_size,
            random_state=random_state,
            elapsed_epochs=elapsed_epochs
        )
        super().__init__(
            window_size=window_size,
            sample_number=self._sequences.sample_number,
            batch_size=batch_size,
            elapsed_epochs=elapsed_epochs,
            support_mirrored_strategy=support_mirrored_strategy,
            random_state=random_state
        )

    def on_epoch_end(self):
        """Shuffles given sequences object."""
        super().on_epoch_end()
        self._sequences.on_epoch_end()
