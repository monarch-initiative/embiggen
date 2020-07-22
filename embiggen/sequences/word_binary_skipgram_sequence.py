"""Keras Sequence object for running BinarySkipGramSequence on texts."""
from typing import Tuple

import numpy as np  # type: ignore
from ensmallen_graph import preprocessing  # pylint: disable=no-name-in-module

from .abstract_word2vec_sequence import AbstractWord2VecSequence


class WordBinarySkipGramSequence(AbstractWord2VecSequence):
    """Keras Sequence object for running BinarySkipGramSequence on texts."""

    def __init__(
        self,
        sequences: np.ndarray,
        batch_size: int,
        vocabulary_size: int,
        negative_samples: float = 7,
        window_size: int = 4,
        shuffle: bool = True,
        seed: int = 42,
        elapsed_epochs: int = 0,
    ):
        """Create new Sequence object.

        Parameters
        -----------------------------
        sequences: np.ndarray,
            List of encoded texts.
        batch_size: int,
            Number of nodes to include in a single batch.
        negative_samples: float = 7,
            Factor of negative samples to use.
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
        self._negative_samples = negative_samples
        self._vocabulary_size = vocabulary_size

        super().__init__(
            sequences=sequences,
            batch_size=batch_size,
            window_size=window_size,
            shuffle=shuffle,
            seed=seed,
            elapsed_epochs=elapsed_epochs
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
            Index corresponding to batch to be rendered.

        Returns
        ---------------
        Tuple of tuples with input data.
        """
        return preprocessing.binary_skipgrams(
            idx+self.elapsed_epochs,
            self._sequences[idx],
            vocabulary_size=self._vocabulary_size,
            window_size=self._window_size,
            negative_samples=self._negative_samples,
            shuffle=self._shuffle
        )
