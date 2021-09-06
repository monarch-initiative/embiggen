"""Keras Sequence object for running CBOW and SkipGram on texts."""
from typing import Tuple

import numpy as np  # type: ignore
from ensmallen import preprocessing  # pylint: disable=no-name-in-module

from .abstract_word2vec_sequence import AbstractWord2VecSequence


class Word2VecSequence(AbstractWord2VecSequence):
    """Keras Sequence object for running CBOW and SkipGram on texts."""

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

        if self._support_mirrored_strategy:
            return (contexts.astype(float), words.astype(float)), None
        return (contexts, words), None
