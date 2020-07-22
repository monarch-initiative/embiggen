"""Abstract Keras Sequence object for running models on huge datasets."""
from keras_mixed_sequence import Sequence


class AbstractSequence(Sequence):
    """Abstract Keras Sequence object for running models on huge datasets."""

    def __init__(
        self,
        batch_size: int,
        samples_number: int,
        window_size: int = 4,
        shuffle: bool = True,
        elapsed_epochs: int = 0,
    ):
        """Create new Sequence object.

        Parameters
        -----------------------------
        batch_size: int,
            Number of nodes to include in a single batch.
        samples_number: int,
            Number of samples that compose this Sequence.
        window_size: int = 4,
            Window size for the local context.
            On the borders the window size is trimmed.
        shuffle: bool = True,
            Wthever to shuffle the vectors.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        """
        self._window_size = window_size
        self._shuffle = shuffle

        super().__init__(
            samples_number=samples_number,
            batch_size=batch_size,
            elapsed_epochs=elapsed_epochs
        )
