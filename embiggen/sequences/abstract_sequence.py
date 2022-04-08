"""Abstract Keras Sequence object for running models on huge datasets."""
from keras_mixed_sequence import Sequence


class AbstractSequence(Sequence):
    """Abstract Keras Sequence object for running models on huge datasets."""

    def __init__(
        self,
        batch_size: int,
        sample_number: int,
        window_size: int = 4,
        elapsed_epochs: int = 0,
        random_state: int = 42
    ):
        """Create new Sequence object.

        Parameters
        -----------------------------
        batch_size: int,
            Number of nodes to include in a single batch.
        sample_number: int,
            Number of samples that compose this Sequence.
        window_size: int = 4,
            Window size for the local context.
            On the borders the window size is trimmed.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        random_state: int = 42,
            Random random_state to make the sequence reproducible.
        """
        self._window_size = window_size
        self._random_state = random_state

        super().__init__(
            sample_number=sample_number,
            batch_size=batch_size,
            elapsed_epochs=elapsed_epochs
        )
