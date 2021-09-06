"""Abstract Keras Sequence object for running models on huge datasets."""
from typing import Union, Tuple, Dict
from keras_mixed_sequence import MixedSequence, VectorSequence
import numpy as np


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

    def __getitem__(self, idx: int) -> Tuple[
        Union[np.ndarray, Dict],
        Union[np.ndarray, Dict]
    ]:
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be rendered.

        Returns
        ---------------
        Return Tuple containing input and output batches.
        """
        input_values, output_values = super().__getitem__(idx)
        if self._directed or np.random.randint(2, dtype=bool):
            return input_values, output_values
        return (input_values[1], input_values[0]), output_values
