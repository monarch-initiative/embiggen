"""CBOW model for sequence embedding."""
from typing import Union, Tuple
from tensorflow.keras.optimizers import Optimizer   # pylint: disable=import-error
from tensorflow.keras.layers import Layer   # pylint: disable=import-error
from .word2vec import Word2Vec


class CBOW(Word2Vec):
    """CBOW model for sequence embedding.

    The CBOW model for graoh embedding receives a list of contexts and tries
    to predict the central word. The model makes use of an NCE loss layer
    during the training process to generate the negatives.
    """

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        model_name: str = "CBOW",
        optimizer: Union[str, Optimizer] = None,
        window_size: int = 16,
        negative_samples: int = 10
    ):
        """Create new CBOW-based Embedder object.

        Parameters
        -------------------------------------------
        vocabulary_size: int,
            Number of terms to embed.
            In a graph this is the number of nodes, while in a text is the
            number of the unique words.
        embedding_size: int,
            Dimension of the embedding.
        model_name: str = "CBOW",
            Name of the model.
        optimizer: Union[str, Optimizer] = None,
            The optimizer to be used during the training of the model.
            By default, if None is provided, Nadam with learning rate
            set at 0.01 is used.
        window_size: int = 16,
            Window size for the local context.
            On the borders the window size is trimmed.
        negative_samples: int = 10,
            The number of negative classes to randomly sample per batch.
            This single sample of negative classes is evaluated for each element in the batch.
        """
        super().__init__(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            model_name=model_name,
            optimizer=optimizer,
            window_size=window_size,
            negative_samples=negative_samples
        )

    def _get_true_input_length(self) -> int:
        """Return length of true input layer."""
        return self._window_size*2

    def _get_true_output_length(self) -> int:
        """Return length of true output layer."""
        return 1

    def _sort_input_layers(
        self,
        true_input_layer: Layer,
        true_output_layer: Layer
    ) -> Tuple[Layer]:
        """Return input layers for training with the same input sequence.

        Parameters
        ----------------------------
        true_input_layer: Layer,
            The input layer that will contain the true input.
        true_output_layer: Layer,
            The input layer that will contain the true output.

        Returns
        ----------------------------
        Return tuple with the tuple of layers.
        """
        return (
            true_input_layer,
            true_output_layer
        )
