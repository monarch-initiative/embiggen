"""Submodule providing flat embedding layer."""
from typing import Dict
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling1D, Layer, Embedding  # pylint: disable=import-error,no-name-in-module


class FlatEmbedding(Layer):
    """Layer implementing simple wrapped embedding layer plus a flatten."""

    def __init__(
        self,
        vocabulary_size: int,
        dimension: int,
        input_length: int,
        mask_zero: bool = False,
        **kwargs: Dict
    ):
        """Create new FlatEmbedding layer.

        Parameters
        ----------------------
        vocabulary_size: int
            The number of elements in the embedding.
        dimension: int
            The dimensionality of the embedding.
        input_length: int
            The expected input length of the embedding.
        mask_zero: bool = False
            Whether to treat zero inputs as if they are a mask.
        **kwargs: Dict,
            Kwargs to pass to the parent Layer class.
        """
        super().__init__(**kwargs)
        self._flatten_layer = None
        self._embedding_layer = None
        self._vocabulary_size = vocabulary_size
        self._dimension = dimension
        self._input_length = input_length
        self._mask_zero = mask_zero

    def build(self, input_shape) -> None:
        """Build the Graph Convolution layer.

        Parameters
        ------------------------------
        input_shape
            Shape of the output of the previous layer.
        """
        self._flatten_layer = GlobalAveragePooling1D()
        self._embedding_layer = Embedding(
            input_dim=self._vocabulary_size,
            output_dim=self._dimension,
            input_length=self._input_length,
            mask_zero=self._mask_zero
        )
        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor
    ) -> tf.Tensor:
        """Returns called flattened embedding.

        Parameters
        ---------------------------
        inputs: tf.Tensor
        """
        return self._flatten_layer(
            self._embedding_layer(inputs)
        )