"""Submodule providing L2 Norm layer."""
from typing import Dict
import tensorflow as tf
from tensorflow.keras.backend import l2_normalize
from tensorflow.keras.layers import Layer, Lambda  # pylint: disable=import-error,no-name-in-module


class L2Norm(Layer):
    """Layer implementing L2 Norm."""

    def __init__(
        self,
        **kwargs: Dict
    ):
        """Create new L2 Norm layer."""
        super().__init__(**kwargs)
        self._norm_layer = None

    def build(self, input_shape) -> None:
        """Build the L2 Norm layer.

        Parameters
        ------------------------------
        input_shape
            Shape of the output of the previous layer.
        """
        self._norm_layer = Lambda(lambda tensor: l2_normalize(tensor, axis=-1))
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
        return self._norm_layer(inputs)
