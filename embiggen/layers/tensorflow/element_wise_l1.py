"""Submodule providing element-wise L1 distance layer."""
from typing import Tuple, Dict
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Subtract, Lambda, Layer  # pylint: disable=import-error,no-name-in-module


class ElementWiseL1(Layer):
    """Layer implementing element-wise L1 distance layer."""

    def __init__(
        self,
        **kwargs: Dict
    ):
        """Create new element-wise L1 distance layer.

        Parameters
        ----------------------
        **kwargs: Dict,
            Kwargs to pass to the parent Layer class.
        """
        super().__init__(**kwargs)
        self._subtraction = None
        self._absolute_value = None

    def build(self, input_shape) -> None:
        """Build the element-wise L1 distance layer.

        Parameters
        ------------------------------
        input_shape
            Shape of the output of the previous layer.
        """
        self._subtraction = Subtract()
        self._absolute_value = Lambda(K.abs)
        super().build(input_shape)

    def call(
        self,
        inputs: Tuple[tf.Tensor],
    ) -> tf.Tensor:
        return self._absolute_value(self._subtraction(inputs))
