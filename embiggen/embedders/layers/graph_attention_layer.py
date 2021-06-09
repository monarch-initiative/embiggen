"""Submodule providing graph convolutional layer.

# References
The layer is implemented as described in [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).
"""
import imp
from typing import Tuple, Union, Dict
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Layer, Dense, Attention
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.constraints import Constraint


class GraphAttention(Layer):
    """Layer implementing graph convolution layer."""

    def __init__(
        self,
        **kwargs: Dict
    ):
        """Create new GraphConvolution layer.

        This layer computes a graph convolution similar to the
        one computed in a normal graph convolution, but instead
        of weighting the incidence matrix using the simmetrically
        normalized laplacian, it uses an attention layer.

        Parameters
        ----------------------
        **kwargs: Dict,
            Kwargs to pass to the parent Layer class.
        """
        super().__init__(**kwargs)

    def build(self, input_shape: Tuple[int, int]) -> None:
        """Build the NCE layer.

        Parameters
        ------------------------------
        input_shape: Tuple[int, int],
            Shape of the output of the previous layer.
        """
        super().build(input_shape)

        self._multi_headed_attention = Attention()
        # Create the layer activation
        self._dropout = Dropout(0.2)


    def call(self, inputs: Tuple[Layer], **kwargs: Dict) -> Layer:
        """Returns called Graph Convolution Layer.

        Parameters
        ---------------------------
        inputs: Tuple[Layer],
            Tuple with vector of labels and inputs.
        """
        features, A = inputs

        hidden = self._dropout(features)

        return self._multi_headed_attention(
            [hidden, hidden], 
            mask=[None, tf.cast(A, bool)]
        )
