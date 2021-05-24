"""Submodule providing graph convolutional layer.

# References
The layer is implemented as described in [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).
"""
import imp
from typing import Tuple, Union, Dict
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import Dropout, Input, Layer
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.constraints import Constraint


class GraphConvolution(Layer):
    """Layer implementing graph convolution layer."""

    def __init__(
        self,
        filter: int,
        activation: str = "relu",
        kernel_initializer: Union[str, Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, Initializer] = 'zeros',
        kernel_regularizer: Union[str, Regularizer] = None,
        bias_regularizer: Union[str, Regularizer] = None,
        kernel_constraint: Union[str, Constraint] = None,
        bias_constraint: Union[str, Constraint] = None,
        dropout_rate: float = 0.5,
        **kwargs: Dict
    ):
        """Create new GraphConvolution layer.

        Parameters
        ----------------------
        filter: int,
            The dimensionality of the output space (i.e. the number of output filters in the convolution).
        activation: str = "relu",
            Activation function to use. If you don't specify anything, relu is used.
        kernel_initializer: Union[str, Initializer] = 'glorot_uniform',
            Initializer for the kernel weights matrix.
        bias_initializer: Union[str, Initializer] = 'zeros',
            Initializer for the bias vector.
        kernel_regularizer: Union[str, Regularizer] = None,
            Regularizer function applied to the kernel weights matrix.
        bias_regularizer: Union[str, Regularizer] = None,
            Regularizer function applied to the bias vector.
        kernel_constraint: Union[str, Constraint] = None,
            Constraint function applied to the kernel matrix.
        bias_constraint: Union[str, Constraint] = None,
            Constraint function applied to the bias vector.
        dropout_rate: float = 0.5,
            Float between 0 and 1. Fraction of the input units to drop.
        **kwargs: Dict,
            Kwargs to pass to the parent Layer class.
        """
        super(GraphConvolution, self).__init__(**kwargs)
        self.filter = filter
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._kernel_constraint = kernel_constraint
        self._bias_constraint = bias_constraint
        self._dropout_rate = dropout_rate

    def build(self, input_shape: Tuple[int, int]) -> None:
        """Build the NCE layer.

        Parameters
        ------------------------------
        input_shape: Tuple[int, int],
            Shape of the output of the previous layer.
        """

        # Create the kernel of the layer
        self._kernel = self.add_weight(
            shape=(input_shape[0][-1], self.filter),
            initializer=self._kernel_initializer,
            regularizer=self._kernel_regularizer,
            constraint=self._kernel_constraint,
            name='kernel'
        )
        # Create the bias of the layer
        self._bias = self.add_weight(
            shape=(self.filter,),
            initializer=self._bias_initializer,
            regularizer=self._kernel_regularizer,
            constraint=self._bias_constraint,
            name='bias'
        )

        # Create the layer activation
        self._activation = activations.get(self._activation)

        # Create the layer activation
        self.dropout = Dropout(self._dropout_rate)

        super().build(input_shape)

    def call(self, inputs: Tuple[Layer], **kwargs: Dict) -> Layer:
        """Returns called Graph Convolution Layer.

        Parameters
        ---------------------------
        inputs: Tuple[Layer],
            Tuple with vector of labels and inputs.
        """
        features, A = inputs
        features = self.dropout(features)
        return self._activation(
            tf.matmul(
                tf.sparse.sparse_dense_matmul(A, features),
                self._kernel
            ) + self._bias
        )
