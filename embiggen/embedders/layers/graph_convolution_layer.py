"""Submodule providing graph convolutional layer.

# References
The layer is implemented as described in [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).

In this version of the implementation, we allow for batch sizes of arbitrary size.
"""
from typing import Tuple, Union, Dict
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Layer, Dense, Concatenate
from tensorflow.keras.constraints import UnitNorm
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.constraints import Constraint


class GraphConvolution(Layer):
    """Layer implementing graph convolution layer."""

    def __init__(
        self,
        units: int,
        activation: str = "relu",
        num_split: int = 1,
        kernel_initializer: Union[str, Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, Initializer] = 'zeros',
        kernel_regularizer: Union[str, Regularizer] = None,
        bias_regularizer: Union[str, Regularizer] = None,
        activity_regularizer: Union[str, Regularizer] = None,
        kernel_constraint: Union[str, Constraint] = None,
        bias_constraint: Union[str, Constraint] = None,
        features_dropout_rate: float = 0.5,
        **kwargs: Dict
    ):
        """Create new GraphConvolution layer.

        Parameters
        ----------------------
        units: int,
            The dimensionality of the output space (i.e. the number of output units).
        activation: str = "relu",
            Activation function to use. If you don't specify anything, relu is used.
        num_split: int = 1,
            Number of splits to use to distribute the steps of
            the graph convolution. This is only needed on graphs
            that have a incidence matrix that exeeds the dimension
            of the GPU memory.
        kernel_initializer: Union[str, Initializer] = 'glorot_uniform',
            Initializer for the kernel weights matrix.
        bias_initializer: Union[str, Initializer] = 'zeros',
            Initializer for the bias vector.
        kernel_regularizer: Union[str, Regularizer] = None,
            Regularizer function applied to the kernel weights matrix.
        bias_regularizer: Union[str, Regularizer] = None,
            Regularizer function applied to the bias vector.
        activity_regularizer: Union[str, Regularizer] = None,
            Regularizer function applied to the output of the activation function.
        kernel_constraint: Union[str, Constraint] = None,
            Constraint function applied to the kernel matrix.
        bias_constraint: Union[str, Constraint] = None,
            Constraint function applied to the bias vector.
        features_dropout_rate: float = 0.5,
            Float between 0 and 1. Fraction of the input units to drop.
        **kwargs: Dict,
            Kwargs to pass to the parent Layer class.
        """
        super().__init__(**kwargs)
        self._units = units
        self._activation = activation
        self._num_split = num_split
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._activity_regularizer = activity_regularizer
        self._kernel_constraint = kernel_constraint
        self._bias_constraint = bias_constraint
        self._features_dropout_rate = features_dropout_rate
        self._dense = None
        self._features_dropout = None
        self._norm = None

    def build(self, input_shape: Tuple[int, int]) -> None:
        """Build the NCE layer.

        Parameters
        ------------------------------
        input_shape: Tuple[int, int],
            Shape of the output of the previous layer.
        """
        self._dense = Dense(
            units=self._units,
            activation=self._activation,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            activity_regularizer=self._activity_regularizer,
        )
        # Create the layer activation
        self._features_dropout = Dropout(self._features_dropout_rate)
        # Create the layer to norm the values
        self._norm = UnitNorm(axis=-1)

        super().build(input_shape)

    def call(self, inputs: Tuple[tf.SparseTensor, tf.Tensor], **kwargs: Dict) -> Layer:
        """Returns called Graph Convolution Layer.

        Parameters
        ---------------------------
        inputs: Tuple[tf.SparseTensor, tf.Tensor],
            Sparse weighted input matrix.
        """
        adjacency, features = inputs
        features = self._features_dropout(features)
        if self._num_split > 1:
            output = Concatenate(axis=0)([
                tf.sparse.sparse_dense_matmul(batch, features)
                for batch in tf.sparse.split(adjacency, num_split=self._num_split, axis=0, )
            ])
        else:
            output = tf.sparse.sparse_dense_matmul(adjacency, features)
        return self._dense(output)
