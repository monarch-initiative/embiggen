"""Submodule providing graph convolutional layer.

# References
The layer is implemented as described in [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).

In this version of the implementation, we allow for batch sizes of arbitrary size.
"""
from typing import Tuple, Union, Dict, Optional
import tensorflow as tf
from tensorflow.python.ops import nn_ops  # pylint: disable=import-error,no-name-in-module
from tensorflow.python.ops import embedding_ops  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import Dropout, Layer, Dense  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.initializers import Initializer  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.regularizers import Regularizer  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.constraints import Constraint  # pylint: disable=import-error,no-name-in-module


class GraphConvolution(Layer):
    """Layer implementing graph convolution layer."""

    def __init__(
        self,
        units: int,
        activation: str = "relu",
        use_bias: bool = True,
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
        use_bias: bool = True,
            Whether to use bias in the layer.
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
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._activity_regularizer = activity_regularizer
        self._kernel_constraint = kernel_constraint
        self._bias_constraint = bias_constraint
        self._features_dropout_rate = features_dropout_rate
        self._internal_product = False
        self._dense = None
        self._use_bias = use_bias
        self._bias = None
        self._features_dropout = None
        self._norm = None

    def build(self, input_shape: Tuple[int, int]) -> None:
        """Build the NCE layer.

        Parameters
        ------------------------------
        input_shape: Tuple[int, int],
            Shape of the output of the previous layer.
        """
        if self._units < input_shape:
            self._internal_product = True
            # TODO! implement internal product
            pass
        self._dense = Dense(
            units=self._units,
            use_bias=self._use_bias,
            activation=self._activation,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            activity_regularizer=self._activity_regularizer,
        )

        self._dense.build(input_shape)
        # Create the layer activation
        self._features_dropout = Dropout(self._features_dropout_rate)
        self._features_dropout.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        adjacency: tf.SparseTensor,
        node_features: tf.Tensor
    ) -> tf.Tensor:
        """Returns called Graph Convolution Layer.

        Parameters
        ---------------------------
        adjaceny: Tuple[tf.SparseTensor, tf.Tensor],
            Sparse weighted input matrix.
        node_features: tf.Tensor
        """
        ids = tf.SparseTensor(
            indices=adjacency.indices,
            values=adjacency.indices[:, 1],
            dense_shape=adjacency.dense_shape
        )
        if self._internal_product:
            # TODO! implement internal product
            pass
        return self._dense(embedding_ops.embedding_lookup_sparse_v2(
            self._features_dropout(node_features),
            ids,
            adjacency,
            combiner='sum'
        ))
