"""Submodule providing graph convolutional layer.

# References
The layer is implemented as described in [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).

In this version of the implementation, we allow for batch sizes of arbitrary size.
"""
from typing import Tuple, Union, Dict, Optional, List
import tensorflow as tf
from tensorflow.python.ops import embedding_ops  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import Dropout, Layer, Dense  # pylint: disable=import-error,no-name-in-module
from embiggen.layers.tensorflow.l2_norm import L2Norm


class GraphConvolution(Layer):
    """Layer implementing graph convolution layer."""

    def __init__(
        self,
        units: int,
        activation: str = "relu",
        dropout_rate: Optional[float] = 0.5,
        apply_norm: bool = False,
        **kwargs: Dict
    ):
        """Create new GraphConvolution layer.

        Parameters
        ----------------------
        units: int,
            The dimensionality of the output space (i.e. the number of output units).
        activation: str = "relu",
            Activation function to use. If you don't specify anything, relu is used.
        dropout_rate: Optional[float] = 0.5,
            Float between 0 and 1. Fraction of the input units to drop.
            If the provided value is either zero or None the dropout rate
            is not applied.
        **kwargs: Dict,
            Kwargs to pass to the parent Layer class.
        """
        super().__init__(**kwargs)
        self._units = units
        self._activation = activation
        if dropout_rate == 0.0:
            dropout_rate = None
        self._dropout_rate = dropout_rate
        self._apply_norm = apply_norm
        self._dense_layers = []
        self._dropout_layer = None
        self._l2_norm = None

    def build(self, input_shape) -> None:
        """Build the Graph Convolution layer.

        Parameters
        ------------------------------
        input_shape
            Shape of the output of the previous layer.
        """
        if len(input_shape) == 0:
            raise ValueError(
                "The provided input of the Graph Convolution layer "
                "is empty. It should contain exactly two elements, "
                "the adjacency matrix and the node features."
            )

        if len(input_shape) == 1:
            raise ValueError(
                "The provided input of the Graph Convolution layer "
                "has a single element. It should contain exactly two elements, "
                "the adjacency matrix and the node features."
            )
        for node_feature_shape in input_shape[1:]:
            dense_layer = Dense(
                units=self._units,
                activation=self._activation,
            )
            dense_layer.build(node_feature_shape)
            self._dense_layers.append(dense_layer)

        if self._dropout_rate is not None:
            self._dropout_layer = Dropout(self._dropout_rate)
            self._dropout_layer.build(node_feature_shape)
        else:
            self._dropout_layer = lambda x: x

        if self._apply_norm:
            self._l2_norm = L2Norm()
            self._l2_norm.build(node_feature_shape)
        else:
            self._l2_norm = lambda x: x

        super().build(input_shape)

    def call(
        self,
        inputs: Tuple[Union[tf.Tensor, List[tf.Tensor], tf.SparseTensor]],
    ) -> tf.Tensor:
        """Returns called Graph Convolution Layer.

        Parameters
        ---------------------------
        inputs: Tuple[Union[tf.Tensor, tf.SparseTensor]],
        """
        adjacency, node_features = inputs[0], inputs[1:]

        ids = tf.SparseTensor(
            indices=adjacency.indices,
            values=adjacency.indices[:, 1],
            dense_shape=adjacency.dense_shape
        )

        return [
            self._l2_norm(dense(embedding_ops.embedding_lookup_sparse_v2(
                self._dropout_layer(node_feature),
                ids,
                adjacency,
                combiner='mean'
            )))
            for dense, node_feature in zip(
                self._dense_layers,
                node_features
            )
        ]
