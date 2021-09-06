"""Submodule providing a graph attention layer.

# References
The layer is implemented as described in [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf).

In this version of the implementation, we allow for batch sizes of arbitrary size.
"""
from typing import Tuple, Union, Dict, Optional
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dropout, Layer, Dense, LeakyReLU
from tensorflow.keras.constraints import UnitNorm
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.constraints import Constraint


class GraphAttention(Layer):
    """Layer implementing graph convolution layer."""

    def __init__(
        self,
        units: int,
        nodes_number: Optional[int] = None,
        node_feature_number: Optional[int] = None,
        node_features: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        trainable: Union[str, bool] = "auto",
        activation: str = "relu",
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
        nodes_number: Optional[int] = None,
            Number of nodes in the considered.
            If the node features are provided, the nodes number is extracted by the node features.
        node_feature_number: Optional[int] = None,
            Number of node features.
            If the node features are provided, the features number is extracted by the node features.
        node_features: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            Vector with the provided node features.
        trainable: Union[str, bool] = "auto",
            Whether to make the node features trainable.
            By default, with "auto", the embedding is trainable if no node features where provided.
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
        if node_features is not None:
            if isinstance(node_features, pd.DataFrame):
                node_features = node_features.values
            if len(node_features.shape) != 2:
                raise ValueError(
                    (
                        "The node features are expected to be with a two-dimensional shape "
                        "but the provided node features have shape {}."
                    ).format(node_features.shape)
                )
            nodes_number, node_feature_number = node_features.shape
        trainable_supported_values = ("auto", True, False)
        if trainable not in trainable_supported_values:
            raise ValueError(
                (
                    "The provided value for `trainable`, '{}', is not among the supported values '{}'."
                ).format(trainable, trainable_supported_values)
            )
        if trainable == "auto":
            trainable = node_features is None
        self._units = units
        self._nodes_number = nodes_number
        self._node_features_number = node_feature_number
        self._activation = activation
        self._trainable = trainable
        self._node_features = node_features
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._activity_regularizer = activity_regularizer
        self._kernel_constraint = kernel_constraint
        self._bias_constraint = bias_constraint
        self._features_dropout_rate = features_dropout_rate
        self._dense = None
        self._leaky_relu = None
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
            activation="linear",
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint
        )
        # Create the layer activation
        self._features_dropout = Dropout(self._features_dropout_rate)
        # Create the node features embedding
        self._node_features = self.add_weight(
            name="node_features",
            shape=(self._nodes_number, self._node_features_number),
            initializer="glorot_normal",
            trainable=self._trainable,
            weights=None if self._node_features is None else [
                self._node_features
            ]
        )
        # Create the leaky relu layer
        self._leaky_relu = LeakyReLU()
        # Create the layer to norm the values
        self._norm = UnitNorm(axis=-1)

        super().build(input_shape)

    def call(self, node_ids: tf.SparseTensor, **kwargs: Dict) -> Layer:
        """Returns called Graph Convolution Layer.

        Parameters
        ---------------------------
        inputs: Tuple[Layer],
            Tuple with vector of labels and inputs.
        """
        hidden_features = self._dense(
            self._features_dropout(self._node_features)
        )

        unnormalized_attention_scores = K.exp(self._leaky_relu(hidden_features))
        normalization_coefficients = tf.nn.embedding_lookup_sparse(
            unnormalized_attention_scores,
            node_ids,
            None,
            combiner="sum"
        )
        attention_scores = unnormalized_attention_scores / normalization_coefficients

        return tf.keras.activations.get(self._activation)(tf.nn.embedding_lookup_sparse(
            hidden_features,
            node_ids,
            attention_scores
        ))
