"""Graph Convolutional Neural Network (GCNN) model for graph embedding."""
from typing import Dict, List, Union, Optional

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from extra_keras_metrics import get_minimal_multiclass_metrics
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.constraints import Constraint
from tqdm.keras import TqdmCallback
from tensorflow.keras.models import Model  # pylint: disable=import-error
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error

from ensmallen_graph import EnsmallenGraph
from embiggen.embedders.layers import GraphConvolution


class GraphConvolutionalNeuralNetwork:
    """Graph Convolutional Neural Network (GCNN) model for graph embedding."""

    def __init__(
        self,
        graph: EnsmallenGraph,
        num_split: int = 1,
        use_weights: Union[str, bool] = "auto",
        node_features_number: Optional[int] = None,
        node_features: Optional[pd.DataFrame] = None,
        number_of_hidden_layers: int = 1,
        number_of_units_per_hidden_layer: Union[int, List[int]] = 16,
        activations_per_hidden_layer: Union[str, List[str]] = "relu",
        kernel_initializer: Union[str, Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, Initializer] = 'zeros',
        kernel_regularizer: Union[str, Regularizer] = None,
        bias_regularizer: Union[str, Regularizer] = None,
        activity_regularizer: Union[str, Regularizer] = None,
        kernel_constraint: Union[str, Constraint] = None,
        bias_constraint: Union[str, Constraint] = None,
        features_dropout_rate: float = 0.5,
        optimizer: Union[str, Optimizer] = "nadam",
    ):
        """Create new GloVe-based Embedder object.

        Parameters
        -------------------------------
        graph: EnsmallenGraph,
            The data for which to build the model.
        num_split: int = 1,
            Number of splits to use to distribute the steps of
            the graph convolution. This is only needed on graphs
            that have a incidence matrix that exeeds the dimension
            of the GPU memory.
        number_of_units_per_hidden_layer: Union[int, List[int]] = 16,
            Number of units per hidden layer.
        use_weights: Union[str, bool] = "auto",
            Whether to expect weights in input to execute the graph convolution.
            The weights may be used in order to compute for instance a weighting
            using the symmetric normalized Laplacian method.
        nodes_number: Optional[int] = None,
            Number of nodes in the considered.
            If the node features are provided, the nodes number is extracted by the node features.
        node_features_number: Optional[int] = None,
            Number of node features.
            If the node features are provided, the features number is extracted by the node features.
        node_features: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            Vector with the provided node features.
        trainable: Union[str, bool] = "auto",
            Whether to make the node features trainable.
            By default, with "auto", the embedding is trainable if no node features where provided.
        number_of_hidden_layers: int = 1,
            Number of graph convolution layer.
        number_of_units_per_hidden_layer: Union[int, List[int]] = 16,
            Number of units per hidden layer.
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

        """
        if isinstance(number_of_units_per_hidden_layer, int):
            number_of_units_per_hidden_layer = [
                number_of_units_per_hidden_layer
            ] * number_of_hidden_layers

        if isinstance(activations_per_hidden_layer, str):
            activations_per_hidden_layer = [
                activations_per_hidden_layer
            ] * number_of_hidden_layers

        if len(number_of_units_per_hidden_layer) != number_of_hidden_layers:
            raise ValueError(
                "The number of hidden layers must match"
                "the number of the hidden units per layer provided"
            )

        if len(activations_per_hidden_layer) != number_of_hidden_layers:
            raise ValueError(
                "The number of hidden layers must match"
                "the number of the activations per layer provided"
            )

        use_weights_supported_values = ("auto", True, False)
        if use_weights not in use_weights_supported_values:
            raise ValueError(
                (
                    "The provided value for `use_weights`, '{}', is not among the supported values '{}'."
                ).format(use_weights, use_weights_supported_values)
            )
        if node_features is not None and any(node_features.index != graph.get_node_names()):
            raise ValueError(
                "The provided node features DataFrame is not aligned with the "
                "provided graph nodes."
            )
        if node_features_number is None and node_features is None:
            raise ValueError(
                "Eiter the number of node features or the node features "
                "themselves must be provided."
            )
        if use_weights == "auto":
            use_weights = graph.has_edge_weights()
        if node_features is not None:
            node_features_number = node_features.shape[-1]
        self._use_weights = use_weights
        self._num_split = num_split
        self._node_features_number = node_features_number
        self._node_features = node_features
        self._nodes_number = graph.get_nodes_number()
        self._node_types_number = graph.get_node_types_number()
        number_of_units_per_hidden_layer[-1] = self._node_types_number
        self._number_of_hidden_layers = number_of_hidden_layers
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._activity_regularizer = activity_regularizer
        self._kernel_constraint = kernel_constraint
        self._bias_constraint = bias_constraint
        self._features_dropout_rate = features_dropout_rate
        self._number_of_units_per_hidden_layer = number_of_units_per_hidden_layer
        self._multi_label = graph.has_multilabel_node_types()
        activations_per_hidden_layer[-1] = "sigmoid" if self._multi_label or self._node_types_number == 1 else "softmax"
        self._activations_per_hidden_layer = activations_per_hidden_layer
        self._optimizer = optimizer
        self._model = self._build_model()
        self._compile_model()

    def _build_model(self):
        """Create new GCN model."""
        adjacency_matrix = Input(
            shape=(self._nodes_number,),
            sparse=True
        )

        input_graph_convolution = GraphConvolution(
            self._number_of_units_per_hidden_layer[0],
            activation=self._activations_per_hidden_layer[0],
            num_split=self._num_split,
            features_dropout_rate=self._features_dropout_rate,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )

        if self._node_features is None:
            node_features = input_graph_convolution.add_weight(
                name="node_features",
                shape=(self._nodes_number, self._node_features_number),
                trainable=True,
                initializer="glorot_normal",
                dtype=tf.float32
            )
        else:
            node_features = tf.Variable(
                initial_value=self._node_features.values,
                trainable=False,
                validate_shape=True,
                name="node_features",
                shape=(self._nodes_number, self._node_features_number),
                dtype=tf.float32
            )

        hidden = input_graph_convolution(adjacency_matrix, node_features)
        for i in range(1, self._number_of_hidden_layers):
            hidden = GraphConvolution(
                self._number_of_units_per_hidden_layer[i],
                activation=self._activations_per_hidden_layer[i],
                num_split=self._num_split,
                features_dropout_rate=self._features_dropout_rate,
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                activity_regularizer=self._activity_regularizer,
                kernel_constraint=self._kernel_constraint,
                bias_constraint=self._bias_constraint,
            )(adjacency_matrix, hidden)

        return Model(
            inputs=adjacency_matrix,
            outputs=hidden,
            name="GCN"
        )

    def _compile_model(self) -> Model:
        """Compile model."""
        self._model.compile(
            loss='binary_crossentropy' if self._node_types_number == 1 or self._multi_label else "categorical_crossentropy",
            optimizer=self._optimizer,
            weighted_metrics=get_minimal_multiclass_metrics()
        )

    def summary(self):
        """Print model summary."""
        self._model.summary()

    def fit(
        self,
        train_graph: EnsmallenGraph,
        batch_size: Union[int, str] = "auto",
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 10,
        reduce_lr_min_delta: float = 0.001,
        reduce_lr_patience: int = 5,
        validation_graph: EnsmallenGraph = None,
        epochs: int = 1000,
        early_stopping_monitor: str = "loss",
        early_stopping_mode: str = "min",
        reduce_lr_monitor: str = "loss",
        reduce_lr_mode: str = "min",
        reduce_lr_factor: float = 0.9,
        verbose: int = 1,
        **kwargs: Dict
    ) -> pd.DataFrame:
        """Return pandas dataframe with training history.

        Parameters
        -----------------------
        train_graph: EnsmallenGraph,
            Graph to use for the training.
        batch_size: Union[int, str] = "auto",
            Batch size for the training epochs.
            If the model has a single GCN layer it is possible
            to specify a variable batch size.
        early_stopping_min_delta: float,
            Minimum delta of metric to stop the training.
        early_stopping_patience: int,
            Number of epochs to wait for when the given minimum delta is not
            achieved after which trigger early stopping.
        reduce_lr_min_delta: float,
            Minimum delta of metric to reduce learning rate.
        reduce_lr_patience: int,
            Number of epochs to wait for when the given minimum delta is not
            achieved after which reducing learning rate.
        validation_graph: EnsmallenGraph = None,
            Tuple to use for the validation.
        epochs: int = 10000,
            Epochs to train the model for.
        early_stopping_monitor: str = "loss",
            Metric to monitor for early stopping.
        early_stopping_mode: str = "min",
            Direction of the variation of the monitored metric for early stopping.
        reduce_lr_monitor: str = "loss",
            Metric to monitor for reducing learning rate.
        reduce_lr_mode: str = "min",
            Direction of the variation of the monitored metric for learning rate.
        reduce_lr_factor: float = 0.9,
            Factor for reduction of learning rate.
        verbose: int = 1,
            Wethever to show the loading bar.
            Specifically, the options are:
            * 0 or False: No loading bar.
            * 1 or True: Showing only the loading bar for the epochs.
            * 2: Showing loading bar for both epochs and batches.
        **kwargs: Dict,
            Additional kwargs to pass to the Keras fit call.

        Raises
        -----------------------
        ValueError,
            If given verbose value is not within the available set (-1, 0, 1).

        Returns
        -----------------------
        Dataframe with training history.
        """
        if verbose == True:
            verbose = 1
        if verbose == False:
            verbose = 0
        if verbose not in {0, 1, 2}:
            raise ValueError(
                "Given verbose value is not valid, as it must be either "
                "a boolean value or 0, 1 or 2."
            )

        if train_graph.has_singleton_nodes():
            raise ValueError(
                "The GCN model does not support operations on graph containing "
                "singletons. You need to either drop the singleton nodes or "
                "add to them a singleton."
            )

        if train_graph.is_multigraph():
            raise ValueError(
                "The GCN model does not support operations on a multigraph. "
                "You need to drop the parallel edges in order to execute this "
                "model."
            )
        if batch_size == "auto":
            batch_size = train_graph.get_nodes_number()

        if self._number_of_hidden_layers != 1 and batch_size != train_graph.get_nodes_number():
            raise ValueError(
                "If the number of GCN layers is greater than 1, "
                "the batch size must be equal to the number of "
                "nodes in the graph."
            )

        if not train_graph.has_edge_weights() and self._use_weights:
            raise ValueError(
                "The model expected a training graph with edge weights!"
            )

        if validation_graph is not None and not validation_graph.has_edge_weights() and self._use_weights:
            raise ValueError(
                "The model expected a validation graph with edge weights!"
            )

        adjacency_matrix = tf.SparseTensor(
            train_graph.get_edge_node_ids(directed=True),
            train_graph.get_edge_weights() if train_graph.has_edge_weights(
            ) and self._use_weights else tf.ones(train_graph.get_directed_edges_number()),
            (train_graph.get_nodes_number(), train_graph.get_nodes_number()),
        )

        if validation_graph:
            validation_data = (
                adjacency_matrix,
                validation_graph.get_one_hot_encoded_node_types(),
                validation_graph.get_one_hot_encoded_node_types().any(axis=1)
            )
        else:
            validation_data = None

        callbacks = kwargs.pop("callbacks", ())
        return pd.DataFrame(self._model.fit(
            adjacency_matrix, train_graph.get_one_hot_encoded_node_types(),
            sample_weight=train_graph.get_one_hot_encoded_node_types().any(axis=1),
            validation_data=validation_data,
            epochs=epochs,
            verbose=False,
            batch_size=batch_size,
            callbacks=[
                EarlyStopping(
                    monitor=early_stopping_monitor,
                    min_delta=early_stopping_min_delta,
                    patience=early_stopping_patience,
                    mode=early_stopping_mode,
                ),
                ReduceLROnPlateau(
                    monitor=reduce_lr_monitor,
                    min_delta=reduce_lr_min_delta,
                    patience=reduce_lr_patience,
                    factor=reduce_lr_factor,
                    mode=reduce_lr_mode,
                ),
                *((TqdmCallback(verbose=verbose-1),) if verbose > 0 else ()),
                *callbacks
            ],
            **kwargs
        ).history)
