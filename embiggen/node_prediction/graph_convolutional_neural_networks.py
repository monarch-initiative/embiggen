"""Graph Convolutional Neural Network (GCNN) model for graph embedding."""
from typing import Dict, List, Union

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from extra_keras_metrics import get_minimal_multiclass_metrics
from tensorflow.keras.layers import Input
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.constraints import Constraint
from tqdm.keras import TqdmCallback
from tensorflow.keras.models import Model  # pylint: disable=import-error
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error

from ensmallen_graph import EnsmallenGraph
from ..embedders.layers import GraphConvolution


class GraphConvolutionalNeuralNetwork:
    """Graph Convolutional Neural Network (GCNN) model for graph embedding."""

    def __init__(
        self,
        features_number: int,
        number_of_outputs: int,
        nodes_number: int,
        number_of_hidden_layers: int = 1,
        number_of_units_per_hidden_layer: Union[int, List[int]] = 16,
        kernel_initializer: Union[str, Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, Initializer] = 'zeros',
        kernel_regularizer: Union[str, Regularizer] = None,
        bias_regularizer: Union[str, Regularizer] = None,
        activity_regularizer: Union[str, Regularizer] = None,
        kernel_constraint: Union[str, Constraint] = None,
        bias_constraint: Union[str, Constraint] = None,
        dropout_rate: float = 0.5,
        optimizer: Union[str, Optimizer] = "nadam",
    ):
        """Create new GloVe-based Embedder object.

        Parameters
        -------------------------------
        features_number: int,
            Number of features.
        number_of_outputs: int,
            Number of output classes.
        nodes_number: int,
            Number of nodes of the considered graph.
        number_of_hidden_layers: int = 1,
            Number of graph convolution layer.
            CONSIDER WELL BEFORE USING MORE THAN ONE.
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
        dropout_rate: float = 0.5,
            Float between 0 and 1. Fraction of the input units to drop.

        """
        if isinstance(number_of_units_per_hidden_layer, int):
            number_of_units_per_hidden_layer = [
                number_of_units_per_hidden_layer
            ] * number_of_hidden_layers

        if len(number_of_units_per_hidden_layer) != number_of_hidden_layers:
            raise ValueError(
                "The number of hidden layers must match"
                "the number of the hidden units per layer provided"
            )
        self._features_number = features_number
        self._nodes_number = nodes_number
        self._number_of_outputs = number_of_outputs
        self._number_of_hidden_layers = number_of_hidden_layers
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._activity_regularizer = activity_regularizer
        self._kernel_constraint = kernel_constraint
        self._bias_constraint = bias_constraint
        self._dropout_rate = dropout_rate
        self._number_of_units_per_hidden_layer = number_of_units_per_hidden_layer
        self._optimizer = optimizer
        self._model = self._build_model()
        self._compile_model()

    def _build_model(self):
        """Create new GCN model."""
        hidden = features = Input(
            shape=(self._features_number,), batch_size=self._nodes_number,)

        symmetrically_normalized_adjacency_matrix = Input(
            shape=(self._nodes_number,),
            batch_size=self._nodes_number,
            sparse=True
        )

        for i, number_of_units in enumerate(self._number_of_units_per_hidden_layer):
            if i + 1 == self._number_of_hidden_layers:
                number_of_units = self._number_of_outputs
                if self._number_of_outputs == 1:
                    activation = "sigmoid"
                else:
                    activation = "softmax"
            else:
                activation = "relu"
            hidden = GraphConvolution(
                number_of_units,
                activation=activation,
                dropout_rate=self._dropout_rate,
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                activity_regularizer=self._activity_regularizer,
                kernel_constraint=self._kernel_constraint,
                bias_constraint=self._bias_constraint,
            )([hidden, symmetrically_normalized_adjacency_matrix])

        return Model(
            inputs=[
                features,
                symmetrically_normalized_adjacency_matrix
            ],
            outputs=hidden,
            name="GCN"
        )

    def _compile_model(self) -> Model:
        """Compile model."""
        self._model.compile(
            loss='binary_crossentropy' if self._number_of_outputs == 1 else "categorical_crossentropy",
            optimizer=self._optimizer,
            weighted_metrics=get_minimal_multiclass_metrics()
        )

    def summary(self):
        """Print model summary."""
        self._model.summary()

    def fit(
        self,
        train_graph: EnsmallenGraph,
        node_features: pd.DataFrame,
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 10,
        reduce_lr_min_delta: float = 0.001,
        reduce_lr_patience: int = 5,
        validation_graph: EnsmallenGraph = None,
        epochs: int = 10000,
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
        node_features: pd.DataFrame,
            Features for all on the graph nodes.
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

        if not all(node_features.index == train_graph.get_node_names()):
            raise ValueError(
                'The index of the provided dataframe'
                'does not map to the given graph nodes.'
            )

        symmetrically_normalized_graph = train_graph.get_unweighted_symmetric_normalized_transformed_graph()
        symmetrically_normalized_adjacency_matrix = tf.SparseTensor(
            symmetrically_normalized_graph.get_edge_node_ids(directed=True),
            symmetrically_normalized_graph.get_edge_weights(),
            (train_graph.get_nodes_number(), train_graph.get_nodes_number())
        )

        training_input_data = (
            node_features.values,
            symmetrically_normalized_adjacency_matrix,
        )

        if validation_graph:
            validation_data = (
                training_input_data,
                validation_graph.get_one_hot_encoded_node_types(),
                validation_graph.get_one_hot_encoded_node_types().any(axis=1)
            )
        else:
            validation_data = None

        callbacks = kwargs.pop("callbacks", ())
        return pd.DataFrame(self._model.fit(
            training_input_data, train_graph.get_one_hot_encoded_node_types(),
            sample_weight=train_graph.get_one_hot_encoded_node_types().any(axis=1),
            validation_data=validation_data,
            shuffle=False,
            epochs=epochs,
            verbose=False,
            batch_size=self._nodes_number,
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
