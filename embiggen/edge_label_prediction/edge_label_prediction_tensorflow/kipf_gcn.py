"""Kipf GCN model for edge-labek prediction."""
from typing import List, Union, Optional, Dict, Any, Type

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import Input, Concatenate, Flatten, Dense  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.models import Model  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error,no-name-in-module

from ensmallen import Graph
from ...layers.tensorflow import GraphConvolution
from ...utils.tensorflow_utils import graph_to_sparse_tensor
from ...utils.normalize_model_structural_parameters import normalize_model_list_parameter
from ..edge_label_prediction_model import AbstractEdgeLabelPredictionModel
import tensorflow as tf
import copy


class KipfGCNEdgeLabelPrediction(AbstractEdgeLabelPredictionModel):
    """Kipf GCN model for edge-label prediction."""

    def __init__(
        self,
        epochs: int = 1000,
        batch_size: int = 2**10,
        number_of_gcn_body_layers: int = 2,
        number_of_gcn_head_layers: int = 1,
        number_of_ffnn_body_layers: int = 2,
        number_of_ffnn_head_layers: int = 1,
        number_of_units_per_gcn_body_layer: Union[int, List[int]] = 128,
        number_of_units_per_gcn_head_layer: Union[int, List[int]] = 128,
        number_of_units_per_ffnn_body_layer: Union[int, List[int]] = 128,
        number_of_units_per_ffnn_head_layer: Union[int, List[int]] = 128,
        features_dropout_rate: float = 0.2,
        optimizer: Union[str, Optimizer] = "LazyAdam",
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 10,
        reduce_lr_min_delta: float = 0.001,
        reduce_lr_patience: int = 5,
        early_stopping_monitor: str = "loss",
        early_stopping_mode: str = "min",
        reduce_lr_monitor: str = "loss",
        reduce_lr_mode: str = "min",
        reduce_lr_factor: float = 0.9,
        use_class_weights: bool = True,
        use_laplacian: bool = True,
        verbose: bool = True
    ):
        """Create new Kipf GCN object.

        Parameters
        -------------------------------
        epochs: int = 1000
            Epochs to train the model for.
        batch_size: int = 2**10
            While the size of the convolution is always fixed to the number
            of nodes in the graph, it is possible to specify the batch size of
            the edge label prediction task.
        number_of_gcn_body_layers: int = 2
            Number of layers in the body subsection of the GCN section of the model.
        number_of_gcn_head_layers: int = 1
            Number of layers in the head subsection of the GCN section of the model.
        number_of_ffnn_body_layers: int = 2
            Number of layers in the body subsection of the FFNN section of the model.
        number_of_ffnn_head_layers: int = 1
            Number of layers in the head subsection of the FFNN section of the model.
        number_of_units_per_gcn_body_layer: Union[int, List[int]] = 128
            Number of units per gcn body layer.
        number_of_units_per_gcn_head_layer: Union[int, List[int]] = 128
            Number of units per gcn head layer.
        number_of_units_per_ffnn_body_layer: Union[int, List[int]] = 128
            Number of units per ffnn body layer.
        number_of_units_per_ffnn_head_layer: Union[int, List[int]] = 128
            Number of units per ffnn head layer.
        features_dropout_rate: float = 0.3
            Float between 0 and 1.
            Fraction of the input units to dropout.
        optimizer: str = "LazyAdam"
            The optimizer to use while training the model.
            By default, we use `LazyAdam`, which should be faster
            than Adam when handling sparse gradients such as the one
            we are using to train this model.
            When the tensorflow addons module is not available,
            we automatically switch back to `Adam`.
        early_stopping_min_delta: float
            Minimum delta of metric to stop the training.
        early_stopping_patience: int
            Number of epochs to wait for when the given minimum delta is not
            achieved after which trigger early stopping.
        reduce_lr_min_delta: float
            Minimum delta of metric to reduce learning rate.
        reduce_lr_patience: int
            Number of epochs to wait for when the given minimum delta is not
            achieved after which reducing learning rate.
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
        use_class_weights: bool = True
            Whether to use class weights to rebalance the loss relative to unbalanced classes.
            Learn more about class weights here: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        use_laplacian: bool = True
            Whether to use laplacian transform before training on the graph.
        verbose: bool = True
            Whether to show loading bars.
        """
        super().__init__()
        self._number_of_units_per_gcn_body_layer = normalize_model_list_parameter(
            number_of_units_per_gcn_body_layer,
            number_of_gcn_body_layers,
            object_type=int
        )
        self._number_of_units_per_gcn_head_layer = normalize_model_list_parameter(
            number_of_units_per_gcn_head_layer,
            number_of_gcn_head_layers,
            object_type=int
        )
        self._number_of_units_per_ffnn_body_layer = normalize_model_list_parameter(
            number_of_units_per_ffnn_body_layer,
            number_of_ffnn_body_layers,
            object_type=int
        )
        self._number_of_units_per_ffnn_head_layer = normalize_model_list_parameter(
            number_of_units_per_ffnn_head_layer,
            number_of_ffnn_head_layers,
            object_type=int
        )

        self._epochs = epochs
        self._use_class_weights = use_class_weights
        self._features_dropout_rate = features_dropout_rate
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._use_laplacian = use_laplacian

        self._early_stopping_min_delta = early_stopping_min_delta
        self._early_stopping_patience = early_stopping_patience
        self._reduce_lr_min_delta = reduce_lr_min_delta
        self._reduce_lr_patience = reduce_lr_patience
        self._early_stopping_monitor = early_stopping_monitor
        self._early_stopping_mode = early_stopping_mode
        self._reduce_lr_monitor = reduce_lr_monitor
        self._reduce_lr_mode = reduce_lr_mode
        self._reduce_lr_factor = reduce_lr_factor

        self._verbose = verbose
        self._model = None
        self.history = None

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            epochs=1,
            number_of_units_per_gcn_body_layer=8,
            number_of_units_per_gcn_head_layer=8,
            number_of_units_per_ffnn_body_layer=8,
            number_of_units_per_ffnn_head_layer=8,
        )

    def clone(self) -> Type["KipfGCNEdgeLabelPrediction"]:
        """Return copy of self."""
        return copy.deepcopy(self)

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return dict(
            number_of_units_per_gcn_body_layer=self._number_of_units_per_gcn_body_layer,
            number_of_units_per_gcn_head_layer=self._number_of_units_per_gcn_head_layer,
            number_of_units_per_ffnn_body_layer=self._number_of_units_per_ffnn_body_layer,
            number_of_units_per_ffnn_head_layer=self._number_of_units_per_ffnn_head_layer,
            epochs=self._epochs,
            use_class_weights=self._use_class_weights,
            features_dropout_rate=self._features_dropout_rate,
            optimizer=self._optimizer,
            early_stopping_min_delta=self._early_stopping_min_delta,
            early_stopping_patience=self._early_stopping_patience,
            reduce_lr_min_delta=self._reduce_lr_min_delta,
            reduce_lr_patience=self._reduce_lr_patience,
            early_stopping_monitor=self._early_stopping_monitor,
            early_stopping_mode=self._early_stopping_mode,
            reduce_lr_monitor=self._reduce_lr_monitor,
            reduce_lr_mode=self._reduce_lr_mode,
            reduce_lr_factor=self._reduce_lr_factor,
        )

    def _build_model(
        self,
        graph: Graph,
        node_features: List[np.ndarray],
        edge_feature_sizes: List[int] = None
    ):
        """Create new GCN model."""
        source_nodes = Input((1,), name="sources", dtype=tf.int32)
        destination_nodes = Input((1,), name="destinations", dtype=tf.int32)
        edge_features = [] if edge_feature_sizes is None else [
            Input((node_features_size,))
            for node_features_size in node_features_sizes
        ]

        adjacency_matrix = graph_to_sparse_tensor(
            graph,
            use_weights=graph.has_edge_weights() and not self._use_laplacian,
            use_laplacian=self._use_laplacian
        )
        node_features_sizes = [
            node_feature.shape[1]
            for node_feature in node_features
        ]
        node_features = [
            tf.Variable(
                initial_value=node_feature.astype(np.float32),
                trainable=False,
                validate_shape=True,
                shape=node_feature.shape,
                dtype=np.float32
            )
            for node_feature in node_features
        ]

        submodules_outputs = []
        submodules_output_sizes = []

        # Build the various submodules, one for each feature.
        for hidden, last_hidden_size in zip(node_features, node_features_sizes):
            # Building the body of the model.
            for units in self._number_of_units_per_gcn_body_layer:
                gcn_hidden = GraphConvolution(
                    units=units,
                    features_dropout_rate=self._features_dropout_rate,
                )

                gcn_hidden.build(last_hidden_size)

                hidden = gcn_hidden(adjacency_matrix, hidden)
                last_hidden_size = units

            submodules_outputs.append(hidden)
            submodules_output_sizes.append(last_hidden_size)

        hidden = Concatenate()(submodules_outputs)
        last_hidden_size = sum(submodules_output_sizes)

        # Building the head of the model.
        for units in self._number_of_units_per_gcn_head_layer:
            gcn_hidden = GraphConvolution(
                units=units,
                features_dropout_rate=self._features_dropout_rate,
            )

            gcn_hidden.build(last_hidden_size)

            hidden = gcn_hidden(adjacency_matrix, hidden)
            last_hidden_size = units

        gcn_node_features = hidden
        ffnn_outputs = []

        source_and_destination_features = []

        for nodes in (source_nodes, destination_nodes):
            source_and_destination_features.append(Flatten()(tf.nn.embedding_lookup(
                gcn_node_features,
                ids=nodes
            )))

        for hidden in edge_features + source_and_destination_features:
            # Building the body of the model.
            for units in self._number_of_units_per_ffnn_body_layer:
                hidden = Dense(
                    units=units,
                    activation="relu"
                )(hidden)

            ffnn_outputs.append(hidden)
        
        hidden = Concatenate()(ffnn_outputs)

        # Building the head of the model.
        for units in self._number_of_units_per_ffnn_head_layer:
            hidden = Dense(
                units=units,
                activation="relu"
            )(hidden)

        # Adding the last layer of the model.
        if self.is_binary_prediction_task():
            units = 1
            activation = "sigmoid"
        else:
            units = graph.get_edge_types_number()
            activation = "softmax"

        output = Dense(
            units=units,
            activation=activation
        )(hidden)

        # Building the the model.

        model = Model(
            inputs=[source_nodes, destination_nodes, *edge_features],
            outputs=output,
            name=self.model_name().replace(" ", "_")
        )

        # Compiling the model.
        if self.is_multilabel_prediction_task() or self.is_binary_prediction_task():
            loss = "binary_crossentropy"
        else:
            loss = "sparse_categorical_crossentropy"

        try:
            if self._optimizer == "LazyAdam":
                import tensorflow_addons as tfa
                optimizer = tfa.optimizers.LazyAdam(0.001)
        except:
            optimizer = "adam"

        model.compile(
            loss=loss,
            optimizer=optimizer
        )

        return model

    @staticmethod
    def model_name() -> str:
        return "Kipf GCN"

    def _fit(
        self,
        graph: Graph,
        node_features: List[np.ndarray],
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Return pandas dataframe with training history.

        Parameters
        -----------------------
        graph: Graph,
            The graph whose edges are to be embedded and edge types extracted.
            It can either be an Graph or a list of lists of edges.
        node_features: List[np.ndarray]
            The node features to be used in the training of the model.
        edge_features: Optional[List[np.ndarray]] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the graph.

        Returns
        -----------------------
        Dataframe with training history.
        """
        try:
            from tqdm.keras import TqdmCallback
            traditional_verbose = False
        except AttributeError:
            traditional_verbose = True

        edges_number = graph.get_number_of_directed_edges()
        edge_types_number = graph.get_edge_types_number()

        if self._use_class_weights:
            class_weight = {
                edge_type_id: edges_number / count / edge_types_number
                for edge_type_id, count in graph.get_edge_type_id_counts_hashmap().items()
            }
        else:
            class_weight = None

        model = self._build_model(
            graph,
            node_features,
            edge_feature_sizes = None if edge_features is None else [
                edge_feature.shape[1]
                for edge_feature in edge_features
            ]
        )

        if self.is_multilabel_prediction_task():
            # TODO! support multilabel prediction!
            raise NotImplementedError(
                "Currently we do not support multi-label edge-label prediction "
                f"in the {self.model_name()} from the {self.library_name()} "
                f"as it is implemented in the {self.__class__.__name__} class."
            )
        elif self.is_binary_prediction_task():
            edge_labels = graph.get_known_edge_type_ids() == 1
        else:
            edge_labels = graph.get_known_edge_type_ids()

        self.history = model.fit(
            (
                graph.get_directed_source_nodes_with_known_edge_types(),
                graph.get_directed_destination_nodes_with_known_edge_types(),
                *(() if edge_features is None else edge_features)
            ),
            edge_labels,
            epochs=self._epochs,
            verbose=traditional_verbose and self._verbose > 0,
            batch_size=self._batch_size,
            class_weight=class_weight,
            callbacks=[
                EarlyStopping(
                    monitor=self._early_stopping_monitor,
                    min_delta=self._early_stopping_min_delta,
                    patience=self._early_stopping_patience,
                    mode=self._early_stopping_mode,
                ),
                ReduceLROnPlateau(
                    monitor=self._reduce_lr_monitor,
                    min_delta=self._reduce_lr_min_delta,
                    patience=self._reduce_lr_patience,
                    factor=self._reduce_lr_factor,
                    mode=self._reduce_lr_mode,
                ),
                *((TqdmCallback(
                    verbose=1,
                    leave=False
                ),)
                    if not traditional_verbose and self._verbose > 0 else ()),
            ],
        )

        self._model = model

    def _predict_proba(
        self,
        graph: Graph,
        node_features: np.ndarray,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Run predictions on the provided graph."""
        return self._model.predict(
            (graph.get_directed_source_node_ids(), graph.get_directed_destination_node_ids()),
            batch_size=self._batch_size,
            verbose=False
        )

    def _predict(
        self,
        graph: Graph,
        node_features: np.ndarray,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Run predictions on the provided graph."""
        return self._predict_proba(
            graph,
            node_features=node_features,
            edge_features=edge_features
        ).argmax(axis=-1)

    @staticmethod
    def requires_edge_weights() -> bool:
        return True

    @staticmethod
    def requires_positive_edge_weights() -> bool:
        return False

    @staticmethod
    def requires_node_types() -> bool:
        return False

    @staticmethod
    def library_name() -> str:
        """Return name of the model."""
        return "TensorFlow"

    @staticmethod
    def can_use_edge_weights() -> bool:
        """Returns whether the model can optionally use edge weights."""
        return True

    def is_using_edge_weights(self) -> bool:
        """Returns whether the model is parametrized to use edge weights."""
        return not self._use_laplacian

    @staticmethod
    def can_use_node_types() -> bool:
        """Returns whether the model can optionally use node types."""
        return False

    def is_using_node_types(self) -> bool:
        """Returns whether the model is parametrized to use node types."""
        return False