"""Kipf GCN model for node-label prediction."""
from typing import List, Union, Optional, Dict, Any, Type

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import Input, Concatenate  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.models import Model  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error,no-name-in-module

from ensmallen import Graph
from ...layers.tensorflow import GraphConvolution
from ...utils.tensorflow_utils import graph_to_sparse_tensor
from ...utils.normalize_model_structural_parameters import normalize_model_list_parameter
from ..node_label_prediction_model import AbstractNodeLabelPredictionModel
import copy


class KipfGCNNodeLabelPrediction(AbstractNodeLabelPredictionModel):
    """Kipf GCN model for node-label prediction."""

    def __init__(
        self,
        epochs: int = 1000,
        number_of_body_layers: int = 2,
        number_of_head_layers: int = 1,
        number_of_units_per_body_layer: Union[int, List[int]] = 128,
        number_of_units_per_head_layer: Union[int, List[int]] = 128,
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
        number_of_units_per_hidden_layer: Union[int, List[int]] = 128
            Number of units per hidden layer.
        number_of_hidden_layers: int = 3
            Number of graph convolution layer.
        number_of_units_per_hidden_layer: Union[int, List[int]] = 128
            Number of units per hidden layer.
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
        self._number_of_units_per_body_layer = normalize_model_list_parameter(
            number_of_units_per_body_layer,
            number_of_body_layers,
            object_type=int
        )
        self._number_of_units_per_head_layer = normalize_model_list_parameter(
            number_of_units_per_head_layer,
            number_of_head_layers,
            object_type=int
        )

        self._epochs = epochs
        self._use_class_weights = use_class_weights
        self._features_dropout_rate = features_dropout_rate
        self._optimizer = optimizer
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
            number_of_units_per_body_layer=8,
            number_of_units_per_head_layer=8
        )

    def clone(self) -> Type["KipfGCNNodeLabelPrediction"]:
        """Return copy of self."""
        return copy.deepcopy(self)

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return dict(
            number_of_units_per_body_layer=self._number_of_units_per_body_layer,
            number_of_units_per_head_layer=self._number_of_units_per_head_layer,
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
        node_features_sizes: List[int]
    ):
        """Create new GCN model."""
        adjacency_matrix = Input((None,), sparse=True)
        node_features = [
            Input((node_features_size,))
            for node_features_size in node_features_sizes
        ]

        submodules_outputs = []
        submodules_output_sizes = []

        # Build the various submodules, one for each feature.
        for hidden, last_hidden_size in zip(node_features, node_features_sizes):
            # Building the body of the model.
            for units in self._number_of_units_per_body_layer:
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
        for units in self._number_of_units_per_head_layer:
            gcn_hidden = GraphConvolution(
                units=units,
                features_dropout_rate=self._features_dropout_rate,
            )

            gcn_hidden.build(last_hidden_size)

            hidden = gcn_hidden(adjacency_matrix, hidden)
            last_hidden_size = units
        
        # Adding the last layer of the model.
        if self.is_binary_prediction_task():
            units = 1
            activation = "sigmoid"
        elif self.is_multilabel_prediction_task():
            units = graph.get_node_types_number()
            activation = "sigmoid"
        else:
            units = graph.get_node_types_number()
            activation = "softmax"

        gcn_head = GraphConvolution(
            units=units,
            features_dropout_rate=self._features_dropout_rate,
            activation=activation
        )
        gcn_head.build(last_hidden_size)

        output = gcn_head(adjacency_matrix, hidden)

        # Building the the model.

        model = Model(
            inputs=[adjacency_matrix, *node_features],
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

        nodes_number = graph.get_nodes_number()
        node_types_number = graph.get_node_types_number()

        if self._use_class_weights:
            class_weight = {
                node_type_id: nodes_number / count / node_types_number
                for node_type_id, count in graph.get_node_type_id_counts_hashmap().items()
            }
        else:
            class_weight = None

        model = self._build_model(
            graph,
            node_features_sizes=[
                node_feature.shape[1]
                for node_feature in node_features
            ]
        )

        adjacency_matrix = graph_to_sparse_tensor(
            graph,
            use_weights=graph.has_edge_weights() and not self._use_laplacian,
            use_laplacian=self._use_laplacian
        )

        if self.is_multilabel_prediction_task():
            node_labels = graph.get_one_hot_encoded_node_types()
        elif self.is_binary_prediction_task():
            node_labels = graph.get_boolean_node_type_ids()
        else:
            node_labels = graph.get_single_label_node_type_ids()

        self.history = model.fit(
            (adjacency_matrix, *node_features),
            node_labels,
            # This is a known hack to get around limitations from the current
            # implementation that handles the sample weights in TensorFlow.
            sample_weight=pd.Series(
                graph.get_known_node_types_mask().astype(np.float32)
            ),
            epochs=self._epochs,
            verbose=traditional_verbose and self._verbose > 0,
            batch_size=nodes_number,
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
        adjacency_matrix = graph_to_sparse_tensor(
            graph,
            use_weights=graph.has_edge_weights() and not self._use_laplacian,
            use_laplacian=self._use_laplacian
        )
        return self._model.predict(
            (adjacency_matrix, node_features),
            batch_size=graph.get_nodes_number(),
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
    def requires_edge_types() -> bool:
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
    def can_use_edge_types() -> bool:
        """Returns whether the model can optionally use edge types."""
        return False

    def is_using_edge_types(self) -> bool:
        """Returns whether the model is parametrized to use edge types."""
        return False