"""Kipf GCN model for edge prediction."""
from typing import List, Union, Optional, Dict, Any, Type

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import Input, Flatten, Concatenate, Dense, Embedding, Add, GlobalAveragePooling1D  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.models import Model  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error,no-name-in-module

from ensmallen import Graph
from ...layers.tensorflow import GraphConvolution
from ...utils.tensorflow_utils import graph_to_sparse_tensor
from ...utils.normalize_model_structural_parameters import normalize_model_list_parameter
from ..edge_prediction_model import AbstractEdgePredictionModel
from ...sequences.tensorflow_sequences import EdgePredictionSequence, EdgePredictionTrainingSequence
import tensorflow as tf
import copy


class KipfGCNEdgePrediction(AbstractEdgePredictionModel):
    """Kipf GCN model for edge prediction."""

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
        use_node_embedding: bool = True,
        node_embedding_size: int = 50,
        use_node_types: Union[bool, "str"] = "auto",
        node_type_embedding_size: int = 50,
        negative_samples_rate: float = 0.5,
        training_sample_only_edges_with_heterogeneous_node_types: bool = False,
        use_edge_metrics: bool = True,
        random_state: int = 42,
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
        use_node_embedding: bool = True
            Whether to use a node embedding layer to let the model automatically
            learn an embedding of the nodes.
        node_embedding_size: int = 50
            Size of the node embedding.
        use_node_types: Union[bool, str] = "auto"
            Whether to use the node types while training the model.
            By default, automatically uses them if the graph has them.
        node_type_embedding_size: int = 50
            Size of the embedding for the node types.
        negative_samples_rate: float = 0.5
            Factor of negatives to use in every batch.
            For example, with a batch size of 128 and negative_samples_rate equal
            to 0.5, there will be 64 positives and 64 negatives.
        training_sample_only_edges_with_heterogeneous_node_types: bool = False
            Whether to only sample edges between heterogeneous node types.
            This may be useful when training a model to predict between
            two portions in a bipartite graph.
        use_edge_metrics: bool = True
            Whether to use the edge metrics from traditional edge prediction.
            These metrics currently include:
            - Adamic Adar
            - Jaccard Coefficient
            - Resource allocation index
            - Preferential attachment
        random_state: int = 42
            Random state to reproduce the training samples.
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
        self._use_edge_metrics = use_edge_metrics
        self._use_node_embedding = use_node_embedding
        self._node_embedding_size = node_embedding_size
        self._use_node_types = use_node_types
        self._node_type_embedding_size = node_type_embedding_size
        self._negative_samples_rate = negative_samples_rate
        self._training_sample_only_edges_with_heterogeneous_node_types = training_sample_only_edges_with_heterogeneous_node_types
        self._random_state = random_state

        self._verbose = verbose
        self._model = None
        self._training_graph = None
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

    def clone(self) -> Type["KipfGCNEdgePrediction"]:
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
            use_edge_metrics = self._use_edge_metrics,
            use_node_types = self._use_node_types,
            node_type_embedding_size = self._node_type_embedding_size,
            negative_samples_rate = self._negative_samples_rate,
            training_sample_only_edges_with_heterogeneous_node_types = self._training_sample_only_edges_with_heterogeneous_node_types,
            random_state = self._random_state,
        )

    @property
    def use_node_types(self) -> bool:
        """Returns whether the model uses node types."""
        if self._use_node_types == "auto":
            return self._training_graph.has_node_types()
        return self._use_node_types

    def _build_model(
        self,
        graph: Graph,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
    ):
        """Create new GCN model."""
        self._training_graph = graph        
        
        source_nodes = Input((1,), name="sources", dtype=tf.int32)
        destination_nodes = Input((1,), name="destinations", dtype=tf.int32)

        if self.use_node_types:
            source_node_types = Input(
                (graph.get_maximum_multilabel_count(),),
                name="source_node_types",
                dtype=tf.int32
            )
            destination_node_types = Input(
                (graph.get_maximum_multilabel_count(),),
                name="destination_node_types",
                dtype=tf.int32
            )
        else:
            source_node_types = destination_node_types = None
        
        if self._use_edge_metrics:
            edge_metrics = Input(
                (graph.get_number_of_available_edge_metrics(),),
                name="edge_metrics",
                dtype=tf.float32
            )
        else:
            edge_metrics = None

        adjacency_matrix = graph_to_sparse_tensor(
            graph,
            use_weights=graph.has_edge_weights() and not self._use_laplacian,
            use_laplacian=self._use_laplacian
        )

        if node_features is not None:
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
        else:
            node_features_sizes = []
            node_features = []

        if self._use_node_embedding:
            node_embedding = tf.Variable(
                initial_value=np.random.uniform(size=(
                    (graph.get_nodes_number(),
                    self._node_embedding_size)
                )),
                trainable=True,
                validate_shape=True,
                dtype=tf.float32
            )
            node_features.append(node_embedding)
            node_features_sizes.append(self._node_embedding_size)

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

        source_and_destination_features = []

        for nodes in (source_nodes, destination_nodes):
            source_and_destination_features.append(Flatten()(tf.nn.embedding_lookup(
                gcn_node_features,
                ids=nodes
            )))

        if self.use_node_types:
            max_node_types = graph.get_maximum_multilabel_count()
            multilabel = graph.has_multilabel_node_types()
            unknown_node_types = graph.has_unknown_node_types()
            node_types_offset = int(multilabel or unknown_node_types)
            node_type_embedding_layer = Embedding(
                input_dim=graph.get_node_types_number() + node_types_offset,
                output_dim=self._node_type_embedding_size,
                input_length=max_node_types,
                name="node_type_embeddings",
                mask_zero=multilabel or unknown_node_types
            )

            reshaping_dense_layer = Dense(
                units=last_hidden_size,
                activation="relu",
            )

            source_and_destination_features = [
                Add()([
                    reshaping_dense_layer(GlobalAveragePooling1D()(
                        node_type_embedding_layer(node_type_input)
                    )),
                    node_embedding
                ])
                for node_type_input, node_embedding in zip(
                    (source_node_types, destination_node_types),
                    source_and_destination_features
                )
            ]

        if node_type_features is not None and self._use_node_types:
            node_type_features = [
                tf.Variable(
                    initial_value=node_type_feature.astype(np.float32),
                    trainable=False,
                    validate_shape=True,
                    shape=node_type_feature.shape,
                    dtype=np.float32
                )
                for node_type_feature in node_type_features
            ]
            source_and_destination_features = [
                Add()([
                    reshaping_dense_layer(GlobalAveragePooling1D()(
                        tf.nn.embedding_lookup(
                            node_type_feature,
                            ids=node_type_input
                        ),
                    )),
                    node_embedding
                ])
                for node_type_feature in node_type_features
                for node_type_input, node_embedding in zip(
                    (source_node_types, destination_node_types),
                    source_and_destination_features
                )
            ]

        if self._use_edge_metrics:
            source_and_destination_features.append(edge_metrics)

        ffnn_outputs = []

        for hidden in source_and_destination_features:
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

        output = Dense(
            units=1,
            activation="sigmoid"
        )(hidden)

        # Building the the model.

        model = Model(
            inputs=[
                input_layer
                for input_layer in (
                    source_nodes,
                    source_node_types,
                    destination_nodes,
                    destination_node_types,
                    edge_metrics
                )
                if input_layer is not None
            ],
            outputs=output,
            name=self.model_name().replace(" ", "_")
        )

        try:
            if self._optimizer == "LazyAdam":
                import tensorflow_addons as tfa
                optimizer = tfa.optimizers.LazyAdam(0.001)
            else:
                optimizer = self._optimizer
        except:
            optimizer = "adam"

        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics="accuracy"
        )

        return model

    @staticmethod
    def model_name() -> str:
        return "Kipf GCN"

    def _fit(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Return pandas dataframe with training history.

        Parameters
        -----------------------
        graph: Graph,
            The graph whose edges are to be embedded and edge types extracted.
            It can either be an Graph or a list of lists of edges.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[List[np.ndarray]] = None
            The node features to be used in the training of the model.
        edge_features: Optional[List[np.ndarray]] = None
            NOT SUPPORTED

        Returns
        -----------------------
        Dataframe with training history.
        """
        try:
            from tqdm.keras import TqdmCallback
            traditional_verbose = False
        except AttributeError:
            traditional_verbose = True

        if node_features is None and not self._use_node_embedding:
            raise ValueError(
                "Neither node features were provided nor the node "
                "embedding was enabled through the `use_node_embedding` "
                "parameter. If you do not provide node features or use an embedding layer "
                "it does not make sense to use a GCN model."
            )      

        if support is None:
            support = graph  

        model = self._build_model(
            support,
            node_features=node_features,
            node_type_features=node_type_features
        )

        sequence = EdgePredictionTrainingSequence(
            graph,
            use_node_types=self.use_node_types,
            use_edge_metrics=self._use_edge_metrics,
            batch_size=self._batch_size,
            negative_samples_rate=self._negative_samples_rate,
            sample_only_edges_with_heterogeneous_node_types=self._training_sample_only_edges_with_heterogeneous_node_types,
            random_state=self._random_state
        )
        self.history = model.fit(
            sequence,
            steps_per_epoch=sequence.steps_per_epoch,
            epochs=self._epochs,
            verbose=traditional_verbose and self._verbose > 0,
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
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Run predictions on the provided graph."""
        sequence = EdgePredictionSequence(
            graph,
            graph_used_in_training=self._training_graph,
            use_node_types=self.use_node_types,
            use_edge_metrics=self._use_edge_metrics,
            batch_size=self._batch_size,
        )
        return self._model.predict(
            sequence,
            steps=sequence.steps_per_epoch,
            verbose=False
        )

    def _predict(
        self,
        graph: Graph,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Run predictions on the provided graph."""
        return self._predict_proba(
            graph,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features
        ) > 0.5

    @staticmethod
    def requires_edge_weights() -> bool:
        return False

    @staticmethod
    def requires_positive_edge_weights() -> bool:
        return False

    @staticmethod
    def requires_node_types() -> bool:
        return False

    @staticmethod
    def requires_edge_types() -> bool:
        return False

    @staticmethod
    def library_name() -> str:
        """Return name of the model."""
        return "TensorFlow"

    @staticmethod
    def can_use_node_types() -> bool:
        """Returns whether the model can optionally use node types."""
        return True

    def is_using_node_types(self) -> bool:
        """Returns whether the model is parametrized to use node types."""
        return True

    @staticmethod
    def can_use_edge_types() -> bool:
        """Returns whether the model can optionally use edge types."""
        return False

    def is_using_edge_types(self) -> bool:
        """Returns whether the model is parametrized to use edge types."""
        return False

    @staticmethod
    def can_use_edge_weights() -> bool:
        """Returns whether the model can optionally use edge weights."""
        return True

    def is_using_edge_weights(self) -> bool:
        """Returns whether the model is parametrized to use edge weights."""
        return not self._use_laplacian