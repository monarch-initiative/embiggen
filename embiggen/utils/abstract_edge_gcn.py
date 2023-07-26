"""GCN model for edge prediction."""
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from ensmallen import Graph
from tensorflow.keras.layers import (  # pylint: disable=import-error,no-name-in-module
    Activation, Add, Average, Concatenate, Dense, Dot, Flatten, Input, Maximum, Layer,
    Minimum, Multiply, Subtract, BatchNormalization, Dropout)
from tensorflow.keras.models import \
    Model  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error,no-name-in-module
from userinput.utils import must_be_in_set

from embiggen.layers.tensorflow import (ElementWiseL1, ElementWiseL2,
                                        EmbeddingLookup, FlatEmbedding)
from embiggen.utils.abstract_edge_feature import AbstractEdgeFeature
from embiggen.utils.abstract_gcn import AbstractGCN
from embiggen.utils.abstract_models import abstract_class
from embiggen.utils.normalize_model_structural_parameters import \
    normalize_model_list_parameter
from embiggen.utils.number_to_ordinal import number_to_ordinal


@abstract_class
class AbstractEdgeGCN(AbstractGCN):
    """GCN model for edge prediction."""

    def __init__(
        self,
        kernels: Optional[Union[str, List[str]]],
        epochs: int = 1000,
        number_of_graph_convolution_layers: int = 2,
        number_of_units_per_graph_convolution_layers: Union[int,
                                                            List[int]] = 128,
        number_of_ffnn_body_layers: int = 2,
        number_of_ffnn_head_layers: int = 1,
        number_of_units_per_ffnn_body_layer: Union[int, List[int]] = 128,
        number_of_units_per_ffnn_head_layer: Union[int, List[int]] = 128,
        dropout_rate: float = 0.3,
        batch_size: Optional[int] = None,
        apply_norm: bool = False,
        combiner: str = "sum",
        edge_embedding_methods: Optional[Union[List[str],
                                               str]] = "Concatenate",
        optimizer: Union[str, Optimizer] = "adam",
        early_stopping_min_delta: float = 0.0001,
        early_stopping_patience: int = 20,
        reduce_lr_min_delta: float = 0.0001,
        reduce_lr_patience: int = 5,
        early_stopping_monitor: str = "loss",
        early_stopping_mode: str = "min",
        reduce_lr_monitor: str = "loss",
        reduce_lr_mode: str = "min",
        reduce_lr_factor: float = 0.9,
        use_class_weights: bool = True,
        use_edge_metrics: bool = False,
        random_state: int = 42,
        use_node_embedding: bool = False,
        node_embedding_size: int = 50,
        use_node_type_embedding: bool = False,
        node_type_embedding_size: int = 50,
        use_edge_type_embedding: bool = False,
        edge_type_embedding_size: int = 50,
        residual_convolutional_layers: bool = False,
        siamese_node_feature_module: bool = False,
        handling_multi_graph: str = "warn",
        node_feature_names: Optional[Union[str, List[str]]] = None,
        node_type_feature_names: Optional[Union[str, List[str]]] = None,
        edge_type_feature_names: Optional[Union[str, List[str]]] = None,
        verbose: bool = True
    ):
        """Create new Kipf GCN object.

        Parameters
        -------------------------------
        kernels: Optional[Union[str, List[str]]]
            The type of normalization to use. It can either be:
            * "Weights", to just use the graph weights themselves.
            * "Left Normalized Laplacian", for the left normalized Laplacian.
            * "Right Normalized Laplacian", for the right normalized Laplacian.
            * "Symmetric Normalized Laplacian", for the symmetric normalized Laplacian.
            * "Transposed Left Normalized Laplacian", for the transposed left normalized Laplacian.
            * "Transposed Right Normalized Laplacian", for the transposed right normalized Laplacian.
            * "Transposed Symmetric Normalized Laplacian", for the transposed symmetric normalized Laplacian.
            * "Weighted Left Normalized Laplacian", for the weighted left normalized Laplacian.
            * "Weighted Right Normalized Laplacian", for the weighted right normalized Laplacian.
            * "Weighted Symmetric Normalized Laplacian", for the weighted symmetric normalized Laplacian.
            * "Transposed Weighted Left Normalized Laplacian", for the transposed weighted left normalized Laplacian.
            * "Transposed Weighted Right Normalized Laplacian", for the transposed weighted right normalized Laplacian.
            * "Transposed Weighted Symmetric Normalized Laplacian", for the transposed weighted symmetric normalized Laplacian.
        epochs: int = 1000
            Epochs to train the model for.
        number_of_graph_convolution_layers: int = 2
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
        dropout_rate: float = 0.3
            Float between 0 and 1.
            Fraction of the input units to dropout.
        batch_size: Optional[int] = None
            Batch size to use while training the model.
            If None, the batch size will be the number of nodes.
            In all model parametrization that involve a number of graph
            convolution layers, the batch size will be the number of nodes.
        apply_norm: bool = False
            Whether to normalize the output of the convolution operations,
            after applying the level activations.
        combiner: str = "mean"
            A string specifying the reduction op.
            Currently "mean", "sqrtn" and "sum" are supported. 
            "sum" computes the weighted sum of the embedding results for each row.
            "mean" is the weighted sum divided by the total weight.
            "sqrtn" is the weighted sum divided by the square root of the sum of the squares of the weights.
            Defaults to mean.
        edge_embedding_method: str = "Concatenate"
            The edge embedding method to use to put togheter the
            source and destination node features, which includes:
            - Concatenate
            - Average
            - Hadamard
            - L1
            - L2
            - Maximum
            - Minimum
            - Add
            - Subtract
            - Dot
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
        use_node_embedding: bool = False
            Whether to use a node embedding layer to let the model automatically
            learn an embedding of the nodes.
        node_embedding_size: int = 50
            Size of the node embedding.
        use_node_types: Union[bool, str] = "auto"
            Whether to use the node types while training the model.
            By default, automatically uses them if the graph has them.
        node_type_embedding_size: int = 50
            Size of the embedding for the node types.
        training_unbalance_rate: float = 1.0
            Unbalance rate for the training non-existing edges.
        use_edge_metrics: bool = False
            Whether to use the edge metrics from traditional edge prediction.
            These metrics currently include:
            - Adamic Adar
            - Jaccard Coefficient
            - Resource allocation index
            - Preferential attachment
        random_state: int = 42
            Random state to reproduce the training samples.
        use_node_embedding: bool = False
            Whether to use a node embedding layer that is automatically learned
            by the model while it trains. Please do be advised that by using
            a node embedding layer you are making a closed-world assumption,
            and this model will not work on graphs with a different node vocabulary.
        node_embedding_size: int = 50
            Dimension of the node embedding.
        use_node_type_embedding: bool = False
            Whether to use a node type embedding layer that is automatically learned
            by the model while it trains. Please do be advised that by using
            a node type embedding layer you are making a closed-world assumption,
            and this model will not work on graphs with a different node vocabulary.
        node_type_embedding_size: int = 50
            Dimension of the node type embedding.
        use_edge_type_embedding: bool = False
            Whether to use a edge type embedding layer that is automatically learned
            by the model while it trains. Please do be advised that by using
            a edge type embedding layer you are making a closed-world assumption,
            and this model will not work on graphs with a different edge vocabulary.
        edge_type_embedding_size: int = 50
            Dimension of the edge type embedding.
        residual_convolutional_layers: bool = False
            Whether to use residual connections in the convolutional layers.
        siamese_node_feature_module: bool = False
            Whether to use a siamese module for the node features.
        handling_multi_graph: str = "warn"
            How to behave when dealing with multigraphs.
            Possible behaviours are:
            - "warn"
            - "raise"
            - "drop"
        node_feature_names: Optional[Union[str, List[str]]] = None
            Names of the node features.
            This is used as the layer names.
        node_type_feature_names: Optional[Union[str, List[str]]] = None
            Names of the node type features.
            This is used as the layer names.
        edge_type_feature_names: Optional[Union[str, List[str]]] = None
            Names of the edge type features.
            This is used as the layer names.
        verbose: bool = True
            Whether to show loading bars.
        """
        super().__init__(
            kernels=kernels,
            epochs=epochs,
            number_of_graph_convolution_layers=number_of_graph_convolution_layers,
            number_of_units_per_graph_convolution_layers=number_of_units_per_graph_convolution_layers,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            apply_norm=apply_norm,
            combiner=combiner,
            optimizer=optimizer,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_patience=early_stopping_patience,
            reduce_lr_min_delta=reduce_lr_min_delta,
            reduce_lr_patience=reduce_lr_patience,
            early_stopping_monitor=early_stopping_monitor,
            early_stopping_mode=early_stopping_mode,
            reduce_lr_monitor=reduce_lr_monitor,
            reduce_lr_mode=reduce_lr_mode,
            reduce_lr_factor=reduce_lr_factor,
            use_class_weights=use_class_weights,
            use_node_embedding=use_node_embedding,
            node_embedding_size=node_embedding_size,
            use_node_type_embedding=use_node_type_embedding,
            node_type_embedding_size=node_type_embedding_size,
            residual_convolutional_layers=residual_convolutional_layers,
            handling_multi_graph=handling_multi_graph,
            node_feature_names=node_feature_names,
            node_type_feature_names=node_type_feature_names,
            verbose=verbose,
            random_state=random_state
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

        if edge_embedding_methods is None:
            edge_embedding_methods: List[str] = []

        if not isinstance(edge_embedding_methods, list):
            edge_embedding_methods: List[str] = [edge_embedding_methods]

        edge_embedding_methods: List[str] = [
            must_be_in_set(
                edge_embedding_method,
                self.get_available_edge_embedding_methods(),
                "edge embedding method"
            )
            for edge_embedding_method in edge_embedding_methods
        ]

        if isinstance(edge_type_feature_names, str):
            edge_type_feature_names = [edge_type_feature_names]

        self._siamese_node_feature_module: bool = siamese_node_feature_module
        self._edge_embedding_methods: List[str] = edge_embedding_methods
        self._edge_type_feature_names: Optional[List[str]
                                                ] = edge_type_feature_names
        self._use_edge_metrics = use_edge_metrics
        self._use_edge_type_embedding = use_edge_type_embedding
        self._edge_type_embedding_size = edge_type_embedding_size
        self._use_node_types = None

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            AbstractGCN.smoke_test_parameters(),
            number_of_units_per_ffnn_body_layer=8,
            number_of_units_per_ffnn_head_layer=8,
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return dict(
            **AbstractGCN.parameters(self),
            edge_embedding_methods=self._edge_embedding_methods,
            siamese_node_feature_module=self._siamese_node_feature_module,
            number_of_units_per_ffnn_body_layer=self._number_of_units_per_ffnn_body_layer,
            number_of_units_per_ffnn_head_layer=self._number_of_units_per_ffnn_head_layer,
            use_edge_metrics=self._use_edge_metrics,
            use_edge_type_embedding=self._use_edge_type_embedding,
            edge_type_embedding_size=self._edge_type_embedding_size,
            edge_type_feature_names=self._edge_type_feature_names,
        )

    def _get_model_training_output(self, graph: Graph) -> Optional[np.ndarray]:
        """Returns training output tuple."""
        return None

    def _get_model_training_sample_weights(self, graph: Graph) -> Optional[np.ndarray]:
        """Returns training output tuple."""
        return None

    @classmethod
    def get_available_edge_embedding_methods(cls) -> List[str]:
        """Returns a list of the available edge embedding methods."""
        return [
            "Concatenate",
            "Average",
            "Hadamard",
            "Maximum",
            "Minimum",
            "Add",
            "Subtract",
            "L1",
            "L2",
            "Dot",
        ]

    def _build_model(
        self,
        graph: Graph,
        graph_convolution_model: Model,
        edge_type_features: List[np.ndarray],
        edge_features: List[Union[Type[AbstractEdgeFeature], np.ndarray]],
    ) -> Type[Model]:
        """Create new GCN model."""
        assert isinstance(graph, Graph)

        if self._use_node_embedding or self.has_kernels():
            source_nodes = Input(
                (1,),
                name="Sources",
                dtype=tf.int32
            )
            destination_nodes = Input(
                (1,),
                name="Destinations",
                dtype=tf.int32
            )
        else:
            source_nodes = None
            destination_nodes = None

        source_features = []
        source_feature_names = []
        destination_features = []
        destination_feature_names = []
        shared_features = []
        shared_feature_names = []

        if self._use_edge_type_embedding:
            edge_types = Input(
                (1,),
                name="EdgeTypes",
                dtype=tf.int16
            )
            edge_type_embedding = FlatEmbedding(
                vocabulary_size=graph.get_number_of_edge_types(),
                dimension=self._edge_type_embedding_size,
                input_length=1,
                name="EdgeTypesEmbedding"
            )(edge_types)
            shared_feature_names.append("Edge Type Embedding")
            shared_features.append(edge_type_embedding)
        else:
            edge_types = None

        # When the model does not have graph convolution layers,
        # we need to handle differently the eventually requested node embedding
        # layer, which is no longer part of the graph convolution model as we
        # do not have any more necessarily a batch size equal to the number of nodes.
        if self._use_node_embedding and not self.has_convolutional_layers():
            node_embedding = FlatEmbedding(
                vocabulary_size=graph.get_number_of_nodes(),
                dimension=self._node_embedding_size,
                input_length=1,
                name="NodesEmbedding"
            )

            source_features.append(node_embedding(source_nodes))
            source_feature_names.append("Source Node Embedding")
            destination_features.append(node_embedding(destination_nodes))
            destination_feature_names.append("Destination Node Embedding")

        # If the model does not have graph convolution layers,
        # and since we are surely currently building an edge-level model,
        # we need to consider the input features that come from the graph convolution model
        # which in this case are half for the source nodes and half for the destination nodes.

        features = graph_convolution_model.output
        if not isinstance(features, list):
            features = [features]

        if (
            len(features) == 0 and
            not self._use_node_embedding and
            not self._use_edge_type_embedding and
            len(edge_features) == 0 and
            len(edge_type_features) == 0 and
            not self._use_edge_metrics
        ):
            raise ValueError(
                "You have not provided any features to the model."
            )

        if len(features) > 0:
            source_feature_names.append("Source Node Features")
            destination_feature_names.append("Destination Node Features")

            if not self.has_kernels():
                assert len(features) % 2 == 0
                source_related_features = features[:len(features) // 2]
                destination_related_features = features[len(features) // 2:]

                if len(source_related_features) > 1:
                    source_related_features = Concatenate(
                        name="ConcatenatedSourceFeatures",
                        axis=-1
                    )(source_related_features)
                else:
                    source_related_features = source_related_features[0]

                if len(destination_related_features) > 1:
                    destination_related_features = Concatenate(
                        name="ConcatenatedDestinationFeatures",
                        axis=-1
                    )(destination_related_features)
                else:
                    destination_related_features = destination_related_features[0]

                # The features have shape (batch size, number of features),
                # with the batch size generally SIGNIFICANTLY LESS than the number of nodes.
                # We do not need to execute an embedding lookup here,
                # as the features are already queries in the training
                # sequence.
                source_features.append(source_related_features)
                destination_features.append(destination_related_features)
            else:
                if len(features) > 1:
                    features = Concatenate(
                        name="ConcatenatedNodeFeatures",
                        axis=-1
                    )(features)
                else:
                    features = features[0]

                # The features have shape (number of nodes, number of features)
                # and we need to extract the features for the source and destination nodes.
                # For this reason, we need to query the features using the source and destination nodes
                # using an embedding lookup.
                source_features.append(EmbeddingLookup(name=f"{source_nodes.name}Features")((
                    source_nodes,
                    features
                )))
                destination_features.append(EmbeddingLookup(name=f"{destination_nodes.name}Features")((
                    destination_nodes,
                    features
                )))

        edge_feature_inputs = []

        if self._use_edge_metrics:
            edge_metrics = Input(
                (graph.get_number_of_available_edge_metrics(),),
                name="Edge Metrics",
                dtype=tf.float32
            )
            edge_feature_inputs.append(edge_metrics)
            shared_feature_names.append("EdgeMetrics")
            shared_features.append(edge_metrics)
        else:
            edge_metrics = None

        edge_feature_names = []

        i = 0
        for edge_feature in edge_features:
            if isinstance(edge_feature, AbstractEdgeFeature):
                for feature_name in edge_feature.get_feature_dictionary_keys():
                    new_feature_name = f"{edge_feature.get_feature_name()}{feature_name}"
                    adjusted_new_feature_name = new_feature_name
                    counter = 0
                    while adjusted_new_feature_name in edge_feature_names:
                        counter += 1
                        adjusted_new_feature_name = new_feature_name + \
                            str(counter)

                    edge_feature_names.append(adjusted_new_feature_name)
            else:
                if len(edge_features) > 1:
                    ordinal = number_to_ordinal(i+1)
                else:
                    ordinal = ""

                edge_feature_names.append(
                    f"{ordinal}EdgeFeature"
                )
                i += 1

        shared_feature_names.extend(edge_feature_names)

        i = 0
        for edge_feature in edge_features:
            if isinstance(edge_feature, AbstractEdgeFeature):
                for feature_name, feature_shape in edge_feature.get_feature_dictionary_shapes().items():
                    dimension = feature_shape[0]
                    for additional_dimension in feature_shape[1:]:
                        dimension *= additional_dimension
                    edge_feature_input = Input(
                        shape=(dimension,),
                        name=edge_feature_names[i],
                    )
                    hidden = edge_feature_input

                    # If we are dealing with a feature that is not flat:
                    if len(feature_shape) > 1:
                        # We flatten it.
                        hidden = Flatten(
                            name=f"{edge_feature_names[i]}Flatten"
                        )(hidden)

                    edge_feature_inputs.append(edge_feature_input)
                    shared_features.append(hidden)
                    i += 1
            elif isinstance(edge_feature, np.ndarray):
                edge_feature_input = Input(
                    shape=edge_feature.shape[1:],
                    name=edge_feature_names[i],
                )
                edge_feature_inputs.append(edge_feature_input)
                shared_features.append(edge_feature_input)
                i += 1
            else:
                raise NotImplementedError(
                    f"Edge feature of type {type(edge_feature)} is not supported."
                    "Please provide an instance of AbstractEdgeFeature or a numpy array."
                )

        edge_type_feature_names: List[str] = []
        if self._edge_type_feature_names is None:
            if len(edge_type_features) > 1:
                edge_type_feature_names = [
                    f"{number_to_ordinal(i+1)} Edge Type Feature"
                    for i in range(len(edge_type_features))
                ]
            elif len(edge_type_features) == 1:
                edge_type_feature_names = ["Edge Type Feature"]
        else:
            edge_type_feature_names = self._edge_type_feature_names

        if len(edge_type_feature_names) != len(edge_type_features):
            raise ValueError(
                f"You have provided {len(edge_type_feature_names)} "
                f"edge type feature names but you have provided {len(edge_type_features)} "
                f"edge type features to the model."
            )

        assert edge_type_features is not None
        assert edge_type_feature_names is not None
        assert len(edge_type_features) == len(edge_type_feature_names)

        for edge_type_feature, edge_type_feature_name in zip(edge_type_features, edge_type_feature_names):
            if isinstance(edge_type_feature, pd.DataFrame):
                edge_type_feature = edge_type_feature.values
            if isinstance(edge_type_feature, np.ndarray):
                edge_type_feature_input = Input(
                    shape=edge_type_feature.shape[1:],
                    name=edge_type_feature_name,
                )
                edge_feature_inputs.append(edge_type_feature_input)
                shared_features.append(edge_type_feature_input)
            else:
                raise NotImplementedError(
                    f"Edge type feature of type {type(edge_type_feature)} is not supported."
                    "Please provide an instance of numpy array or a pandas DataFrame."
                )

        shared_feature_names.extend(edge_type_feature_names)

        ffnn_outputs = []

        if len(source_features) > 0 and len(destination_features) > 0:
            if self._siamese_node_feature_module:
                siamese_layers = []
                for source_feature_name in source_feature_names:
                    partial_siamese_layers = []
                    source_feature_name = source_feature_name.replace(
                        "Source", "").replace(" ", "")
                    for i, units in enumerate(self._number_of_units_per_ffnn_body_layer):
                        assert isinstance(units, int)
                        if len(self._number_of_units_per_ffnn_body_layer) > 1:
                            ordinal = number_to_ordinal(i+1)
                        else:
                            ordinal = ""
                        layer_name = f"{ordinal}{source_feature_name}SiameseDense"
                        self._add_layer_name(layer_name)
                        layer = Dense(
                            units=units,
                            activation="relu",
                            name=layer_name
                        )
                        assert issubclass(
                            type(layer), Layer), f"Expected a layer, but found {type(layer)}."
                        partial_siamese_layers.append(layer)
                    if len(partial_siamese_layers) > 0:
                        partial_siamese_layers.append(BatchNormalization())
                    siamese_layers.append(partial_siamese_layers)
                for zip_iter in (
                    zip(source_features, siamese_layers),
                    zip(destination_features, siamese_layers)
                ):
                    for hidden, partial_siamese_layers in zip_iter:
                        this_ffnn_output = []
                        # Building the body of the model.
                        for siamese_layer in partial_siamese_layers:
                            assert issubclass(type(
                                siamese_layer), Layer), f"Expected a layer, but found {type(siamese_layer)}."
                            hidden = siamese_layer(hidden)
                        this_ffnn_output.append(hidden)
                    ffnn_outputs.append(this_ffnn_output)
            else:
                for hiddens, feature_names in zip(
                    (source_features, destination_features),
                    (source_feature_names, destination_feature_names)
                ):
                    this_ffnn_output = []
                    assert len(hiddens) == len(feature_names)
                    assert len(set(feature_names)) == len(
                        feature_names), f"Expected unique feature names, but found duplicates: {feature_names}."
                    for hidden, feature_name in zip(hiddens, feature_names):
                        # Building the body of the model.
                        feature_name = feature_name.replace(" ", "")
                        for i, units in enumerate(self._number_of_units_per_ffnn_body_layer):
                            assert isinstance(units, int)
                            assert not isinstance(hidden, list)
                            if len(self._number_of_units_per_ffnn_body_layer) > 1:
                                ordinal = number_to_ordinal(i+1)
                            else:
                                ordinal = ""
                            layer_name = f"{ordinal}{feature_name}Dense"
                            self._add_layer_name(layer_name)
                            hidden = Dense(
                                units=units,
                                activation="relu",
                                name=layer_name
                            )(hidden)
                        if len(self._number_of_units_per_ffnn_body_layer) > 0:
                            hidden = BatchNormalization()(hidden)
                        this_ffnn_output.append(hidden)
                    ffnn_outputs.append(this_ffnn_output)
            source_feature_hidden = ffnn_outputs[0]
            destination_feature_hidden = ffnn_outputs[1]
        else:
            source_feature_hidden = []
            destination_feature_hidden = []

        if len(shared_features) > 0:
            this_ffnn_output = []
            for hidden, feature_name in zip(shared_features, shared_feature_names):
                # Building the body of the model.
                feature_name = feature_name.replace(" ", "")
                for i, units in enumerate(self._number_of_units_per_ffnn_body_layer):
                    assert isinstance(units, int)
                    assert not isinstance(hidden, list)
                    if len(self._number_of_units_per_ffnn_body_layer) > 1:
                        ordinal = number_to_ordinal(i+1)
                    else:
                        ordinal = ""
                    layer_name = f"{ordinal}{feature_name}Dense"
                    self._add_layer_name(layer_name)
                    hidden = Dense(
                        units=units,
                        activation="relu",
                        name=layer_name
                    )(hidden)
                this_ffnn_output.append(hidden)
            shared_feature_hidden = this_ffnn_output
            ffnn_outputs.append(this_ffnn_output)
        else:
            shared_feature_hidden = []

        if len(source_feature_hidden) > 1:
            source_feature_hidden = Concatenate(
                name="ConcatenatedProcessedSourceFeatures",
                axis=-1
            )(source_feature_hidden)
        elif len(source_feature_hidden) == 1:
            source_feature_hidden = source_feature_hidden[0]
        else:
            source_feature_hidden = None

        if len(destination_feature_hidden) > 1:
            destination_feature_hidden = Concatenate(
                name="ConcatenatedProcessedDestinationFeatures",
                axis=-1
            )(destination_feature_hidden)
        elif len(destination_feature_hidden) == 1:
            destination_feature_hidden = destination_feature_hidden[0]
        else:
            destination_feature_hidden = None

        edge_embedding_layers = []

        assert (source_feature_hidden is None) == (destination_feature_hidden is None), (
            "Source and destination feature hidden must be both None or both not None. "
            f"Source feature hidden: {source_feature_hidden}. "
            f"Destination feature hidden: {destination_feature_hidden}."
        )

        if len(self._edge_embedding_methods) > 0 and source_feature_hidden is None and destination_feature_hidden is None:
            raise ValueError(
                "You have provided edge embedding methods but you have not provided "
                "any source or destination node features - It is unclear how to embed the edges. "
            )

        if len(self._edge_embedding_methods) == 0 and source_feature_hidden is not None and destination_feature_hidden is not None:
            raise ValueError(
                "You have provided source and destination node features but you have not provided "
                "any edge embedding methods - It is unclear how to embed the edges. "
            )

        for edge_embedding_method in self._edge_embedding_methods:
            if edge_embedding_method == "Concatenate":
                edge_embedding_layers.append(Concatenate(
                    name="NodeConcatenate",
                    axis=-1
                )([
                    source_feature_hidden,
                    destination_feature_hidden
                ]))
            elif edge_embedding_method == "Average":
                edge_embedding_layers.append(Average(
                    name="NodeAverage"
                )([
                    source_feature_hidden,
                    destination_feature_hidden
                ]))
            elif edge_embedding_method == "Hadamard":
                edge_embedding_layers.append(Multiply(
                    name="NodeHadamard"
                )([
                    source_feature_hidden,
                    destination_feature_hidden
                ]))
            elif edge_embedding_method == "Maximum":
                edge_embedding_layers.append(Maximum(
                    name="NodeMaximum"
                )([
                    source_feature_hidden,
                    destination_feature_hidden
                ]))
            elif edge_embedding_method == "Minimum":
                edge_embedding_layers.append(Minimum(
                    name="NodeMinimum"
                )([
                    source_feature_hidden,
                    destination_feature_hidden
                ]))
            elif edge_embedding_method == "Add":
                edge_embedding_layers.append(Add(
                    name="NodeAdd"
                )([
                    source_feature_hidden,
                    destination_feature_hidden
                ]))
            elif edge_embedding_method == "Subtract":
                edge_embedding_layers.append(Subtract(
                    name="NodeSubtract"
                )([
                    source_feature_hidden,
                    destination_feature_hidden
                ]))
            elif edge_embedding_method == "L1":
                edge_embedding_layers.append(ElementWiseL1(
                    name="NodeL1"
                )([
                    source_feature_hidden,
                    destination_feature_hidden
                ]))
            elif edge_embedding_method == "L2":
                edge_embedding_layers.append(ElementWiseL2(
                    name="NodeL2"
                )([
                    source_feature_hidden,
                    destination_feature_hidden
                ]))
            elif edge_embedding_method == "Dot":
                edge_embedding_layers.append(Dot(
                    name="NodeDot",
                    normalize=True,
                    axes=-1
                )([
                    source_feature_hidden,
                    destination_feature_hidden
                ]))

        if len(edge_embedding_layers) > 1:
            edge_embedding_layer_concatenation = Concatenate(
                name="EdgeEmbeddings",
                axis=-1
            )(edge_embedding_layers)
        elif len(edge_embedding_layers) == 1:
            edge_embedding_layer_concatenation = edge_embedding_layers[0]
        else:
            edge_embedding_layer_concatenation = None

        shared_feature_hidden = [
            BatchNormalization(

            )(shared_feature_hidden)
            for shared_feature_hidden in shared_feature_hidden
        ]

        if len(shared_feature_hidden) > 0 and edge_embedding_layer_concatenation is not None:
            hidden = Concatenate(
                name="EdgeFeatures",
                axis=-1
            )([
                edge_embedding_layer_concatenation,
                *shared_feature_hidden
            ])
        elif len(shared_feature_hidden) > 0:
            hidden = Concatenate(
                name="EdgeFeatures",
                axis=-1
            )(shared_feature_hidden)
        elif edge_embedding_layer_concatenation is not None:
            hidden = edge_embedding_layer_concatenation

        hidden = Dropout(
            rate=self._dropout_rate,
            name="EdgeFeaturesDropout"
        )(hidden)

        # Building the head of the model.
        for i, units in enumerate(self._number_of_units_per_ffnn_head_layer):
            if len(self._number_of_units_per_ffnn_head_layer) > 1:
                ordinal = number_to_ordinal(i+1)
            else:
                ordinal = ""
            hidden = Dense(
                units=units,
                activation="relu",
                name=f"{ordinal}HeadDense"
            )(hidden)

        output_activation = self.get_output_activation_name()

        if not isinstance(output_activation, str) or issubclass(type(output_activation), Activation):
            raise ValueError(
                f"The provided output activation {output_activation} "
                "is not a valid activation function name."
            )

        output = Dense(
            units=self.get_output_classes(graph),
            activation=output_activation,
            name="Output"
        )(hidden)

        inputs = [
            input_layer
            for input_layer in (
                source_nodes,
                destination_nodes,
                edge_types,
                *edge_feature_inputs,
                *graph_convolution_model.inputs,
            )
            if input_layer is not None
        ]

        # Building the the model.
        model = Model(
            inputs=inputs,
            outputs=output,
            name=self.model_name().replace(" ", "_")
        )

        model.compile(
            loss=self.get_loss_name(),
            optimizer=self._optimizer,
            metrics="accuracy"
        )

        return model

    @classmethod
    def requires_node_types(cls) -> bool:
        return False

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        return True

    def is_edge_level_task(self) -> bool:
        """Returns whether the task is edge level."""
        return True

    def is_using_node_types(self) -> bool:
        """Returns whether the model is parametrized to use node types."""
        return self._use_node_type_embedding or super().is_using_node_types()
