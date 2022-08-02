"""GCN model for edge prediction."""
from typing import List, Union, Optional, Dict, Any, Type, Tuple

import numpy as np
from tensorflow.keras.layers import Input, Concatenate, Dense, Average, Multiply, Maximum, Minimum, Add, Subtract, Dot  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.models import Model  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error,no-name-in-module

from ensmallen import Graph
from tensorflow.keras.utils import Sequence
from embiggen.utils.normalize_model_structural_parameters import normalize_model_list_parameter
from embiggen.sequences.tensorflow_sequences import GCNEdgePredictionSequence
from embiggen.utils.abstract_models import abstract_class
from embiggen.utils.abstract_gcn import AbstractGCN
from embiggen.utils.number_to_ordinal import number_to_ordinal
from embiggen.layers.tensorflow import EmbeddingLookup, ElementWiseL1, ElementWiseL2
import tensorflow as tf


@abstract_class
class AbstractEdgeGCN(AbstractGCN):
    """GCN model for edge prediction."""

    def __init__(
        self,
        epochs: int = 1000,
        number_of_graph_convolution_layers: int = 2,
        number_of_units_per_graph_convolution_layers: Union[int, List[int]] = 128,
        number_of_ffnn_body_layers: int = 2,
        number_of_ffnn_head_layers: int = 1,
        number_of_units_per_ffnn_body_layer: Union[int, List[int]] = 128,
        number_of_units_per_ffnn_head_layer: Union[int, List[int]] = 128,
        dropout_rate: float = 0.3,
        apply_norm: bool = False,
        combiner: str ="sum",
        edge_embedding_method: str = "Concatenate",
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
        use_simmetric_normalized_laplacian: bool = True,
        use_node_embedding: bool = False,
        node_embedding_size: int = 50,
        use_node_type_embedding: bool = False,
        node_type_embedding_size: int = 50,
        handling_multi_graph: str = "warn",
        node_feature_names: Optional[List[str]] = None,
        node_type_feature_names: Optional[List[str]] = None,
        edge_feature_names: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """Create new Kipf GCN object.

        Parameters
        -------------------------------
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
            - Concatenation
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
        training_sample_only_edges_with_heterogeneous_node_types: bool = False
            Whether to only sample edges between heterogeneous node types.
            This may be useful when training a model to predict between
            two portions in a bipartite graph.
        use_edge_metrics: bool = False
            Whether to use the edge metrics from traditional edge prediction.
            These metrics currently include:
            - Adamic Adar
            - Jaccard Coefficient
            - Resource allocation index
            - Preferential attachment
        random_state: int = 42
            Random state to reproduce the training samples.
        use_simmetric_normalized_laplacian: bool = True
            Whether to use laplacian transform before training on the graph.
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
        handling_multi_graph: str = "warn"
            How to behave when dealing with multigraphs.
            Possible behaviours are:
            - "warn"
            - "raise"
            - "drop"
        node_feature_names: Optional[List[str]] = None
            Names of the node features.
            This is used as the layer names.
        node_type_feature_names: Optional[List[str]] = None
            Names of the node type features.
            This is used as the layer names.
        edge_feature_names: Optional[List[str]] = None
            Names of the edge features.
            This is used as the layer names.
        verbose: bool = True
            Whether to show loading bars.
        """
        AbstractGCN.__init__(
            self,
            epochs=epochs,
            number_of_graph_convolution_layers=number_of_graph_convolution_layers,
            number_of_units_per_graph_convolution_layers=number_of_units_per_graph_convolution_layers,
            dropout_rate=dropout_rate,
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
            use_simmetric_normalized_laplacian=use_simmetric_normalized_laplacian,
            use_node_embedding=use_node_embedding,
            node_embedding_size=node_embedding_size,
            use_node_type_embedding=use_node_type_embedding,
            node_type_embedding_size=node_type_embedding_size,
            handling_multi_graph=handling_multi_graph,
            node_feature_names=node_feature_names,
            node_type_feature_names=node_type_feature_names,
            verbose=verbose,
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

        self._edge_embedding_method = edge_embedding_method
        self._use_edge_metrics = use_edge_metrics
        self._edge_feature_names = edge_feature_names
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
            number_of_units_per_ffnn_body_layer=self._number_of_units_per_ffnn_body_layer,
            number_of_units_per_ffnn_head_layer=self._number_of_units_per_ffnn_head_layer,
            use_edge_metrics=self._use_edge_metrics,
        )

    def _get_model_prediction_input(
        self,
        graph: Graph,
        support: Graph,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> Tuple[Union[np.ndarray, Type[Sequence]]]:
        """Returns dictionary with class weights."""
        return GCNEdgePredictionSequence(
            graph,
            support=support,
            kernel=self.convert_graph_to_kernel(support),
            node_features=node_features,
            return_node_ids=self._use_node_embedding,
            return_node_types=self.is_using_node_types(),
            node_type_features=node_type_features,
            use_edge_metrics=self._use_edge_metrics,
            edge_features=edge_features
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
            "Concatenation",
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
        edge_features: Optional[List[np.ndarray]] = None,
    ):
        """Create new GCN model."""
        # If we make the node embedding, we are making a closed world assumption,
        # while without we do not have them we can keep an open world assumption.
        # Basically, without node embedding we can have different vocabularies,
        # while with node embedding the vocabulary becomes fixed.
        nodes_number = graph.get_number_of_nodes() if self._use_node_embedding else None

        source_nodes = Input(
            (1,),
            batch_size=nodes_number,
            name="Sources",
            dtype=tf.int32
        )
        destination_nodes = Input(
            (1,),
            batch_size=nodes_number,
            name="Destinations",
            dtype=tf.int32
        )

        feature_names = [
            node_input_layer.name
            for node_input_layer in (
                source_nodes,
                destination_nodes
            )
        ]

        source_and_destination_features = [
            EmbeddingLookup(name=f"{feature_name}Features")((
                node_input_layer,
                graph_convolution_model.output
            ))
            for node_input_layer, feature_name in zip((
                source_nodes,
                destination_nodes
            ), feature_names)
        ]

        if self._use_edge_metrics:
            edge_metrics = Input(
                (graph.get_number_of_available_edge_metrics(),),
                batch_size=nodes_number,
                name="Edge Metrics",
                dtype=tf.float32
            )
            feature_names.append("EdgeMetrics")
            source_and_destination_features.append(edge_metrics)
        else:
            edge_metrics = None

        edge_feature_inputs = []
        if edge_features is None:
            edge_features = []
        edge_feature_names = self._edge_feature_names
        if edge_feature_names is None:
            edge_feature_names = [
                f"{number_to_ordinal(i+1)}EdgeFeature"
                for i in range(len(edge_features))
            ]
        if len(edge_feature_names) != len(edge_features):
            raise ValueError(
                f"You have provided {len(edge_feature_names)} "
                f"edge feature names but you have provided {len(edge_features)} "
                "edge features to the model."
            )
        for edge_feature, feature_name in zip(edge_features, edge_feature_names):
            feature_names.append(feature_name)
            edge_feature_input = Input(
                shape=edge_feature.shape[1:],
                batch_size=nodes_number,
                name=feature_name,
            )
            edge_feature_inputs.append(edge_feature_input)
            source_and_destination_features.append(edge_feature_input)

        ffnn_outputs = []

        for hidden, feature_name in zip(source_and_destination_features, feature_names):
            # Building the body of the model.
            for i, units in enumerate(self._number_of_units_per_ffnn_body_layer):
                hidden = Dense(
                    units=units,
                    activation="relu",
                    name=f"{number_to_ordinal(i+1)}{feature_name}Dense"
                )(hidden)

            ffnn_outputs.append(hidden)

        source_and_destination_features = ffnn_outputs[:2]
        other_features = ffnn_outputs[2:]

        if self._edge_embedding_method == "Concatenate":
            hidden = Concatenate(
                name="NodeConcatenation",
                axis=-1
            )(source_and_destination_features)
        elif self._edge_embedding_method == "Average":
            hidden = Average(
                name="NodeAverage"
            )(source_and_destination_features)
        elif self._edge_embedding_method == "Hadamard":
            hidden = Multiply(
                name="NodeHadamard"
            )(source_and_destination_features)
        elif self._edge_embedding_method == "Maximum":
            hidden = Maximum(
                name="NodeMaximum"
            )(source_and_destination_features)
        elif self._edge_embedding_method == "Minimum":
            hidden = Minimum(
                name="NodeMinimum"
            )(source_and_destination_features)
        elif self._edge_embedding_method == "Add":
            hidden = Add(
                name="NodeAdd"
            )(source_and_destination_features)
        elif self._edge_embedding_method == "Subtract":
            hidden = Subtract(
                name="NodeSubtract"
            )(source_and_destination_features)
        elif self._edge_embedding_method == "L1":
            hidden = ElementWiseL1(
                name="NodeL1"
            )(source_and_destination_features)
        elif self._edge_embedding_method == "L2":
            hidden = ElementWiseL2(
                name="NodeL2"
            )(source_and_destination_features)
        elif self._edge_embedding_method == "Dot":
            hidden = Dot(
                name="NodeDot",
                normalize=True,
                axes=-1
            )(source_and_destination_features)

        if len(other_features) > 0:
            hidden = Concatenate(
                name="EdgeFeatures",
                axis=-1
            )([
                hidden,
                *other_features
            ])

        # Building the head of the model.
        for i, units in enumerate(self._number_of_units_per_ffnn_head_layer):
            hidden = Dense(
                units=units,
                activation="relu",
                name=f"{number_to_ordinal(i+1)}HeadDense"
            )(hidden)

        output = Dense(
            units=self.get_output_classes(graph),
            activation=self.get_output_activation_name(),
            name="Output"
        )(hidden)

        inputs=[
            input_layer
            for input_layer in (
                source_nodes,
                destination_nodes,
                edge_metrics,
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

    def _predict_proba(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Run predictions on the provided graph."""
        predictions = super()._predict_proba(
            graph,
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features
        )
        # The model will padd the predictions with a few zeros
        # in order to run the GCN portion of the model, which
        # always requires a batch size equal to the nodes number.
        return predictions[:graph.get_number_of_directed_edges()]

    @classmethod
    def requires_node_types(cls) -> bool:
        return False

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        return True

    def is_using_node_types(self) -> bool:
        """Returns whether the model is parametrized to use node types."""
        return self._use_node_type_embedding
