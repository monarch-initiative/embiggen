"""GCN model for node-label prediction."""
from typing import List, Union, Optional, Dict, Type, Tuple

import numpy as np
from tensorflow.keras.layers import Dense  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.models import Model  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error,no-name-in-module

from ensmallen import Graph
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from embiggen.utils.abstract_gcn import AbstractGCN
from embiggen.utils.normalize_model_structural_parameters import normalize_model_list_parameter
from embiggen.node_label_prediction.node_label_prediction_model import AbstractNodeLabelPredictionModel


class GCNNodeLabelPrediction(AbstractGCN, AbstractNodeLabelPredictionModel):
    """GCN model for node-label prediction."""

    def __init__(
        self,
        epochs: int = 1000,
        number_of_graph_convolution_layers: int = 2,
        number_of_head_layers: int = 1,
        number_of_units_per_graph_convolution_layers: Union[int, List[int]] = 128,
        number_of_units_per_head_layer: Union[int, List[int]] = 128,
        dropout_rate: float = 0.1,
        optimizer: Union[str, Optimizer] = "adam",
        early_stopping_min_delta: float = 0.0001,
        early_stopping_patience: int = 30,
        reduce_lr_min_delta: float = 0.001,
        reduce_lr_patience: int = 20,
        early_stopping_monitor: str = "loss",
        early_stopping_mode: str = "min",
        reduce_lr_monitor: str = "loss",
        reduce_lr_mode: str = "min",
        reduce_lr_factor: float = 0.9,
        use_class_weights: bool = True,
        use_simmetric_normalized_laplacian: bool = True,
        use_node_embedding: bool = True,
        node_embedding_size: int = 50,
        handling_multi_graph: str = "warn",
        node_feature_names: Optional[List[str]] = None,
        node_type_feature_names: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """Create new Kipf GCN object.

        Parameters
        -------------------------------
        epochs: int = 1000
            Epochs to train the model for.
        batch_size: int = 2**10
            Batch size to train the model.
        number_of_units_per_hidden_layer: Union[int, List[int]] = 128
            Number of units per hidden layer.
        number_of_hidden_layers: int = 3
            Number of graph convolution layer.
        number_of_units_per_hidden_layer: Union[int, List[int]] = 128
            Number of units per hidden layer.
        dropout_rate: float = 0.3
            Float between 0 and 1.
            Fraction of the input units to dropout.
        optimizer: str = "Adam"
            The optimizer to use while training the model.
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
        use_simmetric_normalized_laplacian: bool = True
            Whether to use laplacian transform before training on the graph.
        use_node_embedding: bool = True
            Whether to use a node embedding layer that is automatically learned
            by the model while it trains. Please do be advised that by using
            a node embedding layer you are making a closed-world assumption,
            and this model will not work on graphs with a different node vocabulary.
        node_embedding_size: int = 50
            Dimension of the node embedding.
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
        verbose: bool = True
            Whether to show loading bars.
        """
        AbstractNodeLabelPredictionModel.__init__(self)
        AbstractGCN.__init__(
            self,
            epochs=epochs,
            number_of_graph_convolution_layers=number_of_graph_convolution_layers,
            number_of_units_per_graph_convolution_layers=number_of_units_per_graph_convolution_layers,
            dropout_rate=dropout_rate,
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
            handling_multi_graph=handling_multi_graph,
            node_feature_names=node_feature_names,
            node_type_feature_names=node_type_feature_names,
            verbose=verbose,
        )
        self._number_of_units_per_head_layer = normalize_model_list_parameter(
            number_of_units_per_head_layer,
            number_of_head_layers,
            object_type=int,
            can_be_empty=True
        )

    def _build_model(
        self,
        graph: Graph,
        graph_convolution_model: Model,
        edge_features: Optional[List[np.ndarray]] = None,
    ):
        """Create new GCN model."""
        hidden = graph_convolution_model.output

        # Building the head of the model.
        for units in self._number_of_units_per_head_layer:
            hidden = Dense(
                units=units,
                activation="ReLU"
            )(hidden)

        output = Dense(
            units=self.get_output_classes(graph),
            activation=self.get_output_activation_name(),
            name="Output"
        )(hidden)

        # Building the the model.
        model = Model(
            inputs=graph_convolution_model.inputs,
            outputs=output,
            name=self.model_name()
        )

        model.compile(
            loss=self.get_loss_name(),
            optimizer=self._optimizer,
            weighted_metrics="accuracy"
        )

        return model

    def get_output_classes(self, graph:Graph) ->int:
        """Returns number of output classes."""
        return graph.get_node_types_number()

    def _get_class_weights(self, graph: Graph) -> Dict[int, float]:
        """Returns dictionary with class weights."""
        nodes_number = graph.get_nodes_number()
        node_types_number = graph.get_node_types_number()
        return {
            node_type_id: nodes_number / count / node_types_number
            for node_type_id, count in graph.get_node_type_id_counts_hashmap().items()
        }

    def _get_model_training_input(
        self,
        graph: Graph,
        support: Graph,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> Tuple[Union[np.ndarray, Type[Sequence]]]:
        """Returns training input tuple."""
        return (
            self._graph_to_kernel(support),
            *(
                ()
                if node_features is None
                else node_features
            ),
            *(
                (graph.get_node_ids(),)
                if self._use_node_embedding
                else ()
            )
        )

    def _get_model_training_output(
        self,
        graph: Graph,
    ) -> Optional[np.ndarray]:
        """Returns training output tuple."""
        if self.is_multilabel_prediction_task():
            return graph.get_one_hot_encoded_node_types()
        if self.is_binary_prediction_task():
            return graph.get_boolean_node_type_ids()
        return graph.get_single_label_node_type_ids()

    def _get_model_training_sample_weights(
        self,
        graph: Graph,
    ) -> Optional[np.ndarray]:
        """Returns training output tuple."""
        return graph.get_known_node_types_mask().astype(tf.float32)

    def _get_model_prediction_input(
        self,
        graph: Graph,
        support: Graph,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> Tuple[Union[np.ndarray, Type[Sequence]]]:
        """Returns dictionary with class weights."""
        return self._get_model_training_input(
            graph,
            support,
            node_features,
            node_type_features,
            edge_features
        )

    @staticmethod
    def requires_edge_types() -> bool:
        return False

    @staticmethod
    def can_use_edge_types() -> bool:
        """Returns whether the model can optionally use edge types."""
        return False

    def is_using_edge_types(self) -> bool:
        """Returns whether the model is parametrized to use edge types."""
        return False

    @staticmethod
    def model_name() -> str:
        return "GCN"