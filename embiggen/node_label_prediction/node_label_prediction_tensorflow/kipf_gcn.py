"""GCN model for node-label prediction."""
from typing import List, Union, Optional, Dict, Any
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error,no-name-in-module

from embiggen.node_label_prediction.node_label_prediction_tensorflow.gcn import GCNNodeLabelPrediction


class KipfGCNNodeLabelPrediction(GCNNodeLabelPrediction):
    """Kipf GCN model for node-label prediction."""

    def __init__(
        self,
        epochs: int = 1000,
        number_of_graph_convolution_layers: int = 2,
        number_of_head_layers: int = 1,
        number_of_units_per_graph_convolution_layers: Union[int, List[int]] = 128,
        number_of_units_per_head_layer: Union[int, List[int]] = 128,
        dropout_rate: float = 0.1,
        apply_norm: bool = False,
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
        random_state: int = 42,
        use_node_embedding: bool = False,
        node_embedding_size: int = 50,
        handling_multi_graph: str = "warn",
        node_feature_names: Optional[List[str]] = None,
        node_type_feature_names: Optional[List[str]] = None,
        verbose: bool = False
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
        random_state: int = 42
            The random state to use to reproduce the training.
        use_node_embedding: bool = False
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
        verbose: bool = False
            Whether to show loading bars.
        """
        super().__init__(
            epochs=epochs,
            number_of_graph_convolution_layers=number_of_graph_convolution_layers,
            number_of_head_layers=number_of_head_layers,
            number_of_units_per_graph_convolution_layers=number_of_units_per_graph_convolution_layers,
            number_of_units_per_head_layer=number_of_units_per_head_layer,
            dropout_rate=dropout_rate,
            apply_norm=apply_norm,
            combiner="sum",
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
            random_state=random_state,
            use_simmetric_normalized_laplacian=True,
            use_node_embedding=use_node_embedding,
            node_embedding_size=node_embedding_size,
            handling_multi_graph=handling_multi_graph,
            node_feature_names=node_feature_names,
            node_type_feature_names=node_type_feature_names,
            verbose=verbose,
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        removed = [
            "combiner",
        ]
        return dict(
            **{
                key: value
                for key, value in super().parameters().items()
                if key not in removed
            }
        )

    @classmethod
    def model_name(cls) -> str:
        return "Kipf GCN"