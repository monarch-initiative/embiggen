"""GraphSAGE model for edge prediction."""
from typing import List, Union, Optional, Type, Dict, Any
from tensorflow.keras.optimizers import Optimizer
from embiggen.edge_prediction.edge_prediction_tensorflow.gcn import GCNEdgePrediction


class GraphSAGEEdgePrediction(GCNEdgePrediction):
    """GraphSAGE model for edge prediction."""

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
        edge_embedding_method: str = "Concatenate",
        optimizer: Union[str, Type[Optimizer]] = "adam",
        early_stopping_min_delta: float = 0.0001,
        early_stopping_patience: int = 20,
        reduce_lr_min_delta: float = 0.0001,
        reduce_lr_patience: int = 5,
        early_stopping_monitor: str = "loss",
        early_stopping_mode: str = "min",
        reduce_lr_monitor: str = "loss",
        reduce_lr_mode: str = "min",
        reduce_lr_factor: float = 0.9,
        avoid_false_negatives: bool = True,
        training_unbalance_rate: float = 1.0,
        training_sample_only_edges_with_heterogeneous_node_types: bool = False,
        use_edge_metrics: bool = False,
        random_state: int = 42,
        use_node_embedding: bool = False,
        node_embedding_size: int = 50,
        use_node_type_embedding: bool = False,
        node_type_embedding_size: int = 50,
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
        optimizer: str = "adam"
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
        avoid_false_negatives: bool = True
            Whether to avoid sampling false negatives.
            This check makes the sampling a bit slower, and generally
            the rate of collision is extremely low.
            Consider disabling this when the task can account for this.
        training_unbalance_rate: float = 1.0
            The amount of negatives to be sampled during the training of the model.
            By default this is 1.0, which means that the same number of positives and
            negatives in the training are of the same cardinality.
        training_sample_only_edges_with_heterogeneous_node_types: bool = False
            Whether to sample exclusively edges between nodes with different node types.
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
        use_edge_metrics: bool = False
            Whether to use the edge metrics from traditional edge prediction.
            These metrics currently include:
            - Adamic Adar
            - Jaccard Coefficient
            - Resource allocation index
            - Preferential attachment
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
        verbose: bool = False
            Whether to show loading bars.
        """
        super().__init__(
            epochs=epochs,
            number_of_graph_convolution_layers=number_of_graph_convolution_layers,
            number_of_units_per_graph_convolution_layers=number_of_units_per_graph_convolution_layers,
            number_of_ffnn_body_layers=number_of_ffnn_body_layers,
            number_of_ffnn_head_layers=number_of_ffnn_head_layers,
            number_of_units_per_ffnn_body_layer=number_of_units_per_ffnn_body_layer,
            number_of_units_per_ffnn_head_layer=number_of_units_per_ffnn_head_layer,
            dropout_rate=dropout_rate,
            combiner="mean",
            apply_norm=True,
            edge_embedding_method=edge_embedding_method,
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
            avoid_false_negatives=avoid_false_negatives,
            training_unbalance_rate=training_unbalance_rate,
            training_sample_only_edges_with_heterogeneous_node_types=training_sample_only_edges_with_heterogeneous_node_types,
            use_edge_metrics=use_edge_metrics,
            random_state=random_state,
            use_simmetric_normalized_laplacian=False,
            use_node_embedding=use_node_embedding,
            node_embedding_size=node_embedding_size,
            use_node_type_embedding=use_node_type_embedding,
            node_type_embedding_size=node_type_embedding_size,
            handling_multi_graph=handling_multi_graph,
            node_feature_names=node_feature_names,
            node_type_feature_names=node_type_feature_names,
            verbose=verbose,
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        removed = [
            "apply_norm",
            "use_simmetric_normalized_laplacian",
            "combiner"
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
        return "GraphSAGE"