"""GCN model for edge prediction."""
from typing import List, Union, Optional, Dict, Any, Type

import numpy as np
from tensorflow.keras.optimizers import (
    Optimizer,
)  # pylint: disable=import-error,no-name-in-module

from ensmallen import Graph
from embiggen.edge_prediction.edge_prediction_model import AbstractEdgePredictionModel
from embiggen.utils.abstract_edge_gcn import AbstractEdgeGCN
from embiggen.sequences.tensorflow_sequences import (
    GCNEdgePredictionTrainingSequence,
    GCNEdgePredictionSequence,
)
from embiggen.utils import AbstractEdgeFeature


class GCNEdgePrediction(AbstractEdgeGCN, AbstractEdgePredictionModel):
    """GCN model for edge prediction."""

    def __init__(
        self,
        kernels: Optional[Union[str, List[str]]] = ["Symmetric Normalized Laplacian",],
        epochs: int = 1000,
        number_of_graph_convolution_layers: int = 2,
        number_of_units_per_graph_convolution_layers: Union[int, List[int]] = 128,
        number_of_ffnn_body_layers: int = 2,
        number_of_ffnn_head_layers: int = 1,
        number_of_units_per_ffnn_body_layer: Union[int, List[int]] = 128,
        number_of_units_per_ffnn_head_layer: Union[int, List[int]] = 128,
        dropout_rate: float = 0.2,
        batch_size: Optional[int] = None,
        number_of_batches_per_epoch: Optional[int] = None,
        apply_norm: bool = False,
        combiner: str = "sum",
        edge_embedding_methods: Union[List[str], str] = "Concatenate",
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
        avoid_false_negatives: bool = True,
        training_unbalance_rate: float = 1.0,
        use_edge_metrics: bool = False,
        random_state: int = 42,
        use_node_embedding: bool = False,
        node_embedding_size: int = 50,
        use_node_type_embedding: bool = False,
        node_type_embedding_size: int = 50,
        use_edge_type_embedding: bool = False,
        edge_type_embedding_size: int = 50,
        residual_convolutional_layers: bool = False,
        siamese_node_feature_module: bool = True,
        handling_multi_graph: str = "warn",
        node_feature_names: Optional[Union[str, List[str]]] = None,
        node_type_feature_names: Optional[Union[str, List[str]]] = None,
        edge_type_feature_names: Optional[Union[str, List[str]]] = None,
        verbose: bool = False,
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
        number_of_batches_per_epoch: Optional[int] = None
            Number of batches to use per epoch.
            By default, this is None, which means that the number of batches
            will be equal to the number of directed edges divided by the batch size.
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
        verbose: bool = False
            Whether to show loading bars.
        """
        AbstractEdgePredictionModel.__init__(self, random_state=random_state)
        AbstractEdgeGCN.__init__(
            self,
            kernels=kernels,
            epochs=epochs,
            number_of_graph_convolution_layers=number_of_graph_convolution_layers,
            number_of_units_per_graph_convolution_layers=number_of_units_per_graph_convolution_layers,
            number_of_ffnn_body_layers=number_of_ffnn_body_layers,
            number_of_ffnn_head_layers=number_of_ffnn_head_layers,
            number_of_units_per_ffnn_body_layer=number_of_units_per_ffnn_body_layer,
            number_of_units_per_ffnn_head_layer=number_of_units_per_ffnn_head_layer,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            combiner=combiner,
            apply_norm=apply_norm,
            edge_embedding_methods=edge_embedding_methods,
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
            use_class_weights=False,
            use_edge_metrics=use_edge_metrics,
            random_state=random_state,
            use_node_embedding=use_node_embedding,
            node_embedding_size=node_embedding_size,
            use_node_type_embedding=use_node_type_embedding,
            node_type_embedding_size=node_type_embedding_size,
            use_edge_type_embedding=use_edge_type_embedding,
            edge_type_embedding_size=edge_type_embedding_size,
            residual_convolutional_layers=residual_convolutional_layers,
            siamese_node_feature_module=siamese_node_feature_module,
            handling_multi_graph=handling_multi_graph,
            node_feature_names=node_feature_names,
            node_type_feature_names=node_type_feature_names,
            edge_type_feature_names=edge_type_feature_names,
            verbose=verbose,
        )
        self._number_of_batches_per_epoch = number_of_batches_per_epoch
        self._avoid_false_negatives = avoid_false_negatives
        self._training_unbalance_rate = training_unbalance_rate

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        removed = [
            "use_class_weights",
        ]
        return dict(
            **{
                key: value
                for key, value in AbstractEdgeGCN.parameters(self).items()
                if key not in removed
            },
            training_unbalance_rate=self._training_unbalance_rate,
        )

    def _get_model_prediction_input(
        self,
        graph: Graph,
        support: Graph,
        node_features: Optional[List[np.ndarray]],
        node_type_features: Optional[List[np.ndarray]],
        edge_type_features: Optional[List[np.ndarray]],
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ],
    ) -> GCNEdgePredictionSequence:
        """Returns dictionary with class weights."""
        return GCNEdgePredictionSequence(
            graph,
            support=support,
            kernels=self.convert_graph_to_kernels(support),
            batch_size=self.get_batch_size_from_graph(graph),
            node_features=node_features,
            return_node_ids=self._use_node_embedding,
            return_edge_node_ids=self._use_node_embedding or self.has_kernels(),
            return_node_types=self._use_node_type_embedding,
            return_edge_types=self._use_edge_type_embedding,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            use_edge_metrics=self._use_edge_metrics,
            edge_features=edge_features,
        )

    def _get_model_training_input(
        self,
        graph: Graph,
        support: Graph,
        node_features: Optional[List[np.ndarray]],
        node_type_features: Optional[List[np.ndarray]],
        edge_type_features: Optional[List[np.ndarray]],
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ],
    ) -> GCNEdgePredictionTrainingSequence:
        """Returns training input tuple."""
        return GCNEdgePredictionTrainingSequence(
            graph,
            kernels=self.convert_graph_to_kernels(support),
            support=support,
            batch_size=self.get_batch_size_from_graph(graph),
            number_of_batches_per_epoch=self._number_of_batches_per_epoch,
            node_features=node_features,
            edge_features=edge_features,
            return_node_ids=self._use_node_embedding,
            return_edge_node_ids=self._use_node_embedding or self.has_kernels(),
            return_node_types=self._use_node_type_embedding,
            return_edge_types=self._use_edge_type_embedding,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            use_edge_metrics=self._use_edge_metrics,
            avoid_false_negatives=self._avoid_false_negatives,
            negative_samples_rate=self._training_unbalance_rate
            / (self._training_unbalance_rate + 1.0),
            random_state=self._random_state,
        )

    def get_output_classes(self, graph: Graph) -> int:
        """Returns number of output classes."""
        return 1

    @classmethod
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can optionally use edge types."""
        return True

    def _predict_proba(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[
            Union[
                Type[AbstractEdgeFeature],
                List[Union[Type[AbstractEdgeFeature], np.ndarray]],
            ]
        ] = None,
    ) -> np.ndarray:
        """Run predictions on the provided graph."""
        predictions = super()._predict_proba(
            graph,
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
        )
        # The model will padd the predictions with a few zeros
        # in order to run the GCN portion of the model, which
        # always requires a batch size equal to the nodes number.
        return predictions[: graph.get_number_of_directed_edges()]

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            number_of_batches_per_epoch=1,
            **super().smoke_test_parameters(),
        )

    def is_using_edge_types(self) -> bool:
        """Returns whether the model is parametrized to use edge types."""
        return self._use_edge_type_embedding or super().is_using_edge_types()

    @classmethod
    def requires_edge_types(cls) -> bool:
        """Returns whether the model requires edge types."""
        return False

    @classmethod
    def requires_edge_type_features(cls) -> bool:
        return False

    @classmethod
    def requires_edge_features(cls) -> bool:
        return False

    @classmethod
    def can_use_edge_type_features(cls) -> bool:
        return True

    @classmethod
    def can_use_edge_features(cls) -> bool:
        return True

    @classmethod
    def model_name(cls) -> str:
        return "Everything Bagel GCN"