"""Kipf GCN model for node-label prediction."""
import copy
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import compress_pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from ensmallen import Graph
from keras_mixed_sequence import Sequence
from tensorflow.keras.callbacks import (  # pylint: disable=import-error,no-name-in-module
    EarlyStopping, ReduceLROnPlateau,
    TerminateOnNaN
)
from tensorflow.keras.layers import Input
from tensorflow.keras.models import \
    Model  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error,no-name-in-module
from userinput.utils import must_be_in_set

from embiggen.layers.tensorflow import (EmbeddingLookup, FlatEmbedding,
                                        GraphConvolution, L2Norm)
from embiggen.utils import AbstractEdgeFeature
from embiggen.utils.abstract_models import (AbstractClassifierModel,
                                            abstract_class)
from embiggen.utils.normalize_model_structural_parameters import \
    normalize_model_list_parameter
from embiggen.utils.number_to_ordinal import number_to_ordinal


def graph_to_sparse_tensor(
    graph: Graph,
    kernel: str,
    handling_multi_graph: str = "warn",
) -> tf.SparseTensor:
    """Returns provided graph as sparse Tensor.

    Parameters
    -------------------
    graph: Graph,
        The graph to convert.
    kernel: str
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
    handling_multi_graph: str = "warn"
        How to behave when dealing with multigraphs.
        Possible behaviours are:
        - "warn", which warns the user and drops the multi-edges.
        - "raise"
        - "drop"

    Raises
    -------------------
    ValueError,
        If the weights are requested but the graph does not contain any.
    ValueError,
        If the graph contains singletons.
    ValueError,
        If the graph is a multigraph.

    Returns
    -------------------
    SparseTensor with (weighted) adjacency matrix.
    """
    if "Weighted" in kernel:
        use_weights = True
        kernel = kernel.replace("Weighted ", "")
    else:
        use_weights = False

    if "Transposed" in kernel:
        transpose = True
        kernel = kernel.replace("Transposed ", "")
    else:
        transpose = False

    if use_weights and not graph.has_edge_weights():
        raise ValueError(
            "Edge weights were requested but the provided graph "
            "does not contain any edge weight."
        )

    if graph.has_singleton_nodes():
        raise ValueError(
            f"In the provided {graph.get_name()} graph there are "
            f"{graph.get_number_of_singleton_nodes()} singleton nodes."
            "The GCN model does not support operations on graph containing "
            "singletons. You can either choose to drop singletons from "
            "the graph by using the `graph.remove_singleton_nodes()` "
            "method or alternatively you can add selfloops to them by "
            "using the `graph.add_selfloops()` method."
        )

    if graph.is_multigraph():
        message = (
            "The GCN model does not currently support convolutions on a multigraph. "
            "We are dropping the parallel edges before computing the adjacency matrix."
        )
        if handling_multi_graph == "warn":
            warnings.warn(message)
        elif handling_multi_graph == "raise":
            raise ValueError(message)

        graph = graph.remove_parallel_edges()

    # We transpose the graph if requested, though the operation is skipped
    # if we are computing the transposed of an undirected graph. A warning
    # is raised in this case.
    if transpose:
        if graph.is_directed():
            graph = graph.to_transposed()
        else:
            warnings.warn(
                "You are trying to compute the transposed of an undirected graph. "
                "The transposed of an undirected graph is the same graph. "
                "This operation is skipped."
            )

    if kernel == "Weights":
        edge_node_ids = graph.get_directed_edge_node_ids()
        kernel_weights = graph.get_directed_edge_weights()
    elif kernel == "Left Normalized Laplacian":
        edge_node_ids, kernel_weights = graph.get_left_normalized_laplacian_coo_matrix()
    elif kernel == "Right Normalized Laplacian":
        (
            edge_node_ids,
            kernel_weights,
        ) = graph.get_right_normalized_laplacian_coo_matrix()
    elif kernel == "Symmetric Normalized Laplacian":
        (
            edge_node_ids,
            kernel_weights,
        ) = graph.get_symmetric_normalized_laplacian_coo_matrix()
    else:
        raise ValueError(
            f"Kernel {kernel} is not supported. "
            "Supported kernels are: "
            f"{', '.join(AbstractGCN.supported_kernels)}."
        )
    kernel_weights = np.abs(kernel_weights)
    if use_weights and kernel != "Weights":
        kernel_weights = kernel_weights * graph.get_directed_edge_weights()

    # We check that no NaNs are present in the kernel weights.
    number_of_nans = np.isnan(kernel_weights).sum()
    if number_of_nans > 0:
        raise ValueError(
            f"The provided graph contains {number_of_nans} NaNs in the kernel weights."
        )
    
    # We check that no value is set to zero.
    number_of_zeros = (kernel_weights == 0).sum()
    if number_of_zeros > 0:
        raise ValueError(
            f"The provided graph contains {number_of_zeros} zeros in the kernel weights."
        )
    
    return tf.sparse.reorder(
        tf.SparseTensor(
            edge_node_ids,
            kernel_weights,
            (graph.get_number_of_nodes(), graph.get_number_of_nodes()),
        )
    )


@abstract_class
class AbstractGCN(AbstractClassifierModel):
    """Abstract base GCN."""

    supported_kernels = [
        "Weights",
        "Left Normalized Laplacian",
        "Right Normalized Laplacian",
        "Symmetric Normalized Laplacian",
        "Transposed Left Normalized Laplacian",
        "Transposed Right Normalized Laplacian",
        "Transposed Symmetric Normalized Laplacian",
        "Weighted Left Normalized Laplacian",
        "Weighted Right Normalized Laplacian",
        "Weighted Symmetric Normalized Laplacian",
        "Trasposed Weighted Left Normalized Laplacian",
        "Trasposed Weighted Right Normalized Laplacian",
        "Trasposed Weighted Symmetric Normalized Laplacian",
    ]

    def __init__(
        self,
        kernels: Optional[Union[str, List[str]]],
        epochs: int = 1000,
        number_of_graph_convolution_layers: int = 2,
        number_of_units_per_graph_convolution_layers: Union[int, List[int]] = 128,
        dropout_rate: float = 0.5,
        batch_size: Optional[int] = None,
        apply_norm: bool = False,
        combiner: str = "sum",
        optimizer: Union[str, Optimizer] = "adam",
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
        random_state: int = 42,
        use_node_embedding: bool = False,
        node_embedding_size: int = 50,
        use_node_type_embedding: bool = False,
        node_type_embedding_size: int = 50,
        residual_convolutional_layers: bool = False,
        handling_multi_graph: str = "warn",
        node_feature_names: Optional[Union[str, List[str]]] = None,
        node_type_feature_names: Optional[Union[str, List[str]]] = None,
        verbose: bool = True,
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
        number_of_graph_convolution_layers: int = 3
            Number of graph convolution layer.
        number_of_units_per_graph_convolution_layers: Union[int, List[int]] = 128
            Number of units per hidden layer.
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
        residual_convolutional_layers: bool = False
            Whether to use residual connections and concatenate all the convolutional
            layers together before the first dense layer.
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
        verbose: bool = True
            Whether to show loading bars.
        """
        self._number_of_graph_convolution_layers = number_of_graph_convolution_layers
        self._number_of_units_per_graph_convolution_layers = (
            normalize_model_list_parameter(
                number_of_units_per_graph_convolution_layers,
                number_of_graph_convolution_layers,
                object_type=int,
                can_be_empty=True,
            )
        )

        if kernels is None:
            kernels = []

        if isinstance(kernels, str):
            kernels = [kernels]

        if not isinstance(kernels, list):
            raise TypeError(
                f"Provided kernels should be either a string or a list, "
                f"but found {type(kernels)}."
            )

        self._kernels = [
            must_be_in_set(kernel, self.supported_kernels, "kernel")
            for kernel in kernels
        ]

        if self.has_convolutional_layers() and not self.has_kernels():
            raise ValueError(
                "You are trying to create a GCN model with convolutional layers "
                "but you have not provided any kernel to use."
            )
        if not self.has_convolutional_layers() and self.has_kernels():
            raise ValueError(
                "You are trying to create a GCN model without convolutional layers "
                "but you have provided kernels to use."
            )

        self._combiner = combiner
        self._epochs = epochs
        self._use_class_weights = use_class_weights
        self._dropout_rate = dropout_rate
        self._optimizer = optimizer
        self._apply_norm = apply_norm
        self._handling_multi_graph = handling_multi_graph
        self._use_node_embedding = use_node_embedding
        self._node_embedding_size = node_embedding_size
        self._use_node_type_embedding = use_node_type_embedding
        self._node_type_embedding_size = node_type_embedding_size

        if isinstance(node_feature_names, str):
            node_feature_names = [node_feature_names]

        if isinstance(node_type_feature_names, str):
            node_type_feature_names = [node_type_feature_names]

        self._node_feature_names = node_feature_names
        self._node_type_feature_names = node_type_feature_names

        if residual_convolutional_layers and not self.has_kernels():
            raise ValueError(
                "You are trying to create a GCN model with residual convolutional layers "
                "but you have not provided any kernel to use."
            )

        self._residual_convolutional_layers = residual_convolutional_layers

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
        self._layer_names: Set[str] = set()
        self.history = None

        super().__init__(random_state=random_state)

        self._batch_size = batch_size

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            epochs=1,
            number_of_units_per_graph_convolution_layers=2,
            handling_multi_graph="drop",
        )

    def _add_layer_name(self, layer_name: str):
        """Add layer name to the set of layer names."""
        if layer_name in self._layer_names:
            raise ValueError(
                f"You are trying to add a layer with name {layer_name} "
                f"but a layer with the same name has already been added."
            )

        self._layer_names.add(layer_name)

    def clone(self) -> Type["AbstractGCN"]:
        """Return copy of self."""
        return copy.deepcopy(self)

    def get_batch_size_from_graph(self, graph: Graph) -> int:
        """Returns batch size to use for the given graph."""
        if self.has_convolutional_layers() or self._batch_size is None:
            return graph.get_number_of_nodes()
        return self._batch_size

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return dict(
            number_of_units_per_graph_convolution_layers=self._number_of_units_per_graph_convolution_layers,
            epochs=self._epochs,
            apply_norm=self._apply_norm,
            combiner=self._combiner,
            use_class_weights=self._use_class_weights,
            dropout_rate=self._dropout_rate,
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
            use_node_embedding=self._use_node_embedding,
            node_embedding_size=self._node_embedding_size,
            use_node_type_embedding=self._use_node_type_embedding,
            node_type_embedding_size=self._node_type_embedding_size,
            residual_convolutional_layers=self._residual_convolutional_layers,
            handling_multi_graph=self._handling_multi_graph,
            node_feature_names=self._node_feature_names,
            node_type_feature_names=self._node_type_feature_names,
            verbose=self._verbose,
        )

    def plot(self, show_shapes: bool = True, **kwargs: Dict):
        """Plot model using dot.
        
        Parameters
        -----------------------
        show_shapes: bool = True
            Whether to show shapes of the layers.
        kwargs: Dict
            Additional arguments to pass to the plot function.
        """
        if self._model is None:
            raise ValueError(
                "You are trying to plot a model that has not been compiled yet. "
                "You should call the `compile` or `fit` methods before calling `plot`."
            )
        return tf.keras.utils.plot_model(
            self._model,
            show_layer_names=True,
            expand_nested=True,
            show_layer_activations=True,
            show_shapes=show_shapes,
            **kwargs,
        )
    
    def summary(self, **kwargs: Dict):
        """Print summary of the model.
        
        Parameters
        -----------------------
        kwargs: Dict
            Additional arguments to pass to the summary function.
        """
        if self._model is None:
            raise ValueError(
                "You are trying to print the summary of a model that has not been compiled yet. "
                "You should call the `compile` or `fit` methods before calling `summary`."
            )
        return self._model.summary(**kwargs)

    def _get_class_weights(self, graph: Graph) -> Dict[int, float]:
        """Returns dictionary with class weights."""
        raise NotImplementedError(
            "The method `get_class_weights` should be implemented "
            "in the child classes of `AbstractGCN`, but is missing "
            f"in the class {self.__class__.__name__}."
        )

    def _get_model_training_input(
        self,
        graph: Graph,
        support: Graph,
        node_features: Optional[List[np.ndarray]],
        node_type_features: Optional[List[np.ndarray]],
        edge_type_features: Optional[List[np.ndarray]],
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[np.ndarray]]],
    ) -> Tuple[Union[np.ndarray, Type[Sequence]]]:
        """Returns training input tuple."""
        raise NotImplementedError(
            "The method `get_model_training_input` should be implemented "
            "in the child classes of `AbstractGCN`, but is missing "
            f"in the class {self.__class__.__name__}."
        )

    def _get_model_training_output(
        self,
        graph: Graph,
    ) -> Optional[np.ndarray]:
        """Returns training output tuple."""
        raise NotImplementedError(
            "The method `get_model_training_output` should be implemented "
            "in the child classes of `AbstractGCN`, but is missing "
            f"in the class {self.__class__.__name__}."
        )

    def _get_model_training_sample_weights(
        self,
        graph: Graph,
    ) -> Optional[np.ndarray]:
        """Returns training output tuple."""
        raise NotImplementedError(
            "The method `_get_model_training_sample_weights` should be implemented "
            "in the child classes of `AbstractGCN`, but is missing "
            f"in the class {self.__class__.__name__}."
        )

    def _get_model_prediction_input(
        self,
        graph: Graph,
        support: Graph,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[
            Union[
                Type[AbstractEdgeFeature],
                List[Union[np.ndarray, Type[AbstractEdgeFeature]]],
            ]
        ] = None,
    ) -> Tuple[Union[np.ndarray, Type[Sequence]]]:
        """Returns dictionary with class weights."""
        raise NotImplementedError(
            "The method `get_model_prediction_input` should be implemented "
            "in the child classes of `AbstractGCN`, but is missing "
            f"in the class {self.__class__.__name__}."
        )

    def _build_model(
        self,
        graph: Graph,
        graph_convolution_model: Model,
        edge_type_features: List[np.ndarray],
        edge_features: List[Union[np.ndarray, Type[AbstractEdgeFeature]]],
    ) -> Type[Model]:
        """Returns GCN model."""
        raise NotImplementedError(
            "The method `_build_model` should be implemented "
            "in the child classes of `AbstractGCN`, but is missing "
            f"in the class {self.__class__.__name__}."
        )

    def _build_graph_convolution_model(
        self,
        graph: Graph,
        node_features: List[np.ndarray],
        node_type_features: List[np.ndarray],
    ) -> Model:
        """Create new GCN model."""
        # We create the list we will use to collect the input features.
        input_features = []
        hidden = []

        if self.has_kernels() and len(node_features) == 0 and len(node_type_features) == 0 and not self._use_node_embedding and not self._use_node_type_embedding:
            raise ValueError(
                "You are trying to create a GCN model with convolutional layers "
                "but you have not provided any node or node type feature to use, "
                "and you are not using node or node type embeddings."
            )

        kernels = []
        for kernel in self._kernels:
            self._add_layer_name(kernel)
            kernels.append(Input(
                shape=(graph.get_number_of_nodes(),),
                batch_size=graph.get_number_of_nodes(),
                sparse=True,
                name=kernel,
            ))

        # When we are not creating a node-level model but an edge-label or
        # edge-prediction model, and the model is not using convolutional layers,
        # we need to differentiate th source node and destination node features,
        # both for the node themselves and for the associated node types.

        # This is necessary specifically when the batch size is not the number of nodes,
        # as we no longer can assume a dense range of features for each of the nodes
        # in the source nodes and destination nodes feature set.

        # One important observation, is that we cannot any longer create in this portion
        # of the model the node embedding layer, as we do not have the source and destination
        # node ids at this level - we will need to add this node embedding layer within the
        # abstract edge GCN model.

        # What we can do here, is handle in a centralized manner the source and destination
        # node and node type features, and then pass them to the abstract edge GCN model
        # as a list of features, where the first half of the features are the source node
        # features and the second half of the features are the destination node features.

        if not self.has_kernels() and self.is_edge_level_task():
            prefixes = ("Source ", "Destination ")
        else:
            prefixes = ("",)

        for prefix in prefixes:
            # We create the input layers for all of the node and node type features.
            for features, feature_names, feature_category in (
                (node_features, self._node_feature_names, f"{prefix}node feature"),
                (
                    node_type_features,
                    self._node_type_feature_names,
                    f"{prefix}node type feature",
                ),
            ):
                if feature_names is not None and len(features) == 0:
                    raise ValueError(
                        f"You have provided {len(feature_names)} {feature_category} names, "
                        f"but then you have not provided any {feature_category}. Either "
                        f"provide the {feature_category} to the compile or fit method, or "
                        "remove the feature names from the constructor call."
                    )
                if len(features) > 0:
                    if feature_names is None:
                        if len(features) > 1:
                            feature_names = [
                                f"{number_to_ordinal(i+1)} {feature_category}"
                                for i in range(len(features))
                            ]
                        else:
                            feature_names = [feature_category.capitalize()]
                    else:
                        feature_names = [
                            f"{prefix}{feature_name}"
                            for feature_name in feature_names
                        ]
                    if len(feature_names) != len(features):
                        raise ValueError(
                            f"You have provided {len(feature_names)} "
                            f"{feature_category} names but you have provided {len(features)} "
                            f"{feature_category}s to the model. Specifically, the provided "
                            f"feature names are {feature_names}."
                        )
                    new_input_features = []
                    for node_feature, node_feature_name in zip(
                        features, feature_names
                    ):
                        self._add_layer_name(node_feature_name)
                        new_input_features.append(Input(
                            shape=node_feature.shape[1:],
                            name=node_feature_name,
                        ))
                    input_features.extend(new_input_features)
                    hidden.extend(new_input_features)

            # We create the node embedding layer if we are executing convolutional layers
            # or, if we are not executing convolutional layers, if we are not executing
            # an edge-level task. In fact, when we are executing an edge-level task and
            # we are not executing convolutional layers, we will need to create the node
            # embedding layer within the abstract edge GCN model.
            if self._use_node_embedding and (
                self.has_kernels() or not self.is_edge_level_task()
            ):
                node_ids = Input(shape=(1,), name="Nodes", dtype=tf.int32)
                input_features.append(node_ids)

                node_embedding = FlatEmbedding(
                    vocabulary_size=graph.get_number_of_nodes(),
                    dimension=self._node_embedding_size,
                    input_length=1,
                    name="NodesEmbedding",
                )(node_ids)
                hidden.append(node_embedding)

            if self._use_node_type_embedding:
                node_type_ids = Input(
                    shape=(graph.get_maximum_multilabel_count(),),
                    name=f"{prefix}Node Types",
                    dtype=tf.int32,
                )
                input_features.append(node_type_ids)

                space_adjusted_input_layer_name = node_type_ids.name.replace(" ", "")
                use_masking = graph.has_multilabel_node_types() or graph.has_unknown_node_types()

                node_type_embedding = FlatEmbedding(
                    vocabulary_size=graph.get_number_of_node_types() + (1 if use_masking else 0),
                    dimension=self._node_type_embedding_size,
                    input_length=graph.get_maximum_multilabel_count(),
                    mask_zero=use_masking,
                    name=f"{space_adjusted_input_layer_name}Embedding",
                )(node_type_ids)
                hidden.append(node_type_embedding)

        starting_hidden = hidden

        output_hiddens = []

        for kernel, kernel_name in zip(kernels, self._kernels):
            hidden = starting_hidden
            # Building the body of the model.
            for i, units in enumerate(
                self._number_of_units_per_graph_convolution_layers
            ):
                if len(self._number_of_units_per_graph_convolution_layers) > 1:
                    ordinal = number_to_ordinal(i + 1)
                else:
                    ordinal = ""
                sanitized_kernel_name = kernel_name.replace(" ", "")
                assert len(hidden) > 0
                hidden = GraphConvolution(
                    units=units,
                    combiner=self._combiner,
                    dropout_rate=self._dropout_rate,
                    apply_norm=self._apply_norm,
                    name=f"{ordinal}{sanitized_kernel_name}GraphConvolution",
                )((kernel, *hidden))
                if self._residual_convolutional_layers:
                    output_hiddens.extend(hidden)
            if not self._residual_convolutional_layers:
                output_hiddens.extend(hidden)

        if len(output_hiddens) == 0:
            output_hiddens = starting_hidden

        # Returning the convolutional portion of the model.
        return Model(
            inputs=[
                input_layer
                for input_layer in (*kernels, *input_features)
                if input_layer is not None
            ],
            outputs=output_hiddens,
        )

    def get_model_expected_input_shapes(self, graph: Graph) -> Dict[str, Tuple[int]]:
        """Return dictionary with expected input shapes."""
        if self._model is None:
            raise RuntimeError(
                "You need to fit the model before you can "
                "retrieve the expected input shapes."
            )

        return {
            input_layer.name: tuple(
                [
                    self.get_batch_size_from_graph(graph)
                    if dimension is None
                    else dimension
                    for dimension in tuple(input_layer.shape)
                ]
            )
            for input_layer in self._model.inputs
        }

    def compile(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[
            Union[
                Type[AbstractEdgeFeature],
                List[Union[np.ndarray, Type[AbstractEdgeFeature]]],
            ]
        ] = None,
    ) -> pd.DataFrame:
        """Return pandas dataframe with training history.

        Parameters
        -----------------------
        graph: Graph,
            The graph whose edges are to be embedded and edge types extracted.
        node_features: Optional[List[np.ndarray]]
            The node features to be used in the training of the model.
        node_type_features: Optional[List[np.ndarray]]
            The node type features to be used in the training of the model.
        edge_type_features: Optional[List[np.ndarray]]
            The edge type features to be used in the training of the model.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Union[np.ndarray, Type[AbstractEdgeFeature]]]]] = None
            The edge features to be used in the training of the model.

        Returns
        -----------------------
        Dataframe with training history.
        """
        if (
            self.has_kernels()
            and node_features is None
            and not self._use_node_embedding
            and node_type_features is None
            and not self._use_node_type_embedding
        ):
            raise ValueError(
                "Neither node features were provided nor the node "
                "embedding was enabled through the `use_node_embedding` "
                "parameter. If you do not provide node features or use a node embedding layer "
                "nor use a node type embedding layer and neiher use node type features, "
                "it does not make sense to use a GCN model."
            )

        node_features=self.normalize_node_features(
            graph=graph,
            support=support,
            random_state=self._random_state,
            node_features=node_features,
            allow_automatic_feature=True,
        )
        node_type_features=self.normalize_node_type_features(
            graph=graph,
            support=support,
            random_state=self._random_state,
            node_type_features=node_type_features,
            allow_automatic_feature=True,
        )
        edge_type_features=self.normalize_edge_type_features(
            graph=graph,
            support=support,
            random_state=self._random_state,
            edge_type_features=edge_type_features,
            allow_automatic_feature=True,
        )
        edge_features=self.normalize_edge_features(
            graph=graph,
            support=support,
            random_state=self._random_state,
            edge_features=edge_features,
            allow_automatic_feature=True,
        )

        self._model: Type[Model] = self._build_model(
            support,
            graph_convolution_model=self._build_graph_convolution_model(
                graph,
                node_features=node_features,
                node_type_features=node_type_features,
            ),
            edge_type_features=edge_type_features,
            edge_features=edge_features,
        )

    def _fit(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[
            Union[
                Type[AbstractEdgeFeature],
                List[Union[np.ndarray, Type[AbstractEdgeFeature]]],
            ]
        ] = None,
    ) -> pd.DataFrame:
        """Return pandas dataframe with training history.

        Parameters
        -----------------------
        graph: Graph,
            The graph whose edges are to be embedded and edge types extracted.
            It can either be an Graph or a list of lists of edges.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph.
        node_features: Optional[List[np.ndarray]]
            The node features to be used in the training of the model.
        node_type_features: Optional[List[np.ndarray]]
            The node type features to be used in the training of the model.
        edge_type_features: Optional[List[np.ndarray]]
            The edge type features to be used in the training of the model.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Union[np.ndarray, Type[AbstractEdgeFeature]]]]] = None
            The edge features to be used in the training of the model.

        Returns
        -----------------------
        Dataframe with training history.
        """
        try:
            from tqdm.keras import TqdmCallback

            traditional_verbose = False
        except AttributeError:
            traditional_verbose = True

        if support is None:
            support = graph

        class_weight = (
            self._get_class_weights(graph) if self._use_class_weights else None
        )

        if self._model is None:
            self.compile(
                graph,
                support=support,
                node_features=node_features,
                node_type_features=node_type_features,
                edge_type_features=edge_type_features,
                edge_features=edge_features,
            )

        # Within the expected input shapes, we do not have the batch size itself.
        # The batch size is left implicit as a None value, but we need to add it
        # to the expected input shapes to check that the input shapes are correct.

        expected_input_shapes = self.get_model_expected_input_shapes(graph)

        model_input = self._get_model_training_input(
            graph,
            support=support,
            edge_features=edge_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            node_features=node_features,
        )

        if not isinstance(model_input, tuple) and not issubclass(
            type(model_input), Sequence
        ):
            raise RuntimeError(
                "The model input should be a subclass of `Sequence` or a tuple "
                f"but it is `{type(model_input)}`. "
                "This is an internal error, please open an issue at "
                "GRAPE's GitHub page."
            )

        if issubclass(type(model_input), Sequence):
            sequence_input_shapes = [
                tuple(feature.shape) for feature in model_input[0][0]
            ]
        elif isinstance(model_input, tuple):
            sequence_input_shapes = [tuple(feature.shape) for feature in model_input]
        else:
            raise RuntimeError(
                "The model input should be a subclass of `Sequence` or a tuple "
                f"but it is `{type(model_input)}`. "
                "This is an internal error, please open an issue at "
                "GRAPE's GitHub page."
            )

        if len(expected_input_shapes) != len(sequence_input_shapes):
            raise RuntimeError(
                f"We expected {len(expected_input_shapes)} inputs "
                f"and we received {len(sequence_input_shapes)} inputs. "
                "Specifically, the input shapes we expected were "
                f"{expected_input_shapes}, but the training sequence "
                f"provided the input shapes {sequence_input_shapes}. "
                "This is an internal error, please open an issue at "
                "GRAPE's GitHub page."
            )

        for (layer_name, layer_input_shape), input_shape in zip(
            expected_input_shapes.items(), sequence_input_shapes
        ):
            if (
                (layer_input_shape[1:] != input_shape[1:])
                and not self.has_convolutional_layers()
                or layer_input_shape != input_shape
                and self.has_convolutional_layers()
            ):
                raise RuntimeError(
                    f"We expected {len(expected_input_shapes)} inputs "
                    f"and we received {len(sequence_input_shapes)} inputs. "
                    f"The input shape of the layer `{layer_name}` "
                    f"should be `{layer_input_shape}` but it is `{input_shape}`. "
                    "Specifically, the input shapes we expected were "
                    f"{expected_input_shapes}, but the training sequence "
                    f"provided the input shapes {sequence_input_shapes}. "
                    "This is an internal error, please open an issue at "
                    "GRAPE's GitHub page."
                )

        self.history = self._model.fit(
            x=model_input,
            y=self._get_model_training_output(graph),
            sample_weight=self._get_model_training_sample_weights(graph),
            epochs=self._epochs,
            verbose=traditional_verbose and self._verbose > 0,
            batch_size=self.get_batch_size_from_graph(graph),
            shuffle=False,
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
                TerminateOnNaN(),
                *(
                    (
                        TqdmCallback(
                            verbose=1 if "edge" in self.task_name().lower() else 0,
                            leave=False,
                        ),
                    )
                    if not traditional_verbose and self._verbose
                    else ()
                ),
            ],
        )

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
                List[Union[np.ndarray, Type[AbstractEdgeFeature]]],
            ]
        ] = None,
    ) -> pd.DataFrame:
        """Run predictions on the provided graph."""
        if not graph.has_edges():
            return np.array([])

        if support is None:
            support = graph

        model_input = self._get_model_prediction_input(
            graph,
            support,
            node_features,
            node_type_features,
            edge_type_features,
            edge_features,
        )

        if issubclass(type(model_input), Sequence):
            sequence_input_shapes = [
                tuple(feature.shape) for feature in model_input[0][0]
            ]
        elif isinstance(model_input, tuple):
            sequence_input_shapes = [tuple(feature.shape) for feature in model_input]
        else:
            raise RuntimeError(
                "The model input should be a subclass of `Sequence` or a tuple "
                f"but it is `{type(model_input)}`. "
                "This is an internal error, please open an issue at "
                "GRAPE's GitHub page."
            )

        expected_input_shapes = self.get_model_expected_input_shapes(graph)

        if len(expected_input_shapes) != len(sequence_input_shapes):
            raise RuntimeError(
                f"We expected {len(expected_input_shapes)} inputs "
                f"and we received {len(sequence_input_shapes)} inputs. "
                "Specifically, the input shapes we expected were "
                f"{expected_input_shapes}, but the prediction sequence "
                f"provided the input shapes {sequence_input_shapes}. "
                "This is an internal error, please open an issue at "
                "GRAPE's GitHub page."
            )

        for (layer_name, layer_input_shape), input_shape in zip(
            expected_input_shapes.items(), sequence_input_shapes
        ):
            if layer_input_shape[1:] != input_shape[1:]:
                raise RuntimeError(
                    f"We expected {len(expected_input_shapes)} inputs "
                    f"and we received {len(sequence_input_shapes)} inputs. "
                    f"The input shape of the layer `{layer_name}` "
                    f"should be `{layer_input_shape}` but it is `{input_shape}`. "
                    "Specifically, the input shapes we expected were "
                    f"{expected_input_shapes}, but the prediction sequence "
                    f"provided the input shapes {sequence_input_shapes}. "
                    "This is an internal error, please open an issue at "
                    "GRAPE's GitHub page."
                )

        return self._model.predict(
            model_input, batch_size=self.get_batch_size_from_graph(graph), verbose=False
        )

    def _predict(
        self,
        graph: Graph,
        support: Graph,
        node_features: List[np.ndarray],
        node_type_features: List[np.ndarray],
        edge_type_features: List[np.ndarray],
        edge_features: List[Union[np.ndarray, Type[AbstractEdgeFeature]]],
    ) -> pd.DataFrame:
        """Run predictions on the provided graph."""
        predictions = self._predict_proba(
            graph,
            support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
        )

        if self.is_binary_prediction_task() or self.is_multilabel_prediction_task():
            return predictions > 0.5

        return predictions.argmax(axis=-1)

    def get_output_activation_name(self) -> str:
        """Return activation of the output."""
        # Adding the last layer of the model.
        if self.is_binary_prediction_task() or self.is_multilabel_prediction_task():
            return "sigmoid"
        return "softmax"

    def get_loss_name(self) -> str:
        """Return model loss."""
        # Adding the last layer of the model.
        if self.is_binary_prediction_task() or self.is_multilabel_prediction_task():
            return "binary_crossentropy"
        return "sparse_categorical_crossentropy"

    def get_output_classes(self, graph: Graph) -> int:
        """Returns number of output classes."""
        raise NotImplementedError(
            "The method `get_output_classes` should be implemented "
            "in the child classes of `AbstractGCN`, but is missing "
            f"in the class {self.__class__.__name__}."
        )

    def has_convolutional_layers(self) -> bool:
        """Returns whether the present model has convolutional layers."""
        return self._number_of_graph_convolution_layers

    def has_kernels(self) -> bool:
        """Returns whether the present model has kernels."""
        return len(self._kernels) > 0

    def convert_graph_to_kernels(self, graph: Graph) -> Optional[tf.SparseTensor]:
        """Returns provided graph converted to a sparse Tensor.

        Implementation details
        ---------------------------
        Do note that when the model does not have convolutional layers
        the model will return None, as to avoid allocating like object for
        apparently no reason.
        """
        if not self.has_kernels():
            return None
        return [
            graph_to_sparse_tensor(
                graph,
                kernel=kernel,
                handling_multi_graph=self._handling_multi_graph,
            )
            for kernel in self._kernels
        ]

    @classmethod
    def requires_edge_weights(cls) -> bool:
        return False

    @classmethod
    def requires_positive_edge_weights(cls) -> bool:
        return False

    @classmethod
    def library_name(cls) -> str:
        """Return name of the model."""
        return "TensorFlow"

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return True

    def is_using_edge_weights(self) -> bool:
        """Returns whether the model is parametrized to use edge weights."""
        return any(["Weighted" in kernel for kernel in self._kernels])

    def is_edge_level_task(self) -> bool:
        """Returns whether the task is edge level."""
        return False

    @classmethod
    def load(cls, path: str) -> "Self":
        """Load a saved version of the model from the provided path.

        Parameters
        -------------------
        path: str
            Path from where to load the model.
        """
        with tf.keras.utils.custom_object_scope(
            {
                "GraphConvolution": GraphConvolution,
                "EmbeddingLookup": EmbeddingLookup,
                "FlatEmbedding": FlatEmbedding,
                "L2Norm": L2Norm,
            }
        ):
            return compress_pickle.load(path)

    def dump(self, path: str):
        """Dump the current model at the provided path.

        Parameters
        -------------------
        path: str
            Path from where to dump the model.
        """
        compress_pickle.dump(self, path)
