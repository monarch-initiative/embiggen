"""Siamese network for node-embedding including optionally node types and edge types."""
import warnings
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from ensmallen_graph import EnsmallenGraph
from tensorflow.keras import backend as K  # pylint: disable=import-error
from tensorflow.keras.constraints import UnitNorm
from tensorflow.keras.layers import Embedding, Flatten  # pylint: disable=import-error
from tensorflow.keras.layers import (Add, Concatenate, Dot,
                                     GlobalAveragePooling1D, Input)
from tensorflow.keras.models import Model  # pylint: disable=import-error
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error

from ..sequences import EdgePredictionSequence
from .embedder import Embedder


class Siamese(Embedder):
    """Siamese network for node-embedding including optionally node types and edge types."""

    def __init__(
        self,
        graph: EnsmallenGraph,
        use_node_types: Union[bool, str] = "auto",
        node_types_combination: str = "Dot",
        use_edge_types: Union[bool, str] = "auto",
        node_embedding_size: int = 100,
        node_type_embedding_size: int = 100,
        edge_type_embedding_size: int = 100,
        relu_bias: float = 1.0,
        embedding: Union[np.ndarray, pd.DataFrame] = None,
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
        model_name: str = "TransE",
        optimizer: Union[str, Optimizer] = None
    ):
        """Create new sequence Embedder model.

        Parameters
        -------------------------------------------
        vocabulary_size: int = None,
            Number of terms to embed.
            In a graph this is the number of nodes, while in a text is the
            number of the unique words.
            If None, the seed embedding must be provided.
            It is not possible to provide both at once.
        node_embedding_size: int = 100,
            Dimension of the embedding.
            If None, the seed embedding must be provided.
            It is not possible to provide both at once.
        node_type_embedding_size: int = 100,
            Dimension of the embedding for the node types.
        edge_type_embedding_size: int = 100,
            Dimension of the embedding for the edge types.
        relu_bias: float = 1.0,
            The bias to use for the ReLu.
        embedding: Union[np.ndarray, pd.DataFrame] = None,
            The seed embedding to be used.
            Note that it is not possible to provide at once both
            the embedding and either the vocabulary size or the embedding size.
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
            Optional extra features to be used during the computation
            of the embedding. The features must be available for all the
            elements considered for the embedding.
        model_name: str = "Word2Vec",
            Name of the model.
        optimizer: Union[str, Optimizer] = "nadam",
            The optimizer to be used during the training of the model.
        window_size: int = 4,
            Window size for the local context.
            On the borders the window size is trimmed.
        negative_samples_rate: int = 10,
            The number of negative classes to randomly sample per batch.
            This single sample of negative classes is evaluated for each element in the batch.
        """
        self._model_name = model_name
        if graph.has_disconnected_nodes():
            warnings.warn(
                "The graph contains disconnected nodes: these nodes will "
                "not be embedded in a semantically sensible way, but "
                "will only obtain a random node embedding vector which is "
                "from all other nodes."
            )
        if use_node_types == "auto":
            use_node_types = graph.has_node_types() and not graph.has_unknown_node_types()
        if use_node_types:
            if not graph.has_node_types():
                raise ValueError(
                    "Node types are to be used to embed the given graph, "
                    "but the graph does not have node types"
                )
            if graph.has_unknown_node_types():
                raise ValueError(
                    "Node types are to be used to embed the given graph, "
                    "but the graph contains unknown node types and this "
                    "type of model is not designed in order to handle "
                    "unknown node types."
                )
            if graph.has_singleton_node_types():
                warnings.warn(
                    "The node types will be used in order to compute the node "
                    "embedding, but there are some singleton node types: these "
                    "node types will not capture any characteristic that is not "
                    "already captured by the node embedding, and may be an error "
                    "in the pipeline you have used to create this graph."
                )
            self._multilabel_node_types = graph.has_multilabel_node_types()
            self._max_node_types = graph.get_maximum_node_types_number()
            self._node_types_number = graph.get_node_types_number()
        if use_edge_types == "auto":
            use_edge_types = graph.has_edge_types() and not graph.has_unknown_edge_types()
        if use_edge_types:
            if not graph.has_edge_types():
                raise ValueError(
                    "Edge types are to be used to embed the given graph, "
                    "but the graph does not have edge types"
                )
            if graph.has_unknown_edge_types():
                raise ValueError(
                    "Edge types are to be used to embed the given graph, "
                    "but the graph contains unknown edge types and this "
                    "type of model is not designed in order to handle "
                    "unknown edge types."
                )
            if graph.has_singleton_edge_types():
                warnings.warn(
                    "The edge types will be used in order to compute the edge "
                    "embedding, but there are some singleton edge types: these "
                    "edge types will not capture any characteristic that is not "
                    "already captured by the edge embedding, and may be an error "
                    "in the pipeline you have used to create this graph."
                )
            self._edge_types_number = graph.get_edge_types_number()

        self._node_types_combination = node_types_combination
        self._use_node_types = use_node_types
        self._use_edge_types = use_edge_types
        self._node_type_embedding_size = node_type_embedding_size
        self._edge_type_embedding_size = edge_type_embedding_size
        self._graph = graph
        self._relu_bias = relu_bias

        super().__init__(
            vocabulary_size=graph.get_nodes_number(),
            embedding_size=node_embedding_size,
            embedding=embedding,
            extra_features=extra_features,
            optimizer=optimizer
        )

    def _build_model(self):
        """Return Node2Vec model."""
        # Creating the inputs layers
        input_layers = []
        source_nodes_input = Input((1,), name="source_nodes")
        input_layers.append(source_nodes_input)
        if self._use_node_types:
            source_node_types_input = Input(
                (self._max_node_types,), name="source_node_types")
            input_layers.append(source_node_types_input)

        destination_nodes_input = Input((1,), name="destination_nodes")
        input_layers.append(destination_nodes_input)
        if self._use_node_types:
            destination_node_types_input = Input(
                (self._max_node_types,), name="destination_node_types")
            input_layers.append(destination_node_types_input)

        if self._use_edge_types:
            edge_types_input = Input((1,), name="destination_edge_types")
            input_layers.append(edge_types_input)
        else:
            edge_types_input = None

        # Creating the embedding layer for the contexts
        node_embedding_layer = Embedding(
            input_dim=self._vocabulary_size,
            output_dim=self._embedding_size,
            input_length=1,
            name=Embedder.EMBEDDING_LAYER_NAME,
        )

        src_node_embedding = UnitNorm()(
            Flatten()(node_embedding_layer(source_nodes_input)))
        dst_node_embedding = UnitNorm()(
            Flatten()(node_embedding_layer(destination_nodes_input)))

        if self._use_node_types:
            node_type_embedding_layer = Embedding(
                input_dim=self._node_types_number +
                int(self._multilabel_node_types),
                output_dim=self._node_type_embedding_size,
                input_length=self._max_node_types,
                name="node_type_embedding_layer",
                embeddings_constraint=UnitNorm(),
                mask_zero=self._multilabel_node_types
            )
            source_node_types_embedding = node_type_embedding_layer(
                source_node_types_input
            )
            destination_node_types_embedding = node_type_embedding_layer(
                destination_node_types_input
            )
            global_average_layer = GlobalAveragePooling1D()
            source_node_types_embedding = global_average_layer(
                source_node_types_embedding
            )
            destination_node_types_embedding = global_average_layer(
                destination_node_types_embedding
            )

            if self._node_types_combination == "Dot":
                node_types_concatenation = Dot(axes=2)
            elif self._node_types_combination == "Add":
                node_types_concatenation = Add(axes=2)
            elif self._node_types_combination == "Concatenate":
                node_types_concatenation = Concatenate(axes=2)
            else:
                raise ValueError(
                    "Supported node types concatenations are Dot, Add and Concatenate."
                )
            src_node_embedding = node_types_concatenation([
                src_node_embedding,
                source_node_types_embedding
            ])
            dst_node_embedding = node_types_concatenation([
                dst_node_embedding,
                destination_node_types_embedding
            ])

        if self._use_edge_types:
            edge_type_embedding = Embedding(
                input_dim=self._edge_types_number,
                output_dim=self._edge_type_embedding_size,
                input_length=1,
                name="edge_type_embedding_layer",
            )(edge_types_input)
            edge_type_embedding = UnitNorm()(Flatten()(edge_type_embedding))
        else:
            edge_type_embedding = None

        # Creating the actual model
        model = Model(
            inputs=input_layers,
            outputs=self._build_output(
                src_node_embedding,
                dst_node_embedding,
                edge_type_embedding,
                edge_types_input
            ),
            name=self._model_name
        )
        return model

    def _build_output(
        self,
        src_node_embedding: tf.Tensor,
        dst_node_embedding: tf.Tensor,
        edge_type_embedding: Optional[tf.Tensor] = None,
        edge_types_input: Optional[tf.Tensor] = None,
    ):
        """Return output of the model."""
        raise NotImplementedError(
            "The method _build_output must be implemented in the "
            "classes that extend the Siamese base class."
        )

    def _siamese_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        """Compute the siamese loss function.

        Parameters
        ---------------------------
        y_true: tf.Tensor,
            The true values Tensor for this batch.
        y_pred: tf.Tensor,
            The predicted values Tensor for this batch.

        Returns
        ---------------------------
        Loss function score related to this batch.
        """
        # TODO: check what happens with and without relu
        y_true = tf.cast(y_true, "float32")
        return self._relu_bias + K.mean(
            (1 - 2 * y_true) * y_pred,
            axis=-1
        )

    def get_embedding_dataframe(self) -> pd.DataFrame:
        """Return terms embedding using given index names."""
        return super().get_embedding_dataframe(self._graph.get_node_names())

    def _compile_model(self) -> Model:
        """Compile model."""
        self._model.compile(
            loss=self._siamese_loss,
            optimizer=self._optimizer
        )

    @property
    def embedding(self) -> np.ndarray:
        """Return model embeddings.

        Raises
        -------------------
        NotImplementedError,
            If the current embedding model does not have an embedding layer.
        """
        # TODO create multiple getters for the various embedding layers.
        return Embedder.embedding.fget(self)  # pylint: disable=no-member

    def fit(
        self,
        batch_size: int = 2**12,
        negative_samples_rate: float = 1.0,
        avoid_false_negatives: bool = False,
        support_mirror_strategy: bool = False,
        graph_to_avoid: EnsmallenGraph = None,
        batches_per_epoch: Union[int, str] = "auto",
        elapsed_epochs: int = 0,
        epochs: int = 100,
        early_stopping_monitor: str = "loss",
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 10,
        early_stopping_mode: str = "min",
        reduce_lr_monitor: str = "loss",
        reduce_lr_min_delta: float = 0.001,
        reduce_lr_patience: int = 2,
        reduce_lr_mode: str = "min",
        reduce_lr_factor: float = 0.9,
        verbose: int = 2,
        **kwargs: Dict
    ) -> pd.DataFrame:
        """Return pandas dataframe with training history.

        Parameters
        -----------------------
        graph: EnsmallenGraph,
            Graph to embed.
        epochs: int = 100,
            Epochs to train the model for.
        early_stopping_monitor: str = "loss",
            Metric to monitor for early stopping.
        early_stopping_min_delta: float = 0.1,
            Minimum delta of metric to stop the training.
        early_stopping_patience: int = 5,
            Number of epochs to wait for when the given minimum delta is not
            achieved after which trigger early stopping.
        early_stopping_mode: str = "min",
            Direction of the variation of the monitored metric for early stopping.
        reduce_lr_monitor: str = "loss",
            Metric to monitor for reducing learning rate.
        reduce_lr_min_delta: float = 1,
            Minimum delta of metric to reduce learning rate.
        reduce_lr_patience: int = 3,
            Number of epochs to wait for when the given minimum delta is not
            achieved after which reducing learning rate.
        reduce_lr_mode: str = "min",
            Direction of the variation of the monitored metric for learning rate.
        reduce_lr_factor: float = 0.9,
            Factor for reduction of learning rate.
        verbose: int = 2,
            Wethever to show the loading bar.
            Specifically, the options are:
            * 0 or False: No loading bar.
            * 1 or True: Showing only the loading bar for the epochs.
            * 2: Showing loading bar for both epochs and batches.
        **kwargs: Dict,
            Additional kwargs to pass to the Keras fit call.

        Returns
        -----------------------
        Dataframe with training history.
        """
        return super().fit(
            EdgePredictionSequence(
                graph=self._graph,
                batch_size=batch_size,
                avoid_false_negatives=avoid_false_negatives,
                support_mirror_strategy=support_mirror_strategy,
                graph_to_avoid=graph_to_avoid,
                use_node_types=self._use_node_types,
                use_edge_types=self._use_edge_types,
                negative_samples_rate=negative_samples_rate,
                elapsed_epochs=elapsed_epochs,
                batches_per_epoch=batches_per_epoch
            ),
            epochs=epochs,
            early_stopping_monitor=early_stopping_monitor,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_patience=early_stopping_patience,
            early_stopping_mode=early_stopping_mode,
            reduce_lr_monitor=reduce_lr_monitor,
            reduce_lr_min_delta=reduce_lr_min_delta,
            reduce_lr_patience=reduce_lr_patience,
            reduce_lr_mode=reduce_lr_mode,
            reduce_lr_factor=reduce_lr_factor,
            verbose=verbose,
            **kwargs
        )
