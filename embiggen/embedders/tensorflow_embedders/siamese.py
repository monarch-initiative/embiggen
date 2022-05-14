"""Siamese network for node-embedding including optionally node types and edge types."""
from re import S
import warnings
from typing import Dict, List, Union, Tuple, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from ensmallen import Graph
from tensorflow.keras import \
    backend as K  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.constraints import \
    UnitNorm  # pylint: disable=import-error,no-name-in-module,no-name-in-module
from tensorflow.keras.layers import \
    Embedding  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import (  # pylint: disable=import-error,no-name-in-module
    Add,
    GlobalAveragePooling1D,
    Input
)
from tensorflow.keras.models import Model
from embiggen.utils import abstract_class
from ...sequences import EdgePredictionSequence
from .tensorflow_embedder import TensorFlowEmbedder


@abstract_class
class Siamese(TensorFlowEmbedder):
    """Siamese network for node-embedding including optionally node types and edge types."""

    def __init__(
        self,
        embedding_size: int = 100,
        node_type_embedding_size: int = 100,
        edge_type_embedding_size: int = 100,
        relu_bias: float = 1.0,
        epochs: int = 10,
        batch_size: int = 2**14,
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 1,
        learning_rate_plateau_min_delta: float = 0.001,
        learning_rate_plateau_patience: int = 1,
        use_mirrored_strategy: bool = False,
        optimizer: str = "sgd",
    ):
        """Create new sequence TensorFlowEmbedder model.

        Parameters
        -------------------------------------------
        embedding_size: int = 100
            Dimension of the embedding.
            If None, the seed embedding must be provided.
            It is not possible to provide both at once.
        node_type_embedding_size: int = 100
            Dimension of the embedding for the node types.
        edge_type_embedding_size: int = 100
            Dimension of the embedding for the edge types.
        relu_bias: float = 1.0
            The bias to use for the ReLu.
        epochs: int = 10
            Number of epochs to train the model for.
        batch_size: int = 2**14
            Batch size to use during the training.
        early_stopping_min_delta: float = 0.001
            The minimum variation in the provided patience time
            of the loss to not stop the training.
        early_stopping_patience: int = 1
            The amount of epochs to wait for better training
            performance.
        learning_rate_plateau_min_delta: float = 0.001
            The minimum variation in the provided patience time
            of the loss to not reduce the learning rate.
        learning_rate_plateau_patience: int = 1
            The amount of epochs to wait for better training
            performance without decreasing the learning rate.
        use_mirrored_strategy: bool = False
            Whether to use mirrored strategy.
        optimizer: str = "sgd"
            The optimizer to be used during the training of the model.
        """
        self._node_type_embedding_size = node_type_embedding_size
        self._edge_type_embedding_size = edge_type_embedding_size
        self._relu_bias = relu_bias

        super().__init__(
            embedding_size=embedding_size,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_patience=early_stopping_patience,
            learning_rate_plateau_min_delta=learning_rate_plateau_min_delta,
            learning_rate_plateau_patience=learning_rate_plateau_patience,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            use_mirrored_strategy=use_mirrored_strategy,
        )

    def _build_model(self, graph: Graph):
        """Return Siamese model."""
        # Creating the inputs layers
        srcs, dsts = Input((1,)), Input((1,))
        srcs_node_types = dsts_node_types = edge_types_input = None

        self._multilabel_node_types = graph.has_multilabel_node_types()
        self._max_node_types = graph.get_maximum_multilabel_count()
        self._node_types_number = graph.get_node_types_number()

        # Creating the embedding layer for the contexts
        node_embedding_layer = Embedding(
            input_dim=graph.get_nodes_number(),
            output_dim=self._embedding_size,
            input_length=1,
            name="node_embedding"
        )

        # Get the node embedding
        srcs_embedding = UnitNorm(axis=-1)(node_embedding_layer(srcs))
        dsts_embedding = UnitNorm(axis=-1)(node_embedding_layer(dsts))

        if self.requires_node_types():
            max_node_types = graph.get_maximum_multilabel_count()
            multilabel = graph.has_multilabel_node_types()
            srcs_node_types = Input((max_node_types,),)
            dsts_node_types = Input((max_node_types,),)
            node_types_number = (
                graph.get_node_types_number() +
                int(multilabel)
            )
            node_type_embedding_layer = Embedding(
                input_dim=node_types_number,
                output_dim=self._node_type_embedding_size,
                input_length=self._max_node_types,
                name="node_type_embedding",
                mask_zero=multilabel
            )

            srcs_node_types_embedding = node_type_embedding_layer(
                srcs_node_types
            )
            dsts_node_types_embedding = node_type_embedding_layer(
                dsts_node_types
            )

            if multilabel:
                global_average_layer = GlobalAveragePooling1D()
                srcs_node_types_embedding = global_average_layer(
                    srcs_node_types_embedding
                )
                dsts_node_types_embedding = global_average_layer(
                    dsts_node_types_embedding
                )

            srcs_embedding = UnitNorm(axis=-1)(Add()([
                srcs_embedding,
                srcs_node_types_embedding
            ]))
            dsts_embedding = UnitNorm(axis=-1)(Add()([
                dsts_embedding,
                dsts_node_types_embedding
            ]))

        if self.requires_edge_types():
            edge_types = Input((1,))
            edge_type_embedding = UnitNorm(axis=-1)(Embedding(
                input_dim=self._edge_types_number,
                output_dim=self._edge_type_embedding_size,
                input_length=1,
                name="edge_type_embedding",
            )(edge_types))

        (dsts_embedding, dsts_embedding) = self._build_output(
            srcs_embedding,
            dsts_embedding,
            edge_type_embedding,
            edge_types_input
        )

        output = K.sum(K.square(srcs_embedding - dsts_embedding), axis=-1)

        # Creating the actual model
        model = Model(
            inputs=[
                input_layer
                for input_layer in (
                    srcs,
                    srcs_node_types,
                    dsts,
                    dsts_node_types,
                    edge_types
                )
                if input_layer is not None
            ],
            outputs=output,
            name=self.model_name()
        )

        model.compile(
            loss=self._siamese_loss,
            optimizer=self._optimizer
        )

        return model

    def _build_output(
        self,
        source_node_embedding: tf.Tensor,
        destination_node_embedding: tf.Tensor,
        *args: List[tf.Tensor],
        **kwargs: Dict[str, tf.Tensor],
    ):
        """Return output of the model.

        Parameters
        ----------------------
        source_node_embedding: tf.Tensor,
            Embedding of the source node.
        destination_node_embedding: tf.Tensor,
            Embedding of the destination node.
        args: List[tf.Tensor],
            Additional tensors that may be used
            in subclasses of this model.
        kwargs: Dict[str, tf.Tensor],
            Additional tensors that may be used
            in subclasses of this model.

        Returns
        ----------------------
        The distance for the Siamese network.
        """
        raise NotImplementedError(
            "The method `_build_output` should be implemented in the child "
            "classes of the Siamese model, and is missing in the class "
            f"called {self.__class__.__name__}."
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
        return K.relu(self._relu_bias + (1 - 2 * tf.cast(y_true, "float32")) * y_pred)

    def _get_steps_per_epoch(self, graph: Graph) -> int:
        """Returns number of steps per epoch.

        Parameters
        ------------------
        graph: Graph
            The graph to compute the number of steps.
        """
        return max(graph.get_nodes_number() // self._batch_size, 1)

    def _build_input(
        self,
        graph: Graph,
        verbose: bool
    ) -> Tuple[np.ndarray]:
        """Returns values to be fed as input into the model.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        verbose: bool
            Whether to show loading bars.
            Not used in this context.
        """
        try:
            AUTOTUNE = tf.data.AUTOTUNE
        except:
            AUTOTUNE = tf.data.experimental.AUTOTUNE

        return (
            EdgePredictionSequence(
                graph=graph,
                batch_size=self._batch_size,
                avoid_false_negatives=False,
                use_node_types=self.requires_node_types(),
                use_edge_types=self.requires_edge_types(),
            ).into_dataset()
            .repeat()
            .prefetch(AUTOTUNE), )

    @staticmethod
    def requires_nodes_sorted_by_decreasing_node_degree() -> bool:
        return False

    def _extract_embeddings(
        self,
        graph: Graph,
        model: Model,
        return_dataframe: bool
    ) -> Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
        """Returns embedding from the model.

        Parameters
        ------------------
        graph: Graph
            The graph that was embedded.
        model: Model
            The Keras model used to embed the graph.
        return_dataframe: bool
            Whether to return a dataframe of a numpy array.
        """
        embedding = self.get_layer_weights(
            "node_embedding",
            model
        )
        if return_dataframe:
            return pd.DataFrame(
                embedding,
                index=graph.get_node_names()
            )
        return embedding