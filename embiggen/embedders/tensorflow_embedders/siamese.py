"""Siamese network for node-embedding including optionally node types and edge types."""
from typing import Dict, Tuple, Any, Optional

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
from embiggen.utils.abstract_models import abstract_class
from embiggen.sequences.tensorflow_sequences import SiameseSequence, KGSiameseSequence
from embiggen.embedders.tensorflow_embedders.tensorflow_embedder import TensorFlowEmbedder


@abstract_class
class Siamese(TensorFlowEmbedder):
    """Siamese network for node-embedding including optionally node types and edge types."""

    def __init__(
        self,
        embedding_size: int = 100,
        relu_bias: float = 1.0,
        epochs: int = 10,
        batch_size: int = 2**10,
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 5,
        learning_rate_plateau_min_delta: float = 0.001,
        learning_rate_plateau_patience: int = 2,
        use_mirrored_strategy: bool = False,
        optimizer: str = "nadam",
        enable_cache: bool = False
    ):
        """Create new sequence Siamese model.

        Parameters
        -------------------------------------------
        embedding_size: int = 100
            Dimension of the embedding.
            If None, the seed embedding must be provided.
            It is not possible to provide both at once.
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
        optimizer: str = "nadam"
            The optimizer to be used during the training of the model.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
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
            enable_cache=enable_cache
        )

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **TensorFlowEmbedder.smoke_test_parameters(),
        )

    def parameters(self) -> Dict[str, Any]:
        return {
            **super().parameters(),
            **dict(
                relu_bias = self._relu_bias
            )
        }

    def _build_model(self, graph: Graph):
        """Return Siamese model."""
        # Creating the inputs layers
        inputs = [
            Input((1,), dtype=tf.int32)
            for _ in range(4)
        ]

        edge_types = Input((1,), dtype=tf.int32)

        # Creating the embedding layer for the contexts
        node_embedding_layer = Embedding(
            input_dim=graph.get_nodes_number(),
            output_dim=self._embedding_size,
            input_length=1,
            name="node_embeddings"
        )

        # Get the node embedding
        node_embeddings = [
            UnitNorm(axis=-1)(node_embedding_layer(node_input))
            for node_input in inputs
        ]

        if self.requires_node_types():
            max_node_types = graph.get_maximum_multilabel_count()
            multilabel = graph.has_multilabel_node_types()
            unknown_node_types = graph.has_unknown_node_types()
            node_types_offset = int(multilabel or unknown_node_types)
            node_type_inputs = [
                Input((max_node_types,), dtype=tf.int32)
                for _ in range(4)
            ]

            node_type_embedding_layer = Embedding(
                input_dim=graph.get_node_types_number() + node_types_offset,
                output_dim=self._embedding_size,
                input_length=max_node_types,
                name="node_type_embeddings",
                mask_zero=multilabel or unknown_node_types
            )

            node_embeddings = [
                Add()([
                    GlobalAveragePooling1D()(
                        node_type_embedding_layer(node_type_input)
                    ),
                    node_embedding
                ])
                for node_type_input, node_embedding in zip(
                    node_type_inputs,
                    node_embeddings
                )
            ]
        else:
            node_type_inputs = []

        inputs.extend(node_type_inputs)
        inputs.append(edge_types)

        edge_types_number = graph.get_edge_types_number()
        unknown_edge_types = graph.has_unknown_edge_types()
        edge_types_offset = int(unknown_edge_types)
        edge_type_embedding = GlobalAveragePooling1D()(Embedding(
            input_dim=edge_types_number,
            output_dim=self._embedding_size,
            input_length=1 + edge_types_offset,
            mask_zero=unknown_edge_types,
            name="edge_type_embeddings",
        )(edge_types))

        (
            srcs_embedding,
            dsts_embedding,
            not_srcs_embedding,
            not_dsts_embedding,
            edge_type_embedding
        ) = self._build_output(
            *node_embeddings,
            edge_type_embedding,
            edge_types
        )

        loss = K.relu(self._relu_bias + tf.norm(
            srcs_embedding + edge_type_embedding - dsts_embedding,
            axis=-1
        ) - tf.norm(
            not_srcs_embedding + edge_type_embedding - not_dsts_embedding,
            axis=-1
        ))

        # Creating the actual model
        model = Model(
            inputs=inputs,
            outputs=loss,
            name=self.model_name()
        )

        model.add_loss(loss)

        model.compile(optimizer=self._optimizer)

        return model

    def _build_output(
        self,
        srcs_embedding: tf.Tensor,
        dsts_embedding: tf.Tensor,
        not_srcs_embedding: tf.Tensor,
        not_dsts_embedding: tf.Tensor,
        edge_type_embedding: tf.Tensor,
        edge_type_input: Optional[Input]
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Returns the five input tensors, arbitrarily changed.

        Parameters
        ----------------------
        srcs_embedding: tf.Tensor
            Embedding of the source node.
        dsts_embedding: tf.Tensor
            Embedding of the destination node.
        not_srcs_embedding: tf.Tensor
            Embedding of the fake source node.
        not_dsts_embedding: tf.Tensor
            Embedding of the fake destination node.
        edge_type_embedding: tf.Tensor
            Embedding of the edge types.
        edge_type_input: Optional[Input]
            Input of the edge types. This is not None
            only when there is one such input in the
            model, when edge types are requested.
        """
        raise NotImplementedError(
            "The method `_build_output` should be implemented in the child "
            "classes of the Siamese model, and is missing in the class "
            f"called {self.__class__.__name__}."
        )

    def _get_steps_per_epoch(self, graph: Graph) -> int:
        """Returns number of steps per epoch.

        Parameters
        ------------------
        graph: Graph
            The graph to compute the number of steps.
        """
        return max(graph.get_number_of_directed_edges() // self._batch_size, 1)

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

        if self.requires_node_types():
            sequence = KGSiameseSequence(
                graph=graph,
                batch_size=self._batch_size,
            )
        else:
            sequence = SiameseSequence(
                graph=graph,
                batch_size=self._batch_size,
            )
        return (
            sequence.into_dataset()
            .repeat()
            .prefetch(AUTOTUNE), )

    @staticmethod
    def requires_nodes_sorted_by_decreasing_node_degree() -> bool:
        return False

    @staticmethod
    def is_topological() -> bool:
        return True

    @staticmethod
    def requires_edge_weights() -> bool:
        return False

    @staticmethod
    def requires_positive_edge_weights() -> bool:
        return False

    @staticmethod
    def can_use_edge_weights() -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False

    def is_using_edge_weights(self) -> bool:
        """Returns whether the model is parametrized to use edge weights."""
        return False

    @staticmethod
    def can_use_edge_types() -> bool:
        """Returns whether the model can optionally use edge types."""
        return True

    def is_using_edge_types(self) -> bool:
        """Returns whether the model is parametrized to use edge types."""
        return True