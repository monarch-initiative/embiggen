"""Siamese network for node-embedding including optionally node types and edge types."""
from typing import Dict, Tuple, Any, Optional

import numpy as np
import tensorflow as tf
from ensmallen import Graph
from tensorflow.keras import \
    backend as K  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import (  # pylint: disable=import-error,no-name-in-module
    Input, ReLU
)
from tensorflow.keras.models import Model
from embiggen.layers.tensorflow import FlatEmbedding, ElementWiseL2, ElementWiseL1
from embiggen.utils.abstract_models import abstract_class
from embiggen.sequences.tensorflow_sequences import SiameseSequence
from embiggen.embedders.tensorflow_embedders.tensorflow_embedder import TensorFlowEmbedder


@abstract_class
class Siamese(TensorFlowEmbedder):
    """Siamese network for node-embedding including optionally node types and edge types."""

    def __init__(
        self,
        embedding_size: int = 100,
        relu_bias: float = 1.0,
        epochs: int = 50,
        batch_size: int = 2**8,
        early_stopping_min_delta: float = 0.0001,
        early_stopping_patience: int = 10,
        learning_rate_plateau_min_delta: float = 0.0001,
        learning_rate_plateau_patience: int = 5,
        norm: str = "L2",
        use_mirrored_strategy: bool = False,
        optimizer: str = "nadam",
        verbose: bool = False,
        enable_cache: bool = False,
        random_state: int = 42
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
        epochs: int = 50
            Number of epochs to train the model for.
        batch_size: int = 2**8
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
        norm: str = "L2"
            Normalization to use.
            Can either be `L1` or `L2`.
        use_mirrored_strategy: bool = False
            Whether to use mirrored strategy.
        optimizer: str = "nadam"
            The optimizer to be used during the training of the model.
        verbose: bool = False
            Whether to show loading bars.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        random_state: Optional[int] = None
            The random state to use if the model is stocastic.
        """
        self._relu_bias = relu_bias
        self._norm = norm
        super().__init__(
            embedding_size=embedding_size,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_patience=early_stopping_patience,
            learning_rate_plateau_min_delta=learning_rate_plateau_min_delta,
            learning_rate_plateau_patience=learning_rate_plateau_patience,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            verbose=verbose,
            use_mirrored_strategy=use_mirrored_strategy,
            enable_cache=enable_cache,
            random_state=random_state
        )

    def parameters(self) -> Dict[str, Any]:
        return dict(
            **super().parameters(),
            **dict(
                relu_bias=self._relu_bias,
                norm=self._norm
            )
        )

    def _build_model(self, graph: Graph):
        """Return Siamese model."""
        # Creating the inputs layers
        inputs = [
            Input((1,), dtype=tf.int32, name=node_name)
            for node_name in (
                "Sources",
                "Destinations",
                "Corrupted Sources",
                "Corrupted Destinations",
            )
        ]

        # Creating the embedding layer for the contexts
        node_embedding_layer = FlatEmbedding(
            vocabulary_size=graph.get_number_of_nodes(),
            dimension=self._embedding_size,
            input_length=1,
            name="NodeEmbedding"
        )

        # Get the node embedding
        node_embeddings = [
            node_embedding_layer(node_input)
            for node_input in inputs
        ]

        (
            edge_types,
            regularization,
            srcs_embedding,
            dsts_embedding,
            not_srcs_embedding,
            not_dsts_embedding,
        ) = self._build_output(
            *node_embeddings,
            graph
        )

        if self._norm == "L2":
            norm_layer = ElementWiseL2
        else:
            norm_layer = ElementWiseL1

        if dsts_embedding is not None:
            srcs_embedding = norm_layer()([
                srcs_embedding,
                dsts_embedding
            ])

        if not_dsts_embedding is not None:
            not_srcs_embedding = norm_layer()([
                not_srcs_embedding,
                not_dsts_embedding
            ])

        loss = ReLU()(
            self._relu_bias + srcs_embedding - not_srcs_embedding
        ) + regularization

        if edge_types is not None:
            inputs.append(edge_types)

        # Creating the actual model
        model = Model(
            inputs=inputs,
            outputs=loss,
            name=self.model_name().replace(" ", "")
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
        graph: Graph
    ) -> Tuple[Optional[Input], tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Returns the inputs, if any, the regularization loss, and the received node embedding arbitrarily modified.

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
        graph: Graph
            Graph whose structure is to be used to build
            the model.
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
    ) -> Tuple[np.ndarray]:
        """Returns values to be fed as input into the model.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        sequence = SiameseSequence(
            graph=graph,
            batch_size=self._batch_size,
            return_edge_types=self.requires_edge_types()
        )
        return (
            sequence.into_dataset()
            .repeat()
            .prefetch(tf.data.AUTOTUNE), )

    @classmethod
    def requires_nodes_sorted_by_decreasing_node_degree(cls) -> bool:
        return False

    @classmethod
    def is_topological(cls) -> bool:
        return True

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False
