"""Abstract class to implement embedding methods based on edge prediction."""
from typing import Dict, Tuple, Any, Optional, Union, List

import numpy as np
import tensorflow as tf
from ensmallen import Graph
from tensorflow.keras.layers import (  # pylint: disable=import-error,no-name-in-module
    Input
)
from tensorflow.keras.models import Model
from embiggen.utils.abstract_models import abstract_class
from embiggen.sequences.tensorflow_sequences import EdgePredictionTrainingSequence
from embiggen.embedders.tensorflow_embedders.tensorflow_embedder import TensorFlowEmbedder


@abstract_class
class EdgePredictionBasedTensorFlowEmbedders(TensorFlowEmbedder):
    """Abstract class to implement embedding methods based on edge prediction."""

    def __init__(
        self,
        embedding_size: int = 500,
        negative_samples_rate: float = 0.5,
        epochs: int = 200,
        batch_size: int = 2**10,
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 10,
        learning_rate_plateau_min_delta: float = 0.001,
        learning_rate_plateau_patience: int = 5,
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
        negative_samples_rate: float = 0.5
            Factor of negatives to use in every batch.
            For example, with a batch size of 128 and negative_samples_rate equal
            to 0.5, there will be 64 positives and 64 negatives.
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
        self._negative_samples_rate = negative_samples_rate

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
                negative_samples_rate=self._negative_samples_rate
            )
        }

    def _build_edge_prediction_based_model(
        self,
        graph: Graph,
        sources: tf.Tensor,
        destinations: tf.Tensor
    ) -> Union[List[tf.Tensor], tf.Tensor]:
        """Return the model implementation.

        Parameters
        -------------------
        graph: Graph

        sources: tf.Tensor
            The source nodes to be used in the model.
        destinations: tf.Tensor
            The destinations nodes to be used in the model.
        """
        raise NotImplementedError(
            "The method `_build_edge_prediction_based_model` has not been "
            f"implemented in the model `{self.model_name()}`, library `{self.library_name()}`, "
            f"specifically with the class name {self.__class__.__name__}."
        )

    def _build_model(self, graph: Graph):
        """Return Siamese model."""
        # Creating the inputs layers
        source_nodes = Input(
            shape=(1,),
            dtype=tf.int32,
            name="Sources"
        )
        destination_nodes = Input(
            shape=(1,),
            dtype=tf.int32,
            name="Destinations"
        )

        # Creating the actual model
        model = Model(
            inputs=[source_nodes, destination_nodes],
            outputs=self._build_edge_prediction_based_model(
                graph=graph,
                sources=source_nodes,
                destinations=destination_nodes
            ),
            name=self.model_name().replace(" ", "")
        )

        model.compile(
            optimizer=self._optimizer,
            loss="binary_crossentropy",
            metrics="accuracy"
        )

        return model

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

        sequence = EdgePredictionTrainingSequence(
            graph=graph,
            negative_samples_rate=self._negative_samples_rate,
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
    def requires_node_types() -> bool:
        return False

    @staticmethod
    def requires_edge_types() -> bool:
        return False

    @staticmethod
    def requires_positive_edge_weights() -> bool:
        return False

    @staticmethod
    def can_use_node_types() -> bool:
        """Returns whether the model can optionally use node types."""
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
        return False

    def is_using_edge_types(self) -> bool:
        """Returns whether the model is parametrized to use edge types."""
        return False
