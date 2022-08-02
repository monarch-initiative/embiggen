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
        embedding_size: int = 100,
        negative_samples_rate: float = 0.5,
        epochs: int = 500,
        batch_size: int = 2**10,
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 10,
        learning_rate_plateau_min_delta: float = 0.001,
        learning_rate_plateau_patience: int = 5,
        use_mirrored_strategy: bool = False,
        activation: str = "sigmoid",
        loss: str = "binary_crossentropy",
        optimizer: str = "nadam",
        verbose: bool = False,
        enable_cache: bool = False,
        random_state: int = 42
    ):
        """Create new Edge-predicton based model.

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
        epochs: int = 500
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
        activation: str = "sigmoid"
            The activation to be used.
            For LINE models, this is the Sigmoid, while
            for the HOPE models this is a linear activation.
        loss: str = "binary_crossentropy"
            Loss to be minimezed during the training of the model.
            For LINE models, this is a  binary ceossentropy (since the output is linear)
            while for the HOPE models this is an Mean squared error.
        optimizer: str = "nadam"
            The optimizer to be used during the training of the model.
        verbose: bool = False
            Whether to show the loading bar while training the model.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        random_state: Optional[int] = None
            The random state to use if the model is stocastic.
        """
        self._negative_samples_rate = negative_samples_rate
        self._activation = activation
        self._loss = loss

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
                negative_samples_rate=self._negative_samples_rate,
                activation=self._activation
            )
        )

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
            loss=self._loss,
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

    def _build_sequence(
        self,
        graph: Graph,
    ) -> EdgePredictionTrainingSequence:
        """Returns values to be fed as input into the model.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        return EdgePredictionTrainingSequence(
            graph=graph,
            negative_samples_rate=self._negative_samples_rate,
            batch_size=self._batch_size,
        )

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
        return (self._build_sequence(graph), )

    @classmethod
    def requires_nodes_sorted_by_decreasing_node_degree(cls) -> bool:
        return False

    @classmethod
    def is_topological(cls) -> bool:
        return True

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        return False

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False

    @classmethod
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can optionally use edge types."""
        return False
