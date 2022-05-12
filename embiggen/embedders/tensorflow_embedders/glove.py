"""GloVe model for graph and words embedding."""
from typing import Dict, List, Tuple, Union, Any

import numpy as np
import pandas as pd
from ensmallen import Graph
import tensorflow as tf
from tensorflow.keras import backend as K  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import Add  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import Dot, Embedding, Flatten, Input  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.models import Model  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.optimizers import \
    Optimizer, Nadam  # pylint: disable=import-error,no-name-in-module

from .abstract_random_walked_based_embedder_model import AbstractRandomWalkBasedEmbedderModel


class GloVe(AbstractRandomWalkBasedEmbedderModel):
    """GloVe model for graph and words embedding.

    The GloVe model for graph embedding receives two words and is asked to
    predict its cooccurrence probability.
    """

    SOURCE_NODES_EMBEDDING = "SOURCE_NODES_EMBEDDING"
    DESTINATION_NODES_EMBEDDING = "DESTINATION_NODES_EMBEDDING"

    def __init__(
        self,
        embedding_size: int = 100,
        alpha: float = 0.75,
        use_bias: bool = True,
        siamese: bool = False,
        epochs: int = 10,
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 5,
        learning_rate_plateau_min_delta: float = 0.001,
        learning_rate_plateau_patience: int = 3,
        window_size: int = 4,
        walk_length: int = 128,
        iterations: int = 1,
        return_weight: float = 1.0,
        explore_weight: float = 1.0,
        change_node_type_weight: float = 1.0,
        change_edge_type_weight: float = 1.0,
        max_neighbours: int = 100,
        normalize_by_degree: bool = False,
        random_state: int = 42,
        optimizer: str = "sgd",
        use_mirrored_strategy: bool = False
    ):
        """Create new GloVe-based TensorFlowEmbedder object.

        Parameters
        -------------------------------
        embedding_size: int = 100
            Dimension of the embedding.
        alpha: float = 0.75
            Alpha to use for the function.
        use_bias: bool = True
            Whether to use the bias in the GloVe model.
            Consider that these weights are excluded from
            the model embedding.
        siamese: bool = False
            Whether to use the siamese modality and share the embedding
            weights between the source and destination nodes.
        epochs: int = 10
            Number of epochs to train the model for.
        early_stopping_min_delta: float
            The minimum variation in the provided patience time
            of the loss to not stop the training.
        early_stopping_patience: int
            The amount of epochs to wait for better training
            performance.
        learning_rate_plateau_min_delta: float
            The minimum variation in the provided patience time
            of the loss to not reduce the learning rate.
        learning_rate_plateau_patience: int
            The amount of epochs to wait for better training
            performance without decreasing the learning rate.
        window_size: int = 4
            Window size for the local context.
            On the borders the window size is trimmed.
        walk_length: int = 128
            Maximal length of the walks.
        iterations: int = 1
            Number of iterations of the single walks.
        return_weight: float = 1.0
            Weight on the probability of returning to the same node the walk just came from
            Having this higher tends the walks to be
            more like a Breadth-First Search.
            Having this very high  (> 2) makes search very local.
            Equal to the inverse of p in the Node2Vec paper.
        explore_weight: float = 1.0
            Weight on the probability of visiting a neighbor node
            to the one we're coming from in the random walk
            Having this higher tends the walks to be
            more like a Depth-First Search.
            Having this very high makes search more outward.
            Having this very low makes search very local.
            Equal to the inverse of q in the Node2Vec paper.
        change_node_type_weight: float = 1.0
            Weight on the probability of visiting a neighbor node of a
            different type than the previous node. This only applies to
            colored graphs, otherwise it has no impact.
        change_edge_type_weight: float = 1.0
            Weight on the probability of visiting a neighbor edge of a
            different type than the previous edge. This only applies to
            multigraphs, otherwise it has no impact.
        max_neighbours: int = 100
            Number of maximum neighbours to consider when using approximated walks.
            By default, None, we execute exact random walks.
            This is mainly useful for graphs containing nodes with high degrees.
        normalize_by_degree: bool = False
            Whether to normalize the random walk by the node degree
            of the destination node degrees.
        random_state: int = 42
            The random state to reproduce the training sequence.
        optimizer: str = "sgd"
            Optimizer to use during the training.
        use_mirrored_strategy: bool = False
            Whether to use mirrored strategy.
        """
        self._alpha = alpha
        self._siamese = siamese
        self._use_bias = use_bias

        super().__init__(
            window_size=window_size,
            walk_length=walk_length,
            iterations=iterations,
            return_weight=return_weight,
            explore_weight=explore_weight,
            change_node_type_weight=change_node_type_weight,
            change_edge_type_weight=change_edge_type_weight,
            max_neighbours=max_neighbours,
            normalize_by_degree=normalize_by_degree,
            random_state=random_state,
            embedding_size=embedding_size,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_patience=early_stopping_patience,
            learning_rate_plateau_min_delta=learning_rate_plateau_min_delta,
            learning_rate_plateau_patience=learning_rate_plateau_patience,
            epochs=epochs,
            optimizer=optimizer,
            use_mirrored_strategy=use_mirrored_strategy,
        )

    def parameters(self) -> Dict[str, Any]:
        return {
            **super().parameters(),
            **dict(
                alpha=self._alpha,
                siamese=self._siamese,
                use_bias=self._use_bias
            )
        }

    def _glove_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        """Compute the glove loss function.

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
        return K.sum(
            K.pow(K.clip(y_true, 0.0, 1.0), self._alpha) *
            K.square(y_pred - K.log(y_true)),
            axis=-1
        )

    @staticmethod
    def model_name() -> str:
        """Returns name of the model."""
        return "GloVe"

    def _build_model(self, graph: Graph) -> Model:
        """Create new Glove model.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        # Creating the input layers
        sources = Input((1,), dtype=tf.int32)
        destinations = Input((1,), dtype=tf.int32)

        sources_embedding_layer = Embedding(
            input_dim=graph.get_nodes_number(),
            output_dim=self._embedding_size,
            input_length=1,
            name=self.SOURCE_NODES_EMBEDDING
        )
        sources_embedding = sources_embedding_layer(sources)

        if self._siamese:
            destinations_embedding = sources_embedding_layer(destinations)
        else:
            destinations_embedding = Embedding(
                input_dim=graph.get_nodes_number(),
                output_dim=self._embedding_size,
                input_length=1,
                name=self.DESTINATION_NODES_EMBEDDING
            )(destinations)

        # Creating the dot product of the embedding layers
        prediction = Dot(axes=2)([
            sources_embedding,
            destinations_embedding
        ])

        # Creating the biases layer
        if self._use_bias:
            biases = [
                Embedding(graph.get_nodes_number(), 1,
                          input_length=1)(input_layer)
                for input_layer in (sources, destinations)
            ]
            prediction = Add()([prediction, *biases])

        # Creating the model
        model = Model(
            inputs=[sources, destinations],
            outputs=prediction,
            name=self.name()
        )

        model.compile(
            loss=self._glove_loss,
            optimizer=self._optimizer
        )

        return model

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
        """
        sources, destinations, frequencies = graph.cooccurence_matrix(
            walk_length=self._walk_length,
            window_size=self._window_size,
            iterations=self._iterations,
            return_weight=self._return_weight,
            explore_weight=self._explore_weight,
            change_edge_type_weight=self._change_edge_type_weight,
            change_node_type_weight=self._change_node_type_weight,
            max_neighbours=self._max_neighbours,
            random_state=self._random_state,
            normalize_by_degree=self._normalize_by_degree,
            verbose=verbose
        )

        return (
            (sources, destinations),
            frequencies
        )
