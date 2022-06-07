"""Abstract class for graph embedding models."""
from typing import Dict, Union, Optional, Tuple, Any

import numpy as np
import pandas as pd
from ensmallen import Graph
import tensorflow as tf
from tensorflow.keras import Model
from embiggen.sequences.tensorflow_sequences import Node2VecSequence
from embiggen.embedders.tensorflow_embedders.abstract_random_walked_based_embedder_model import AbstractRandomWalkBasedEmbedderModel
from embiggen.utils.abstract_models import abstract_class, EmbeddingResult


@abstract_class
class Node2Vec(AbstractRandomWalkBasedEmbedderModel):
    """Abstract class for sequence embedding models."""

    def __init__(
        self,
        number_of_negative_samples: int = 5,
        batch_size: int = 128,
        embedding_size: int = 100,
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
        use_mirrored_strategy: bool = False,
        enable_cache: bool = False
    ):
        """Create new abstract Node2Vec model.

        Parameters
        -------------------------------
        number_of_negative_samples: int = 5
            The number of negative classes to randomly sample per batch.
            This single sample of negative classes is evaluated for each element in the batch.
        batch_size: int = 128
            The number of nodes to consider for each walk.
        embedding_size: int = 100
            Dimension of the embedding.
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
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._number_of_negative_samples = number_of_negative_samples

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
            batch_size=batch_size,
            optimizer=optimizer,
            use_mirrored_strategy=use_mirrored_strategy,
            enable_cache=enable_cache
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return dict(
            **super().parameters(),
            number_of_negative_samples=self._number_of_negative_samples,
        )

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **AbstractRandomWalkBasedEmbedderModel.smoke_test_parameters(),
            number_of_negative_samples=1
        )

    def _get_steps_per_epoch(self, graph: Graph) -> Tuple[Any]:
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

        return (Node2VecSequence(
            graph,
            walk_length=self._walk_length,
            batch_size=self._batch_size,
            iterations=self._iterations,
            window_size=self._window_size,
            return_weight=self._return_weight,
            explore_weight=self._explore_weight,
            change_node_type_weight=self._change_node_type_weight,
            change_edge_type_weight=self._change_edge_type_weight,
            max_neighbours=self._max_neighbours,
            random_state=self._random_state,
        ).into_dataset()\
            .repeat()\
            .prefetch(AUTOTUNE), )

    @staticmethod
    def requires_nodes_sorted_by_decreasing_node_degree() -> bool:
        return True

    def _extract_embeddings(
        self,
        graph: Graph,
        model: Model,
        return_dataframe: bool
    ) -> EmbeddingResult:
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
        node_embedding = self.get_layer_weights(
            "node_embedding",
            model
        )
        if return_dataframe:
            node_embedding = pd.DataFrame(
                node_embedding,
                index=graph.get_node_names()
            )
        
        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            node_embeddings=node_embedding
        )