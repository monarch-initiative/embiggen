"""Abstract class for graph embedding models."""
from typing import Dict, Union, Optional

import numpy as np
import pandas as pd
from ensmallen import Graph
from tensorflow.keras.optimizers import Optimizer, Nadam  # pylint: disable=import-error,no-name-in-module
import tensorflow as tf
from ...sequences import Node2VecSequence
from .cbow import CBOW
from .skipgram import SkipGram
from ..embedders_utils import enforce_sorted_graph, detect_graph_node_embedding_oddities


class Node2Vec:
    """Abstract class for sequence embedding models."""

    def __init__(
        self,
        graph: Graph,
        word2vec_model: Union[CBOW, SkipGram],
        embedding_size: int = 100,
        embedding: Union[np.ndarray, pd.DataFrame] = None,
        optimizer: Union[str, Optimizer] = None,
        number_of_negative_samples: int = 5,
        walk_length: int = 128,
        batch_size: int = 256,
        iterations: int = 16,
        window_size: int = 4,
        return_weight: float = 1.0,
        explore_weight: float = 1.0,
        change_node_type_weight: float = 1.0,
        change_edge_type_weight: float = 1.0,
        max_neighbours: Optional[int] = 100,
        elapsed_epochs: int = 0,
        random_state: int = 42,
        dense_node_mapping: Optional[Dict[int, int]] = None,
        use_gradient_centralization: bool = True,
        siamese: bool = False
    ):
        """Create new sequence TensorFlowEmbedder model.

        Parameters
        -------------------------------------------
        graph: Graph,
            Graph to be embedded.
        word2vec_model: Word2Vec,
            Word2Vec model to use.
        embedding_size: int = 100,
            Dimension of the embedding.
        embedding: Union[np.ndarray, pd.DataFrame] = None,
            The seed embedding to be used.
            Note that it is not possible to provide at once both
            the embedding and either the vocabulary size or the embedding size.
        optimizer: Union[str, Optimizer] = None,
            The optimizer to be used during the training of the model.
            By default, if None is provided, Nadam with learning rate
            set at 0.01 is used.
        window_size: int = 4,
            Window size for the local context.
            On the borders the window size is trimmed.
        number_of_negative_samples: int = 5,
            The number of negative classes to randomly sample per batch.
            This single sample of negative classes is evaluated for each element in the batch.
        walk_length: int = 128,
            Maximal length of the walks.
        batch_size: int = 256,
            Number of nodes to include in a single batch.
        iterations: int = 16,
            Number of iterations of the single walks.
        window_size: int = 4,
            Window size for the local context.
            On the borders the window size is trimmed.
        return_weight: float = 1.0,
            Weight on the probability of returning to the same node the walk just came from
            Having this higher tends the walks to be
            more like a Breadth-First Search.
            Having this very high  (> 2) makes search very local.
            Equal to the inverse of p in the Node2Vec paper.
        explore_weight: float = 1.0,
            Weight on the probability of visiting a neighbor node
            to the one we're coming from in the random walk
            Having this higher tends the walks to be
            more like a Depth-First Search.
            Having this very high makes search more outward.
            Having this very low makes search very local.
            Equal to the inverse of q in the Node2Vec paper.
        change_node_type_weight: float = 1.0,
            Weight on the probability of visiting a neighbor node of a
            different type than the previous node. This only applies to
            colored graphs, otherwise it has no impact.
        change_edge_type_weight: float = 1.0,
            Weight on the probability of visiting a neighbor edge of a
            different type than the previous edge. This only applies to
            multigraphs, otherwise it has no impact.
        max_neighbours: Optional[int] = 100,
            Number of maximum neighbours to consider when using approximated walks.
            By default, None, we execute exact random walks.
            This is mainly useful for graphs containing nodes with extremely high degrees.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        random_state: int = 42,
            The random state to reproduce the training sequence.
        dense_node_mapping: Optional[Dict[int, int]] = None,
            Mapping to use for converting sparse walk space into a dense space.
            This object can be created using the method (available from the
            graph object created using Graph)
            called `get_dense_node_mapping` that returns a mapping from
            the non trap nodes (those from where a walk could start) and
            maps these nodes into a dense range of values.
        use_gradient_centralization: bool = True,
            Whether to wrap the provided optimizer into a normalized
            one that centralizes the gradient.
            It is automatically enabled if the current version of
            TensorFlow supports gradient transformers.
            More detail here: https://arxiv.org/pdf/2004.01461.pdf
        siamese: bool = False
            Whether to use the siamese modality and share the embedding
            weights between the source and destination nodes.
        """
        enforce_sorted_graph(graph)
        detect_graph_node_embedding_oddities(graph)
        self._graph = graph
        self._sequence = Node2VecSequence(
            self._graph,
            walk_length=walk_length,
            batch_size=batch_size,
            iterations=iterations,
            window_size=window_size,
            return_weight=return_weight,
            explore_weight=explore_weight,
            change_node_type_weight=change_node_type_weight,
            change_edge_type_weight=change_edge_type_weight,
            max_neighbours=max_neighbours,
            elapsed_epochs=elapsed_epochs,
            random_state=random_state,
            dense_node_mapping=dense_node_mapping,
        )
        self._model = word2vec_model(
            vocabulary_size=self._graph.get_nodes_number(),
            embedding=embedding,
            embedding_size=embedding_size,
            optimizer=Nadam(
                learning_rate=0.02
            ) if optimizer is None else optimizer,
            window_size=window_size,
            number_of_negative_samples=number_of_negative_samples,
            use_gradient_centralization=use_gradient_centralization,
            siamese=siamese,
        )

    def fit(
        self,
        epochs: int = 100,
        early_stopping_monitor: str = "loss",
        early_stopping_min_delta: float = 0.5,
        early_stopping_patience: int = 2,
        early_stopping_mode: str = "min",
        reduce_lr_monitor: str = "loss",
        reduce_lr_min_delta: float = 0.1,
        reduce_lr_patience: int = 2,
        reduce_lr_mode: str = "min",
        reduce_lr_factor: float = 0.9,
        verbose: int = 2,
        **kwargs: Dict
    ) -> pd.DataFrame:
        """Return pandas dataframe with training history.

        Parameters
        -----------------------
        epochs: int = 10000,
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
        try:
            AUTOTUNE = tf.data.AUTOTUNE
        except:
            AUTOTUNE = tf.data.experimental.AUTOTUNE
        return self._model.fit(
            self._sequence
                .into_dataset()
                .repeat()
                .prefetch(  # Overlap producer and consumer works
                    AUTOTUNE
                ),
            steps_per_epoch=self._sequence.steps_per_epoch,
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

    def summary(self):
        """Print model summary."""
        self._model.summary()

    @property
    def embedding(self) -> np.ndarray:
        """Return model embeddings."""
        return self._model.embedding

    @property
    def trainable(self) -> bool:
        """Return whether the embedding layer can be trained.

        Raises
        -------------------
        NotImplementedError,
            If the current embedding model does not have an embedding layer.
        """
        return self._model.trainable

    @trainable.setter
    def trainable(self, trainable: bool):
        """Set whether the embedding layer can be trained or not.

        Parameters
        -------------------
        trainable: bool,
            Whether the embedding layer can be trained or not.
        """
        self._model.trainable = trainable

    def get_embedding_dataframe(self) -> pd.DataFrame:
        """Return terms embedding using given index names."""
        return self._model.get_embedding_dataframe(self._graph.get_node_names())

    def save_embedding(self, path: str):
        """Save terms embedding using given index names.

        Parameters
        -----------------------------
        path: str,
            Save embedding as csv to given path.
        """
        self._model.save_embedding(path, self._graph.get_node_names())

    @property
    def name(self) -> str:
        """Return model name."""
        return self._model.name

    def save_weights(self, path: str):
        """Save model weights to given path.

        Parameters
        ---------------------------
        path: str,
            Path where to save model weights.
        """
        self._model.save_weights(path)

    def load_weights(self, path: str):
        """Load model weights from given path.

        Parameters
        ---------------------------
        path: str,
            Path from where to load model weights.
        """
        self._model.load_weights(path)
