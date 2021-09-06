"""GloVe model for graph node embedding."""
from typing import Dict, Union

import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import \
    Optimizer   # pylint: disable=import-error,no-name-in-module

from ensmallen import Graph

from .glove import GloVe
from ..utils import validate_window_size


class GraphGloVe(GloVe):
    """GloVe model for graph and words embedding.

    The GloVe model for graoh embedding receives two words and is asked to
    predict its cooccurrence probability.
    """

    def __init__(
        self,
        graph: Graph,
        embedding_size: int = 100,
        embedding: Union[np.ndarray, pd.DataFrame] = None,
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
        optimizer: Union[str, Optimizer] = None,
        alpha: float = 0.75,
        directed: bool = False,
        walk_length: int = 128,
        iterations: int = 16,
        window_size: int = 4,
        return_weight: float = 1.0,
        explore_weight: float = 1.0,
        change_node_type_weight: float = 1.0,
        change_edge_type_weight: float = 1.0,
        max_neighbours: int = None,
        support_mirrored_strategy: bool = False,
        random_state: int = 42,
        dense_node_mapping: Dict[int, int] = None,
        use_gradient_centralization: bool = True,
    ):
        """Create new GloVe-based Embedder object.

        Parameters
        ----------------------------
        vocabulary_size: int,
            Number of terms to embed.
            In a graph this is the number of nodes, while in a text is the
            number of the unique words.
        embedding_size: int,
            Dimension of the embedding.
        embedding: Union[np.ndarray, pd.DataFrame] = None,
            The seed embedding to be used.
            Note that it is not possible to provide at once both
            the embedding and either the vocabulary size or the embedding size.
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
            Optional extra features to be used during the computation
            of the embedding. The features must be available for all the
            elements considered for the embedding.
        optimizer: Union[str, Optimizer] = "nadam",
            The optimizer to be used during the training of the model.
            By default, if None is provided, Nadam with learning rate
            set at 0.01 is used.
        alpha: float = 0.75,
            Alpha to use for the function.
        directed: bool = False,
            Whether to treat the data as directed or not.
        walk_length: int = 128,
            Maximal length of the walks.
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
        max_neighbours: int = None,
            Number of maximum neighbours to consider when using approximated walks.
            By default, None, we execute exact random walks.
            This is mainly useful for graphs containing nodes with extremely high degrees.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        support_mirrored_strategy: bool = False,
            Wethever to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        random_state: int = 42,
            The random state to reproduce the training sequence.
        dense_node_mapping: Dict[int, int] = None,
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
        """
        self._graph = graph
        self._walk_length = walk_length
        self._iterations = iterations
        self._window_size = validate_window_size(window_size)
        self._return_weight = return_weight
        self._explore_weight = explore_weight
        self._change_node_type_weight = change_node_type_weight
        self._change_edge_type_weight = change_edge_type_weight
        self._max_neighbours = max_neighbours
        self._support_mirrored_strategy = support_mirrored_strategy
        self._random_state = random_state
        self._dense_node_mapping = dense_node_mapping
        super().__init__(
            alpha=alpha,
            random_state=random_state,
            vocabulary_size=self._graph.get_nodes_number(),
            embedding_size=embedding_size,
            embedding=embedding,
            extra_features=extra_features,
            optimizer=optimizer,
            use_gradient_centralization=use_gradient_centralization
        )

    def get_embedding_dataframe(self) -> pd.DataFrame:
        """Return terms embedding using given index names."""
        return super().get_embedding_dataframe(self._graph.get_node_names())

    def fit(
        self,
        epochs: int = 1000,
        batch_size: int = 2**20,
        early_stopping_monitor: str = "loss",
        early_stopping_min_delta: float = 0.00001,
        early_stopping_patience: int = 100,
        early_stopping_mode: str = "min",
        reduce_lr_monitor: str = "loss",
        reduce_lr_min_delta: float = 0.0001,
        reduce_lr_patience: int = 50,
        reduce_lr_mode: str = "min",
        reduce_lr_factor: float = 0.95,
        verbose: int = 2,
        **kwargs: Dict
    ) -> pd.DataFrame:
        """Return pandas dataframe with training history.

        Parameters
        -----------------------
        epochs: int = 1000,
            Epochs to train the model for.
        batch_size: int = 2**20,
            The batch size.
            Tipically batch sizes for the GloVe model can be immense.
        early_stopping_monitor: str = "loss",
            Metric to monitor for early stopping.
        early_stopping_min_delta: float = 0.001,
            Minimum delta of metric to stop the training.
        early_stopping_patience: int = 10,
            Number of epochs to wait for when the given minimum delta is not
            achieved after which trigger early stopping.
        early_stopping_mode: str = "min",
            Direction of the variation of the monitored metric for early stopping.
        reduce_lr_monitor: str = "loss",
            Metric to monitor for reducing learning rate.
        reduce_lr_min_delta: float = 0.01,
            Minimum delta of metric to reduce learning rate.
        reduce_lr_patience: int = 10,
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

        Raises
        -----------------------
        ValueError,
            If given verbose value is not within the available set (-1, 0, 1).

        Returns
        -----------------------
        Dataframe with training history.
        """
        sources, destinations, frequencies = self._graph.cooccurence_matrix(
            walk_length=self._walk_length,
            window_size=self._window_size,
            iterations=self._iterations,
            return_weight=self._return_weight,
            explore_weight=self._explore_weight,
            change_edge_type_weight=self._change_edge_type_weight,
            change_node_type_weight=self._change_node_type_weight,
            dense_node_mapping=self._dense_node_mapping,
            max_neighbours=self._max_neighbours,
            random_state=self._random_state,
            verbose=verbose > 0
        )
        if self._support_mirrored_strategy:
            sources = sources.astype(float)
            destinations = destinations.astype(float)
        return super().fit(
            (sources, destinations), frequencies,
            epochs=epochs,
            batch_size=batch_size,
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
