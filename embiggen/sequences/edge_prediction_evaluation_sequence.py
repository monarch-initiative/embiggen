"""Keras Sequence for running Neural Network on graph edge prediction."""
from typing import Tuple, Union, Optional
import warnings

import numpy as np
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence


class EdgePredictionEvaluationSequence(Sequence):
    """Keras Sequence for running Neural Network on graph edge prediction."""

    def __init__(
        self,
        positive_graph: Graph,
        negative_graph: Optional[Graph] = None,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        use_edge_metrics: bool = False,
        batch_size: int = 2**10,
        support_mirrored_strategy: bool = False,
        filter_none_values: bool = True,
        batches_per_epoch: Union[int, str] = "auto",
        elapsed_epochs: int = 0,
        random_state: int = 42
    ):
        """Create new EdgePredictionSequence object.

        Parameters
        --------------------------------
        positive_graph: Graph
            The graph from which to sample the positive edges.
        negative_graph: Optional[Graph] = None
            The graph from which to sample the negative edges.
        use_node_types: bool = False
            Whether to return the node types.
        use_edge_types: bool = False
            Whether to return the edge types.
        use_edge_metrics: bool = False
            Whether to return the edge metrics.
        batch_size: int = 2**10
            The batch size to use.
        support_mirrored_strategy: bool = False
            Whether to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        filter_none_values: bool = True
            Whether to filter None values.
        batches_per_epoch: Union[int, str] = "auto"
            Number of batches per epoch.
            If auto, it is used: `10 * edges number /  batch size`
        elapsed_epochs: int = 0
            Number of elapsed epochs to init state of generator.
        random_state: int = 42
            The random_state to use to make extraction reproducible.

        Raises
        --------------------------
        ValueError
            If the two graphs do not share the same vocabulary and therefore
            are not compatible with each other (or the model, likely).
        ValueError
            If the positive and negative graphs overlap.
        ValueError
            If either of the provided graphs is empty.
        """
        if negative_graph is not None:
            if not positive_graph.is_compatible(negative_graph):
                raise ValueError(
                    "The two provided positive and negative graphs are "
                    "not compatible with each other, for instance they may not "
                    "share the same dictionary of nodes."
                )
            if positive_graph.overlaps(negative_graph):
                raise ValueError(
                    "The provided positive and negative graphs overlap "
                    "with each other, making the definition of true negative "
                    "and true positive within this evaluation ambigous."
                )
            if not negative_graph.has_edges():
                raise ValueError(
                    "The provided negative graph is empty, that is, it has no edges."
                )
            if positive_graph.has_selfloops() ^ negative_graph.has_selfloops():
                warnings.warn(
                    "Please be advides that in either the provided positive "
                    "or negative graphs there are present selfloops. "
                    "If such odd topologies were also exclusively "
                    "present during the training of this model in only either the positive "
                    "or negative edges, the model might learn that self-loops are always "
                    "positive and/or negative. This might be okay or not according "
                    "to the experimental setup you have designed."
                )

        if not positive_graph.has_edges():
            raise ValueError(
                "The provided positive graph is empty, that is, it has no edges."
            )

        self._positive_graph = positive_graph
        self._negative_graph = negative_graph
        self._support_mirrored_strategy = support_mirrored_strategy
        self._random_state = random_state
        self._use_node_types = use_node_types
        self._use_edge_types = use_edge_types
        self._filter_none_values = filter_none_values
        self._use_edge_metrics = use_edge_metrics
        if batches_per_epoch == "auto":
            edges_number = positive_graph.get_edges_number()
            if negative_graph is not None:
                edges_number += negative_graph.get_edges_number()
            batches_per_epoch = max(
                edges_number // batch_size,
                1
            )
        self._batches_per_epoch = batches_per_epoch
        super().__init__(
            sample_number=batches_per_epoch*batch_size,
            batch_size=batch_size,
            elapsed_epochs=elapsed_epochs
        )

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be returned.

        Returns
        ---------------
        Return Tuple containing X and Y numpy arrays corresponding to given batch index.
        """
        if idx * self.batch_size < self._positive_graph.get_edges_number():
            sources, source_node_types, destinations, destination_node_types, edge_metrics, edge_types = self._positive_graph.get_edge_prediction_chunk_mini_batch(
                idx,
                return_node_types=self._use_node_types,
                return_edge_types=False,
                return_edge_metrics=self._use_edge_metrics,
                batch_size=self.batch_size,
            )
            labels = np.ones_like(sources, dtype=bool)
        elif self._negative_graph is not None:
            sources, source_node_types, destinations, destination_node_types, edge_metrics, edge_types = self._negative_graph.get_edge_prediction_chunk_mini_batch(
                idx,
                return_node_types=self._use_node_types,
                return_edge_types=False,
                return_edge_metrics=self._use_edge_metrics,
                batch_size=self.batch_size,
            )
            labels = np.zeros_like(sources, dtype=bool)

        if self._support_mirrored_strategy:
            sources = sources.astype(float)
            destinations = destinations.astype(float)
            if self._use_node_types:
                source_node_types = source_node_types.astype(float)
                destination_node_types = destination_node_types.astype(float)
            if self._use_edge_types:
                edge_types = edge_types.astype(float)
            if self._use_edge_metrics:
                edge_metrics = edge_metrics.astype(float)
        return [
            value
            for value in (
                sources, source_node_types, destinations, destination_node_types, edge_metrics, edge_types,
            )
            if not self._filter_none_values or value is not None
        ], labels
