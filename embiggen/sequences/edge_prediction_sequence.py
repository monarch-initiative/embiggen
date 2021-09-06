"""Keras Sequence for running Neural Network on graph edge prediction."""
from typing import Tuple, Union

import numpy as np
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence


class EdgePredictionSequence(Sequence):
    """Keras Sequence for running Neural Network on graph edge prediction."""

    def __init__(
        self,
        graph: Graph,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        use_edge_metrics: bool = False,
        batch_size: int = 2**10,
        negative_samples_rate: float = 0.5,
        avoid_false_negatives: bool = False,
        support_mirrored_strategy: bool = False,
        graph_to_avoid: Graph = None,
        batches_per_epoch: Union[int, str] = "auto",
        elapsed_epochs: int = 0,
        random_state: int = 42
    ):
        """Create new EdgePredictionSequence object.

        Parameters
        --------------------------------
        graph: Graph,
            The graph from which to sample the edges.
        use_node_types: bool = False,
            Whether to return the node types.
        use_edge_types: bool = False,
            Whether to return the edge types.
        use_edge_metrics: bool = False,
            Whether to return the edge metrics.
        batch_size: int = 2**10,
            The batch size to use.
        negative_samples_rate: float = 0.5,
            Factor of negatives to use in every batch.
            For example, with a batch size of 128 and negative_samples_rate equal
            to 0.5, there will be 64 positives and 64 negatives.
        avoid_false_negatives: bool = False,
            Whether to filter out false negatives.
            By default False.
            Enabling this will slow down the batch generation while (likely) not
            introducing any significant gain to the model performance.
        support_mirrored_strategy: bool = False,
            Wethever to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        graph_to_avoid: Graph = None,
            Graph to avoid when generating the edges.
            This can be the validation component of the graph, for example.
            More information to how to generate the holdouts is available
            in the Graph package.
        batches_per_epoch: Union[int, str] = "auto",
            Number of batches per epoch.
            If auto, it is used: `10 * edges number /  batch size`
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        random_state: int = 42,
            The random_state to use to make extraction reproducible.
        """
        self._graph = graph
        self._negative_samples_rate = negative_samples_rate
        self._avoid_false_negatives = avoid_false_negatives
        self._support_mirrored_strategy = support_mirrored_strategy
        self._graph_to_avoid = graph_to_avoid
        self._random_state = random_state
        self._use_node_types = use_node_types
        self._use_edge_types = use_edge_types
        self._use_edge_metrics = use_edge_metrics
        if batches_per_epoch == "auto":
            batches_per_epoch = max(
                5 * graph.get_directed_edges_number() // batch_size,
                1
            )
        self._batches_per_epoch = batches_per_epoch
        self._nodes = np.array(self._graph.get_node_names())
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
        sources, source_node_types, destinations, destination_node_types, edge_metrics, edge_types, labels = self._graph.get_edge_prediction_mini_batch(
            (self._random_state + idx) * (1 + self.elapsed_epochs),
            return_node_types=self._use_node_types,
            return_edge_types=self._use_edge_types,
            return_edge_metrics=self._use_edge_metrics,
            batch_size=self.batch_size,
            negative_samples_rate=self._negative_samples_rate,
            avoid_false_negatives=self._avoid_false_negatives,
            graph_to_avoid=self._graph_to_avoid,
        )
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
            if value is not None
        ], labels
