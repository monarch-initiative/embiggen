"""Keras Sequence for running Neural Network on graph edge prediction."""
from typing import Tuple

import numpy as np
import tensorflow as tf
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence
from embiggen.utils.tensorflow_utils import tensorflow_version_is_higher_or_equal_than


class EdgePredictionTrainingSequence(Sequence):
    """Keras Sequence for running Neural Network on graph edge prediction."""

    def __init__(
        self,
        graph: Graph,
        use_node_types: bool = False,
        use_edge_metrics: bool = False,
        batch_size: int = 2**10,
        negative_samples_rate: float = 0.5,
        avoid_false_negatives: bool = False,
        graph_to_avoid: Graph = None,
        sample_only_edges_with_heterogeneous_node_types: bool = False,
        random_state: int = 42
    ):
        """Create new EdgePredictionSequence object.

        Parameters
        --------------------------------
        graph: Graph,
            The graph from which to sample the edges.
        use_node_types: bool = False,
            Whether to return the node types.
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
        graph_to_avoid: Graph = None,
            Graph to avoid when generating the edges.
            This can be the validation component of the graph, for example.
            More information to how to generate the holdouts is available
            in the Graph package.
        sample_only_edges_with_heterogeneous_node_types: bool = False
            Whether to only sample edges between heterogeneous node types.
            This may be useful when training a model to predict between
            two portions in a bipartite graph.
        random_state: int = 42,
            The random_state to use to make extraction reproducible.
        """
        self._graph = graph
        self._negative_samples_rate = negative_samples_rate
        self._avoid_false_negatives = avoid_false_negatives
        self._graph_to_avoid = graph_to_avoid
        self._random_state = random_state
        self._use_node_types = use_node_types
        self._use_edge_metrics = use_edge_metrics
        self._sample_only_edges_with_heterogeneous_node_types = sample_only_edges_with_heterogeneous_node_types
        self._current_index = 0
        super().__init__(
            sample_number=graph.get_number_of_directed_edges(),
            batch_size=batch_size,
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
        sources, source_node_types, destinations, destination_node_types, edge_metrics, labels = self._graph.get_edge_prediction_mini_batch(
            (self._random_state + idx) * (1 + self.elapsed_epochs),
            return_node_types=self._use_node_types,
            return_edge_metrics=self._use_edge_metrics,
            batch_size=self.batch_size,
            negative_samples_rate=self._negative_samples_rate,
            avoid_false_negatives=self._avoid_false_negatives,
            sample_only_edges_with_heterogeneous_node_types=self._sample_only_edges_with_heterogeneous_node_types,
            graph_to_avoid=self._graph_to_avoid,
        )

        return (tuple([
            value
            for value in (
                sources, source_node_types,
                destinations, destination_node_types,
                edge_metrics,
            )
            if value is not None
        ]), labels)
