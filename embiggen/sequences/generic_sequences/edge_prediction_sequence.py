"""Keras Sequence for running Neural Network on graph edge prediction."""
from typing import Tuple

import numpy as np
from ensmallen import Graph  # pylint: disable=no-name-in-module


class EdgePredictionSequence:
    """Keras Sequence for running Neural Network on graph edge prediction."""

    def __init__(
        self,
        graph: Graph,
        graph_used_in_training: Graph,
        return_node_types: bool,
        return_edge_types: bool,
        use_edge_metrics: bool,
        batch_size: int = 2**10,
    ):
        """Create new EdgePredictionSequence object.

        Parameters
        --------------------------------
        graph: Graph
            The graph whose edges are to be predicted.
        graph_used_in_training: Graph
            The graph that was used while training the current
            edge prediction model.
        return_node_types: bool
            Whether to return the node types.
        return_edge_types: bool
            Whether to return the edge types.
        use_edge_metrics: bool = False
            Whether to return the edge metrics.
        batch_size: int = 2**10,
            The batch size to use.
        """
        if not graph.has_compatible_node_vocabularies(graph_used_in_training):
            raise ValueError(
                f"The provided graph {graph.get_name()} does not have a node vocabulary "
                "that is compatible with the provided graph used in training."
            )
        self._graph = graph
        self._graph_used_in_training = graph_used_in_training
        self._return_node_types = return_node_types
        self._return_edge_types = return_edge_types
        self._use_edge_metrics = use_edge_metrics
        self._batch_size = batch_size

    def __len__(self) -> int:
        """Returns length of sequence."""
        return int(np.ceil(self._graph.get_number_of_directed_edges() / self._batch_size))

    def __getitem__(self, idx: int):
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be returned.

        Returns
        ---------------
        Return Tuple containing X and Y numpy arrays corresponding to given batch index.
        """
        return (tuple([
            value
            for value in self._graph_used_in_training.get_edge_prediction_chunk_mini_batch(
                idx,
                graph=self._graph,
                batch_size=self._batch_size,
                return_node_types=self._return_node_types,
                return_edge_types=self._return_edge_types,
                return_edge_metrics=self._use_edge_metrics,
            )
            if value is not None
        ]),)
