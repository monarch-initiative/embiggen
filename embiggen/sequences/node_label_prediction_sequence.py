"""Keras Sequence for running Neural Network on graph node-label embedding."""
from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from ensmallen import Graph
from keras_mixed_sequence import Sequence


class NodeLabelPredictionSequence(Sequence):
    """Keras Sequence for running Neural Network on graph node-label prediction."""

    def __init__(
        self,
        graph: Graph,
        max_neighbours: int = None,
        include_central_node: bool = False,
        return_edge_weights: bool = False,
        batch_size: int = 2**8,
        elapsed_epochs: int = 0,
        random_state: int = 42,
        support_mirrored_strategy: bool = False
    ):
        """Create new Node Label Prediction Sequence.
        """
        self._graph = graph
        self._random_state = random_state
        self._include_central_node = include_central_node
        self._return_edge_weights = return_edge_weights
        self._max_neighbours = max_neighbours
        self._support_mirrored_strategy = support_mirrored_strategy
        super().__init__(
            sample_number=graph.get_directed_edges_number(),
            batch_size=batch_size,
            elapsed_epochs=elapsed_epochs
        )

    def __getitem__(self, idx: int) -> Tuple[Union[tf.RaggedTensor, Tuple[tf.RaggedTensor]], np.ndarray]:
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be returned.

        Returns
        ---------------
        Return Tuple containing X and Y numpy arrays corresponding to given batch index.
        """
        neighbours, weights, labels = self._graph.get_node_label_prediction_mini_batch(
            idx=(self._random_state + idx) * (1 + self.elapsed_epochs),
            batch_size=self._batch_size,
            include_central_node=self._include_central_node,
            return_edge_weights=self._return_edge_weights,
            max_neighbours=self._max_neighbours
        )

        if self._support_mirrored_strategy:
            neighbours = neighbours.astype(float)
        if self._return_edge_weights:
            return (tf.ragged.constant(neighbours), tf.ragged.constant(weights)), labels
        return tf.ragged.constant(neighbours), labels
