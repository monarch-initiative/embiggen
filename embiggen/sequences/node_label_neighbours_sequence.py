"""Keras Sequence for running Neural Network on graph node-label embedding."""
from typing import Tuple

import numpy as np
from ensmallen_graph import EnsmallenGraph
from keras_mixed_sequence import VectorSequence


class NodeLabelNeighboursSequence(VectorSequence):
    """Keras Sequence for running Neural Network on graph node-label prediction."""

    def __init__(
        self,
        graph: EnsmallenGraph,
        node_ids: np.ndarray,
        max_neighbours: int = None,
        batch_size: int = 2**8,
        elapsed_epochs: int = 0,
        random_state: int = 42,
        support_mirror_strategy: bool = False
    ):
        """Create new LinkPredictionSequence object.

        Parameters
        --------------------------------
        graph: EnsmallenGraph,
            The graph from which to sample the edges.
        node_ids: np.ndarray = None,
            IDs of the nodes to consider.
        batch_size: int = 2**8,
            The batch size to use.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        random_state: int = 42,
            The random state to use to make extraction reproducible.
        support_mirror_strategy: bool = False,
            Wethever to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        """
        self._graph = graph
        if max_neighbours is None:
            max_neighbours = self._graph.degrees_median()
        self._max_neighbours = max_neighbours
        self._support_mirror_strategy = support_mirror_strategy
        super().__init__(
            vector=node_ids,
            batch_size=batch_size,
            random_state=random_state,
            elapsed_epochs=elapsed_epochs,
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
        nodes = super().__getitem__(idx)
        neighbours = np.zeros((nodes.size, self._max_neighbours))
        for i, node in enumerate(nodes):
            node_neighbours = self._graph.get_filtered_neighbours(node)
            if node_neighbours.size > self._max_neighbours:
                node_neighbours = np.random.choice(
                    node_neighbours,
                    size=self._max_neighbours,
                    replace=False
                )
            # The plus one is needed to handle nodes with less than max neighbours
            # such nodes are represented with zeros and in the embedding layer
            # are masked.
            neighbours[i, node_neighbours.size] = node_neighbours + 1
        if self._support_mirror_strategy:
            return neighbours.astype(float)
        return neighbours
