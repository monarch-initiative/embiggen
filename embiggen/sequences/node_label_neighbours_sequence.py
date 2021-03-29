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
        max_neighbours: int = None,
        include_central_node: bool = True,
        batch_size: int = 2**8,
        elapsed_epochs: int = 0,
        shuffle: bool = True,
        random_state: int = 42,
        support_mirror_strategy: bool = False
    ):
        """Create new NoLaN sequence.

        Parameters
        --------------------------------
        graph: EnsmallenGraph,
            The graph from which to sample the edges.
        max_neighbours: int = None,
            Number of neighbours to consider.
            If None, the graph mean is used.
        include_central_node: bool = False,
            Whether to include the central node.
            In our experiments, this lead to overfitting.
        batch_size: int = 2**8,
            The batch size to use.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        shuffle: bool = True,
            Whether to shuffle data.
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
        self._include_central_node = include_central_node
        self._support_mirror_strategy = support_mirror_strategy
        super().__init__(
            batch_size=batch_size,
            vector=np.array([
                node_id
                for node_id, node_type in enumerate(self._graph.get_node_types())
                if node_type is not None
            ]),
            random_state=random_state,
            shuffle=shuffle,
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
        neighbours, labels = self._graph.get_node_label_prediction_tuple_by_node_ids(
            node_ids=super().__getitem__(idx),
            random_state=self._random_state + idx,
            include_central_node=self._include_central_node,
            offset=1,
            max_neighbours=self._max_neighbours
        )

        if self._support_mirror_strategy:
            return neighbours.astype(float), labels
        return neighbours, labels
