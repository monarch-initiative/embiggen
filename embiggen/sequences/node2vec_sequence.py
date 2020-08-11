"""Keras Sequence object for running CBOW and SkipGram on graph walks."""
from typing import Tuple

import numpy as np  # type: ignore

from .abstract_node2vec_sequence import AbstractNode2VecSequence


class Node2VecSequence(AbstractNode2VecSequence):
    """Keras Sequence object for running CBOW and SkipGram on graph walks."""

    def __getitem__(self, idx: int) -> Tuple[Tuple[np.ndarray, np.ndarray], None]:
        """Return batch corresponding to given index.

        The return tuple of tuples is composed of an inner tuple, containing
        the words vector and the vector of vectors of the contexts.
        Depending on the order of the input_layers of the models that can
        accept these data format, one of the vectors is used as training
        input and the other one is used as the output for the NCE loss layer.

        The words vectors and contexts vectors contain numeric IDs, that
        represent the index of the words' embedding column.

        The true output value is None, since no loss function is used after
        the NCE loss, that is implemented as a layer, and this vastly improves
        the speed of the training process since it does not require to allocate
        empty vectors of considerable size for the one-hot encoding process.

        A batch returns words and contexts from:

            (number of nodes provided in a batch) *
            (number of iterations of random walks per node) *
            (walk length - window_size*2)

        different contexts.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be rendered.

        Returns
        ---------------
        Tuple of tuples with input data.
        """
        return self._graph.node2vec(
            idx,
            self._batch_size,
            self._walk_length,
            iterations=self._iterations,
            window_size=self._window_size,
            shuffle=self._shuffle,
            min_length=self._min_length,
            return_weight=self._return_weight,
            explore_weight=self._explore_weight,
            change_node_type_weight=self._change_node_type_weight,
            change_edge_type_weight=self._change_edge_type_weight,
            dense_nodes_mapping=self._dense_nodes_mapping
        ), None
