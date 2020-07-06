from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
import numpy as np  # type: ignore
from typing import Tuple
from .node2vec_sequence import Node2VecSequence


class NodeCBOWSequence(Node2VecSequence):

    def __getitem__(self, idx: int) -> Tuple[Tuple[np.ndarray, np.ndarray], None]:
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be rendered.

        Returns
        ---------------
        
        """
        return self._graph.cbow(
            idx,
            self._batch_size,
            self._length,
            iterations=self._iterations,
            window_size=self._window_size,
            shuffle=self._shuffle,
            min_length=self._min_length,
            return_weight=self._return_weight,
            explore_weight=self._explore_weight,
            change_node_type_weight=self._change_node_type_weight,
            change_edge_type_weight=self._change_edge_type_weight,
        ), None