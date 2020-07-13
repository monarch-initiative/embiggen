from typing import Tuple

import numpy as np  # type: ignore
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from tensorflow.keras.utils import Sequence


class AbstractNode2VecSequence(Sequence):

    def __init__(
        self,
        graph: EnsmallenGraph,
        length: int,
        batch_size: int,
        iterations: int = 1,
        window_size: int = 4,
        shuffle: bool = True,
        min_length: int = 1,
        return_weight: float = 1.0,
        explore_weight: float = 1.0,
        change_node_type_weight: float = 1.0,
        change_edge_type_weight: float = 1.0,
    ):
        """Create new Node2Vec Sequence object.

        Parameters
        -----------------------------
        graph: EnsmallenGraph,
            The graph from from where to extract the walks.
        length: int,
            Maximal length of the walks.
            In directed graphs, when traps are present, walks may be shorter.
        batch_size: int,
            Number of nodes to include in a single batch.
        iterations: int = 1,
            Number of iterations of the single walks.
        window_size: int = 4,
            Window size for the local context.
            On the borders the window size is trimmed.
        shuffle: bool = True,
            Wthever to shuffle the vectors.
        min_length: int = 1,
            Minimum length of the walks.
            In directed graphs, when traps are present, walks shorter than
            this amount are removed. This should be two times the window_size.
        return_weight: float = 1.0,
            Weight on the probability of returning to node coming from
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
        """
        self._graph = graph
        self._length = length
        self._batch_size = batch_size
        self._iterations = iterations
        self._window_size = window_size
        self._shuffle = shuffle
        self._min_length = min_length
        self._return_weight = return_weight
        self._explore_weight = explore_weight
        self._change_node_type_weight = change_node_type_weight
        self._change_edge_type_weight = change_edge_type_weight

    def on_epoch_end(self):
        """Shuffle private bed object on every epoch end."""

    def __len__(self) -> int:
        """Return length of bed generator."""
        return int(np.ceil(self._graph.get_nodes_number() / float(self._batch_size)))

    @property
    def steps_per_epoch(self) -> int:
        """Return length of bed generator."""
        return len(self)

    def __getitem__(self, idx: int) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Return batch corresponding to given index.

        This method must be implemented in the child classes.
        """
        raise NotImplementedError(
            "The method __getitem__ must be implemented in the child classes."
        )
