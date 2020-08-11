"""Abstract Keras Sequence object for running models on graph walks."""
from typing import Dict

from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from .abstract_sequence import AbstractSequence


class AbstractNode2VecSequence(AbstractSequence):
    """Abstract Keras Sequence object for running models on graph walks."""

    def __init__(
        self,
        graph: EnsmallenGraph,
        walk_length: int,
        batch_size: int,
        iterations: int = 1,
        window_size: int = 4,
        shuffle: bool = True,
        min_length: int = 1,
        return_weight: float = 1.0,
        explore_weight: float = 1.0,
        change_node_type_weight: float = 1.0,
        change_edge_type_weight: float = 1.0,
        elapsed_epochs: int = 0,
        dense_nodes_mapping: Dict[int, int] = None
    ):
        """Create new Node2Vec Sequence object.

        Parameters
        -----------------------------
        graph: EnsmallenGraph,
            The graph from from where to extract the walks.
        walk_length: int,
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
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        dense_nodes_mapping: Dict[int, int] = None,
            Mapping to use for converting sparse walk space into a dense space.
            This object can be created using the method available from graph
            called `get_dense_nodes_mapping` that returns a mapping from
            the non trap nodes (those from where a walk could start) and
            maps these nodes into a dense range of values.
        """
        self._graph = graph
        self._walk_length = walk_length
        self._iterations = iterations
        self._min_length = min_length
        self._return_weight = return_weight
        self._explore_weight = explore_weight
        self._change_node_type_weight = change_node_type_weight
        self._change_edge_type_weight = change_edge_type_weight
        self._dense_nodes_mapping = dense_nodes_mapping

        super().__init__(
            batch_size=batch_size,
            samples_number=self._graph.get_not_trap_nodes_number(),
            window_size=window_size,
            elapsed_epochs=elapsed_epochs
        )
