from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
import numpy as np  # type: ignore
from typing import Tuple
from .node2vec_sequence import Node2VecSequence


class NodeSkipGramSequence(Node2VecSequence):

    def __init__(self, *args, negative_samples: float = 7.0, graph_to_avoid: EnsmallenGraph = None, **kwargs):
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
        negative_samples: float = 7,
            Factor of negative samples to use.
        graph_to_avoid: EnsmallenGraph = None,
            The graph portion to be avoided. Can be usefull when using
            holdouts where a portion of the graph is completely hidden,
            and is not to be used neither for negatives nor positives.
        iterations: int = 1,
            Number of iterations of the single walks.
        window_size: int = 4,
            Window size for the local context.
            On the borders the window size is trimmed.
        shuffle: bool = True,
            Wthever to shuffle the vectors.
        min_length: int = 8,
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
        super().__init__(*args, **kwargs)
        self._negative_samples = negative_samples
        self._graph_to_avoid = graph_to_avoid

    def __getitem__(self, idx: int) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be rendered.

        Returns
        ---------------
        Tuple of tuples with vector of integer with words, contexts and labels.
        """
        return self._graph.skipgrams(
            idx,
            self._batch_size,
            self._length,
            iterations=self._iterations,
            window_size=self._window_size,
            negative_samples=self._negative_samples,
            shuffle=self._shuffle,
            min_length=self._min_length,
            return_weight=self._return_weight,
            explore_weight=self._explore_weight,
            change_node_type_weight=self._change_node_type_weight,
            change_edge_type_weight=self._change_edge_type_weight,
            graph_to_avoid=self._graph_to_avoid
        )
