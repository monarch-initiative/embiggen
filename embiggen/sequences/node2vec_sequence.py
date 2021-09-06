"""Keras Sequence object for running CBOW and SkipGram on graph walks."""
from typing import Tuple, Dict

import numpy as np  # type: ignore
from ensmallen import Graph  # pylint: disable=no-name-in-module
from .abstract_sequence import AbstractSequence


class Node2VecSequence(AbstractSequence):
    """Keras Sequence object for running models on graph walks."""

    def __init__(
        self,
        graph: Graph,
        walk_length: int = 128,
        batch_size: int = 256,
        iterations: int = 16,
        window_size: int = 4,
        return_weight: float = 1.0,
        explore_weight: float = 1.0,
        change_node_type_weight: float = 1.0,
        change_edge_type_weight: float = 1.0,
        max_neighbours: int = None,
        elapsed_epochs: int = 0,
        support_mirrored_strategy: bool = False,
        random_state: int = 42,
        dense_node_mapping: Dict[int, int] = None,
    ):
        """Create new Node2Vec Sequence object.

        Parameters
        -----------------------------
        graph: Graph,
            The graph from from where to extract the walks.
        walk_length: int = 128,
            Maximal length of the walks.
        batch_size: int = 256,
            Number of nodes to include in a single batch.
        iterations: int = 16,
            Number of iterations of the single walks.
        window_size: int = 4,
            Window size for the local context.
            On the borders the window size is trimmed.
        return_weight: float = 1.0,
            Weight on the probability of returning to the same node the walk just came from
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
            THIS IS AN EXPERIMENTAL FEATURE!
        change_edge_type_weight: float = 1.0,
            Weight on the probability of visiting a neighbor edge of a
            different type than the previous edge. This only applies to
            multigraphs, otherwise it has no impact.
            THIS IS AN EXPERIMENTAL FEATURE!
        max_neighbours: int = None,
            Number of maximum neighbours to consider when using approximated walks.
            By default, None, we execute exact random walks.
            This is mainly useful for graphs containing nodes with extremely high degrees.
            THIS IS AN EXPERIMENTAL FEATURE!
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        support_mirrored_strategy: bool = False,
            Wethever to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        random_state: int = 42,
            The random state to reproduce the training sequence.
        dense_node_mapping: Dict[int, int] = None,
            Mapping to use for converting sparse walk space into a dense space.
            This object can be created using the method (available from the
            graph object created using Graph)
            called `get_dense_node_mapping` that returns a mapping from
            the non trap nodes (those from where a walk could start) and
            maps these nodes into a dense range of values.
            THIS IS AN EXPERIMENTAL FEATURE!
        """
        self._graph = graph
        self._walk_length = walk_length
        self._iterations = iterations
        self._return_weight = return_weight
        self._explore_weight = explore_weight
        self._max_neighbours = max_neighbours
        self._change_node_type_weight = change_node_type_weight
        self._change_edge_type_weight = change_edge_type_weight
        self._dense_node_mapping = dense_node_mapping

        super().__init__(
            batch_size=batch_size,
            sample_number=self._graph.get_unique_source_nodes_number(),
            window_size=window_size,
            elapsed_epochs=elapsed_epochs,
            support_mirrored_strategy=support_mirrored_strategy,
            random_state=random_state
        )

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
            Index corresponding to batch to be returned.

        Returns
        ---------------
        Tuple of tuples with input data.
        """
        contexts_batch, words_batch = self._graph.node2vec(
            batch_size=self._batch_size,
            walk_length=self._walk_length,
            window_size=self._window_size,
            iterations=self._iterations,
            return_weight=self._return_weight,
            explore_weight=self._explore_weight,
            change_node_type_weight=self._change_node_type_weight,
            change_edge_type_weight=self._change_edge_type_weight,
            dense_node_mapping=self._dense_node_mapping,
            max_neighbours=self._max_neighbours,
            random_state=self._random_state + idx + self.elapsed_epochs
        )

        outputs = [contexts_batch, words_batch]

        if self._support_mirrored_strategy:
            outputs = [
                output.astype(float)
                for output in outputs
            ]

        return outputs, None
