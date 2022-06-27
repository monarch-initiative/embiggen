"""Keras Sequence object for running CBOW and SkipGram on graph walks."""
from typing import Tuple, Dict, Optional

import numpy as np  # type: ignore
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence
import tensorflow as tf
from embiggen.utils.tensorflow_utils import tensorflow_version_is_higher_or_equal_than


class Node2VecSequence(Sequence):
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
        max_neighbours: Optional[int] = 100,
        random_state: int = 42,
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
        max_neighbours: Optional[int] = 100,
            Number of maximum neighbours to consider when using approximated walks.
            By default, None, we execute exact random walks.
            This is mainly useful for graphs containing nodes with extremely high degrees.
            THIS IS AN EXPERIMENTAL FEATURE!
        random_state: int = 42,
            The random state to reproduce the training sequence.
        """
        self._graph = graph
        self._walk_length = walk_length
        self._iterations = iterations
        self._return_weight = return_weight
        self._explore_weight = explore_weight
        self._max_neighbours = max_neighbours
        self._change_node_type_weight = change_node_type_weight
        self._change_edge_type_weight = change_edge_type_weight
        self._window_size = window_size
        self._random_state = random_state
        self._current_index = 0

        super().__init__(
            sample_number=self._graph.get_number_of_unique_source_nodes(),
            batch_size=batch_size,
        )

    def __call__(self):
        """Return next batch using an infinite generator model."""
        self._current_index += 1
        return self[self._current_index]

    def into_dataset(self) -> tf.data.Dataset:
        """Return dataset generated out of the current sequence instance.

        Implementative details
        ---------------------------------
        This method handles the conversion of this Keras Sequence into
        a TensorFlow dataset, also handling the proper dispatching according
        to what version of TensorFlow is installed in this system.

        Returns
        ----------------------------------
        Dataset to be used for the training of a model
        """

        #################################################################
        # Handling kernel creation when TensorFlow is a modern version. #
        #################################################################

        number_of_skipgrams = self._batch_size * self._iterations * \
            (self._walk_length - self._window_size * 2)
        
        if tensorflow_version_is_higher_or_equal_than("2.5.0"):
            input_tensor_specs = []

            # Shapes of the source and destination node IDs
            input_tensor_specs.append(tf.TensorSpec(
                shape=(number_of_skipgrams, self._window_size*2),
                dtype=tf.int32
            ))
            input_tensor_specs.append(tf.TensorSpec(
                shape=(number_of_skipgrams, ),
                dtype=tf.int32
            ))

            return tf.data.Dataset.from_generator(
                self,
                output_signature=(
                    (
                        *input_tensor_specs,
                    ),
                )
            )

        return tf.data.Dataset.from_generator(
            self,
            output_types=(
                (
                    tf.int32,
                    tf.int32
                ),
            ),
            output_shapes=(
                (
                    tf.TensorShape([number_of_skipgrams, self._window_size*2]),
                    tf.TensorShape([number_of_skipgrams, ])
                ),
            )
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
            max_neighbours=self._max_neighbours,
            random_state=self._random_state + idx + self.elapsed_epochs
        )

        return (((contexts_batch, words_batch), ), )
