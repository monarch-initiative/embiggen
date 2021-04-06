"""Keras Sequence for running Neural Network on graph edge prediction."""
from typing import Tuple

import numpy as np
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence


class EdgePredictionSequence(Sequence):
    """Keras Sequence for running Neural Network on graph edge prediction."""

    def __init__(
        self,
        graph: EnsmallenGraph,
        batch_size: int = 2**10,
        negative_samples: float = 1.0,
        avoid_false_negatives: bool = False,
        support_mirror_strategy: bool = False,
        graph_to_avoid: EnsmallenGraph = None,
        batches_per_epoch: bool = 2**8,
        elapsed_epochs: int = 0,
        random_state: int = 42
    ):
        """Create new EdgePredictionSequence object.

        Parameters
        --------------------------------
        graph: EnsmallenGraph,
            The graph from which to sample the edges.
        batch_size: int = 2**10,
            The batch size to use.
        negative_samples: float = 1.0,
            Factor of negatives to use in every batch.
            For example, with a batch size of 128 and negative_samples equal
            to 1.0, there will be 64 positives and 64 negatives.
        avoid_false_negatives: bool = False,
            Whether to filter out false negatives.
            By default False.
            Enabling this will slow down the batch generation while (likely) not
            introducing any significant gain to the model performance.
        support_mirror_strategy: bool = False,
            Wethever to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        graph_to_avoid: EnsmallenGraph = None,
            Graph to avoid when generating the edges.
            This can be the validation component of the graph, for example.
            More information to how to generate the holdouts is available
            in the EnsmallenGraph package.
        batches_per_epoch: bool = 2**8,
            Number of batches per epoch.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        random_state: int = 42,
            The random_state to use to make extraction reproducible.
        """
        self._graph = graph
        self._negative_samples = negative_samples
        self._avoid_false_negatives = avoid_false_negatives
        self._support_mirror_strategy = support_mirror_strategy
        self._graph_to_avoid = graph_to_avoid
        self._random_state = random_state
        self._nodes = np.array(self._graph.get_node_names())
        super().__init__(
            sample_number=batches_per_epoch*batch_size,
            batch_size=batch_size,
            elapsed_epochs=elapsed_epochs
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
        left, right, labels = self._graph.link_prediction_ids(
            self._random_state + idx + self.elapsed_epochs,
            batch_size=self.batch_size,
            negative_samples=self._negative_samples,
            avoid_false_negatives=self._avoid_false_negatives,
            graph_to_avoid=self._graph_to_avoid,
        )
        if self._support_mirror_strategy:
            left = left.astype(float)
            right = right.astype(float)
        return (left, right), labels
