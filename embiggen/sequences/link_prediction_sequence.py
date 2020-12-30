"""Keras Sequence for running Neural Network on graph link prediction."""
from typing import Callable, Tuple, Union

import numpy as np
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence


class LinkPredictionSequence(Sequence):
    """Keras Sequence for running Neural Network on graph link prediction."""

    def __init__(
        self,
        graph: EnsmallenGraph,
        embedding: np.ndarray,
        method: Union[str, Callable] = "Hadamard",
        batch_size: int = 2**10,
        negative_samples: float = 1.0,
        avoid_false_negatives: bool = False,
        graph_to_avoid: EnsmallenGraph = None,
        batches_per_epoch: bool = 2**8,
        elapsed_epochs: int = 0,
        seed: int = 42
    ):
        """Create new LinkPredictionSequence object.

        Parameters
        --------------------------------
        graph: EnsmallenGraph,
            The graph from which to sample the edges.
        embedding: np.ndarray,
            This is a numpy array and NOT a pandas DataFrame because we need
            to quickly load the embedding for the nodes and a DataFrame is too slow.
        method: str = "Hadamard",
            Method to use for the embedding.
            Can either be 'Hadamard', 'Average', 'L1', 'AbsoluteL1' or 'L2'.
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
        graph_to_avoid: EnsmallenGraph = None,
            Graph to avoid when generating the links.
            This can be the validation component of the graph, for example.
            More information to how to generate the holdouts is available
            in the EnsmallenGraph package.
        batches_per_epoch: bool = 2**8,
            Number of batches per epoch.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        seed: int = 42,
            The seed to use to make extraction reproducible.
        """
        self._graph = graph
        self._graph.set_embedding(embedding)
        self._negative_samples = negative_samples
        self._avoid_false_negatives = avoid_false_negatives
        self._graph_to_avoid = graph_to_avoid
        self._method = method
        self._seed = seed
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
        return self._graph.link_prediction(
            self._seed + idx + self.elapsed_epochs,
            batch_size=self.batch_size,
            method=self._method,
            negative_samples=self._negative_samples,
            avoid_false_negatives=self._avoid_false_negatives,
            graph_to_avoid=self._graph_to_avoid,
        )
