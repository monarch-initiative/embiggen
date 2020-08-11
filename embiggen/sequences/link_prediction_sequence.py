"""Keras Sequence for running Neural Netwok on graph link prediction."""
from typing import Callable, Tuple, Union

import numpy as np
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence

from ..transformers import EdgeTransformer


class LinkPredictionSequence(Sequence):
    """Keras Sequence for running Neural Netwok on graph link prediction."""

    def __init__(
        self,
        graph: EnsmallenGraph,
        embedding: np.ndarray,
        method: Union[str, Callable] = "hadamard",
        batch_size: int = 2**10,
        negative_samples: float = 1.0,
        graph_to_avoid: EnsmallenGraph = None,
        batches_per_epoch: bool = 2**8,
        avoid_self_loops: bool = False,
        elapsed_epochs: int = 0
    ):
        """Create new LinkPredictionSequence object.

        Parameters
        --------------------------------
        graph: EnsmallenGraph,
            The graph from which to sample the edges.
        embedding: np.ndarray,
            The embedding of the nodes.
        method: str = "hadamard",
            Method to use for the embedding.
            Can either be 'hadamard', 'average', 'weightedL1', 'weightedL2' or
            a custom lambda that receives two numpy arrays with the nodes
            embedding and returns the edge embedding.
        batch_size: int = 2**10,
            The batch size to use.
        negative_samples: float = 1.0,
            Factor of negatives to use in every batch.
            For example, with a batch size of 128 and negative_samples equal
            to 1.0, there will be 64 positives and 64 negatives.
        graph_to_avoid: EnsmallenGraph = None,
            Graph to avoid when generating the links.
            This can be the validation component of the graph, for example.
            More information to how to generate the holdouts is available
            in the EnsmallenGraph package.
        batches_per_epoch: bool = 2**8,
            Number of batches per epoch.
        avoid_self_loops: bool = False,
            If the self loops must be filtered away from the result.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        """
        self._graph = graph
        self._negative_samples = negative_samples
        self._graph_to_avoid = graph_to_avoid
        self._avoid_self_loops = avoid_self_loops
        self._transformer = EdgeTransformer(method)
        self._transformer.fit(embedding)
        super().__init__(
            samples_number=batches_per_epoch,
            batch_size=batch_size,
            elapsed_epochs=elapsed_epochs
        )

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be rendered.

        Returns
        ---------------
        Return Tuple containing X and Y numpy arrays corresponding to given batch index.
        """
        edges, labels = self._graph.link_prediction(
            idx + self.elapsed_epochs,
            batch_size=self.batch_size,
            negative_samples=self._negative_samples,
            graph_to_avoid=self._graph_to_avoid,
            avoid_self_loops=self._avoid_self_loops
        )
        return (
            self._transformer.transform(
                edges[:, 0],
                edges[:, 1]
            ),
            labels
        )
