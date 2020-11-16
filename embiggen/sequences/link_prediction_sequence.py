"""Keras Sequence for running Neural Network on graph link prediction."""
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence

from ..transformers import EdgeTransformer


class LinkPredictionSequence(Sequence):
    """Keras Sequence for running Neural Network on graph link prediction."""

    def __init__(
        self,
        graph: EnsmallenGraph,
        embedding: pd.DataFrame,
        method: Union[str, Callable] = "hadamard",
        batch_size: int = 2**10,
        negative_samples: float = 1.0,
        graph_to_avoid: EnsmallenGraph = None,
        batches_per_epoch: bool = 2**8,
        elapsed_epochs: int = 0,
        support_mirror_strategy: bool = False,
        seed: int = 42
    ):
        """Create new LinkPredictionSequence object.

        Parameters
        --------------------------------
        graph: EnsmallenGraph,
            The graph from which to sample the edges.
        embedding: pd.DataFrame,
            The embedding of the nodes.
            This is a pandas DataFrame and NOT a numpy array because we need
            to be able to remap correctly the vector embeddings in case of
            graphs that do not respect the same internal node mapping but have
            the same node set. It is possible to remap such graphs using
            Ensmallen's remap method but it may be less intuitive to users.
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
            If the self loops must be filtered away from the result.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        support_mirror_strategy: bool = False,
            Wethever to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        seed: int = 42,
            The seed to use to make extraction reproducible.
        """
        self._graph = graph
        self._negative_samples = negative_samples
        self._graph_to_avoid = graph_to_avoid
        self._transformer = EdgeTransformer(method)
        self._transformer.fit(embedding)
        self._support_mirror_strategy = support_mirror_strategy
        self._seed = seed
        self._nodes = np.array(self._graph.get_node_names())
        super().__init__(
            sample_number=batches_per_epoch,
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
        edges, labels = self._graph.link_prediction(
            self._seed + idx + self.elapsed_epochs,
            batch_size=self.batch_size,
            negative_samples=self._negative_samples,
            graph_to_avoid=self._graph_to_avoid
        )
        edge_embeddings = self._transformer.transform(
            self._nodes[edges[:, 0]],
            self._nodes[edges[:, 1]]
        )
        if self._support_mirror_strategy:
            return edge_embeddings.astype(float), labels.astype(float)
        return edge_embeddings, labels
