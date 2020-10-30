"""LinkPredictionTransformer class to convert graphs to edge embeddings to execute link prediction."""
from typing import Tuple
import pandas as pd
import numpy as np
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module

from .graph_transformer import GraphTransformer


class LinkPredictionTransformer:
    """LinkPredictionTransformer class to convert graphs to edge embeddings."""

    def __init__(self, method: str = "hadamard"):
        """Create new LinkPredictionTransformer object.

        Parameters
        ------------------------
        method: str = "hadamard",
            Method to use for the embedding.
            Can either be 'hadamard', 'average', 'weightedL1', 'weightedL2' or
            a custom lambda that receives two numpy arrays with the nodes
            embedding and returns the edge embedding.
        """
        self._transformer = GraphTransformer(method=method)

    def fit(self, embedding: pd.DataFrame):
        """Fit the model.

        Parameters
        -------------------------
        embedding: pd.DataFrame,
            Embedding to use to fit the transformer.
            This is a pandas DataFrame and NOT a numpy array because we need
            to be able to remap correctly the vector embeddings in case of
            graphs that do not respect the same internal node mapping but have
            the same node set. It is possible to remap such graphs using
            Ensmallen's remap method but it may be less intuitive to users.
        """
        self._transformer.fit(embedding)

    def transform(
        self,
        positive_graph: EnsmallenGraph,
        negative_graph: EnsmallenGraph,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return edge embedding for given graph using provided method.

        Parameters
        --------------------------
        graph: EnsmallenGraph,
            The graph whose edges are to embed.

        Raises
        --------------------------
        ValueError,
            If embedding is not fitted.

        Returns
        --------------------------
        Tuple with X and y values.
        """
        edge_embeddings = np.vstack([
            self._transformer.transform(positive_graph),
            self._transformer.transform(negative_graph),
        ])
        edge_labels = np.concatenate([
            np.ones(positive_graph.get_edges_number()),
            np.zeros(negative_graph.get_edges_number())
        ])
        numpy_random_state = np.random.RandomState(  # pylint: disable=no-member
            seed=random_state
        )
        indices = numpy_random_state.permutation(edge_labels.size)

        return edge_embeddings[indices], edge_labels[indices]
