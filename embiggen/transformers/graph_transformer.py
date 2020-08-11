"""GraphTransformer class to convert graphs to edge embeddings."""
import numpy as np
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module

from .edge_transformer import EdgeTransformer


class GraphTransformer:
    """GraphTransformer class to convert graphs to edge embeddings."""

    def __init__(self, method: str = "hadamard"):
        """Create new GraphTransformer object.

        Parameters
        ------------------------
        method: str = "hadamard",
            Method to use for the embedding.
            Can either be 'hadamard', 'average', 'weightedL1', 'weightedL2' or
            a custom lambda that receives two numpy arrays with the nodes
            embedding and returns the edge embedding.
        """
        self._transformer = EdgeTransformer(method=method)

    def fit(self, embedding: np.ndarray):
        """Fit the model.

        Parameters
        -------------------------
        embedding: np.ndarray,
            Embedding to use to fit the transformer.
        """
        self._transformer.fit(embedding)

    def transform(self, graph: EnsmallenGraph) -> np.ndarray:
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
        Numpy array of embeddings.
        """
        return self._transformer.transform(graph.sources, graph.destinations)
