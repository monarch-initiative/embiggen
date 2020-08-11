"""NodeTransformer class to convert nodes to edge embeddings."""
import numpy as np


class NodeTransformer:
    """NodeTransformer class to convert nodes to edge embeddings."""

    def __init__(self):
        """Create new NodeTransformer object."""
        self._embedding = None

    def fit(self, embedding: np.ndarray):
        """Fit the model.

        Parameters
        -------------------------
        embedding: np.ndarray,
            Embedding to use to fit the transformer.
        """
        self._embedding = embedding

    def transform(self, nodes: np.ndarray) -> np.ndarray:
        """Return embeddings from given node.

        Parameters
        --------------------------
        nodes: np.ndarray,
            Vector of nodes whose embedding is to be returned.

        Raises
        --------------------------
        ValueError,
            If embedding is not fitted.

        Returns
        --------------------------
        Numpy array of embeddings.
        """
        if self._embedding is None:
            raise ValueError(
                "Transformer was not fitted yet."
            )
        return self._embedding[nodes]
