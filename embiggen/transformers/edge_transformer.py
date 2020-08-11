"""EdgeTransformer class to convert edges to edge embeddings."""
import numpy as np
from .node_transformer import NodeTransformer


class EdgeTransformer:
    """EdgeTransformer class to convert edges to edge embeddings."""

    methods = {
        "hadamard": np.multiply,
        "average": lambda x1, x2: np.mean([x1, x2], axis=0),
        "weightedL1": lambda x1, x2: np.abs(x1 - x2),
        "weightedL2": lambda x1, x2: (x1 - x2)**2
    }

    def __init__(self, method: str = "hadamard"):
        """Create new EdgeTransformer object.

        Parameters
        ------------------------
        method: str = "hadamard",
            Method to use for the embedding.
            Can either be 'hadamard', 'average', 'weightedL1', 'weightedL2' or
            a custom lambda that receives two numpy arrays with the nodes
            embedding and returns the edge embedding.
        """
        if isinstance(method, str) and method not in EdgeTransformer.methods:
            raise ValueError((
                "Given method '{}' is not supported. "
                "Supported methods are {}, or alternatively a lambda."
            ).format(
                method, ", ".join(list(EdgeTransformer.methods.keys()))
            ))
        self._method = EdgeTransformer.methods[method]
        self._transformer = NodeTransformer()

    def fit(self, embedding: np.ndarray):
        """Fit the model.

        Parameters
        -------------------------
        embedding: np.ndarray,
            Embedding to use to fit the transformer.
        """
        self._transformer.fit(embedding)

    def transform(self, sources: np.ndarray, destinations: np.ndarray) -> np.ndarray:
        """Return embedding for given edges using provided method.

        Parameters
        --------------------------
        sources: np.ndarray,
            Vector of source nodes whose embedding is to be returned.
        destinations:np.ndarray,
            Vector of destination nodes whose embedding is to be returned.

        Raises
        --------------------------
        ValueError,
            If embedding is not fitted.

        Returns
        --------------------------
        Numpy array of embeddings.
        """
        return self._method(
            self._transformer.transform(sources),
            self._transformer.transform(destinations),
        )
