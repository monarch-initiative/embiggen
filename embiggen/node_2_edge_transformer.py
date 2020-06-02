from typing import Dict, List, Tuple
from .graph import Graph
import numpy as np  # type: ignore


class N2ETransformer:

    methods = {
        "hadamard": lambda x1, x2: np.multiply(x1, x2),
        "average": lambda x1, x2: np.mean([x1, x2], axis=0),
        "weightedL1": lambda x1, x2: np.abs(x1 - x2),
        "weightedL2": lambda x1, x2: (x2 - x2)**2
    }

    def __init__(self, embedding: np.ndarray, method: str = "hadamard"):
        """Create a new N2ETransformer object.

        Parameters
        ----------------------
        embedding: np.ndarray,
            Nodes embedding.
        method: str = "hadamard",
            Method to use to transform the nodes embedding to edges.

        Returns
        ----------------------
        A new N2ETransformer object.
        """

        if method not in N2ETransformer.methods:
            raise ValueError((
                "Given method {method} is not supported. "
                "The only supported methods are {methods}."
            ).format(
                method=method,
                methods=", ".join(N2ETransformer.methods.keys())
            ))
        self._method = np.vectorize(N2ETransformer.methods[method])
        self._embedding = embedding

    def transform(self, G: Graph) -> np.ndarray:
        """Return embedded edges from given graph nodes.

        Parameters
        ---------------------
        G: Graph,
            The graph whose nodes are to be embedded.

        Returns
        ---------------------
        The embedded edges.
        """
        return self._method(*self.transform_nodes(G))

    def transform_nodes(self, G: Graph) -> Tuple[np.ndarray, np.ndarray]:
        """Return nodes from given graph.

        Parameters
        ---------------------
        G: Graph,
            The graph whose nodes are to be embedded.

        Returns
        ---------------------
        The embedded edges.
        """
        return self._embedding[G.sources], self._embedding[G.destinations]