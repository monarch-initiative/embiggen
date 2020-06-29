from typing import Tuple
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
import numpy as np  # type: ignore


class Node2EdgeTransformer:

    methods = {
        "hadamard": lambda x1, x2: np.multiply(),
        "average": lambda x1, x2: np.mean([x1, x2], axis=0),
        "weightedL1": lambda x1, x2: np.abs(x1 - x2),
        "weightedL2": lambda x1, x2: (x2 - x2)**2
    }

    def __init__(self, method: str = "hadamard"):
        """Create a new Node2EdgeTransformer object.

        Parameters
        ----------------------
        method: str = "hadamard",
            Method to use to transform the nodes embedding to edges.

        Raises
        ----------------------
        ValueError,
            If the given embedding method is not supported.

        Returns
        ----------------------
        A new N2ETransformer object.
        """

        if method not in Node2EdgeTransformer.methods:
            raise ValueError((
                "Given method {method} is not supported. "
                "The only supported methods are {methods}."
            ).format(
                method=method,
                methods=", ".join(Node2EdgeTransformer.methods.keys())
            ))
        self._method = Node2EdgeTransformer.methods[method]
        self._embedding = None

    def fit(self,  embedding: np.ndarray):
        """Fit the Node2EdgeTransformer model.

        Parameters
        ----------------------
        embedding: np.ndarray,
            Nodes embedding.
        """
        self._embedding = embedding

    def transform_edges(self, G: EnsmallenGraph) -> np.ndarray:
        """Return embedded edges from given graph nodes.

        Parameters
        ---------------------
        G: EnsmallenGraph,
            The graph whose nodes are to be embedded.

        Raises
        --------------------
        ValueError,
            If model has not been fitted.

        Returns
        ---------------------
        The embedded edges.
        """
        return self._method(*self.transform_nodes(G))

    def transform_nodes(self, G: EnsmallenGraph) -> Tuple[np.ndarray, np.ndarray]:
        """Return nodes from given graph.

        Parameters
        ---------------------
        G: EnsmallenGraph,
            The graph whose nodes are to be embedded.

        Raises
        --------------------
        ValueError,
            If model has not been fitted.

        Returns
        ---------------------
        The embedded edges.
        """
        if self._embedding is None:
            raise ValueError("Model has not been fitted.")
        return self._embedding[G.sources], self._embedding[G.destinations]
