"""EdgeTransformer class to convert edges to edge embeddings."""
from typing import List

import numpy as np
import pandas as pd

from .node_transformer import NodeTransformer


class EdgeTransformer:
    """EdgeTransformer class to convert edges to edge embeddings."""

    methods = {
        "Hadamard": lambda x1, x2: np.multiply(x1, x2, out=x1),
        "Sum": lambda x1, x2: np.add(x1, x2, out=x1),
        "Average": lambda x1, x2: np.divide(np.add(x1, x2, out=x1), 2, out=x1),
        "L1": lambda x1, x2: np.subtract(x1, x2, out=x1),
        "AbsoluteL1": lambda x1, x2: np.abs(np.subtract(x1, x2, out=x1), out=x1),
        "L2": lambda x1, x2: np.power(np.subtract(x1, x2, out=x1), 2, out=x1),
        "Concatenate": lambda x1, x2: np.hstack((x1, x2)),
    }

    def __init__(self, method: str = "Hadamard"):
        """Create new EdgeTransformer object.

        Parameters
        ------------------------
        method: str = "Hadamard",
            Method to use for the embedding.
            Can either be 'Hadamard', 'Sum', 'Average', 'L1', 'AbsoluteL1', 'L2' or 'Concatenate'.
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

    def transform(self, sources: List[str], destinations: List[str], aligned_node_mapping: bool = False) -> np.ndarray:
        """Return embedding for given edges using provided method.

        Parameters
        --------------------------
        sources: List[str],
            List of source nodes whose embedding is to be returned.
        destinations: List[str],
            List of destination nodes whose embedding is to be returned.
        aligned_node_mapping: bool = False,
            This parameter specifies wheter the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated edge embedding
            will be meaningless.

        Raises
        --------------------------
        ValueError,
            If embedding is not fitted.

        Returns
        --------------------------
        Numpy array of embeddings.
        """
        return self._method(
            self._transformer.transform(
                sources,
                aligned_node_mapping
            ),
            self._transformer.transform(
                destinations,
                aligned_node_mapping
            )
        )
