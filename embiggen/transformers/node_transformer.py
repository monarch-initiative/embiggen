"""NodeTransformer class to convert nodes to edge embeddings."""
from typing import List
import numpy as np
import pandas as pd


class NodeTransformer:
    """NodeTransformer class to convert nodes to edge embeddings."""

    def __init__(self):
        """Create new NodeTransformer object."""
        self._embedding = None

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
        self._embedding = embedding

    def transform(self, nodes: List[str]) -> np.ndarray:
        """Return embeddings from given node.

        Parameters
        --------------------------
        nodes: List[str],
            List of nodes whose embedding is to be returned.

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
        return self._embedding.loc[nodes].values
