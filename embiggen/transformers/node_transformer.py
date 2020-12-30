"""NodeTransformer class to convert nodes to edge embeddings."""
from typing import List, Union
import numpy as np
import pandas as pd


class NodeTransformer:
    """NodeTransformer class to convert nodes to edge embeddings."""

    def __init__(self):
        """Create new NodeTransformer object."""
        self._embedding = None
        self._embedding_numpy = None

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
        if not isinstance(embedding, pd.DataFrame):
            raise ValueError("Given embedding is not a pandas DataFrame.")
        self._embedding = embedding
        self._embedding_numpy = embedding.to_numpy()

    def transform(self, nodes: Union[List[str], List[int]], aligned_node_mapping: bool = False) -> np.ndarray:
        """Return embeddings from given node.

        Parameters
        --------------------------
        nodes: Union[List[str], List[int]],
            List of nodes whose embedding is to be returned.
            By default this should be a list of strings, if the
            aligned_node_mapping is setted, then this methods also accepts
            a list of ints.
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
        if self._embedding is None:
            raise ValueError(
                "Transformer was not fitted yet."
            )

        if aligned_node_mapping:
            return self._embedding_numpy[nodes]

        return self._embedding.loc[nodes].to_numpy()
