"""NodeTransformer class to convert nodes to edge embeddings."""
from typing import List, Union
import numpy as np
import pandas as pd


class NodeTransformer:
    """NodeTransformer class to convert nodes to edge embeddings."""

    def __init__(
        self,
        numeric_node_ids: bool = False,
        aligned_node_mapping: bool = False,
        support_mirrored_strategy: bool = False,
    ):
        """Create new NodeTransformer object.

        Parameters
        -------------------
        numeric_node_ids: bool = False,
            Wether to return the numeric node IDs instead of the node embedding.
        aligned_node_mapping: bool = False,
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated edge embedding
            will be meaningless.
        support_mirrored_strategy: bool = False,
            Wethever to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        """
        self._numeric_node_ids = numeric_node_ids
        self._support_mirrored_strategy = support_mirrored_strategy
        self._embedding = None
        self._embedding_numpy = None
        self._aligned_node_mapping = aligned_node_mapping

    @property
    def numeric_node_ids(self) -> bool:
        """Return whether the transformer returns numeric node IDs."""
        return self._numeric_node_ids

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
            if not self._aligned_node_mapping:
                raise ValueError("Given embedding is not a pandas DataFrame.")
            self._embedding_numpy = embedding
        else:
            self._embedding = embedding
            self._embedding_numpy = embedding.to_numpy()

    def transform(self, nodes: Union[List[str], List[int]]) -> np.ndarray:
        """Return embeddings from given node.

        Parameters
        --------------------------
        nodes: Union[List[str], List[int]],
            List of nodes whose embedding is to be returned.
            By default this should be a list of strings, if the
            aligned_node_mapping is setted, then this methods also accepts
            a list of ints.

        Raises
        --------------------------
        ValueError,
            If embedding is not fitted.

        Returns
        --------------------------
        Numpy array of embeddings.
        """
        if self._embedding_numpy is None and not self.numeric_node_ids:
            raise ValueError(
                "Transformer was not fitted yet."
            )

        if self._aligned_node_mapping:
            if self.numeric_node_ids:
                if self._support_mirrored_strategy:
                    return nodes.astype(float)
                return nodes
            return self._embedding_numpy[nodes]

        if self.numeric_node_ids:
            ids = self._embedding.index.get_indexer(nodes)
            if self._support_mirrored_strategy:
                return ids.astype(float)
            return ids
        return self._embedding.loc[nodes].to_numpy()
