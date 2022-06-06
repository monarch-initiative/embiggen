"""NodeTransformer class to convert nodes to edge embeddings."""
from typing import List, Union
import numpy as np
import pandas as pd
from ensmallen import Graph

class NodeTransformer:
    """NodeTransformer class to convert nodes to edge embeddings."""

    def __init__(
        self,
        numeric_node_ids: bool = False,
        aligned_node_mapping: bool = False,
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
        """
        self._numeric_node_ids = numeric_node_ids
        self._node_feature = None
        self._aligned_node_mapping = aligned_node_mapping

    @property
    def numeric_node_ids(self) -> bool:
        """Return whether the transformer returns numeric node IDs."""
        return self._numeric_node_ids

    def fit(self, node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]):
        """Fit the model.

        Parameters
        -------------------------
        node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
            Node feature to use to fit the transformer.
        """
        if not isinstance(node_feature, list):
            node_feature = [node_feature]

        if len(node_feature) == 0:
            raise ValueError(
                "The provided list of features is empty!"
            )

        # We check if any of the provided node features
        # is neither a numpy array nor a pandas dataframe.
        for nf in node_feature:
            if not isinstance(nf, (pd.DataFrame, np.ndarray)):
                raise ValueError(
                    (
                        "One of the provided node features is not "
                        "neither a pandas DataFrame nor a numpy array, but "
                        "of type {node_feature_type}. It is not clear "
                        "what to do with this feature."
                    ).format(
                        node_feature_type=type(node_feature)
                    )
                )

        # We check if, while the parameters for alignment
        # has not been provided, numpy arrays were provided.
        # This would be an issue as we cannot check for alignment
        # in numpy arrays.
        if not self._aligned_node_mapping and any(
            isinstance(nf, np.ndarray)
            for nf in node_feature
        ):
            raise ValueError(
                "A numpy array feature was provided while the "
                "aligned node mapping parameter was set to false. "
                "If you intend to specify that you are providing a numpy "
                "array node feature that is aligned with the node vocabulary "
                "of the graph set the `aligned_node_mapping` parameter "
                "to True."
            )

        if self._aligned_node_mapping:
            self._node_feature = np.hstack([
                nf.to_numpy() if isinstance(nf, pd.DataFrame) else nf
                for nf in node_feature
            ])
            if not self._node_feature.data.c_contiguous:
                self._node_feature = np.ascontiguousarray(self._node_feature)
        else:
            self._node_feature = pd.concat(node_feature, axis=1)

    def transform(self, nodes: Union[Graph, List[str], List[int]]) -> np.ndarray:
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
        if self._node_feature is None and not self.numeric_node_ids:
            raise ValueError(
                "Transformer was not fitted yet."
            )

        if self._aligned_node_mapping:
            if self.numeric_node_ids:
                return nodes
            if isinstance(nodes, Graph):
                return self._node_feature
            return self._node_feature[nodes]

        if self.numeric_node_ids:
            return self._node_feature.index.get_indexer(nodes)
        
        if isinstance(nodes, Graph):
            nodes = nodes.get_node_names()
        
        return self._node_feature.loc[nodes].to_numpy()
