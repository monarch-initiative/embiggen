"""GraphTransformer class to convert graphs to edge embeddings."""
from typing import List, Union
import pandas as pd
import numpy as np
from ensmallen import Graph  # pylint: disable=no-name-in-module

from .edge_transformer import EdgeTransformer


class GraphTransformer:
    """GraphTransformer class to convert graphs to edge embeddings."""

    def __init__(
        self,
        method: str = "Hadamard",
        aligned_node_mapping: bool = False,
        support_mirrored_strategy: bool = False,
    ):
        """Create new GraphTransformer object.

        Parameters
        ------------------------
        method: str = "hadamard",
            Method to use for the embedding.
            Can either be 'Hadamard', 'Sum', 'Average', 'L1', 'AbsoluteL1', 'L2' or 'Concatenate'.
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
        self._transformer = EdgeTransformer(
            method=method,
            aligned_node_mapping=aligned_node_mapping,
            support_mirrored_strategy=support_mirrored_strategy
        )
        self._aligned_node_mapping = aligned_node_mapping

    @property
    def numeric_node_ids(self) -> bool:
        """Return whether the transformer returns numeric node IDs."""
        return self._transformer.numeric_node_ids

    @property
    def method(self) -> str:
        """Return the used edge embedding method."""
        return self._transformer.method

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

    def transform(
        self,
        graph: Union[Graph, np.ndarray, List[List[str]], List[List[int]]],
    ) -> np.ndarray:
        """Return edge embedding for given graph using provided method.

        Parameters
        --------------------------
        graph: Union[Graph, np.ndarray, List[List[str]], List[List[int]]],
            The graph whose edges are to embed.
            It can either be an Graph or a list of lists of edges.

        Raises
        --------------------------
        ValueError,
            If embedding is not fitted.

        Returns
        --------------------------
        Numpy array of embeddings.
        """
        if isinstance(graph, Graph):
            if self._aligned_node_mapping:
                graph = graph.get_edge_node_ids(directed=False)
            else:
                graph = graph.get_edge_node_names(directed=False)
        if isinstance(graph, List):
            graph = np.array(graph)
        if isinstance(graph, np.ndarray):
            sources = graph[:, 0]
            destinations = graph[:, 1]
        return self._transformer.transform(sources, destinations)
