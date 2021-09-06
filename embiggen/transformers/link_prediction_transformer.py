"""LinkPredictionTransformer class to convert graphs to edge embeddings to execute link prediction."""
from typing import Tuple, Union, List
import pandas as pd
import numpy as np
from ensmallen import Graph  # pylint: disable=no-name-in-module

from .graph_transformer import GraphTransformer


class LinkPredictionTransformer:
    """LinkPredictionTransformer class to convert graphs to edge embeddings."""

    def __init__(
        self,
        method: str = "Hadamard",
        aligned_node_mapping: bool = False,
        support_mirrored_strategy: bool = False,
    ):
        """Create new LinkPredictionTransformer object.

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
        self._transformer = GraphTransformer(
            method=method,
            aligned_node_mapping=aligned_node_mapping,
            support_mirrored_strategy=support_mirrored_strategy
        )

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
        positive_graph: Union[Graph, np.ndarray, List[List[str]], List[List[int]]],
        negative_graph: Union[Graph, np.ndarray, List[List[str]], List[List[int]]],
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return edge embedding for given graph using provided method.

        Parameters
        --------------------------
        positive_graph: Union[Graph, List[List[str]], List[List[int]]],
            The graph whose edges are to be embedded and labeled as positives.
            It can either be an Graph or a list of lists of edges.
        negative_graph: Union[Graph, List[List[str]], List[List[int]]],
            The graph whose edges are to be embedded and labeled as positives.
            It can either be an Graph or a list of lists of edges.
        random_state: int = 42,
            The random state to use to shuffle the labels.

        Raises
        --------------------------
        ValueError,
            If embedding is not fitted.

        Returns
        --------------------------
        Tuple with X and y values.
        """
        positive_edge_embedding = self._transformer.transform(positive_graph)
        negative_edge_embedding = self._transformer.transform(negative_graph)
        edge_embeddings = np.vstack([
            positive_edge_embedding,
            negative_edge_embedding
        ])
        edge_labels = np.concatenate([
            np.ones(positive_edge_embedding.shape[0]),
            np.zeros(negative_edge_embedding.shape[0])
        ])
        numpy_random_state = np.random.RandomState(  # pylint: disable=no-member
            seed=random_state
        )
        indices = numpy_random_state.permutation(edge_labels.size)

        return edge_embeddings[indices], edge_labels[indices]
