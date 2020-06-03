from typing import Tuple
from ..graph import Graph
from .node_2_edge_transformer import Node2EdgeTransformer
import numpy as np  # type: ignore


class GraphPartitionTransfomer:

    def __init__(self, method: str = "hadamard"):
        """Create a new GraphPartitionTransfomer object.

        It transforms a tuple formed of the positive partition of edges
        and the negative partition of edges into a tuple of vectors and labels
        to be used for training purposes.

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
        A new GraphPartitionTransfomer object.
        """
        self._transformer = Node2EdgeTransformer(method=method)

    def fit(self,  embedding: np.ndarray):
        """Fit the GraphPartitionTransfomer model.

        Parameters
        ----------------------
        embedding: np.ndarray,
            Nodes embedding.
        """
        self._transformer.fit(embedding)

    def _get_labels(self, positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
        """Return training labels for given graph partitions.

        Parameters
        ----------------------
        positive: np.ndarray,
            The positive partition of the graph.
        negative: np.ndarray,
            The negative partition of the graph.

        Returns
        ----------------------
        Labels from the partitions.
        """
        return np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])

    def transform(self, positive: Graph, negative: Graph) -> Tuple[np.ndarray, np.ndarray]:
        """Return X and y data for training.

        Parameters
        ----------------------
        positive: Graph,
            The positive partition of the Graph.
        negative: Graph,
            The negative partition of the Graph.

        Raises
        --------------------
        ValueError,
            If model has not been fitted.

        Returns
        ----------------------
        Tuple of X and y to be used for training.
        """

        positive_embedding = self._transformer.transform(positive)
        negative_embedding = self._transformer.transform(negative)

        return (
            np.concatenate([positive_embedding, negative_embedding]),
            self._get_labels(positive_embedding, negative_embedding)
        )

    def transform_nodes(self, positive: Graph, negative: Graph) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray]:
        """Return X and y data for training.

        Parameters
        ----------------------
        positive: Graph,
            The positive partition of the Graph.
        negative: Graph,
            The negative partition of the Graph.

        Returns
        ----------------------
        Triple of X for the source nodes,
        X for the destination nodes and the labels to be used for training.
        """
        pos_src, pos_dst = self._transformer.transform_nodes(positive)
        neg_src, neg_dst = self._transformer.transform_nodes(negative)

        return (
            np.concatenate([pos_dst, neg_dst]),
            np.concatenate([pos_src, neg_src]),
            self._get_labels(pos_src, neg_src)
        )
