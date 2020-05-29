from typing import Dict, List, Tuple
from .graph import Graph
from .node_2_edge_transformer import N2ETransformer
import numpy as np  # type: ignore


class GraphPartitionTransfomer:

    def __init__(self, embedding: Dict[str, List[float]], method: str = "hadamard"):
        """Create a new GraphPartitionTransfomer object.

        It transforms a tuple formed of the positive partition of edges
        and the negative partition of edges into a tuple of vectors and labels
        to be used for training purposes.

        Parameters
        ----------------------
        embedding: Dict[str, List[float]],
            Dictionary containing the nodes embedding.
        method: str = "hadamard",
            Method to use to transform the nodes embedding to edges.

        Returns
        ----------------------
        A new GraphPartitionTransfomer object.
        """
        self._transformer = N2ETransformer(embedding, method=method)

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

    def transform_edges(self, positive: Graph, negative: Graph) -> Tuple[np.ndarray, np.ndarray]:
        """Return X and y data for training.

        Parameters
        ----------------------
        positive: Graph,
            The positive partition of the Graph.
        negative: Graph,
            The negative partition of the Graph.

        Returns
        ----------------------
        Tuple of X and y to be used for training.
        """
        positive_embedding = self._transformer.transform_edges(positive)
        negative_embedding = self._transformer.transform_edges(negative)

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
        Tuple of X and y to be used for training.
        """
        positive_sink, positive_source = self._transformer.transform_nodes(
            positive
        )
        negative_sink, negative_source = self._transformer.transform_nodes(
            negative
        )

        return (
            np.concatenate([positive_source, negative_source]),
            np.concatenate([positive_sink, negative_sink]),
            self._get_labels(positive_sink, negative_sink)
        )
