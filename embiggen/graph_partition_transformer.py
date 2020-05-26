from typing import Dict, List, Tuple
from .csf_graph import CSFGraph
from .node_2_edge_transformer import N2ETransformer
import numpy as np


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

    def transform(self, positive: CSFGraph, negative: CSFGraph) -> Tuple[np.ndarray]:
        """Return X and y data for training.

        Parameters
        ----------------------
        positive: CSFGraph,
            The positive partition of the Graph.
        negative: CSFGraph,
            The negative partition of the Graph.

        Returns
        ----------------------
        Tuple of X and y to be used for training.
        """
        positive_embedding = self._transformer.transform(positive)
        negative_embedding = self._transformer.transform(negative)

        return (
            np.concatenate([positive_embedding, negative_embedding]),
            np.concatenate([
                np.ones(len(positive_embedding)),
                np.zeros(len(negative_embedding))
            ])
        )
