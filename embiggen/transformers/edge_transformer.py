"""EdgeTransformer class to convert edges to edge embeddings."""
from typing import List, Union, Optional

import numpy as np
import pandas as pd

from .node_transformer import NodeTransformer


def get_hadamard_edge_embedding(
    source_node_embedding: np.ndarray,
    destination_node_embedding: np.ndarray
) -> np.ndarray:
    """Return Hadamard edge embedding of the two nodes.

    Parameters
    --------------------------
    source_node_embedding: np.ndarray
        Numpy array with the embedding of the source node.
    destination_node_embedding: np.ndarray
        Numpy array with the embedding of the destination node.

    Returns
    --------------------------
    Numpy array with the Hadamard edge embedding.
    """
    return np.multiply(
        source_node_embedding,
        destination_node_embedding
    )


def get_sum_edge_embedding(
    source_node_embedding: np.ndarray,
    destination_node_embedding: np.ndarray
) -> np.ndarray:
    """Return sum edge embedding of the two nodes.

    Parameters
    --------------------------
    source_node_embedding: np.ndarray
        Numpy array with the embedding of the source node.
    destination_node_embedding: np.ndarray
        Numpy array with the embedding of the destination node.

    Returns
    --------------------------
    Numpy array with the sum edge embedding.
    """
    return np.add(
        source_node_embedding,
        destination_node_embedding
    )


def get_average_edge_embedding(
    source_node_embedding: np.ndarray,
    destination_node_embedding: np.ndarray
) -> np.ndarray:
    """Return average edge embedding of the two nodes.

    Parameters
    --------------------------
    source_node_embedding: np.ndarray
        Numpy array with the embedding of the source node.
    destination_node_embedding: np.ndarray
        Numpy array with the embedding of the destination node.

    Returns
    --------------------------
    Numpy array with the average edge embedding.
    """
    return np.divide(
        get_sum_edge_embedding(
            source_node_embedding,
            destination_node_embedding
        ),
        2.0
    )


def get_l1_edge_embedding(
    source_node_embedding: np.ndarray,
    destination_node_embedding: np.ndarray
) -> np.ndarray:
    """Return L1 edge embedding of the two nodes.

    Parameters
    --------------------------
    source_node_embedding: np.ndarray
        Numpy array with the embedding of the source node.
    destination_node_embedding: np.ndarray
        Numpy array with the embedding of the destination node.

    Returns
    --------------------------
    Numpy array with the L1 edge embedding.
    """
    return np.subtract(
        source_node_embedding,
        destination_node_embedding
    )


def get_absolute_l1_edge_embedding(
    source_node_embedding: np.ndarray,
    destination_node_embedding: np.ndarray
) -> np.ndarray:
    """Return Absolute L1 edge embedding of the two nodes.

    Parameters
    --------------------------
    source_node_embedding: np.ndarray
        Numpy array with the embedding of the source node.
    destination_node_embedding: np.ndarray
        Numpy array with the embedding of the destination node.

    Returns
    --------------------------
    Numpy array with the Absolute L1 edge embedding.
    """
    return np.abs(
        get_l1_edge_embedding(
            source_node_embedding,
            destination_node_embedding
        )
    )


def get_squared_l2_edge_embedding(
    source_node_embedding: np.ndarray,
    destination_node_embedding: np.ndarray
) -> np.ndarray:
    """Return Squared L2 edge embedding of the two nodes.

    Parameters
    --------------------------
    source_node_embedding: np.ndarray
        Numpy array with the embedding of the source node.
    destination_node_embedding: np.ndarray
        Numpy array with the embedding of the destination node.

    Returns
    --------------------------
    Numpy array with the Squared L2 edge embedding.
    """
    return np.power(
        get_l1_edge_embedding(
            source_node_embedding,
            destination_node_embedding
        ),
        2.0
    )


def get_l2_edge_embedding(
    source_node_embedding: np.ndarray,
    destination_node_embedding: np.ndarray
) -> np.ndarray:
    """Return L2 edge embedding of the two nodes.

    Parameters
    --------------------------
    source_node_embedding: np.ndarray
        Numpy array with the embedding of the source node.
    destination_node_embedding: np.ndarray
        Numpy array with the embedding of the destination node.

    Returns
    --------------------------
    Numpy array with the L2 edge embedding.
    """
    return np.sqrt(
        get_squared_l2_edge_embedding(
            source_node_embedding,
            destination_node_embedding
        )
    )


def get_concatenate_edge_embedding(
    source_node_embedding: np.ndarray,
    destination_node_embedding: np.ndarray
) -> np.ndarray:
    """Return concatenate edge embedding of the two nodes.

    Parameters
    --------------------------
    source_node_embedding: np.ndarray
        Numpy array with the embedding of the source node.
    destination_node_embedding: np.ndarray
        Numpy array with the embedding of the destination node.

    Returns
    --------------------------
    Numpy array with the concatenate edge embedding.
    """
    return np.hstack((
        source_node_embedding,
        destination_node_embedding
    ))


def get_min_edge_embedding(
    source_node_embedding: np.ndarray,
    destination_node_embedding: np.ndarray
) -> np.ndarray:
    """Return min edge embedding of the two nodes.

    Parameters
    --------------------------
    source_node_embedding: np.ndarray
        Numpy array with the embedding of the source node.
    destination_node_embedding: np.ndarray
        Numpy array with the embedding of the destination node.

    Returns
    --------------------------
    Numpy array with the min edge embedding.
    """
    return np.min(
        [
            source_node_embedding,
            destination_node_embedding
        ],
        axis=0
    )


def get_max_edge_embedding(
    source_node_embedding: np.ndarray,
    destination_node_embedding: np.ndarray
) -> np.ndarray:
    """Return max edge embedding of the two nodes.

    Parameters
    --------------------------
    source_node_embedding: np.ndarray
        Numpy array with the embedding of the source node.
    destination_node_embedding: np.ndarray
        Numpy array with the embedding of the destination node.

    Returns
    --------------------------
    Numpy array with the max edge embedding.
    """
    return np.max(
        [
            source_node_embedding,
            destination_node_embedding
        ],
        axis=0
    )

def get_indices_edge_embedding(
    source_node_embedding: np.ndarray,
    destination_node_embedding: np.ndarray
) -> np.ndarray:
    """Placeholder method."""
    return np.vstack((
        source_node_embedding,
        destination_node_embedding
    )).T


class EdgeTransformer:
    """EdgeTransformer class to convert edges to edge embeddings."""

    methods = {
        "Hadamard": get_hadamard_edge_embedding,
        "Sum": get_sum_edge_embedding,
        "Average": get_average_edge_embedding,
        "L1": get_l1_edge_embedding,
        "AbsoluteL1": get_absolute_l1_edge_embedding,
        "SquaredL2": get_squared_l2_edge_embedding,
        "L2": get_l2_edge_embedding,
        "Concatenate": get_concatenate_edge_embedding,
        "Min": get_min_edge_embedding,
        "Max": get_max_edge_embedding,
        None: get_indices_edge_embedding,
    }

    def __init__(
        self,
        method: str = "Hadamard",
        aligned_node_mapping: bool = False,
    ):
        """Create new EdgeTransformer object.

        Parameters
        ------------------------
        method: str = "Hadamard",
            Method to use for the embedding.
            If None is used, we return instead the numeric tuples.
            Can either be 'Hadamard', 'Min', 'Max', 'Sum', 'Average',
            'L1', 'AbsoluteL1', 'SquaredL2', 'L2' or 'Concatenate'.
        aligned_node_mapping: bool = False,
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated edge embedding
            will be meaningless.
        """
        if isinstance(method, str) and method not in EdgeTransformer.methods:
            raise ValueError((
                "Given method '{}' is not supported. "
                "Supported methods are {}, or alternatively a lambda."
            ).format(
                method, ", ".join(list(EdgeTransformer.methods.keys()))
            ))
        self._transformer = NodeTransformer(
            numeric_node_ids=method is None,
            aligned_node_mapping=aligned_node_mapping,
        )
        self._method_name = method
        self._method = EdgeTransformer.methods[self._method_name]

    @property
    def numeric_node_ids(self) -> bool:
        """Return whether the transformer returns numeric node IDs."""
        return self._transformer.numeric_node_ids

    @property
    def method(self) -> str:
        """Return the used edge embedding method."""
        return self._method_name

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

        Raises
        -------------------------
        ValueError,
            If the given method is None there is no need to call the fit method.
        """
        if self._method is None:
            raise ValueError(
                "There is no need to call the fit when edge method is None."
            )
        self._transformer.fit(embedding)

    def transform(
        self,
        sources:Union[List[str], List[int]],
        destinations:Union[List[str], List[int]],
        edge_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return embedding for given edges using provided method.

        Parameters
        --------------------------
        sources:Union[List[str], List[int]]
            List of source nodes whose embedding is to be returned.
        destinations:Union[List[str], List[int]]
            List of destination nodes whose embedding is to be returned.
        edge_features: Optional[np.ndarray] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the provided graph.

        Raises
        --------------------------
        ValueError
            If embedding is not fitted.
        ValueError
            If the edge features are provided and do not have the correct shape.

        Returns
        --------------------------
        Numpy array of embeddings.
        """
        if edge_features is not None and len(edge_features.shape) != 2:
            raise ValueError(
                (
                    "The provided edge features should have a bidimensional shape, "
                    "but the provided one has shape {}."
                ).format(edge_features.shape)
            )

        edge_embeddings = self._method(
            self._transformer.transform(sources),
            self._transformer.transform(destinations)
        )

        if edge_features is not None:
            if edge_features.shape[0] != edge_embeddings.shape[0]:
                raise ValueError(
                    (
                        "The provided edge features should have a sample for each of the edges "
                        "in the graph, which are {}, but were {}."
                    ).format(
                        edge_embeddings.shape[0],
                        edge_features.shape[0]
                    )
                )
            edge_embeddings = np.hstack([
                edge_embeddings,
                edge_features
            ])

        return edge_embeddings