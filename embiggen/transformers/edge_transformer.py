"""EdgeTransformer class to convert edges to edge embeddings."""
from typing import List, Union, Optional

import numpy as np
import pandas as pd
from userinput.utils import closest
from ensmallen import express_measures
from embiggen.utils.abstract_models import format_list
from embiggen.transformers.node_transformer import NodeTransformer


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


def get_l2_distance(
    source_node_embedding: np.ndarray,
    destination_node_embedding: np.ndarray
) -> np.ndarray:
    """Return L2 distance of the two nodes.

    Parameters
    --------------------------
    source_node_embedding: np.ndarray
        Numpy array with the embedding of the source node.
    destination_node_embedding: np.ndarray
        Numpy array with the embedding of the destination node.

    Returns
    --------------------------
    Numpy array with the L2 distance.
    """
    return np.sqrt(np.sum(np.power(
        source_node_embedding - destination_node_embedding,
        2.0
    ), axis=1)).reshape((-1, 1))


def get_cosine_similarity(
    embedding: np.ndarray,
    source_node_ids: np.ndarray,
    destination_node_ids: np.ndarray
) -> np.ndarray:
    """Return cosine similarity of the two nodes.

    Parameters
    --------------------------
    embedding: np.ndarray
        Numpy array with the embedding matrix.
    source_node_ids: np.ndarray
        Numpy array with the ids of the source node.
    destination_node_ids: np.ndarray
        Numpy array with the ids of the destination node.

    Returns
    --------------------------
    Numpy array with the cosine similarity.
    """
    if not source_node_ids.data.c_contiguous:
        source_node_ids = np.ascontiguousarray(source_node_ids)
    if not destination_node_ids.data.c_contiguous:
        destination_node_ids = np.ascontiguousarray(destination_node_ids)
    return express_measures.cosine_similarity_from_indices_unchecked(
        matrix=embedding,
        sources=source_node_ids,
        destinations=destination_node_ids
    ).reshape((-1, 1))

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
        "L2Distance": get_l2_distance,
        "CosineSimilarity": get_cosine_similarity,
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
        if isinstance(method, str) and method.lower() not in [
            None if method_name is None else method_name.lower()
            for method_name in EdgeTransformer.methods
        ]:
            raise ValueError((
                "Given method '{}' is not supported. "
                "Supported methods are {}, or alternatively a lambda. "
                "Maybe you meant {}?"
            ).format(
                method,
                format_list([method for method in EdgeTransformer.methods.keys() if method is not None]),
                closest(method, [
                    method_name
                    for method_name in EdgeTransformer.methods
                    if method_name is not None
                ])
            ))
        self._transformer = NodeTransformer(
            numeric_node_ids=method is None,
            aligned_node_mapping=aligned_node_mapping,
        )
        self._method_name = method
        if self._method_name is None:
            self._method = EdgeTransformer.methods[None]
        else:
            self._method = {
                None if method_name is None else method_name.lower(): callback
                for method_name, callback in EdgeTransformer.methods.items()
            }[self._method_name.lower()]

    @property
    def numeric_node_ids(self) -> bool:
        """Return whether the transformer returns numeric node IDs."""
        return self._transformer.numeric_node_ids

    @property
    def method(self) -> str:
        """Return the used edge embedding method."""
        return self._method_name

    def fit(self, node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]):
        """Fit the model.

        Parameters
        -------------------------
        node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
            Node feature to use to fit the transformer.

        Raises
        -------------------------
        ValueError
            If the given method is None there is no need to call the fit method.
        """
        if self._method is None:
            raise ValueError(
                "There is no need to call the fit when edge method is None, "
                "as the transformer will exclusively return the numeric node "
                "indices and not any node feature."
            )
        self._transformer.fit(node_feature)

    def transform(
        self,
        sources: Union[List[str], List[int]],
        destinations: Union[List[str], List[int]],
        edge_features: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> np.ndarray:
        """Return embedding for given edges using provided method.

        Parameters
        --------------------------
        sources:Union[List[str], List[int]]
            List of source nodes whose embedding is to be returned.
        destinations:Union[List[str], List[int]]
            List of destination nodes whose embedding is to be returned.
        edge_features: Optional[Union[np.ndarray, List[np.ndarray]]] = None
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
        if self.method == "CosineSimilarity":
            if (
                not isinstance(sources, np.ndarray) or
                not isinstance(destinations, np.ndarray)
            ):
                raise NotImplementedError(
                    "The Cosine Similarity is currently implemented exclusively for "
                    "numpy arrays of type uint32, but you have provided objects of type "
                    f"{type(sources)} and {type(destinations)}. "
                )
            if (
                sources.dtype != np.uint32 or
                destinations.dtype != np.uint32
            ):
                raise NotImplementedError(
                    "The Cosine Similarity is currently implemented exclusively for "
                    "numpy arrays of type uint32, but you have provided objects of type "
                    f"{sources.dtype} and {destinations.dtype}. "
                )
            if self._transformer._node_feature.dtype != np.float32:
                raise NotImplementedError(
                    "The Cosine Similarity is currently implemented exclusively for "
                    "node features contained in a numpy array of type float32, but "
                    f"you have provided an object of type {self._transformer._node_feature.dtype}."
                )

            edge_embeddings = self._method(
                embedding=self._transformer._node_feature,
                source_node_ids=sources,
                destination_node_ids=destinations,
            )
        else:
            edge_embeddings = self._method(
                self._transformer.transform(sources),
                self._transformer.transform(destinations)
            )

        if edge_features is not None:
            if not isinstance(edge_features, list):
                edge_features = [edge_features]
            
            for edge_feature in edge_features:
                if len(edge_feature.shape) != 2:
                    raise ValueError(
                        (
                            "The provided edge features should have a bidimensional shape, "
                            "but the provided one has shape {}."
                        ).format(edge_feature.shape)
                    )
                if edge_feature.shape[0] != edge_embeddings.shape[0]:
                    raise ValueError(
                        (
                            "The provided edge features should have a sample for each of the edges "
                            "in the graph, which are {}, but were {}."
                        ).format(
                            edge_embeddings.shape[0],
                            edge_feature.shape[0]
                        )
                    )
            edge_embeddings = np.hstack([
                edge_embeddings,
                *edge_features
            ])

        return edge_embeddings
