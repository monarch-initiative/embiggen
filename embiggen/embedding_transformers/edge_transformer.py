"""EdgeTransformer class to convert edges to edge embeddings."""
from typing import List, Optional, Union, Optional

import numpy as np
import pandas as pd
from ensmallen import express_measures
from userinput.utils import must_be_in_set

from embiggen.embedding_transformers.node_transformer import NodeTransformer


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


def get_l1_norm_edge_embedding(
    edge_embedding: np.ndarray
) -> np.ndarray:
    """Return L1 norm for each edge embedding row.

    Parameters
    --------------------------
    edge_embedding: np.ndarray
        Numpy array with the edge embeddings.

    Returns
    --------------------------
    Numpy array with the L1 norm scalar scores.
    """
    assert edge_embedding.ndim == 2
    return np.abs(edge_embedding).sum(axis=1, keepdims=True)


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


def get_l2_norm_edge_embedding(
    edge_embedding: np.ndarray
) -> np.ndarray:
    """Return L2 norm for each edge embedding row.

    Parameters
    --------------------------
    edge_embedding: np.ndarray
        Numpy array with the edge embeddings.

    Returns
    --------------------------
    Numpy array with the L2 norm scalar scores.
    """
    assert edge_embedding.ndim == 2
    return np.sqrt(np.power(edge_embedding, 2.0).sum(axis=1, keepdims=True))


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
    source_node_embedding: np.ndarray,
    destination_node_embedding: np.ndarray
) -> np.ndarray:
    """Return cosine similarity of the two nodes.

    Parameters
    --------------------------
    source_node_embedding: np.ndarray
        Numpy array with the embedding of the source node.
    destination_node_embedding: np.ndarray
        Numpy array with the embedding of the destination node.

    Returns
    --------------------------
    Numpy array with the cosine similarity.
    """
    assert source_node_embedding.dtype == destination_node_embedding.dtype
    assert source_node_embedding.shape == destination_node_embedding.shape
    hadamard_product = source_node_embedding * destination_node_embedding
    norm = (
        get_l2_norm_edge_embedding(source_node_embedding) *
        get_l2_norm_edge_embedding(destination_node_embedding)
    )
    norm[norm < 1e-6] = 1e-6
    return np.sum(hadamard_product, axis=1, keepdims=True) / norm


def get_concatenate_edge_embedding(
    source_node_embedding: np.ndarray,
    destination_node_embedding: np.ndarray
) -> np.ndarray:
    """Return Concatenate edge embedding of the two nodes.

    Parameters
    --------------------------
    source_node_embedding: np.ndarray
        Numpy array with the embedding of the source node.
    destination_node_embedding: np.ndarray
        Numpy array with the embedding of the destination node.

    Returns
    --------------------------
    Numpy array with the Concatenate edge embedding.
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
    }

    def __init__(
        self,
        methods: Union[List[str], str] = "Hadamard",
        aligned_mapping: bool = False,
    ):
        """Create new EdgeTransformer object.

        Parameters
        ------------------------
        method: Union[List[str], str] = "Hadamard",
            Method to use for the embedding.
            If None is used, we return instead the numeric tuples.
            If multiple edge embedding are provided, they
            will be Concatenated and fed to the model.
            The supported edge embedding methods are:
             * Hadamard: element-wise product
             * Sum: element-wise sum
             * Average: element-wise mean
             * L1: element-wise subtraction
             * AbsoluteL1: element-wise subtraction in absolute value
             * SquaredL2: element-wise subtraction in squared value
             * L2: element-wise squared root of squared subtraction
             * Concatenate: Concatenate of source and destination node features
             * Min: element-wise minimum
             * Max: element-wise maximum
             * L2Distance: vector-wise L2 distance - this yields a scalar
             * CosineSimilarity: vector-wise cosine similarity - this yields a scalar
        aligned_mapping: bool = False,
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated edge embedding
            will be meaningless.
        """
        self._transformer = NodeTransformer(
            aligned_mapping=aligned_mapping,
        )

        if not isinstance(methods, list):
            methods = [methods]

        normalized_methods = []

        for method_name in methods:
            if not isinstance(method_name, str):
                raise ValueError(
                    "The provided method name should be a string, but we got "
                    f"{type(method_name)} instead."
                )
            normalized_methods.append(must_be_in_set(
                method_name,
                set(self.methods.keys()),
                "edge embedding method"
            ))

        self._methods = [
            self.methods[method_name]
            for method_name in normalized_methods
        ]
        self._method_names = normalized_methods
        self._edge_type_features = []

    def fit(
        self,
        node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
        node_type_feature: Optional[Union[pd.DataFrame, np.ndarray,
                                          List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray,
                                           List[Union[pd.DataFrame, np.ndarray]]]] = None,
    ):
        """Fit the model.

        Parameters
        -------------------------
        node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
            Node feature to use to fit the transformer.
        node_type_feature: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            Node type feature to use to fit the transformer.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            Edge type feature to use to fit the transformer.
        """
        if edge_type_features is None:
            edge_type_features = []

        if not isinstance(edge_type_features, list):
            edge_type_features = [edge_type_features]

        # It also a good idea to check whether the edge type features provided
        # have all the same shape, and whether they contain any NaN value, so
        # to catch them early rather than later.

        # We expect that the edge type features can either be NumPy arrays of
        # Pandas Dataframes. If they are pandas dataframes, we will be able
        # to use their index as a way to map eventual edge types ids provided
        # with the string name.
        for edge_type_feature in edge_type_features:
            if not isinstance(edge_type_feature, (pd.DataFrame, np.ndarray)):
                raise ValueError(
                    "The provided edge type features should be either Pandas Dataframes or "
                    f"Numpy arrays, but we got {type(edge_type_feature)} instead."
                )

            if isinstance(edge_type_feature, pd.DataFrame):
                if edge_type_feature.index.hasnans:
                    raise ValueError(
                        "The provided edge type features should "
                        "not have NaN values in their index."
                    )

                if edge_type_feature.index.has_duplicates:
                    raise ValueError(
                        "The provided edge type features should "
                        "not have duplicated values in their index."
                    )

                if pd.isna(edge_type_feature).any():
                    raise ValueError(
                        "The provided edge type features should "
                        "not have NaN values in their values."
                    )

            if isinstance(edge_type_feature, np.ndarray):
                if np.isnan(edge_type_feature).any():
                    raise ValueError(
                        "The provided edge type features should "
                        "not have NaN values in their values."
                    )

        self._edge_type_features = edge_type_features
        self._transformer.fit(
            node_feature=node_feature,
            node_type_feature=node_type_feature,
        )

    def has_edge_type_features(self) -> bool:
        """Return whether the transformer has edge type features."""
        return len(self._edge_type_features) > 0
    
    def has_node_type_features(self) -> bool:
        """Return whether the transformer has node type features."""
        return self._transformer.has_node_type_features()
    
    def is_aligned_mapping(self) -> bool:
        """Return whether the transformer has aligned mapping."""
        return self._transformer.is_aligned_mapping()

    def has_numpy_edge_type_features(self) -> bool:
        """Returns whether any of the edge type features provided is a numpy array."""
        return any([
            isinstance(edge_type_feature, np.ndarray)
            for edge_type_feature in self._edge_type_features
        ])

    def transform(
        self,
        sources: Union[List[str], List[int]],
        destinations: Union[List[str], List[int]],
        source_node_types: Optional[Union[List[Optional[List[str]]],
                                          List[Optional[List[int]]]]] = None,
        destination_node_types: Optional[Union[List[Optional[List[str]]],
                                               List[Optional[List[int]]]]] = None,
        edge_types: Optional[Union[List[str], List[int], np.ndarray]] = None,
        edge_features: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> np.ndarray:
        """Return embedding for given edges using provided method.

        Parameters
        --------------------------
        sources:Union[List[str], List[int]]
            List of source nodes whose embedding is to be returned.
        destinations:Union[List[str], List[int]]
            List of destination nodes whose embedding is to be returned.
        source_node_types: Optional[Union[List[Optional[List[str]]], List[Optional[List[int]]]]] = None,
            List of source node types whose embedding is to be returned.
            This can be either a list of strings, or a graph, or if the
            aligned_mapping is setted, then this methods also accepts
            a list of ints.
        destination_node_types: Optional[Union[List[Optional[List[str]]], List[Optional[List[int]]]]] = None,
            List of destination node types whose embedding is to be returned.
            This can be either a list of strings, or a graph, or if the
            aligned_mapping is setted, then this methods also accepts
            a list of ints.
        edge_types: Optional[Union[List[str], List[int], np.ndarray]] = None
            List of edge types whose embedding is to be returned.
        edge_features: Optional[Union[np.ndarray, List[np.ndarray]]] = None
            Optional edge features to be used as input Concatenated
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
        if self.has_edge_type_features() and edge_types is None:
            raise ValueError(
                "The edge type features are provided, but no edge types are provided."
            )
        
        if self.has_node_type_features() and source_node_types is None:
            raise ValueError(
                "The node type features are provided, but no source node types are provided."
            )
        
        if len(sources) != len(destinations):
            raise ValueError(
                "The provided sources and destinations should have the same length, "
                f"but we got {len(sources)} and {len(destinations)} instead."
            )
        
        # The source, destination and edge types, if provided, must have the same length.
        if edge_types is not None and len(destinations) != len(edge_types):
            raise ValueError(
                "The provided sources, destinations and edge types should have the same length, "
                f"but we got {len(sources)}, {len(destinations)} and {len(edge_types)} instead."
            )

        edge_type_features: List[np.ndarray] = []

        for edge_type_feature in self._edge_type_features:
            if isinstance(edge_type_feature, pd.DataFrame):
                if isinstance(edge_types[0], str):
                    edge_type_features.append(
                        edge_type_feature.loc[edge_types].values
                    )
                elif isinstance(edge_types[0], (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                    edge_type_features.append(
                        edge_type_feature.iloc[edge_types].values
                    )
                else:
                    raise ValueError(
                        "The provided edge types should be either strings or integers, but we got "
                        f"{type(edge_types[0])} instead."
                    )
            elif isinstance(edge_type_feature, np.ndarray):
                if isinstance(edge_types[0], str):
                    raise ValueError(
                        "Since the edge type features are provided as numpy arrays, "
                        "the edge types should be provided as integers. "
                        f"We got instead {type(edge_types[0])}."
                    )
                elif isinstance(edge_types[0], (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                    edge_type_features.append(
                        edge_type_feature[edge_types]
                    )
                else:
                    raise ValueError(
                        "The provided edge types should be either strings or integers, but we got "
                        f"{type(edge_types[0])} instead."
                    )
            else:
                raise ValueError(
                    "The provided edge type features should be either Pandas Dataframes or "
                    f"Numpy arrays, but we got {type(edge_type_feature)} instead."
                )
            
        assert len(self._edge_type_features) == len(edge_type_features)
        
        edge_embeddings: List[np.ndarray] = []
        if self._transformer.is_fit():
            for method in self._methods:
                edge_embedding = method(
                    self._transformer.transform(
                        sources,
                        node_types=source_node_types
                    ),
                    self._transformer.transform(
                        destinations,
                        node_types=destination_node_types
                    )
                )
                assert not np.isnan(edge_embedding).any(), (
                    "The provided edge embedding should not have NaN values, but we got "
                    f"a numpy array with shape {edge_embedding.shape} and NaN values. "
                    f"The object was obtained using the method {method}."
                )
                edge_embeddings.append(edge_embedding)

        if edge_features is None:
            edge_features = []

        if not isinstance(edge_features, list):
            edge_features = [edge_features]

        for edge_feature in edge_features:
            if not isinstance(edge_feature, np.ndarray):
                raise ValueError(
                    "The provided edge features should be numpy arrays, but we got "
                    f"{type(edge_feature)} instead."
                )

            if len(edge_embeddings) > 0 and edge_feature.shape[0] != edge_embeddings[0].shape[0]:
                raise ValueError(
                    "The provided edge features should have a sample for each of the edges "
                    f"in the graph, which are {sources.shape[0]}, with embedding "
                    f"shape {edge_embeddings[0].shape}, but has shape {edge_feature.shape}."
                )

        if all([
            len(features) == 0
            for features in (
                edge_features,
                edge_type_features,
                edge_embeddings
            )
        ]):
            raise ValueError(
                "At least one of the provided edge features, edge type features or edge embeddings "
                "should be provided."
            )

        expected_shape: Optional[int] = None
        for features, feature_kind in (
            (edge_features, "edge features"),
            (edge_type_features, "edge type features"),
            (edge_embeddings, "edge embeddings")
        ):
            if len(features) > 0 and expected_shape is None:
                # We imprint the expected shape on the
                # first feature available.
                expected_shape = features[0].shape[0]
            # We check that all the features have the same first dimension.
            for feature in features:
                assert not np.isnan(feature).any(), (
                    "The provided edge features should not have NaN values, but we got "
                    f"a numpy array with shape {feature.shape} and NaN values. "
                    f"It is a {feature_kind}."
                )
                if feature.shape[0] != expected_shape:
                    raise ValueError(
                        "The provided edge features should have a sample for each of the edges "
                        f"in the graph, which are {expected_shape}, but we got {feature.shape[0]}."
                    )


        result = np.hstack([
            *[
                edge_embedding.reshape((expected_shape, -1))
                for edge_embedding in edge_embeddings
            ],
            *[
                edge_feature.reshape((expected_shape, -1))
                for edge_feature in edge_features
            ],
            *[
                edge_type_feature.reshape((expected_shape, -1))
                for edge_type_feature in edge_type_features
            ]
        ])

        return result
