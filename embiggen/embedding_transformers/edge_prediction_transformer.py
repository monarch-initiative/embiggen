"""EdgePredictionTransformer class to convert graphs to edge embeddings to execute edge prediction."""
from typing import Tuple, Union, List, Optional
import pandas as pd
import numpy as np
from ensmallen import Graph  # pylint: disable=no-name-in-module

from embiggen.embedding_transformers.graph_transformer import GraphTransformer


class EdgePredictionTransformer:
    """EdgePredictionTransformer class to convert graphs to edge embeddings."""

    def __init__(
        self,
        method: str = "Hadamard",
        aligned_mapping: bool = False,
    ):
        """Create new EdgePredictionTransformer object.

        Parameters
        ------------------------
        method: str = "hadamard",
            Method to use for the embedding.
            Can either be 'Hadamard', 'Sum', 'Average', 'L1', 'AbsoluteL1', 'L2' or 'Concatenate'.
        aligned_mapping: bool = False,
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated edge embedding
            will be meaningless.
        """
        self._transformer = GraphTransformer(
            method=method,
            aligned_mapping=aligned_mapping,
        )

    def fit(
        self,
        node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
        node_type_feature: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
    ):
        """Fit the model.

        Parameters
        -------------------------
        node_feature: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
            Node feature to use to fit the transformer.
        node_type_feature: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            Node type feature to use to fit the transformer.

        Raises
        -------------------------
        ValueError
            If the given method is None there is no need to call the fit method.
        """
        self._transformer.fit(
            node_feature,
            node_type_feature=node_type_feature
        )

    def transform(
        self,
        positive_graph: Union[Graph, np.ndarray, List[List[str]], List[List[int]]],
        negative_graph: Union[Graph, np.ndarray, List[List[str]], List[List[int]]],
        edge_features: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        random_state: int = 42,
        shuffle: bool = False
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
        edge_features: Optional[Union[np.ndarray, List[np.ndarray]]] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the provided graph.
        random_state: int = 42,
            The random state to use to shuffle the labels.
        shuffle: bool = False
            Whether to shuffle the samples

        Raises
        --------------------------
        ValueError,
            If embedding is not fitted.

        Returns
        --------------------------
        Tuple with X and y values.
        """
        if isinstance(positive_graph, Graph) and isinstance(negative_graph, Graph):
            if not positive_graph.has_compatible_node_vocabularies(negative_graph):
                raise ValueError(
                    "The provided positive and negative graphs are not compatible. "
                    "Possible causes for this may be a different node vocabulary, "
                    "presence or absence of node types or different node type vocabulary, "
                    "the presence or absence of edge weights or types in one of the "
                    "two graphs."
                )

        if edge_features is not None:
            if not isinstance(edge_features, list):
                edge_features = [edge_features]
            
            if isinstance(positive_graph, Graph):
                number_of_positive_edges = positive_graph.get_number_of_directed_edges()
            else:
                number_of_positive_edges = len(positive_graph)
            
            positive_edge_features = [
                edge_feature[:number_of_positive_edges]
                for edge_feature in edge_features
            ]
            negative_edge_features  = [
                edge_feature[number_of_positive_edges:]
                for edge_feature in edge_features
            ]
        else:
            positive_edge_features = negative_edge_features = None

        positive_edge_embedding = self._transformer.transform(
            positive_graph,
            node_types=positive_graph,
            edge_features=positive_edge_features
        )
        negative_edge_embedding = self._transformer.transform(
            negative_graph,
            node_types=negative_graph,
            edge_features=negative_edge_features
        )

        edge_embeddings = np.vstack([
            positive_edge_embedding,
            negative_edge_embedding
        ])

        edge_labels = np.concatenate([
            np.ones(positive_edge_embedding.shape[0]),
            np.zeros(negative_edge_embedding.shape[0])
        ])

        if shuffle:
            numpy_random_state = np.random.RandomState(  # pylint: disable=no-member
                seed=random_state
            )

            indices = numpy_random_state.permutation(edge_labels.shape[0])

            edge_embeddings, edge_labels = edge_embeddings[indices], edge_labels[indices]

        return edge_embeddings, edge_labels