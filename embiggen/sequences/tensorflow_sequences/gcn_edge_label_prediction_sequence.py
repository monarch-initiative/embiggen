"""Keras Sequence for edge-label prediction GNN and GCN."""
from typing import List, Optional, Union, Type, Tuple

import numpy as np
import tensorflow as tf
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import VectorSequence, Sequence
from embiggen.utils import AbstractEdgeFeature
from embiggen.sequences.tensorflow_sequences.gcn_edge_prediction_training_sequence import GCNEdgePredictionTrainingSequence


class GCNEdgeLabelPredictionSequence(Sequence):
    """Keras Sequence for running Neural Network on graph edge-label prediction."""

    def __init__(
        self,
        graph: Graph,
        support: Graph,
        kernel: tf.SparseTensor,
        return_node_types: bool = False,
        return_node_ids: bool = False,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[Union[np.ndarray, Type[AbstractEdgeFeature], List[Union[Type[AbstractEdgeFeature], np.ndarray]]]] = None,
        use_edge_metrics: bool = False,
    ):
        """Create new Open-world assumption GCN training sequence for edge prediction.

        Parameters
        --------------------------------
        graph: Graph,
            The graph from which to sample the edges.
        support: Graph
            The graph to be used for the topological metrics.
        kernel: tf.SparseTensor
            The kernel to be used for the convolutions.
        return_node_types: bool = False
            Whether to return the node types.
        return_edge_types: bool = False
            Whether to return the edge types.
        return_node_ids: bool = False
            Whether to return the node IDs.
            These are needed when an embedding layer is used.
        node_features: List[np.ndarray]
            The node features to be used.
        node_type_features: Optional[List[np.ndarray]]
            The node type features to be used.
            For instance, these could be BERT embeddings of the
            description of the node types.
            When the graph has multilabel node types,
            we will average the features.
        edge_features: Optional[List[np.ndarray]] = None,
            The edge features to be used.
            For instance, these could be BERT embeddings of the
            description of the edges.
        use_edge_metrics: bool = False
            Whether to return the edge metrics.
        """
        super().__init__(
            sample_number=graph.get_number_of_edges(),
            batch_size=graph.get_number_of_nodes(),
        )

        # We use the GCN edge prediction training sequence to avoid
        # duplicating the complex logic that handles the computation
        # of the kernels, node and node type metrics.
        self._gcn_edge_prediction_training_sequence = GCNEdgePredictionTrainingSequence(
            graph=graph,
            support=support,
            kernel=kernel,
            return_node_types=return_node_types,
            return_node_ids=return_node_ids,
            node_features=node_features,
            node_type_features=node_type_features,
        )

        # Differently from the GCN edge prediction training sequence, the edge-label
        # prediction training sequence can handle the dense edge features, if provided.
        # We proceed to validate the provided edge features, if any.
        # The edge features that are supported as:
        # - One or more numpy array of shape (number_of_edges, number_of_features)
        # - One or more edge feature classes

        if edge_features is None:
            edge_features = []

        if not isinstance(edge_features, list):
            edge_features = [edge_features]

        rasterized_edge_features = []

        if use_edge_metrics:
            rasterized_edge_features.append(
                support.get_all_edge_metrics(
                    normalize=True,
                    subgraph=graph,
                )
            )

        for edge_feature in edge_features:
            if not isinstance(edge_feature, np.ndarray) and not issubclass(type(edge_feature), AbstractEdgeFeature):
                raise ValueError(
                    "The edge features should be provided as a numpy array or "
                    f"an edge feature class. Got {type(edge_feature)} instead."
                )
            if isinstance(edge_feature, np.ndarray) and len(edge_feature.shape) != 2:
                raise ValueError(
                    "The edge features should be provided as a numpy array with "
                    "shape (number_of_edges, number_of_features). "
                    f"Got {edge_feature.shape} instead."
                )
            if isinstance(edge_feature, np.ndarray) and edge_feature.shape[0] != graph.get_number_of_edges():
                raise ValueError(
                    "The edge features should be provided as a numpy array with "
                    f"shape (number_of_edges, number_of_features). "
                    f"Got {edge_feature.shape} instead, while the number of edges is {graph.get_number_of_edges()}."
                )
            if isinstance(edge_feature, np.ndarray):
                rasterized_edge_features.append(edge_feature)
            elif issubclass(type(edge_feature), AbstractEdgeFeature):
                if not edge_feature.is_fit():
                    edge_feature.fit(support=support)
                for feature in edge_feature.get_edge_feature_from_graph(
                    graph=graph,
                    support=support,
                ).values():
                    rasterized_edge_features.append(feature.reshape(feature.shape[0], -1))

        self._edge_features = [
            VectorSequence(
                rasterized_edge_feature,
                batch_size=graph.get_number_of_nodes(),
            )
            for rasterized_edge_feature in rasterized_edge_features
        ]
        self._sources = VectorSequence(
            graph.get_source_node_ids(graph.is_directed()),
            batch_size=graph.get_number_of_nodes(),
        )
        self._destinations = VectorSequence(
            graph.get_destination_node_ids(graph.is_directed()),
            batch_size=graph.get_number_of_nodes(),
        )
        
    def get_node_features(self) -> Tuple[np.ndarray]:
        """Return the node features."""
        return self._gcn_edge_prediction_training_sequence.get_node_features()

    def __getitem__(self, idx: int):
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be returned.

        Returns
        ---------------
        Return Tuple containing X and Y numpy arrays corresponding to given batch index.
        """
        edge_features = [
            edge_feature[idx]
            for edge_feature in self._edge_features
        ]
        sources = self._sources[idx]
        destinations = self._destinations[idx]

        # If this last batch is smaller than the batch size, we need to pad it.
        # This is necessary because in GCNs, the batch size is fixed to the number of nodes.
        delta = self.batch_size - sources.shape[0]
        if delta > 0:
            edge_features = [
                np.pad(edge_feature, [(0, delta), (0, 0)])
                for edge_feature in edge_features
            ]
            sources = np.pad(sources, (0, delta))
            destinations = np.pad(destinations, (0, delta))

        assert sources.shape[0] == destinations.shape[0] == self.batch_size

        for edge_feature in edge_features:
            assert edge_feature.shape[0] == self.batch_size

        # We need to reshape the source and destination nodes to be of shape
        # (batch_size, 1) instead of (batch_size, ). This is necessary because
        # this way these features match exactly the shape of the expected input
        # of the model.

        sources = sources.reshape(-1, 1)
        destinations = destinations.reshape(-1, 1)

        return (
            (
                sources,
                destinations,
                *edge_features,
                *self.get_node_features()
            ),
        )
