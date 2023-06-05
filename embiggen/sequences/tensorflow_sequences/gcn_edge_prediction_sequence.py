"""Keras Sequence for Open-world assumption GCN."""
from typing import Tuple, List, Optional, Union, Type

import numpy as np
import tensorflow as tf
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence
from embiggen.sequences.generic_sequences import EdgePredictionSequence
from embiggen.utils import AbstractEdgeFeature


class GCNEdgePredictionSequence(Sequence):
    """Keras Sequence for running Neural Network on graph edge prediction."""

    def __init__(
        self,
        graph: Graph,
        support: Graph,
        kernel: tf.SparseTensor,
        return_node_types: bool = False,
        return_edge_types: bool = False,
        return_node_ids: bool = False,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None,
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
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None,
            The edge features to be used.
        use_edge_metrics: bool = False
            Whether to return the edge metrics.
        """
        if not graph.has_edges():
            raise ValueError(
                f"An empty instance of graph {graph.get_name()} was provided!"
            )
        
        if (
            return_node_types or
            node_type_features is not None
        ) and graph.has_unknown_node_types():
            raise ValueError(
                f"The provided graph {graph.get_name()} "
                "contains unknown node types but node types "
                "have been requested for the sequence."
            )

        if edge_features is None:
            edge_features = []

        if not isinstance(edge_features, list):
            edge_features = [edge_features]

        if support is None:
            support = graph

        # We verify that the provided edge features are valid
        for edge_feature in edge_features:
            if not issubclass(type(edge_feature), AbstractEdgeFeature):
                raise NotImplementedError(
                    f"The provided edge feature of type {type(edge_feature)} "
                    "is not supported. Please provide an object of type "
                    "subclass of AbstractEdgeFeature."
                )

            if not edge_feature.fit():
                edge_feature.fit(support)

        self._kernel = kernel
        if node_features is None:
            node_features = []
        self._node_features = [
            node_feature.astype(np.float32)
            for node_feature in node_features
        ]

        # We need to reshape the node IDs into a column vector
        # so that they match exactly the shape expected by the
        # embedding layer of the model.
        self._node_ids = graph.get_node_ids().reshape(-1, 1) if return_node_ids else None

        if return_node_types or node_type_features is not None:
            if graph.has_multilabel_node_types():
                node_types = graph.get_one_hot_encoded_node_types()
            else:
                node_types = graph.get_single_label_node_type_ids()

        if node_type_features is not None:
            if graph.has_multilabel_node_types():
                self._node_type_features = []
                minus_node_types = node_types - 1
                node_types_mask = node_types == 0
                for node_type_feature in node_type_features:
                    self._node_type_features.append(np.ma.array(
                        node_type_feature[minus_node_types],
                        mask=np.repeat(
                            node_types_mask,
                            node_type_feature.shape[1]
                        ).reshape((
                            *node_types.shape,
                            node_type_feature.shape[1]
                        ))
                    ).mean(axis=-2).data)
            else:
                self._node_type_features = [
                    node_type_feature[node_types]
                    for node_type_feature in node_type_features
                ]
        else:
            self._node_type_features = []

        self._sequence = EdgePredictionSequence(
            graph=graph,
            support=support,
            return_node_types=False,
            return_edge_types=return_edge_types,
            use_edge_metrics=use_edge_metrics,
            batch_size=support.get_number_of_nodes()
        )

        # We need to reshape the node types, when they are a flat array,
        # to be of shape (batch_size, 1) so that they can have a shape
        # identical to the shape expected from the model. Such reshaping
        # procedure comes to virtually no cost as it is just a view of
        # the original array.
        if return_node_types and not graph.has_multilabel_node_types():
            node_types = node_types.reshape(-1, 1)

        self._node_types = node_types if return_node_types else None

        self._edge_features = [] if edge_features is None else edge_features

        self._current_index = 0
        super().__init__(
            sample_number=graph.get_number_of_directed_edges(),
            batch_size=graph.get_number_of_nodes(),
        )

    def use_edge_metrics(self) -> bool:
        """Return whether to use edge metrics."""
        return self._sequence.use_edge_metrics()
    
    def return_node_types(self) -> bool:
        """Return whether to return node types."""
        return self._sequence.return_node_types()
    
    def return_edge_types(self) -> bool:
        """Return whether to return edge types."""
        return self._sequence.return_edge_types()
    
    def get_graph(self) -> Graph:
        """Return graph."""
        return self._sequence.get_graph()
    
    def get_support(self) -> Graph:
        """Return support graph."""
        return self._sequence.get_support()

    def get_kernel(self) -> tf.SparseTensor:
        """Return kernel."""
        return self._kernel

    def get_node_features(self) -> Tuple[np.ndarray]:
        """Return the node features."""
        return tuple([
            node_feature
            for node_feature in (
                self._kernel,
                *self._node_features,
                *self._node_type_features,
                self._node_ids,
                self._node_types,
            )
            if node_feature is not None
        ])
    
    def get_edge_features_from_edge_node_ids(self, sources: np.ndarray, destinations: np.ndarray) -> Tuple[np.ndarray]:
        """Returns the edge features associated with the provided edge node IDs.
        
        Parameters
        ---------------------------
        sources: np.ndarray,
            The source node IDs.
        destinations: np.ndarray,
            The destination node IDs.
            
        Returns
        ---------------------------
        The edge features associated with the provided edge node IDs.
        
        """
        rasterized_edge_features = []
        for edge_feature in self._edge_features:
            for edge_feature_name, rasterized_edge_feature in edge_feature.get_edge_feature_from_edge_node_ids(
                support=self.get_support(),
                sources=sources,
                destinations=destinations
            ).keys():
                if not isinstance(rasterized_edge_feature, np.ndarray):
                    raise ValueError(
                        f"The provided edge feature {edge_feature_name} "
                        f"from the edge feature extractor {edge_feature.get_feature_name()}"
                        f"is not a numpy array but a {type(rasterized_edge_feature)}."
                    )
                if rasterized_edge_feature.shape[0] != sources.shape[0]:
                    raise ValueError(
                        f"The provided edge feature {edge_feature_name} "
                        f"from the edge feature extractor {edge_feature.get_feature_name()}"
                        "has a different number of rows than the "
                        "number of edges in the graph. "
                        f"Expected {sources.shape[0]} but found {rasterized_edge_feature.shape[0]}."
                    )
                rasterized_edge_features.append(
                    rasterized_edge_feature.reshape(sources.shape[0], -1)
                )
        return tuple(rasterized_edge_features)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be returned.

        Returns
        ---------------
        Return Tuple containing X and Y numpy arrays corresponding to given batch index.
        """
        values = list(self._sequence[idx][0])
        sources = values[0]
        destinations = values[1]

        edge_features: Tuple[np.ndarray] = self.get_edge_features_from_edge_node_ids(sources, destinations)

        # We reshape both sources and destinations to be of shape
        # (batch_size, 1) so that they can be used as inputs for
        # the embedding layer.
        values[0] = values[0].reshape(-1, 1)
        values[1] = values[1].reshape(-1, 1)

        # If necessary, we add the padding as the last batch may be
        # smaller than the required size (number of nodes).
        delta = self.batch_size - values[0].shape[0]
        if delta > 0:
            values = [
                np.pad(value, (0, delta) if len(value.shape)
                       == 1 else [(0, delta), (0, 0)])
                for value in values
                if value is not None
            ]
            edge_features = [
                np.pad(edge_feature, [(0, delta), (0, 0)])
                for edge_feature in edge_features
            ]

        return (tuple([
            value
            for value in (
                *values,
                *edge_features,
                *self.get_node_features(),
            )
            if value is not None
        ]),)
