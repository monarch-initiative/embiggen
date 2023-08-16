"""Keras Sequence for Open-world assumption GCN."""
from typing import List, Optional, Tuple, Type, Union

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
        kernels: Optional[List[tf.SparseTensor]],
        batch_size: int,
        return_node_types: bool = False,
        return_edge_types: bool = False,
        return_node_ids: bool = False,
        return_edge_node_ids: bool = True,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_type_features: Optional[List[np.ndarray]] = None,
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
        kernels: Optional[List[tf.SparseTensor]]
            The kernel to be used for the convolutions.
        batch_size: int
            The batch size to use.
        return_node_types: bool = False
            Whether to return the node types.
        return_edge_types: bool = False
            Whether to return the edge types.
        return_node_ids: bool = False
            Whether to return the node IDs.
            These are needed when an embedding layer is used.
        return_edge_node_ids: bool = True
            Whether to return the edge node IDs.
            These are needed when an edge feature extractor is used.
        node_features: List[np.ndarray]
            The node features to be used.
        node_type_features: Optional[List[np.ndarray]]
            The node type features to be used.
            For instance, these could be BERT embeddings of the
            description of the node types.
            When the graph has multilabel node types,
            we will average the features.
        edge_type_features: Optional[List[np.ndarray]]
            The edge type features to be used.
            For instance, these could be BERT embeddings of the
            description of the edge types.
            When the graph has multilabel edge types,
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
        
        if edge_features is None:
            edge_features = []

        if not isinstance(edge_features, list):
            edge_features = [edge_features]

        if edge_type_features is None:
            edge_type_features = []
        
        if not isinstance(edge_type_features, list):
            edge_type_features = [edge_type_features]

        if support is None:
            support = graph

        if (
            return_node_types or
            len(node_type_features) > 0
        ) and graph.has_unknown_node_types():
            raise ValueError(
                f"The provided graph {graph.get_name()} "
                "contains unknown node types but node types "
                "have been requested for the sequence."
            )
        
        if (
            return_edge_types or
            len(edge_type_features) > 0
        ) and graph.has_unknown_edge_types():
            raise ValueError(
                f"The provided graph {graph.get_name()} "
                "contains unknown edge types but edge types "
                "have been requested for the sequence."
            )

        # We verify that the provided edge features are valid
        for edge_feature in edge_features:
            if not issubclass(type(edge_feature), AbstractEdgeFeature):
                raise NotImplementedError(
                    f"The provided edge feature of type {type(edge_feature)} "
                    "is not supported. Please provide an object of type "
                    "subclass of AbstractEdgeFeature."
                )

            if not edge_feature.is_fit():
                edge_feature.fit(support)

        if kernels is None:
            kernels = []

        if not isinstance(kernels, list):
            kernels = [kernels]

        for kernel in kernels:
            assert kernel is not None, "The provided kernel is None."

        self._kernels = kernels
        if node_features is None:
            node_features = []
        self._node_features = [
            node_feature.astype(np.float32)
            for node_feature in node_features
        ]

        # We need to reshape the node IDs into a column vector
        # so that they match exactly the shape expected by the
        # embedding layer of the model.
        self._node_ids = graph.get_node_ids().reshape(-1, 1) if return_node_ids and self.has_kernels() else None
        self._return_edge_node_ids = return_edge_node_ids

        if return_node_types or len(node_type_features) > 0:
            if graph.has_multilabel_node_types():
                maximal_multilabel_number = graph.get_maximum_multilabel_count()
                node_types = np.zeros((graph.get_number_of_nodes(), maximal_multilabel_number), dtype=np.int32)
                for node_id in range(graph.get_number_of_nodes()):
                    node_type: Optional[np.ndarray] = graph.get_node_type_ids_from_node_id(node_id)
                    if node_type is not None:
                        # We add one to the node type IDs so that the
                        # 0 value can be used for masking.
                        node_types[node_id, :len(node_type)] = node_type + 1
            else:
                node_types = graph.get_single_label_node_type_ids()

            assert node_types.shape[0] == graph.get_number_of_nodes(), (
                f"The provided node types have a different number of rows "
                f"than the number of nodes in the graph. Expected {graph.get_number_of_nodes()} "
                f"but found {node_types.shape[0]}. This is an internal error, please "
                "open an issue at the Embiggen GitHub repository."
            )

        if len(node_type_features) > 0:
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

        self._edge_type_features = edge_type_features
        self._return_edge_types = return_edge_types

        self._sequence = EdgePredictionSequence(
            graph=graph,
            support=support,
            return_node_types=False,
            return_edge_types=self.return_edge_types() or self.has_edge_type_features(),
            use_edge_metrics=use_edge_metrics,
            batch_size=batch_size
        )

        # We need to reshape the node types, when they are a flat array,
        # to be of shape (batch_size, 1) so that they can have a shape
        # identical to the shape expected from the model. Such reshaping
        # procedure comes to virtually no cost as it is just a view of
        # the original array.
        if return_node_types and not graph.has_multilabel_node_types():
            node_types = node_types.reshape(-1, 1)

        self._node_types = node_types if return_node_types else None

        self._edge_features = edge_features

        self._current_index = 0
        super().__init__(
            sample_number=graph.get_number_of_directed_edges(),
            batch_size=batch_size
        )

    def use_edge_metrics(self) -> bool:
        """Return whether to use edge metrics."""
        return self._sequence.use_edge_metrics()
    
    def return_node_types(self) -> bool:
        """Return whether to return node types."""
        return self._sequence.return_node_types()
    
    def return_edge_types(self) -> bool:
        """Return whether to return edge types."""
        return self._return_edge_types
    
    def get_graph(self) -> Graph:
        """Return graph."""
        return self._sequence.get_graph()
    
    def get_support(self) -> Graph:
        """Return support graph."""
        return self._sequence.get_support()

    def get_kernels(self) -> List[tf.SparseTensor]:
        """Return kernels."""
        return self._kernels
    
    def has_kernels(self) -> bool:
        """Return whether we have kernels."""
        return len(self._kernels) > 0
    
    def return_edge_node_ids(self) -> bool:
        """Return whether to return edge node IDs."""
        return self._return_edge_node_ids
    
    def get_edge_type_features(self, edge_type_ids: np.ndarray) -> Tuple[np.ndarray]:
        """Return edge type features."""
        edge_type_features = []

        for edge_type_feature in self._edge_type_features:
            edge_type_features.append(
                edge_type_feature[edge_type_ids]
            )

        return tuple(edge_type_features)
    
    def has_edge_type_features(self) -> bool:
        """Return whether we have edge type features."""
        return len(self._edge_type_features) > 0
    
    def get_node_features(
        self,
        sources: Optional[np.ndarray]=None,
        destinations: Optional[np.ndarray]=None
    ) -> Tuple[np.ndarray]:
        """Return the node features associated with the provided node IDs.
        
        Parameters
        ---------------------------
        sources: Optional[np.ndarray],
            The source node IDs.
        destinations: Optional[np.ndarray],
            The destination node IDs.

        Implementative details
        ---------------------------
        When the source and destination node IDs are not provided,
        we return the node features associated with all node IDs,
        as is expected when dealing with whole graph convolutions.

        When the source and destination node IDs are provided,
        we MUST not be in a graph convolution scenario, and we
        need to check that the provided kernels are an empty list
        otherwise we find ourselves in an illegal state.
        Furthermore, we must check that the length of the provided
        source and destination node IDs is the same, otherwise
        we find ourselves in an illegal state.

        Raises
        ---------------------------
        ValueError,
            When the provided source and destination node IDs
            are not both None or both not None.
        ValueError,
            When the provided source and destination node IDs
            have different lengths.
        ValueError,
            When the provided source and destination node IDs
            are not None but the provided kernels are not an
            empty list.
        """
        if not self.has_kernels() and (sources is None or destinations is None):
            raise ValueError(
                "The provided source and destination node IDs "
                "must be provided when kernels are not provided."
            )
        if sources is None and destinations is None:
            return tuple([
                node_feature
                for node_feature in (
                    *self._kernels,
                    *self._node_features,
                    *self._node_type_features,
                    self._node_ids,
                    self._node_types,
                )
                if node_feature is not None
            ])
        if sources is None or destinations is None:
            raise ValueError(
                "The provided source and destination node IDs "
                "must be either both None or both not None."
            )
        if len(sources) != len(destinations):
            raise ValueError(
                "The provided source and destination node IDs "
                "must have the same length."
            )
        
        if len(self._kernels) > 0:
            raise ValueError(
                "The provided source and destination node IDs "
                "must be None when kernels are provided."
            )
        
        return tuple([
            node_feature[node_ids.flatten()]
            for node_ids in (sources, destinations)
            for node_feature in (
                *self._node_features,
                *self._node_type_features,
                # Note that we do not return the node IDs as they are not needed when
                # we are not running a whole graph convolution.
                # self._node_ids,
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
            ).items():
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

        current_batch_size = sources.shape[0]

        edge_features: Tuple[np.ndarray] = self.get_edge_features_from_edge_node_ids(
            sources,
            destinations
        )

        edge_type_features = self.get_edge_type_features(values[2]) if self.has_edge_type_features() else []

        # If the edge types are NOT requested as the model does not have an edge type
        # embedding layer, we remove the edge type IDs from the values, since we have
        # retrieved them solely to index correctly the edge type features.
        if self.has_edge_type_features() and not self.return_edge_types():
            values[2] = None

        # We reshape both sources and destinations to be of shape
        # (batch_size, 1) so that they can be used as inputs for
        # the embedding layer.
        values[0] = values[0].reshape(-1, 1)
        values[1] = values[1].reshape(-1, 1)

        # We remove the edge node IDs if they are not requested.
        if not self._return_edge_node_ids:
            values[0] = None
            values[1] = None

        # If we have to return the edge types, we need to reshape
        # them to be of shape (batch_size, 1) so that they can be
        # used as inputs for the embedding layer.
        if self.return_edge_types():
            values[2] = values[2].reshape(-1, 1)

        # If necessary, we add the padding as the last batch may be
        # smaller than the required size (number of nodes).
        delta = self.batch_size - current_batch_size
        if delta > 0 and self.has_kernels():
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
            edge_type_features = [
                np.pad(edge_type_feature, [(0, delta), (0, 0)])
                for edge_type_feature in edge_type_features
            ]

        return (tuple([
            value
            for value in (
                *values,
                *edge_features,
                *edge_type_features,
                *(self.get_node_features() if self.has_kernels() else self.get_node_features(
                    sources=sources,
                    destinations=destinations
                ))
            )
            if value is not None
        ]),)
