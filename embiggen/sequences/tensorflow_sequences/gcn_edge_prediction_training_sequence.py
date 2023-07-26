"""Keras Sequence for Open-world assumption GCN."""
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import Sequence

from embiggen.sequences.tensorflow_sequences.gcn_edge_prediction_sequence import (
    GCNEdgePredictionSequence,
)
from embiggen.utils import AbstractEdgeFeature


class GCNEdgePredictionTrainingSequence(Sequence):
    """Keras Sequence for running Neural Network on graph edge prediction."""

    def __init__(
        self,
        graph: Graph,
        kernels: Optional[List[tf.SparseTensor]],
        batch_size: int,
        number_of_batches_per_epoch: Optional[int],
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[AbstractEdgeFeature]] = None,
        return_node_ids: bool = False,
        return_edge_node_ids: bool = True,
        return_node_types: bool = False,
        return_edge_types: bool = False,
        use_edge_metrics: bool = False,
        negative_samples_rate: float = 0.5,
        avoid_false_negatives: bool = False,
        graph_to_avoid: Graph = None,
        
        random_state: int = 42,
    ):
        """Create new Open-world assumption GCN training sequence for edge prediction.

        Parameters
        --------------------------------
        graph: Graph,
            The graph from which to sample the edges.
        kernels: Optional[List[tf.SparseTensor]]
            The kernel to be used for the convolutions.
        batch_size: int
            The batch size to use.
        number_of_batches_per_epoch: Optional[int]
            The number of batches to use per epoch.
            If None, the number of batches will be equal to the number of edges.
        support: Optional[Graph] = None
            The graph to use to compute the edge metrics.
        node_features: Optonal[List[np.ndarray]] = None
            The node features to be used.
        node_type_features: Optional[List[np.ndarray]]
            The node type features to be used.
            For instance, these could be BERT embeddings of the
            description of the node types.
            When the graph has multilabel node types,
            we will average the features.
        edge_type_features: Optional[List[np.ndarray]] = None
            The edge type features to be used.
            For instance, these could be BERT embeddings of the
            description of the edge types.
        edge_features: Optional[List[AbstractEdgeFeature]] = None,
            The edge features to be used.
        return_node_ids: bool = False
            Whether to return the node IDs.
            These are needed when a node embedding layer is used.
        return_edge_node_ids: bool = True
            Whether to return the edge node IDs.
        return_node_types: bool = False,
            Whether to return the node type IDs.
            These are needed when a node type embedding layer is used.
        return_edge_types: bool = False,
            Whether to return the edge type IDs.
            These are needed when an edge type embedding layer is used.
        use_edge_metrics: bool = False
            Whether to return the edge metrics.
        negative_samples_rate: float = 0.5
            Factor of negatives to use in every batch.
            For example, with a batch size of 128 and negative_samples_rate equal
            to 0.5, there will be 64 positives and 64 negatives.
        avoid_false_negatives: bool = False,
            Whether to filter out false negatives.
            By default False.
            Enabling this will slow down the batch generation while (likely) not
            introducing any significant gain to the model performance.
        graph_to_avoid: Graph = None,
            Graph to avoid when generating the edges.
            This can be the validation component of the graph, for example.
            More information to how to generate the holdouts is available
            in the Graph package.
        random_state: int = 42,
            The random_state to use to make extraction reproducible.
        """

        self._negative_samples_rate = negative_samples_rate
        self._avoid_false_negatives = avoid_false_negatives
        self._graph_to_avoid = graph_to_avoid
        self._random_state = random_state
        self._current_index = 0

        self._prediction_sequence = GCNEdgePredictionSequence(
            graph=graph,
            support=support,
            kernels=kernels,
            batch_size=batch_size,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            return_node_ids=return_node_ids,
            return_edge_node_ids=return_edge_node_ids,
            return_node_types=return_node_types,
            return_edge_types=return_edge_types,
            use_edge_metrics=use_edge_metrics,
        )

        if number_of_batches_per_epoch is None:
            sample_number = graph.get_number_of_edges()
        else:
            sample_number = number_of_batches_per_epoch * batch_size

        super().__init__(
            sample_number=sample_number,
            batch_size=batch_size,
        )

    def get_node_features(
        self,
        sources: Optional[np.ndarray] = None,
        destinations: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray]:
        """Return node features."""
        return self._prediction_sequence.get_node_features(
            sources=sources, destinations=destinations
        )

    def use_edge_metrics(self) -> bool:
        """Return whether to use edge metrics."""
        return self._prediction_sequence.use_edge_metrics()

    def get_graph(self) -> Graph:
        """Return graph."""
        return self._prediction_sequence.get_graph()

    def get_support(self) -> Graph:
        """Return support."""
        return self._prediction_sequence.get_support()

    def get_kernel(self) -> tf.SparseTensor:
        """Return kernel."""
        return self._prediction_sequence.get_kernel()

    def has_kernels(self) -> bool:
        """Return whether kernels are available."""
        return self._prediction_sequence.has_kernels()

    def return_edge_types(self) -> bool:
        """Return whether to return edge types."""
        return self._prediction_sequence.return_edge_types()

    def has_edge_type_features(self) -> bool:
        """Return whether to return edge type features."""
        return self._prediction_sequence.has_edge_type_features()
    
    def return_edge_node_ids(self) -> bool:
        """Return whether to return edge node IDs."""
        return self._prediction_sequence.return_edge_node_ids()

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
        (
            sources,
            _,
            destinations,
            _,
            edge_types,
            edge_metrics,
            labels,
        ) = self.get_graph().get_edge_prediction_mini_batch(
            (self._random_state + idx) * (1 + self.elapsed_epochs),
            return_node_types=False,
            return_edge_types=self.return_edge_types() or self.has_edge_type_features(),
            return_edge_metrics=self.use_edge_metrics(),
            batch_size=self.batch_size,
            sample_only_edges_with_heterogeneous_node_types=False,
            negative_samples_rate=self._negative_samples_rate,
            avoid_false_negatives=self._avoid_false_negatives,
            support=self.get_support(),
            graph_to_avoid=self._graph_to_avoid,
        )

        edge_features: Tuple[
            np.ndarray
        ] = self._prediction_sequence.get_edge_features_from_edge_node_ids(
            sources, destinations
        )

        edge_type_features = (
            self._prediction_sequence.get_edge_type_features(edge_types)
            if self.has_edge_type_features()
            else []
        )

        # We need to reshape the sources and destinations to be
        # column vectors, as expected by the model input shapes.
        sources = sources.reshape((-1, 1))
        destinations = destinations.reshape((-1, 1))

        if self.return_edge_types():
            edge_types = edge_types.reshape((-1, 1))

        return (
            tuple(
                [
                    value
                    for value in (
                        sources if self.return_edge_node_ids() else None,
                        destinations if self.return_edge_node_ids() else None,
                        (edge_types if self.return_edge_types() else None),
                        edge_metrics,
                        *edge_features,
                        *edge_type_features,
                        *(
                            self.get_node_features()
                            if self.has_kernels()
                            else self.get_node_features(
                                sources=sources, destinations=destinations
                            )
                        ),
                    )
                    if value is not None
                ]
            ),
            labels,
        )
