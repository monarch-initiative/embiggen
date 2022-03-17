"""Keras Sequence for running GNN on graph edge prediction."""
from typing import Union, List, Optional

import numpy as np
import pandas as pd
from ensmallen import Graph  # pylint: disable=no-name-in-module
from keras_mixed_sequence import VectorSequence
from tqdm.auto import tqdm


class GNNBipartiteEdgePredictionSequence(VectorSequence):
    """Keras Sequence for running GNN on graph edge prediction."""

    def __init__(
        self,
        graph: Graph,
        sources: np.ndarray,
        destinations: np.ndarray,
        node_features: Optional[Union[pd.DataFrame, List[pd.DataFrame], np.ndarray, List[np.ndarray]]] = None,
        use_node_types: bool = False,
        return_node_ids: bool = True,
        return_labels: bool = True,
        support_mirrored_strategy: bool = False,
        random_state: int = 42
    ):
        """Create new GNNEdgePredictionSequence object.

        Parameters
        --------------------------------
        TODO: Update
        """
        self._sources = sources
        self._destinations = destinations
        self._graph = graph

        self._source_node_type_ids = None
        self._destination_node_type_ids = None

        if use_node_types:
            self._source_node_type_ids = np.zeros((
                sources.size,
                graph.get_maximum_multilabel_count()
            ), dtype=np.uint32)
            for i, source_node_id in enumerate(tqdm(
                sources,
                desc="Computing source node types",
                leave=False,
                dynamic_ncols=True
            )):
                node_type_ids = graph.get_node_type_ids_from_node_id(
                    source_node_id
                )
                for j, node_type_id in enumerate(node_type_ids):
                    self._source_node_type_ids[i, j] = node_type_id + 1
            self._destination_node_type_ids = np.zeros((
                destinations.size,
                graph.get_maximum_multilabel_count()
            ), dtype=np.uint32)
            for i, destination_node_id in enumerate(tqdm(
                destinations,
                desc="Computing destination node types",
                leave=False,
                dynamic_ncols=True
            )):
                node_type_ids = graph.get_node_type_ids_from_node_id(
                    destination_node_id
                )
                for j, node_type_id in enumerate(node_type_ids):
                    # Here we need to offset the node type IDs by one
                    # because we use the node type zero as the padding
                    self._destination_node_type_ids[i, j] = node_type_id + 1

        self._node_features = [
            node_feature.values
            if isinstance(node_feature, pd.DataFrame)
            else node_feature
            for node_feature in node_features
        ]
        self._destination_node_features = [
            node_feature[self._destinations]
            for node_feature in self._node_features
        ]
        self._use_node_types = use_node_types
        self._return_node_ids = return_node_ids
        self._return_labels = return_labels

        if self._return_node_ids:
            self._destination_node_features.append(self._destinations)
        if self._use_node_types:
            self._destination_node_features.append(
                self._destination_node_type_ids)
        self._support_mirrored_strategy = support_mirrored_strategy
        super().__init__(
            sources,
            batch_size=1,
            random_state=random_state
        )

    def __getitem__(self, idx: int) -> List[np.ndarray]:
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be returned.

        Returns
        ---------------
        Return Tuple containing X and Y numpy arrays corresponding to given batch index.
        """
        sources = np.full_like(
            self._destinations,
            super().__getitem__(idx)[0]
        )
        source_node_types = None
        if self._source_node_type_ids is not None:
            source_node_types = np.tile(
                self._source_node_type_ids[idx],
                (self._destinations.size, 1)
            )

        source_ids = None
        if self._return_node_ids:
            source_ids = sources

        X = [
            node_feature[sources]
            for node_feature in self._node_features
        ] + [
            value
            for value in (
                source_ids, source_node_types
            )
            if value is not None
        ] + self._destination_node_features

        if self._return_labels:
            labels = np.fromiter(
                (
                    self._graph.has_edge_from_node_ids(
                        source_id,
                        destination_id
                    )
                    for source_id, destination_id in zip(
                        sources,
                        self._destinations
                    )
                ),
                dtype=np.bool
            )
            return X, labels
        return X
