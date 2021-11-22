"""Keras Sequence for running GNN on graph edge prediction."""
from typing import Tuple, Union, List, Optional

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
                    self._source_node_type_ids[i, j] = node_type_id
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
                    self._destination_node_type_ids[i, j] = node_type_id

        self._node_features = [
            node_feature.values
            if isinstance(node_feature, pd.DataFrame)
            else node_feature
            for node_feature in node_features
        ]
        self._use_node_types = use_node_types
        self._return_node_ids = return_node_ids
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
        sources = np.full_like(self._destinations, super().__getitem__(idx)[0])
        source_node_types = None
        if self._source_node_type_ids is not None:
            source_node_types = np.tile(
                self._source_node_type_ids[idx],
                (1, self._destinations.size)
            )

        source_ids = None
        if self._return_node_ids:
            source_ids = sources

        destination_ids = None
        if self._return_node_ids:
            destination_ids = self._destinations

        return [
            node_feature[sources]
            for node_feature in self._node_features
        ] + [
            value
            for value in (
                source_ids, source_node_types
            )
            if value is not None
        ] + [
            node_feature[self._destinations]
            for node_feature in self._node_features
        ] + [
            value
            for value in (
                destination_ids, self._destination_node_type_ids
            )
            if value is not None
        ]
