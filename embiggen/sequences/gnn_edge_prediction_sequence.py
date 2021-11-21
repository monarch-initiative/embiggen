"""Keras Sequence for running GNN on graph edge prediction."""
from typing import Tuple, Union, List, Optional

import numpy as np
import pandas as pd
from ensmallen import Graph  # pylint: disable=no-name-in-module
from .edge_prediction_sequence import EdgePredictionSequence


class GNNEdgePredictionSequence(EdgePredictionSequence):
    """Keras Sequence for running GNN on graph edge prediction."""

    def __init__(
        self,
        graph: Graph,
        node_features: Optional[Union[pd.DataFrame, List[pd.DataFrame], np.ndarray, List[np.ndarray]]] = None,
        use_node_types: bool = False,
        use_edge_metrics: bool = False,
        batch_size: int = 2**10,
        negative_samples_rate: float = 0.5,
        avoid_false_negatives: bool = False,
        support_mirrored_strategy: bool = False,
        batches_per_epoch: Union[int, str] = "auto",
        elapsed_epochs: int = 0,
        random_state: int = 42
    ):
        """Create new GNNEdgePredictionSequence object.

        Parameters
        --------------------------------
        graph: Graph,
            The graph from which to sample the edges.
        use_node_types: bool = False,
            Whether to return the node types.
        use_edge_types: bool = False,
            Whether to return the edge types.
        use_edge_metrics: bool = False,
            Whether to return the edge metrics.
        batch_size: int = 2**10,
            The batch size to use.
        negative_samples_rate: float = 0.5,
            Factor of negatives to use in every batch.
            For example, with a batch size of 128 and negative_samples_rate equal
            to 0.5, there will be 64 positives and 64 negatives.
        avoid_false_negatives: bool = False,
            Whether to filter out false negatives.
            By default False.
            Enabling this will slow down the batch generation while (likely) not
            introducing any significant gain to the model performance.
        support_mirrored_strategy: bool = False,
            Whether to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        filter_none_values: bool = True
            Whether to filter None values.
        graph_to_avoid: Graph = None,
            Graph to avoid when generating the edges.
            This can be the validation component of the graph, for example.
            More information to how to generate the holdouts is available
            in the Graph package.
        batches_per_epoch: Union[int, str] = "auto",
            Number of batches per epoch.
            If auto, it is used: `10 * edges number /  batch size`
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        random_state: int = 42,
            The random_state to use to make extraction reproducible.
        """
        self._node_features = [
            node_feature.values
            if isinstance(node_feature, pd.DataFrame)
            else node_feature
            for node_feature in node_features
        ]
        super().__init__(
            graph,
            use_node_types=use_node_types,
            use_edge_types=False,
            use_edge_metrics=use_edge_metrics,
            batch_size=batch_size,
            negative_samples_rate=negative_samples_rate,
            avoid_false_negatives=avoid_false_negatives,
            support_mirrored_strategy=support_mirrored_strategy,
            filter_none_values=False,
            batches_per_epoch=batches_per_epoch,
            elapsed_epochs=elapsed_epochs,
            random_state=random_state
        )

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
        (sources, source_node_types, destinations,
         destination_node_types, _, _), labels = super().__getitem__(idx)
        return [
            node_feature[sources]
            for node_feature in self._node_features
        ] + [
            node_feature[destinations]
            for node_feature in self._node_features
        ] + [
            value
            for value in (
                sources, destinations, source_node_types, destination_node_types
            )
            if value is not None
        ], labels
