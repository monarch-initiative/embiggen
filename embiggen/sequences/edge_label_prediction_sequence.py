"""Keras Sequence for running Neural Network on graph edge prediction."""
from typing import Union, Tuple, List
import numpy as np

from ensmallen import Graph  # pylint: disable=no-name-in-module
from .edge_prediction_sequence import EdgePredictionSequence


class EdgeLabelPredictionSequence(EdgePredictionSequence):
    """Keras Sequence for running a Neural Network on graph edge-label prediction."""

    def __init__(
        self,
        graph: Graph,
        use_node_types: bool = False,
        use_edge_metrics: bool = False,
        batch_size: int = 2**10,
        support_mirrored_strategy: bool = False,
        graph_to_avoid: Graph = None,
        batches_per_epoch: Union[int, str] = "auto",
        elapsed_epochs: int = 0,
        random_state: int = 42
    ):
        """Create new EdgeLabekPredictionSequence object.

        Parameters
        --------------------------------
        graph: Graph,
            The graph from which to sample the edges.
        use_node_types: bool = False,
            Whether to return the node types.
        use_edge_metrics: bool = False,
            Whether to return the edge metrics.
        batch_size: int = 2**10,
            The batch size to use.
        support_mirrored_strategy: bool = False,
            Wethever to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
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
        super().__init__(
            graph,
            use_node_types=use_node_types,
            use_edge_types=True,
            return_only_edges_with_known_edge_types=True,
            use_edge_metrics=use_edge_metrics,
            batch_size=batch_size,
            negative_samples_rate=0,
            avoid_false_negatives=False,
            support_mirrored_strategy=support_mirrored_strategy,
            graph_to_avoid=graph_to_avoid,
            batches_per_epoch=batches_per_epoch,
            elapsed_epochs=elapsed_epochs,
            random_state=random_state
        )

    def __getitem__(self, idx: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be returned.

        Returns
        ---------------
        Return Tuple containing X and Y numpy arrays corresponding to given batch index.
        """
        edge_data, _ = super().__getitem__(idx)
        return edge_data[:-1], edge_data[-1]
