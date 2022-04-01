"""Keras Sequence for running Neural Network on graph edge prediction."""
from typing import Union, Tuple, List, Optional
import numpy as np

from ensmallen import Graph

from embiggen.sequences.edge_label_prediction_sequence import EdgeLabelPredictionSequence  # pylint: disable=no-name-in-module


class BinaryEdgeLabelPredictionSequence(EdgeLabelPredictionSequence):
    """Keras Sequence for running a Neural Network on graph edge-label prediction."""

    def __init__(
        self,
        graph: Graph,
        positive_edge_type: Optional[Union[str, int]] = None,
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
        graph: Graph
            The graph from which to sample the edges.
        positive_edge_type: Optional[Union[str, int]] = None
            The type for the positive class.
            If None, we check if the provided graph has two classes,
            and if so we use the minority class as the positive class.
            If the graph does not have two classes, we will raise an error.
        use_node_types: bool = False
            Whether to return the node types.
        use_edge_metrics: bool = False
            Whether to return the edge metrics.
        batch_size: int = 2**10
            The batch size to use.
        support_mirrored_strategy: bool = False
            Wethever to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        graph_to_avoid: Graph = None
            Graph to avoid when generating the edges.
            This can be the validation component of the graph, for example.
            More information to how to generate the holdouts is available
            in the Graph package.
        batches_per_epoch: Union[int, str] = "auto"
            Number of batches per epoch.
            If auto, it is used: `10 * edges number /  batch size`
        elapsed_epochs: int = 0
            Number of elapsed epochs to init state of generator.
        random_state: int = 42
            The random_state to use to make extraction reproducible.
        """
        if positive_edge_type is None:
            if graph.get_edge_types_number() != 2:
                raise ValueError(
                    "The provided edge types number is None, so we would use "
                    "as positive class the class with least edges, but the "
                    "graph you have provided does not have two classes and "
                    "therefore we cannot infer exactly which of the minority "
                    "classes you would want to use."
                )
            else:
                edge_types_counts = graph.get_edge_type_names_counts_hashmap()
                positive_edge_type = min(
                    edge_types_counts,
                    key=edge_types_counts.get
                )

        if not isinstance(positive_edge_type, (int, str)):
            raise ValueError(
                (
                    "The positive edge type must be either an integer "
                    "or a string but the provided one is a {}."
                ).format(
                    type(positive_edge_type)
                )
            )
        if positive_edge_type < 0:
            raise ValueError(
                (
                    "The positive edge type must be a positive integer "
                    "but the provided edge type was {}."
                ).format(
                    type(positive_edge_type)
                )
            )

        # If the provided positive edge type is a string, we need
        # to convert this into the corresponding numeric type.
        # Do note that within Ensmallen the type is checked for existance,
        # that is when a non existant type is provided a ValueError exception
        # will be raised.
        # Furthermore, if the current graph does not contain edge type at all
        # an exception will be raised.
        if isinstance(positive_edge_type, str):
            positive_edge_type = graph.get_edge_type_id_from_edge_type_name(
                positive_edge_type
            )

        # We check if the provided positive edge type
        if positive_edge_type >= graph.get_edge_types_number():
            raise ValueError(
                (
                    "The current graph has {} edge types but "
                    "the provided edge type was {}."
                ).format(
                    graph.get_edge_types_number(),
                    positive_edge_type
                )
            )

        self._positive_edge_type = positive_edge_type
        super().__init__(
            graph,
            use_node_types=use_node_types,
            use_edge_metrics=use_edge_metrics,
            batch_size=batch_size,
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
        edge_data, edge_types = super().__getitem__(idx)
        return edge_data, edge_types == self._positive_edge_type
