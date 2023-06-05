"""Keras Sequence for running Neural Network on graph edge prediction."""
import numpy as np
from ensmallen import Graph  # pylint: disable=no-name-in-module
from embiggen.embedders.ensmallen_embedders.hyper_sketching import HyperSketching


class EdgePredictionSequence:
    """Keras Sequence for running Neural Network on graph edge prediction."""

    def __init__(
        self,
        graph: Graph,
        support: Graph,
        return_node_types: bool,
        return_edge_types: bool,
        use_edge_metrics: bool,
        batch_size: int = 2**10,
    ):
        """Create new EdgePredictionSequence object.

        Parameters
        --------------------------------
        graph: Graph
            The graph whose edges are to be predicted.
        support: Graph
            The graph that was used while training the current
            edge prediction model.
        return_node_types: bool
            Whether to return the node types.
        return_edge_types: bool
            Whether to return the edge types.
        use_edge_metrics: bool = False
            Whether to return the edge metrics.
        batch_size: int = 2**10,
            The batch size to use.
        """
        if not graph.has_compatible_node_vocabularies(support):
            raise ValueError(
                f"The provided graph {graph.get_name()} does not have a node vocabulary "
                "that is compatible with the provided graph used in training."
            )
        
        if support is None:
            support = graph

        self._graph = graph
        self._support = support
        self._return_node_types = return_node_types
        self._return_edge_types = return_edge_types
        self._use_edge_metrics = use_edge_metrics
        self._batch_size = batch_size

    def __len__(self) -> int:
        """Returns length of sequence."""
        return int(np.ceil(self._graph.get_number_of_directed_edges() / self._batch_size))
    
    def use_edge_metrics(self) -> bool:
        """Return whether to use edge metrics."""
        return self._use_edge_metrics
    
    def return_node_types(self) -> bool:
        """Return whether to return node types."""
        return self._return_node_types
    
    def return_edge_types(self) -> bool:
        """Return whether to return edge types."""
        return self._return_edge_types
    
    def get_graph(self) -> Graph:
        """Return graph."""
        return self._graph
    
    def get_support(self) -> Graph:
        """Return support graph."""
        return self._support
    

    def __getitem__(self, idx: int):
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be returned.

        Returns
        ---------------
        """
        return (tuple([
            value
            for value in self._support.get_edge_prediction_chunk_mini_batch(
                idx,
                graph=self._graph,
                batch_size=self._batch_size,
                return_node_types=self._return_node_types,
                return_edge_types=self._return_edge_types,
                return_edge_metrics=self.use_edge_metrics(),
            )
            if value is not None
        ]),)
