from typing import List, Tuple, Dict, Set
import numpy as np  # type: ignore
from numba.experimental import jitclass  # type: ignore
from numba import typed, types, njit, prange  # type: ignore
from .numba_graph import NumbaGraph
from .graph_types import (
    nodes_mapping_type,
    numpy_nodes_type,
    numba_vector_nodes_type
)


class DirectedGraph:

    def __init__(self, **kwargs):
        """Crate a new instance of a NumbaDirectedGraph with given edges.

        Parameters
        -------------------------
        nodes: List[str],
            List of node names.
        sources_names: List[int],
            List of the source nodes in edges of the graph.
        destinations_names: List[int],
            List of the destination nodes in edges of the graph.
        preprocess: bool = True,
            Wethever to preprocess this graph for random walk.

        Raises
        -------------------------
        ValueError,
            If given sources length does not match destinations length.
        """

        self._graph = NumbaGraph(**kwargs)

    @property
    def core(self) -> NumbaGraph:
        """Return graph engine."""
        return self._graph

    @property
    def preprocessed(self) -> bool:
        """Return boolean representing if the graph was preprocessed."""
        return self._graph.preprocessed

    @property
    def nodes_number(self) -> int:
        """Return integer with the length of the graph."""
        return self._graph.nodes_number

    @property
    def sources(self) -> List[int]:
        """Return list of source nodes of the graph."""
        return self._graph.sources

    @property
    def destinations(self) -> List[int]:
        """Return list of destinations nodes of the graph."""
        return self._graph.destinations

    @property
    def has_traps(self) -> bool:
        """Return boolean representing if graph has traps."""
        return self._graph.has_traps

    def is_node_trap(self, node: int) -> bool:
        """Return boolean representing if node is a dead end.

        Parameters
        ----------
        node: int,
            Node numeric ID.

        Returns
        -----------------
        Boolean True if node is a trap.
        """
        return self._graph.is_node_trap

    def extract_node_neighbour(self, src: int) -> Tuple[int, int]:
        """Return a random adjacent node to the one associated to node.
        The destination is extracted by using the normalized weights
        of the edges as probability distribution.

        Parameters
        ----------
        src: int
            The index of the source node that is to be considered.

        Returns
        -------
        A tuple containing the index of a random adiacent node to given
        source node and the ID of th edge used for the transition between
        the two nodes.
        """
        return self._graph.extract_node_neighbour(src)

    def extract_edge_neighbour(self, edge: int) -> Tuple[int, int]:
        """Return a random adiacent edge to the one associated to edge.
        The Random is extracted by using the normalized weights of the edges
        as probability distribution.

        Parameters
        ----------
        edge: int
            The index of the egde that is to be considered.

        Returns
        -------
        The index of a random adiacent edge to edge.
        """
        return self._graph.extract_edge_neighbour(edge)
