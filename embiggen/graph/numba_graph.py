from typing import List, Tuple, Set
import numpy as np  # type: ignore
from numba.experimental import jitclass  # type: ignore
from numba import typed, types  # type: ignore
from .alias_method import alias_draw
from random import choice
from .build_alias import build_alias_edges, build_alias_nodes
from .graph_types import (
    numba_vector_nodes_type,
    numba_edges_type,
    alias_list_type,
    edges_type_list
)


@jitclass([
    ('_destinations', numba_vector_nodes_type),
    ('_sources', numba_vector_nodes_type),
    ('_neighbors', types.ListType(edges_type_list)),
    ('_nodes_alias', types.ListType(alias_list_type)),
    ('_edges_alias', types.ListType(alias_list_type)),
    ('_has_traps', types.boolean),
    ('_uniform', types.boolean),
    ('_nodes_number', types.uint64)
])
class NumbaGraph:

    def __init__(
        self,
        nodes_number: int,
        sources: List[int],
        destinations: List[int],
        edges_set: Set[Tuple[int, int]],
        node_types: List[np.uint16] = None,
        edge_types: List[np.uint16] = None,
        weights: List[float] = None,
        uniform: bool = True,
        default_weight: float = 1.0,
        return_weight: float = 1.0,
        explore_weight: float = 1.0,
        change_node_type_weight: float = 1.0,
        change_edge_type_weight: float = 1.0,
    ):
        """Crate a new instance of a NumbaGraph with given edges.

        Parameters
        -------------------------
        nodes_number: int,
            Number of nodes in the graph.
        sources: List[int],
            List of the source nodes in edges of the graph.
        destinations: List[int],
            List of the destination nodes in edges of the graph.
        edges_set: Set[Tuple[int, int]],
            Set of unique edges in the graph.
        node_types: List[np.uint16] = None,
            The node types for each node.
            This is an optional parameter to make the graph behave as if it
            is colored within the walk.
        edge_types: List[np.uint16],
            The edge types for each source and sink.
            This is an optional parameter to make the graph behave as if it
            is a multigraph within the walk.
        weights: List[float] = None,
            The weights for each source and sink. By default None. If you want
            to specify a single value to be used for every weight, use the
            default_weight parameter.
        uniform: bool = True,
            Wethever if the weights for the nodes are close to be uniform.
        default_weight: int = 1.0,
            The default weight to use when no weight is provided.
        return_weight : float = 1.0,
            Weight on the probability of returning to node coming from
            Having this higher tends the walks to be
            more like a Breadth-First Search.
            Having this very high  (> 2) makes search very local.
            Equal to the inverse of p in the Node2Vec paper.
        explore_weight : float = 1.0,
            Weight on the probability of visiting a neighbor node
            to the one we're coming from in the random walk
            Having this higher tends the walks to be
            more like a Depth-First Search.
            Having this very high makes search more outward.
            Having this very low makes search very local.
            Equal to the inverse of q in the Node2Vec paper.
        change_node_type_weight: float = 1.0,
            Weight on the probability of visiting a neighbor node of a
            different type than the previous node. This only applies to
            colored graphs, otherwise it has no impact.
        change_edge_type_weight: float = 1.0,
            Weight on the probability of visiting a neighbor edge of a
            different type than the previous edge. This only applies to
            multigraphs, otherwise it has no impact.

        Raises
        -------------------------
        ValueError,
            If given edge types length does not match destinations length.
        ValueError,
            If given weights list length does not match destinations length
        ValueError,
            If given node types has not the same length of nodes numbers.
        ValueError,
            If return_weight is not a strictly positive real number.
        ValueError,
            If explore_weight is not a strictly positive real number.
        ValueError,      
            If change_node_type_weight is not a strictly positive real number.
        ValueError,
            If change_edge_type_weight is not a strictly positive real number.

        """
        if edge_types is not None and len(edge_types) != len(destinations):
            raise ValueError(
                "Given edge types length does not match destinations length."
            )
        if weights is not None and len(weights) != len(destinations):
            raise ValueError(
                "Given weights length does not match destinations length."
            )
        if node_types is not None and nodes_number != len(node_types):
            raise ValueError(
                "Given node types has not the same length of given nodes number."
            )
        if return_weight <= 0:
            raise ValueError("Given return weight is not a positive number")
        if explore_weight <= 0:
            raise ValueError("Given explore weight is not a positive number")
        if change_node_type_weight <= 0:
            raise ValueError(
                "Given change_node_type_weigh is not a positive number"
            )
        if change_edge_type_weight <= 0:
            raise ValueError(
                "Given change_edge_type_weight is not a positive number"
            )

        self._destinations = destinations
        self._sources = sources
        self._nodes_number = nodes_number
        self._uniform = uniform or weights is None

        # Each node has a list of neighbors.
        # These lists are initialized as empty.
        self._neighbors = typed.List.empty_list(edges_type_list)
        for _ in range(nodes_number):
            self._neighbors.append(
                typed.List.empty_list(numba_edges_type)
            )

        # The following proceedure ASSUMES that the edges only appear
        # in a single direction. This must be handled in the preprocessing
        # of the graph parsing proceedure.
        for i, (src, dst) in enumerate(zip(self._destinations, self._sources)):
            # Appending outbound edge ID to SRC list.
            self._neighbors[src].append(i)

        # Creating the node alias list, which contains tuples composed of
        # the list of indices of the opposite extraction events and the list
        # of probabilities for the extraction of edges neighbouring the nodes.
        if not self._uniform:
            self._nodes_alias = build_alias_nodes(self._neighbors, weights)

        # Creating the edges alias list, which contains tuples composed of
        # the list of indices of the opposite extraction events and the list
        # of probabilities for the extraction of edges neighbouring the edges.
        self._edges_alias = build_alias_edges(
            edges_set=edges_set,
            nodes_neighboring_edges=self._neighbors,
            node_types=node_types,
            edge_types=edge_types,
            weights=weights,
            default_weight=default_weight,
            sources=self._sources,
            destinations=self._destinations,
            return_weight=return_weight,
            explore_weight=explore_weight,
            change_node_type_weight=change_node_type_weight,
            change_edge_type_weight=change_edge_type_weight
        )

        # To verify if this graph has some walker traps, meaning some nodes
        # that do not have any neighbors, we have to iterate on the list of
        # neighbors and to check if at least a node has no neighbors.
        # If such a condition is met, we cannot anymore do the simple random
        # walk assuming that all the walks have the same length, but we need
        # to create a random walk with variable length, hence a list of lists.

        self._has_traps = False
        for src in range(nodes_number):
            if self.is_node_trap(src):
                self._has_traps = True
                break

    @property
    def nodes_number(self) -> int:
        """Return integer with the length of the graph."""
        return self._nodes_number

    @property
    def sources(self) -> List[int]:
        """Return list of source nodes of the graph."""
        return self._sources

    @property
    def destinations(self) -> List[int]:
        """Return list of destinations nodes of the graph."""
        return self._destinations

    @property
    def has_traps(self) -> bool:
        """Return boolean representing if graph has traps."""
        return self._has_traps

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
        return len(self._neighbors[node]) == 0

    def _extract_transition_informations(
        self,
        src: int,
        j: List[int],
        q: List[float]
    ) -> Tuple[int, int]:
        """Return tuple with new destination and used edge.

        Parameters
        ---------------------
        src: int,
            The previous source node.
        j: List[int],
            The indices to use for alias method.
        q: List[float]
            The probabilities to use for alias method.

        Returns
        ---------------------
        Tuple containing the new destination and the used edge.
        """
        # Get the new edge, extracted randomly using alias method draw.
        edge = self._neighbors[src][alias_draw(j, q)]
        # Get the destination of the chosen edge.
        dst = self._destinations[edge]
        # Return the obtained tuple.
        return dst, edge

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
        # If the graph is uniform, we do not need to use advanced proceedures
        # to get the next random weight and we can just use random choise to
        # accomplish that.
        if self._uniform:
            edge = choice(self._neighbors[src])
            dst = self._destinations[edge]
            return dst, edge
        # Get the information relative to the source node, composed of a tuple:
        # - The numpy array of opposite events for the alias method (j)
        # - The probabilities for the extractions for the alias method (q)
        j, q = self._nodes_alias[src]
        # Get the tuple of the new transition, composed of the new destination
        # and the edge used for the transition.
        return self._extract_transition_informations(src, j, q)

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
        # We retrieve the destination of edge currently used, which can be
        # considered the source of the edge we are looking for now.
        src = self._destinations[edge]
        # Get the information relative to the source node, composed of a tuple:
        # - The numpy array of opposite events for the alias method (j)
        # - The probabilities for the extractions for the alias method (q)
        j, q = self._edges_alias[edge]
        # Get the tuple of the new transition, composed of the new destination
        # and the edge used for the transition.
        return self._extract_transition_informations(src, j, q)
