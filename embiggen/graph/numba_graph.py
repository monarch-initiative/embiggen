from typing import Dict, List, Set, Tuple

import numpy as np  # type: ignore
from numba import njit, prange, typed, types, objmode  # type: ignore
from numba.experimental import jitclass  # type: ignore
from ..utils import numba_log

from .alias_method import alias_draw
from .build_alias import build_alias_edges, build_alias_nodes
from .graph_types import (
    alias_list_type, edges_type_list, nodes_mapping_type,
    numba_edges_type, numba_nodes_type,
    numba_vector_edges_type, numba_vector_nodes_type,
    numpy_edges_type, numpy_nodes_type,
    numpy_nodes_colors_type, numpy_edges_colors_type
)


@njit(parallel=True)
def process_edges(
    sources_names: List[str],
    destinations_names: List[str],
    mapping: Dict[str, int]
) -> Tuple[List[int], List[int], Set[Tuple[int, int]]]:
    """Return triple with mapped sources, destinations and edges set.

    Parameters
    --------------------------
    sources_names: List[str],
        List of the sources names.
    destinations_names: List[str],
        List of the destinations names.
    mapping: Dict[str, int],
        Dictionary mapping of the nodes.

    Returns
    --------------------------
    Triple with:
        - The mapped sources using given mapping, as a list of integers.
        - The mapped destinations using given mapping, as a list of integers.
        - The mapped edge sets using given mapping, as a set of tuples.
    """
    edges_number = len(sources_names)
    sources = np.empty(edges_number, dtype=numpy_nodes_type)
    destinations = np.empty(edges_number, dtype=numpy_nodes_type)
    edges_set = set()

    for i in prange(edges_number):  # pylint: disable=not-an-iterable
        # Store the sources into the sources vector
        sources[i] = mapping[str(sources_names[i])]
        # Store the destinations into the destinations vector
        destinations[i] = mapping[str(destinations_names[i])]

    # Creating edge set.
    for edge in zip(sources, destinations):
        # Adding the edge to the set
        edges_set.add(edge)

    return sources, destinations, edges_set


@njit(parallel=True)
def process_traps(neighbors: List[List[int]]) -> List[bool]:
    traps = np.empty(len(neighbors), dtype=np.bool_)
    for src in prange(len(neighbors)):  # pylint: disable=not-an-iterable
        traps[src] = neighbors[np.int64(src)].size == 0
    return traps


@jitclass([
    ('_destinations', numba_vector_nodes_type),
    ('_sources', numba_vector_nodes_type),
    ('_nodes_mapping', types.DictType(*nodes_mapping_type)),
    ('_reverse_nodes_mapping', types.ListType(types.string)),
    ('_neighbors', types.ListType(numba_vector_edges_type)),
    ('_nodes_alias', types.ListType(alias_list_type)),
    ('_edges_alias', types.ListType(alias_list_type)),
    ('_traps', types.boolean[:]),
    ('_has_traps', types.boolean),
    ('_uniform', types.boolean),
    ('_preprocessed', types.boolean),
    ('_nodes_number', types.uint64)
])
class NumbaGraph:

    def __init__(
        self,
        nodes: List[str],
        sources_names: List[str],
        destinations_names: List[str],
        node_types: List[np.uint16] = None,
        edge_types: List[np.uint16] = None,
        weights: List[float] = None,
        uniform: bool = True,
        directed: bool = True,
        preprocess: bool = True,
        default_weight: float = 1.0,
        return_weight: float = 1.0,
        explore_weight: float = 1.0,
        change_node_type_weight: float = 1.0,
        change_edge_type_weight: float = 1.0,
    ):
        """Crate a new instance of a NumbaGraph with given edges.

        Parameters
        -------------------------
        nodes_number: List[str],
            Number of nodes in the graph.
        sources_names: List[str],
            List of the source nodes in edges of the graph.
        destinations_names: List[str],
            List of the destination nodes in edges of the graph.
        edges_set: Set[Tuple[int, int]],
            Set of unique edges in the graph.
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
        if len(edge_types) > 0 and len(edge_types) != len(destinations_names):
            raise ValueError(
                "Given edge types length does not match destinations length."
            )
        if len(weights) > 0 and len(weights) != len(destinations_names):
            raise ValueError(
                "Given weights length does not match destinations length."
            )
        if len(node_types) > 0 and len(nodes) != len(node_types):
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

        if len(sources_names) != len(destinations_names):
            raise ValueError(
                "Given sources length does not match destinations length."
            )

        if not directed:
            numba_log("Building undirected graph.")
            # Cunting self-loops
            numba_log("Counting self-loops.")
            loops_mask = np.zeros(len(sources_names), dtype=np.bool_)
            for i, (src, dst) in enumerate(zip(sources_names, destinations_names)):
                loops_mask[i] = str(src) == str(dst)

            total_loops = loops_mask.sum()
            total_orig_edges = len(sources_names)
            total_edges = (total_orig_edges-total_loops)*2 + total_loops

            numba_log("Building undirected graph sources.")
            full_sources = np.empty(total_edges, dtype=sources_names.dtype)
            full_sources[:total_orig_edges] = sources_names
            full_sources[total_orig_edges:] = sources_names[~loops_mask]
            sources_names = full_sources

            numba_log("Building undirected graph destinations.")
            full_destinations = np.empty(
                total_edges, dtype=destinations_names.dtype)
            full_destinations[:total_orig_edges] = destinations_names
            full_destinations[total_orig_edges:] = destinations_names[~loops_mask]
            destinations_names = full_destinations

            if len(node_types) > 0:
                numba_log("Building undirected graph node types.")
                full_node_types = np.empty(
                    total_edges, dtype=numpy_nodes_colors_type)
                full_node_types[:total_orig_edges] = node_types
                full_node_types[total_orig_edges:] = node_types[~loops_mask]
                node_types = full_node_types

            if len(edge_types) > 0:
                numba_log("Building undirected graph edge types.")
                full_edge_types = np.empty(
                    total_edges, dtype=numpy_edges_colors_type)
                full_edge_types[:total_orig_edges] = edge_types
                full_edge_types[total_orig_edges:] = edge_types[~loops_mask]
                edge_types = full_edge_types

            if len(weights) > 0:
                numba_log("Building undirected graph weights.")
                full_weights = np.empty(total_edges, dtype=np.float64)
                full_weights[:total_orig_edges] = weights
                full_weights[total_orig_edges:] = weights[~loops_mask]
                weights = full_weights

        # Creating mapping and reverse mapping of nodes and integer ID.
        # The map looks like the following:
        # {
        #   "node_1_id": 0,
        #   "node_2_id": 1
        # }
        #
        # The reverse mapping is just a list of the nodes.
        #
        numba_log("Processing nodes mapping")
        self._nodes_mapping = typed.Dict.empty(*nodes_mapping_type)
        self._reverse_nodes_mapping = typed.List.empty_list(types.string)
        for i, node in enumerate(nodes):
            self._nodes_mapping[str(node)] = np.uint32(i)
            self._reverse_nodes_mapping.append(str(node))

        numba_log("Processing edges mapping")
        # Transform the lists of names into IDs
        self._sources, self._destinations, edges_set = process_edges(
            sources_names, destinations_names, self._nodes_mapping
        )

        self._preprocessed = preprocess

        if not preprocess:
            numba_log(
                "No further preprocessing for random walk has been required. "
                "Stopping graph preprocessing before alias building."
            )
            return

        self._nodes_number = len(nodes)
        self._uniform = uniform or len(weights) == 0

        numba_log("Processing neighbours")

        # Each node has a list of neighbors.
        # These lists are initialized as empty.
        neighbors = typed.List.empty_list(edges_type_list)
        for _ in range(self.nodes_number):
            neighbors.append(
                typed.List.empty_list(numba_edges_type)
            )

        # The following proceedure ASSUMES that the edges only appear
        # in a single direction. This must be handled in the preprocessing
        # of the graph parsing proceedure.
        for i, src in enumerate(self._sources):
            # Appending outbound edge ID to SRC list.
            neighbors[src].append(i)

        self._neighbors = typed.List.empty_list(numba_vector_edges_type)
        for src in range(self.nodes_number):
            src = np.int64(src)
            neighs = neighbors[src]
            self._neighbors.append(
                np.empty(len(neighs), dtype=numpy_edges_type))
            for i, neigh in enumerate(neighs):
                self._neighbors[src][i] = neigh

        # Creating the node alias list, which contains tuples composed of
        # the list of indices of the opposite extraction events and the list
        # of probabilities for the extraction of edges neighbouring the nodes.
        if not self._uniform:
            numba_log("Processing nodes alias.")

            self._nodes_alias = build_alias_nodes(
                self._neighbors, weights
            )

        numba_log("Processing edges alias")

        # Creating the edges alias list, which contains tuples composed of
        # the list of indices of the opposite extraction events and the list
        # of probabilities for the extraction of edges neighbouring the edges.
        self._edges_alias = build_alias_edges(
            edges_set=edges_set,
            nodes_neighboring_edges=self._neighbors,
            node_types=node_types,
            edge_types=edge_types,
            weights=weights,
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

        numba_log("Searching for traps in the graph.")
        self._traps = process_traps(self._neighbors)
        self._has_traps = self._traps.any()
        numba_log("Completed graph preprocessing for random walks.")

    @property
    def preprocessed(self) -> int:
        """Return integer with the length of the graph."""
        return self._preprocessed

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
        return self._traps[node]

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
            edge = np.random.choice(self._neighbors[src])
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
