from typing import Dict, List, Set, Tuple

import numpy as np  # type: ignore
from numba import njit, prange, typed, types, objmode  # type: ignore
from numba.experimental import jitclass  # type: ignore
from ..utils import numba_log

from .alias_method import alias_draw, alias_setup
from .graph_types import (
    alias_list_type, edges_type_list, nodes_mapping_type,
    numba_edges_type, numba_nodes_type,
    numba_vector_edges_type, numba_vector_nodes_type, numba_probs_type,
    numpy_edges_type, numpy_nodes_type,
    numba_vector_nodes_colors_type, numba_vector_edges_colors_type,
    numpy_nodes_colors_type, numpy_edges_colors_type,
    numpy_probs_type, edges_keys_tuple,
    numpy_indices_type
)


@jitclass([
    ('_destinations', numba_vector_nodes_type),
    ('_sources', numba_vector_nodes_type),
    ('_nodes_mapping', types.DictType(*nodes_mapping_type)),
    ('_reverse_nodes_mapping', types.ListType(types.string)),
    ('_unique_edges', types.Set(edges_keys_tuple)),
    ('_outbound_edges', numba_vector_edges_type),
    ('_nodes_alias', types.ListType(alias_list_type)),
    ('_edges_alias', types.ListType(alias_list_type)),
    ('_uniform', types.boolean),
    ('_directed', types.boolean),
    ('_preprocessed', types.boolean),
    ('_nodes_number', types.uint64),
    ('_weights', numba_probs_type),
    ('_node_types', numba_vector_nodes_colors_type),
    ('_edge_types', numba_vector_edges_colors_type),
    ('_return_weight', types.float64),
    ('_explore_weight', types.float64),
    ('_change_node_type_weight', types.float64),
    ('_change_edge_type_weight', types.float64),
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
        if edge_types.size and edge_types.size != destinations_names.size:
            raise ValueError(
                "Given edge types length does not match destinations length."
            )
        if weights.size and weights.size != destinations_names.size:
            raise ValueError(
                "Given weights length does not match destinations length."
            )
        if node_types.size and nodes.size != node_types.size:
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

        self._preprocessed = False
        self._directed = directed
        self._weights = weights
        self._node_types = node_types
        self._edge_types = edge_types
        self._nodes_number = len(nodes)
        self._return_weight = return_weight
        self._explore_weight = explore_weight
        self._change_node_type_weight = change_node_type_weight
        self._change_edge_type_weight = change_edge_type_weight

        if not self.directed:
            sources_names, destinations_names = self._process_undirected_graph(
                sources_names,
                destinations_names
            )

        self._process_nodes_mapping(nodes)

        numba_log("Processing edges mapping")
        # Transform the lists of names into IDs
        self._sources, self._destinations, self._unique_edges = process_edges(
            sources_names, destinations_names, self._nodes_mapping
        )

        # Sorting given edges using sources as index.
        numba_log("Sorting edges values")
        self._sort_edge_values()

        self._uniform = self._process_is_uniform(uniform)

        numba_log("Processing outbound edges")
        self._outbound_edges = self.compute_outbound_edges()

        numba_log("Completed construction of graph object.")

    def build_graph_alias(self):
        """Create objects necessary for quick random search over graph.

        Note that these objects can get VERY big, for example in a graph with
        15 million edges they get up to around 80GBs.

        Consider using the lazy random walk that renders the probabilities as
        the the walk proceeds as an alternative solution when the graph gets
        too big for this quick walk.

        After the walk is executed, to keep the graph object but destroy these
        big objects, call the method graph.destroy_graph_alias().
        """
        # Creating the node alias list, which contains tuples composed of
        # the list of indices of the opposite extraction events and the list
        # of probabilities for the extraction of edges neighbouring the nodes.
        if not self._uniform:
            numba_log("Processing nodes alias.")
            self._nodes_alias = build_alias_nodes(self)
        else:
            numba_log("Skipping nodes alias building since graph is uniform.")

        # Creating the edges alias list, which contains tuples composed of
        # the list of indices of the opposite extraction events and the list
        # of probabilities for the extraction of edges neighbouring the edges.
        numba_log("Processing edges alias")
        self._edges_alias = build_alias_edges(self)
        numba_log("Completed graph preprocessing for random walks.")

        # Marking current graph as preprocessed
        self._preprocessed = True

    def destroy_graph_alias(self):
        """Destroys object related to graph alias."""
        self._preprocessed = False
        self._nodes_alias = typed.List.empty_list(alias_list_type)
        self._edges_alias = typed.List.empty_list(alias_list_type)

    def _sort_edge_values(self):
        """Sort the values so that we can compute the neighbours faster."""
        sorted_indices = np.argsort(self._sources)
        self._sources = self._sources[sorted_indices]
        self._destinations = self._destinations[sorted_indices]
        # If edge types are provided we need to sort them.
        if self._edge_types.size:
            self._edge_types = self._edge_types[sorted_indices]
        # If weights are provided we need to sort them.
        if self._weights.size:
            self._weights = self._weights[sorted_indices]

    def _process_is_uniform(self, uniform: bool) -> bool:
        """Compute if we can process the graph as an uniform or not.
        An uniform graph is a graph where we can ignore the weights.ƒ
        """
        return (
            # if the user told us to.
            uniform or
            # or he didn't give us weights AND
            self._weights.size == 0 and (
                # there are no node-types or we don't have to change the weights
                # based on the node-types
                self._node_types.size == 0 or (
                    1.0 - self._change_node_type_weight) < 1e8
            )
        )

    def _process_nodes_mapping(self, nodes: List[str]):
        """Create the outbound and backward mapping from each node to it's ID"""
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
            self._nodes_mapping[str(node)] = numpy_nodes_type(i)
            self._reverse_nodes_mapping.append(str(node))

    def _process_undirected_graph(self,
                                  sources_names: List[str],
                                  destinations_names: List[str]
                                  ) -> Tuple[List[str], List[str]]:
        """Parse the inputs of the class in an optimal way for undirected graphs

        Parameters
        ----------
        sources_names: List[str],
            List of the source nodes in edges of the graph.
        destinations_names: List[str],
            List of the destination nodes in edges of the graph.

        Returns
        -------
        The two updated lists of sources_names and destinations_names.
        """
        numba_log("Building undirected graph.")
        # Cunting self-loops
        numba_log("Counting self-loops.")
        loops_mask = np.empty(len(sources_names), dtype=np.bool_)
        for i, (src_name, dst_name) in enumerate(zip(sources_names, destinations_names)):
            loops_mask[i] = str(src_name) == str(dst_name)

        total_loops = loops_mask.sum()
        total_orig_edges = len(sources_names)
        total_edges = (total_orig_edges-total_loops)*2 + total_loops

        numba_log("Building undirected graph sources.")
        full_sources = np.empty(total_edges, dtype=sources_names.dtype)
        full_sources[:total_orig_edges] = sources_names
        full_sources[total_orig_edges:] = destinations_names[~loops_mask]

        numba_log("Building undirected graph destinations.")
        full_destinations = np.empty(
            total_edges, dtype=destinations_names.dtype)
        full_destinations[:total_orig_edges] = destinations_names
        full_destinations[total_orig_edges:] = sources_names[~loops_mask]

        if self._edge_types.size:
            numba_log("Building undirected graph edge types.")
            full_edge_types = np.empty(
                total_edges, dtype=numpy_edges_colors_type)
            full_edge_types[:total_orig_edges] = self._edge_types
            full_edge_types[total_orig_edges:] = self._edge_types[~loops_mask]
            self._edge_types = full_edge_types

        if self._weights.size:
            numba_log("Building undirected graph weights.")
            full_weights = np.empty(total_edges, dtype=np.float64)
            full_weights[:total_orig_edges] = self._weights
            full_weights[total_orig_edges:] = self._weights[~loops_mask]
            self._weights = full_weights

        return full_sources, full_destinations

    def compute_outbound_edges(self) -> List[int]:
        """Return offset sets of outbound outbound_edges."""
        # IF THE EDGES ARE SORTED, we can use the destinations array to get the
        # foward edges of a node.
        #
        # The idea is to store the number of outbound edges for each NODE.
        # E.G.    _
        #   s1 d1  |
        #   s1 d2  | -> 3
        #   s1 d3 _|
        #   s4 d1  | -> 2
        #   s4 d3 _|
        #   s6 d2  | -> 2
        #   s6 d3 _|
        # for optimization we can sum the number of outbound edges
        # so it becomes an offset.
        # Then, we can compute it using the two array:
        # [d1, d2, d3, d1, d3, d2, d3]
        # [3, 3, 3, 5, 5, 7, 7]
        # now to get the outbound edges of s2 we get its index, s2 -> 1
        # then we access the index of s2 and the previous index.
        # Finally the outbound_edges are d[3:5]
        last_src = 0
        # preallocate the outbound_edges to avoid multiple reallocation
        outbound_edges = np.empty(self.nodes_number, dtype=numpy_edges_type)
        # we iterate on the sources because they have foward edges
        for i, src in enumerate(self._sources):
            if last_src != src:
                # Assigning to range instead of single value, so that traps
                # have as delta between previous and next node zero.
                outbound_edges[last_src:src] = i
                last_src = src
        # Fix the last nodes foward edges by propagating the last_count because
        # if we haven't already filled the array,
        # all the remaining nodes are traps
        outbound_edges[src:] = i + 1
        return outbound_edges

    @property
    def preprocessed(self) -> bool:
        """Return boolean representing if the graph was preprocessed."""
        return self._preprocessed

    @property
    def directed(self) -> bool:
        """Return boolean representing if the graph is directed."""
        return self._directed

    @property
    def uniform(self) -> bool:
        """Return boolean representing if the graph has uniform nodes weights."""
        return self._uniform

    @property
    def nodes_number(self) -> int:
        """Return integer with the number of nodes of the graph."""
        return self._nodes_number

    @property
    def edges_number(self) -> int:
        """Return integer with the number of edges of the graph."""
        return self._sources.size

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
        for node in range(self.nodes_number):
            if self.is_node_trap(node):
                return True
        return False

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
        _min, _max = self._get_min_max_edge(node)
        return _min == _max

    def is_edge_trap(self, edge: int) -> bool:
        """Return boolean representing if edge is a dead end.

        Parameters
        ----------
        edge: int,
            Edge numeric ID.

        Returns
        -----------------
        Boolean True if edge is a trap.
        """
        return self.is_node_trap(self._destinations[edge])

    def _get_min_max_edge(self, node: int) -> Tuple[int, int]:
        """Return tuple with minimum and maximum edge for given node.
        This method is used to retrieve the indices needed to extract the 
        neighbors from the destination array.

        Parameters
        ---------------------------
        node: int,
            The id of the node to access.

        Returns
        ----------------------------
        Tuple with minimum and maximum edge index for given node.
        """
        min_edge = 0 if node == 0 else self._outbound_edges[node-1]
        max_edge = self._outbound_edges[node]
        return min_edge, max_edge

    def get_node_transition_weights(self, node: int) -> Tuple[List[float], List[int]]:
        """Return transition weights vector for given node.

        NB: the returned weights are NOT normalized.

        Parameters
        ----------------------------
        node: int,
            Node for which to compute the transition weights.

        Returns
        -----------------------------
        Tuple with transition weights and transition destinations.
        """
        # Retrieve edge boundaries.
        min_edge, max_edge = self._get_min_max_edge(node)
        # If weights are given
        if self._weights.size:
            # We retrieve the weights relative to these transitions
            transition_weights = self._weights[min_edge:max_edge]
        else:
            # Otherwise we start wi AND
            transition_weights = np.ones(
                max_edge-min_edge,
                dtype=numpy_probs_type
            )

        destinations = self._destinations[min_edge:max_edge]

        ############################################################
        # Handling of the change node type parameter               #
        ############################################################
        # If the node types were given:
        if self._node_types.size:
            # if the destination node type matches the neighbour
            # destination node type (we are not changing the node type)
            # we weigth using the provided change_node_type_weight weight.
            mask = self._node_types[node] == self._node_types[destinations]
            transition_weights[mask] /= self._change_node_type_weight

        return transition_weights, destinations

    def get_edge_transition_weights(self, edge: int) -> List[float]:
        """Return transition weights vector for given edge.

        NB: the returned weights are NOT normalized.

        Parameters
        ----------------------------
        edge: int,
            Edge for which to compute the transition weights.

        Returns
        -----------------------------
        Vector of the transition weights
        """
        # Get the source and destination for current edge.
        src, dst = self._sources[edge], self._destinations[edge]

        # Compute the transition weights relative to the node weights.
        transition_weights, destinations = self.get_node_transition_weights(
            dst)
        min_edge, max_edge = self._get_min_max_edge(dst)

        ############################################################
        # Handling of the change edge type parameter               #
        ############################################################

        if self._edge_types.size:
            # Similarly if the neighbour edge type matches the previous
            # edge type (we are not changing the edge type)
            # we weigth using the provided change_edge_type_weight weight.
            mask = self._edge_types[edge] == self._edge_types[min_edge:max_edge]
            transition_weights[mask] /= self._change_edge_type_weight

        ############################################################
        # Handling of the Q parameter: the return coefficient      #
        ############################################################

        # If the neigbour matches with the source, hence this is
        # a backward loop like the following:
        # SRC -> DST
        #  ▲     /
        #   \___/
        #
        # We weight the edge weight with the given return weight.
        is_looping_back = destinations == src
        transition_weights[is_looping_back] *= self._return_weight

        ############################################################
        # Handling of the P parameter: the exploration coefficient #
        ############################################################
        
        for i, ndst in enumerate(destinations):
            # If there is no branch from the destination to the source node
            # it means that the destination can lead to more exploration
            if (ndst, src) not in self._unique_edges:
                # Hence we apply the explore weight
                transition_weights[i] *= self._explore_weight

        return transition_weights

    def _extract_transition(self, src: int, j: List[int], q: List[float]) -> Tuple[int, int]:
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
        previous_edge, _ = self._get_min_max_edge(src)
        edge = previous_edge + alias_draw(j, q)
        # Get the destination of the chosen edge.
        dst = self._destinations[edge]
        # Return the obtained tuple.
        return dst, edge

    def extract_node_neighbour(self, src: int) -> Tuple[int, int]:
        """Return a random adjacent node to the one associated to node.
        The destination is extracted by using the normalized weights
        of the edges as probability distribution.

        For the uniform case, we can just extract a random neighbour. 
        (This is actually not equivalent because we "ignore" that
        different nodetypes can change the probability but it's an approximation
        that allows us to don't build and save the node_outbound_edges struct which
        is requires a cospicuos ammount of memory.)

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
        if self.uniform:
            edge = np.random.randint(*self._get_min_max_edge(src))
            dst = self._destinations[edge]
            return dst, edge
        # Get the information relative to the source node, composed of a tuple:
        # - The numpy array of opposite events for the alias method (j)
        # - The probabilities for the extractions for the alias method (q)
        j, q = self._nodes_alias[src]
        # Get the tuple of the new transition, composed of the new destination
        # and the edge used for the transition.
        return self._extract_transition(src, j, q)

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
        return self._extract_transition(src, j, q)


@njit(parallel=True)
def process_edges(
    sources_names: List[str],
    destinations_names: List[str],
    mapping: Dict[str, int]
) -> Tuple[List[int], List[int]]:
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
    """
    edges_number = len(sources_names)
    sources = np.empty(edges_number, dtype=numpy_nodes_type)
    destinations = np.empty(edges_number, dtype=numpy_nodes_type)

    for i in prange(edges_number):  # pylint: disable=not-an-iterable
        # Store the sources into the sources vector
        sources[i] = mapping[str(sources_names[i])]
        # Store the destinations into the destinations vector
        destinations[i] = mapping[str(destinations_names[i])]

    unique_edges = set()

    for edge in zip(sources, destinations):
        unique_edges.add(edge)
    return sources, destinations, unique_edges


@njit
def build_default_alias_vectors(
    number: int
) -> Tuple[List[int], List[float]]:
    """Return empty alias vectors to be populated in
        build_alias_nodes below

    Parameters
    -----------------------
    number: int,
        Number of aliases to setup for.

    Returns
    -----------------------
    Returns default alias vectors.
    """
    alias = typed.List.empty_list(alias_list_type)
    empty_j = np.empty(0, dtype=numpy_indices_type)
    empty_q = np.empty(0, dtype=numpy_probs_type)
    for _ in range(number):
        alias.append((empty_j, empty_q))
    return alias


@njit(parallel=True)
def build_alias_nodes(graph: NumbaGraph) -> List[Tuple[List[int], List[float]]]:
    """Return aliases for nodes to use for alias method for 
       selecting from discrete distribution.

    Parameters
    -----------------------
    graph: NumbaGraph,
        Graph for which to compute the alias nodes.

    Returns
    -----------------------
    Lists of tuples representing node aliases 
    """

    alias = build_default_alias_vectors(graph.nodes_number)

    for i in prange(graph.nodes_number):  # pylint: disable=not-an-iterable
        node = np.int64(i)

        # Do not call the alias setup if the node is a trap.
        # Because that node will have no neighbors and thus the necessity
        # of setupping the alias method to efficently extract the neighbour.
        if graph.is_node_trap(node):
            continue

        # Compute the weights for the transition.
        weights, _ = graph.get_node_transition_weights(node)
        # Compute the alias method setup for given transition
        alias[node] = alias_setup(weights/weights.sum())
    return alias


@njit(parallel=True)
def build_alias_edges(graph: NumbaGraph) -> List[Tuple[List[int], List[float]]]:
    """Return aliases for edges to use for alias method for 
       selecting from discrete distribution.

    Parameters
    -----------------------
    graph: NumbaGraph,
        Graph for which to compute the alias edges.

    Returns
    -----------------------
    Lists of tuples representing edges aliases.
    """

    alias = build_default_alias_vectors(graph.edges_number)

    for i in prange(graph.edges_number):  # pylint: disable=not-an-iterable
        edge = np.int64(i)

        # Do not call the alias setup if the edge is a traps.
        # Because that edge will have no neighbors and thus the necessity
        # of setupping the alias method to efficently extract the neighbour.
        if graph.is_edge_trap(edge):
            continue

        weights = graph.get_edge_transition_weights(edge)

        # Finally we assign the obtained alias method probabilities.
        alias[edge] = alias_setup(weights/weights.sum())
    return alias
