from typing import List, Tuple
import numpy as np  # type: ignore
from numba.experimental import jitclass  # type: ignore
from numba import typed, types, njit, prange  # type: ignore
from .alias_method import alias_draw, alias_setup
from IPython import embed

numba_nodes_type = types.uint32
numpy_nodes_type = np.uint32
numba_edges_type = types.int64
numpy_edges_type = np.int64

# key and value types
edges_type_list = types.ListType(numba_edges_type)
float_list = types.ListType(types.float64)
edges_type = (types.UniTuple(numba_nodes_type, 2), numba_edges_type)
nodes_type = (types.string, numba_nodes_type)
alias_method_list_type = types.Tuple((types.int16[:], types.float64[:]))


@njit(parallel=True)
def build_alias_nodes(nodes, weights, nodes_mapping, nodes_neighboring_edges):
    # TODO! WRITE A DOCSTRING HERE!
    empty_j = np.empty(0, dtype=np.int16)
    empty_q = np.empty(0, dtype=np.float64)
    nodes_alias = typed.List.empty_list(alias_method_list_type)

    for _ in nodes:
        nodes_alias.append((empty_j, empty_q))

    for i in prange(len(nodes)):  # pylint: disable=not-an-iterable
        k = np.int64(i)
        src = nodes_mapping[str(nodes[k])]
        neighboring_edges = nodes_neighboring_edges[src]
        neighboring_edges_number = len(neighboring_edges)

        # Do not call the alias setup if the node is a trap.
        # Because that node will have no neighbors and thus the necessity
        # of setupping the alias method to efficently extract the neighbour.
        if neighboring_edges_number == 0:
            #j, q = np.zeros(0, dtype=np.int16), np.zeros(0, dtype=np.float64)
            continue

        probs = np.zeros(neighboring_edges_number, dtype=np.float64)
        for j, neighboring_edge in enumerate(neighboring_edges):
            probs[j] = weights[neighboring_edge]

        nodes_alias[k] = alias_setup(probs/probs.sum())
    return nodes_alias


@njit(parallel=True)
def build_alias_edges(
    edges_set,
    nodes_neighboring_edges,
    node_types,
    edge_types,
    weights,
    sources,
    destinations,
    return_weight,
    explore_weight,
    change_node_type_weight,
    change_edge_type_weight,
):
    # TODO! WRITE A DOCSTRING HERE!
    empty_j = np.empty(0, dtype=np.int16)
    empty_q = np.empty(0, dtype=np.float64)
    edges_alias = typed.List.empty_list(alias_method_list_type)

    for _ in range(len(sources)):
        edges_alias.append((empty_j, empty_q))

    for i in prange(len(sources)): # pylint: disable=not-an-iterable
        k = np.int64(i)
        src = sources[k]
        dst = destinations[k]
        neighboring_edges = nodes_neighboring_edges[dst]
        neighboring_edges_number = len(neighboring_edges)

        # Do not call the alias setup if the edge is a trap.
        # Because that edge will have no neighbors and thus the necessity
        # of setupping the alias method to efficently extract the neighbour.
        if neighboring_edges_number == 0:
            continue

        probs = np.zeros(neighboring_edges_number, dtype=np.float64)
        destination_type = node_types[dst]
        edge_type = edge_types[k]

        for index, neighboring_edge in enumerate(neighboring_edges):
            # We get the weight for the edge from the destination to
            # the neighbour.
            weight = weights[neighboring_edge]
            # Then we retrieve the neigh_dst node type.
            # And if the destination node type matches the neighbour
            # destination node type (we are not changing the node type)
            # we weigth using the provided change_node_type_weight weight.
            neighbor_destination = destinations[neighboring_edge]
            neighbor_type = node_types[neighbor_destination]
            if destination_type == neighbor_type:
                weight /= change_node_type_weight
            # Similarly if the neighbour edge type matches the previous
            # edge type (we are not changing the edge type)
            # we weigth using the provided change_edge_type_weight weight.
            if edge_type == edge_types[neighboring_edge]:
                weight /= change_edge_type_weight
            # If the neigbour matches with the source, hence this is
            # a backward loop like the following:
            # SRC -> DST
            #  ▲     /
            #   \___/
            #
            # We weight the edge weight with the given return weight.
            if neighbor_destination == src:
                weight = weight * return_weight
            # If the backward loop does not exist, we multiply the weight
            # of the edge by the weight for moving forward and explore more.
            elif (neighbor_destination, src) not in edges_set:
                weight = weight * explore_weight
            # Then we store these results into the probability vector.
            probs[index] = weight
        edges_alias[k] = alias_setup(probs/probs.sum())
    return edges_alias


@jitclass([
    ('_destinations', types.int64[:]),
    ('_sources', types.int64[:]),
    ('_nodes_mapping', types.DictType(*nodes_type)),
    ('_reverse_nodes_mapping', types.ListType(types.string)),
    ('_nodes_neighboring_edges', types.ListType(edges_type_list)),
    ('_nodes_alias', types.ListType(alias_method_list_type)),
    ('_edges_alias', types.ListType(alias_method_list_type)),
    ('has_traps', types.boolean),
    ('random_walk_preprocessing', types.boolean),
])
class NumbaGraph:

    def __init__(
        self,
        edges: np.ndarray,  # Array of strings
        weights: np.ndarray,  # Array of floats, same as the weights type
        nodes: np.ndarray,  # Array of strings
        node_types: np.ndarray,  # Array of integers, int16
        edge_types: np.ndarray,  # Array of integers, int16
        return_weight: float = 1,
        explore_weight: float = 1,
        change_node_type_weight: float = 1,
        change_edge_type_weight: float = 1,
        random_walk_preprocessing: bool = True
    ):
        """Crate a new instance of a undirected graph with given edges.

        Parameters
        ---------------------
        # TODO! UPDATE DOCSTRING!!!
        edges: np.ndarray,
            List of edges of the graph.
        nodes: np.ndarray,
            List of the nodes of the graph.
        weights: np.ndarray,
            The weights for each source and sink.
        node_types: np.ndarray,
            The node types for each node.
        node_types: np.ndarray,
            The edge types for each source and sink.
        return_weight : float = 1,
            Weight on the probability of returning to node coming from
            Having this higher tends the walks to be
            more like a Breadth-First Search.
            Having this very high  (> 2) makes search very local.
            Equal to the inverse of p in the Node2Vec paper.
        explore_weight : float = 1,
            Weight on the probability of visiting a neighbor node
            to the one we're coming from in the random walk
            Having this higher tends the walks to be
            more like a Depth-First Search.
            Having this very high makes search more outward.
            Having this very low makes search very local.
            Equal to the inverse of q in the Node2Vec paper.
        change_node_type_weight: float = 1,
            Weight on the probability of visiting a neighbor node of a
            different type than the previous node. This only applies to
            colored graphs, otherwise it has no impact.
        change_edge_type_weight: float = 1,
            Weight on the probability of visiting a neighbor edge of a
            different type than the previous edge. This only applies to
            multigraphs, otherwise it has no impact.
        random_walk_preprocessing: bool = True,
            Wethever to encode the graph to run afterwards a random walk.

        Returns
        ---------------------
        New instance of graph.
        """

        # Algorithm for preprocessing the graph:
        #
        #   1 - Computing nodes neighbouring edges.
        #
        #       A - Allocate the list of empty lists with len = len(nodes)
        #       B - Iterate sequentially over the enumerate(sources)
        #       C - Given an edge (i, src) -> n2e_neighbours[src].append(i)
        #
        #   2 - Compute edges j (int16) and q (float64) vectors for alias method.
        #
        #   3 - Compute destinations.
        #
        # Create the mapping of nodes and edges to indices
        # Compute the neighbours of each index
        #   - in such a way that if the index is >= k it's a trap
        #       - This can be done only after having built the first neighbours vector.
        #       - The i-th node index can be computed as len(nodes) - i
        #
        #   Random Walk:
        #       Given an edge e, we access to the alias list to extract j, q
        #           index = alias_draw(*alias[e])
        #       Then we get the destination node with destinations[e]
        #           then we get the neighbours of the dest node with n2e_neihbours[destionation[e]]
        #           and finally we can exract the indexth node

        #           new_edge = n2e_neighbours[destinations[e]][alias_draw(*alias[e])]

        self.random_walk_preprocessing = random_walk_preprocessing

        # Creating mapping of nodes and integer ID.
        # The map looks like the following:
        # {
        #   "node_1_id": 0,
        #   "node_2_id": 1
        # }
        self._nodes_mapping = typed.Dict.empty(*nodes_type)
        self._reverse_nodes_mapping = typed.List.empty_list(types.string)
        for i, node in enumerate(nodes):
            self._nodes_mapping[str(node)] = i
            self._reverse_nodes_mapping.append(str(node))

        if self.random_walk_preprocessing:
            # Each node has a list of neighbors.
            # These lists are initialized as empty.
            self._nodes_neighboring_edges = typed.List.empty_list(
                edges_type_list)
            for _ in range(len(nodes)):
                self._nodes_neighboring_edges.append(
                    typed.List.empty_list(numba_edges_type)
                )

        # Allocating the vectors of the mappings
        edges_set = set()
        self._destinations = np.empty(len(edges), dtype=numpy_nodes_type)
        self._sources = np.empty(len(edges), dtype=numpy_nodes_type)

        # The following proceedure ASSUMES that the edges only appear
        # in a single direction. This must be handled in the preprocessing
        # of the graph parsing proceedure.
        for i, (source, destination) in enumerate(edges):
            # Create the sources numeric ID
            src = self._nodes_mapping[str(source)]
            # Create the destinations numeric ID
            dst = self._nodes_mapping[str(destination)]
            # Store the destinations into the destinations vector
            self._destinations[i] = dst
            # Store the sources into the sources vector
            self._sources[i] = src
            # If the preprocessing is required we compute the neighbours
            if self.random_walk_preprocessing:
                # Appending outbound edge ID to SRC list.
                self._nodes_neighboring_edges[src].append(i)
                # Storing the edges mapping.
                edges_set.add((src, dst))

        if not self.random_walk_preprocessing:
            return

        # Creating struct saving all the data relative to the nodes.
        # This structure is composed by a list of three values:
        #
        # - The node neighbors
        # - The vector of indices of minor probabilities events
        # - The vector of probabilities for the extraction of the neighbors
        #
        # A very similar struct is also used for the edges.
        #

        # The following are empty versions of j and q, so to use only one
        # instance of these objects.

        # TODO! Consider parallelizing this thing.
        self._nodes_alias = build_alias_nodes(
            nodes,
            weights,
            self._nodes_mapping,
            self._nodes_neighboring_edges
        )

        # Creating struct saving all the data relative to the edges.
        # This structure is composed by a list of three values:
        #
        # - The edges neighbors
        # - The vector of indices of minor probabilities events
        # - The vector of probabilities for the extraction of the neighbors
        #
        # A very similar struct is also used for the nodes.
        #

        # TODO! Consider parallelizing this thing.
        self._edges_alias = build_alias_edges(
            edges_set,
            self._nodes_neighboring_edges,
            node_types,
            edge_types,
            weights,
            self._sources,
            self._destinations,
            return_weight,
            explore_weight,
            change_edge_type_weight,
            change_edge_type_weight
        )

        # To verify if this graph has some walker traps, meaning some nodes
        # that do not have any neighbors, we have to iterate on the list of
        # neighbors and to check if at least a node has no neighbors.
        # If such a condition is met, we cannot anymore do the simple random
        # walk assuming that all the walks have the same length, but we need
        # to create a random walk with variable length, hence a list of lists.

        self.has_traps = False
        for src in range(len(self._nodes_alias)):
            if self.is_node_trap(src):
                self.has_traps = True
                break

    @property
    def nodes_number(self) -> int:
        """Return the total number of nodes in the graph.

        Returns
        -------
        The total number of nodes in the graph.
        """
        return len(self._nodes_alias)

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
        return len(self._nodes_neighboring_edges[node]) == 0

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

    def neighbors(self, node: str) -> List:
        """Return neighbors of given node.

        Parameters
        ---------------------
        node: str,
            The node whose neigbours are to be identified.

        Returns
        ---------------------
        List of neigbours of given node.
        """
        # Get the numeric ID of the node
        node_id = self._nodes_mapping[node]
        # We get the node neighbours
        neighbours = self._nodes_neighboring_edges[node_id]
        # And translate them back.
        return [
            # For each neighbor edge, we need to get first the destination ID
            # and then remap the destination ID to the original node name
            self._reverse_nodes_mapping[self._destinations[neighbor]]
            for neighbor in neighbours
        ]

    def degree(self, node: str) -> int:
        """Return degree of given node.

        Parameters
        ---------------------
        node: str,
            The node whose neigbours are to be identified.

        Returns
        ---------------------
        Number of neighbors of given node.
        """
        return len(self.neighbors(node))

    def extract_transition_informations(self, src: int, j: np.ndarray, q: np.ndarray) -> Tuple[int, int]:
        # TODO! Add docstring!
        neighbor_index = alias_draw(j, q)
        neighbours = self._nodes_neighboring_edges[src]
        edge = neighbours[neighbor_index]
        # Get the destination of the chosen edge.
        dst = self._destinations[edge]
        # Return the obtained tuple
        return dst, edge

    def extract_random_node_neighbour(self, src: int) -> Tuple[int, int]:
        """Return a random adiacent node to the one associated to node.
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
        # !TODO! UPDATE THIS METHOD!
        # Get the information relative to the source node, composed of a tuple:
        # - The numpy array of opposite events for the alias method (j)
        # - The probabilities for the extractions for the alias method (q)
        j, q = self._nodes_alias[src]

        return self.extract_transition_informations(src, j, q)

    def extract_random_edge_neighbour(self, edge: int) -> Tuple[int, int]:
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
        # !TODO! UPDATE THIS METHOD!
        # We retrieve the destination of edge currently used, which can be
        # considered the source of the edge we are looking for now.
        src = self._destinations[edge]
        # Get the information relative to the source node, composed of a tuple:
        # - The numpy array of opposite events for the alias method (j)
        # - The probabilities for the extractions for the alias method (q)
        j, q = self._edges_alias[edge]

        return self.extract_transition_informations(src, j, q)
