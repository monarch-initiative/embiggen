from typing import List, Union, Tuple, Dict
import numpy as np
from numba import jitclass, typed, types, njit, prange
from .alias_method import alias_draw, alias_setup


# key and value types
keys_tuple = types.UniTuple(types.int64, 2)
kv_ty = (keys_tuple, types.int64)
integer_list = types.ListType(types.int64)
float_list = types.ListType(types.float64)
triple_list = types.Tuple((integer_list, types.int64[:], types.float64[:]))


@jitclass([
    ('_edges', types.DictType(*kv_ty)),
    ('_edges_indices', types.ListType(keys_tuple)),
    ('_nodes_alias', types.ListType(triple_list)),
    ('_edges_alias', types.ListType(triple_list)),
])
class Graph:

    def __init__(
        self,
        edges: List[Tuple[str, str]],
        weights: Union[List],
        nodes: List[str],
        nodes_type: List[int],
        nodes_types_number: int,
        directed: List[bool],
        return_weight: float = 1,
        explore_weight: float = 1,
        jump_weight: float = 1
    ):
        """Crate a new instance of a undirected graph with given edges.

        Parameters
        ---------------------
        edges: List[Tuple[str, str]],
            List of edges of the graph.
        nodes: List[str],
            List of the nodes of the graph.
        weights: List[float],
            The weights for each source and sink.
        TODO: check where to put the s in the names of variables
        node_types: List[int],
            The node types for each source and sink.
        nodes_types_number: int,
            Number of unique node types.
        directed: List[bool],
            The edges directions for each source and sink.
        return_weight : float in (0, inf),
            Weight on the probability of returning to node coming from
            Having this higher tends the walks to be
            more like a Breadth-First Search.
            Having this very high  (> 2) makes search very local.
            Equal to the inverse of p in the Node2Vec paper.
        explore_weight : float in (0, inf),
            Weight on the probability of visiting a neighbor node
            to the one we're coming from in the random walk
            Having this higher tends the walks to be
            more like a Depth-First Search.
            Having this very high makes search more outward.
            Having this very low makes search very local.
            Equal to the inverse of q in the Node2Vec paper.
        jump_weight : float in (0, inf),
            Weight on the probability of visiting a neighbor node
            that has a different node type than the current node.

        Returns
        ---------------------
        New instance of graph.
        """

        nodes_number = len(nodes)

        # Creating mapping of nodes and integer ID.
        # The map looks like the following:
        # {
        #   "node_1_id": 0,
        #   "node_2_id": 1
        # }
        # This is only a method variable and not a class variable because
        # it is only used during the constractor.
        nodes_mapping = typed.Dict.empty(types.string, types.int64)
        for i, node in enumerate(nodes):
            nodes_mapping[node] = i

        # Creating mapping of edges and integer ID.
        # The map looks like the following:
        # {
        #   (0, 1): 0,
        #   (0, 2): 1
        # }
        # This is a class variable and not a method variable because it is
        # also used for the counter translation.
        self._edges = typed.Dict.empty(*kv_ty)

        # Each node has a list of neighbours.
        # These lists are initialized as empty.
        nodes_neighbours = typed.List.empty_list(integer_list)
        # Each node has a list of neighbours weights.
        # These lists are initialized as empty, if a weight
        neighbours_weights = typed.List.empty_list(float_list)

        for _ in range(nodes_number):
            nodes_neighbours.append(typed.List.empty_list(types.int64))
            neighbours_weights.append(typed.List.empty_list(types.float64))

        # The following proceedure ASSUMES that the edges only appear
        # in a single direction. This must be handled in the preprocessing
        # of the graph parsing proceedure.
        i = 0
        for k, (start_name, end_name) in enumerate(edges):
            src, dst = nodes_mapping[start_name], nodes_mapping[end_name]
            if (src, dst) not in self._edges:
                self._edges[(src, dst)] = i
                nodes_neighbours[src].append(dst)
                neighbours_weights[src].append(weights[k])
                i += 1
            # If the edge is not-directed we add the inverse to be able to
            # convert undirected graph to a directed one.
            if not directed[k] and (dst, src) not in self._edges:
                self._edges[(dst, src)] = i
                nodes_neighbours[dst].append(src)
                neighbours_weights[dst].append(weights[k])
                i += 1

        # Compute edges
        self._edges_indices = typed.List.empty_list(keys_tuple)
        for edge in self._edges:
            self._edges_indices.append(edge)

        # Compute edges neighbours to avoid having to make a double search
        # or a dictionary: this enables us to do everything after this
        # preprocessing step using purely numba.
        edges_neighbours = typed.List.empty_list(integer_list)
        for _ in range(len(self._edges)):
            edges_neighbours.append(typed.List.empty_list(types.int64))
        # Since we construct the dual graph, we need the neighbours of the edges
        # Therefore if we have a graph like
        # A -> B --> C
        #        \-> D
        # it would became:
        # AB --> BC.
        #    \-> BD
        # So the neighbours of each edge (start, end) are all the
        # edges in the form (end, neigh) where neigh is any neighbour of end.
        for (_, dst), edge_neighbours in zip(self._edges_indices, edges_neighbours):
            for neigh in nodes_neighbours[dst]:
                edge_neighbours.append(self._edges[dst, neigh])

        # Creating struct saving all the data relative to the nodes.
        # This structure is composed by a list of three values:
        #
        # - The node neighbours
        # - The vector of indices of minor probabilities events
        # - The vector of probabilities for the extraction of the neighbours
        #
        # A very similar struct is also used for the edges.
        #
        self._nodes_alias = typed.List.empty_list(triple_list)
        for node_neighbours, neighbour_weights in zip(nodes_neighbours, neighbours_weights):
            # Allocate a vector with the length of the possible types
            types_count = np.zeros(nodes_types_number, dtype=np.int64)
            # A variable for the unique types
            unique_types = 0
            # Then we iterate on the vector of the available neighbours
            for neighbour in node_neighbours:
                # And we count the number of neighbours for each given type.
                if types_count[nodes_type[neighbour]] == 0:
                    unique_types += 1
                types_count[nodes_type[neighbour]] += 1

            types_count = jump_weight / (types_count*unique_types)

            probs = np.empty(len(neighbour_weights))

            for i, weight in enumerate(neighbour_weights):
                probs[i] = weight
            j, q = alias_setup(probs/probs.sum())
            self._nodes_alias.append((node_neighbours, j, q))

        # Creating struct saving all the data relative to the edges.
        # This structure is composed by a list of three values:
        #
        # - The edges neighbours
        # - The vector of indices of minor probabilities events
        # - The vector of probabilities for the extraction of the neighbours
        #
        # A very similar struct is also used for the nodes.
        #
        self._edges_alias = typed.List.empty_list(triple_list)
        for (src, dst), edge_neighbours in zip(self._edges_indices, edges_neighbours):
            probs = np.empty(len(edge_neighbours))
            for index, neighbour in enumerate(edge_neighbours):
                # We get the weight for the edge from the destination to
                # the neighbour.
                weight = neighbours_weights[dst][index]
                # If the neigbour matches with the source, hence this is
                # a backward loop like the following:
                # SRC -> DST
                #  ▲     /
                #   \___/
                #
                # We weight the edge weight with the given return weight.
                if neighbour == src:
                    weight = weight * return_weight
                # If the backward loop does not exist, we multiply the weight
                # of the edge by the weight for moving forward and explore more.
                elif (neighbour, src) not in self._edges:
                    weight = weight * explore_weight
                # Then we store these results into the probability vector.
                probs[index] = weight

            j, q = alias_setup(probs/probs.sum())
            self._edges_alias.append((edge_neighbours, j, q))

    @property
    def nodes_number(self) -> int:
        """Return the total number of nodes in the graph.

        Returns
        -------
        The total number of nodes in the graph.
        """
        return len(self._nodes_alias)

    def get_edge_id(self, src: int, dst: int) -> int:
        """Return the numeric id for the curresponding edge.

        Parameters
        ----------
        src: int,
            The start node of the edge
        dst: int,
            The end node of the edge

        Returns
        -----------------
        Edge numeric ID.
        """
        return self._edges[src, dst]

    def get_edge_destination(self, edge: int) -> int:
        """Return the endpoint of the given edge ID.

        Parameters
        ----------
        edge: int,
            The id of the edge.

        Returns
        -----------------
        Return destination id of given edge.
        """
        return self._edges_indices[edge][1]

    def extract_random_node_neighbour(self, node: int) -> int:
        """Return a random adiacent node to the one associated to node.
        The Random is extracted by using the normalized weights of the edges
        as probability distribution. 

        Parameters
        ----------
        node: int
            The index of the node that is to be considered.

        Returns
        -------
        The index of a random adiacent node to node.
        """
        neighbours, j, q = self._nodes_alias[node]
        return neighbours[alias_draw(j, q)]

    def extract_random_edge_neighbour(self, edge: int) -> int:
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
        neighbours, j, q = self._edges_alias[edge]
        return neighbours[alias_draw(j, q)]

    def random_walk(self, number: int = 100, length: int = 100) -> np.ndarray:
        """Return random walks on graph.

        Parameters
        ---------------------
        number: int = 100,
            Number of walks to execute.
        length: int = 100,
            Length of the walks to execute.

        Returns
        ----------------------
        Numpy array with the walks.
        """
        return random_walk(self, number, length)


# This function is out of the class because otherwise we would not be able
# to activate the parallel=True flag.
@njit(parallel=True)
def random_walk(graph: Graph, number: int, length: int) -> np.ndarray:
    """Return a list of graph walks

    Parameters
    ----------
    graph:Graph
        The graph on which the random walks will be done.
    number: int,
        Number of walks to execute.
    length:int,
        The length of the walks in edges traversed.

    Returns
    -------
    Numpy array with all the walks.
    """
    all_walks = np.empty((number, graph.nodes_number, length), dtype=np.int64)

    # We can use prange to parallelize the walks and the iterations on the
    # graph nodes.
    for i in prange(number):  # pylint: disable=not-an-iterable
        for src in prange(graph.nodes_number):  # pylint: disable=not-an-iterable
            walk = all_walks[i][src]
            # TODO: if the todo below is green-lighted also the following
            # two lines have to be rewritten to only include node IDs.
            walk[0] = src
            walk[1] = dst = graph.extract_random_node_neighbour(src)
            edge = graph.get_edge_id(src, dst)
            for index in range(2, length):
                edge = graph.extract_random_edge_neighbour(edge)
                # TODO: the following line might not be needed at all!
                walk[index] = graph.get_edge_destination(edge)
    return all_walks
