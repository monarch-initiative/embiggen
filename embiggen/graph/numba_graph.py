from typing import List, Tuple
import numpy as np  # type: ignore
from numba.experimental import jitclass  # type: ignore
from numba import typed, types, prange  # type: ignore
from .alias_method import alias_draw, alias_setup

# key and value types
keys_tuple = types.UniTuple(types.int64, 2)
keys_triple = types.UniTuple(types.int64, 3)
kv_ty = (keys_triple, types.int64)
integer_list = types.ListType(types.int64)
float_list = types.ListType(types.float64)
triple_list = types.Tuple((integer_list, types.int64[:], types.float64[:]))


@jitclass([
    ('_edges', types.DictType(*kv_ty)),
    ('_edges_indices', types.ListType(keys_triple)),
    ('_nodes_alias', types.ListType(triple_list)),
    ('_nodes_mapping', types.DictType(types.string, types.int64)),
    ('_nodes_reverse_mapping', types.ListType(types.string)),
    ('_grouped_edge_types', types.DictType(keys_tuple, integer_list)),
    ('_node_types', types.int64[:]),
    ('_edge_types', integer_list),
    ('_neighbors_edge_types', types.ListType(integer_list)),
    ('_edges_alias', types.ListType(triple_list)),
    ('has_traps', types.boolean),
    ('random_walk_preprocessing', types.boolean),
])
class NumbaGraph:

    def __init__(
        self,
        edges: List[Tuple[str, str]],
        weights: np.ndarray,
        nodes: np.ndarray,
        node_types: np.ndarray,
        edge_types: np.ndarray,
        directed: np.ndarray,
        return_weight: float = 1,
        explore_weight: float = 1,
        change_node_type_weight: float = 1,
        change_edge_type_weight: float = 1,
        random_walk_preprocessing: bool = True
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
        node_types: List[int],
            The node types for each node.
        node_types: List[int],
            The edge types for each source and sink.
        directed: List[bool],
            The edges directions for each source and sink.
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

        self.random_walk_preprocessing = random_walk_preprocessing
        nodes_number = len(nodes)

        # Storing node types
        self._node_types = node_types

        # Creating mapping of nodes and integer ID.
        # The map looks like the following:
        # {
        #   "node_1_id": 0,
        #   "node_2_id": 1
        # }
        self._nodes_mapping = typed.Dict.empty(types.string, types.int64)
        self._nodes_reverse_mapping = typed.List.empty_list(types.string)
        for i, node in enumerate(nodes):
            self._nodes_mapping[str(node)] = i
            self._nodes_reverse_mapping.append(str(node))

        # Creating mapping of edges and integer ID.
        # The map looks like the following:
        # {
        #   (0, 1): 0,
        #   (0, 2): 1
        # }
        self._edges = typed.Dict.empty(*kv_ty)

        if self.random_walk_preprocessing:
            # Each node has a list of neighbors.
            # These lists are initialized as empty.
            nodes_neighbors = typed.List.empty_list(integer_list)
            # Each node has a list of neighbors weights.
            # These lists are initialized as empty, if a weight
            neighbors_weights = typed.List.empty_list(float_list)
            # Each node has a list of the edges types.
            # These lists are initialized as empty, if a weight
            self._neighbors_edge_types = typed.List.empty_list(integer_list)

            for _ in range(nodes_number):
                nodes_neighbors.append(typed.List.empty_list(types.int64))
                neighbors_weights.append(typed.List.empty_list(types.float64))
                self._neighbors_edge_types.append(
                    typed.List.empty_list(types.int64)
                )

            # Dictionary of lists of types.
            self._grouped_edge_types = typed.Dict.empty(
                keys_tuple, integer_list)

        # The following proceedure ASSUMES that the edges only appear
        # in a single direction. This must be handled in the preprocessing
        # of the graph parsing proceedure.
        i = 0
        for k, ((start_name, end_name), edge_type) in enumerate(zip(edges, edge_types)):
            src = self._nodes_mapping[str(start_name)]
            dst = self._nodes_mapping[str(end_name)]
            if (src, dst, edge_type) not in self._edges:
                self._edges[(src, dst, edge_type)] = i
                i += 1
                if self.random_walk_preprocessing:
                    if (src, dst) not in self._grouped_edge_types:
                        self._grouped_edge_types[src, dst] = typed.List.empty_list(
                            types.int64)
                    self._grouped_edge_types[src, dst].append(edge_type)
                    nodes_neighbors[src].append(dst)
                    self._neighbors_edge_types[src].append(edge_type)
                    neighbors_weights[src].append(weights[k])
            # If the edge is not-directed we add the inverse to be able to
            # convert undirected graph to a directed one.
            if not directed[k] and (dst, src, edge_type) not in self._edges:
                self._edges[(dst, src, edge_type)] = i
                i += 1
                if self.random_walk_preprocessing:
                    if (dst, src) not in self._grouped_edge_types:
                        self._grouped_edge_types[dst, src] = typed.List.empty_list(
                            types.int64)
                    self._grouped_edge_types[dst, src].append(edge_type)
                    nodes_neighbors[dst].append(src)
                    self._neighbors_edge_types[dst].append(edge_type)
                    neighbors_weights[dst].append(weights[k])

        # Compute edges
        self._edges_indices = typed.List.empty_list(keys_triple)
        for edge in self._edges:
            self._edges_indices.append(edge)

        if not self.random_walk_preprocessing:
            return

        # Compute edges neighbors to avoid having to make a double search
        # or a dictionary: this enables us to do everything after this
        # preprocessing step using purely numba.
        edges_neighbors = typed.List.empty_list(integer_list)
        for _ in range(len(self._edges)):
            edges_neighbors.append(typed.List.empty_list(types.int64))
        # Since we construct the dual graph, we need the neighbors of the edges
        # Therefore if we have a graph like
        # A -> B --> C
        #        \-> D
        # it would became:
        # AB --> BC.
        #    \-> BD
        # So the neighbors of each edge (start, end) are all the
        # edges in the form (end, neigh) where neigh is any neighbour of end.
        for (_, dst, edge_type), edge_neighbors in zip(self._edges_indices, edges_neighbors):
            # For each node in the neighbourhood
            for neigh in nodes_neighbors[dst]:
                # We iterate the available edge_types of the edge
                for neigh_edge_type in self._grouped_edge_types[dst, neigh]:
                    # And retreve the edge id for the specific type.
                    edge_neighbors.append(
                        self._edges[dst, neigh, neigh_edge_type])

        # Creating struct saving all the data relative to the nodes.
        # This structure is composed by a list of three values:
        #
        # - The node neighbors
        # - The vector of indices of minor probabilities events
        # - The vector of probabilities for the extraction of the neighbors
        #
        # A very similar struct is also used for the edges.
        #
        self._nodes_alias = typed.List.empty_list(triple_list)
        for node_neighbors, neighbour_weights in zip(nodes_neighbors, neighbors_weights):
            probs = np.zeros(len(neighbour_weights))
            for i, weight in enumerate(neighbour_weights):
                probs[i] = weight
            # Do not call the alias setup if the node is a trap.
            # Because that node will have no neighbors and thus the necessity
            # of setupping the alias method to efficently extract the neighbour.
            if len(neighbour_weights):
                j, q = alias_setup(probs/probs.sum())
            else:
                j, q = np.zeros(0, dtype=np.int64), np.zeros(0)
            self._nodes_alias.append((node_neighbors, j, q))

        # Creating struct saving all the data relative to the edges.
        # This structure is composed by a list of three values:
        #
        # - The edges neighbors
        # - The vector of indices of minor probabilities events
        # - The vector of probabilities for the extraction of the neighbors
        #
        # A very similar struct is also used for the nodes.
        #
        self._edges_alias = typed.List.empty_list(triple_list)
        for (src, dst, edge_type), edge_neighbors in zip(self._edges_indices, edges_neighbors):
            probs = np.zeros(len(edge_neighbors))
            destination_type = self._node_types[dst]
            for index, neigh in enumerate(edge_neighbors):
                # We get the weight for the edge from the destination to
                # the neighbour.
                weight = neighbors_weights[dst][index]
                # We retrieve the edge informations which is a triple like
                # the following:
                # (source, destination, edge_type)
                _, neigh_dst, neigh_edge_type = self._edges_indices[neigh]
                # Then we retrieve the neigh_dst node type.
                neigh_dst_node_type = self._node_types[neigh_dst]
                # And if the destination node type matches the neighbour
                # destination node type (we are not changing the node type)
                # we weigth using the provided change_node_type_weight weight.
                if destination_type == neigh_dst_node_type:
                    weight /= change_node_type_weight
                # Similarly if the neighbour edge type matches the previous
                # edge type (we are not changing the edge type)
                # we weigth using the provided change_edge_type_weight weight.
                if edge_type == neigh_edge_type:
                    weight /= change_edge_type_weight
                # If the neigbour matches with the source, hence this is
                # a backward loop like the following:
                # SRC -> DST
                #  ▲     /
                #   \___/
                #
                # We weight the edge weight with the given return weight.
                if neigh == src:
                    weight = weight * return_weight
                # If the backward loop does not exist, we multiply the weight
                # of the edge by the weight for moving forward and explore more.
                elif (neigh, src) not in self._grouped_edge_types:
                    weight = weight * explore_weight
                # Then we store these results into the probability vector.
                probs[index] = weight
            # Do not call the alias setup if the edge is a trap.
            # Because that edge will have no neighbors and thus the necessity
            # of setupping the alias method to efficently extract the neighbour.
            if len(edge_neighbors):
                j, q = alias_setup(probs/probs.sum())
            else:
                j, q = np.zeros(0, dtype=np.int64), np.zeros(0)

            self._edges_alias.append((edge_neighbors, j, q))

        # To verify if this graph has some walker traps, meaning some nodes
        # that do not have any neighbors, we have to iterate on the list of
        # neighbors and to check if at least a node has no neighbors.
        # If such a condition is met, we cannot anymore do the simple random
        # walk assuming that all the walks have the same length, but we need
        # to create a random walk with variable length, hence a list of lists.

        self.has_traps = False
        for src in range(nodes_number):
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
        return len(self._nodes_alias[node][0]) == 0

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
        return len(self._edges_alias[edge][0]) == 0

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
        return [
            self._nodes_reverse_mapping[neighbor]
            for neighbor in self._nodes_alias[self._nodes_mapping[node]][0]
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
        # Get the information relative to the source node, composed of a triple:
        # - The list of the node's neighbours
        # - The numpy array of opposite events for the alias method (j)
        # - The probabilities for the extractions for the alias method (q)
        neighbors, j, q = self._nodes_alias[src]
        # We extract using the alias method the index of the transition edge
        neighbor_index = alias_draw(j, q)
        # We then retrieve the edge type corresponding to the edge used for
        # the transition.
        edge_type = self._neighbors_edge_types[src][neighbor_index]
        # We retrieve the ID of the destination node
        dst = neighbors[neighbor_index]
        # Finally we retrieve the ID of the edge curresponding at the position
        # given by the triple (src, dst, edge_type)
        edge = self._edges[src, dst, edge_type]
        return dst, edge

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
        neighbors, j, q = self._edges_alias[edge]
        return neighbors[alias_draw(j, q)]
