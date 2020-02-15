import numpy as np
import os.path

from collections import defaultdict
# from .edge import Edge

from xn2v.csf_graph.edge import Edge


class CSFGraph:
    """Class converts data in to a compressed storage format graph.

    A file of data that is assumed to contain rows of graph edges, where each edge is a space-delimited string
    containing two nodes and an edge weight (e.g. 'g1 g2 10') is read in and converted into several dictionaries
    where node and edge types are indexed by integers.

    Attributes:
        edgetype2count_dictionary: A dictionary which stores the count of edges by edge type. In this dictionary,
            the keys are strings which represent an edge's type (i.e. a string that is comprised of the first character
            of each node) and values are integers, which represent the count of the edge type. For example:
                {'gg': 5, 'pp': 5, 'gp': 5, 'dg': 3, 'dd': 3}
        nodetype2count_dictionary: A dictionary which stores the count of nodes by node type. In this dictionary,
            the keys are strings which represent a node type (i.e. a string that is comprised of the first character
            of a node) and values are integers, which represent the count of the node type. For example:
                {'d': 3, 'g': 4, 'p': 4}
        edge_to: A numpy array with length the number of edges. Each index in the array contains the index of an edge,
            such that it can be used to look up the destination nodes that an edge in a given index points to.
        edge_weight: A numpy array with length the number of edges. Each item in the array represents an edge's index
            and each value in the array contains an edge weight.
        offset_to_edge_: A numpy array with length the number of unique nodes +1. Each index in the array represents a
            node and the value stored at each node's index is the total number of edges coming out of that node.
        node_to_index_map: A dictionary where keys contain node labels and values contain the integer index of each
            node from the sorted list of unique nodes in the graph.
        index_to_node_map: A dictionary where keys contain the integer index of each node from the sorted list of
            unique nodes in the graph and values contain node labels.
    """

    def __init__(self, filepath):
        if filepath is None:
            raise TypeError('ERROR: Need to pass path of file with edges')
        if not isinstance(filepath, str):
            raise TypeError('ERROR: filepath argument must be string')
        if not os.path.exists(filepath):
            raise TypeError('ERROR: Could not find graph file {}'.format(filepath))

        # create variables to store node and edge information
        nodes, edges = set(), set()
        self.edgetype2count_dictionary: defaultdict = defaultdict(int)
        self.nodetype2count_dictionary: defaultdict = defaultdict(int)
        self.node_to_index_map: defaultdict = defaultdict(int)
        self.index_to_node_map: defaultdict = defaultdict(str)

        # read in and process edge data, creating a dictionary that stores edge information.
        # filepath = 'tests/data/small_graph.txt'
        with open(filepath) as f:
            for line in f:
                fields = line.rstrip('\n').split()
                node_a = fields[0]
                node_b = fields[1]

                if len(fields) == 2:
                    # if no weight provided, assign a default value of 1.0
                    weight = 1.0
                else:
                    try:
                        weight = float(fields[2])
                    except ValueError:
                        print('ERROR: Could not parse weight field (must be an integer): {}'.format(fields[2]))
                        continue

                # add nodes
                nodes.add(node_a)
                nodes.add(node_b)

                # process edges y initializing instances of each edge and it's inverse
                edge = Edge(node_a, node_b, weight)
                inverse_edge = Edge(node_b, node_a, weight)
                edges.add(edge)
                edges.add(inverse_edge)

                # add the string representation of the edge to dictionary
                self.edgetype2count_dictionary[edge.get_edge_type_string()] += 1

        # convert node sets to numpy arrays, sorted alphabetically on on their source element
        node_list = sorted(nodes)

        # create node data dictionaries
        for i in range(len(node_list)):
            self.node_to_index_map[node_list[i]] = i
            self.index_to_node_map[i] = node_list[i]

        # initialize edge arrays
        # convert edge sets to numpy arrays, sorted alphabetically on on their source element
        edge_list = sorted(edges)
        total_edge_count = len(edge_list)
        total_vertex_count = len(node_list)

        self.edge_to: np.ndarray = np.zeros(total_edge_count, dtype=np.int32)
        self.edge_weight: np.ndarray = np.zeros(total_edge_count, dtype=np.int32)
        # self.proportion_of_different_neighbors = np.zeros(total_vertex_count, dtype=np.float32)
        self.offset_to_edge_: np.ndarray = np.zeros(total_vertex_count + 1, dtype=np.int32)

        # create the graph - this done by performing three passes over the data
        # first pass: count # of edges emanating from each source id
        index2edge_count = defaultdict(int)

        for edge in edge_list:
            source_index = self.node_to_index_map[edge.node_a]
            index2edge_count[source_index] += 1

        # second pass: set the offset_to_edge_ according to the number of edges emanating from each source ids
        self.offset_to_edge_[0], offset, i = 0, 0, 0

        for n in node_list:
            node_type = n[0]
            self.nodetype2count_dictionary[node_type] += 1
            source_index = self.node_to_index_map[n]
            # n_edges can be zero here, that is OK
            n_edges = index2edge_count[source_index]
            i += 1
            offset += n_edges
            self.offset_to_edge_[i] = offset

        # third pass: add the actual edges
        current_source_index = -1
        j, offset = 0, 0

        # use the offset variable to keep track of how many edges we have already entered for a given source index
        for edge in edge_list:
            source_index = self.node_to_index_map[edge.node_a]
            dest_index = self.node_to_index_map[edge.node_b]

            if source_index != current_source_index:
                current_source_index = source_index
                offset = 0  # start a new block
            else:
                # go to next index (for a new destination of the previous source)
                offset += 1

            self.edge_to[j] = dest_index
            self.edge_weight[j] = edge.weight
            j += 1

    def nodes(self):
        """Method which returns a list of graph nodes."""

        return list(self.node_to_index_map.keys())

    def nodes_as_integers(self):
        """Returns a list of integers representing the location of each node from the alphabetically-sorted list."""

        return list(self.index_to_node_map.keys())

    def node_count(self):
        """Returns an integer that contains the total number of unique nodes in the graph"""

        return len(self.node_to_index_map)

    def edge_count(self):
        """Returns an integer that contains the total number of unique edges in the graph"""

        return len(self.edge_to)

    def weight(self, source: str, dest: str):
        """Takes user provided strings, representing node names for a source and destination node, and returns
        weights for each edge that exists between these nodes.

        Assumptions:
            - Assumes that there is a valid edge between source and destination nodes.

        Args:
            source: The name of a source node.
            dest: The name of a destination node.

        Returns:
            The weight of the edge that exists between the source and destination nodes.
        """

        # get indices for each user-provided node string
        source_idx = self.node_to_index_map[source]
        dest_idx = self.node_to_index_map[dest]

        # get edge weights for nodes
        for i in range(self.offset_to_edge_[source_idx], self.offset_to_edge_[source_idx + 1]):
            if dest_idx == self.edge_to[i]:
                return self.edge_weight[i]

        # We should never get here -- leaving this in, but there are more pythonic ways to handle this error
        raise TypeError("Could not identify edge between {} and {}".format(source, dest))

    def weight_from_ints(self, source_idx: int, dest_idx: int):
        """Takes user provided integers, representing indices for a source and destination node, and returns
        weights for each edge that exists between these nodes.

        Assumptions:
            - Assumes that there is a valid edge between source and destination nodes.

        Args:
            source_idx: The index of a source node.
            dest_idx: The index of a destination node.

        Returns:
            The weight of the edge that exists between the source and destination nodes.
        """

        for i in range(self.offset_to_edge_[source_idx], self.offset_to_edge_[source_idx + 1]):
            if dest_idx == self.edge_to[i]:
                return self.edge_weight[i]

        # We should never get here -- this error should never happen!
        # leaving this in, but there are more pythonic ways to handle this error
        raise TypeError("Could not identify edge between {} and {}".format(source_idx, dest_idx))

    def neighbors(self, source: str):
        """Gets a list of node names, which are the neighbors of the user-provided source node.

        Args:
            source: The name of a source node.

        Returns:
            neighbors: A list of neighbors (i.e. strings representing node names) for a source node.
        """

        neighbors = []
        source_idx = self.node_to_index_map[source]

        for i in range(self.offset_to_edge_[source_idx], self.offset_to_edge_[source_idx + 1]):
            nbr = self.index_to_node_map[self.edge_to[i]]
            neighbors.append(nbr)

        return neighbors

    def neighbors_as_ints(self, source_idx: int):
        """Gets a list of node indices, which are the neighbors of the user-provided source node.

        Args:
            source_idx: The index of a source node.

        Returns:
            neighbors_ints: A list of indices of neighbors for a source node.
        """

        neighbors_ints = []

        for i in range(self.offset_to_edge_[source_idx], self.offset_to_edge_[source_idx + 1]):
            nbr = self.edge_to[i]
            neighbors_ints.append(nbr)

        return neighbors_ints

    def has_edge(self, src: str, dest: str):
        """Checks if an edge exists between src and dest node.

        Args:
            src:
            dest:

        Returns:
            A boolean value is returned, where True means an exists and False means no edge exists.
        """

        source_idx = self.node_to_index_map[src]
        dest_idx = self.node_to_index_map[dest]

        for i in range(self.offset_to_edge_[source_idx], self.offset_to_edge_[source_idx + 1]):
            nbr_idx = self.edge_to[i]

            if nbr_idx == dest_idx:
                return True

        return False

    @staticmethod
    def same_nodetype(n1: str, n2: str):
        """Encodes the node type using the first character of the node label. For instance, g1 and g2 have the same
        node type but g2 and p5 do not.

        Args:
            n1: A string containing an node name.
            n2: A string containing an node name.

        Returns:
            A boolean value is returned, where True means the user-provided nodes are of the same type and False
            means that they are of different types.
        """

        if n1[0] == n2[0]:
            return True
        else:
            return False

    def edges(self):
        """Creates an edge list, where nodes are coded by their string names, for the graph.

        Returns:
            edge_list: A list of tuples for all edges in the graph, where each tuple contains two strings that
                represent the name of each node in an edge. For instance, ('gg1', 'gg2').
        """

        edge_list = []

        for source_idx in range(len(self.offset_to_edge_) - 1):
            src = self.index_to_node_map[source_idx]
            for j in range(self.offset_to_edge_[source_idx], self.offset_to_edge_[source_idx + 1]):
                nbr_idx = self.edge_to[j]
                nbr = self.index_to_node_map[nbr_idx]
                tpl = (src, nbr)
                edge_list.append(tpl)

        return edge_list

    def edges_as_ints(self):
        """Creates an edge list, where nodes are coded by integers, for the graph.

        Returns:
            edge_list: A list of tuples for all edges in the graph, where each tuple contains two integers that
                represent each node in an edge. For instance, (2, 3).
        """

        edge_list = []

        for source_idx in range(len(self.offset_to_edge_) - 1):
            for j in range(self.offset_to_edge_[source_idx], self.offset_to_edge_[source_idx + 1]):
                nbr_idx = self.edge_to[j]
                tpl = (source_idx, nbr_idx)
                edge_list.append(tpl)

        return edge_list

    def get_node_to_index_map(self):
        """Returns a dictionary of the nodes in the graph and their corresponding integers."""

        return self.node_to_index_map

    def get_node_index(self, node: str):
        """Checks if a node exists in the node_to_index_map and if it is, the integer for that node is returned.

         Args:
             node: A string containing an node name.

        Returns:
            An integer index for the user-provided node.
        """

        if node not in self.node_to_index_map:
            raise ValueError('Could not find {} in node-to-index map'.format(node))
        else:
            return self.node_to_index_map.get(node)

    def get_index_to_node_map(self):
        """Returns a dictionary of the nodes in the graph and their corresponding integers."""

        return self.index_to_node_map

    def __str__(self):
        """Prints a string containing the total number of nodes and edges in the graph."""

        return 'CSFGraph(nodes: {}, edges: {})'.format(self.node_count(),  self.edge_count())

    def print_edge_type_distribution(self):
        """Prints node and edge type information for the graph.

        Note. This function is intended to be used for debugging or logging.

        Returns:
            -Node Type Counts: A string containing counts of node types.
            - Edge Type Counts:
                - A string containing the total counts of edges according to type.
                - If the graph only has one edge type, the total edge count is returned.
        """

        for n, count in self.nodetype2count_dictionary.items():
            print('node type {} - count: {}'.format(n, count))

        if len(self.edgetype2count_dictionary) < 2:
            print('edge count: {}'.format(self.edge_count()))
        else:
            for category, count in self.edgetype2count_dictionary.items():
                print('{} - count: {}'.format(category, count))
