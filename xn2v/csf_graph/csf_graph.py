import logging
import os.path
import numpy as np  # type: ignore
from collections import defaultdict
from typing import Optional, Any, Union, Set, Dict, List, Tuple

from xn2v.csf_graph.edge import Edge


class CSFGraphNoSubjectColumnError(Exception):
    pass


class CSFGraphNoObjectColumnError(Exception):
    pass


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
        node_to_index_map: A dictionary where keys contain node labels and values contain the integer index of each
            node from the sorted list of unique nodes in the graph.
        index_to_node_map: A dictionary where keys contain the integer index of each node from the sorted list of
            unique nodes in the graph and values contain node labels.
        nodetype_to_index_map: A dictionary where keys contain node type and values contain the integer index of each
            node from the sorted list of unique nodes in the graph.
        index_to_nodetype_map: A dictionary where keys contain the integer index of each node from the sorted list of
            unique nodes in the graph and values contain node types.
        edge_to: A numpy array with length the number of edges. Each index in the array contains the index of an edge,
            such that it can be used to look up the destination nodes that an edge in a given index points to.
        edge_weight: A numpy array with length the number of edges. Each item in the array represents an edge's index
            and each value in the array contains an edge weight.
        offset_to_edge_: A numpy array with length the number of unique nodes +1. Each index in the array represents a
            node and the value stored at each node's index is the total number of edges coming out of that node.

    Raises:
        - TypeError: If the filepath attribute is empty.
        - TypeError: If the filepath attribute must be type str.
        - ValueError: If the file referenced by filepath cannot be found.
    """

    def __init__(self, edge_file: str, node_file: str = None, default_weight=1):
        if not os.path.exists(edge_file):
            raise ValueError('Could not find edge file {}'.format(edge_file))

        # create variables to store node and edge information
        nodes: Set[Union[int, str]] = set()
        edges: Set[Edge] = set()

        self.subject_column_name = 'subject'
        self.object_column_name = 'object'
        self.edge_label_column_name = 'edge_label'
        self.default_edge_type = 'biolink:Association'

        self.edgetype2count_dictionary: Dict[str, int] = defaultdict(int)
        self.nodetype2count_dictionary: Dict[str, int] = defaultdict(int)
        self.node_to_index_map:  Dict[str, int] = defaultdict(int)
        self.index_to_node_map:  Dict[int, str] = defaultdict(str)
        self.nodetype_to_index_map:  Dict[str, list] = defaultdict(list)
        self.index_to_nodetype_map:  Dict[int, str] = defaultdict(str)
        self.edgetype_to_index_map:  Dict[str, list] = defaultdict(list)
        self.index_to_edgetype_map:  Dict[int, str] = defaultdict(str)

        # read in and process edge data, creating a dictionary that stores edge information
        header_info = self.parse_header(edge_file)
        with open(edge_file) as f:
            if not header_info['is_legacy']:  # legacy edge files don't have headers
                _ = f.readline()  # throw away header

            for line in f:
                fields = line.rstrip('\n').split()
                items = dict(zip(header_info['header_items'], fields))
                node_a = items[self.subject_column_name]
                node_b = items[self.object_column_name]
                if self.edge_label_column_name in items:
                    edge_type = items[self.edge_label_column_name]
                else:
                    edge_type = self.default_edge_type

                if 'weight' not in items:
                    # no weight provided. Assign a default value
                    weight = default_weight
                else:
                    try:
                        weight = float(items['weight'])
                    except ValueError as e:
                        logging.error("[ERROR] Could not parse weight field " +
                                      "(must be an integer): {}".format(
                                          items['weight']))
                # add nodes
                nodes.add(node_a)
                nodes.add(node_b)

                # process edges y initializing instances of each edge and it's inverse
                edge = Edge(node_a, node_b, weight, edge_type)
                inverse_edge = Edge(node_b, node_a, weight, edge_type)
                edges.add(edge)
                edges.add(inverse_edge)

                # add the string representation of the edge to dictionary
                self.edgetype2count_dictionary[edge.get_edge_type_string()] += 1

        # convert node sets to numpy arrays, sorted alphabetically on on their source element
        node_list: List = sorted(nodes)
        edge_list: List = sorted(edges)

        # create edge data dictionaries
        for i in range(len(edge_list)):
            this_type = edge_list[i].edge_type
            if this_type not in self.edgetype_to_index_map:
                self.edgetype_to_index_map[this_type] = []
            self.edgetype_to_index_map[this_type].append(i)
            self.index_to_edgetype_map[i] = this_type

        # create node data dictionaries
        for i in range(len(node_list)):
            self.node_to_index_map[node_list[i]] = i
            self.index_to_node_map[i] = node_list[i]

        # initialize edge arrays - convert edge sets to numpy arrays, sorted alphabetically on on their source element
        edge_list = sorted(edges)
        total_edge_count = len(edge_list)
        total_vertex_count = len(node_list)

        self.edge_to: np.ndarray = np.zeros(total_edge_count, dtype=np.int32)
        self.edge_weight: np.ndarray = np.zeros(total_edge_count, dtype=np.int32)
        # self.proportion_of_different_neighbors = np.zeros(total_vertex_count, dtype=np.float32)
        self.offset_to_edge_: np.ndarray = np.zeros(total_vertex_count + 1, dtype=np.int32)

        # create the graph - this done in three steps
        # step 1: count # of edges emanating from each source id
        index2edge_count: Dict[int, int] = defaultdict(int)

        for edge in edge_list:
            source_index = self.node_to_index_map[edge.node_a]
            index2edge_count[source_index] += 1

        # step 2: set the offset_to_edge_ according to the number of edges emanating from each source ids
        self.offset_to_edge_[0], offset, i = 0, 0, 0

        for n in node_list:
            node_type = n[0]
            self.nodetype2count_dictionary[node_type] += 1
            source_index = self.node_to_index_map[n]
            n_edges = index2edge_count[source_index]  # n_edges can be zero here
            i += 1
            offset += n_edges
            self.offset_to_edge_[i] = offset

        # step 3: add the actual edges
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
                offset += 1  # go to next index (for a new destination of the previous source)

            self.edge_to[j] = dest_index
            self.edge_weight[j] = edge.weight
            j += 1

    def parse_header(self, edge_file: str) -> dict:
        with open(edge_file, 'r') as fh:
            header_info: Dict[str, Union[list, bool]] = {}
            header_info['is_legacy'] = False

            header = fh.readline()
            header_items = header.strip().split('\t')
            header_info['header_items'] = header_items

            if self.subject_column_name not in header_items and \
                    self.object_column_name not in header_items:
                logging.warning(
                    "Didn't find subject or object in header - probably a legacy edge file")
                header_info['is_legacy'] = True
                if len(header_items) == 2:
                    header_info['header_items'] = [self.subject_column_name,
                                                   self.object_column_name]
                elif len(header_items) == 3:
                    header_info['header_items'] = [self.subject_column_name,
                                                   self.object_column_name,
                                                   'weight']
                else:
                    logging.error('Legacy edge file should have 2 or 3 columns' +
                                  '(node1 node2 [weight])\n{}'.format(header))
                    raise ValueError
            elif self.subject_column_name not in header_items:
                raise CSFGraphNoSubjectColumnError(
                    "Edge file should have a {} column", self.subject_column_name)
            elif self.object_column_name not in header_items:
                raise CSFGraphNoObjectColumnError(
                    "Edge file should have an {} column", self.object_column_name)
            return header_info

    def nodes(self) -> List[str]:
        """Returns a list of graph nodes."""
        return list(self.node_to_index_map.keys())

    def nodes_as_integers(self) -> List[int]:
        """Returns a list of integers representing the location of each node from the alphabetically-sorted list."""

        return list(self.index_to_node_map.keys())

    def node_count(self) -> int:
        """Returns an integer that contains the total number of unique nodes in the graph"""

        return len(self.node_to_index_map)

    def edge_count(self) -> int:
        """Returns an integer that contains the total number of unique edges in the graph"""

        return len(self.edge_to)

    def weight(self, source: str, dest: str) -> Optional[Union[int, float]]:
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
            else:
                pass

        return None

    def weight_from_ints(self, source_idx: int, dest_idx: int) -> Optional[Union[int, float]]:
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
            else:
                pass

        return None

    def neighbors(self, source: str) -> List[str]:
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

    def neighbors_as_ints(self, source_idx: int) -> List[int]:
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

    def has_edge(self, src: str, dest: str) -> bool:
        """Checks if an edge exists between src and dest node.

        Args:
            src: The name of a source node.
            dest: The name of a destination node.

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
    def same_nodetype(n1: str, n2: str) -> bool:
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

    def edges(self) -> List[Tuple[str, str]]:
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

    def edges_as_ints(self) -> List[Tuple[int, Any]]:
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

    def get_node_to_index_map(self) -> Dict[str, int]:
        """Returns a dictionary of the nodes in the graph and their corresponding integers."""

        return self.node_to_index_map

    def get_node_index(self, node: str) -> Optional[int]:
        """Checks if a node exists in the node_to_index_map and if it is, the integer for that node is returned.

         Args:
             node: A string containing an node name.

        Returns:
            An integer index for the user-provided node.

        Raises:
            ValueError: If a referenced node is not included in self.node_to_index_map.
        """

        if node not in self.node_to_index_map:
            raise ValueError('Could not find {} in self.node_to_index_map'.format(node))
        else:
            return self.node_to_index_map.get(node)

    def get_index_to_node_map(self) -> Dict[int, str]:
        """Returns a dictionary of the nodes in the graph and their corresponding integers."""

        return self.index_to_node_map

    def __str__(self) -> str:
        """Prints a string containing the total number of nodes and edges in the graph."""

        return 'CSFGraph(nodes: {}, edges: {})'.format(self.node_count(),  self.edge_count())

    def print_edge_type_distribution(self) -> None:
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
                print("%s - count: %d" % (category, count))

    def node_degree(self, node):
        """
        :param node
        :return: degree of node
        """
        return len(self.neighbors(node))
