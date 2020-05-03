class Edge:
    """Class to organize edge data during the construction of the graph. Note that we do not need Edge objects in the
    final graph (edges are encoded using arrays).

    Attributes:
        node_a: A string containing a node (e.g. "g1").
        node_b: A string containing a node (e.g. "g2").
        edge_type: A string specifying edge type, preferably an association from the
        biolink model - either biolink:Association, or one of it's children. See here:
        https://biolink.github.io/biolink-model/docs/Association.html
        weight: A float representing an edge weight.
    """

    def __init__(self, n_a: str, n_b: str,  w: float, e_t: str='biolink:Association'):
        self.node_a: str = n_a
        self.node_b: str = n_b
        self.weight: float = w
        self.edge_type: str = e_t

    def get_edge_type_string(self):
        """Retrieves the edge type. The goal of this method is to help with summarizing the
        output of the graph, by providing a way to show how many homogeneous and how many heterogeneous edges there are.

        Assumptions:
            - The nodes are from an undirected graph. This matters because each representative node character is sorted
              alphabetically.

        Returns:
            edge_type: A string containing the edge type
        """

        return self.edge_type

    def __hash__(self):
        """Generates a hash from a tuple containing two node strings. For example: ('g1', 'g2) is hashed.

        Assumptions:
            - Weight is not included in the hash function.
            - Users are responsible for not including the same edge twice with different weights.

        Returns:
            hashed_edge: An integer hash of a tuple containing two node strings.
        """

        hashed_edge: int = hash((self.node_a, self.node_b))

        return hashed_edge

    def __eq__(self, other):
        """Determines whether two edge tuples (i.e. ("g1", "g2")) are equal.

        Assumption: For determining equality, edge weight is disregarded.

        Returns:
            edge_equality: A bool indicating whether or not the two edge tuples are equal.
        """

        edge_equality = (self.node_a, self.node_b) == (other.node_a, other.node_b)

        return edge_equality

    def __ne__(self, other):
        """Determines if two objects are equal.

        Return:
            edge_equality: A bool indicating whether or not the two objects are equal.
        """

        equality = not (self == other)

        return equality

    def __lt__(self, other):
        """Used to sort edge lists; sort on (1) source node and (2) destination node.

        Returns:
            A boolean indicating whether or not a current node is less than another node.
        """

        if self.node_a == other.node_a:
            return self.node_b < other.node_b

        return self.node_a < other.node_a
