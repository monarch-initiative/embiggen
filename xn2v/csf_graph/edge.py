class Edge:
    """
    Class to organize edge data during the construction of the graph. We do not need
    Edge objects in the final graph (edges are encoded using arrays)
    """

    def __init__(self, nA, nB, w):
        self.nodeA = nA
        self.nodeB = nB
        self.weight = w

    def get_edge_type_string(self):
        '''
        For the summary output of the graph, we would like to show how many homogeneous and
        how many heterogeneous edges there are. We expect nodes types to be coded using the
        first characters of the nodes. For instance, g1-g3 would we coded as 'gg' and 'g1-p45'
        would be coded as 'gp'. We assume an undirected graph. We sort the characters alphabetically.
        :return:
        '''
        na = self.nodeA[0]
        nb = self.nodeB[0]
        return "".join(sorted([na, nb]))

    def __hash__(self):
        """
        Do not include weight in the hash function, we are interested in
        the edges. Users are responsible for not including the same edge
        twice with different weights
        """
        return hash((self.nodeA, self.nodeB))

    def __eq__(self, other):
        """
        For equality, we disregard the edge weight
        """
        return (self.nodeA, self.nodeB) == (other.nodeA, other.nodeB)

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        """
        We sort on (1) source node and (2) destination node
        """
        if self.nodeA == other.nodeA:
            return self.nodeB < other.nodeB
        return self.nodeA < other.nodeA
