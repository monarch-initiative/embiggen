from unittest import TestCase

import networkx as nx
import os.path
import numpy as np
from n2v import N2vGraph


def calculate_total_probs(j, q):
    """
    Use the alias method to calculate the total probabilites of the discrete events
    :param j: alias vector
    :param q: alias-method probabilities
    :return:
    """
    N = len(j)
    probs = np.zeros(N)
    for i in range(N):
        p = q[i]
        probs[i] += p
        if p < 1.0:
            alias_index = j[i]
            probs[alias_index] += 1 - p
    s = np.sum(probs)
    return probs/s

class TestHetGraph(TestCase):

    def setUp(self):
        inputfile = os.path.join(os.path.dirname(__file__), 'data', 'small_het_graph.txt')
        g = nx.read_edgelist(inputfile, nodetype=str, data=(('weight', float),), create_using=nx.DiGraph())
        g = g.to_undirected() #adding this to show the graph is not firected.
        self.graph = g
        self.nodes = g.nodes()
        self.g1index = self.__get_index('g1')
        self.d1index = self.__get_index('d1')

    def __get_index(self, node):
        i = 0
        for n in self.nodes:
            if n == node:
                return i
            i += 1
        raise Exception("Could not find {} in self.nodes".format(node))

    def get_index_of_neighbor(self, node, neighbor):
        """
        Searches for the index of a neighbor of node
        If there is no neighbor, return None
        :param node:
        :param neighbor:
        :return:
        """
        neighbors = sorted(self.graph.neighbors(node))
        i = 0
        for n in neighbors:
            if n == neighbor:
                return i
            i += 1
        return None

    def test_gamma_probs(self):
        """
        Test that our rebalancing methods cause us to go from g1 to d1 with a probability of gamma = 1/k
        where k=3, i.e., the number of different node types
        :return:
        """
        p = 1
        q = 1
        gamma = float(1) / float(3)
        g = N2vGraph(self.graph, p, q, gamma, True)
        src = 'g0'
        dst = 'g1'
        g0g1tuple = (src, dst) # this is a key of the dictionary alias_edge_tuple
        alias_edge_tuple = g.retrieve_alias_edges()
        g0g1edges = alias_edge_tuple.get(g0g1tuple)
        # g0g1 edges has the alias map and probabilities as a 2-tuple
        # The following code searches for the index of d1
        # We want the probability of going from g1 to d1 -- it should be equal to gamma, because there is only one g1 to disease edge
        idx = self.get_index_of_neighbor('g1', 'd1')
        self.assertIsNotNone(idx)
        probs = calculate_total_probs(g0g1edges[0], g0g1edges[1])
        d1prob = probs[idx] #total probability of going from g1 to d1
        self.assertAlmostEqual(1.0 / 3.0, d1prob)

    def test_gamma_probs_2(self):
        """
        Test that our rebalancing methods cause us to go from g1 to d1 with a probability of gamma = 1/k
        where k=3, i.e., the number of different node types
        :return:
        """
        p = 1
        q = 1
        gamma = float(1) / float(3)
        g = N2vGraph(self.graph, p, q, gamma, True)
        src = 'g1'
        dst = 'g2'
        g1g2tuple = (src, dst) # this is a key of the dictionary alias_edge_tuple
        alias_edge_tuple = g.retrieve_alias_edges()
        g1g2edges = alias_edge_tuple.get(g1g2tuple)
        # g1g2 edges has the alias map and probabilities as a 2-tuple
        # The following code searches for the index of g1
        # We want the probability of going from g2 to g1 -- it should be equal to 1 because there is only one neighbor to g2
        # and it is in the same network
        idx = self.get_index_of_neighbor('g2', 'g1')
        self.assertIsNotNone(idx)
        probs = calculate_total_probs(g1g2edges[0], g1g2edges[1])
        g1prob = probs[idx] #total probability of going from g2 to g1
        self.assertAlmostEqual(1.0, g1prob)


    def test_gamma_probs_3(self):
        """
        Test that our rebalancing methods cause us to go from g1 to d1 with a probability of gamma = 1/k
        where k=3, i.e., the number of different node types
        :return:
        """
        p = 1
        q = 1
        gamma = float(1) / float(3)
        g = N2vGraph(self.graph, p, q, gamma, True)
        src = 'g1'
        dst = 'p1'
        g1p1tuple = (src, dst) # this is a key of the dictionary alias_edge_tuple
        alias_edge_tuple = g.retrieve_alias_edges()
        g1p1edges = alias_edge_tuple.get(g1p1tuple)
        # g1p1 edges has the alias map and probabilities as a 2-tuple
        # The following code searches for the index of p2
        # We want the probability of going from p1 to p2 -- it should be equal
        idx = self.get_index_of_neighbor('p1', 'g1')
        self.assertIsNotNone(idx)
        probs = calculate_total_probs(g1p1edges[0], g1p1edges[1])
        g1prob = probs[idx] #total probability of going from p1 to g1.
        # p1 has g1 as a neighbor in different network and has 29 protein neighbors in the same network.
        #The probability of going back to g1 is gamma which is 1/3
        self.assertAlmostEqual(1.0/3.0, g1prob)


    def test_gamma_probs_4(self):
        """
        Assume we have traversed the g1-p1 edge.
        Test that our rebalancing methods cause us to go from p1 to g1 with a probability of gamma = 1/k
        where k=3, i.e., the number of different node types
        :return:
        """
        p = 1
        q = 1
        gamma = float(1) / float(3)
        g = N2vGraph(self.graph, p, q, gamma, True)
        src = 'g1'
        dst = 'p1'
        g1p1tuple = (src, dst) # this is a key of the dictionary alias_edge_tuple
        alias_edge_tuple = g.retrieve_alias_edges()
        g1p1edges = alias_edge_tuple.get(g1p1tuple)
        # g1p1 edges has the alias map and probabilities as a 2-tuple
        # The following code searches for the index of p2
        # We want the probability of going from p1 to p2 -- it should be equal
        idx = self.get_index_of_neighbor('p1', 'p2')
        self.assertIsNotNone(idx)
        probs = calculate_total_probs(g1p1edges[0], g1p1edges[1])
        p2prob = probs[idx] #total probability of going from p1 to p2.
        # p1 has g1 as a neighbor in different network and has 29 protein neighbors in the same network.
        #The probability of going  to p2 is gamma which is (2/3*29) ~ 0.023
        self.assertAlmostEqual(2.0/87.0, p2prob)
