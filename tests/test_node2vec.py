from unittest import TestCase

import networkx as nx
import os.path
import numpy as np
from hn2v.hetnode2vec import Graph


def calculate_total_probs(j, q):
    """
    Use the alias method to calculate the total probabilities of the discrete events
    :param j: alias vector
    :param q: alias-method probabilities
    :return:
    """
    N = len(j)
    # Total probs has the total probabilities of the events in the vector
    # Thus, it goes from the alias model, where the probability of one event
    # can be divided in two slots, to a simple 'multinomial' vector of probabilities
    total_probs = np.zeros(N)
    for i in range(N):
        p = q[i]
        total_probs[i] += p
        if p < 1.0:
            alias_index = j[i]
            total_probs[alias_index] += 1 - p
    s = np.sum(total_probs)
    return total_probs/s


class TestGraph(TestCase):

    def setUp(self):
        inputfile = os.path.join(os.path.dirname(__file__), 'data', 'small_graph.txt')
        g = nx.read_edgelist(inputfile, nodetype=str, data=(('weight', float),), create_using=nx.DiGraph())
        g = g.to_undirected()
        self.graph = g

    def test_simple_graph(self):
        is_directed = False
        p = 1
        q = 1
        gamma = 1
        g = Graph(self.graph, is_directed, p, q, gamma)
        src='g1'
        dst='g2'
        j_alias, q_alias = g.get_alias_edge(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # there are 4 outgoing edges from g2: g1, g3, p1, p2
        self.assertEqual(4, len(j_alias))
        #recreate the original probabilities. They should be 0.2, 0.2, 0.4, 0.2
        original_probs = calculate_total_probs(j_alias,q_alias)
        self.assertAlmostEqual(0.2, original_probs[0])
        self.assertAlmostEqual(0.2, original_probs[1])
        self.assertAlmostEqual(0.4, original_probs[2])
        self.assertAlmostEqual(0.2, original_probs[3])


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

    def testGraphNodeCounts(self):
        """
        #We expect 101 gene nodes, 20 disease nodes, and 30 protein nodes

        :return:
        """
        g = self.graph
        n = g.number_of_nodes()
        self.assertEqual(151, n)

    def testGraphEdgeCounts(self):
        """
         #We expect 100 g<->g edges, 1 g<->d edge, 19 d<->d edges, and 29 p<->p edges
        :return:
        """
        g = self.graph
        m = g.number_of_edges()
        self.assertEqual(150, m)

    def test_raw_probs(self):
        is_directed = False
        p = 1
        q = 1
        gamma = 1
        g = Graph(self.graph, is_directed, p, q, gamma, True)
        src = 'g0'
        dst = 'g1'
        j_alias, q_alias = g.get_alias_edge(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # there are 102 outgoing edges from g1. one edge from g1 to g0, 99 edges from g1 to g_i, i=2,...,100, one edge from g1 to d1
        self.assertEqual(102, len(j_alias))
        # recreate the original probabilities. They should be a vector of length 102 where all values are 10.0/102.0.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(10.0/1020.0, original_probs[self.d1index])


    # def test_gamma2(self):
    #     """
    #     We are using gamma2 to record the proportion of nodes with different edge types from the original node
    #     :return:
    #     """
    #     is_directed = False
    #     p = 1
    #     q = 1
    #     gamma = 1
    #     g = Graph(self.graph, is_directed, p, q, gamma)
    #     #g1 has 100 gene neighbors, 1 protein node, 1 disease node
    #     # thus the proportion of different nodes is 2/102
    #     gamma2 = self.graph.node['g1']['gamma2']
    #     #we expect gamma2 to be self.gamma/proportion of different node types
    #     expected = g.gamma / (2.0/102.0)
    #     self.assertAlmostEqual(expected, gamma2)

    def test_calculate_proportion_of_different_neighbors(self):
        """
        There is one neighbor to g2 which is g1. So, gamma2 is 1.
        :return:
        """
        is_directed = False
        p = 1
        q = 1
        gamma = float(1) / float(3)
        g = Graph(self.graph, is_directed, p, q, gamma, True)
        node ='g2'
        g.calculate_proportion_of_different_neighbors(node)
        self.assertEqual(1, self.graph.node['g2']['gamma2'])


    def test_calculate_proportion_of_different_neighbors_2(self):
        """
        g1 has 2 neighbors in different networks and 100 neighbors in the same network.
        So, prop =  2/102. Thus, gamma2 is (1/3) / (2/102) = 51/3 = 17
        :return:
        """
        is_directed = False
        p = 1
        q = 1
        gamma = float(1) / float(3)
        g = Graph(self.graph, is_directed, p, q, gamma, True)
        node ='g1'
        g.calculate_proportion_of_different_neighbors(node)
        self.assertEqual((1.0/3.0)/(2.0/102.0), self.graph.node['g1']['gamma2'])

    def test_alias_edge_2_length_1(self):
        """
        There are 102 edges from g_1. So, the length of j_alias is 102.
        :return:
        """
        is_directed = False
        p = 1
        q = 1
        gamma = float(1) / float(3)
        g = Graph(self.graph, is_directed, p, q, gamma, True)
        src = 'g0'
        dst = 'g1'
        j_alias, q_alias = g.get_alias_edgeHN2V(src, dst)
        self.assertEqual(102, len(j_alias))

    def test_alias_edge_2_length_2(self):
        """
        Tests if the lengths of j_alias array and q_alias array are equal.
        :return:
        """
        is_directed = False
        p = 1
        q = 1
        gamma = float(1) / float(3)
        g = Graph(self.graph, is_directed, p, q, gamma, True)
        src = 'g1'
        dst = 'd1'
        j_alias, q_alias = g.get_alias_edgeHN2V(src, dst)
        self.assertEqual(len(q_alias), len(j_alias))

    def test_gamma_probs(self):
        """
        Test that our rebalancing methods cause us to go from g1 to d1 with a probability of gamma = 1/k
        where k=3, i.e., the number of different node types
        :return:
        """
        is_directed = False
        p = 1
        q = 1
        gamma = float(1)/float(3)
        g = Graph(self.graph, is_directed, p, q, gamma, True)
        src = 'g0'
        dst = 'g1'
        j_alias, q_alias = g.get_alias_edgeHN2V(src, dst)
        alias_tuple = g.retrieve_alias_nodes()
        # recreate the original probabilities.
        original_probs = calculate_total_probs(j_alias, q_alias)
        numerator = g.gamma / (2.0/102.0)
        denominator = 100.0/102.0 + 102.0 * g.gamma
        expected = numerator/denominator
        self.assertAlmostEqual(1.0/3.0, original_probs[self.d1index])




