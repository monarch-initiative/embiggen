from unittest import TestCase

import networkx as nx
import os.path
import numpy as np
from n2v import N2vGraph


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
        p = 1
        q = 1
        gamma = 1
        g = N2vGraph(self.graph, p, q, gamma)
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
        p = 1
        q = 1
        gamma = 1
        g = N2vGraph(self.graph, p, q, gamma, True)
        src = 'g0'
        dst = 'g1'
        j_alias, q_alias = g.get_alias_edge(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # there are 102 outgoing edges from g1. one edge from g1 to g0, 99 edges from g1 to g_i, i=2,...,100, one edge from g1 to d1
        self.assertEqual(102, len(j_alias))
        # recreate the original probabilities. They should be a vector of length 102 where all values are 10.0/102.0.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(10.0/1020.0, original_probs[self.d1index])



