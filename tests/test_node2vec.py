from unittest import TestCase

import os.path
from xn2v import CSFGraph
from xn2v import N2vGraph
from tests.utils.utils import calculate_total_probs


class TestGraph(TestCase):

    def setUp(self):
        inputfile = os.path.join(os.path.dirname(__file__), 'data', 'small_graph.txt')
        g = CSFGraph(inputfile)
        self.graph = g

    def test_raw_probs_simple_graph_1(self):
        p = 1
        q = 1
        gamma = 1
        g = N2vGraph(self.graph, p, q, gamma,doxn2v=True)
        #src = self.graph.get_node_index('g1')
        #dst = self.graph.get_node_index('g2')
        src = 'g1'
        dst = 'g2'
        edge = (src, dst)
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        #orig_edge, [j_alias, q_alias] = g.get_alias_edge(src, dst)
        #self.assertEqual(orig_edge, edge,
                        # "get_alias_edge didn't send back the original edge")
        self.assertEqual(len(j_alias), len(q_alias))
        # there are 4 outgoing edges from g2: g1, g3, p1, p2
        self.assertEqual(4, len(j_alias))
        # recreate the original probabilities. They should be 0.125, 0.125, 0.5, 0.25
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(0.125, original_probs[0])
        self.assertAlmostEqual(0.125, original_probs[1])
        self.assertAlmostEqual(0.5, original_probs[2])
        self.assertAlmostEqual(0.25, original_probs[3])

    def test_raw_probs_simple_graph_2(self):
        p = 1
        q = 1
        gamma = 1
        g = N2vGraph(self.graph, p, q, gamma, doxn2v=True)
        # src = self.graph.get_node_index('g0')
        # dst = self.graph.get_node_index('g1')
        src = 'g1'
        dst = 'g4'
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # outgoing edges from g4: g1, g3, p4, d2
        self.assertEqual(4, len(j_alias))
        # recreate the original probabilities. They should be a vector of length 4.
        original_probs = calculate_total_probs(j_alias, q_alias)
        #self.assertAlmostEqual(1.0/4.0, original_probs[1])


class TestHetGraph(TestCase):

    def setUp(self):
        inputfile = os.path.join(os.path.dirname(__file__), 'data', 'small_het_graph.txt')
        g = CSFGraph(inputfile)
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
        n = g.node_count()
        self.assertEqual(151, n)

    def testGraphEdgeCounts(self):
        """
         # We expect 100 g<->g edges, 1 g<->d edge, 19 d<->d edges, and 29 p<->p edges
         # Note that currently we have 2 directed edges for each undirected edge. This
         # means that edge_count() returns 300. This is an implementation detail that
         # may change in the future.
        :return:
        """
        g = self.graph
        m = g.edge_count()
        self.assertEqual(300, m)

    def test_raw_probs_1(self):
        p = 1
        q = 1
        gamma = 1
        g = N2vGraph(self.graph, p, q, gamma, doxn2v=True)
        #src = self.graph.get_node_index('g0')
        #dst = self.graph.get_node_index('g1')
        src = 'g1'
        dst = 'g2'
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # outgoing edges from g2: g1
        self.assertEqual(1, len(j_alias))
        # recreate the original probabilities. They should be a vector of length 1 with value 1.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(1.0, original_probs[0])


    def test_raw_probs_2(self):
        p = 1
        q = 1
        gamma = 1
        g = N2vGraph(self.graph, p, q, gamma, doxn2v=True)
        #src = self.graph.get_node_index('g0')
        #dst = self.graph.get_node_index('g1')
        src = 'g1'
        dst = 'p1'
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # outgoing edges from p1: g1, p2, ..., p30
        self.assertEqual(30, len(j_alias))
        # recreate the original probabilities. They should be a vector of length 1 with value 1.
        original_probs = calculate_total_probs(j_alias, q_alias)
        #self.assertAlmostEqual(1.0/30.0, original_probs[1])


