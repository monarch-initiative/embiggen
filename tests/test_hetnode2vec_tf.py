from unittest import TestCase

import os.path
from xn2v import CSFGraph
from xn2v.hetnode2vec import N2vGraph
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
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # there are 4 outgoing edges from g2: g1, g3, p1, p2
        self.assertEqual(4, len(j_alias))
        # recreate the original probabilities. They should be 0.25, 0.25, 0.3333, 0.1666
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(0.25, original_probs[0])#prob from g2 to g1: (1-1/2)/2 = 1/4
        self.assertAlmostEqual(0.25, original_probs[1])#prob from g2 to g3: (1-1/2)/2 = 1/4
        self.assertAlmostEqual(1.0/3.0, original_probs[2])#prob from g2 to p1: ((1/2)/2)*20 = 5 --> 5/7.5 = 1/3 (weight of edge:20)
        self.assertAlmostEqual(1.0/6.0, original_probs[3])#prob from g2 to p2: ((1/2)/2)*10 = 2.5 --> 2.5/7.5 = 1/6 (weight of edge:10)

    def test_raw_probs_simple_graph_2(self):
        p = 1
        q = 1
        gamma = 1
        g = N2vGraph(self.graph, p, q, gamma, doxn2v=True)
        src = 'g1'
        dst = 'g4'
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # outgoing edges from g4: d2, g1, g3, p4
        self.assertEqual(4, len(j_alias))
        # recreate the original probabilities. They should be a vector of length 4.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(1.0/3.0, original_probs[0])#prob from g4 to d2: gamma/number of node types of neighbors of g4
        # = 1/3
        self.assertAlmostEqual(1.0/6.0, original_probs[1])# prob from g4 to g1: (1-2/3)/2 = 1/6
        self.assertAlmostEqual(1.0/6.0, original_probs[2])#prob from g4 to g3: (1-2/3)/2 = 1/6
        self.assertAlmostEqual(1.0/3.0, original_probs[3])#prob from g4 to p4: 1/3

    def test_raw_probs_simple_graph_3(self):
        p = 1
        q = 1
        gamma = 1
        g = N2vGraph(self.graph, p, q, gamma, doxn2v=True)
        src = 'g2'
        dst = 'p2'
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # outgoing edges from p2: g1, p1, p3, p4
        self.assertEqual(4, len(j_alias))
        # recreate the original probabilities. They should be a vector of length 4.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(1.0 / 2.0, original_probs[0]) #prob from p2 to g1: 1/2
        self.assertAlmostEqual(1.0 / 6.0, original_probs[1]) #prob from p2 to p1: (1-1/2)/3 = 1/6
        self.assertAlmostEqual(1.0 / 6.0, original_probs[2])#prob from p2 to p3: (1-1/2)/3 = 1/6
        self.assertAlmostEqual(1.0 / 6.0, original_probs[3])#prob from p2 to p4: (1-1/2)/3 = 1/6

    def test_raw_probs_simple_graph_4(self):
        p = 1
        q = 1
        gamma = 1.0/8.0
        g = N2vGraph(self.graph, p, q, gamma, doxn2v=True)
        src = 'g2'
        dst = 'p2'
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # outgoing edges from p2: g1, p1, p3, p4
        self.assertEqual(4, len(j_alias))
        # recreate the original probabilities. They should be a vector of length 4.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(1.0 / 16.0, original_probs[0]) #prob from p2 to g1: gamma/2 = (1/8)/2 = 1/16
        self.assertAlmostEqual(15.0 / 48.0, original_probs[1])#prob from p2 to p1: (1-gamma/2)/3 = (15/16)/3 = 15/48
        self.assertAlmostEqual(15.0 / 48.0, original_probs[2])#prob from p2 to p3: (1-gamma/2)/3 = (15/16)/3 = 15/48
        self.assertAlmostEqual(15.0 / 48.0, original_probs[3])#prob from p2 to p4: (1-gamma/2)/3 = (15/16)/3 = 15/48

    def test_raw_probs_simple_graph_5(self):
        p = 1
        q = 1
        gamma = 1.0 / 8.0
        g = N2vGraph(self.graph, p, q, gamma, doxn2v=True)
        src = 'g1'
        dst = 'g4'
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # outgoing edges from g4: d2, g1, g3, p4
        self.assertEqual(4, len(j_alias))
        # recreate the original probabilities. They should be a vector of length 4.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(1.0 / 24.0, original_probs[0])#prob from g4 to d2: gamma/3 = 1/24
        self.assertAlmostEqual(11.0 / 24.0, original_probs[1])#prob from g4 to g1: (1- 2*gamma/3)/2 = 11/24
        self.assertAlmostEqual(11.0 / 24.0, original_probs[2])#prob from g4 to g3: (1- 2*gamma/3)/2 = 11/24
        self.assertAlmostEqual(1.0 / 24.0, original_probs[3])#prob from g4 to p4: gamma/3 = 1/24

    def test_raw_probs_simple_graph_6(self):
        p = 1
        q = 1
        gamma = 7.0 / 8.0
        g = N2vGraph(self.graph, p, q, gamma, doxn2v=True)
        src = 'g1'
        dst = 'g4'
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # outgoing edges from g4: d2, g1, g3, p4
        self.assertEqual(4, len(j_alias))
        # recreate the original probabilities. They should be a vector of length 4.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(7.0 / 24.0, original_probs[0]) #prob from g4 to d2: gamma/3 = 7/24
        self.assertAlmostEqual(10.0 / 48.0, original_probs[1])#prob from g4 to g1: (1-2gamma/3)/2 = 10/48
        self.assertAlmostEqual(10.0 / 48.0, original_probs[2])#prob from g4 to g3: (1-2gamma/3)/2 = 10/48
        self.assertAlmostEqual(7.0 / 24.0, original_probs[3])#prob from g4 to p4: gamma/3 = 7/24


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
        src = 'g1'
        dst = 'g2'
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # outgoing edges from g2: g1
        self.assertEqual(1, len(j_alias))
        # recreate the original probabilities. They should be a vector of length 1 with value 1.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(1.0, original_probs[0])



    def test_raw_probs_3(self):
        p = 1
        q = 1
        gamma = 1
        g = N2vGraph(self.graph, p, q, gamma, doxn2v=True)
        src = 'g1'
        dst = 'p1'
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # outgoing edges from p1: g1, p2, ..., p30
        self.assertEqual(30, len(j_alias))
        # recreate the original probabilities. They should be a vector of length 1 with value 1.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(1.0 / 2, original_probs[0])
        self.assertAlmostEqual(1.0/58.0, original_probs[1])

    def test_raw_probs_4(self):
        p = 1
        q = 1
        gamma = 1
        g = N2vGraph(self.graph, p, q, gamma, doxn2v=True)
        src = 'g1'
        dst = 'd1'
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # outgoing edges from d1: d2, ..., d20, g1
        self.assertEqual(20, len(j_alias))
        # recreate the original probabilities.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(1.0 / 38, original_probs[0])
        self.assertAlmostEqual(1.0 / 2, original_probs[19])

    def test_raw_probs_5(self):
        p = 1
        q = 1
        gamma = 1.0/3.0
        g = N2vGraph(self.graph, p, q, gamma, doxn2v=True)
        src = 'g1'
        dst = 'p1'
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # outgoing edges from p1: g1, p2, ..., p30
        self.assertEqual(30, len(j_alias))
        # recreate the original probabilities.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(1.0 / 6.0, original_probs[0])
        self.assertAlmostEqual(5/174.0, original_probs[25])
        self.assertAlmostEqual(5/174.0, original_probs[29])

    def test_raw_probs_6(self):
        p = 1
        q = 1
        gamma = 1
        g = N2vGraph(self.graph, p, q, gamma, doxn2v=True)
        src = 'd1'
        dst = 'g1'
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # outgoing edges from g1: d1, g0, g2, ...., g100, p1
        self.assertEqual(102, len(j_alias))
        # recreate the original probabilities. They should be a vector of length 104.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(1.0 / 3.0, original_probs[0]) #prob from g1 to d1
        self.assertAlmostEqual(1.0 / 300.0, original_probs[1])#prob from g1 to g0
        self.assertAlmostEqual(1.0 / 300.0, original_probs[30])#prob from g1 to g30
        self.assertAlmostEqual(1.0 / 3.0, original_probs[101])#prob from g1 to p1

    def test_raw_probs_7(self):
        p = 1
        q = 1
        gamma = 1.0/2.0
        g = N2vGraph(self.graph, p, q, gamma, doxn2v=True)
        src = 'd1'
        dst = 'g1'
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # outgoing edges from g1: d1, g0, g2, ...., g100, p1
        self.assertEqual(102, len(j_alias))
        # recreate the original probabilities. They should be a vector of length 104.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(1.0 / 6.0, original_probs[0]) #prob from g1 to d1
        self.assertAlmostEqual(2.0/300.0, original_probs[1])#prob from g1 to g0
        self.assertAlmostEqual(2.0 / 300.0, original_probs[30])#prob from g1 to g30
        self.assertAlmostEqual(1.0 / 6.0, original_probs[101])#prob from g1 to p1

    def test_raw_probs_8(self):
        p = 1
        q = 1
        gamma = 1.0/2.0
        g = N2vGraph(self.graph, p, q, gamma, doxn2v=True)
        src = 'd2'
        dst = 'd1'
        [j_alias, q_alias] = g.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # outgoing edges from d1: d2, ...., d20, g1
        self.assertEqual(20, len(j_alias))
        # recreate the original probabilities. They should be a vector of length 104.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(3.0 / 76.0, original_probs[0]) #prob from d1 to d2
        self.assertAlmostEqual(3.0/ 76.0, original_probs[1])#prob from d1 to d3
        self.assertAlmostEqual(3.0 / 76.0, original_probs[10])#prob from d1 to d12
        self.assertAlmostEqual(1.0 / 4.0, original_probs[19])#prob from d1 to g1


class TestHetGraph2(TestCase):

    def setUp(self):
        infile = os.path.join(os.path.dirname(__file__), 'data', 'small_het_graph.txt')
        g = CSFGraph(infile)
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
        Test that our rebalancing methods cause us to go from g1 to d1 with a probability of gamma/k
        where k=3, i.e., the number of different node types
        :return:
        """
        p = 1
        q = 1
        gamma = float(1) / float(3)
        g = N2vGraph(self.graph, p, q, gamma, True)
        src = 'g0'
        dst = 'g1'
        g0g1tuple = (src, dst)  # this is a key of the dictionary alias_edge_tuple
        alias_edge_tuple = g.retrieve_alias_edges()
        g0g1edges = alias_edge_tuple.get(g0g1tuple)
        # g0g1 edges has the alias map and probabilities as a 2-tuple
        # The following code searches for the index of d1
        # We want the probability of going from g1 to d1 -- it should be equal to gamma, because there is only one g1 to disease edge
        idx = self.get_index_of_neighbor('g1', 'd1')
        self.assertIsNotNone(idx)
        probs = calculate_total_probs(g0g1edges[0], g0g1edges[1])
        d1prob = probs[idx]  # total probability of going from g1 to d1 is (gamma/3) = 1/9, because d1 is in different network
        #and is the only neighbor of g1 in this network. g1 is connected to 3 nodetypes.
        self.assertAlmostEqual(1.0 / 9.0, d1prob)

    def test_gamma_probs_2(self):
        """
        Test that our rebalancing methods cause us to go from g2 to g1 with a probability 1
        :return:
        """
        p = 1
        q = 1
        gamma = float(1) / float(3)
        g = N2vGraph(self.graph, p, q, gamma, True)
        src = 'g1'
        dst = 'g2'
        g1g2tuple = (src, dst)  # this is a key of the dictionary alias_edge_tuple
        alias_edge_tuple = g.retrieve_alias_edges()
        g1g2edges = alias_edge_tuple.get(g1g2tuple)
        # g1g2 edges has the alias map and probabilities as a 2-tuple
        # The following code searches for the index of g1
        # We want the probability of going from g2 to g1 -- it should be equal to 1 because there is only one neighbor to g2
        # and it is in the same network
        idx = self.get_index_of_neighbor('g2', 'g1')
        self.assertIsNotNone(idx)
        probs = calculate_total_probs(g1g2edges[0], g1g2edges[1])
        g1prob = probs[idx]  # total probability of going from g2 to g1
        self.assertAlmostEqual(1.0, g1prob)

    def test_gamma_probs_3(self):
        """
        Test that our rebalancing methods cause us to go from p1 to g1 with a probability of gamma/ k
        where k=2, i.e., the number of different node types
        :return:
        """
        p = 1
        q = 1
        gamma = float(1) / float(3)
        g = N2vGraph(self.graph, p, q, gamma, True)
        src = 'g1'
        dst = 'p1'
        g1p1tuple = (src, dst)  # this is a key of the dictionary alias_edge_tuple
        alias_edge_tuple = g.retrieve_alias_edges()
        g1p1edges = alias_edge_tuple.get(g1p1tuple)
        # g1p1 edges has the alias map and probabilities as a 2-tuple
        # The following code searches for the index of g1
        # We want the probability of going from p1 to g1 -- it should be equal
        idx = self.get_index_of_neighbor('p1', 'g1')
        self.assertIsNotNone(idx)
        probs = calculate_total_probs(g1p1edges[0], g1p1edges[1])
        g1prob = probs[idx]  # total probability of going from p1 to g1.
        ## p1 has g1 as a neighbor in different network and has 29 protein neighbors in the same network.
        # The probability of going back to g1 is gamma/2 which is (1/3)/2 = 1/6 because p1 is connected to 2 networks
        # (gens and proteins). p1 is the only neighbor of g1 in proteins.
        self.assertAlmostEqual(1.0 / 6.0, g1prob)

    def test_gamma_probs_4(self):
        """
        Assume we have traversed the g1-p1 edge.
        Test that our rebalancing methods cause us to go from p1 to p2 with a probability (1 - gamma/k)/n where
        where k=2, i.e., the number of different node types connected to p1 and n=29 is the number of nodes in protein set ]
        #connected to p1.
        :return:
        """
        p = 1
        q = 1
        gamma = float(1) / float(3)
        g = N2vGraph(self.graph, p, q, gamma, True)
        src = 'g1'
        dst = 'p1'
        g1p1tuple = (src, dst)  # this is a key of the dictionary alias_edge_tuple
        alias_edge_tuple = g.retrieve_alias_edges()
        g1p1edges = alias_edge_tuple.get(g1p1tuple)
        # g1p1 edges has the alias map and probabilities as a 2-tuple
        # The following code searches for the index of p2
        # We want the probability of going from p1 to p2 --
        idx = self.get_index_of_neighbor('p1', 'p2')
        self.assertIsNotNone(idx)
        probs = calculate_total_probs(g1p1edges[0], g1p1edges[1])
        p2prob = probs[idx]  # total probability of going from p1 to p2.
        # p1 has g1 as a neighbor in different network and has 29 protein neighbors in the same network.
        # The probability of going  to p2 is (1-gamma/2)/29 which is (5/(6*29))
        self.assertAlmostEqual(5.0 / 174.0, p2prob)
