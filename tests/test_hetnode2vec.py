from unittest import TestCase, skip

import os.path
from xn2v import CSFGraph
from xn2v import N2vGraph
from tests.utils.utils import calculate_total_probs


class TestSimpleHetGraph(TestCase):

    def setUp(self):
        inputfile = os.path.join(os.path.dirname(__file__), 'data', 'small_graph.txt')
        g = CSFGraph(inputfile)
        self.graph = g
        self.p = 1
        self.q = 1
        self.gamma = 1
        self.n2v_walk = N2vGraph(self.graph, self.p, self.q, self.gamma, doxn2v=True)

    def test_j_alias_q_alias_length(self):
        """Test that j_alias and q_alias lengths are equal, and the correct length

        destination node is g2, which has 4 neighbors: g1, g3, p1, p2, so *_alias
        should both be length 4
        :return:
        """
        src = 'g1'
        dst = 'g2'
        [j_alias, q_alias] = self.n2v_walk.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        self.assertEqual(4, len(j_alias))

    def test_raw_probs_simple_graph_1(self):
        src = 'g1'
        dst = 'g2'
        [j_alias, q_alias] = self.n2v_walk.get_alias_edge_xn2v(src, dst)
        # recreate the original probabilities. They should be 0.125, 0.125, 0.5, 0.25
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(0.125, original_probs[0])
        self.assertAlmostEqual(0.125, original_probs[1])
        self.assertAlmostEqual(0.5, original_probs[2])
        self.assertAlmostEqual(0.25, original_probs[3])

    @skip
    def test_raw_probs_simple_graph_2(self):# This test need to be checked.
        src = 'g1'
        dst = 'g4'
        [j_alias, q_alias] = self.n2v_walk.get_alias_edge_xn2v(src, dst)
        self.assertEqual(len(j_alias), len(q_alias))
        # outgoing edges from g4: g1, g3, p4, d2
        self.assertEqual(4, len(j_alias))
        # recreate the original probabilities. They should be a vector of length 4.
        original_probs = calculate_total_probs(j_alias, q_alias) #some original probs are not positive!!
        self.assertAlmostEqual(1.0/4.0, original_probs[1]) #Check probability!


class TestComplexHetGraph(TestCase):

    def setUp(self):
        inputfile = os.path.join(os.path.dirname(__file__), 'data', 'small_het_graph.txt')
        self.graph = CSFGraph(inputfile)
        self.nodes = self.graph.nodes()
        self.g1index = self.__get_index('g1')
        self.d1index = self.__get_index('d1')
        p = 1
        q = 1
        gamma = 1
        self.n2v_walk_default_params = N2vGraph(self.graph, p, q, gamma, doxn2v=True)

    def __get_index(self, node):
        if node in self.nodes:
            return self.nodes.index(node)
        else:
            raise Exception("Could not find {} in self.nodes".format(node))

    def _get_index_of_neighbor(self, node, neighbor):
        """
        Searches for the index of a neighbor of node
        If there is no neighbor, return None
        :param node:
        :param neighbor:
        :return:
        """
        neighbors = sorted(self.graph.neighbors(node))

        if neighbor in neighbors:
            return neighbors.index(neighbor)
        else:
            return None

    def testGraphNodeCounts(self):
        """
        #We expect 101 gene nodes, 20 disease nodes, and 30 protein nodes

        :return:
        """
        self.assertEqual(151,  self.graph.node_count())

    def testGraphEdgeCounts(self):
        """
         # We expect 100 g<->g edges, 1 g<->d edge, 19 d<->d edges, and 29 p<->p edges
         # Note that currently we have 2 directed edges for each undirected edge. This
         # means that edge_count() returns 300. This is an implementation detail that
         # may change in the future.
        :return:
        """
        self.assertEqual(300, self.graph.edge_count())

    def test_raw_probs_1(self):
        """The intention here (I think) is to test aliases and probabilities for
        nodes that are only connected to one other node. Here that's g2 which is only
        connected to g1.

        j_alias, q_alias, and original probabilities should be len(1), and
        original probabilities should have one entry which is AlmostEqual to 1.0
        """
        src = 'g1'
        dst = 'g2'
        [j_alias, q_alias] = self.n2v_walk_default_params.get_alias_edge_xn2v(src, dst)
        # recreate the original probabilities. They should be a vector of length 1 with value 1.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(1.0, original_probs[0])

    @skip
    def test_raw_probs_2(self):# This test needs to be checked.
        """The intention here is to test aliases and probabilities for
        a node (p1) that is connected to 29 other protein nodes (p2 - 30)
        and one gene node (g1)

        j_alias, and q_alias, and original probabilities should be len(30). g1 should
        have one probability, all the p* nodes should have another
        """
        src = 'g1'
        dst = 'p1'
        [j_alias, q_alias] = self.n2v_walk_default_params.get_alias_edge_xn2v(src, dst)
        # recreate the original probabilities.
        original_probs = calculate_total_probs(j_alias, q_alias)
        self.assertAlmostEqual(1.0/30.0, original_probs[1])#check the probability

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
        src_as_int = self.graph.node_to_index_map.get(src)
        dst_as_int = self.graph.node_to_index_map.get(dst)
        g0g1tuple = (src_as_int, dst_as_int)  # this is a key of the dictionary alias_edge_tuple
        alias_edge_tuple = g.retrieve_alias_edges()
        g0g1edges = alias_edge_tuple.get(g0g1tuple)
        # g0g1 edges has the alias map and probabilities as a 2-tuple
        # The following code searches for the index of d1
        # We want the probability of going from g1 to d1 -- it should be equal to gamma, because there is only one g1 to disease edge
        idx = self._get_index_of_neighbor('g1', 'd1')
        self.assertIsNotNone(idx)
        probs = calculate_total_probs(g0g1edges[0], g0g1edges[1])
        d1prob = probs[idx]  # total probability of going from g1 to d1
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
        src_as_int = self.graph.node_to_index_map.get(src)
        dst_as_int = self.graph.node_to_index_map.get(dst)
        g1g2tuple = (src_as_int, dst_as_int)  # this is a key of the dictionary alias_edge_tuple
        alias_edge_tuple = g.retrieve_alias_edges()
        g1g2edges = alias_edge_tuple.get(g1g2tuple)
        # g1g2 edges has the alias map and probabilities as a 2-tuple
        # The following code searches for the index of g1
        # We want the probability of going from g2 to g1 -- it should be equal to 1 because there is only one neighbor to g2
        # and it is in the same network
        idx = self._get_index_of_neighbor('g2', 'g1')
        self.assertIsNotNone(idx)
        probs = calculate_total_probs(g1g2edges[0], g1g2edges[1])
        g1prob = probs[idx]  # total probability of going from g2 to g1
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
        src_as_int = self.graph.node_to_index_map.get(src)
        dst_as_int = self.graph.node_to_index_map.get(dst)
        g1p1tuple = (src_as_int, dst_as_int)  # this is a key of the dictionary alias_edge_tuple
        alias_edge_tuple = g.retrieve_alias_edges()
        g1p1edges = alias_edge_tuple.get(g1p1tuple)
        # g1p1 edges has the alias map and probabilities as a 2-tuple
        # The following code searches for the index of p2
        # We want the probability of going from p1 to p2 -- it should be equal
        idx = self._get_index_of_neighbor('p1', 'g1')
        self.assertIsNotNone(idx)
        probs = calculate_total_probs(g1p1edges[0], g1p1edges[1])
        g1prob = probs[idx]  # total probability of going from p1 to g1.
        # p1 has g1 as a neighbor in different network and has 29 protein neighbors in the same network.
        # The probability of going back to g1 is gamma which is 1/3
        self.assertAlmostEqual(1.0 / 3.0, g1prob)

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
        src_as_int = self.graph.node_to_index_map.get(src)
        dst_as_int = self.graph.node_to_index_map.get(dst)
        g1p1tuple = (src_as_int, dst_as_int)  # this is a key of the dictionary alias_edge_tuple
        alias_edge_tuple = g.retrieve_alias_edges()
        g1p1edges = alias_edge_tuple.get(g1p1tuple)
        # g1p1 edges has the alias map and probabilities as a 2-tuple
        # The following code searches for the index of p2
        # We want the probability of going from p1 to p2 -- it should be equal
        idx = self._get_index_of_neighbor('p1', 'p2')
        self.assertIsNotNone(idx)
        probs = calculate_total_probs(g1p1edges[0], g1p1edges[1])
        p2prob = probs[idx]  # total probability of going from p1 to p2.
        # p1 has g1 as a neighbor in different network and has 29 protein neighbors in the same network.
        # The probability of going  to p2 is gamma which is (2/3*29) ~ 0.023
        self.assertAlmostEqual(2.0 / 87.0, p2prob)
