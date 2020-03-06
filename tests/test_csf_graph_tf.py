import os.path
import tensorflow as tf

from unittest import TestCase

from xn2v.csf_graph.csf_graph_tf import CSFGraph


class TestCSFGraph(TestCase):

    def setUp(self):

        input_file = os.path.join(os.path.dirname(__file__), 'data', 'small_graph.txt')
        g = CSFGraph(input_file)
        self.g = g

        return str(g)

    def test_notnull(self):

        self.assertIsNotNone(self.g)

    def test_get_node_to_index_map(self):

        self.assertIsNotNone(self.g.get_node_to_index_map())

    def test_get_index_to_node_map(self):

        self.assertIsNotNone(self.g.get_index_to_node_map())

    def test_count_nodes(self):

        # the graph in small_graph.txt has 11 nodes
        expected_num = 11
        self.assertEqual(expected_num, self.g.node_count())

    def test_count_edges(self):

        # the graph is transformed into an undirected graph by adding the inverse of each edge
        # 21 edges are included in the file, thus we expect 2*21=42 edges
        expected_num = 42
        self.assertEqual(expected_num, self.g.edge_count())

    def test_get_weight(self):

        # for d1<->d3
        expected = 10
        weight = self.g.weight('d1', 'd3')
        self.assertEqual(expected, weight)

        # for g3<->g4
        expected = 7
        weight = self.g.weight('g3', 'g4')
        self.assertEqual(expected, weight)

        return None

    def test_get_neighbors1(self):

        # the neighbors of p4 are g4, p2, p3
        nbrs = self.g.neighbors('p4')
        self.assertEqual(['g4', 'p2', 'p3'], nbrs)

        return None

    def test_get_neighbors_as_ints_1(self):

        # The neighbors of p4 are g4, p2, p3 and their indices are 6, 8, 9
        p4_idx = self.g.node_to_index_map['p4']
        nbrs = self.g.neighbors_as_ints(p4_idx)
        self.assertEqual([6, 8, 9], nbrs)

    def test_get_neighbors2(self):

        # the neighbors of g2 are g1, g3, p1, p2
        nbrs = self.g.neighbors('g2')
        self.assertEqual(['g1', 'g3', 'p1', 'p2'], nbrs)

    def test_get_neighbors_as_ints_2(self):

        g2_idx = self.g.node_to_index_map['g2']

        # the neighbors of g2 are g1, g3, p1, p2 and their indices are 3, 5, 7, 8
        nbrs = self.g.neighbors_as_ints(g2_idx)
        self.assertEqual([3, 5, 7, 8], nbrs)

    def test_check_nodetype(self):

        self.assertTrue(self.g.same_nodetype('g1', 'g2'))
        self.assertFalse(self.g.same_nodetype('g1', 'p3'))

    def test_edges(self):
        """Test the method that should get all edges as tuples wc -l small_graph.txt reveals 21 edges, but the Graph
        class creates a total of 42 edges to make an undirected graph.
        """

        edge_list = self.g.edges()

        if isinstance(edge_list, tf.RaggedTensor):
            self.assertEqual(42, edge_list.shape[0])
        else:
            self.assertEqual(42, len([edge for edge in edge_list]))

        ###########################
        # EDGE: p4, p3 - valid edge
        t1 = tf.Variable(('p4', 'p3'), dtype=tf.string)

        # find edge in tensor stack
        t1_edge_search_result = [edge for edge in edge_list if all(x for x in (edge == t1).numpy())]

        # if edge is in tensor stack, length of search results should be 1
        t1_edge_search_result_len = len(t1_edge_search_result)
        self.assertTrue(1 == t1_edge_search_result_len)

        ###########################
        # EDGE: p3, p1 - valid edge
        t2 = tf.Variable(('p3', 'p1'), dtype=tf.string)

        # find edge in tensor stack
        t2_edge_search_result = [edge for edge in edge_list if all(x for x in (edge == t2).numpy())]

        # if edge is in tensor stack, length of search results should be 1
        t2_edge_search_result_len = len(t2_edge_search_result)
        self.assertTrue(1 == t2_edge_search_result_len)

        ###########################
        # EDGE: made up - invalid edge
        made_up = tf.Variable(('z1', 'q123'), dtype=tf.string)

        # find edge in tensor stack
        made_up_edge_search_result = [edge for edge in edge_list if all(x for x in (edge == made_up).numpy())]

        # if edge is in tensor stack, length of search results should be 1
        made_up_edge_search_result_len = len(made_up_edge_search_result)
        self.assertFalse(1 == made_up_edge_search_result_len)
