import os.path

from unittest import TestCase

from xn2v import CSFGraph


class TestCSFGraph(TestCase):
    def setUp(self):
        input_file = os.path.join(os.path.dirname(__file__), 'data', 'unweighted_small_graph.txt')
        g = CSFGraph(input_file)
        self.g = g

        return str(g)

    def test_notnull(self):
        self.assertIsNotNone(self.g)

        return None

    def test_get_node_to_index_map(self):
        self.g.get_node_to_index_map()

        return None

    def test_get_index_to_node_map(self):
        self.g.get_index_to_node_map()

        return None

    def test_print_edge_type_distribution(self):
        self.g.print_edge_type_distribution()

        return None

    def test_count_nodes(self):
        # the graph in unweighted_small_graph.txt has 11 nodes
        expected_num = 11
        self.assertEqual(expected_num, self.g.node_count())

        return None

    def test_count_edges(self):
        # note that the graph is transformed into an undirected graph by adding the inverse of each edge
        # 21 edges are included in the file, thus we expect 2*21=42 edges
        expected_num = 42
        self.assertEqual(expected_num, self.g.edge_count())

        return None

    def test_get_weight(self):
        # for d1<->d3
        expected = 1
        weight = self.g.weight('d1', 'd3')
        self.assertEqual(expected, weight)

        # for g3<->g4
        expected = 1
        weight = self.g.weight('g3', 'g4')
        self.assertEqual(expected, weight)

        return None

    def test_get_neighbors1(self):
        # the neighbors of p4 are g4, p2, p3
        nbrs = self.g.neighbors('p4')
        # print(nbrs)
        self.assertEqual(['g4', 'p2', 'p3'], nbrs)

        return None

    def test_nodes_as_integers(self):
        self.g.nodes_as_integers()

        return None

    def test_get_neighbors2(self):
        # the neighbors of g2 are g1, g3, p1, p2
        nbrs = self.g.neighbors('g2')
        # print(nbrs)
        self.assertEqual(['g1', 'g3', 'p1', 'p2'], nbrs)

        return None

    def test_check_nodetype(self):
        self.assertTrue(self.g.same_nodetype('g1', 'g2'))
        self.assertFalse(self.g.same_nodetype('g1', 'p3'))

        return None

    def test_edges(self):
        """Test the method that should get all edges as tuples wc -l unweighted_small_graph.txt reveals 21 edges,
        but the Graph class creates a total of 42 edges to make an undirected graph.
        """

        edge_list = self.g.edges()
        self.assertEqual(42, len(edge_list))

        # p1 p3 and p3 p1 are valid edges
        t1 = ('p1', 'p3')
        self.assertTrue(t1 in edge_list)

        t2 = ('p3', 'p1')
        self.assertTrue(t2 in edge_list)

        made_up = ('z1', 'q123')
        self.assertFalse(made_up in edge_list)

        return None
