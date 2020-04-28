from unittest import TestCase
import os.path
from xn2v import CSFGraph
from xn2v.csf_graph.csf_graph import CSFGraphNoSubjectColumnError, \
    CSFGraphNoObjectColumnError


class TestCSFGraph(TestCase):
    def setUp(self):
        data_dir = os.path.join(os.path.dirname( __file__), 'data')
        self.legacy_edge_file = os.path.join(data_dir, 'small_graph_LEGACY.txt')
        self.tsv_no_subject = os.path.join(data_dir, 'small_graph_edges_NO_SUBJECT.tsv')
        self.tsv_no_object = os.path.join(data_dir, 'small_graph_edges_NO_OBJECT.tsv')
        self.edge_file = os.path.join(data_dir, 'small_graph_edges.tsv')
        self.node_file = os.path.join(data_dir, 'small_graph_nodes.tsv')
        g = CSFGraph(edge_file=self.edge_file)
        self.g = g
        str(g)

    def test_notnull(self):
        self.assertIsNotNone(self.g)

    #
    # check maps
    #
    def test_get_node_to_index_map(self):
        self.assertIsNotNone(self.g.get_node_to_index_map())

    def test_get_index_to_node_map(self):
        self.assertIsNotNone(self.g.get_index_to_node_map())

    # check nodetype to index map
    def test_csfgraph_constructor_makes_nodetype_to_index_map(self):
        self.assertIsInstance(self.g.nodetype_to_index_map, dict)

    # check index to nodetype map
    def test_csfgraph_constructor_makes_index_to_nodetype_map(self):
        self.assertIsInstance(self.g.index_to_nodetype_map, dict)

    # check edgetype to index map
    def test_csfgraph_constructor_makes_edgetype_to_index_map(self):
        self.assertIsInstance(self.g.edgetype_to_index_map, dict)

    # check index to edgetype map
    def test_csfgraph_constructor_makes_index_to_edgetype_map(self):
        self.assertIsInstance(self.g.index_to_edgetype_map, dict)

    def test_csfgraph_constructor_requires_arg(self):
        with self.assertRaises(Exception) as context:
            CSFGraph()  # missing edge arg
            self.assertTrue(str('missing' in context.exception))

    def test_csfgraph_constructor_check_for_subject_columns(self):
        with self.assertRaises(CSFGraphNoSubjectColumnError) as context:
            CSFGraph(edge_file=self.tsv_no_subject)  # file doesn't have subject col

    def test_csfgraph_constructor_check_for_object_column(self):
        with self.assertRaises(CSFGraphNoObjectColumnError) as context:
            CSFGraph(edge_file=self.tsv_no_object)  # file doesn't have object col

    def test_csfgraph_constructor_accepts_edge_file(self):
        g = CSFGraph(edge_file=self.edge_file)

    def test_csfgraph_constructor_accepts_node_file(self):
        g = CSFGraph(edge_file=self.edge_file, node_file=self.node_file)

    def test_count_nodes_legacy_edge_file(self):
        g = CSFGraph(edge_file=self.legacy_edge_file)
        self.assertEqual(3, g.node_count())

    def test_count_edges_legacy_edge_file(self):
        g = CSFGraph(edge_file=self.legacy_edge_file)
        self.assertEqual(6, g.edge_count())

    def test_count_nodes(self):
        # The graph in small_graph.txt has 11 nodes
        expected_num = 11
        self.assertEqual(expected_num, self.g.node_count())

    def test_count_edges(self):
        # Note that the graph is transformed into an undirected graph by
        # adding the inverse of each edge
        # 21 edges are included in the file, thus we expect 2*21=42 edges
        expected_num = 42
        self.assertEqual(expected_num, self.g.edge_count())

        return None

    def test_get_weight(self):
        expected = 10  # for d1<->d3
        weight = self.g.weight('d1', 'd3')
        self.assertEqual(expected, weight)
        expected = 10  # for g3<->g4
        weight = self.g.weight('g3', 'g4')
        self.assertEqual(expected, weight)

        return None

    def test_get_neighbors1(self):
        # the neighbors of p4 are g4, p2, p3
        nbrs = self.g.neighbors('p4')
        # print(nbrs)
        self.assertEqual(['g4', 'p2', 'p3'], nbrs)

        return None

    def test_get_neighbors_as_ints_1(self):
        # The neighbors of p4 are g4, p2, p3 and their indices are 6, 8, 9
        p4_idx = self.g.node_to_index_map['p4']
        nbrs = self.g.neighbors_as_ints(p4_idx)
        # print("index of g4 = {}, index of p2 = {}, index of p3 = {}".format(self.g.node_to_index_map['g4'],
        # self.g.node_to_index_map['p2'],self.g.node_to_index_map['p3']))
        self.assertEqual([6, 8, 9], nbrs)

    def test_get_neighbors2(self):
        # the neighbors of g2 are g1, g3, p1, p2
        nbrs = self.g.neighbors('g2')
        # print(nbrs)
        self.assertEqual(['g1', 'g3', 'p1', 'p2'], nbrs)

        return None

    def test_get_neighbors_as_ints_2(self):
        g2_idx = self.g.node_to_index_map['g2']
        # the neighbors of g2 are g1, g3, p1, p2 and their indices are 3, 5, 7, 8
        nbrs = self.g.neighbors_as_ints(g2_idx)
        # print("index of g1 = {}, index of g3 = {}, index of p1 = {}, index of p2 = {}".
        # format(self.g.node_to_index_map['g1'], self.g.node_to_index_map['g3'], self.g.node_to_index_map['p1'],
        # self.g.node_to_index_map['p2']))
        self.assertEqual([3, 5, 7, 8], nbrs)

        return None

    def test_check_nodetype(self):
        self.assertTrue(self.g.same_nodetype('g1', 'g2'))
        self.assertFalse(self.g.same_nodetype('g1', 'p3'))

        return None

    def test_edges(self):
        """Test the method that should get all edges as tuples wc -l small_graph.txt reveals 21 edges, but the Graph
        class creates a total of 42 edges to make an undirected graph.
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

    def test_degree(self):
        #test the degrees of nodes
        node_1 = "g1"
        node_2 = "p2"
        node_3 = "d3"
        self.assertEqual(5, self.g.node_degree(node_1))
        self.assertEqual(4, self.g.node_degree(node_2))
        self.assertEqual(3, self.g.node_degree(node_3))

