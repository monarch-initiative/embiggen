import os.path

from unittest import TestCase

from xn2v.csf_graph.csf_graph_tf import CSFGraph, CSFGraphNoSubjectColumnError, \
    CSFGraphNoObjectColumnError
from tests.utils.utils import gets_tensor_length, searches_tensor


class TestCSFGraph(TestCase):

    def setUp(self):
        data_dir = os.path.join(os.path.dirname( __file__), 'data')

        self.edge_file = os.path.join(data_dir, 'small_graph_edges.tsv')
        self.node_file = os.path.join(data_dir, 'small_graph_nodes.tsv')

        # # legacy and non-standard test files
        self.tsv_no_subject = os.path.join(data_dir, 'small_graph_edges_NO_SUBJECT.tsv')
        self.tsv_no_object = os.path.join(data_dir, 'small_graph_edges_NO_OBJECT.tsv')
        self.legacy_edge_file = os.path.join(data_dir, 'small_graph_LEGACY.txt')
        self.node_file_missing_nodes = os.path.join(data_dir, 'small_graph_nodes_MISSING_NODES.tsv')

        self.graph = CSFGraph(self.edge_file)

        return None

    def test_notnull(self):

        self.assertIsNotNone(self.graph)

    def test_get_node_to_index_map(self):

        self.assertIsNotNone(self.graph.get_node_to_index_map())

    def test_get_index_to_node_map(self):

        self.assertIsNotNone(self.graph.get_index_to_node_map())

    def test_count_nodes_legacy_edge_file(self):
        g = CSFGraph(edge_file=self.legacy_edge_file)
        self.assertEqual(3, g.node_count())

    def test_csfgraph_checks_for_subject_columns(self):
        with self.assertRaises(CSFGraphNoSubjectColumnError) as context:
            CSFGraph(edge_file=self.tsv_no_subject)  # file doesn't have subject col

    def test_csfgraph_checks_for_object_column(self):
        with self.assertRaises(CSFGraphNoObjectColumnError) as context:
            CSFGraph(edge_file=self.tsv_no_object)  # file doesn't have object col

    def test_csfgraph_makes_nodetype_to_index_map(self):
        self.assertIsInstance(self.graph.nodetype_to_index_map, dict)

    def test_csfgraph_assigns_default_nodetype_to_nodetype_to_index_map(self):
        self.assertIsInstance(self.graph.nodetype_to_index_map, dict)
        self.assertEqual(self.graph.nodetype_to_index_map[self.graph.default_node_type],
                         list(range(self.graph.node_count())))

    def test_csfgraph_populates_nodetype_to_index_map(self):
        het_g = CSFGraph(edge_file=self.edge_file, node_file=self.node_file)
        self.assertEqual(het_g.nodetype_to_index_map['biolink:Disease'], [0, 1, 2])

    def test_csfgraph_makes_index_to_nodetype_map(self):
        self.assertIsInstance(self.graph.index_to_nodetype_map, dict)

    def test_csfgraph_populates_index_to_nodetype_map(self):
        het_g = CSFGraph(edge_file=self.edge_file, node_file=self.node_file)
        self.assertEqual(11, len(het_g.index_to_nodetype_map))
        self.assertEqual(het_g.index_to_nodetype_map[0], 'biolink:Disease')

    def test_csfgraph_assigns_default_node_type_to_index_to_nodetype_map(self):
        self.assertEqual(self.graph.index_to_nodetype_map[0], self.graph.default_node_type)

    def test_csfgraph_tolerates_missing_node_info(self):
        het_g = CSFGraph(edge_file=self.edge_file,
                         node_file=self.node_file_missing_nodes)
        self.assertEqual(het_g.index_to_nodetype_map[2], het_g.default_node_type)

    # check edgetype to index map
    def test_csfgraph_makes_edgetype_to_index_map(self):
        self.assertIsInstance(self.graph.edgetype_to_index_map, dict)

    def test_csfgraph_populates_edgetype_to_index_map(self):
        self.assertCountEqual(self.graph.edgetype_to_index_map.keys(),
                              ['biolink:interacts_with',
                               'biolink:molecularly_interacts_with'])
        self.assertEqual(40, len(self.graph.edgetype_to_index_map['biolink:interacts_with']))
        self.assertEqual(2, len(self.graph.edgetype_to_index_map['biolink:molecularly_interacts_with']))

    def test_csfgraph_makes_index_to_edgetype_map(self):
        self.assertIsInstance(self.graph.index_to_edgetype_map, dict)

    def test_csfgraph_populates_index_to_edgetype_map(self):
        self.assertEqual(42, len(self.graph.index_to_edgetype_map))
        self.assertEqual(self.graph.index_to_edgetype_map[0], 'biolink:interacts_with')
        self.assertEqual(self.graph.index_to_edgetype_map[34], 'biolink:molecularly_interacts_with')
        self.assertEqual(self.graph.index_to_edgetype_map[40], 'biolink:molecularly_interacts_with')

    def test_count_nodes(self):

        # the graph in small_graph.txt has 11 nodes
        expected_num = 11
        self.assertEqual(expected_num, self.graph.node_count())

    def test_count_edges(self):

        # the graph is transformed into an undirected graph by adding the inverse of each edge
        # 21 edges are included in the file, thus we expect 2*21=42 edges
        expected_num = 42
        self.assertEqual(expected_num, self.graph.edge_count())

    def test_get_weight(self):

        # for d1<->d3
        expected = 10
        weight = self.graph.weight('d1', 'd3')
        self.assertEqual(expected, weight)

        # for g3<->g4 (weight is 10)
        expected = 10
        weight = self.graph.weight('g3', 'g4')
        self.assertEqual(expected, weight)

        return None

    def test_get_neighbors1(self):

        # the neighbors of p4 are g4, p2, p3
        nbrs = self.graph.neighbors('p4')
        self.assertEqual(['g4', 'p2', 'p3'], nbrs)

        return None

    def test_get_neighbors_as_ints_1(self):

        # The neighbors of p4 are g4, p2, p3 and their indices are 6, 8, 9
        p4_idx = self.graph.node_to_index_map['p4']
        nbrs = self.graph.neighbors_as_ints(p4_idx)
        self.assertEqual([6, 8, 9], nbrs)

    def test_get_neighbors2(self):

        # the neighbors of g2 are g1, g3, p1, p2
        nbrs = self.graph.neighbors('g2')
        self.assertEqual(['g1', 'g3', 'p1', 'p2'], nbrs)

    def test_get_neighbors_as_ints_2(self):

        g2_idx = self.graph.node_to_index_map['g2']

        # the neighbors of g2 are g1, g3, p1, p2 and their indices are 3, 5, 7, 8
        nbrs = self.graph.neighbors_as_ints(g2_idx)
        self.assertEqual([3, 5, 7, 8], nbrs)

    def test_check_nodetype(self):

        self.assertTrue(self.graph.same_nodetype('g1', 'g2'))
        self.assertFalse(self.graph.same_nodetype('g1', 'p3'))

    def test_edges(self):
        """Test the method that should get all edges as tuples wc -l small_graph.txt reveals 21 edges, but the Graph
        class creates a total of 42 edges to make an undirected graph.
        """

        edge_list = self.graph.edges()

        # test length of tensor
        tensor_length = gets_tensor_length(edge_list)
        self.assertEqual(42, tensor_length)

        ###########################
        # EDGE: p4, p3 - valid edge
        edge1 = ('p4', 'p3')
        edge1_search_result = searches_tensor(edge1, edge_list)
        self.assertTrue(edge1_search_result)

        ###########################
        # EDGE: p3, p1 - valid edge
        edge_2 = ('p3', 'p1')
        edge2_search_result = searches_tensor(edge_2, edge_list)
        self.assertTrue(edge2_search_result)

        ###########################
        # EDGE: made up - invalid edge
        made_up = ('z1', 'q123')
        made_up_search_result = searches_tensor(made_up, edge_list)
        self.assertFalse(made_up_search_result)
