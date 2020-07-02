from unittest import TestCase
import os.path
from embiggen import CSFGraph
from embiggen import Edge
from embiggen.csf_graph.csf_graph import CSFGraphNoSubjectColumnError, \
    CSFGraphNoObjectColumnError


class TestCSFGraph(TestCase):
    def setUp(self):
        data_dir = os.path.join(os.path.dirname( __file__), 'data')

        # files for canonical test graph
        self.edge_file = os.path.join(data_dir, 'small_graph_edges.tsv')
        self.node_file = os.path.join(data_dir, 'small_graph_nodes.tsv')

        # legacy and non-standard test files
        self.legacy_edge_file = os.path.join(data_dir, 'small_graph_LEGACY.txt')
        self.tsv_no_subject = os.path.join(data_dir, 'small_graph_edges_NO_SUBJECT.tsv')
        self.tsv_no_object = os.path.join(data_dir, 'small_graph_edges_NO_OBJECT.tsv')
        self.node_file_missing_nodes = os.path.join(data_dir, 'small_graph_nodes_MISSING_NODES.tsv')

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

    def test_edgetype2count_dictionary(self):
        self.assertIsInstance(self.g.edgetype2count_dictionary, dict)
        self.assertEqual(self.g.edgetype2count_dictionary['biolink:molecularly_interacts_with'], 1)
        self.assertEqual(self.g.edgetype2count_dictionary['biolink:interacts_with'], 20)

    def test_nodetype2count_dictionary(self):
        het_g = CSFGraph(edge_file=self.edge_file, node_file=self.node_file)
        self.assertIsInstance(self.g.nodetype2count_dictionary, dict)
        self.assertEqual(self.g.nodetype2count_dictionary['biolink:NamedThing'], 11)
        self.assertEqual(het_g.nodetype2count_dictionary['biolink:Disease'], 3)

    # check nodetype to index map
    def test_csfgraph_makes_nodetype_to_index_map(self):
        self.assertIsInstance(self.g.nodetype_to_index_map, dict)

    def test_csfgraph_assigns_default_nodetype_to_nodetype_to_index_map(self):
        self.assertIsInstance(self.g.nodetype_to_index_map, dict)
        self.assertEqual(self.g.nodetype_to_index_map[self.g.default_node_type],
                         list(range(self.g.node_count())))

    def test_csfgraph_populates_nodetype_to_index_map(self):
        het_g = CSFGraph(edge_file=self.edge_file, node_file=self.node_file)
        self.assertEqual(het_g.nodetype_to_index_map['biolink:Disease'], [0, 1, 2])

    # check index to nodetype map
    def test_csfgraph_makes_index_to_nodetype_map(self):
        self.assertIsInstance(self.g.index_to_nodetype_map, dict)

    def test_csfgraph_populates_index_to_nodetype_map(self):
        het_g = CSFGraph(edge_file=self.edge_file, node_file=self.node_file)
        self.assertEqual(11, len(het_g.index_to_nodetype_map))
        self.assertEqual(het_g.index_to_nodetype_map[0], 'biolink:Disease')

    def test_csfgraph_assigns_default_node_type_to_index_to_nodetype_map(self):
        self.assertEqual(self.g.index_to_nodetype_map[0], self.g.default_node_type)

    def test_csfgraph_tolerates_missing_node_info(self):
        het_g = CSFGraph(edge_file=self.edge_file,
                         node_file=self.node_file_missing_nodes)
        self.assertEqual(het_g.index_to_nodetype_map[2], het_g.default_node_type)

    # edgetype to index map
    def test_csfgraph_makes_edgetype_to_index_map(self):
        self.assertIsInstance(self.g.edgetype_to_index_map, dict)

    def test_csfgraph_populates_edgetype_to_index_map(self):
        self.assertCountEqual(self.g.edgetype_to_index_map.keys(),
                              ['biolink:interacts_with',
                               'biolink:molecularly_interacts_with'])
        self.assertEqual(40, len(self.g.edgetype_to_index_map['biolink:interacts_with']))
        self.assertEqual(2, len(self.g.edgetype_to_index_map['biolink:molecularly_interacts_with']))

    # check index to edgetype map
    def test_csfgraph_constructor_makes_index_to_edgetype_map(self):
        self.assertIsInstance(self.g.index_to_edgetype_map, dict)

    def test_csfgraph_populates_index_to_edgetype_map(self):
        self.assertEqual(42, len(self.g.index_to_edgetype_map))
        self.assertEqual(self.g.index_to_edgetype_map[0], 'biolink:interacts_with')
        self.assertEqual(self.g.index_to_edgetype_map[34], 'biolink:molecularly_interacts_with')
        self.assertEqual(self.g.index_to_edgetype_map[40], 'biolink:molecularly_interacts_with')

    def test_csfgraph_requires_arg(self):
        with self.assertRaises(Exception) as context:
            CSFGraph()  # missing edge arg
            self.assertTrue(str('missing' in context.exception))

    def test_csfgraph_checks_for_subject_columns(self):
        with self.assertRaises(CSFGraphNoSubjectColumnError) as context:
            CSFGraph(edge_file=self.tsv_no_subject)  # file doesn't have subject col

    def test_csfgraph_checks_for_object_column(self):
        with self.assertRaises(CSFGraphNoObjectColumnError) as context:
            CSFGraph(edge_file=self.tsv_no_object)  # file doesn't have object col

    def test_csfgraph_accepts_edge_file(self):
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

    def test_edge_type_1(self):
        node_1 = "g1"
        node_2 = "p1"
        edge_type = self.g.edgetype(node_1, node_2)
        expected = "biolink:interacts_with"
        self.assertEqual(expected,edge_type)

    def test_edge_type_2(self):
        node_1 = "p2"
        node_2 = "p4"
        edge_type = self.g.edgetype(node_1, node_2)
        expected = "biolink:molecularly_interacts_with"
        self.assertEqual(expected,edge_type)