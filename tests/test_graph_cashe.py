from unittest import TestCase

import os.path
from embiggen import CSFGraph
from embiggen.random_walk_generator import N2vGraph
from embiggen.utils import serialize, deserialize


class TestGraphCache(TestCase):

    def setUp(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')

        # these pass the tests okay
        node_file = os.path.join(data_dir, 'small_graph_nodes.tsv')
        edge_file = os.path.join(data_dir, 'small_graph_edges.tsv')

        g = CSFGraph(edge_file=edge_file, node_file=node_file)
        self.graph = g

    def test_cache_graph_state(self):
        """
        Test caching of N2vGraph state into a pickle.
        """
        p = 1
        q = 1
        gamma = 1
        num_walks = 10
        walk_length = 1
        g = N2vGraph(self.graph, p, q)
        walks1 = g.simulate_walks(num_walks, walk_length)
        serialize(g, 'N2vGraph.pkl')