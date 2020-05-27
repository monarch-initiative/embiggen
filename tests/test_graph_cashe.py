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

    def test_caching(self):
        """
        Test caching of random walks in N2vGraph by,
            - creating the N2vGraph
            - running simulate_walks and storing the walks as a property of N2vGraph
        """
        p = 1
        q = 1
        num_walks = 10
        walk_length = 1
        g = N2vGraph(self.graph, p, q)
        g.simulate_walks(num_walks, walk_length)
        assert len(g.random_walks_map) > 0

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

    def test_cache_graph_state_restore(self):
        """
        Test restore of N2vGraph state from a pickle.
        """
        g = deserialize('N2vGraph.pkl')
        k = (10, 1)
        assert k in g.random_walks_map.keys()
