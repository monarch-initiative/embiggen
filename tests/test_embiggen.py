import os.path

from unittest import TestCase

from embiggen import Graph, GraphFactory, Embiggen


class TestCBOWconstruction(TestCase):
    """Use the test files in test/data/ppismall to construct a CBOW and test that we can get some random walks."""

    def test_get_cbow_batch(self):
        graph = GraphFactory().read_csv(
            "tests/data/rand_100nodes_5000edges.graph",
            edge_has_header=False,
            start_nodes_column=0,
            end_nodes_column=1,
            weights_column=2,
            return_weight=10,
            explore_weight=10
        )
        embedder = Embiggen()
        embedder.fit(graph, walks_number=10)
        embedder.transform(graph, graph)