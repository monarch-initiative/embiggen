import os.path

from unittest import TestCase

from embiggen import Graph, GraphFactory, Embiggen


class TestCBOWconstruction(TestCase):
    """Use the test files in test/data/ppismall to construct a CBOW and test that we can get some random walks."""

    def test_get_cbow_batch(self):
        graph = GraphFactory().read_csv(
            "tests/data/small_9606.protein.actions.txt",
            start_nodes_column="item_id_a",
            end_nodes_column="item_id_b",
            weights_column="score"
        )
        embedder = Embiggen()
        embedder.fit(graph)
        embedder.transform(graph, graph)