import os.path
import pytest
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
        with pytest.raises(ValueError):
            Embiggen(embedding_method="unsupported")
        # TODO! the following test should be re-enabled once the fit method
        # of the embiggen and all word2vec classes has been properly updated.
        # with pytest.raises(ValueError):
        #     Embiggen().embedding
        # TODO! the following test should be re-enabled once the fit method
        # of the embiggen and all word2vec classes has been properly updated.
        # with pytest.raises(ValueError):
        #     Embiggen().save_embedding("test.csv")
        embedder.fit(graph, walks_number=10)
        embedder.transform(graph, graph)
        embedder.transform_nodes(graph, graph)
        embedder.save_embedding("embedding.csv")
        embedder.load_embedding("embedding.csv")
        os.remove("embedding.csv")
        with pytest.raises(ValueError):
            embedder.load_embedding("this_file_does_not_exists.csv")