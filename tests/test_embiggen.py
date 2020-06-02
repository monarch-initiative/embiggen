import os.path
import pytest
from unittest import TestCase
from tqdm.auto import tqdm
from embiggen import Graph, GraphFactory, Embiggen


class TestEmbiggen(TestCase):
    """Test suite for the class Embiggen"""

    def setUp(self):
        self.graphs = []

        factory = GraphFactory()
        directed_factory = GraphFactory(default_directed=True)
        # For heterogeneous test
        self.graphs.append(factory.read_csv(
            "tests/data/small_graph_edges.tsv",
            "tests/data/small_graph_nodes.tsv",
            return_weight=10,
            explore_weight=10,
            change_type_weight=10
        ))

        # For homogeneous test
        self.graphs.append(factory.read_csv(
            "tests/data/rand_100nodes_5000edges.graph",
            edge_file_has_header=False,
            start_nodes_column=0,
            end_nodes_column=1,
            weights_column=2,
            return_weight=10,
            explore_weight=10
        ))

        # For heterogeneous directed test
        # TODO! Uncomment for testing for the ragged tensors!
        # self.graphs.append(directed_factory.read_csv(
        #     "tests/data/small_graph_edges.tsv",
        #     "tests/data/small_graph_nodes.tsv",
        #     return_weight=10,
        #     explore_weight=10,
        #     change_type_weight=10
        # ))

        # For homogeneous directed test
        # TODO! Uncomment for testing for the ragged tensors!
        # self.graphs.append(directed_factory.read_csv(
        #     "tests/data/rand_100nodes_5000edges.graph",
        #     edge_file_has_header=False,
        #     start_nodes_column=0,
        #     end_nodes_column=1,
        #     weights_column=2,
        #     return_weight=10,
        #     explore_weight=10
        # ))

    def test_cbow_on_embiggen(self):
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

        for graph in tqdm(self.graphs, desc="Testing embiggen"):
            embedder.fit(graph)
            embedder.transform(graph, graph)
            embedder.transform_nodes(graph, graph)
            embedder.save_embedding("embedding.csv")
            embedder.load_embedding("embedding.csv")
            os.remove("embedding.csv")

        with pytest.raises(ValueError):
            embedder.load_embedding("this_file_does_not_exists.csv")