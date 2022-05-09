"""Unit test class for GraphTransformer objects."""
from unittest import TestCase
from ensmallen import Graph  # pylint: disable=no-name-in-module
from embiggen import GraphTransformer, GloVe, EdgeTransformer
import pickle
import os


class TestGraphTransformer(TestCase):
    """Unit test class for GraphTransformer objects."""

    def setUp(self):
        """Setup objects for running tests on GraphTransformer objects class."""
        self._embedding_size = 50
        self._graph: Graph = Graph.from_csv(
            edge_path="tests/data/small_ppi.tsv",
            sources_column="subject",
            destinations_column="object",
            directed=False,
            weights_column="weight"
        )
        self._transfomer = None
        self._embedding = GloVe(
            vocabulary_size=self._graph.get_nodes_number(),
            embedding_size=self._embedding_size
        ).get_embedding_dataframe(self._graph.get_node_names())

    def test_graph_transformer(self):
        """Test to verify that graph transformation returns expected shape."""
        for aligned_node_mapping in (True, False):
            for embedding_method in EdgeTransformer.methods:
                self._transfomer = GraphTransformer(
                    method=embedding_method,
                    aligned_node_mapping=aligned_node_mapping
                )
                self._transfomer.fit(self._embedding)
                embedded_edges = self._transfomer.transform(self._graph)
                if embedding_method == "Concatenate":
                    self.assertEqual(
                        embedded_edges.shape,
                        (self._graph.get_undirected_edges_number(),
                         self._embedding_size*2)
                    )
                elif embedding_method is None:
                    self.assertEqual(
                        embedded_edges.shape,
                        (self._graph.get_undirected_edges_number(), 2)
                    )
                elif "Distance" in embedding_method:
                    self.assertEqual(
                        embedded_edges.shape,
                        (self._graph.get_undirected_edges_number(), )
                    )
                else:
                    self.assertEqual(
                        embedded_edges.shape,
                        (self._graph.get_undirected_edges_number(),
                         self._embedding_size)
                    )

    def test_graph_transformer_picklability(self):
        """Test to verify that graph transformation returns expected shape."""
        self._transfomer = GraphTransformer()
        self._transfomer.fit(self._embedding)
        self._transfomer.transform(self._graph)
        path = "test_pickled_object.pkl"
        with open(path, "wb") as f:
            pickle.dump(self._transfomer, f)
        os.remove(path)
