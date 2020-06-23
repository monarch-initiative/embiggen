import os.path
import pytest
from unittest import TestCase
from tqdm.auto import tqdm
from embiggen.utils import TextEncoder
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from embiggen.embedders import CBOW, SkipGram, GloVe
import tensorflow as tf


class TestGraphEmbedding(TestCase):
    """Test suite for the class Embiggen"""

    def setUp(self):
        self._graph = EnsmallenGraph(
            edge_path="tests/data/small_het_graph_edges.tsv",
            sources_column="subject",
            destinations_column="object",
            directed=False,
            weights_column="weight",
            default_weight=1,
            node_path="tests/data/small_het_graph_nodes.tsv",
            nodes_column="id",
            node_types_column="category",
            default_node_type="biolink:NamedThing"
        )

    def test_skipgram(self):
        X = tf.ragged.constant(self._graph.walk(10, 80, 0, 1, 1, 1, 1))
        embedder_model = SkipGram()
        embedder_model.fit(X, self._graph.get_nodes_number())
        embedder_model.embedding
