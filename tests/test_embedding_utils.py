# import needed libraries
import os.path
import shutil
import tensorflow as tf
import unittest

from typing import Dict, List

from embiggen import CSFGraph
from embiggen import N2vGraph
from embiggen.word2vec import ContinuousBagOfWordsWord2Vec
from embiggen.utils import get_embedding, write_embeddings, load_embeddings

# TODO: add test for calculates_cosine_similarity in utils/embedding_utils


class TestEmbeddingUtils(unittest.TestCase):
    """Class to test the methods designed to work with embeddings from the embedding utility script."""

    def setUp(self):

        # read in sample data
        current_directory = os.path.dirname(__file__)
        self.data_dir = os.path.join(current_directory, 'data')
        pos_train = os.path.abspath(
            self.data_dir + '/ppismall/pos_train_edges')

        # read data into graph
        training_graph = CSFGraph(pos_train)
        worddictionary = training_graph.get_node_to_index_map()
        self.reverse_worddictionary = training_graph.get_index_to_node_map()

        # generate random walks
        n2v_graph = N2vGraph(training_graph, 1, 1)
        walks = n2v_graph.simulate_walks(5, 10)

        # learn embeddings
        self.model = ContinuousBagOfWordsWord2Vec(walks,
                                                  worddictionary=worddictionary,
                                                  reverse_worddictionary=self.reverse_worddictionary,
                                                  num_epochs=2)

        # create temporary directory to write data to
        self.temp_dir_loc = os.path.abspath(self.data_dir + '/temp')
        os.mkdir(self.temp_dir_loc)

        return None

    def tests_save_embedding(self):
        """Tests the writes_embeddings method."""
        path = f"{self.temp_dir_loc}/sample_embedding_data.csv"
        self.model.save(path)
        self.assertTrue(os.path.exists(path))

    def tearDown(self):

        # remove temp directory
        shutil.rmtree(self.temp_dir_loc)

        return None
