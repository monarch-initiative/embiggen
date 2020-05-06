# import needed libraries
import os.path
import shutil
import tensorflow as tf
import unittest

from typing import Dict, List

from xn2v import CSFGraph
from xn2v import N2vGraph
from xn2v.word2vec import ContinuousBagOfWordsWord2Vec
from xn2v.utils import get_embedding, write_embeddings, load_embeddings

# TODO: add test for calculates_cosine_similarity in utils/embedding_utils


class TestEmbeddingUtils(unittest.TestCase):
    """Class to test the methods designed to work with embeddings from the embedding utility script."""

    def setUp(self):

        # read in sample data
        current_directory = os.path.dirname(__file__)
        self.data_dir = os.path.join(current_directory, 'data')
        pos_train = os.path.abspath(self.data_dir + '/ppismall/pos_train_edges')

        # read data into graph
        training_graph = CSFGraph(pos_train)
        worddictionary = training_graph.get_node_to_index_map()
        self.reverse_worddictionary = training_graph.get_index_to_node_map()

        # generate random walks
        n2v_graph = N2vGraph(training_graph, 1, 1, 1, False)
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

    def tests_get_embedding(self):
        """Tests the gets_embedding method."""

        # check function fails when
        self.assertRaises(ValueError, get_embedding, 0, None)

        # check that data is returned and that the data is a tensor
        self.assertIsInstance(get_embedding(0, self.model.embedding), tf.Tensor)

        return None

    def tests_write_embeddings(self):
        """Tests the writes_embeddings method."""

        # check that data is written
        write_embeddings(self.temp_dir_loc + '/sample_embedding_data.txt',
                         self.model.embedding,
                         self.reverse_worddictionary)

        self.assertTrue(os.path.exists(self.temp_dir_loc + '/sample_embedding_data.txt'))

        return None

    def tests_load_embeddings(self):
        """tests the load_embeddings method."""

        # write out embedding data
        write_embeddings(self.temp_dir_loc + '/sample_embedding_data.txt',
                         self.model.embedding,
                         self.reverse_worddictionary)

        # read in embeddings
        embedding_map = load_embeddings(self.temp_dir_loc + '/sample_embedding_data.txt')

        # make sure embeddings are read in as a dictionary
        self.assertIsInstance(embedding_map, Dict)

        # make sure that the embedding value is a list of floats
        sample_entry = embedding_map[list(embedding_map.keys())[0]]
        self.assertIsInstance(sample_entry, List)
        self.assertIsInstance(sample_entry[0], float)

    def tearDown(self):

        # remove temp directory
        shutil.rmtree(self.temp_dir_loc)

        return None
