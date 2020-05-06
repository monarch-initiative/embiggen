import os.path
import tensorflow as tf

from unittest import TestCase

from xn2v.text_encoder import TextEncoder  # update once refactor is complete
from tests.utils.utils import gets_tensor_length


class TestTextEncoderSentences(TestCase):
    """Test version of text encoder that extracts/converts words to integers and returns a tf.data.Dataset if list of
    sentences are the same length OR a tf.RaggedTensor if the list of sentences differ in length.
    """

    def setUp(self):

        # read in data
        infile = os.path.join(os.path.dirname(__file__), 'data', 'greatExpectations3.txt')

        # run text encoder
        encoder = TextEncoder(filename=infile, data_type='sentences')
        self.data, self.count, self.dictionary, self.reverse_dictionary = encoder.build_dataset()

    def testWordCounts(self):

        # convert wordcount to dictionary
        wordcount = {item[0]: item[1] for item in self.count}

        # test conditions
        self.assertEqual(1, wordcount['spiders'])
        self.assertEqual(2, wordcount['twig'])
        self.assertEqual(2, wordcount['blade'])

        # stopwords should have been removed
        self.assertEqual(None, wordcount.get('the'))

    def test_number_of_sentences(self):

        # test length of tensor
        tensor_length = gets_tensor_length(self.data)
        self.assertEqual(3, tensor_length)


class TestTextEncoderEnBlock(TestCase):
    """Test version of text encoder that extracts/converts words to integers and returns a tf.data.Dataset."""

    def setUp(self):

        # read in data
        infile = os.path.join(os.path.dirname(__file__), 'data', 'greatExpectations3.txt')

        # run text encoder
        encoder = TextEncoder(filename=infile, data_type='words')
        self.data, self.count, self.dictionary, self.reverse_dictionary = encoder.build_dataset()

    def testWordCounts(self):

        # convert wordcount to dictionary
        wordcount = {item[0]: item[1] for item in self.count}

        # test conditions
        # Apr 26, Peter removed the following, we no longer want to use the tf.data.Dataset API for this
        # self.assertEqual(True, isinstance(self.data, tf.data.Dataset))
        # instead this should be a simple tensor
        self.assertTrue(isinstance(self.data, tf.Tensor))
        self.assertEqual(1, wordcount['spiders'])
        self.assertEqual(2, wordcount['twig'])
        self.assertEqual(2, wordcount['blade'])
        self.assertEqual(None, wordcount.get('the'))  # Stopwords should have been removed
