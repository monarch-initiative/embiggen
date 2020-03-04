import os.path
import tensorflow as tf

from unittest import TestCase

from xn2v.text_encoder_tf import TextEncoder  # update once refactor is complete


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

        if isinstance(self.data, tf.RaggedTensor):
            self.assertEqual(3, self.data.shape[0])
        else:
            self.assertEqual(3, len([sentence for sentence in self.data]))


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
        self.assertEqual(True, isinstance(self.data, tf.data.Dataset))
        self.assertEqual(1, wordcount['spiders'])
        self.assertEqual(2, wordcount['twig'])
        self.assertEqual(2, wordcount['blade'])
        self.assertEqual(None, wordcount.get('the'))  # Stopwords should have been removed
