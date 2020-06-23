import os.path
import tensorflow as tf

from unittest import TestCase

from embiggen.transformers import TextTransformer
from tests.utils.utils import gets_tensor_length


class TestTextTransformerSentences(TestCase):
    """Test version of text encoder that extracts/converts words to integers and returns a tf.data.Dataset if list of
    sentences are the same length OR a tf.RaggedTensor if the list of sentences differ in length.
    """

    def setUp(self):

        # read in data
        self.infile = os.path.join(os.path.dirname(__file__), 'data', 'greatExpectations3.txt')

        # run text encoder
        encoder = TextTransformer(filename=self.infile,
                              payload_index=None,
                              header=None,
                              delimiter=None,
                              data_type='sentences',
                              stopwords=None,
                              minlen=10)

        self.data, self.count, self.dictionary, self.reverse_dictionary = encoder.build_dataset()

    def testClassInitializationDataType(self):
        """Tests the initialization of the class for the data_type parameter."""

        # when passing a word other than "words" or "sentences"
        self.assertRaises(TypeError, TextTransformer, self.infile, 'line')

    def testParseSuccess(self):

        # check returned data types
        self.assertIsInstance(self.data, tf.RaggedTensor)
        self.assertIsInstance(self.count, list)
        self.assertIsInstance(self.dictionary, dict)
        self.assertIsInstance(self.reverse_dictionary, dict)

    def testWordCounts(self):

        # convert wordcount to dictionary
        wordcount = {item[0]: item[1] for item in self.count}

        # test conditions
        self.assertEqual(1, wordcount['spiders'])
        self.assertEqual(2, wordcount['twig'])
        self.assertEqual(2, wordcount['blade'])

        # stopwords should have been removed
        self.assertEqual(None, wordcount.get('the'))

    def testNumberOfSentences(self):

        # test length of tensor
        tensor_length = gets_tensor_length(self.data)
        self.assertEqual(3, tensor_length)


class TestTextTransformerEnBlock(TestCase):
    """Test version of text encoder that extracts/converts words to integers and returns a tf.data.Dataset."""

    def setUp(self):

        # read in data
        infile = os.path.join(os.path.dirname(__file__), 'data', 'greatExpectations3.txt')

        # run text encoder
        encoder = TextTransformer(filename=infile,
                              payload_index=None,
                              header=None,
                              delimiter=None,
                              data_type='words',
                              stopwords=None,
                              minlen=10)

        self.data, self.count, self.dictionary, self.reverse_dictionary = encoder.build_dataset()

    def testParseSuccess(self):

        # check returned data types
        self.assertIsInstance(self.data, tf.Tensor)
        self.assertIsInstance(self.count, list)
        self.assertIsInstance(self.dictionary, dict)
        self.assertIsInstance(self.reverse_dictionary, dict)

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
        self.assertEqual(None, wordcount.get('the'))  # stopwords should have been removed


class TestCsvTextTransformer(TestCase):
    """Test version of text encoder that extracts/converts words to integers and returns a tf.data.Dataset if list of
        sentences are the same length OR a tf.RaggedTensor if the list of sentences differ in length.
        """

    def setUp(self):
        # read in data
        self.infile2 = os.path.join(os.path.dirname(__file__), 'data', 'pubmed20n1015excerpt.txt')

        # run text encoder
        encoder = TextTransformer(filename=self.infile2,
                              payload_index=2,
                              header=None,
                              delimiter='\t',
                              data_type='sentences',
                              stopwords=None,
                              minlen=10)

        self.data, self.count, self.dictionary, self.reverse_dictionary = encoder.build_dataset()

    def testClassInitializationPayloadIndex(self):
        """Tests the initialization of the class for the payload_index parameter."""

        # when passing a non-integer
        self.assertRaises(TypeError, TextTransformer, self.infile2, 'two', None, None, 'words', None, 10)

    def testClassInitializationHeader(self):
        """Tests the initialization of the class for the header parameter."""

        # when passing a non-integer
        self.assertRaises(TypeError, TextTransformer, self.infile2, 2, 'two', None, 'words', None, 10)

    def testClassInitializationDelimiter(self):
        """Tests the initialization of the class for the delimiter parameter."""

        # when passing something other than a string
        self.assertRaises(TypeError, TextTransformer, self.infile2, 2, None, [','], 'words', None, 10)

    def testClassInitializationMinLen(self):
        """Tests the initialization of the class for the minlen parameter."""

        # when passing something other than an int
        self.assertRaises(TypeError, TextTransformer, self.infile2, 2, None, [','], 'words', None, 'ten')

    def testParseSuccess(self):

        # check returned data types
        self.assertIsInstance(self.data, tf.RaggedTensor)
        self.assertIsInstance(self.count, list)
        self.assertIsInstance(self.dictionary, dict)
        self.assertIsInstance(self.reverse_dictionary, dict)

    def testCsvParserIngestedTenLines(self):

        expected_number_of_rows = 10
        i = 0

        for _ in self.data:
            i += 1

        self.assertEqual(expected_number_of_rows, i)
