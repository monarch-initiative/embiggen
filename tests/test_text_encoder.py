from unittest import TestCase

import os.path
from _collections import defaultdict
from n2v import TextEncoder


class TestTextEncoderSentences(TestCase):
    def setUp(self):
        inputfile = os.path.join(os.path.dirname(__file__), 'data', 'greatExpectations3.txt')
        encoder = TextEncoder(inputfile)
        data, count, dictionary, reverse_dictionary = encoder.build_dataset_in_sentences()
        self.data = data
        self.count = count
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary

    def testWordCounts(self):
        print(self.count)
        wordcount = defaultdict(int)
        for item in self.count:
            wordcount[item[0]] = item[1]
        self.assertEqual(1,wordcount['spiders'])
        self.assertEqual(2, wordcount['twig'])
        self.assertEqual(2, wordcount['blade'])
        self.assertEqual(0, wordcount['the']) #Stopwords shhould have been removed

    def test_number_of_sentences(self):
        # Our text has three sentences
        print(self.data)
        self.assertEqual(3, len(self.data))


class TestTextEncoderEnBlock(TestCase):
    """
    Test version of text encoder that extracts/converts words to integers and returns then in a single list.
    """
    def setUp(self):
        inputfile = os.path.join(os.path.dirname(__file__), 'data', 'greatExpectations3.txt')
        encoder = TextEncoder(inputfile)
        data, count, dictionary, reverse_dictionary = encoder.build_dataset()
        self.data = data
        self.count = count
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary

    def testWordCounts(self):
        print(self.count)
        wordcount = defaultdict(int)
        for item in self.count:
            wordcount[item[0]] = item[1]
        self.assertEqual(1,wordcount['spiders'])
        self.assertEqual(2, wordcount['twig'])
        self.assertEqual(2, wordcount['blade'])
        self.assertEqual(0, wordcount['the']) #Stopwords shhould have been removed
