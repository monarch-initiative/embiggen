import collections
import re
import os
import string
import bz2
from math import ceil

stopwords = set(['the', 'a', 'an', 'another', 'for', 'an', 'nor', 'but', 'or', 'yet', 'so'])


class TextEncoder:
    """
    This class takes as input a file containing text that we want to
    encode as integers for Word2Vec. It cleanses the data, splits the
    text into sentences, and returns a list of lists, where the entries
    are lists of integers corresponding to non-stop words of the sentences.
    The Word2Vec implementations in this package are intended to support
    graph embedding operations and te text/NLP functionalities are just for
    demonstration.
    """

    def __init__(self, filename):
        if filename is None:
            raise TypeError("Need to pass filename to constructor")
        if not isinstance(filename, str):
            raise TypeError("File name arguments must be a string")
        if not os.path.exists(filename):
            raise TypeError("Could not find file: " + filename)
        self.filename = filename

    @staticmethod
    def clean_text(text):
        """
        :param text: A line of text, typically a line from a file
        :return: a cleansed version of the input text.
        """
        text = text.lower()

        text = text.replace("\n", "")
        text = text.replace("\xad", "")
        text = text.replace("'ve", " have")
        text = text.replace("'t", " not")
        text = text.replace("'s", " is")
        text = text.replace("'m", " am")

        # Numbers with Word
        text = text.replace("0", " zero ")
        text = text.replace("1", " one ")
        text = text.replace("2", " two ")
        text = text.replace("3", " three ")
        text = text.replace("4", " four ")
        text = text.replace("5", " five ")
        text = text.replace("6", " six ")
        text = text.replace("7", " seven ")
        text = text.replace("8", " eight ")
        text = text.replace("9", " nine ")
        text = text.replace("10", " nine ")

        punc = set(string.punctuation)
        for p in punc:
            text = text.replace(p, " ")
        text = " ".join(text.split())
        return text

    def read_databz2(self, filename):
        """
        Extract the first file enclosed in a zip (bz2) file as a list of words
        and pre-processes it using the nltk python library
        """
        ## NOTE NEED TO CONVERT TO SENTENCE WISE EXTRACTION
        with bz2.BZ2File(filename) as f:
            data = []
            file_size = os.stat(filename).st_size
            chunk_size = 1024 * 1024  # reading 1 MB at a time as the dataset is moderately large
            print('Reading data...')
            for i in range(ceil(file_size // chunk_size) + 1):
                bytes_to_read = min(chunk_size, file_size - (i * chunk_size))
                file_string = f.read(bytes_to_read).decode('utf-8')
                file_string = file_string.lower()
                # tokenizes a string to words residing in a list
                file_string = file_string.split()
                data.extend(file_string)
        return data

    def __read_data(self):
        """
        Extract the first file enclosed in a zip file as a list of words
        and pre-processes it using the nltk python library
        """
        data = []
        with open(self.filename) as f:
            print('Reading data from %s' % self.filename)
            for line in f:
                line = TextEncoder.clean_text(line)
                words = line.split()
                taker_words = []
                for word in words:
                    if not word in stopwords:
                        taker_words.append(word)
                data.extend(taker_words)
        return data

    def __read_data_in_sentences(self):
        """
        Extract the first file enclosed in a zip file as a list of words
        and pre-processes it using the nltk python library
        :param: Name of a file with text to use for word2vec
        :return: list of words from the file
        """
        data = []
        c = 0
        with open(self.filename) as f:
            filecontents = f.read().replace('\n', '')
            sentences = re.split("(?<=[.!?])\\s+", filecontents)
        for sent in sentences:
            sent = TextEncoder.clean_text(sent)
            # tokenizes a string to a list of words
            words = sent.split()
            taker_words = []
            for word in words:
                if not word in stopwords:
                    taker_words.append(word)
                    c += 1
            data.append(taker_words)
        print("Extracted {} non-stop words and {} sentences".format(c, len(data)))
        return data

    # we restrict our vocabulary size to 50000
    vocabulary_size = 50000

    def build_dataset(self):
        count = [['UNK', -1]]
        words = self.__read_data()
        vocabulary_size = min(50000, len(set(words)))
        # Gets only the vocabulary_size most common words as the vocabulary
        # All the other words will be replaced with UNK token
        count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
        dictionary = dict()

        # Create an ID for each word by giving the current length of the dictionary
        # And adding that item to the dictionary
        for word, _ in count:
            dictionary[word] = len(dictionary)

        data = list()
        unk_count = 0
        # Traverse through all the text we have and produce a list
        # where each element corresponds to the ID of the word found at that index
        for word in words:
            # If word is in the dictionary use the word ID,
            # else use the ID of the special token "UNK"
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1
            data.append(index)

        # update the count variable with the number of UNK occurences
        count[0][1] = unk_count

        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        # Make sure the dictionary is of size of the vocabulary
        assert len(dictionary) == vocabulary_size

        return data, count, dictionary, reverse_dictionary

    def build_dataset_in_sentences(self):
        """
        Create a dataset for word2vec that consists of all of the sentences from the original text
        with the words encoded as integers
        """
        sentences = self.__read_data_in_sentences()
        count = [['UNK', -1]]
        flat_list_of_words = [item for sublist in sentences for item in sublist]
        vocabulary_size = min(50000, len(set(flat_list_of_words)))
        # Gets only the vocabulary_size most common words as the vocabulary
        # All the other words will be replaced with UNK token
        # A counter is a container that stores elements as dictionary keys,
        # and their counts are stored as dictionary values.
        # Extend appends a list. We append 49,999 to get 50,000
        count.extend(collections.Counter(flat_list_of_words).most_common(vocabulary_size - 1))
        dictionary = dict()

        # Create an ID for each word by giving the current length of the dictionary
        # And adding that item to the dictionary
        for word, _ in count:
            dictionary[word] = len(dictionary)

        data = list()
        unk_count = 0
        # Traverse through all the text we have and produce a list of lists
        # each list entry is a sentence
        # where each element corresponds to the ID of the word found at that index
        for sen in sentences:
            integer_sentence = []  # encode the send
            for word in sen:
                # If word is in the dictionary use the word ID,
                # else use the ID of the special token "UNK"
                if word in dictionary:
                    index = dictionary[word]
                else:
                    index = 0  # dictionary['UNK']
                    unk_count = unk_count + 1
                integer_sentence.append(index)
            data.append(integer_sentence)

        # update the count variable with the number of UNK occurences
        count[0][1] = unk_count

        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        # Make sure the dictionary is of size of the vocabulary
        assert len(dictionary) == vocabulary_size
        del sentences  # Hint to reduce memory.
        # log.info('Most common words (+UNK) {}', str(count[:5)])
        # log.info('Sample data: {}', str(data[:10]))
        return data, count, dictionary, reverse_dictionary


data_index = 0
