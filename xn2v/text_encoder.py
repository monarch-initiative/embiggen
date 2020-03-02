import bz2
import collections
import os
import re
import string

from math import ceil

from pandas.core.common import flatten  # type: ignore
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer  # type: ignore
from typing import Any, Dict, List, Tuple


class TextEncoder:
    """
    This class takes as input a file containing text that we want to encode as integers for Word2Vec. It cleanses the
    data, splits the text into sentences, and returns a list of lists, where the entries are lists of integers
    corresponding to non-stop words of the sentences.

    The Word2Vec implementations in this package are intended to support graph embedding operations and text/NLP
    functionality is included for demonstration.

    Attributes:
        filename: a filepath and name to text which needs encoding.
        stopwords: a set of stopwords. If nothing is passed by user a default list of stopwords is utilized.

    Raises:
        ValueError: If the filename is None.
        TypeError: If the filename attribute is not a string.
        TypeError: If the file referenced by filename could not be found.
    """

    def __init__(self, filename: str, stopwords: set = None):

        if filename is None:
            raise ValueError('filename cannot be None')
        elif not isinstance(filename, str):
            raise TypeError('filename must be a string')
        elif not os.path.exists(filename):
            raise TypeError('Could not find file referenced by filename: {}'.format(filename))
        else:
            self.filename = filename

        self.stopwords = {'the', 'a', 'an', 'another', 'for', 'an', 'nor', 'but', 'or', 'yet', 'so'} \
            if stopwords is None else stopwords

    @staticmethod
    def clean_text(text: str) -> str:
        """Takes a text string and performs several tasks that are intended to clean the text including making the
        text lowercase, undoing contractions,

        Args:
            text: A line of text, typically a line of text from a file.

        Returns:
            A cleansed version of the input text.
        """

        # ensure text is lower case
        text = text.lower()

        # undo contractions
        text = text.replace("\n", "")
        text = text.replace("\xad", "")
        text = text.replace("'ve", " have")
        text = text.replace("'t", " not")
        text = text.replace("'s", " is")
        text = text.replace("'m", " am")

        # replace numbers with words representing number
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
        text = text.replace("10", " ten ")

        # remove punctuation - gets non-ascii + ascii punctuation
        text = re.sub(r'[^\w\s]', ' ', text.translate(str.maketrans('', '', string.punctuation)))

        return ' '.join(text.split())

    def read_databz2(self) -> List[str]:
        """ Extracts the first file enclosed in a zip (bz2) file as a list of words and pre-processes it using the
        nltk python library.

        Returns:
            A list of words.
        """

        # NOTE NEED TO CONVERT TO SENTENCE WISE EXTRACTION
        with bz2.BZ2File(self.filename) as file:
            data = []
            file_size = os.stat(self.filename).st_size
            chunk_size = 1024 * 1024  # reading 1 MB at a time as the dataset is moderately large

            print('Reading data from {filename}'.format(filename=self.filename))

            for i in range(ceil(file_size // chunk_size) + 1):
                bytes_to_read = min(chunk_size, file_size - (i * chunk_size))
                file_string = file.read(bytes_to_read).decode('utf-8')
                # lowercase and tokenize into list of words
                data.extend(file_string.lower().split())

        return data

    def __read_data(self) -> List[str]:
        """Extract the first file enclosed in a zip file as a list of words and pre-processes it using the nltk
        python library.

        Returns:
            A list of cleaned words.
        """

        data = []

        with open(self.filename) as file:
            print('Reading data from {filename}'.format(filename=self.filename))

            for line in file:
                cleaned_line = self.clean_text(line)
                data.extend([word for word in cleaned_line.split() if word not in self.stopwords])

        return data

    def __read_data_in_sentences(self) -> List[List[str]]:
        """
        Extract the first file enclosed in a zip file as a list of words and pre-processes it using the nltk python
        library.

        Returns:
            A nested list of cleaned words from the file.
        """

        data = []
        count = 0

        with open(self.filename) as file:
            print('Reading data from {filename}'.format(filename=self.filename))

            sentences = re.split("(?<=[.!?])\\s+", file.read().replace('\n', ''))

        for sent in sentences:
            sent = self.clean_text(sent)
            words = sent.split()
            keep_words = []

            for word in words:
                if word not in self.stopwords:
                    keep_words.append(word)
                    count += 1

            data.append(keep_words)

        print("Extracted {} non-stop words and {} sentences".format(count, len(data)))

        return data

    def build_dataset(self) -> Tuple[List, List, Dict[Any, Any], Dict[Any, Any]]:
        """Builds a dataset by traversing over a input text and compiling a list of the most commonly occurring words.

        Returns:
            data: A list of the most commonly occurring word indices.
            count: A list of tuples, where the first item in each tuple is a word and the second is the word frequency.
            dictionary: A dictionary where the keys are words and the values are the word id.
            reversed dictionary: A dictionary that is the reverse of the dictionary object mentioned above.

        Raises:
            ValueError: If the length of the dictionary does not match vocabulary_size.
        """

        count = [['UNK', -1]]
        words = self.__read_data()
        vocabulary_size = min(50000, len(set(words)))

        # gets the vocabulary_size of most common words, all the other words will be replaced with UNK token
        count += [list(x) for x in collections.Counter(words).most_common(vocabulary_size - 1)]
        dictionary: Dict = dict()

        # create an ID for each word from current length of the dictionary
        for word, _ in count:
            dictionary[word] = len(dictionary)

        # traverse text and produce a list where each element corresponds to the ID of the word at that idx
        data, unk_count = list(), 0

        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1

            data.append(index)

        # update the count variable with the number of UNK occurrences
        count[0][1] = unk_count

        # make sure the dictionary is of size of the vocabulary
        if len(dictionary) != vocabulary_size:
            raise ValueError('The length of the dictionary does not match vocabulary_size.')

        return data, count, dictionary, dict(zip(dictionary.values(), dictionary.keys()))

    def build_dataset_in_sentences(self) -> Tuple[List, List, Dict, Dict]:
        """ Creates a dataset for word2vec that consists of all of the sentences from the original text with the
        words encoded as integers. The list is created by first getting only the vocabulary_size of the most common
        words. All the other words will be replaced with an UNK token.

        Returns:
            data: A list of the most commonly occurring word indices.
            count: A list of tuples, where the first item in each tuple is a word and the second is the word frequency.
            dictionary: A dictionary where the keys are words and the values are the word id.
            reversed dictionary: A dictionary that is the reverse of the dictionary object mentioned above.

        ValueError: If the length of the dictionary does not match vocabulary_size.
        """

        sentences = self.__read_data_in_sentences()
        count = [['UNK', -1]]
        flat_list_of_words = [item for sublist in sentences for item in sublist]
        vocabulary_size = min(50000, len(set(flat_list_of_words)))

        # a counter container stores elements as dict keys and their counts are stored as values
        count += [list(x) for x in collections.Counter(flat_list_of_words).most_common(vocabulary_size - 1)]
        dictionary: Dict = dict()

        # create an ID for each word by giving the current length of the dictionary
        for word, _ in count:
            dictionary[word] = len(dictionary)

        data, unk_count = list(), 0

        # traverse text to produce a list of lists, each list is a sentence and each element is the word id at that idx
        for sen in sentences:
            integer_sentence = []  # encode the send

            for word in sen:
                if word in dictionary:
                    index = dictionary[word]
                else:
                    index = 0  # dictionary['UNK']
                    unk_count = unk_count + 1

                integer_sentence.append(index)
            data.append(integer_sentence)

        # update the count variable with the number of UNK occurrences
        count[0][1] = unk_count

        # make sure the dictionary is of size of the vocabulary
        if len(dictionary) != vocabulary_size:
            raise ValueError('The length of the dictionary does not match vocabulary_size.')

        # reduce memory
        del sentences

        # log.info('Most common words (+UNK) {}', str(count[:5)])
        # log.info('Sample data: {}', str(data[:10]))
        return data, count, dictionary, dict(zip(dictionary.values(), dictionary.keys()))

    def get_raw_text(self) -> str:
        """Reads in data from a file and returns the data as a single string.

        Returns:
            text: A string of text from the read in file.
        """

        text = open(self.filename).read()

        return text

    def parse_file_into_sentences(self) -> List[str]:
        """Reads data from a file and returns the data as a list of sentences.

        Returns:
            sentences: A list of strings which represent sentences.
        """

        sentences = []

        with open(self.filename) as file:
            print('Reading data from {filename}'.format(filename=self.filename))

            for line in file:
                # calling clean_text since Keras does not undo contractions
                cleaned_line = self.clean_text(line)

                # removing stopwords since Keras does not support removing a defined set of words
                cleaned_lines = ' '.join(self.remove_stopwords(cleaned_line.split(' ')))

                sentences.append(cleaned_lines)

        return sentences

    def remove_stopwords(self, words: List[str]) -> List[str]:
        """Removes words form a list of words if the word occurs in a list of stop words.

        Args:
            words: A list of words.

        Returns:
            A list of words with stop words removed.
        """

        filtered_words = []
        filtered_words.extend([word for word in words if word not in self.stopwords])

        return filtered_words

    def build_dataset_with_keras(self, max_vocab_size=50000) -> Tuple[List, List, Dict, Dict]:
        """A TensorFlow implementation of the text-encoder functionality.

        Note. The Tokenizer method is initialized with 'UNK' as the out-of-vocabulary token. Keras reserves the 0th
        index for padding sequences, the index for  'UNK' will be 1st index max_vocab_size + 1 because Keras reserves
        the 0th index.

        Args:
            max_vocab_size: An integer specifying the maximum vocabulary size.

        Returns:
            flatted_sequences: A list of the most commonly occurring word indices.
            count_as_tuples: A list of tuples, the first item is a word and the second is the word frequency.
            dictionary: A dictionary where the keys are words and the values are the word id.
            reverse_dictionary: A dictionary that is the reverse of the dictionary object mentioned above.

        Raises:
            ValueError: If the length of count_as_tuples does not match max_vocab_size.
        """

        text = self.get_raw_text()
        words = text_to_word_sequence(text, lower=True, filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        words = self.remove_stopwords(words)
        max_vocab_size = min(max_vocab_size, len(set(words)))

        # initialize tokenizer
        tokenizer = Tokenizer(num_words=max_vocab_size + 1,
                              filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                              lower=True, oov_token=['UNK'][0])

        # apply tokenizer to process text
        sentences = self.parse_file_into_sentences()
        tokenizer.fit_on_texts(sentences)
        sequences = tokenizer.texts_to_sequences(sentences)
        flatted_sequences = list(flatten(sequences))  # for downstream compatibility
        count = tokenizer.word_counts
        filtered_count, dictionary = {}, {}  # for downstream compatibility

        for k, v in tokenizer.word_index.items():
            if v <= max_vocab_size:
                if k == 'UNK':
                    filtered_count['UNK'] = 0
                    dictionary['UNK'] = 1
                    continue

                filtered_count[k] = count[k]
                dictionary[k] = v
            else:
                filtered_count['UNK'] += 1

        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        # for downstream compatibility
        count_as_list = [list(x) for x in list(zip(list(filtered_count.keys()), list(filtered_count.values())))]

        if max_vocab_size != len(count_as_list):
            raise ValueError('The length of count_as_tuples does not match max_vocab_size.')
        else:
            return flatted_sequences, count_as_list, dictionary, reverse_dictionary
