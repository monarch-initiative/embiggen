import nltk  # type: ignore
import os
import pandas as pd  # type: ignore
import re
import tensorflow as tf  # type: ignore

from collections import Counter
from more_itertools import unique_everseen  # type: ignore
from pandas.core.common import flatten  # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from typing import Dict, List, Optional, Tuple, Union


class TextEncoder:
    """This class takes as input a file containing text that we want to encode as integers for Word2Vec. It cleanses the
    data, splits the text into sentences, and returns a tf.Tensor (tf.data.Dataset if a single span of text or list of
    sentences of the same length OR a tf.RaggedTensor if the list of sentences differ in length), where the entries are
    lists of integers corresponding to non-stop words of the sentences.

    The Word2Vec implementations in this package are intended to support graph embedding operations and text/NLP
    functionality is included for demonstration.

    Attributes:
        filename: A filepath and name to text which needs encoding.
        payload_index: An integer that if specified is used to process a specific column from an input csv file.
        header: An integer, that if specified contains the row index of the input data containing file header info.
        delimiter: A string containing the file delimiter type.
        data_type: A string which is used to indicate whether or not the data should be read in as a single text
            object or as a list of sentences. Passed values can be "words" or "sentences" (default="sentences").
        stopwords: A set of stopwords. If nothing is passed by user a default list of stopwords is utilized.

    Raises:
        ValueError: If filename is None.
        TypeError: If filename and delimiter (when specified) are not strings.
        TypeError: If payload_index and header (when specified) are not are not integers.
        IOError: If the file referenced by filename could not be found.
        TypeError: If payload_index is not an integer.
        ValueError: If data_type is not "words" or "sentences".
    """

    def __init__(self, filename: str, payload_index: Optional[int] = None, header: Optional[int] = None,
                 delimiter: Optional[str] = None, data_type: Optional[str] = None, stopwords: set = None) -> None:

        # verify filename structure
        if not filename:
            raise ValueError('filename cannot be None')
        elif not isinstance(filename, str):
            raise TypeError('filename must be a string')
        elif not os.path.exists(filename):
            raise IOError('could not find file referenced by filename: {}'.format(filename))
        else:
            self.filename = filename

        if payload_index and not isinstance(payload_index, int):
            raise TypeError('payload_index must be an integer')
        else:
            self.payload_index = payload_index if payload_index else None

            if header and not isinstance(header, int):
                raise TypeError('header must be an integer')
            else:
                self.header = header if header else None

            if delimiter and not isinstance(delimiter, str):
                raise TypeError('delimiter must be a string')
            else:
                self.delimiter = delimiter if delimiter else '\t'

        if data_type and data_type.lower() not in ['sentences', 'words']:
            raise ValueError('data_type must be "words" or "sentences"')
        else:
            self.data_type = data_type.lower() if data_type else 'sentences'

        try:
            self.stopwords = nltk.corpus.stopwords.words('english') if not stopwords else stopwords
        except LookupError:
            nltk.download('stopwords')
            self.stopwords = nltk.corpus.stopwords.words('english') if not stopwords else stopwords

    def clean_text(self, text: str) -> str:
        """Takes a text string and performs several tasks that are intended to clean the text including making the
        text lowercase, undoing contractions, removing punctuation, and removing stop words.

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

        # remove punctuation (gets non-ascii + ascii punctuation)
        text = ' '.join(re.sub(r'[^a-zA-Z\s+]', ' ', text, re.I | re.A).split())

        # remove stopwords
        text = ' '.join([x for x in text.split() if x not in self.stopwords])

        return text

    def process_input_text(self) -> List[str]:
        """Reads in data from a file and returns the data as a single string (if data_type is "words") or as a list
        of strings (if the data_type is "sentences").

        Returns:
            text: A string or list of stings of text from the read in file.
        """

        print('Reading {file} and processing it as {data_type}'.format(file=self.filename, data_type=self.data_type))

        if self.data_type == 'words':
            word_data = open(self.filename).read()
            return self.clean_text(word_data).split()
        else:
            if self.payload_index:
                data = pd.read_csv(self.filename, sep=self.delimiter, header=self.header)
                sentence_data = list(data[list(data).index(self.payload_index)])
            else:
                sentence_data = open(self.filename).readlines()

            return [self.clean_text(sent) for sent in sentence_data]

    def build_dataset(self, max_vocab=50000) -> Tuple[Union[tf.Tensor, tf.RaggedTensor], List, Dict, Dict]:
        """A TensorFlow implementation of the text-encoder functionality.

        Note. The Tokenizer method is initialized with 'UNK' as the out-of-vocabulary token. Keras reserves the 0th
        index for padding sequences, the index for 'UNK' will be 1st index max_vocab_size + 1 because Keras reserves
        the 0th index.

        Args:
            max_vocab: An integer specifying the maximum vocabulary size.

        Returns:
            tensor_data: A tf.Tensor (tf.data.Dataset if a single span of text or list of sentences of the same length
                OR a tf.RaggedTensor if the list of sentences differ in length) the first  item is a word and
                the second is the word frequency.
            count_list: A list of tuples, the first item is a word and the second is the word frequency.
            dictionary: A dictionary where the keys are words and the values are the word id.
            reverse_dictionary: A dictionary that is the reverse of the dictionary object mentioned above.

        Raises:
            ValueError: If the length of count_as_tuples does not match max_vocab_size.
        """

        # read in data
        text = self.process_input_text()

        # get word count and set max_vocab_size
        if self.data_type == 'words':
            word_count = len(set(text))
            max_vocab = min(max_vocab, word_count) + 1
            word_index_list = list(zip(['UNK'] + list(unique_everseen(text)), range(1, word_count + 2)))
            sequences = [[x[1] - 1 for x in word_index_list if word in x[0]][0] for word in text]
        else:
            word_count = len(set([word for sentence in text for word in sentence.split()]))
            max_vocab = min(max_vocab, word_count) + 1
            tokenizer = Tokenizer(num_words=max_vocab, filters='', oov_token=['UNK'][0])
            tokenizer.fit_on_texts(text)
            word_index_list = tokenizer.word_index.items()
            sequences = tokenizer.texts_to_sequences(text)

        # apply tokenizer to unprocessed words
        flattened_sequences = list(flatten(sequences))  # for downstream compatibility
        count = Counter(flattened_sequences)
        filtered_count, dictionary = {}, {}  # for downstream compatibility

        for k, v in word_index_list:
            if v <= max_vocab:
                if k == 'UNK':
                    filtered_count['UNK'] = 0
                    dictionary['UNK'] = 1
                else:
                    filtered_count[k] = count[v - 1]
                    dictionary[k] = v
            else:
                filtered_count['UNK'] += 1

        # update for downstream compatibility
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        count_list = [list(x) for x in list(zip(list(filtered_count.keys()), list(filtered_count.values())))]

        if max_vocab != len(count_list):
            raise ValueError('The length of count_as_tuples does not match max_vocab_size.')
        else:
            if isinstance(sequences, list):
                tensor_data = tf.ragged.constant(sequences)
            else:
                tensor_data = tf.convert_to_tensor(sequences)  # should now be a 1D tensor

        return tensor_data, count_list, dictionary, reverse_dictionary
