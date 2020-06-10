import nltk  # type: ignore
import os
import tensorflow as tf  # type: ignore
import re
import numpy as np

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
        data_type: A string which is used to indicate whether or not the data should be read in as a single text
            object or as a list of sentences. Passed values can be "words" or "sentences" (default="sentences").
        stopwords: A set of stopwords. If nothing is passed by user a default list of stopwords is utilized.
        minlen: minimum length to include a sentence. If a sentence is shorter, it will be skipped.

    Raises:
        ValueError: If the filename is None.
        TypeError: If the filename attribute is not a string.
        IOError: If the file referenced by filename could not be found.
    """

    def __init__(self,
                 filename: str,
                 data_type: Optional[str] = None,
                 stopwords: set = None,
                 minlen: int = 10):

        if filename is None:
            raise ValueError('filename cannot be None')
        if not isinstance(filename, str):
            raise TypeError('filename must be a string')
        if not os.path.exists(filename):
            raise IOError('Could not find file referenced by filename: {}'.format(filename))

        self.filename = filename
        self.data_type = data_type if data_type else 'sentences'
        try:
            self.stopwords = nltk.corpus.stopwords.words('english') if stopwords is None else stopwords
        except LookupError:
            nltk.download('stopwords')
            self.stopwords = nltk.corpus.stopwords.words('english') if stopwords is None else stopwords
        self.minlen = minlen

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

        print('Reading data from {filename} and processing it as {data_type}'.format(filename=self.filename,
                                                                                     data_type=self.data_type))
        sentence_data = open(self.filename).readlines()
        cleaned_sentences = [self.clean_text(sent) for sent in sentence_data]

        if self.data_type == 'words':
            return ' '.join([sent for sent in cleaned_sentences if sent.count(' ') + 1 >= self.minlen]).split()
        else:
            return [sent for sent in cleaned_sentences if sent.count(' ') + 1 >= self.minlen]

    # TODO! Use fit and transform instead of a constructor that does everything.
    def fit(self, corpus):
        pass

    # TODO! Use fit and transform instead of a constructor that does everything.
    def transform(self, X: np.ndarray):
        pass

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
        # TODO: Figure out why is there is if statement and why isn't tokenizer the default.
        if self.data_type == 'words':
            word_count = len(set(text))
            max_vocab = min(max_vocab, word_count) + 1
            word_index_list = list(zip(['UNK'] + list(unique_everseen(text)), range(1, word_count + 2)))
            word_index_dict = dict(word_index_list)
            sequences = [word_index_dict[word] - 1 for word in text]
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

        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))  # for downstream compatibility
        count_list = [list(x) for x in list(zip(list(filtered_count.keys()), list(filtered_count.values())))]

        if max_vocab != len(count_list):
            raise ValueError('The length of count_as_tuples does not match max_vocab_size.')
        else:
            # try:
            #    tensor_data = tf.data.Dataset.from_tensor_slices(sequences)
            # except ValueError:
            #    tensor_data = tf.ragged.constant(sequences)  # for nested lists of differing lengths
            if isinstance(sequences, list):
                tensor_data = tf.ragged.constant(sequences)
            else:
                tensor_data = tf.convert_to_tensor(sequences)  # should now be a 1D tensor

        return tensor_data, count_list, dictionary, reverse_dictionary
