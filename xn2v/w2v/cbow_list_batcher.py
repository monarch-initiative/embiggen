import collections
import numpy as np   # type: ignore

from typing import List, Tuple, Union


from typing import List, Tuple, Union


class CBOWListBatcher:
    """Encapsulate functionality for getting the next batch for Continuous Bag of Words (CBOW). The input is expected
    to be a list of lists of ints, such as we get from node2vec. This class is an implementation detail and should not
    be used outside of this file.

    This class setups Continuous Bag of Words Batch generation for data that is presented as a list of list of
    integers (as is typical for node2vec). Note that we generate batches that consist of all of the data from k windows,
    where k is at least one.

    Attributes:
        data: A list of lists of integers, representing sentences/random walks.
        span: The total size of the sliding window [skip_window central_word skip_window].
        sentence_index: Index of the sentence that will be used next for batch generation.
        word_index: Index of the word in the sentence that will be used next for batch generation.
        data_index: Index of the list that will be used next for batch generation.
        sentence_count: The number of lists in the input data.
        sentence_len: The length of the sentences or walks.
        batch_size: The number of samples to draw for a given training batch.
        max_word_index: An integer representing the position of the last word or node in the sentence or walk.
        window_size: The size of sliding window for continuous bag of words.
        sentences_per_batch: The number of sentences to include in a batch.

    Raises:
        ValueError: If the length of all sentences or walks is not the same.
        ValueError: If less than one sentence/walk for CBOWBatcherListOfLists is provided.
    """

    def __init__(self, data: List[List[Union[str, int]]], window_size: int = 2,
                 sentences_per_batch: int = 1) -> None:

        self.data = data
        self.window_size = window_size
        self.sentences_per_batch = sentences_per_batch

        # span is the total size of the sliding window we look at [ skip_window central_word skip_window ]
        self.span: int = 2 * self.window_size + 1  # [ skip_window target skip_window ]

        # index of the sentence that will be used next for batch generation
        self.sentence_index: int = 0
        self.word_index: int = 0
        self.data_index: int = 0

        # do some q/c
        self.sentence_count: int = len(self.data)

        # enforce that all sentences have the same length
        sentence_len = 0

        for sent in self.data:
            if sentence_len == 0:
                sentence_len = len(sent)
            elif sentence_len != len(sent):
                raise ValueError('Sentence/walk lengths need to be equal.')

        self.sentence_len: int = sentence_len

        # this is the # of exs returned per batch. Int multiples of all of exs from individual sentences are returned
        self.batch_size: int = self.sentences_per_batch * (self.sentence_len - self.span + 1)
        self.max_word_index: int = self.sentence_len - self.span + 1

        if self.sentence_count < 2:
            raise ValueError('Expected more than one sentence/walk for CBOWBatcherListOfLists.')

    def get_next_sentence(self) -> List[Union[int, str]]:
        """Method returns the next sentence available for processing.

        Returns:
            sentence: A list, which represents a single walk (i.e. list of nodes in a path) or a sentence.
        """

        sentence = self.data[self.sentence_index]
        self.sentence_index += 1

        if self.sentence_index == self.sentence_count:
            self.sentence_index = 0  # reset

        return sentence

    def generate_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates the next batch of data for CBOW.

        Returns:
            A list of CBOW data (each stored as a numpy.ndarray) for training; the first item is a batch and the second
            item is the batch labels.
        """

        span = self.span

        # two numpy arrays to hold target (batch) and context words (labels). Note, batch has span-1=2*window_size cols
        batch = np.ndarray(shape=(self.batch_size, span - 1), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int64)

        # The buffer holds the data contained within the span. The deque essentially implements a sliding window
        target_idx = self.window_size  # target word index at the center of the buffer

        for _ in range(self.sentences_per_batch):
            sentence = self.get_next_sentence()

            # buffer is a sliding window with span elements
            buffer: collections.deque = collections.deque(maxlen=span)
            buffer.extend(sentence[0:span - 1])

            word_index = span - 1  # go to end of sliding window

            # For each batch index, we iterate through span elements to fill in the columns of batch array
            for i in range(self.max_word_index):
                # Put the next window into the buffer
                buffer.append(sentence[word_index])
                word_index = word_index + 1
                col_idx = 0

                for j in range(span):
                    # if j == span // 2:
                    if j == target_idx:
                        continue  # i.e., ignore the center word
                    batch[i, col_idx] = buffer[j]
                    col_idx += 1

                labels[i, 0] = buffer[target_idx]

        return batch, labels
