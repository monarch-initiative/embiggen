import collections
import numpy as np

class CBOWListBatcher:
    """
    Encapsulate functionality for getting the next batch for Continuous Bag of Words (CBOW)
    The input is expected to be a list of lists of ints, such as we get from node2vec
    This class is an implementation detail and should not be used outside of this file
    """

    def __init__(self, data, window_size, sentences_per_batch):
        """Setup Continuous Bag of Words Batch generation for data that is presented
        as a list of list of integers (as is typical for node2vec). Note that we generate
        batches that consist of all of the data from k windows, where k is at least one.

        Args:
            data: a list of lists of integers, representing sentences/random walks
            window_size: size of sliding window for continuous bag of words
            sentences_per_batch: number of sentences to include in one batch
        """
        self.data = data
        self.window_size = window_size
        self.sentences_per_batch = sentences_per_batch
        # span is the total size of the sliding window we look at.
        # [ skip_window central_word skip_window ]
        self.span = 2 * self.window_size + 1  # [ skip_window target skip_window ]
        self.sentence_index = 0  # index of the sentence that will be used next for batch generation
        self.word_index = 0  # index of word in the current sentence that will be used next
        self.data_index = 0
        # Do some Q/C
        self.sentence_count = len(self.data)
        # enforce that all sentences have the same length
        sentence_len = None
        for sent in self.data:
            if sentence_len is None:
                sentence_len = len(sent)
            elif sentence_len != len(sent):
                raise TypeError("Sentence lengths need to be equal for sll sentences")
        self.sentence_len = sentence_len
        # This is the number of examples we will return per batch
        # Note that we return integer multiples of all of the examples we can get out of
        # individual sentences
        self.batch_size = self.sentences_per_batch * (self.sentence_len - self.span + 1)
        self.max_word_index = self.sentence_len - self.span + 1
        if self.sentence_count < 2:
            raise TypeError("Expected more than one sentence for CBOWBatcherListOfLists")

    def get_next_sentence(self):
        """Return the next sentence available for processing. Rotate to
        the beginning of the dataset if we are at the end.
        """
        sentence = self.data[self.sentence_index]
        self.sentence_index += 1
        if self.sentence_index == self.sentence_count:
            self.sentence_index = 0  # reset
        return sentence

    def generate_batch(self):
        """
        Generate the next batch of data for CBOW

        Returns:
            A batch CBOW data for training
        """
        span = self.span
        # two numpy arrays to hold target words (batch)
        # and context words (labels)
        # Note that batch has span-1=2*window_size columns
        batch = np.ndarray(shape=(self.batch_size, span - 1), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int64)

        # The buffer holds the data contained within the span.
        # The deque essentially implements a sliding window

        target_idx = self.window_size  # target word index at the center of the buffer

        for _ in range(self.sentences_per_batch):
            sentence = self.get_next_sentence()
            # buffer is a sliding window with span elements
            buffer = collections.deque(maxlen=span)
            buffer.extend(sentence[0:span-1])
            word_index = span - 1  # go to end of sliding window
            # For each batch index, we iterate through span elements
            # to fill in the columns of batch array
            for i in range(self.max_word_index):
                # Put the next window into the buffer
                buffer.append(sentence[word_index])
                word_index = word_index + 1
                col_idx = 0
                for j in range(span):
                    # if j == span // 2:
                    if j == target_idx:
                        continue  # i.e., ignore the center wortd
                    batch[i, col_idx] = buffer[j]
                    col_idx += 1
                labels[i, 0] = buffer[target_idx]

        return batch, labels

