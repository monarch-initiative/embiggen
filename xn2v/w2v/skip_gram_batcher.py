import collections
import numpy as np
import random

from typing import List, Tuple, Union


class SkipGramBatcher:
    """Encapsulate functionality for getting the next batch for SkipGram from a single list of ints (e.g. from a
    single text with no sentences). Use  SkipGramListBatcher for Node2Vec. The input is expected to be a list of
    ints. This class is an implementation detail and should not be used outside of this file.

    Attributes:
        data: A list of lists of integers, representing sentences/random walks.
        batch_size: The number of samples to draw for a given training batch.
        num_skips: How many times to reuse an input to generate a label.
        skip_window: How many words to consider left and right.
        span: The total size of the sliding window [skip_window central_word skip_window].
        data_index: Index of the list that will be used next for batch generation.

    Raises:
        ValueError: If the number of skips cannot be divided by batch_size without a remainder.
        ValueError: If the number of skips is larger than the skip_window.
        TypeError: If data is not a list.
        TypeError: If the items inside data are not integers.
        ValueError: If the number of words or nodes in data is less than skip_window length.
    """

    def __init__(self, data: List[Union[int, str]], batch_size: int, num_skips: int, skip_window: int) -> None:

        self.data = data
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window

        if not batch_size % num_skips == 0:
            raise ValueError('For SkipGram, the number of skips must divide batch_size without remainder')

        if not num_skips <= 2 * skip_window:
            raise ValueError('For SkipGram, the number of skips must not be larger than skip_window')

        if not isinstance(data, list):
            raise TypeError('Data must be a list')

        if not all(x for x in data if isinstance(x, int)):
            raise TypeError('Data must be a list of integers')

        if len(data) < skip_window:
            raise ValueError('Data (%d) is shorter than skip_window (%d)' % (len(data), skip_window))

        self.span = 2 * skip_window + 1
        self.data_index = 0

    def generate_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a training batch for the skip-gram model.

        Assumptions: All of the data is in one and only one list (for instance, the data might derive from a book).

        Returns:
            A list where the first item is a batch and the second item is the batch's labels.
        """

        batch_size = self.batch_size
        span = self.span
        num_skips = self.num_skips
        skip_window = self.skip_window

        batch = np.ndarray(shape=(batch_size, ), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        # get window size (words left and right + current one)
        buffer: collections.deque = collections.deque(maxlen=span)

        if self.data_index + span > len(self.data):
            self.data_index = 0

        buffer.extend(self.data[self.data_index:self.data_index + self.span])
        self.data_index += self.span

        for i in range(batch_size // self.num_skips):
            context_words = [w for w in range(self.span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)

            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]

            if self.data_index == len(self.data):
                # i.e., we are at the end of data and need to reset the index to the beginning
                buffer.extend(self.data[0:span])
                self.data_index = span
            else:
                buffer.append(self.data[self.data_index])
                self.data_index += 1  # i.e., move the sliding window 1 position to the right

        # backtrack a little bit to avoid skipping words in the end of a batch.
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)

        return batch, labels
