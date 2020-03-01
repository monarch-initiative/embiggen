import collections
import numpy as np   # type: ignore
import random


class SkipGramBatcher:
    """
        Encapsulate functionality for getting the next batch for SkipGram from
        a single list of ints (e.g. from a single text with no sentences). Use
        SkipGramListBatcher for Node2Vec.
        The input is expected to be a list of ints
        This class is an implementation detail and should not be used outside of this file
        """

    def __init__(self, data, batch_size, num_skips, skip_window):
        self.data = data
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
        if not batch_size % num_skips == 0:
            raise ValueError("For SkipGram, the number of skips must divide batch_size without remainder")
        if not num_skips <= 2 * skip_window:
            raise ValueError("For SkipGram, the number of skips must not be larger than skip_window")
        if not isinstance(data, list):
            raise TypeError("data must be a list")
        if len(data) < skip_window:
            raise ValueError("Data (%d) is shorter than skip_window (%d)" % (len(data), skip_window))
        first_elem = data[0]
        if not isinstance(first_elem, int):
            raise TypeError("data must be a list of integers")
        self.span = 2 * skip_window + 1
        self.data_index = 0


    def generate_batch(self):
        """
        Generate training batch for the skip-gram model. This assumes that all of the data is in one
        and only one list (for instance, the data might derive from a book). To get batches
        from a list of lists (e.g., node2vec), use the 'next_batch_from_list_of_list' function
        """
        batch_size = self.batch_size
        span = self.span
        num_skips = self.num_skips
        skip_window = self.skip_window

        batch = np.ndarray(shape=batch_size, dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        # get window size (words left and right + current one).

        buffer = collections.deque(maxlen=span)
        if self.data_index + span > len(self.data):
            self.data_index = 0
        buffer.extend(self.data[self.data_index:self.data_index + self.span])
        # print("data", data[self.data_index:self.data_index + span])
        # print("bbuffer", buffer)
        self.data_index += self.span
        for i in range(batch_size // self.num_skips):
            context_words = [w for w in range(self.span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                # j is the index of an element of words_to_use and context_word is that element
                # buffer -- the sliding window
                # buffer[skip_window] -- the center element of the sliding window
                #  buffer[context_word] -- the integer value of the word/node we are trying to predict with the skip-gram model
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
            if self.data_index == len(self.data):
                # i.e., we are at the end of data and need to reset the index to the beginning
                buffer.extend(self.data[0:span])
                self.data_index = span
            else:
                buffer.append(self.data[self.data_index])
                self.data_index += 1  # i.e., move the sliding window 1 position to the right
        # Backtrack a little bit to avoid skipping words in the end of a batch.
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels

