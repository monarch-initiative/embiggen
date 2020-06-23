from scipy.sparse import lil_matrix  # type: ignore
import collections
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
# from embiggen import TextTransformer


class CooccurrenceEncoder:
    """This class takes as input a text or a series or texts (sentences) and generates
    a coocurrence matrix using the list of list (lil) matrix from scipy. It returns
    a dictionary of co-occurences whose key is a tuple (w1,w2) and whose value is the
    coocurrence.
    """

    def __init__(self, input, window_size, vocab_size, batch_size=128):
        """
        If you are starting with text, first call
        data, count, dictionary, reverse_dict = encoder.build_dataset()
        and then pass 'data' as the input parameter for this constructor
        :param window_size: Size of the window surrounding the center word (the total size is two times this)
        """
        self.window_size = window_size
        self.num_samples_per_centerword = 2 * window_size  # number of samples per center word
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        if isinstance(input, tf.RaggedTensor):
            self.is_list = True
        elif isinstance(input, tf.Tensor):
            self.is_list = False
        else:
            raise TypeError("'input' had type {} but we expected a Tensor or RaggedTensor.".format(type(input)))
        self.data = input


    @classmethod
    def from_walks(cls, walks, n_nodes):
        """Generate a co-occurence matrix from a collection of walks from
        a random walk.
        Args:
            walks: An array of integers to use as batch training data.
            n_nodes: Total count of nodes (vertices) in the graph
        Returns:
            None.
        """
        print("Generating coocurrence matrix from walks: ", type(walks), "number of nodes ", n_nodes)


    def _generate_batch_from_sentence(self, sentence):
        """Generate a batch of co-occurence counts for this sentence
      Args:
        sentence: a list of integers representing words or nodes
      Returns:
        batch, labels, weights
      """
        # two numpy arrays to hold target words (batch)
        # and context words (labels)
        # The weight is calculated to reflect the distance from the center word
        # This is the number of context words we sample for a single target word
        num_samples_per_centerword = self.num_samples_per_centerword
        window_size = self.window_size
        num_contexts = len(sentence) - num_samples_per_centerword  # total n of full-length contexts in the sentence
        batch_size = num_contexts * num_samples_per_centerword  # For each context we sample num_samples time
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        weights = np.ndarray(shape=(batch_size), dtype=np.float32)
        # span defines the total window size, where
        # data we consider at an instance looks as follows.
        # [ skip_window target skip_window ]
        span = 2 * window_size + 1
        # This is the number of context words we sample for a single target word
        num_samples_per_centerword = 2 * window_size
        buffer: collections.deque = collections.deque(maxlen=span)
        buffer.extend(sentence[0:span - 1])
        data_index = span - 1
        for i in range(num_contexts):
            buffer.append(sentence[data_index])
            data_index += 1  # move sliding window 1 spot to the right
            k = 0
            for j in list(range(window_size)) + list(range(window_size + 1, 2 * window_size + 1)):
                batch[i * num_samples_per_centerword + k] = buffer[window_size]
                labels[i * num_samples_per_centerword + k, 0] = buffer[j]
                weights[i * num_samples_per_centerword + k] = abs(1.0 / (j - window_size))
                k += 1
        return batch, labels, weights

    def _generate_cooc_from_list(self):
        """
        Generate co-occurence matrix by processing batches of data
        We are using the list of list sparse matrix from scipy
        This function assumes that we get a list of strings. For instance, each
        string could be the text of a pubmed abstract or of a movie review.
        """
        vocab_size = self.vocab_size
        cooc_mat = lil_matrix((vocab_size, vocab_size), dtype=np.float32)
        print(cooc_mat.shape)
        i = 0
        for sequence in self.data:
            batch, labels, weights = self._generate_batch_from_sentence(sequence)
            labels = labels.reshape(-1)  # why is the reshape needed
            # Incrementing the sparse matrix entries accordingly
            for inp, lbl, w in zip(batch, labels, weights):
                cooc_mat[inp, lbl] += (1.0 * w)
            i += 1
            if i % 10 == 0:
                print("Sentence %d" % i)
        return cooc_mat

    def _generate_coor_from_single_string(self):
        """generate coocurence matrix from a single string, e.g., a string that has been generated
        from a book that has not been divided into sentence or chapters
        """
        vocab_size = self.vocab_size
        cooc_mat = lil_matrix((vocab_size, vocab_size), dtype=np.float32)
        print(cooc_mat.shape)
        i = 0
        batch_size = self.batch_size
        data_len = len(self.data)
        # Note that we cannot fully digest all of the data in any one batch
        # if the window length is K and the natch_len is N, then the last
        # window that we get starts at position (N-K). Therefore, if we start
        # the next window at position (N-K)+1, we will get all windows.
        window_len = 1 + 2 * self.window_size
        # we need to make sure that we do not shift outside the boundaries of self.data too
        lastpos = data_len - 1  # index of the last word in data
        shift_len = batch_size = window_len + 1
        data_index = 0
        endpos = data_index + batch_size
        while True:
            if endpos > lastpos:
                break
            current_sequence = self.data[data_index:endpos]
            if len(current_sequence) < window_len:
                break  # We are at the end
            batch, labels, weights = self._generate_batch_from_sentence(current_sequence)
            labels = labels.reshape(-1)  # why is the reshape needed
            # Incrementing the sparse matrix entries accordingly
            for inp, lbl, w in zip(batch, labels, weights):
                cooc_mat[inp, lbl] += (1.0 * w)
            data_index += shift_len
            endpos = data_index + batch_size
            endpos = min(endpos, lastpos)  # takes care of last part of data. Maybe we should just ignore though
        return cooc_mat

    def build_dataset(self):
        """Builds a dataset by traversing over a input text and compiling a list of the most commonly occurring words.

        Returns:
            cooc_dict: A co-occurrence dictionary with key (w1,w2) and value the count
            count: A list of tuples, where the first item in each tuple is a word and the second is the word frequency.
            dictionary: A dictionary where the keys are words and the values are the word id.
            reversed dictionary: A dictionary that is the reverse of the dictionary object mentioned above.
        """
        if self.is_list:
            cooc_mat = self._generate_cooc_from_list()
        else:
            cooc_mat = self._generate_coor_from_single_string()
        cooc_dict = cooc_mat.todok()
        return cooc_dict
